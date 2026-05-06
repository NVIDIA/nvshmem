/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "libfabric.h"
#include <assert.h>
#include <sys/uio.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/types/struct_iovec.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <mutex>
#include <new>
#include <unordered_map>
#include <memory>
#include <utility>
#include <errno.h>
#include <sched.h>

#include "internal/host_transport/cudawrap.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "device_host_transport/nvshmem_constants.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "non_abi/nvshmemx_error.h"
#include "non_abi/nvshmem_build_options.h"  // IWYU pragma: keep
#include "non_abi/nvshmem_version.h"
#include "rdma/fabric.h"
#include "rdma/fi_atomic.h"
#include "rdma/fi_cm.h"
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_eq.h"
#include "rdma/fi_errno.h"
#include "rdma/fi_rma.h"
#include "internal/host_transport/transport.h"
#include "transport_common.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

/* Note - this is required to not break on Slingshot systems
 * where we compile with libfabric < 1.15.
 */
#ifndef FI_OPT_CUDA_API_PERMITTED
#define FI_OPT_CUDA_API_PERMITTED 10
#endif

#define NVSHMEM_STAGED_AMO_WIREDATA_SIZE \
    sizeof(nvshmemt_libfabric_gdr_op_ctx_t) - sizeof(struct fi_context2) - sizeof(fi_addr_t)

namespace {
constexpr size_t max_completions_per_cq_poll = 300;
constexpr size_t max_completions_per_cq_poll_efa = 32;

constexpr uint32_t imm_header_bit_shift = 28;

/* Internal types */
typedef enum {
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDRCOPY_AMO_ACK,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PERFORM_GDRCOPY_AMO_SEND,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_PROCESS_AMOS_GET_NEXT_ACK,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_PROCESS_AMOS_GET_NEXT_NOT_ACK,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_GET_NEXT_SENDS,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_EFA,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_NON_EFA,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_PUT,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_GET,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_GET_NEXT_SENDS,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_SEND,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_AMO_ATOMICMSG,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_GET_NEXT_SENDS,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_SEND,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PUT_SIGNAL_UNORDERED_SEQ,
    NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_ENFORCE_CST
} nvshmemt_libfabric_try_again_call_site_t;

enum class progress_type {
    CompletionsOnly,
    All,
};

enum class completion_kind {
    Invalid,
    PutSignalWrite,
    PutSignalSignal,
    StandalonePut,
    StandalonePutAckReq,
};

/* Internal global variables */
#ifdef NVSHMEM_USE_GDRCOPY
struct gdrcopy_function_table gdrcopy_ftable;
void *gdrcopy_handle = NULL;
gdr_t gdr_desc;
bool use_gdrcopy = false;
#endif

struct nvshmemi_options_s options;
}  // namespace

#define NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, err, label, ...)                             \
    do {                                                                                     \
        if (unlikely((status) != 0)) {                                                       \
            fprintf(stderr, "%s:%d: libfabric status: %d (%s) ", __FILE__, __LINE__, status, \
                    fi_strerror((status) < 0 ? -(status) : (status)));                       \
            fprintf(stderr, __VA_ARGS__);                                                    \
            fprintf(stderr, "\n");                                                           \
            status = err;                                                                    \
            goto label;                                                                      \
        }                                                                                    \
    } while (0)

/* Forward declarations */
static int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport, int qp_index);
static int nvshmemt_libfabric_gdr_process_amo(nvshmem_transport_t transport,
                                              nvshmemt_libfabric_gdr_op_ctx_t *op,
                                              nvshmemt_libfabric_gdr_op_ctx_t **send_elems,
                                              uint32_t sequence_count,
                                              uint8_t preceding_put_count);
static int nvshmemt_libfabric_put_signal_completion(nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    const struct fi_cq_data_entry &entry,
                                                    const fi_addr_t &addr);
static int drain_completions(nvshmem_transport_t transport, int qp_index);
static int nvshmemt_libfabric_progress(nvshmem_transport_t transport, int qp_index);

namespace {
/* Internal functions */
static inline nvshmemt_libfabric_state_t *get_libfabric_state(nvshmem_transport_t transport) {
    return static_cast<nvshmemt_libfabric_state_t *>(transport->state);
}

int get_next_ep(nvshmemt_libfabric_state_t *state, int qp_index) {
    if (qp_index == NVSHMEMX_QP_HOST) {
        return 0; /* Currently only 1 EP defined for the host */
    } else {
        return ((state->proxy_ep_cntr++) % state->num_proxy_domains) + state->num_host_domains;
    }
}

nvshmemt_libfabric_imm_cq_data_hdr_t get_write_with_imm_hdr(uint64_t imm_data) {
    return static_cast<nvshmemt_libfabric_imm_cq_data_hdr_t>(
              static_cast<uint32_t>(imm_data) >> imm_header_bit_shift);
}

completion_kind classify_completion(const struct fi_cq_data_entry &entry) {
    if (!(entry.flags & FI_REMOTE_CQ_DATA)) {
        return completion_kind::PutSignalSignal;
    }

    switch (get_write_with_imm_hdr(entry.data)) {
        case NVSHMEMT_LIBFABRIC_IMM_PUT_SIGNAL_SEQ:
            return completion_kind::PutSignalWrite;
        case NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT:
            return completion_kind::StandalonePut;
        case NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT_WITH_ACK_REQ:
            return completion_kind::StandalonePutAckReq;
        default:
            return completion_kind::Invalid;
    }
}

/*
 * TODO: Make the following more general by using fid_nic field in fi_info.
 */
int ib_iface_get_nic_path(const char *nic_name, const char *nic_class, char **path) {
    int status;

    char device_path[MAXPATHSIZE];
    status = snprintf(device_path, MAXPATHSIZE, "/sys/class/%s/%s/device", nic_class, nic_name);
    if (status < 0 || status >= MAXPATHSIZE) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to fill in device name.\n");
    } else {
        status = NVSHMEMX_SUCCESS;
    }

    *path = realpath(device_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static inline int convert_addr_to_pe(nvshmemt_libfabric_state_t *state,
                                     const nvshmemt_libfabric_endpoint_t &ep,
                                     const fi_addr_t &addr) {
    // addr = pe * state->eps.size() + ep.ep_index
    // so
    // pe = (addr - ep.ep_index) / state->eps.size()
    int num_eps = state->eps.size();
    int base_ep_index = addr - ep.ep_index;
    assert((base_ep_index % num_eps) == 0);

    return base_ep_index / num_eps;
}

static inline nvshmemt_libfabric_signal_state_t &get_signal_state(
    nvshmemt_libfabric_state_t *state, const nvshmemt_libfabric_endpoint_t &ep) {
    if (ep.domain_index < state->num_host_domains) {
        return state->host_signal_state;
    }
    return state->proxy_signal_state;
}

static std::pair<bool, int> accumulate_signal(signal_seq_map &completion_map, uint32_t seq,
                                              nvshmemt_libfabric_gdr_op_ctx_t *op, int delta) {
    int status = NVSHMEMX_SUCCESS;
    bool drainable = false;
    nvshmemt_libfabric_comp_entry_t fresh_entry =
        nvshmemt_libfabric_signal_comp_entry{op, delta};

    auto insert_result = completion_map.insert(seq, fresh_entry);
    auto *signal_entry =
        std::get_if<nvshmemt_libfabric_signal_comp_entry>(insert_result.first);

    NVSHMEMI_NULL_ERROR_JMP(signal_entry, status, NVSHMEMX_ERROR_INTERNAL, out,
                           "unexpected ack entry while accumulating signal completion.\n");

    if (!insert_result.second) {
        if (op != nullptr) signal_entry->op = op;
        signal_entry->progress_count += delta;
    }

    drainable = (signal_entry->progress_count == 0);

out:
    return {drainable, status};
}

/* Return the signal_state for the given EP and conditionally lock its mutex.
 * Host EPs (domain_index < num_host_domains) are shared between user and proxy
 * threads, so the mutex must be held.  Proxy-only EPs are single-threaded under
 * auto_progress, so the lock is skipped for performance. */
static inline std::pair<nvshmemt_libfabric_signal_state_t *,
                        std::unique_lock<std::recursive_mutex>>
get_signal_state_locked(nvshmemt_libfabric_state_t *state,
                        const nvshmemt_libfabric_endpoint_t &ep) {
    bool is_host = ep.domain_index < state->num_host_domains;
    auto *sig = is_host ? &state->host_signal_state : &state->proxy_signal_state;
    std::unique_lock<std::recursive_mutex> lk(sig->mtx, std::defer_lock);
    if (is_host) lk.lock();
    return {sig, std::move(lk)};
}

/* FI_THREAD_COMPLETION: user-thread submissions on host EP (eps[0..num_host_domains])
 * must be serialized with the proxy thread's fi_cq_read on the same CQ.
 * This guard acquires host_ep_progress_lock blockingly iff ep is a host EP. */
struct host_ep_submit_guard {
    nvshmemt_libfabric_state_t *state;
    bool held;
    host_ep_submit_guard(nvshmemt_libfabric_state_t *s,
                         const nvshmemt_libfabric_endpoint_t &ep)
        : state(s), held(false) {
        if (ep.domain_index < state->num_host_domains) {
            state->host_ep_progress_lock.lock();
            held = true;
        }
    }
    ~host_ep_submit_guard() {
        if (held) state->host_ep_progress_lock.unlock();
    }
    host_ep_submit_guard(const host_ep_submit_guard&) = delete;
    host_ep_submit_guard& operator=(const host_ep_submit_guard&) = delete;
};

int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;
    const char *nic_name, *nic_class;
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(t);

    if ((libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) ||
        (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA)) {
        nic_class = "infiniband";
    } else {
        nic_class = "cxi";
    }

    nic_name = libfabric_state->domain_names[dev].name.data();

    status = ib_iface_get_nic_path(nic_name, nic_class, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get_pci_path failed \n");

out:
    return status;
}

int check_signal_delivery_status(nvshmemt_libfabric_state_t *state) {
    if (unlikely(state->signal_delivery_status.load(std::memory_order_acquire) != 0)) {
        return NVSHMEMX_ERROR_INTERNAL;
    }
    return 0;
}

int try_again(nvshmem_transport_t transport, int *status, uint64_t *num_retries,
              nvshmemt_libfabric_try_again_call_site_t call_site, int qp_index,
              progress_type prog_type) {
    if (likely(*status == 0)) {
        return 0;
    }

    if (likely(*status == -FI_EAGAIN)) {
        if (unlikely(*num_retries >= NVSHMEMT_LIBFABRIC_MAX_RETRIES)) {
            fprintf(stderr, "call site: %d, Max amount of libfabric retries reached, %d: %s\n",
                    call_site, *status, fi_strerror(*status * -1));
            *status = NVSHMEMX_ERROR_INTERNAL;
            return 0;
        }
        (*num_retries)++;
        /*
         * CompletionsOnly is used by retries originating from paths that must not
         * re-enter top-level progress while signal_progress_lock is held.
         */
        if (prog_type == progress_type::All) {
            *status = nvshmemt_libfabric_progress(transport, qp_index);
        } else {
            *status = drain_completions(transport, qp_index);
        }
    }

    if (unlikely(*status != 0)) {
        fprintf(stderr, "Error in libfabric operation (%d): %s.\n", *status,
                fi_strerror(*status * -1));
        *status = NVSHMEMX_ERROR_INTERNAL;
        return 0;
    }

    return 1;
}

static inline int get_next_seq_num_with_retry(nvshmem_transport_t transport,
                                               nvshmemt_libfabric_endpoint_seq_counter_t &seq_counter,
                                               uint32_t *sequence_count,
                                               int qp_index,
                                               nvshmemt_libfabric_try_again_call_site_t call_site) {
    uint64_t num_retries = 0;
    int status;
    do {
        int32_t seq_num = seq_counter.next_seq_num();
        if (seq_num < 0) {
            status = -EAGAIN;
        } else {
            *sequence_count = seq_num;
            status = 0;
        }
    } while (try_again(transport, &status, &num_retries, call_site, qp_index,
                       progress_type::All));

    return status;
}

int gdrcopy_amo_ack(nvshmem_transport_t transport, nvshmemt_libfabric_endpoint_t &ep,
                    fi_addr_t dest_addr, uint16_t ack_seq_num, uint8_t ack_count,
                    uint8_t ack_num_ops) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_gdr_op_ctx_t *send_elem;
    nvshmemt_libfabric_gdr_amo_ack_op_t *ack_op;
    uint64_t num_retries = 0;
    int status;
    host_ep_submit_guard host_guard(libfabric_state, ep);

    do {
        status = libfabric_state->op_queue[ep.domain_index]->getNextSends(&send_elem, 1);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDRCOPY_AMO_ACK, ep.qp_index,
                       progress_type::CompletionsOnly));
    if (status) goto out;

    ack_op = reinterpret_cast<nvshmemt_libfabric_gdr_amo_ack_op_t *>(send_elem);
    ack_op->type = NVSHMEMT_LIBFABRIC_AMO_ACK_SEND;
    ack_op->ack.ack_seq_num = ack_seq_num;
    ack_op->ack.ack_count = ack_count;
    ack_op->ack.ack_num_ops = ack_num_ops;
    do {
        status = fi_send(ep.endpoint, (void *)ack_op, sizeof(nvshmemt_libfabric_gdr_amo_ack_op_t),
                         fi_mr_desc(libfabric_state->mrs[ep.domain_index]), dest_addr,
                         &send_elem->ofi_context);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDRCOPY_AMO_ACK, ep.qp_index,
                       progress_type::CompletionsOnly));

    NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "Unable to send atomic ack.\n");
    ep.submitted_ops++;

out:
    return status;
}

template <typename T>
int perform_gdrcopy_amo(nvshmem_transport_t transport, nvshmemt_libfabric_gdr_op_ctx_t *op,
                        nvshmemt_libfabric_gdr_op_ctx_t **send_elems, uint32_t sequence_count,
                        uint8_t preceding_put_count) {
    T old_value, new_value = {};
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_gdr_send_amo_op_t *received_op = &(op->send_amo);
    nvshmemt_libfabric_memhandle_info_t *handle_info;
    volatile T *ptr;
    int status = 0;
    /* Decode float flag from op field */
    bool is_float = (received_op->op & NVSHMEMI_AMO_FLOAT_BIT) != 0;
    received_op->op = (nvshmemi_amo_t)(received_op->op & ~NVSHMEMI_AMO_FLOAT_BIT);
    signal_delivery_done_entry done{};
    /* Save op fields as registers to allow posting op as RX before TX */
    int src_pe = op->send_amo.src_pe;
    fi_addr_t src_addr = op->src_addr;
    bool is_fetch_amo = received_op->op > NVSHMEMI_AMO_END_OF_NONFETCH;
    uint64_t ret_flags = received_op->retflag;
    void *ret_addr = received_op->ret_addr;

    handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
        transport, libfabric_state->cache, received_op->target_addr);
    NVSHMEMI_NULL_ERROR_JMP(handle_info, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to get mem handle for atomic.\n");
#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        ptr = (volatile T *)((char *)handle_info->cpu_ptr +
                             ((char *)received_op->target_addr - (char *)handle_info->ptr));
    } else
#endif
    {
        ptr = (volatile T *)received_op->target_addr;
    }
    old_value = *ptr;

    switch (received_op->op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            /* The static_cast is used to truncate the uint64_t value of swap_add back to its
             * original length */
            new_value = static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            if (is_float) {
                new_value = nvshmemt_float_atomic_add<T>(old_value, received_op->swap_add);
            } else {
                new_value = old_value + static_cast<T>(received_op->swap_add);
            }
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ static_cast<T>(received_op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == static_cast<T>(received_op->comp))
                            ? static_cast<T>(received_op->swap_add)
                            : old_value;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            new_value = old_value;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "RMA/AMO verb %d not implemented\n", received_op->op);
        }
    }

    *ptr = new_value;
    STORE_BARRIER();

    /* Push to done_queue for proxy thread to handle EP ops */
    done.op = op;
    done.send_elems[0] = send_elems[0];
    done.send_elems[1] = send_elems[1];
    done.sequence_count = sequence_count;
    done.preceding_put_count = preceding_put_count;
    done.src_pe = src_pe;
    done.src_addr = src_addr;
    done.ep_index = op->ep_index;
    done.is_fetch_amo = is_fetch_amo;
    done.old_value = static_cast<uint64_t>(old_value);
    done.ret_flags = ret_flags;
    done.ret_addr = ret_addr;
    while (!libfabric_state->signal_done_queue.push(done)) {
        NVSHMEMT_LIBFABRIC_CPU_RELAX();
    }

out:
    return status;
}

/* Process done_queue entries: fi_recv re-post + fi_send response + fi_send ACK.
 * Process 1 per call for low latency. Deadlock-free because process_amos drains
 * done_queue when work_queue is full instead of spinning. */
static int nvshmemt_libfabric_gdr_complete_amos(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    signal_delivery_done_entry done{};
    uint64_t num_retries = 0;
    int send_elems_index = 0;
    int status = 0;

    if (!libfabric_state->signal_done_queue.pop(done)) {
        return 0;
    }

    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[done.ep_index]);
    host_ep_submit_guard host_guard(libfabric_state, ep);

    /* Post recv before posting TX operations to avoid deadlocks */
    status =
        fi_recv(ep.endpoint, (void *)done.op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                fi_mr_desc(libfabric_state->mrs[ep.domain_index]), FI_ADDR_UNSPEC,
                &done.op->ofi_context);
    NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "Unable to re-post recv.\n");

    if (done.is_fetch_amo) {
        nvshmemt_libfabric_gdr_op_ctx_t *resp_op = done.send_elems[send_elems_index];

        resp_op->ret_amo.elem.data = done.old_value;
        resp_op->ret_amo.elem.flag = done.ret_flags;
        resp_op->ret_amo.ret_addr = done.ret_addr;
        resp_op->type = NVSHMEMT_LIBFABRIC_ACK;

        do {
            status = fi_send(ep.endpoint, (void *)resp_op,
                             NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                             fi_mr_desc(libfabric_state->mrs[ep.domain_index]),
                             done.src_addr, &resp_op->ofi_context);
        } while (try_again(transport, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PERFORM_GDRCOPY_AMO_SEND,
                           ep.qp_index, progress_type::CompletionsOnly));
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to respond to atomic request.\n");
        ep.submitted_ops++;
        send_elems_index++;
    }

    /* Return remaining send element(s) to pool — aggregator acquires its own at flush */
    for (int i = send_elems_index; i < 2 && done.send_elems[i]; i++) {
        libfabric_state->op_queue[ep.domain_index]->putToSend(done.send_elems[i]);
    }

    {
        /* Record ack in aggregator instead of sending immediately */
        auto [signal_state_p, sig_lk] = get_signal_state_locked(libfabric_state, ep);
        nvshmemt_libfabric_signal_state_t &signal_state = *signal_state_p;

        if (done.sequence_count == nvshmemt_libfabric_endpoint_seq_counter_t::last_seq_num) {
            status = signal_state.ack_aggregator->record_amo_ack(done.src_pe, transport, ep,
                                                                 done.src_addr);
        } else {
            status = signal_state.ack_aggregator->record_ack(done.src_pe, done.sequence_count,
                                                             transport, ep, done.src_addr,
                                                             done.preceding_put_count);
        }
    }

out:
    return status;
}

static inline void nvshmemt_libfabric_wake_signal_delivery_thread(
        nvshmemt_libfabric_state_t *state) {
    if (state->signal_delivery_futex.exchange(1, std::memory_order_release) == 0) {
        syscall(SYS_futex, &state->signal_delivery_futex, FUTEX_WAKE, 1, NULL, NULL, 0);
    }
}

/*
 * Enqueue a work item into signal_work_queue with backpressure. Acquires signal_work_queue_lock,
 * pushes `work`, and if the SPSC queue is full, drains done_queue via gdr_complete_amos to make
 * room before retrying.
 *
 * The lock is released across the drain because gdr_complete_amos calls progress which can
 * re-enter put_signal_completion and try to acquire this same non-reentrant lock; holding it
 * across the drain would deadlock.
 *
 * Returns 0 on success, NVSHMEMX_ERROR_INTERNAL if the drain fails. The lock is always released
 * before returning
 */
static inline int nvshmemt_libfabric_enqueue_signal_work(nvshmem_transport_t transport,
                                                         const signal_delivery_work_entry &work) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int status;

    while (libfabric_state->signal_work_queue_lock.test_and_set(std::memory_order_acquire))
        NVSHMEMT_LIBFABRIC_CPU_RELAX();
    while (!libfabric_state->signal_work_queue.push(work)) {
        libfabric_state->signal_work_queue_lock.clear(std::memory_order_release);
        status = nvshmemt_libfabric_gdr_complete_amos(transport);
        if (status) {
            NVSHMEMI_WARN_PRINT("Failed call to nvshmemt_libfabric_gdr_complete_amos: %d", status);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        nvshmemt_libfabric_wake_signal_delivery_thread(libfabric_state);
        while (libfabric_state->signal_work_queue_lock.test_and_set(std::memory_order_acquire))
            NVSHMEMT_LIBFABRIC_CPU_RELAX();
    }
    nvshmemt_libfabric_wake_signal_delivery_thread(libfabric_state);
    libfabric_state->signal_work_queue_lock.clear(std::memory_order_release);
    return 0;
}

nvshmemt_libfabric_gdr_op_ctx_t *inplace_copy_sig_op_to_gdr_op(
    nvshmemt_libfabric_gdr_signal_op *sig_op, int ep_index) {
    nvshmemt_libfabric_gdr_op_ctx_t *amo;
    uint8_t op = sig_op->op;
    uint8_t elem_size = sig_op->elem_size;
    uint64_t sig_val = sig_op->sig_val;
    void *target_addr = sig_op->target_addr;
    uint32_t src_pe = sig_op->src_pe;
    uint32_t sequence_count = sig_op->sequence_count;
    uint8_t preceding_put_count = sig_op->preceding_put_count;

    amo = (nvshmemt_libfabric_gdr_op_ctx_t *)sig_op;
    amo->ep_index = ep_index;
    amo->type = NVSHMEMT_LIBFABRIC_MATCH;
    amo->send_amo.op = static_cast<nvshmemi_amo_t>(op);
    amo->send_amo.target_addr = target_addr;
    amo->send_amo.swap_add = sig_val;
    amo->send_amo.src_pe = src_pe;
    amo->send_amo.size = elem_size;
    amo->send_amo.sequence_count = sequence_count;
    amo->send_amo.preceding_put_count = preceding_put_count;

    return amo;
}
}  // anonymous namespace

/* ---- Ack Aggregator Implementation ---- */

int nvshmemt_libfabric_ack_aggregator_t::flush_peer(int pe, nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    fi_addr_t dest_addr) {
    int status;
    auto &pending = pending_per_peer[pe];
    if (pending.total_pending() == 0) return 0;

    status =
        gdrcopy_amo_ack(transport, ep, dest_addr, pending.range_end, pending.range_count,
                        pending.ack_op_count());
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to send coalesced ack.\n");

    /* Clear pending state */
    pending.range_end = 0;
    pending.range_count = 0;
    pending.has_range = false;
    pending.amo_ack_count = 0;
    pending.signal_ack_count = 0;

out:
    return status;
}

int nvshmemt_libfabric_ack_aggregator_t::record_ack(int pe, uint16_t seq_num,
                                                    nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    fi_addr_t dest_addr,
                                                    uint8_t preceding_put_count) {
    int status = 0;
    auto &pending = pending_per_peer[pe];
    pending.age = 0;

    uint16_t start_seq_num = nvshmemt_libfabric_endpoint_seq_counter_t::seq_num_wrapdown(
        (seq_num - preceding_put_count) & nvshmemt_libfabric_endpoint_seq_counter_t::sequence_mask);
    uint16_t new_range_count = static_cast<uint16_t>(preceding_put_count) + 1;
    assert(new_range_count <= UINT8_MAX);

    if (!pending.has_range) {
        pending.range_end = seq_num;
        pending.range_count = static_cast<uint8_t>(new_range_count);
        pending.has_range = true;
    } else {
        uint16_t next_expected = nvshmemt_libfabric_endpoint_seq_counter_t::seq_num_wrapup(
            (pending.range_end + 1) &
            nvshmemt_libfabric_endpoint_seq_counter_t::sequence_mask
        );
        if (start_seq_num == next_expected) {
            uint16_t range_count = static_cast<uint16_t>(pending.range_count) + new_range_count;
            assert(range_count <= UINT8_MAX);
            pending.range_count = static_cast<uint8_t>(range_count);
            pending.range_end = seq_num;
        } else {
            /* Non-contiguous: flush current range, start new one */
            status = flush_peer(pe, transport, ep, dest_addr);
            if (unlikely(status)) return status;
            pending.range_end = seq_num;
            pending.range_count = static_cast<uint8_t>(new_range_count);
            pending.has_range = true;
        }
    }
    assert(static_cast<uint16_t>(pending.signal_ack_count) + pending.amo_ack_count < UINT8_MAX);
    pending.signal_ack_count++;

    if (!pending.is_dirty) {
        pending.is_dirty = true;
        dirty_peers.push_back(pe);
    }

    if (pending.total_pending() >= flush_threshold) {
        status = flush_peer(pe, transport, ep, dest_addr);
    }
    return status;
}

int nvshmemt_libfabric_ack_aggregator_t::record_amo_ack(int pe, nvshmem_transport_t transport,
                                                        nvshmemt_libfabric_endpoint_t &ep,
                                                        fi_addr_t dest_addr) {
    auto &pending = pending_per_peer[pe];
    pending.age = 0;

    assert(static_cast<uint16_t>(pending.signal_ack_count) + pending.amo_ack_count < UINT8_MAX);
    pending.amo_ack_count++;

    if (!pending.is_dirty) {
        pending.is_dirty = true;
        dirty_peers.push_back(pe);
    }

    if (pending.total_pending() >= flush_threshold) {
        return flush_peer(pe, transport, ep, dest_addr);
    }
    return 0;
}

int nvshmemt_libfabric_ack_aggregator_t::flush_all(nvshmem_transport_t transport,
                                                   nvshmemt_libfabric_endpoint_t &ep) {
    nvshmemt_libfabric_state_t *state = (nvshmemt_libfabric_state_t *)transport->state;
    int status = 0;

    for (int pe : dirty_peers) {
        fi_addr_t dest_addr =  static_cast<fi_addr_t>(pe * state->eps.size() + ep.ep_index);
        status = flush_peer(pe, transport, ep, dest_addr);
        pending_per_peer[pe].is_dirty = false;
        if (status) break;
    }
    dirty_peers.clear();

    return status;
}

int nvshmemt_libfabric_ack_aggregator_t::flush_stale(nvshmem_transport_t transport,
                                                     nvshmemt_libfabric_endpoint_t &ep) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int status = 0;

    for (size_t i = 0; i < dirty_peers.size(); ) {
        int pe = dirty_peers[i];
        auto &pending = pending_per_peer[pe];
        if (pending.total_pending() == 0) {
            /* Non-pending entry; remove from dirty_peers (swap-and-pop) */
            pending.is_dirty = false;
            dirty_peers[i] = dirty_peers.back();
            dirty_peers.pop_back();
            continue;
        }
        pending.age++;
        if (pending.age >= max_age) {
            fi_addr_t dest_addr =
                static_cast<fi_addr_t>(pe * libfabric_state->eps.size() + ep.ep_index);
            status = flush_peer(pe, transport, ep, dest_addr);
            if (status) return status;
            /* flush_peer clears pending; remove from dirty_peers (swap-and-pop) */
            pending.is_dirty = false;
            dirty_peers[i] = dirty_peers.back();
            dirty_peers.pop_back();
        } else {
            i++;
        }
    }

    return status;
}

bool nvshmemt_libfabric_ack_aggregator_t::try_extract_for_peer(int pe, uint16_t &range_end,
                                                               uint8_t &range_count,
                                                               uint8_t &signal_ack_count) {
    auto &pending = pending_per_peer[pe];
    if (pending.total_pending() == 0)
        return false;
    range_end = pending.range_end;
    range_count = pending.range_count;
    signal_ack_count = pending.ack_op_count();

    /* Clear pending state */
    pending.range_end = 0;
    pending.range_count = 0;
    pending.has_range = false;
    pending.amo_ack_count = 0;
    pending.signal_ack_count = 0;
    pending.age = 0;

    /* Leave in dirty_peers; flush_stale will clean up non-pending entries */
    return true;
}

/* Private functions with external linkage (local symbols) */
static void nvshmemt_libfabric_apply_ack(nvshmemt_libfabric_signal_state_t &signal_state, int pe,
                                         const nvshmemt_libfabric_ack_payload_t &ack) {
    if (ack.ack_count > 0) {
        uint32_t end_seq =
            nvshmemt_libfabric_endpoint_seq_counter_t::seq_num_wrapup(ack.ack_seq_num);
        signal_state.put_signal_seq_counter[pe].return_acked_range(end_seq, ack.ack_count);
    }
    signal_state.completed_staged_atomics += ack.ack_num_ops;
}

static void nvshmemt_libfabric_put_signal_ack_completion(nvshmemt_libfabric_state_t *state,
                                                         nvshmemt_libfabric_endpoint_t &ep,
                                                         const nvshmemt_libfabric_gdr_amo_ack_op_t &ack_op,
                                                         fi_addr_t addr) {
    auto [signal_state_p, sig_lk] = get_signal_state_locked(state, ep);
    nvshmemt_libfabric_signal_state_t &signal_state = *signal_state_p;

    int pe = convert_addr_to_pe(state, ep, addr);
    nvshmemt_libfabric_apply_ack(signal_state, pe, ack_op.ack);
}

static inline bool is_signal_only_op(nvshmemi_amo_t op) {
    return (op == NVSHMEMI_AMO_SIGNAL || op == NVSHMEMI_AMO_SIGNAL_SET ||
            op == NVSHMEMI_AMO_SIGNAL_ADD || op == NVSHMEMI_AMO_ADD);
}

static inline int nvshmemt_libfabric_gdr_process_ack(nvshmem_transport_t transport,
                                                     nvshmemt_libfabric_gdr_op_ctx_t *op) {
    nvshmemt_libfabric_gdr_ret_amo_op_t *ret = &op->ret_amo;
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_memhandle_info_t *handle_info;
    g_elem_t *elem;
    void *valid_cpu_ptr;

    handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
        transport, libfabric_state->cache, ret->ret_addr);
    if (!handle_info) {
        NVSHMEMI_ERROR_PRINT("Unable to get handle info for atomic response.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    valid_cpu_ptr =
        (void *)((char *)handle_info->cpu_ptr + ((char *)ret->ret_addr - (char *)handle_info->ptr));
    assert(valid_cpu_ptr);
    elem = (g_elem_t *)valid_cpu_ptr;
    elem->data = ret->elem.data;
    elem->flag = ret->elem.flag;

    return 0;
}

static int nvshmemt_libfabric_gdr_process_completion(nvshmem_transport_t transport,
                                                     nvshmemt_libfabric_endpoint_t &ep,
                                                     const struct fi_cq_data_entry &entry,
                                                     const fi_addr_t &addr) {
    int status = 0;
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_state_t *state = get_libfabric_state(transport);
    int domain_idx = ep.domain_index;

    /* Write w/imm doesn't have op->op_context, must be checked first */
    if (entry.flags & FI_REMOTE_CQ_DATA) {
        nvshmemt_libfabric_imm_cq_data_hdr_t imm_header = get_write_with_imm_hdr(entry.data);
        if (NVSHMEMT_LIBFABRIC_IMM_PUT_SIGNAL_SEQ == imm_header ||
            NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT == imm_header ||
            NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT_WITH_ACK_REQ == imm_header) {
            status = nvshmemt_libfabric_put_signal_completion(transport, ep, entry, addr);
            goto out;
        } else {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "Received a write w/imm completion with invalid header type.\n");
        }
    }

    op = container_of(entry.op_context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context);
    /* FI_CONTEXT2 support requires that every operation with a completion has a context */
    assert(op);
    op->src_addr = addr;

    if (entry.flags & FI_SEND) {
        state->op_queue[domain_idx]->putToSend(op);
        ep.completed_ops++;
    } else if (entry.flags & FI_RMA) {
        /* inlined p ops or atomic responses */
        state->op_queue[domain_idx]->putToSend(op);
        ep.completed_ops++;
    } else if ((op->type == NVSHMEMT_LIBFABRIC_MATCH) && (entry.flags & FI_RECV)) {
        /* Must happen after entry.flags & FI_SEND to avoid send completions */
        status = nvshmemt_libfabric_put_signal_completion(transport, ep, entry, addr);
    } else if ((entry.flags & FI_RECV) && (op->type == NVSHMEMT_LIBFABRIC_AMO_ACK_SEND)) {
        nvshmemt_libfabric_gdr_amo_ack_op_t *ack_op =
            reinterpret_cast<nvshmemt_libfabric_gdr_amo_ack_op_t *>(op);
        nvshmemt_libfabric_put_signal_ack_completion(state, ep, *ack_op, addr);
        status = fi_recv(ep.endpoint, (void *)op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                         fi_mr_desc(state->mrs[domain_idx]), FI_ADDR_UNSPEC, &op->ofi_context);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to re-post recv.\n");
    } else if (entry.flags & FI_RECV) {
        op->ep_index = ep.ep_index;
        if (op->type == NVSHMEMT_LIBFABRIC_ACK) {
            status = nvshmemt_libfabric_gdr_process_ack(transport, op);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Unable to process atomic.\n");

            status = fi_recv(ep.endpoint, (void *)op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                             fi_mr_desc(state->mrs[domain_idx]), FI_ADDR_UNSPEC,
                             &op->ofi_context);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Unable to re-post recv.\n");
        } else {
            state->op_queue[domain_idx]->putToRecv(op, NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK);
        }
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Found an invalid message type in an ep completion.\n");
    }

out:
    return status;
}

static int nvshmemt_libfabric_process_completion(nvshmem_transport_t transport, int ep_idx) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);
    int status = 0;

    size_t max_per_poll = (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA)
                              ? max_completions_per_cq_poll_efa
                              : max_completions_per_cq_poll;
    std::array<struct fi_cq_data_entry, max_completions_per_cq_poll> entry;
    std::array<fi_addr_t, max_completions_per_cq_poll> addr;
    ssize_t qstatus;
    int ret = 0;

    qstatus = fi_cq_readfrom(ep.cq, entry.data(), max_per_poll, addr.data());
    /* Note - EFA provider does not support selective completions */
    if (qstatus > 0) {
        if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            for (int j = 0; j < qstatus; j++) {
                status =
                    nvshmemt_libfabric_gdr_process_completion(transport, ep, entry[j], addr[j]);
                if (status) return NVSHMEMX_ERROR_INTERNAL;
            }
        } else {
            NVSHMEMI_WARN_PRINT("Got %zd unexpected events on EP %d\n", qstatus, ep_idx);
            return NVSHMEMX_ERROR_INTERNAL;
        }
    } else if (qstatus < 0 && qstatus != -FI_EAGAIN) {
        /* On call to fi_cq_readerr, Libfabric requires some members of
         * err_entry to be zero-initialized or point to valid data.  For
         * simplicity, just zero out the whole struct.
         */
        struct fi_cq_err_entry err_entry = {};

        ret = fi_cq_readerr(ep.cq, &err_entry, 0);
        if (ret == -FI_EAGAIN) {
            return 0;
        } else if (ret < 0) {
            NVSHMEMI_WARN_PRINT("Unable to read from fi_cq_readerr. RC: %d. Error: %s\n", ret, fi_strerror(-ret));
            return NVSHMEMX_ERROR_INTERNAL;
        }

        NVSHMEMI_WARN_PRINT("Received a CQE with error. RC: %d. Error: %d (%s)", err_entry.err, err_entry.prov_errno,
                            fi_cq_strerror(ep.cq, err_entry.prov_errno, err_entry.err_data, NULL, 0));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

/*
 * Drain CQ completions without calling nvshmemt_libfabric_progress.
 * Safe for try_again paths that cannot re-enter the top-level progress loop.
 */
/* Drain host EP CQs under host_ep_progress_lock. User-thread path (blocking=true)
 * acquires the lock; proxy path (blocking=false) try-locks and skips if the
 * user thread is already draining (they will make progress on their own).
 * Essence of upstream commit 756f773. */
static inline int progress_host_eps(nvshmem_transport_t transport, bool blocking) {
    nvshmemt_libfabric_state_t *state = (nvshmemt_libfabric_state_t *)transport->state;
    if (blocking) {
        state->host_ep_progress_lock.lock();
    } else if (!state->host_ep_progress_lock.try_lock()) {
        return 0; /* user thread is draining; skip */
    }
    int status = 0;
    for (int i = 0; i < state->num_host_domains; i++) {
        status = nvshmemt_libfabric_process_completion(transport, i);
        if (unlikely(status)) break;
    }
    state->host_ep_progress_lock.unlock();
    return status;
}

static int drain_completions(nvshmem_transport_t transport, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    int ep_start_idx;
    int ep_end_idx;
    int status = 0;

    /* Note this is only valid because we have 1 EP per domain */
    if (libfabric_state->use_auto_progress) {
        /* Auto - only progress EPs related to the qp_index */
        if (qp_index == NVSHMEMX_QP_HOST) {
            /* User-thread path: acquire host_ep_progress_lock blocking, drain host EPs,
             * then return (no proxy EPs in HOST path). */
            status = progress_host_eps(transport, /*blocking=*/true);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error in host progress: %d.\n", status);
            goto out;
        }
        /* Proxy path: try-drain host EPs (skip if user is draining), then drain proxy EPs. */
        status = progress_host_eps(transport, /*blocking=*/false);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Error in proxy-drain host progress: %d.\n", status);
        ep_start_idx = libfabric_state->num_host_domains;
        ep_end_idx = libfabric_state->eps.size();
    } else {
        /* Manual - progress all EPs */
        ep_start_idx = 0;
        ep_end_idx = libfabric_state->eps.size();
        qp_index = NVSHMEMX_QP_ALL;
        /* Setting qp_index to ensure all QPs are processed in gdr_process_amos below. */
    }

    for (int i = ep_start_idx; i < ep_end_idx; i++) {
        status = nvshmemt_libfabric_process_completion(transport, i);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to complete progress: %d.\n", status);
    }

out:
    return status;
}

static void *nvshmemt_libfabric_signal_delivery_thread(void *arg) {
    nvshmemt_libfabric_state_t *state = (nvshmemt_libfabric_state_t *)arg;
    nvshmem_transport_t transport = state->signal_delivery_transport;
    signal_delivery_work_entry work{};

    /* Inherit the process CPU affinity so the OS can schedule on any allowed core. */
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }

    while (!state->signal_delivery_stop.load(std::memory_order_relaxed)) {
        if (!state->signal_work_queue.pop(work)) {
            state->signal_delivery_futex.store(0, std::memory_order_release);
            /* Re-check after store to avoid missed wake. */
            if (!state->signal_work_queue.pop(work)) {
                syscall(SYS_futex, &state->signal_delivery_futex, FUTEX_WAIT, 0, NULL, NULL, 0);
                continue;
            }
        }
        int status = nvshmemt_libfabric_gdr_process_amo(transport, work.op, work.send_elems,
                                                        work.sequence_count,
                                                        work.preceding_put_count);
        if (unlikely(status)) {
            NVSHMEMI_WARN_PRINT("Failed call to nvshmemt_libfabric_gdr_process_amo: %d",
                                status);
            state->signal_delivery_status.store(status, std::memory_order_release);
            state->signal_delivery_stop.store(1, std::memory_order_release);
            break;
        }
    }
    return NULL;
}

/* Drain deferred work queue: signal ops that would recurse from completion path,
 * and standalone ack ops that would cause try_again -> progress -> completion
 * re-entry. PR#19. */
static int nvshmemt_libfabric_drain_deferred_work(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_deferred_work_t item;
    int status = 0;

    while (libfabric_state->deferred_work_queue.pop(item)) {
        if (auto *work = std::get_if<signal_delivery_work_entry>(&item)) {
            status = nvshmemt_libfabric_enqueue_signal_work(transport, *work);
        } else if (auto *ack = std::get_if<deferred_ack_entry>(&item)) {
            status = gdrcopy_amo_ack(transport, *ack->ep, ack->src_addr,
                                     ack->ack_payload.ack_seq_num,
                                     ack->ack_payload.ack_count,
                                     ack->ack_payload.ack_num_ops);
        }
        if (unlikely(status)) return status;
    }
    return 0;
}

/*
 * Top-level progress: drains completions, drains deferred work (PR#19),
 * drains completed signal-delivery work, then enqueues pending staged-AMOs.
 */

static int nvshmemt_libfabric_progress(nvshmem_transport_t transport, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    int status = drain_completions(transport, qp_index);
    if (unlikely(status)) return status;

    status = nvshmemt_libfabric_drain_deferred_work(transport);
    if (unlikely(status)) return status;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        int effective_qp = libfabric_state->use_auto_progress ? qp_index : NVSHMEMX_QP_ALL;
        status = check_signal_delivery_status(libfabric_state);
        if (unlikely(status)) return status;

        /* Serialize access to SPSC rings; both host and proxy threads may enter here. */
        while (libfabric_state->signal_progress_lock.test_and_set(std::memory_order_acquire)) {
            NVSHMEMT_LIBFABRIC_CPU_RELAX();
        }
        status = check_signal_delivery_status(libfabric_state);
        if (unlikely(status)) {
            libfabric_state->signal_progress_lock.clear(std::memory_order_release);
            return status;
        }

        /* Drain done_queue first: fi_recv + ACK, freeing space before new work. */
        status = nvshmemt_libfabric_gdr_complete_amos(transport);
        if (unlikely(status)) {
            libfabric_state->signal_progress_lock.clear(std::memory_order_release);
            return NVSHMEMX_ERROR_INTERNAL;
        }
        status = check_signal_delivery_status(libfabric_state);
        if (unlikely(status)) {
            libfabric_state->signal_progress_lock.clear(std::memory_order_release);
            return status;
        }

        /* Dequeue from op_queue, push to work_queue for signal delivery thread */
        status = nvshmemt_libfabric_gdr_process_amos(transport, effective_qp);
        if (unlikely(status)) {
            libfabric_state->signal_progress_lock.clear(std::memory_order_release);
            return NVSHMEMX_ERROR_INTERNAL;
        }

        /* Flush stale coalesced acks */
        if (effective_qp == NVSHMEMX_QP_HOST || effective_qp == NVSHMEMX_QP_ALL) {
            if (libfabric_state->host_signal_state.ack_aggregator) {
                status = libfabric_state->host_signal_state.ack_aggregator->flush_stale(
                    transport, *libfabric_state->eps[0]);
                if (status) {
                    libfabric_state->signal_progress_lock.clear(std::memory_order_release);
                    return NVSHMEMX_ERROR_INTERNAL;
                }
            }
        }

        if (effective_qp != NVSHMEMX_QP_HOST) {
            if (libfabric_state->proxy_signal_state.ack_aggregator) {
                status = libfabric_state->proxy_signal_state.ack_aggregator->flush_stale(
                    transport, *libfabric_state->eps[libfabric_state->num_host_domains]);
                if (status) {
                    libfabric_state->signal_progress_lock.clear(std::memory_order_release);
                    return NVSHMEMX_ERROR_INTERNAL;
                }
            }
        }

        libfabric_state->signal_progress_lock.clear(std::memory_order_release);

        status = check_signal_delivery_status(libfabric_state);
        if (unlikely(status)) return status;

        nvshmemt_libfabric_wake_signal_delivery_thread(libfabric_state);
    }
    return 0;
}

static int nvshmemt_libfabric_proxy_progress(nvshmem_transport_t transport) {
    return nvshmemt_libfabric_progress(transport, NVSHMEMX_QP_DEFAULT);
}

static int nvshmemt_libfabric_gdr_process_amo(nvshmem_transport_t transport,
                                              nvshmemt_libfabric_gdr_op_ctx_t *op,
                                              nvshmemt_libfabric_gdr_op_ctx_t **send_elems,
                                              uint32_t sequence_count,
                                              uint8_t preceding_put_count) {
    int status = 0;

    switch (op->send_amo.size) {
        case 2:
            status = perform_gdrcopy_amo<uint16_t>(transport, op, send_elems, sequence_count, preceding_put_count);
            break;
        case 4:
            status = perform_gdrcopy_amo<uint32_t>(transport, op, send_elems, sequence_count, preceding_put_count);
            break;
        case 8:
            status = perform_gdrcopy_amo<uint64_t>(transport, op, send_elems, sequence_count, preceding_put_count);
            break;
        default:
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "invalid element size encountered %u\n", op->send_amo.size);
    }

out:
    return status;
}

static int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2] = {NULL, NULL};
    size_t num_retries = 0;
    int domain_start_idx;
    int domain_end_idx;
    int status = 0;
    int max_ops = libfabric_state->proxy_request_batch_max;

    /* Drain non-signal AMO requests (e.g., fetch-AMO) from per-domain op_queues. */
    if (qp_index == NVSHMEMX_QP_HOST) {
        domain_start_idx = 0;
        domain_end_idx = libfabric_state->num_host_domains;
    } else if (qp_index == NVSHMEMX_QP_ALL) {
        domain_start_idx = 0;
        domain_end_idx = libfabric_state->domains.size();
    } else {
        domain_start_idx = libfabric_state->num_host_domains;
        domain_end_idx = libfabric_state->domains.size();
    }

    for (int i = domain_start_idx; i < domain_end_idx; i++) {
        int ops_processed = 0;
        do {
            do {
                send_elems[1] = nullptr;
                status = libfabric_state->op_queue[i]->getNextAmoOps(
                    send_elems, &op, NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK);
            } while (
                try_again(transport, &status, &num_retries,
                          NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_PROCESS_AMOS_GET_NEXT_NOT_ACK,
                          qp_index, progress_type::CompletionsOnly));
            num_retries = 0;
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "getNextAmoOps failed.\n");

            if (op) {
                ops_processed++;
                signal_delivery_work_entry work{};
                work.op = op;
                work.send_elems[0] = send_elems[0];
                work.send_elems[1] = send_elems[1];
                work.sequence_count = nvshmemt_libfabric_endpoint_seq_counter_t::last_seq_num;
                work.preceding_put_count = 0;
                status = nvshmemt_libfabric_enqueue_signal_work(transport, work);
                if (status) return status;
            }
        } while (op && ops_processed < max_ops);
    }

out:
    return status;
}

static int nvshmemt_libfabric_put_signal_completion(nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    const struct fi_cq_data_entry &entry,
                                                    const fi_addr_t &addr) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    int status = 0, pe;
    uint32_t map_seq;
    completion_kind kind = classify_completion(entry);

    auto [signal_state_p, sig_lk] = get_signal_state_locked(libfabric_state, ep);
    nvshmemt_libfabric_signal_state_t &signal_state = *signal_state_p;
    signal_seq_map *completion_map = nullptr;

    if (unlikely(kind == completion_kind::Invalid)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Received staged completion with invalid header type.\n");
    }

    if (unlikely(addr == FI_ADDR_NOTAVAIL)) {
        status = -1;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Write w/imm returned with invalid src address.\n");
    }

    pe = convert_addr_to_pe(libfabric_state, ep, addr);
    completion_map = &signal_state.proxy_put_signal_comp_map[pe];

    switch (kind) {
        case completion_kind::StandalonePutAckReq: {
            map_seq =
                static_cast<uint32_t>(entry.data) & nvshmemt_libfabric_endpoint_seq_counter_t::bit_mask;
            uint8_t put_count =
                (static_cast<uint32_t>(entry.data) >>
                 nvshmemt_libfabric_endpoint_seq_counter_t::bit_shift) &
                0xFF;
            nvshmemt_libfabric_comp_entry_t ack_comp_entry =
                nvshmemt_libfabric_put_ack_entry{addr, ep.ep_index, put_count};
            auto insert_result = completion_map->insert(map_seq, ack_comp_entry);
            if (!insert_result.second) {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "duplicate ack entry while processing put completion.\n");
            }
            break;
        }

        case completion_kind::StandalonePut: {
            map_seq =
                static_cast<uint32_t>(entry.data) & nvshmemt_libfabric_endpoint_seq_counter_t::bit_mask;
            const auto [drainable, signal_status] =
                accumulate_signal(*completion_map, map_seq, nullptr, 0);
            status = signal_status;
            if (status) goto out;
            if (!drainable) goto out;
            break;
        }

        case completion_kind::PutSignalWrite: {
            map_seq =
                static_cast<uint32_t>(entry.data) & nvshmemt_libfabric_endpoint_seq_counter_t::bit_mask;
            const auto [drainable, signal_status] =
                accumulate_signal(*completion_map, map_seq, nullptr, -1);
            status = signal_status;
            if (status) goto out;
            if (!drainable) goto out;
            break;
        }

        case completion_kind::PutSignalSignal: {
            nvshmemt_libfabric_gdr_signal_op *sig_op =
                reinterpret_cast<nvshmemt_libfabric_gdr_signal_op *>(
                    container_of(entry.op_context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context));
            map_seq = sig_op->sequence_count;
            int progress_count = static_cast<int>(sig_op->num_writes);
            nvshmemt_libfabric_ack_payload_t piggyback_ack = sig_op->ack;

            /* The EFA provider has an inline send size of 32 bytes.
             * The gdr atomic fi_send message is 72 bytes and does not
             * fit inside the efa_provider's 32 byte inline send window.
             * Hence, we send a 32 byte nvshmemt_libfabric_gdr_signal_op over the wire,
             * and re-arrange the memory in-place to allow for re-use of the gdr atomic
             * code.
             */
            nvshmemt_libfabric_gdr_op_ctx_t *op =
                inplace_copy_sig_op_to_gdr_op(sig_op, ep.ep_index);
            nvshmemt_libfabric_apply_ack(signal_state, pe, piggyback_ack);
            const auto [drainable, signal_status] =
                accumulate_signal(*completion_map, map_seq, op, progress_count);
            status = signal_status;
            if (status) goto out;
            if (!drainable) goto out;
            break;
        }

        case completion_kind::Invalid:
            assert(false && "invalid completion kind should be handled before dispatch");
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "Received staged completion with invalid header type.\n");
    }

    {
        // next_expected_seq is sized by PE during endpoint setup.
        uint32_t &next_seq = signal_state.next_expected_seq[pe];

        while (true) {
            auto *it = completion_map->find(next_seq);

            if (!it) break;

            if (auto *signal_entry = std::get_if<nvshmemt_libfabric_signal_comp_entry>(it)) {
                if (signal_entry->progress_count != 0) break;

                if (signal_entry->op != NULL) {
                    /* Defer signal delivery to avoid recursion:
                     * enqueue_signal_work -> try_again -> process_completions
                     * -> put_signal_completion would re-enter this path. */
                    nvshmemt_libfabric_gdr_op_ctx_t *sig_op = signal_entry->op;
                    signal_delivery_work_entry work{};
                    work.op = sig_op;
                    work.send_elems[0] = NULL;
                    work.send_elems[1] = NULL;
                    work.sequence_count = sig_op->send_amo.sequence_count;
                    work.preceding_put_count = sig_op->send_amo.preceding_put_count;
                    libfabric_state->deferred_work_queue.push(work);
                }
            } else if (auto *ack_entry = std::get_if<nvshmemt_libfabric_put_ack_entry>(it)) {
                /* Defer ACK send to avoid recursion (PR#19):
                 * gdrcopy_amo_ack -> try_again -> process_completions
                 * -> put_signal_completion would re-enter this path. */
                deferred_ack_entry ack{};
                ack.ep = libfabric_state->eps[ack_entry->ep_index].get();
                ack.src_addr = ack_entry->src_addr;
                ack.ack_payload = {(uint16_t)next_seq, ack_entry->put_count, 1};
                libfabric_state->deferred_work_queue.push(ack);
            } else {
                assert(false && "unexpected completion entry type");
            }

            completion_map->erase(next_seq);
            next_seq = nvshmemt_libfabric_endpoint_seq_counter_t::seq_num_wrapup(next_seq + 1);
        }
    }

out:
    return status;
}

static int nvshmemt_libfabric_quiet(struct nvshmem_transport *tcurr, int /*pe*/, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(tcurr);
    int ep_start_idx;
    int ep_end_idx;
    int status = 0;

    /* Note this is only valid because we have 1 EP per domain */
    if (qp_index == NVSHMEMX_QP_HOST) {
        ep_start_idx = 0;
        ep_end_idx = libfabric_state->num_host_domains;
    } else {
        ep_start_idx = libfabric_state->num_host_domains;
        ep_end_idx = libfabric_state->eps.size();
    }

    const nvshmemt_libfabric_signal_state_t &signal_state =
        (qp_index == NVSHMEMX_QP_HOST) ? libfabric_state->host_signal_state
                                       : libfabric_state->proxy_signal_state;

    for (;;) {
        /* Force-flush all pending ACKs so they get sent before checking quiescence */
        if (signal_state.ack_aggregator) {
            int ep_idx = (qp_index == NVSHMEMX_QP_HOST) ? 0 : libfabric_state->num_host_domains;
            status = signal_state.ack_aggregator->flush_all(tcurr, *libfabric_state->eps[ep_idx]);
            if (unlikely(status)) break;
        }

        uint64_t total_submitted = 0;
        uint64_t total_completed = 0;
        {
            /* The proxy thread may modify host EP counters via drain_deferred_work
             * (gdrcopy_amo_ack -> submitted_ops++) while holding host_ep_progress_lock.
             * Take the same lock to get a consistent snapshot. */
            host_ep_submit_guard _quiet_guard(libfabric_state,
                                              *libfabric_state->eps[ep_start_idx]);
            for (int i = ep_start_idx; i < ep_end_idx; i++) {
                total_submitted += libfabric_state->eps[i]->submitted_ops;
                total_completed += libfabric_state->eps[i]->completed_ops;
            }
            total_completed += signal_state.completed_staged_atomics;
        }
        if (total_submitted == total_completed) {
            break;
        }

        if (nvshmemt_libfabric_progress(tcurr, qp_index)) {
            status = NVSHMEMX_ERROR_INTERNAL;
            break;
        }
    }

    return status;
}

static int nvshmemt_libfabric_fence(struct nvshmem_transport *tcurr, int pe, int qp_index,
                                    int /*is_multi*/) {
    int status = nvshmemt_libfabric_quiet(tcurr, pe, qp_index);

    return status;
}

static int nvshmemt_libfabric_show_info(struct nvshmem_transport * /*transport*/, int /*style*/) {
    NVSHMEMI_ERROR_PRINT("libfabric show info not implemented");
    return 0;
}

static int nvshmemt_libfabric_rma_impl(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                       rma_memdesc_t *remote, rma_memdesc_t *local,
                                       rma_bytesdesc_t bytesdesc, int /*qp_index*/,
                                       const nvshmem_transport_op_attrs_t *attrs,
                                       uint32_t *imm_data, nvshmemt_libfabric_endpoint_t &ep) {
    nvshmemt_libfabric_mem_handle_ep_t *remote_handle, *local_handle = NULL;
    void *local_mr_desc = NULL;
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(tcurr);
    size_t op_size;
    uint64_t num_retries = 0;
    int status = 0;
    int target_ep, ep_idx, domain_idx;
    void *context = NULL;

    ep_idx = ep.ep_index;
    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep_idx;
    int slot_impl = (ep.qp_index == NVSHMEMX_QP_HOST) ? 0 : 1;
    bool put_more_follows = verb.desc == NVSHMEMI_OP_PUT && attrs &&
                            (attrs->flags & NVSHMEM_TRANSPORT_OP_FLAG_MORE_FOLLOWS);

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        nvshmemt_libfabric_gdr_op_ctx_t *gdr_ctx;
        do {
            status = libfabric_state->op_queue[domain_idx]->getNextSends(&gdr_ctx, 1);
        } while (try_again(tcurr, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_GET_NEXT_SENDS,
                           ep.qp_index, progress_type::All));
        NVSHMEMI_NULL_ERROR_JMP(gdr_ctx, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "Unable to get context buffer for put request.\n");
        context = &gdr_ctx->ofi_context;
        gdr_ctx->type = NVSHMEMT_LIBFABRIC_RMA;

        /* local->handle may be NULL for small operations (P ops) sent by value/inline */
        if (likely(local->handle != NULL)) {
            local_handle = &((nvshmemt_libfabric_mem_handle_t *)local->handle)->hdls[domain_idx];
            local_mr_desc = local_handle->local_desc;
        }
    }

    remote_handle = &((nvshmemt_libfabric_mem_handle_t *)remote->handle)->hdls[domain_idx];
    op_size = bytesdesc.elembytes * bytesdesc.nelems;

    /* HOT PATH: OP_PUT is the common case, check it first with likely */
    if (likely(verb.desc == NVSHMEMI_OP_PUT)) {
        uintptr_t remote_addr;
        if (libfabric_state->prov_infos[domain_idx]->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
            remote_addr = (uintptr_t)remote->ptr;
        else
            remote_addr = (uintptr_t)remote->offset;

        if (put_more_follows) {
            struct iovec p_op_l_iov = {};
            struct fi_msg_rma p_op_msg = {};
            struct fi_rma_iov p_op_r_iov = {};
            uint64_t msg_flags = FI_MORE;

            if (imm_data) {
                p_op_msg.data = *imm_data;
                msg_flags |= FI_REMOTE_CQ_DATA;
            }

            p_op_msg.msg_iov = &p_op_l_iov;
            p_op_msg.desc = &local_mr_desc;
            p_op_msg.iov_count = 1;
            p_op_msg.addr = target_ep;
            p_op_msg.rma_iov = &p_op_r_iov;
            p_op_msg.rma_iov_count = 1;
            p_op_msg.context = context;

            p_op_l_iov.iov_base = local->ptr;
            p_op_l_iov.iov_len = op_size;

            p_op_r_iov.addr = remote_addr;
            p_op_r_iov.len = op_size;
            p_op_r_iov.key = remote_handle->key;

            do {
                status = fi_writemsg(ep.endpoint, &p_op_msg, msg_flags);
            } while (try_again(tcurr, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_PUT,
                               ep.qp_index, progress_type::All));
            libfabric_state->pending_batch_ep[slot_impl] = status ? -1 : ep_idx;
        } else {
            do {
                if (likely(imm_data != NULL)) {
                    status =
                        fi_writedata(ep.endpoint, local->ptr, op_size, local_mr_desc, *imm_data,
                                     target_ep, remote_addr, remote_handle->key, context);
                } else
                    status = fi_write(ep.endpoint, local->ptr, op_size, local_mr_desc, target_ep,
                                      remote_addr, remote_handle->key, context);
            } while (try_again(tcurr, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_PUT,
                               ep.qp_index, progress_type::All));
        }
    } else if (likely(verb.desc == NVSHMEMI_OP_P)) {
        if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            nvshmemt_libfabric_gdr_op_ctx_t *p_buf =
                container_of(context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context);
            num_retries = 0;
            p_buf->p_op.value = *(uint64_t *)local->ptr;
            assert(imm_data);
            do {
                status =
                    fi_writedata(ep.endpoint, &p_buf->p_op.value, op_size,
                                 fi_mr_desc(libfabric_state->mrs[domain_idx]), *imm_data,
                                 target_ep, (uintptr_t)remote->ptr, remote_handle->key, context);
            } while (try_again(tcurr, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_EFA,
                               ep.qp_index, progress_type::All));
        } else {
            struct iovec p_op_l_iov = {};
            struct fi_msg_rma p_op_msg = {};
            struct fi_rma_iov p_op_r_iov = {};

            p_op_msg.msg_iov = &p_op_l_iov;
            p_op_msg.desc = NULL;  // Local buffer is on the stack
            p_op_msg.iov_count = 1;
            p_op_msg.addr = target_ep;
            p_op_msg.rma_iov = &p_op_r_iov;
            p_op_msg.rma_iov_count = 1;

            p_op_l_iov.iov_base = local->ptr;
            p_op_l_iov.iov_len = op_size;

            if (libfabric_state->prov_infos[domain_idx]->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
                p_op_r_iov.addr = (uintptr_t)remote->ptr;
            else
                p_op_r_iov.addr = (uintptr_t)remote->offset;
            p_op_r_iov.len = op_size;
            p_op_r_iov.key = remote_handle->key;

            /* The p buffer is on the stack so use
             * FI_INJECT to avoid segfaults during async runs.
             */
            do {
                status = fi_writemsg(ep.endpoint, &p_op_msg, FI_INJECT);
            } while (try_again(tcurr, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_NON_EFA,
                               ep.qp_index, progress_type::All));
        }
    } else if (likely(verb.desc == NVSHMEMI_OP_G || verb.desc == NVSHMEMI_OP_GET)) {
        assert(
            !imm_data);  // Write w/ imm not suppored with NVSHMEMI_OP_G/GET on Libfabric transport
        uintptr_t remote_addr;
        if (libfabric_state->prov_infos[domain_idx]->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
            remote_addr = (uintptr_t)remote->ptr;
        else
            remote_addr = (uintptr_t)remote->offset;

        do {
            status = fi_read(ep.endpoint, local->ptr, op_size, local_mr_desc, target_ep,
                             remote_addr, remote_handle->key, context);
        } while (try_again(tcurr, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_GET, ep.qp_index,
                           progress_type::All));
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid RMA operation specified.\n");
    }

out:
    if (!put_more_follows) libfabric_state->pending_batch_ep[slot_impl] = -1;
    ep.submitted_ops++;

    if (unlikely(status)) {
        ep.submitted_ops--;
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an RMA operation: %d (%s).\n",
                             status, fi_strerror(status < 0 ? -status : status));
    }

    return status;
}

static int nvshmemt_libfabric_rma_with_hints(struct nvshmem_transport *tcurr, int pe,
                                             rma_verb_t verb, rma_memdesc_t *remote,
                                             rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                                             int qp_index,
                                             const nvshmem_transport_op_attrs_t *attrs) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(tcurr);
    uint32_t imm_data_val = 0;
    uint32_t *imm_data = NULL;
    int status = 0;
    /*
     * Use the same EP while FI_MORE doorbell is deferred (RMA only). Slot by caller role so host
     * and proxy do not interfere.
     */
    int slot = (qp_index == NVSHMEMX_QP_HOST) ? 0 : 1;
    int ep_idx = (libfabric_state->pending_batch_ep[slot] >= 0)
                     ? libfabric_state->pending_batch_ep[slot]
                     : get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);

    // Generate sequence number for P and PUT operations when ordering is needed
    host_ep_submit_guard host_guard(libfabric_state, ep);
    if (libfabric_state->use_staged_atomics &&
        (verb.desc == NVSHMEMI_OP_P || verb.desc == NVSHMEMI_OP_PUT)) {

        auto [signal_state, sig_lk] = get_signal_state_locked(libfabric_state, ep);
        auto &seq_counter = signal_state->put_signal_seq_counter[pe];
        uint32_t sequence_count;

        status = get_next_seq_num_with_retry(tcurr, seq_counter, &sequence_count, qp_index,
                                             NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_PUT);
        if (status) return status;

        seq_counter.put_count++;

        nvshmemt_libfabric_imm_cq_data_hdr_t header;
        uint8_t put_ack_count = 0;
        if (seq_counter.put_count >= seq_counter.put_ack_freq) {
            /* Make sure we didn't accidentally increment it anywhere else */
            assert(seq_counter.put_count == seq_counter.put_ack_freq);
            put_ack_count = seq_counter.put_count;
            header = NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT_WITH_ACK_REQ;
            seq_counter.put_count = 0;
            ep.submitted_ops++;  // Account for incoming ack
        } else {
            header = NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT;
        }

        imm_data_val = (static_cast<uint32_t>(header) << imm_header_bit_shift) |
                       (static_cast<uint32_t>(put_ack_count) <<
                            nvshmemt_libfabric_endpoint_seq_counter_t::bit_shift) |
                       (sequence_count & nvshmemt_libfabric_endpoint_seq_counter_t::bit_mask);
        imm_data = &imm_data_val;
    }

    return nvshmemt_libfabric_rma_impl(tcurr, pe, verb, remote, local, bytesdesc, qp_index, attrs,
                                       imm_data, ep);
}

static int nvshmemt_libfabric_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                  rma_memdesc_t *remote, rma_memdesc_t *local,
                                  rma_bytesdesc_t bytesdesc, int qp_index) {
    return nvshmemt_libfabric_rma_with_hints(tcurr, pe, verb, remote, local, bytesdesc, qp_index,
                                             nullptr);
}

static int nvshmemt_libfabric_gdr_signal(struct nvshmem_transport *transport, int pe,
                                         void *curetptr, amo_verb_t verb, amo_memdesc_t *remote,
                                         amo_bytesdesc_t bytesdesc, int qp_index,
                                         uint32_t sequence_count, uint16_t num_writes,
                                         nvshmemt_libfabric_endpoint_t &ep,
                                         uint8_t preceding_put_count);

static int nvshmemt_libfabric_gdr_amo(struct nvshmem_transport *transport, int pe, void *curetptr,
                                      amo_verb_t verb, amo_memdesc_t *remote,
                                      amo_bytesdesc_t bytesdesc, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    uint64_t num_retries = 0;
    int target_ep, ep_idx, domain_idx;
    int status = 0;

    ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);
    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep_idx;

    /* Signal-only operations use gdr_signal path with num_writes=0 */
    if (is_signal_only_op(verb.desc)) {
        auto [signal_state, sig_lk] = get_signal_state_locked(libfabric_state, ep);
        auto &seq_counter = signal_state->put_signal_seq_counter[pe];
        uint32_t sequence_count = 0;
        status = get_next_seq_num_with_retry(transport, seq_counter, &sequence_count, qp_index,
                                             NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_GET_NEXT_SENDS);
        if (status) goto out;
        assert(sequence_count != nvshmemt_libfabric_endpoint_seq_counter_t::last_seq_num);

        assert(seq_counter.put_count <= seq_counter.put_ack_freq);
        uint8_t ppc = seq_counter.put_count;
        seq_counter.put_count = 0;

        {
            host_ep_submit_guard host_guard(libfabric_state, ep);
            status = nvshmemt_libfabric_gdr_signal(transport, pe, curetptr, verb, remote,
                                                   bytesdesc, qp_index, sequence_count, 0, ep, ppc);
        }
        goto out;
    }

    /* Fetch operations use full gdr_op_ctx_t */
    nvshmemt_libfabric_gdr_op_ctx_t *amo;
    do {
        status = libfabric_state->op_queue[domain_idx]->getNextSends(&amo, 1);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_GET_NEXT_SENDS, qp_index,
                       progress_type::All));
    NVSHMEMI_NULL_ERROR_JMP(amo, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to retrieve AMO operation.");

    amo->send_amo.op =
        (nvshmemi_amo_t)(verb.desc | (verb.is_float ? NVSHMEMI_AMO_FLOAT_BIT : 0));
    amo->send_amo.target_addr = remote->remote_memdesc.ptr;
    amo->send_amo.ret_addr = remote->retptr;
    amo->send_amo.retflag = remote->retflag;
    amo->send_amo.swap_add = remote->val;
    amo->send_amo.size = bytesdesc.elembytes;
    amo->send_amo.src_pe = transport->my_pe;
    amo->send_amo.comp = remote->cmp;
    amo->type = NVSHMEMT_LIBFABRIC_SEND;

    {
        host_ep_submit_guard host_guard(libfabric_state, ep);
        num_retries = 0;
        do {
            status =
                fi_send(ep.endpoint, (void *)amo, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                        fi_mr_desc(libfabric_state->mrs[domain_idx]), target_ep, &amo->ofi_context);
        } while (try_again(transport, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_SEND, qp_index, progress_type::All));

        if (status) {
            NVSHMEMI_ERROR_PRINT("Received an error when trying to post an AMO operation.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
        } else {
            ep.submitted_ops += 2;
        }
    }

out:
    return status;
}

static int nvshmemt_libfabric_amo(struct nvshmem_transport *transport, int pe, void * /*curetptr*/,
                                  amo_verb_t verb, amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc,
                                  int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_mem_handle_ep_t *remote_handle = NULL, *local_handle = NULL;
    struct fi_msg_atomic amo_msg;
    struct fi_ioc fi_local_iov;
    struct fi_ioc fi_comp_iov;
    struct fi_ioc fi_ret_iov;
    struct fi_rma_ioc fi_remote_iov;
    enum fi_datatype data;
    enum fi_op op;
    uint64_t num_retries = 0;
    int target_ep;
    int ep_idx;
    int domain_idx;
    int status = 0;

    memset(&amo_msg, 0, sizeof(struct fi_msg_atomic));
    memset(&fi_local_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_comp_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_ret_iov, 0, sizeof(struct fi_ioc));
    memset(&fi_remote_iov, 0, sizeof(struct fi_rma_ioc));

    ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);
    host_ep_submit_guard host_guard(libfabric_state, ep);
    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep_idx;

    remote_handle =
        &((nvshmemt_libfabric_mem_handle_t *)remote->remote_memdesc.handle)->hdls[domain_idx];
    if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
        local_handle = &((nvshmemt_libfabric_mem_handle_t *)remote->ret_handle)->hdls[domain_idx];
    }

    if (bytesdesc.elembytes == 8) {
        data = FI_UINT64;
    } else if (bytesdesc.elembytes == 4) {
        data = FI_UINT32;
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid atomic size specified.\n");
    }

    switch (verb.desc) {
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET: {
            op = FI_ATOMIC_WRITE;
            break;
        }
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC:
        case NVSHMEMI_AMO_FETCH_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD: {
            op = FI_SUM;
            break;
        }
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND: {
            op = FI_BAND;
            break;
        }
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR: {
            op = FI_BOR;
            break;
        }
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR: {
            op = FI_BXOR;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            op = FI_ATOMIC_READ;
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            op = FI_CSWAP;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out, "Opcode %d is invalid.\n",
                               verb.desc);
        }
    }

    if (op != FI_ATOMIC_READ) {
        fi_local_iov.addr = &remote->val;
        fi_local_iov.count = 1;
        amo_msg.msg_iov = &fi_local_iov;
        amo_msg.desc = NULL;  // Local operands are on the stack
        amo_msg.iov_count = 1;
    }

    amo_msg.addr = target_ep;

    if (libfabric_state->prov_infos[domain_idx]->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
        fi_remote_iov.addr = (uintptr_t)remote->remote_memdesc.ptr;
    else
        fi_remote_iov.addr = (uintptr_t)remote->remote_memdesc.offset;

    fi_remote_iov.count = 1;
    fi_remote_iov.key = remote_handle->key;
    amo_msg.rma_iov = &fi_remote_iov;
    amo_msg.rma_iov_count = 1;

    amo_msg.datatype = data;
    amo_msg.op = op;

    amo_msg.context = NULL;
    amo_msg.data = 0;

    if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
        fi_ret_iov.addr = remote->retptr;
        fi_ret_iov.count = 1;
        if (verb.desc == NVSHMEMI_AMO_COMPARE_SWAP) {
            fi_comp_iov.addr = &remote->cmp;
            fi_comp_iov.count = 1;
        }
    }

    do {
        if (verb.desc == NVSHMEMI_AMO_COMPARE_SWAP) {
            status = fi_compare_atomicmsg(ep.endpoint, &amo_msg, &fi_comp_iov, NULL, 1, &fi_ret_iov,
                                          &local_handle->local_desc, 1, FI_INJECT);
        } else if (verb.desc < NVSHMEMI_AMO_END_OF_NONFETCH) {
            status = fi_atomicmsg(ep.endpoint, &amo_msg, op == FI_ATOMIC_READ ? 0 : FI_INJECT);
        } else {
            status = fi_fetch_atomicmsg(ep.endpoint, &amo_msg, &fi_ret_iov,
                                        &local_handle->local_desc, 1, FI_INJECT);
        }
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_AMO_ATOMICMSG, qp_index,
                       progress_type::All));

    if (status) goto out;  // Status set by try_again

    ep.submitted_ops++;

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an AMO operation.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
    }
    return status;
}

static int nvshmemt_libfabric_gdr_signal(struct nvshmem_transport *transport, int pe,
                                         void * /*curetptr*/, amo_verb_t verb,
                                         amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc,
                                         int /*qp_index*/, uint32_t sequence_count,
                                         uint16_t num_writes, nvshmemt_libfabric_endpoint_t &ep,
                                         uint8_t preceding_put_count) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(transport);
    nvshmemt_libfabric_gdr_op_ctx_t *context;
    nvshmemt_libfabric_gdr_signal_op_t *signal;
    uint64_t num_retries = 0;
    int target_ep;
    int domain_idx;
    int status = 0;

    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep.ep_index;

    static_assert(sizeof(nvshmemt_libfabric_gdr_op_ctx) >=
                  sizeof(nvshmemt_libfabric_gdr_signal_op_t));
    do {
        status = libfabric_state->op_queue[domain_idx]->getNextSends(&context, 1);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_GET_NEXT_SENDS,
                       ep.qp_index, progress_type::All));
    NVSHMEMI_NULL_ERROR_JMP(context, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to retrieve signal operation buffer.");

    signal = (nvshmemt_libfabric_gdr_signal_op_t *)context;
    signal->type = NVSHMEMT_LIBFABRIC_MATCH;
    signal->op =
        static_cast<uint8_t>(verb.desc | (verb.is_float ? NVSHMEMI_AMO_FLOAT_BIT : 0));
    signal->elem_size = static_cast<uint8_t>(bytesdesc.elembytes);
    signal->sequence_count = sequence_count;
    signal->target_addr = remote->remote_memdesc.ptr;
    signal->sig_val = remote->val;
    signal->num_writes = num_writes;
    signal->src_pe = transport->my_pe;
    signal->preceding_put_count = preceding_put_count;

    /* Piggyback pending ACK for this destination PE if available */
    {
        auto [signal_state_p, sig_lk] = get_signal_state_locked(libfabric_state, ep);
        nvshmemt_libfabric_signal_state_t &signal_state = *signal_state_p;
        uint16_t ack_range_end;
        uint8_t ack_range_count, ack_signal_count;

        if (signal_state.ack_aggregator &&
            signal_state.ack_aggregator->try_extract_for_peer(
                pe, ack_range_end, ack_range_count, ack_signal_count)) {
            signal->ack.ack_seq_num = ack_range_end;
            signal->ack.ack_count = ack_range_count;
            signal->ack.ack_num_ops = ack_signal_count;
        } else {
            signal->ack.ack_seq_num = 0;
            signal->ack.ack_count = 0;
            signal->ack.ack_num_ops = 0;
        }
        signal->reserved = 0;
    }

    num_retries = 0;
    do {
        status =
            fi_send(ep.endpoint, (void *)signal, sizeof(nvshmemt_libfabric_gdr_signal_op_t),
                    fi_mr_desc(libfabric_state->mrs[domain_idx]), target_ep, &context->ofi_context);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_SEND, ep.qp_index,
                       progress_type::All));

    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post a signal operation.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
    } else {
        ep.submitted_ops += 2;
    }

out:
    return status;
}

static int nvshmemt_libfabric_put_signal_unordered(struct nvshmem_transport *tcurr, int pe,
                                                   rma_verb_t write_verb,
                                                   std::vector<rma_memdesc_t> &write_remote,
                                                   std::vector<rma_memdesc_t> &write_local,
                                                   std::vector<rma_bytesdesc_t> &write_bytes_desc,
                                                   amo_verb_t sig_verb, amo_memdesc_t *sig_target,
                                                   amo_bytesdesc_t sig_bytes_desc, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(tcurr);
    uint32_t sequence_count = 0;
    int status = 0;
    uint8_t ppc = 0;

    int ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);

    {
        auto [signal_state, sig_lk] = get_signal_state_locked(libfabric_state, ep);
        auto &seq_counter = signal_state->put_signal_seq_counter[pe];

        /* Get sequence number for this put-signal, with retry */
        status = get_next_seq_num_with_retry(tcurr, seq_counter, &sequence_count, qp_index,
                                             NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PUT_SIGNAL_UNORDERED_SEQ);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT("Error in nvshmemt_put_signal_unordered while waiting for category\n");
            goto out;
        }

        assert(seq_counter.put_count <= seq_counter.put_ack_freq);
        ppc = seq_counter.put_count;
        seq_counter.put_count = 0;
    }

    assert(write_remote.size() == write_local.size() &&
           write_local.size() == write_bytes_desc.size());
    {
        host_ep_submit_guard host_guard(libfabric_state, ep);
        for (size_t i = 0; i < write_remote.size(); i++) {
            status =
                nvshmemt_libfabric_rma_impl(tcurr, pe, write_verb, &write_remote[i], &write_local[i],
                                            write_bytes_desc[i], qp_index, nullptr,
                                            &sequence_count, ep);
            if (unlikely(status)) {
                NVSHMEMI_ERROR_PRINT(
                    "Error in nvshmemt_put_signal_unordered, could not submit write #%lu\n", i);
                goto out;
            }
        }

        assert(libfabric_state->use_staged_atomics == true);
        status =
            nvshmemt_libfabric_gdr_signal(tcurr, pe, NULL, sig_verb, sig_target, sig_bytes_desc,
                                          qp_index, sequence_count, (uint16_t)write_remote.size(), ep, ppc);
    }
out:
    if (status) {
        NVSHMEMI_ERROR_PRINT(
            "Received an error when trying to perform a nvshmem_proxy_put_signal_unordered "
            "operation.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
    }
    return status;
}

static int nvshmemt_libfabric_enforce_cst(struct nvshmem_transport *tcurr) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(tcurr);
    uint64_t num_retries = 0;
    int qp_index;
    int domain_idx;
    int ep_idx_start;
    int ep_idx_end;
    int status;

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        if (libfabric_state->provider != NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
            int temp;
            nvshmemt_libfabric_memhandle_info_t *mem_handle_info;

            mem_handle_info =
                (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(
                    libfabric_state->cache, 0);
            if (!mem_handle_info) {
                goto skip;
            }
            gdrcopy_ftable.copy_from_mapping(mem_handle_info->mh, &temp, mem_handle_info->cpu_ptr,
                                             sizeof(int));
        }
    }

skip:
#endif

    /* Only proxy EPs */
    ep_idx_start = libfabric_state->num_host_domains;
    ep_idx_end = ep_idx_start + libfabric_state->num_proxy_domains;

    for (int ep_idx = ep_idx_start; ep_idx < ep_idx_end; ep_idx++) {
        num_retries = 0;
        qp_index = libfabric_state->eps[ep_idx]->qp_index;
        domain_idx = libfabric_state->eps[ep_idx]->domain_index;
        do {
            struct fi_msg_rma msg;
            struct iovec l_iov;
            struct fi_rma_iov r_iov;
            void *desc = libfabric_state->local_mr_descs[domain_idx];
            uint64_t flags = 0;

            memset(&msg, 0, sizeof(struct fi_msg_rma));
            memset(&l_iov, 0, sizeof(struct iovec));
            memset(&r_iov, 0, sizeof(struct fi_rma_iov));

            l_iov.iov_base = libfabric_state->local_mem_ptr;
            l_iov.iov_len = 8;

            r_iov.addr = 0;  // Zero offset
            r_iov.len = 8;
            r_iov.key = libfabric_state->local_mr_keys[domain_idx];

            msg.msg_iov = &l_iov;
            msg.desc = &desc;
            msg.iov_count = 1;
            msg.rma_iov = &r_iov;
            msg.rma_iov_count = 1;
            msg.context = NULL;
            msg.data = 0;

            if (libfabric_state->prov_infos[domain_idx]->caps & FI_FENCE) flags |= FI_FENCE;

            status = fi_readmsg(libfabric_state->eps[ep_idx]->endpoint, &msg, flags);
        } while (try_again(tcurr, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_ENFORCE_CST, qp_index,
                           progress_type::All));

        libfabric_state->eps[ep_idx]->submitted_ops++;

        /* If try_again errors out, need to break for upper-layer to abort */
        if (unlikely(status != 0)) break;
    }

    return status;
}

static int nvshmemt_libfabric_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                                 nvshmem_transport_t t) {
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(t);
    nvshmemt_libfabric_mem_handle_t *fabric_handle;
    void *curr_ptr;
    int status = 0;

    assert(mem_handle != NULL);
    fabric_handle = (nvshmemt_libfabric_mem_handle_t *)mem_handle;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        nvshmemt_libfabric_memhandle_info_t *handle_info;
        handle_info = (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get(
            t, libfabric_state->cache, fabric_handle->buf);
        if (handle_info != NULL) {
#ifdef NVSHMEM_USE_GDRCOPY
            if ((use_gdrcopy == true) && (handle_info->gdr_mapping_size > 0)) {
                status = gdrcopy_ftable.unmap(gdr_desc, handle_info->mh, handle_info->cpu_ptr_base,
                                              handle_info->gdr_mapping_size);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

                status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info->mh);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");
            }
#endif
            if (libfabric_state->cache != NULL) {
                curr_ptr = handle_info->ptr;
                do {
                    nvshmemt_mem_handle_cache_remove(t, libfabric_state->cache, curr_ptr);
                    curr_ptr = (char *)curr_ptr + (1ULL << t->log2_cumem_granularity);
                } while (curr_ptr < (char *)handle_info->ptr + handle_info->gdr_mapping_size);
            }
        }
    }

    for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
        if (libfabric_state->local_mrs[i] == fabric_handle->hdls[i].mr)
            libfabric_state->local_mrs[i] = NULL;

        int status = fi_close(&fabric_handle->hdls[i].mr->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Error releasing mem handle idx %zu (%d): %s\n", i, status,
                                fi_strerror(status * -1));
        }
    }

    libfabric_state->local_mrs.clear();
    libfabric_state->local_mr_keys.clear();
    libfabric_state->local_mr_descs.clear();

out:
    return status;
}

static int nvshmemt_libfabric_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf,
                                             size_t length, nvshmem_transport_t t,
                                             bool local_only) {
    nvshmemt_libfabric_mem_handle_t *fabric_handle;
    nvshmemt_libfabric_state_t *libfabric_state = get_libfabric_state(t);
    cudaPointerAttributes attr = {};
    struct fi_mr_attr mr_attr;
    struct iovec mr_iovec;
    int status;
    bool is_host = true;
    void *curr_ptr;
    CUdevice gpu_device_id;
    nvshmemt_libfabric_memhandle_info_t *handle_info = NULL;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_info_t info;
#endif

    // for now, error out if mmap is used with libfabric
    // TODO : Add workaround for mmap with libfabric
    status = (t->alias_va_map != NULL && t->alias_va_map->count(buf));
    if (status) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "mmap symmetric buf %p not supported with libfabric currently. Please "
                           "use nvshmem_malloc to allocate symmetric buf\n",
                           buf);
    }

    status = CUPFN(libfabric_state->table, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    assert(mem_handle != NULL);
    fabric_handle = (nvshmemt_libfabric_mem_handle_t *)mem_handle;
    fabric_handle->buf = buf;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to query pointer attributes.\n");
    }

    if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) {
        is_host = false;
    }

    memset(&mr_attr, 0, sizeof(struct fi_mr_attr));
    memset(&mr_iovec, 0, sizeof(struct iovec));

    mr_iovec.iov_base = buf;
    mr_iovec.iov_len = length;
    mr_attr.mr_iov = &mr_iovec;
    mr_attr.iov_count = 1;
    mr_attr.access = FI_READ | FI_WRITE;
    if (!local_only) {
        mr_attr.access |= FI_REMOTE_READ | FI_REMOTE_WRITE;
    }
    mr_attr.offset = 0;
    mr_attr.context = NULL;
    if (!is_host) {
        mr_attr.iface = FI_HMEM_CUDA;
        mr_attr.device.cuda = gpu_device_id;
    } else {
        mr_attr.iface = FI_HMEM_SYSTEM;
    }

    for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
        if (libfabric_state->prov_infos[i]->domain_attr->mr_mode & FI_MR_ENDPOINT) {
            status =
                fi_mr_regattr(libfabric_state->domains[i], &mr_attr, 0, &fabric_handle->hdls[i].mr);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Error registering memory region.\n");

            status =
                fi_mr_bind(fabric_handle->hdls[i].mr, &libfabric_state->eps[i]->endpoint->fid, 0);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Error binding MR to EP %zu.\n", i);

            status = fi_mr_enable(fabric_handle->hdls[i].mr);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Error enabling MR.\n");

            fabric_handle->hdls[i].key = fi_mr_key(fabric_handle->hdls[i].mr);
            fabric_handle->hdls[i].local_desc = fi_mr_desc(fabric_handle->hdls[i].mr);
        } else {
            struct fid_mr *mr;

            status = fi_mr_regattr(libfabric_state->domains[i], &mr_attr, 0, &mr);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Error registering memory region.\n");

            fabric_handle->hdls[i].mr = mr;
            fabric_handle->hdls[i].key = fi_mr_key(mr);
            fabric_handle->hdls[i].local_desc = fi_mr_desc(mr);
        }
    }

    if (!local_only && libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        if (libfabric_state->cache == NULL) {
            status = nvshmemt_mem_handle_cache_init(t, &libfabric_state->cache);
            NVSHMEMI_NZ_ERROR_JMP(status, status, out, "mem handle cache initialization failed.");
        }

        handle_info = (nvshmemt_libfabric_memhandle_info_t *)calloc(
            1, sizeof(nvshmemt_libfabric_memhandle_info_t));
        NVSHMEMI_NULL_ERROR_JMP(handle_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Cannot allocate memory handle info.\n");

        if (!is_host) {
#ifdef NVSHMEM_USE_GDRCOPY
            if (use_gdrcopy) {
                status = gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)buf, length, 0, 0,
                                                   &handle_info->mh);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy pin_buffer failed \n");

                status = gdrcopy_ftable.map(gdr_desc, handle_info->mh, &handle_info->cpu_ptr_base,
                                            length);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy map failed \n");

                status = gdrcopy_ftable.get_info(gdr_desc, handle_info->mh, &info);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy get_info failed \n");

                // remember that mappings start on a 64KB boundary, so let's
                // calculate the offset from the head of the mapping to the
                // beginning of the buffer
                handle_info->cpu_ptr =
                    (void *)((char *)handle_info->cpu_ptr_base + ((char *)buf - (char *)info.va));

                handle_info->gdr_mapping_size = length;
                handle_info->ptr = buf;
                curr_ptr = buf;
                do {
                    status = nvshmemt_mem_handle_cache_add(t, libfabric_state->cache, curr_ptr,
                                                           (void *)handle_info);
                    NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                          "Unable to add key to mem handle info cache");
                    curr_ptr = (char *)curr_ptr + (1ULL << t->log2_cumem_granularity);
                } while (curr_ptr < (char *)buf + length);
            } else
#endif
            {
                NVSHMEMI_ERROR_PRINT(
                    "GDRCopy support not enabled. Unable to register gpu memory handle info.");
                status = NVSHMEMX_ERROR_INVALID_VALUE;
                goto out;
            }
        } else {
            handle_info->ptr = buf;
            handle_info->cpu_ptr = buf;
            handle_info->gdr_mapping_size = 0;
        }
        curr_ptr = buf;
        do {
            status = nvshmemt_mem_handle_cache_add(t, libfabric_state->cache, curr_ptr,
                                                   (void *)handle_info);
            NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                  "Unable to add key to mem handle info cache");
            curr_ptr = (char *)curr_ptr + (1ULL << t->log2_cumem_granularity);
        } while (curr_ptr < (char *)buf + length);
    }

    if (libfabric_state->local_mrs.empty() && !local_only) {
        for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
            libfabric_state->local_mrs.push_back(fabric_handle->hdls[i].mr);
            libfabric_state->local_mr_keys.push_back(fabric_handle->hdls[i].key);
            libfabric_state->local_mr_descs.push_back(fabric_handle->hdls[i].local_desc);
        }
        libfabric_state->local_mem_ptr = buf;
    }

out:
    if (status) {
        if (handle_info) {
            if (libfabric_state->cache) {
                nvshmemt_mem_handle_cache_remove(t, libfabric_state->cache, buf);
            }
            free(handle_info);
        }
    }
    return status;
}

static int nvshmemt_libfabric_can_reach_peer(int *access,
                                             struct nvshmem_transport_pe_info * /*peer_info*/,
                                             nvshmem_transport_t /*t*/) {
    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return 0;
}

static void nvshmemt_libfabric_cleanup_signal_ordering_state(nvshmemt_libfabric_state_t *state)
{
    state->host_signal_state.ack_aggregator.reset();
    state->host_signal_state.put_signal_seq_counter.clear();
    state->host_signal_state.proxy_put_signal_comp_map.clear();
    state->host_signal_state.next_expected_seq.clear();
    state->proxy_signal_state.ack_aggregator.reset();
    state->proxy_signal_state.put_signal_seq_counter.clear();
    state->proxy_signal_state.proxy_put_signal_comp_map.clear();
    state->proxy_signal_state.next_expected_seq.clear();
    state->deferred_work_queue.clear();
}

static int nvshmemt_libfabric_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                                int num_selected_devs, int * /*out_qp_indices*/,
                                                int /*num_qps*/) {
    nvshmemt_libfabric_state_t *state = get_libfabric_state(t);
    std::vector<nvshmemt_libfabric_ep_name_t> all_ep_names;
    std::vector<nvshmemt_libfabric_ep_name_t> local_ep_names;
    struct fi_info *current_info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *address;
    struct fid_mr *mr;
    struct fi_av_attr av_attr;
    struct fi_cq_attr cq_attr;
    size_t ep_namelen = NVSHMEMT_LIBFABRIC_EP_LEN;
    int status = 0;
    size_t total_num_eps;
    size_t num_recvs_per_ep = 0;
    int n_pes = t->n_pes;
    size_t num_sends;
    size_t num_recvs;
    size_t elem_size;
    size_t num_selected_domains;
    uint64_t flags;

    if (state->eps.size()) {
        NVSHMEMI_WARN_PRINT("PE has previously called connect_endpoints()\n");
        goto out_already_connected;
    }

    /* Number of devices to attempt to setup */
    state->num_selected_devs = std::min(num_selected_devs, state->max_nic_per_pe);

    /* Number of domains for host and proxy */
    state->num_host_domains = 1;
    state->num_proxy_domains = state->num_selected_devs;
    num_selected_domains = state->num_host_domains + state->num_proxy_domains;

    /* Check for potential overflow of nvshmemt_libfabric_mem_handle_t */
    if (num_selected_domains > NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE) {
        NVSHMEMI_WARN_PRINT(
            "Selected %d devices, resulting in %zu domains (%d host, %d proxy), "
            "but the libfabric transport supports a max of %zu domains.\n",
            state->num_selected_devs, num_selected_domains, state->num_host_domains,
            state->num_proxy_domains, NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE);

        /* Reduce host domains first (if applicable) */
        int remainder = num_selected_domains - NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE;
        if (state->num_host_domains > 1) {
            state->num_host_domains = std::max(1, state->num_host_domains - remainder);
        }
        num_selected_domains = state->num_host_domains + state->num_proxy_domains;

        /* Then reduce proxy domains until overflow is resolved */
        remainder = num_selected_domains - NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE;
        if (remainder > 0) {
            state->num_proxy_domains = std::max(1, state->num_proxy_domains - remainder);
        }

        /* Minimum is 1 host and 1 proxy */
        num_selected_domains = state->num_host_domains + state->num_proxy_domains;
        NVSHMEMI_CHECK_ERROR_JMP(num_selected_domains > NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE,
                                 status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "Unable to reduce domain count (selected: %zu, required: %zu).\n",
                                 num_selected_domains, NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE);

        state->num_selected_devs = std::max(state->num_host_domains, state->num_proxy_domains);

        NVSHMEMI_WARN_PRINT("Continuing with %d devices with %zu domains (%d host, %d proxy).\n",
                            state->num_selected_devs, num_selected_domains, state->num_host_domains,
                            state->num_proxy_domains);
    }

    /* One-time initializations */
    t->max_op_len = UINT64_MAX;
    state->proxy_ep_cntr = 0;
    state->pending_batch_ep = {-1, -1};

    memset(&cq_attr, 0, sizeof(struct fi_cq_attr));
    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        cq_attr.format = FI_CQ_FORMAT_UNSPEC;
        cq_attr.wait_obj = FI_WAIT_NONE;
        cq_attr.size = 16; /* CQ is only used to capture error events */
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        cq_attr.format = FI_CQ_FORMAT_DATA;
        cq_attr.wait_obj = FI_WAIT_NONE;
        cq_attr.size = 32768;
    }

    memset(&av_attr, 0, sizeof(struct fi_av_attr));
    av_attr.type = FI_AV_TABLE;
    av_attr.count = num_selected_domains * n_pes;
    av_attr.ep_per_node = 1;

    state->prov_infos.resize(num_selected_domains);

    /* Find provider info for each device */
    for (int dev_idx = 0; dev_idx < state->num_selected_devs; dev_idx++) {
        current_info = state->all_prov_info;
        do {
            if (!strncmp(current_info->nic->device_attr->name,
                         state->domain_names[selected_dev_ids[dev_idx]].name.data(),
                         NVSHMEMT_LIBFABRIC_DOMAIN_LEN)) {
                break;
            }
            current_info = current_info->next;
        } while (current_info != NULL);
        NVSHMEMI_NULL_ERROR_JMP(current_info, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "Unable to find the selected fabric.\n");

        /* Constructed such that all host domains first, then proxy domains */
        if (dev_idx < state->num_host_domains) {
            state->prov_infos[dev_idx] = current_info;
        }
        if (dev_idx < state->num_proxy_domains) {
            state->prov_infos[dev_idx + state->num_host_domains] = current_info;
        }

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA &&
            strcmp(current_info->fabric_attr->name, "efa-direct"))
            NVSHMEMI_WARN_PRINT(
                "Libfabric transport is using efa fabric instead of efa-direct, "
                "use libfabric v2.1.0 or newer for improved performance\n");
    }

    /* Allocate out of band AV name exchange buffers */
    local_ep_names.resize(num_selected_domains);
    total_num_eps = num_selected_domains * static_cast<size_t>(n_pes);
    all_ep_names.resize(total_num_eps);

    /* Initialize state-level signal ordering state */
    {
        int npes = t->n_pes;

        /* Compute ack high-water mark from NIC tx depth. Under a pure
         * AMO burst, each signal generates an ack fi_send on the receiver.
         * If the receiver's tx ring fills, ack fi_sends return EAGAIN and
         * the proxy thread retries inside try_again, which polls the CQ
         * and processes more incoming signals — each generating another
         * ack that also hits EAGAIN. This recursive try_again loop
         * prevents the ack backlog from ever draining, leading to
         * deadlock. Capping the sender's outstanding signals below
         * tx_attr.size limits the ack pressure on the receiver.
         * TODO: fix try_again to not do sends during CQ processing. */
        uint32_t min_tx_size = UINT32_MAX;
        for (size_t i = 0; i < state->prov_infos.size(); i++) {
            if (state->prov_infos[i]->tx_attr)
                min_tx_size = std::min(min_tx_size, (uint32_t)state->prov_infos[i]->tx_attr->size);
        }
        /* Scale HWM by npes so aggregate receiver ack pressure is bounded.
         * Per-peer HWM of (tx_size - 256) is a per-pair bound; with N senders
         * targeting one receiver, aggregate pressure = N * HWM. Each inbound
         * signal generates an ack fi_send needing a tx WQE; if aggregate >
         * tx_size the receiver hits EAGAIN and try_again recursively re-enters
         * CQ processing, triggering deadlock (see TODO in this function).
         *
         * Divide base by npes so aggregate = N * (base/N) = base <= tx_size.
         * If base < npes the invariant is unsatisfiable — fail loudly rather
         * than silently allow the deadlock precondition. Long-term fix is to
         * make try_again not issue fi_sends during CQ processing. */
        uint32_t base_hwm = (min_tx_size > 256) ? min_tx_size - 256 : min_tx_size / 2;
        if ((uint32_t)npes > base_hwm) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                "tx_size=%u too small for npes=%d (base_hwm=%u < npes): aggregate ack pressure cannot be bounded below tx ring depth. Use a NIC with deeper tx ring, or enable the long-term try_again fix.\n",
                min_tx_size, npes, base_hwm);
        }
        uint32_t hwm = std::max(1u, base_hwm / (uint32_t)npes);

        /* Link put_ack_freq to hwm so the invariant put_ack_freq <= hwm holds by
         * construction. At high npes (e.g. 64) hwm can drop below the static cap
         * of 64; without this link sparse workloads deadlock because the sender
         * blocks at hwm before put_count reaches put_ack_freq (no ack request is
         * ever emitted). hwm/2 leaves headroom to avoid a race on the boundary. */
        uint32_t put_ack_freq_u = std::max(1u, hwm / 2);
        uint8_t put_ack_freq = static_cast<uint8_t>(
            std::min<uint32_t>(nvshmemt_libfabric_endpoint_seq_counter_t::PUT_ACK_FREQ_CAP,
                               put_ack_freq_u));

        state->host_signal_state.put_signal_seq_counter.resize(npes);
        state->host_signal_state.proxy_put_signal_comp_map.resize(npes);
        state->host_signal_state.next_expected_seq.resize(npes, 0);
        state->host_signal_state.ack_aggregator = std::make_unique<nvshmemt_libfabric_ack_aggregator_t>(npes);
        state->host_signal_state.completed_staged_atomics = 0;
        for (int pe = 0; pe < npes; pe++) {
            state->host_signal_state.put_signal_seq_counter[pe].ack_high_watermark = hwm;
            state->host_signal_state.put_signal_seq_counter[pe].put_ack_freq = put_ack_freq;
        }
        state->proxy_signal_state.put_signal_seq_counter.resize(npes);
        state->proxy_signal_state.proxy_put_signal_comp_map.resize(npes);
        state->proxy_signal_state.next_expected_seq.resize(npes, 0);
        state->proxy_signal_state.ack_aggregator = std::make_unique<nvshmemt_libfabric_ack_aggregator_t>(npes);
        state->proxy_signal_state.completed_staged_atomics = 0;
        for (int pe = 0; pe < npes; pe++) {
            state->proxy_signal_state.put_signal_seq_counter[pe].ack_high_watermark = hwm;
            state->proxy_signal_state.put_signal_seq_counter[pe].put_ack_freq = put_ack_freq;
        }
    }

    for (size_t i = 0; i < state->prov_infos.size(); i++) {
        INFO(state->log_level, "Selected provider %s, fabric %s, nic %s, hmem %s domain %zu/%zu",
             state->prov_infos[i]->fabric_attr->prov_name, state->prov_infos[i]->fabric_attr->name,
             state->prov_infos[i]->nic->device_attr->name,
             state->prov_infos[i]->caps & FI_HMEM ? "yes" : "no", i + 1, state->prov_infos.size());

        if (state->prov_infos[i]->ep_attr->max_msg_size < t->max_op_len)
            t->max_op_len = state->prov_infos[i]->ep_attr->max_msg_size;

        status = fi_fabric(state->prov_infos[i]->fabric_attr, &fabric, NULL);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Failed to allocate fabric.\n");
        state->fabrics.push_back(fabric);

        status = fi_domain(fabric, state->prov_infos[i], &domain, NULL);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Failed to allocate domain.\n");
        state->domains.push_back(domain);

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            num_sends = state->prov_infos[i]->tx_attr->size;
            num_recvs = state->prov_infos[i]->rx_attr->size;
            elem_size = sizeof(nvshmemt_libfabric_gdr_op_ctx_t);
            num_recvs_per_ep = num_recvs;

            state->recv_buf.push_back(
                std::vector<nvshmemt_libfabric_gdr_op_ctx_t>(num_sends + num_recvs));

            status =
                fi_mr_reg(domain, state->recv_buf[i].data(), (num_sends + num_recvs) * elem_size,
                          FI_SEND | FI_RECV | FI_WRITE, 0, 0, 0, &mr, NULL);
            NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "Failed to register EFA msg buffer.\n");
            state->mrs.push_back(mr);

            /* Locking is not required for auto progress, required for manual progress */
            state->op_queue.emplace_back(std::unique_ptr<threadSafeOpQueue>(
                new threadSafeOpQueue(!state->use_auto_progress)));
            NVSHMEMI_NULL_ERROR_JMP(state->op_queue[i], status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                    "Unable to alloc thread-safe op queue struct.\n");
            state->op_queue.back()->putToSendBulk(&state->recv_buf[i][num_recvs], num_sends);
        }

        status = fi_av_open(domain, &av_attr, &address, NULL);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Failed to allocate address vector.\n");
        state->addresses.push_back(address);

        /* Create endpoint resources */
        state->eps.emplace_back(
            std::unique_ptr<nvshmemt_libfabric_endpoint_t>(new nvshmemt_libfabric_endpoint_t()));
        NVSHMEMI_NULL_ERROR_JMP(state->eps[i], status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to alloc nvshmemt_libfabric_endpoint_t struct.\n");
        state->eps[i]->domain_index = i;
        state->eps[i]->ep_index = i;

        if (static_cast<int>(i) < state->num_host_domains) {
            state->eps[i]->qp_index = NVSHMEMX_QP_HOST;
        } else {
            /* QP index value for proxy EPs doesn't matter, as long as it is not NVSHMEMX_QP_HOST.
             * This is strictly used for determining which EPs (e.g. all proxy EPs vs host EP) to
             * progress when FI_PROGRESS_AUTO is enabled.
             */
            state->eps[i]->qp_index = NVSHMEMX_QP_DEFAULT;
        }

        state->eps[i]->submitted_ops = 0;
        state->eps[i]->completed_ops = 0;

        status = fi_cq_open(domain, &cq_attr, &state->eps[i]->cq, NULL);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to open completion queue for endpoint.\n");

        status = fi_endpoint(domain, state->prov_infos[i], &state->eps[i]->endpoint, NULL);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to allocate endpoint.\n");

        /* FI_OPT_CUDA_API_PERMITTED was introduced in libfabric 1.18.0 */
        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            bool prohibit_cuda_api = false;
            status = fi_setopt(&state->eps[i]->endpoint->fid, FI_OPT_ENDPOINT,
                               FI_OPT_CUDA_API_PERMITTED, &prohibit_cuda_api, sizeof(bool));
            if (status == -FI_ENOPROTOOPT) {
                NVSHMEMI_WARN_PRINT(
                    "fi_setopt of FI_OPT_CUDA_API_PERMITTED returned as "
                    "not implemented.\n Not setting. This is expected for libfabric "
                    "versions < 1.18.\n");
            } else if (status) {
                NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(
                    status, NVSHMEMX_ERROR_INTERNAL, out,
                    "Unable to set endpoint CUDA API status.\n");
            }
        }

        /* Bind resources to EP */
        status = fi_ep_bind(state->eps[i]->endpoint, &state->addresses[i]->fid, 0);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to bind endpoint to address vector.\n");

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
            flags = FI_SELECTIVE_COMPLETION | FI_TRANSMIT | FI_RECV;
        } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
            flags = FI_SELECTIVE_COMPLETION | FI_TRANSMIT;
        } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            /* EFA is documented as not supporting FI_SELECTIVE_COMPLETION */
            flags = FI_TRANSMIT | FI_RECV;
        } else {
            NVSHMEMI_ERROR_PRINT(
                "Invalid provider identified. This should be impossible. "
                "Possible memory corruption in the state pointer?");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }

        status = fi_ep_bind(state->eps[i]->endpoint, &state->eps[i]->cq->fid, flags);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to bind endpoint to completion queue.\n");

        status = fi_enable(state->eps[i]->endpoint);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to enable endpoint.\n");

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            std::vector<nvshmemt_libfabric_gdr_op_ctx_t> &op = state->recv_buf[i];
            for (size_t j = 0; j < num_recvs_per_ep; j++) {
                status = fi_recv(state->eps[i]->endpoint, &op[j], NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                                 fi_mr_desc(state->mrs[i]), FI_ADDR_UNSPEC, &op[j].ofi_context);
                NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                                "Unable to post recv to ep.\n");
            }
        }

        status =
            fi_getname(&state->eps[i]->endpoint->fid, local_ep_names[i].name.data(), &ep_namelen);
        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "Unable to get name for endpoint.\n");
        if (ep_namelen > NVSHMEMT_LIBFABRIC_EP_LEN) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Name of EP is too long.");
        }
    }

    assert(state->domains.size() == num_selected_domains);
    assert(state->domains.size() <= NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE);
    if (state->domains.size() > NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE) {
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "libfabric: total domains (%zu) exceeds hdls capacity (%zu)\n",
                              state->domains.size(), NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE);
    }

    /* Perform out of band address exchange */
    status = t->boot_handle->allgather(local_ep_names.data(), all_ep_names.data(),
                                       state->eps.size() * sizeof(nvshmemt_libfabric_ep_name_t),
                                       t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Failed to gather endpoint names.\n");

    /* We need to insert one at a time since each buffer is larger than the address. */
    for (size_t i = 0; i < state->domains.size(); i++) {
        for (size_t j = 0; j < total_num_eps; j++) {
            status = fi_av_insert(state->addresses[i], &all_ep_names[j], 1, NULL, 0, NULL);
            if (status < 0) {
                NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                                "Unable to insert ep names in address vector.\n");
            } else if (status == 0) {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to insert ep names in address vector.\n");
            }

            status = NVSHMEMX_SUCCESS;
        }
    }

    /* Spawn signal delivery thread for EFA staged atomics */
    if (state->use_staged_atomics) {
        state->signal_delivery_stop.store(0, std::memory_order_relaxed);
        state->signal_delivery_status.store(0, std::memory_order_relaxed);
        state->signal_delivery_transport = t;
        status = pthread_create(&state->signal_delivery_thread, NULL,
                                nvshmemt_libfabric_signal_delivery_thread, state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to create signal delivery thread.\n");
    }

out:
    if (status != 0) {
        /* Cleanup state-level signal ordering state */
        nvshmemt_libfabric_cleanup_signal_ordering_state(state);

        for (size_t i = 0; i < state->eps.size(); i++) {
            if (state->eps[i]->endpoint) {
                fi_close(&state->eps[i]->endpoint->fid);
                state->eps[i]->endpoint = NULL;
            }
            if (state->eps[i]->cq) {
                fi_close(&state->eps[i]->cq->fid);
                state->eps[i]->cq = NULL;
            }
        }
        state->recv_buf.clear();
        state->eps.clear();
        state->op_queue.clear();
    }

out_already_connected:
    return status;
}

static int nvshmemt_libfabric_finalize(nvshmem_transport_t transport) {
    int status;

    assert(transport);

    /* Take ownership of the state so destruction runs automatically at function exit. */
    std::unique_ptr<nvshmemt_libfabric_state_t> libfabric_state_owner(
        get_libfabric_state(transport));
    transport->state = nullptr;
    nvshmemt_libfabric_state_t *libfabric_state = libfabric_state_owner.get();

    /* Stop signal delivery thread before tearing down resources */
    if (libfabric_state->use_staged_atomics && libfabric_state->signal_delivery_transport) {
        libfabric_state->signal_delivery_stop.store(1, std::memory_order_seq_cst);
        libfabric_state->signal_delivery_futex.store(1, std::memory_order_release);
        syscall(SYS_futex, &libfabric_state->signal_delivery_futex, FUTEX_WAKE, 1, NULL, NULL, 0);
        pthread_join(libfabric_state->signal_delivery_thread, NULL);
    }

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

    size_t mem_handle_cache_size;
    nvshmemt_libfabric_memhandle_info_t *handle_info = NULL, *previous_handle_info = NULL;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        mem_handle_cache_size = nvshmemt_mem_handle_cache_get_size(libfabric_state->cache);
        for (size_t i = 0; i < mem_handle_cache_size; i++) {
            handle_info =
                (nvshmemt_libfabric_memhandle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(
                    libfabric_state->cache, i);
            if (handle_info && handle_info != previous_handle_info) {
                free(handle_info);
            }
            previous_handle_info = handle_info;
        }

        nvshmemt_mem_handle_cache_fini(libfabric_state->cache);
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
        }
#endif
    }

    /* Since fi_dupinfo is not called, we don't need to clean up prov_infos */
    if (libfabric_state->all_prov_info) {
        fi_freeinfo(libfabric_state->all_prov_info);
    }

    /* Cleanup state-level signal ordering state */
    nvshmemt_libfabric_cleanup_signal_ordering_state(libfabric_state);

    for (size_t i = 0; i < libfabric_state->eps.size(); i++) {
        if (libfabric_state->eps[i]->endpoint) {
            status = fi_close(&libfabric_state->eps[i]->endpoint->fid);
            if (status) {
                NVSHMEMI_WARN_PRINT("Unable to close fabric endpoint.: %d: %s\n", status,
                                    fi_strerror(status * -1));
            }
        }
        if (libfabric_state->eps[i]->cq) {
            status = fi_close(&libfabric_state->eps[i]->cq->fid);
            if (status) {
                NVSHMEMI_WARN_PRINT("Unable to close fabric cq: %d: %s\n", status,
                                    fi_strerror(status * -1));
            }
        }
    }

    for (size_t i = 0; i < libfabric_state->mrs.size(); i++) {
        status = fi_close(&libfabric_state->mrs[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric MR: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    for (size_t i = 0; i < libfabric_state->addresses.size(); i++) {
        status = fi_close(&libfabric_state->addresses[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric address vector: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
        status = fi_close(&libfabric_state->domains[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric domain: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    for (size_t i = 0; i < libfabric_state->fabrics.size(); i++) {
        status = fi_close(&libfabric_state->fabrics[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    libfabric_state_owner.reset();
    free(transport);

    return 0;
}

static int nvshmemi_libfabric_init_state(nvshmem_transport_t t, nvshmemt_libfabric_state_t *state) {
    struct fi_info *all_infos, *current_info;
    size_t num_fabrics_returned = 0;
    int num_devices = 0;
    int status = 0;

    auto deleter = [](fi_info *p) { fi_freeinfo(p); };
    std::unique_ptr<fi_info, decltype(deleter)> hints{fi_allocinfo(), deleter};
    NVSHMEMI_NULL_ERROR_JMP(hints, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate memory for libfabric info hint.");

    hints->addr_format = FI_FORMAT_UNSPEC;
    hints->caps = FI_RMA | FI_HMEM;
    hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_PROV_KEY;

    if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
        hints->caps |= FI_ATOMIC;
        hints->domain_attr->mr_mode |= FI_MR_VIRT_ADDR;
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        /* TODO: Use FI_FENCE to optimize put_with_signal */
        hints->caps |= FI_FENCE | FI_ATOMIC;
        hints->domain_attr->mr_mode |= FI_MR_ENDPOINT;
    } else if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        hints->caps |= FI_MSG | FI_SOURCE;
        hints->domain_attr->mr_mode |= FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_HMEM;
    }

    if (state->use_staged_atomics) {
        hints->mode |= FI_CONTEXT2;
    }

    /* Try FI_PROGRESS_AUTO */
    hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;

    /* Require completion RMA completion at target for correctness of quiet */
    hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;

    hints->ep_attr->type = FI_EP_RDM;  // Reliable datagrams

    status = fi_getinfo(FI_VERSION(NVSHMEMT_LIBFABRIC_MAJ_VER, NVSHMEMT_LIBFABRIC_MIN_VER), NULL,
                        NULL, 0, hints.get(), &all_infos);

    /* Ensure at least one fabric was returned and it matches the selected provider */
    if (!status && strstr(all_infos->fabric_attr->name, options.LIBFABRIC_PROVIDER)) {
        state->use_auto_progress = true;
        INFO(state->log_level, "Using FI_PROGRESS_AUTO with FI_THREAD_COMPLETION.\n");
    } else {
        /* Cleanup and fallback to manual progress */
        if (!status) {
            fi_freeinfo(all_infos);
        }

        INFO(state->log_level, "Falling back to FI_PROGRESS_MANUAL with FI_THREAD_SAFE.\n");
        hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
        hints->domain_attr->threading = FI_THREAD_SAFE;

        status = fi_getinfo(FI_VERSION(NVSHMEMT_LIBFABRIC_MAJ_VER, NVSHMEMT_LIBFABRIC_MIN_VER),
                            NULL, NULL, 0, hints.get(), &all_infos);

        NVSHMEMT_LIBFABRIC_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "No providers matched fi_getinfo query.\n");
    }

    state->all_prov_info = all_infos;
    for (current_info = all_infos; current_info != NULL; current_info = current_info->next) {
        num_fabrics_returned++;
    }

    state->domain_names.resize(num_fabrics_returned);

    /* Only select unique devices. */
    num_devices = 0;
    for (current_info = all_infos; current_info != NULL; current_info = current_info->next) {
        if (!current_info->nic) {
            INFO(state->log_level,
                 "Interface did not return NIC structure to fi_getinfo. Skipping.\n");
            continue;
        }

        if (!current_info->tx_attr) {
            INFO(state->log_level,
                 "Interface did not return TX_ATTR structure to fi_getinfo. Skipping.\n");
            continue;
        }

        TRACE(state->log_level, "fi_getinfo returned provider %s, fabric %s, nic %s",
              current_info->fabric_attr->prov_name, current_info->fabric_attr->name,
              current_info->nic->device_attr->name);

        if (state->provider != NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            if (current_info->tx_attr->inject_size < NVSHMEMT_LIBFABRIC_INJECT_BYTES) {
                INFO(state->log_level,
                     "Disabling interface due to insufficient inject data size. reported %lu, "
                     "expected "
                     "%u",
                     current_info->tx_attr->inject_size, NVSHMEMT_LIBFABRIC_INJECT_BYTES);
                continue;
            }
        }

        if ((current_info->domain_attr->mr_mode & FI_MR_PROV_KEY) == 0) {
            INFO(state->log_level, "Disabling interface due to FI_MR_PROV_KEY support");
            continue;
        }

        for (int i = 0; i <= num_devices; i++) {
            if (!strncmp(current_info->nic->device_attr->name, state->domain_names[i].name.data(),
                         NVSHMEMT_LIBFABRIC_DOMAIN_LEN)) {
                break;
            } else if (i == num_devices) {
                size_t name_len = strlen(current_info->nic->device_attr->name);
                if (name_len >= NVSHMEMT_LIBFABRIC_DOMAIN_LEN) {
                    NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                       "Unable to copy domain name for libfabric transport.");
                }
                (void)strncpy(state->domain_names[num_devices].name.data(),
                              current_info->nic->device_attr->name, NVSHMEMT_LIBFABRIC_DOMAIN_LEN);
                num_devices++;
                break;
            }
        }
    }

    t->n_devices = num_devices;
    t->device_pci_paths = (char **)calloc(t->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(t->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.");
    for (int i = 0; i < t->n_devices; i++) {
        status = get_pci_path(i, &t->device_pci_paths[i], t);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.");
    }

out:
    if (status) {
        nvshmemt_libfabric_finalize(t);
    }

    return status;
}

/* Public Functions */
int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    nvshmemt_libfabric_state_t *libfabric_state = NULL;
    nvshmem_transport_t transport = NULL;
    int status = 0;
    std::unique_ptr<nvshmemt_libfabric_state_t> libfabric_state_owner;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    transport = (nvshmem_transport_t)calloc(1, sizeof(*transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for libfabric transport.");

    libfabric_state_owner = std::make_unique<nvshmemt_libfabric_state_t>();
    libfabric_state = libfabric_state_owner.get();
    libfabric_state->table = table;
    /* Transfer ownership to transport; finalize() reclaims it via unique_ptr. */
    transport->state = libfabric_state_owner.release();

    transport->host_ops.can_reach_peer = nvshmemt_libfabric_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_libfabric_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_libfabric_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_libfabric_release_mem_handle;
    transport->host_ops.rma = nvshmemt_libfabric_rma;
    if (!options.LIBFABRIC_DISABLE_BATCH_RMA) {
        transport->host_ops.rma_with_hints = nvshmemt_libfabric_rma_with_hints;
    }
    transport->host_ops.fence = nvshmemt_libfabric_fence;
    transport->host_ops.quiet = nvshmemt_libfabric_quiet;
    transport->host_ops.finalize = nvshmemt_libfabric_finalize;
    transport->host_ops.show_info = nvshmemt_libfabric_show_info;
    transport->host_ops.progress = nvshmemt_libfabric_proxy_progress;
    transport->host_ops.enforce_cst = nvshmemt_libfabric_enforce_cst;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;

    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    status = nvshmemi_env_options_init(&options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize env options.");

    libfabric_state->log_level = nvshmemt_common_get_log_level(&options);
    libfabric_state->max_nic_per_pe = options.LIBFABRIC_MAX_NIC_PER_PE;

    if (strcmp(options.LIBFABRIC_PROVIDER, "verbs") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_VERBS;
    } else if (strcmp(options.LIBFABRIC_PROVIDER, "cxi") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT;
    } else if (strcmp(options.LIBFABRIC_PROVIDER, "efa") == 0) {
        libfabric_state->provider = NVSHMEMT_LIBFABRIC_PROVIDER_EFA;
    } else {
        NVSHMEMI_WARN_PRINT("Invalid libfabric transport persona '%s'\n",
                            options.LIBFABRIC_PROVIDER);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        transport->atomics_complete_on_quiet = false;
    } else {
        transport->atomics_complete_on_quiet = true;
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (options.DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   libfabric_state->log_level);
        if (!use_gdrcopy) {
            INFO(libfabric_state->log_level,
                 "GDRCopy Initialization failed."
                 " Device memory will not be supported.\n");
        }
    } else {
        INFO(libfabric_state->log_level,
             "GDRCopy requested, but unused by transport. Disabling.\n");
        use_gdrcopy = false;
    }
#else
    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "EFA Provider requires GDRCopy, but it was disabled"
                           " at compile time.\n");
    }
#endif

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        libfabric_state->use_staged_atomics = true;
        transport->host_ops.amo = nvshmemt_libfabric_gdr_amo;
    } else {
        transport->host_ops.amo = nvshmemt_libfabric_amo;
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        transport->host_ops.put_signal = nvshmemt_libfabric_put_signal_unordered;
    } else {
        transport->host_ops.put_signal = nvshmemt_put_signal;
    }

#define NVSHMEMI_SET_ENV_VAR(varname, desired, warning_msg)                              \
    do {                                                                                 \
        const char *env_value = getenv(varname);                                         \
        if (env_value && strcmp(env_value, desired) != 0) {                              \
            NVSHMEMI_WARN_PRINT(warning_msg);                                            \
        } else if (!env_value) {                                                         \
            status = setenv(varname, desired, 1);                                        \
            if (status) {                                                                \
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,                 \
                                   "Failed to set environment variable %s.\n", varname); \
            }                                                                            \
        }                                                                                \
    } while (0)

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_VERBS) {
        /* This MLX5 feature is known to cause issues with device memory read and atomic ops. */
        NVSHMEMI_SET_ENV_VAR(
            "MLX5_SCATTER_TO_CQE", "0",
            "MLX5_SCATTER_TO_CQE is set. This may cause issues with device memory read and "
            "atomic ops if the value is not 0.\n");
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        NVSHMEMI_SET_ENV_VAR(
            "FI_EFA_USE_DEVICE_RDMA", "1",
            "FI_EFA_USE_DEVICE_RDMA is set. This may cause issues with initialization "
            "if the value is not 1.\n");
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT) {
        NVSHMEMI_SET_ENV_VAR("FI_CXI_DISABLE_HMEM_DEV_REGISTER", "1",
                             "FI_CXI_DISABLE_HMEM_DEV_REGISTER is set. This may cause issues with "
                             "initialization if the value is not 1.\n");

        NVSHMEMI_SET_ENV_VAR("FI_CXI_OPTIMIZED_MRS", "0",
                             "FI_CXI_OPTIMIZED_MRS is set. This may cause a hang at runtime "
                             "if the value is not 0.\n");
    }

    NVSHMEMI_SET_ENV_VAR("FI_HMEM_CUDA_USE_GDRCOPY", "1",
                         "FI_HMEM_CUDA_USE_GDRCOPY is set. This may cause issues at runtime "
                         "if the value is not 1.\n");

#undef NVSHMEMI_SET_ENV_VAR

    /* Prepare fabric state information. */
    status = nvshmemi_libfabric_init_state(transport, libfabric_state);
    if (status) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out_clean,
                           "Failed to initialize the libfabric state.\n");
    }

    libfabric_state->proxy_request_batch_max = options.LIBFABRIC_PROXY_REQUEST_BATCH_MAX;
    *t = transport;
out:
    if (status) {
        if (transport) {
            nvshmemt_libfabric_finalize(transport);
        }
    }

out_clean:
    return status;
}
