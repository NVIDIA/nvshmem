/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
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
#include <unordered_map>
#include <memory>
#include <errno.h>
#ifdef NVSHMEM_X86_64
#include <immintrin.h>  // IWYU pragma: keep
#endif
// IWYU pragma: no_include <xmmintrin.h>

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

#define MAX_COMPLETIONS_PER_CQ_POLL 300
#define NVSHMEM_STAGED_AMO_WIREDATA_SIZE \
    sizeof(nvshmemt_libfabric_gdr_op_ctx_t) - sizeof(struct fi_context2) - sizeof(fi_addr_t)

/* Forward declarations */
static int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport);
static int nvshmemt_libfabric_gdr_process_amos_ack(nvshmem_transport_t transport);
static int nvshmemt_libfabric_put_signal_completion(nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    struct fi_cq_data_entry *entry,
                                                    fi_addr_t *addr);
static int nvshmemt_libfabric_process_completions(nvshmem_transport_t transport);
static int nvshmemt_libfabric_progress(nvshmem_transport_t transport);

namespace {
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

/* Internal global variables */
#ifdef NVSHMEM_USE_GDRCOPY
struct gdrcopy_function_table gdrcopy_ftable;
void *gdrcopy_handle = NULL;
gdr_t gdr_desc;
bool use_gdrcopy = false;
#endif

bool use_staged_atomics = false;
std::recursive_mutex gdrRecvMutex;

/* Internal functions */
int get_next_ep(nvshmemt_libfabric_state_t *state, int qp_index) {
    if (qp_index == NVSHMEMX_QP_HOST) {
        return 0; /* Currently only 1 EP defined for the host */
    } else {
        return ((state->proxy_ep_cntr++) % state->num_proxy_domains) + state->num_host_domains;
    }
}

nvshmemt_libfabric_imm_cq_data_hdr_t get_write_with_imm_hdr(uint64_t imm_data) {
    return (nvshmemt_libfabric_imm_cq_data_hdr_t)((uint32_t)imm_data >>
                                                  NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT);
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

int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;
    const char *nic_name, *nic_class;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;

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

int try_again(nvshmem_transport_t transport, int *status, uint64_t *num_retries,
              nvshmemt_libfabric_try_again_call_site_t call_site, bool completions_only = false) {
    if (likely(*status == 0)) {
        return 0;
    }

    if (*status == -FI_EAGAIN) {
        if (*num_retries >= NVSHMEMT_LIBFABRIC_MAX_RETRIES) {
            fprintf(stderr, "call site: %d, Max amount of libfabric retries reached, %d: %s\n",
                    call_site, *status, fi_strerror(*status * -1));
            *status = NVSHMEMX_ERROR_INTERNAL;
            return 0;
        }
        (*num_retries)++;
        if (completions_only) {
            *status = nvshmemt_libfabric_process_completions(transport);
        } else {
            *status = nvshmemt_libfabric_progress(transport);
        }
    }

    if (*status != 0) {
        fprintf(stderr, "Error in libfabric operation (%d): %s.\n", *status,
                fi_strerror(*status * -1));
        *status = NVSHMEMX_ERROR_INTERNAL;
        return 0;
    }

    return 1;
}

int gdrcopy_amo_ack(nvshmem_transport_t transport, nvshmemt_libfabric_endpoint_t &ep,
                    fi_addr_t dest_addr, uint32_t sequence_count, int pe,
                    nvshmemt_libfabric_gdr_op_ctx_t **send_elems) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_op_ctx_t *resp_op = NULL;
    uint64_t num_retries = 0;
    int status;
    uint64_t imm_data = 0;
    uint64_t rkey_index = pe * libfabric_state->domains.size() + ep.domain_index;

    resp_op = send_elems[0];
    imm_data = (NVSHMEMT_LIBFABRIC_IMM_STAGED_ATOMIC_ACK
                << NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT) |
               sequence_count;
    do {
        status = fi_writedata(
            ep.endpoint, resp_op, 0, fi_mr_desc(libfabric_state->mrs[ep.domain_index]), imm_data,
            dest_addr, (uint64_t)libfabric_state->remote_addr_staged_amo_ack[pe],
            libfabric_state->rkey_staged_amo_ack[rkey_index], &resp_op->ofi_context);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDRCOPY_AMO_ACK, true));

    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to write atomic ack.\n");
    ep.submitted_ops++;

out:
    return status;
}

template <typename T>
int perform_gdrcopy_amo(nvshmem_transport_t transport, nvshmemt_libfabric_gdr_op_ctx_t *op,
                        nvshmemt_libfabric_gdr_op_ctx_t **send_elems, uint32_t sequence_count) {
    T old_value, new_value;
    uint64_t num_retries = 0;
    int send_elems_index = 0;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_send_amo_op_t *received_op = &(op->send_amo);
    nvshmemt_libfabric_gdr_op_ctx_t *resp_op = NULL;
    nvshmemt_libfabric_memhandle_info_t *handle_info;
    volatile T *ptr;
    int status = 0;
    /* Save op fields as registers to allow posting op as RX before TX */
    int src_pe = op->send_amo.src_pe;
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[op->ep_index]);
    fi_addr_t src_addr = op->src_addr;
    bool is_fetch_amo = received_op->op > NVSHMEMI_AMO_END_OF_NONFETCH;
    uint64_t ret_flags = received_op->retflag;
    void *ret_addr = received_op->ret_addr;
    int domain_idx = ep.domain_index;

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
            new_value = old_value + static_cast<T>(received_op->swap_add);
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

    /* Post recv before posting TX operations to avoid deadlocks */
    status =
        fi_recv(ep.endpoint, (void *)op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                fi_mr_desc(libfabric_state->mrs[domain_idx]), FI_ADDR_UNSPEC, &op->ofi_context);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to re-post recv.\n");

    if (is_fetch_amo) {
        resp_op = send_elems[send_elems_index];

        resp_op->ret_amo.elem.data = old_value;
        resp_op->ret_amo.elem.flag = ret_flags;
        resp_op->ret_amo.ret_addr = ret_addr;
        resp_op->type = NVSHMEMT_LIBFABRIC_ACK;

        do {
            status = fi_send(ep.endpoint, (void *)resp_op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                             fi_mr_desc(libfabric_state->mrs[domain_idx]), src_addr,
                             &resp_op->ofi_context);
        } while (try_again(transport, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PERFORM_GDRCOPY_AMO_SEND, true));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to respond to atomic request.\n");
        ep.submitted_ops++;
        send_elems_index++;
    }

    status = gdrcopy_amo_ack(transport, ep, src_addr, sequence_count, src_pe,
                             &send_elems[send_elems_index]);
out:
    return status;
}

nvshmemt_libfabric_gdr_op_ctx_t *inplace_copy_sig_op_to_gdr_op(
    nvshmemt_libfabric_gdr_signal_op *sig_op, int ep_index) {
    nvshmemt_libfabric_gdr_op_ctx_t *amo;
    uint16_t op = sig_op->op;
    uint64_t sig_val = sig_op->sig_val;
    void *target_addr = sig_op->target_addr;
    uint32_t src_pe = sig_op->src_pe;
    uint32_t sequence_count = sig_op->sequence_count;

    amo = (nvshmemt_libfabric_gdr_op_ctx_t *)sig_op;
    amo->ep_index = ep_index;
    amo->type = NVSHMEMT_LIBFABRIC_MATCH;
    amo->send_amo.op = (nvshmemi_amo_t)op;
    amo->send_amo.target_addr = target_addr;
    amo->send_amo.swap_add = sig_val;
    amo->send_amo.src_pe = src_pe;
    amo->send_amo.size = 8;
    amo->send_amo.sequence_count = sequence_count;

    return amo;
}
}  // anonymous namespace

/* Private functions with external linkage (local symbols) */
static void nvshmemt_libfabric_put_signal_ack_completion(nvshmemt_libfabric_endpoint_t &ep,
                                                         struct fi_cq_data_entry *entry) {
    uint32_t seq_num = entry->data & NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK;

    if (seq_num != NVSHMEM_STAGED_AMO_SEQ_NUM) {
        ep.put_signal_seq_counter.return_acked_seq_num(seq_num);
    }

    ep.completed_staged_atomics++;
}

static int nvshmemt_libfabric_gdr_process_completion(nvshmem_transport_t transport,
                                                     nvshmemt_libfabric_endpoint_t &ep,
                                                     struct fi_cq_data_entry *entry,
                                                     fi_addr_t *addr) {
    int status = 0;
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int domain_idx = ep.domain_index;

    /* Write w/imm doesn't have op->op_context, must be checked first */
    if (entry->flags & FI_REMOTE_CQ_DATA) {
        nvshmemt_libfabric_imm_cq_data_hdr_t imm_header = get_write_with_imm_hdr(entry->data);
        if (NVSHMEMT_LIBFABRIC_IMM_PUT_SIGNAL_SEQ == imm_header) {
            status = nvshmemt_libfabric_put_signal_completion(transport, ep, entry, addr);
            goto out;
        } else if (NVSHMEMT_LIBFABRIC_IMM_STAGED_ATOMIC_ACK == imm_header) {
            nvshmemt_libfabric_put_signal_ack_completion(ep, entry);
            goto out;
        } else {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "Received a write w/imm completion with invalid header type.\n");
        }
    }

    op = container_of(entry->op_context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context);
    /* FI_CONTEXT2 support requires that every operation with a completion has a context */
    assert(op);
    assert(addr);
    op->src_addr = *addr;

    if (entry->flags & FI_SEND) {
        libfabric_state->op_queue[domain_idx]->putToSend(op);
    } else if (entry->flags & FI_RMA) {
        /* inlined p ops or atomic responses */
        libfabric_state->op_queue[domain_idx]->putToSend(op);
    } else if (op->type == NVSHMEMT_LIBFABRIC_MATCH) {
        /* Must happen after entry->flags & FI_SEND to avoid send completions */
        status = nvshmemt_libfabric_put_signal_completion(transport, ep, entry, addr);
    } else if (entry->flags & FI_RECV) {
        op->ep_index = ep.ep_index;
        if (op->type == NVSHMEMT_LIBFABRIC_ACK) {
            libfabric_state->op_queue[domain_idx]->putToRecv(op, NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK);
        } else {
            libfabric_state->op_queue[domain_idx]->putToRecv(op,
                                                             NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK);
        }
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Found an invalid message type in an ep completion.\n");
    }

out:
    return status;
}

static int nvshmemt_libfabric_process_completions(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int status = 0;
    for (size_t i = 0; i < libfabric_state->eps.size(); i++) {
        uint64_t cnt = fi_cntr_readerr(libfabric_state->eps[i]->counter);

        if (cnt > 0) {
            NVSHMEMI_WARN_PRINT("Nonzero error count progressing EP %zu (%" PRIu64 ")\n", i, cnt);

            struct fi_cq_err_entry err;
            memset(&err, 0, sizeof(struct fi_cq_err_entry));
            ssize_t nerr = fi_cq_readerr(libfabric_state->eps[i]->cq, &err, 0);

            if (nerr > 0) {
                char str[100] = "\0";
                const char *err_str = fi_cq_strerror(libfabric_state->eps[i]->cq, err.prov_errno,
                                                     err.err_data, str, 100);
                NVSHMEMI_WARN_PRINT(
                    "CQ reported error (%d): %s\n\tProvider error: %s\n\tSupplemental error "
                    "info: %s\n",
                    err.err, fi_strerror(err.err), err_str ? err_str : "none",
                    strlen(str) ? str : "none");
            } else if (nerr == -FI_EAGAIN) {
                NVSHMEMI_WARN_PRINT("fi_cq_readerr returned -FI_EAGAIN\n");
            } else {
                NVSHMEMI_WARN_PRINT("fi_cq_readerr returned %zd: %s\n", nerr,
                                    fi_strerror(-1 * nerr));
            }
            return err.err;
        }

        {
            char buf[MAX_COMPLETIONS_PER_CQ_POLL * sizeof(struct fi_cq_data_entry)];
            fi_addr_t src_addr[MAX_COMPLETIONS_PER_CQ_POLL];
            ssize_t qstatus;
            nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[i]);
            do {
                qstatus = fi_cq_readfrom(ep.cq, buf, MAX_COMPLETIONS_PER_CQ_POLL, src_addr);
                /* Note - EFA provider does not support selective completions */
                if (qstatus > 0) {
                    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
                        struct fi_cq_data_entry *entry = (struct fi_cq_data_entry *)buf;
                        fi_addr_t *addr = src_addr;
                        for (int j = 0; j < qstatus; j++, entry++, addr++) {
                            status = nvshmemt_libfabric_gdr_process_completion(transport, ep, entry,
                                                                               addr);
                            if (status) return NVSHMEMX_ERROR_INTERNAL;
                        }
                    } else {
                        NVSHMEMI_WARN_PRINT("Got %zd unexpected events on EP %zu\n", qstatus, i);
                    }
                }
            } while (qstatus > 0);
            if (qstatus < 0 && qstatus != -FI_EAGAIN) {
                NVSHMEMI_WARN_PRINT("Error progressing CQ (%zd): %s\n", qstatus,
                                    fi_strerror(qstatus * -1));
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }
    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        status = nvshmemt_libfabric_gdr_process_amos_ack(transport);
        if (status) {
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }
    return 0;
}

static int nvshmemt_libfabric_progress(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    int status;

    status = nvshmemt_libfabric_process_completions(transport);
    if (status) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        if (gdrRecvMutex.try_lock()) {
            status = nvshmemt_libfabric_gdr_process_amos(transport);
            gdrRecvMutex.unlock();
            if (status) {
                return NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }

    return 0;
}

static int nvshmemt_libfabric_gdr_process_amo(nvshmem_transport_t transport,
                                              nvshmemt_libfabric_gdr_op_ctx_t *op,
                                              nvshmemt_libfabric_gdr_op_ctx_t **send_elems,
                                              uint32_t sequence_count) {
    int status = 0;

    switch (op->send_amo.size) {
        case 2:
            status = perform_gdrcopy_amo<uint16_t>(transport, op, send_elems, sequence_count);
            break;
        case 4:
            status = perform_gdrcopy_amo<uint32_t>(transport, op, send_elems, sequence_count);
            break;
        case 8:
            status = perform_gdrcopy_amo<uint64_t>(transport, op, send_elems, sequence_count);
            break;
        default:
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "invalid element size encountered %u\n", op->send_amo.size);
    }

out:
    return status;
}

static int nvshmemt_libfabric_gdr_process_ack(nvshmem_transport_t transport,
                                              nvshmemt_libfabric_gdr_op_ctx_t *op) {
    nvshmemt_libfabric_gdr_ret_amo_op_t *ret = &op->ret_amo;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
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

static int nvshmemt_libfabric_gdr_process_amos_ack(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2];
    size_t num_retries = 0;
    int status = 0;

    for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
        do {
            do {
                status = libfabric_state->op_queue[i]->getNextAmoOps(
                    send_elems, &op, NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK);
            } while (try_again(transport, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_PROCESS_AMOS_GET_NEXT_ACK,
                               true));
            num_retries = 0;

            if (op) {
                nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[op->ep_index]);
                status = nvshmemt_libfabric_gdr_process_ack(transport, op);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "Unable to process atomic.\n");
                status =
                    fi_recv(ep.endpoint, (void *)op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                            fi_mr_desc(libfabric_state->mrs[i]), FI_ADDR_UNSPEC, &op->ofi_context);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "Unable to re-post recv.\n");
            }
        } while (op);
    }

out:
    return status;
}

static int nvshmemt_libfabric_gdr_process_amos(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2];
    size_t num_retries = 0;
    int status = 0;

    for (size_t i = 0; i < libfabric_state->domains.size(); i++) {
        do {
            do {
                status = libfabric_state->op_queue[i]->getNextAmoOps(
                    send_elems, &op, NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK);
            } while (try_again(
                transport, &status, &num_retries,
                NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_PROCESS_AMOS_GET_NEXT_NOT_ACK, true));
            num_retries = 0;

            if (op) {
                if (op->type == NVSHMEMT_LIBFABRIC_SEND) {
                    assert(send_elems[0] != NULL);
                    assert(send_elems[1] != NULL);
                    status = nvshmemt_libfabric_gdr_process_amo(transport, op, send_elems,
                                                                NVSHMEM_STAGED_AMO_SEQ_NUM);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "Unable to process atomic.\n");
                    /* Reposts recv in perform_gdrcopy_amo() */
                } else if (op->type == NVSHMEMT_LIBFABRIC_MATCH) {
                    assert(send_elems[0] != NULL);
                    assert(send_elems[1] != NULL);
                    status = nvshmemt_libfabric_gdr_process_amo(transport, op, send_elems,
                                                                op->send_amo.sequence_count);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "Unable to process atomic.\n");
                    /* Reposts recv in perform_gdrcopy_amo() */
                }
            }
        } while (op);
    }
out:
    return status;
}

static int nvshmemt_libfabric_put_signal_completion(nvshmem_transport_t transport,
                                                    nvshmemt_libfabric_endpoint_t &ep,
                                                    struct fi_cq_data_entry *entry,
                                                    fi_addr_t *addr) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_signal_op *sig_op = NULL;
    nvshmemt_libfabric_gdr_op_ctx_t *op = NULL;
    bool is_write_comp = entry->flags & FI_REMOTE_CQ_DATA;
    int status = 0, progress_count;
    uint64_t map_key;
    std::unordered_map<uint64_t, std::pair<nvshmemt_libfabric_gdr_op_ctx_t *, int>>::iterator iter;

    if (unlikely(*addr == FI_ADDR_NOTAVAIL)) {
        status = -1;
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Write w/imm returned with invalid src address.\n");
    }

    if (is_write_comp) {
        map_key = *addr << 32 | (uint32_t)entry->data;
        progress_count = -1;
    } else {
        sig_op = (nvshmemt_libfabric_gdr_signal_op *)container_of(
            entry->op_context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context);
        map_key = *addr << 32 | sig_op->sequence_count;
        progress_count = (int)sig_op->num_writes;

        /* The EFA provider has an inline send size of 32 bytes.
         * The gdr atomic fi_send message is 72 bytes and does not
         * fit inside the efa_provider's 32 byte inline send window.
         * Hence, we send a 32 byte nvshmemt_libfabric_gdr_signal_op over the wire,
         * and re-arrange the memory in-place to allow for re-use of the gdr atomic
         * code.
         */
        op = inplace_copy_sig_op_to_gdr_op(sig_op, ep.ep_index);
    }

    iter = ep.proxy_put_signal_comp_map->find(map_key);
    if (iter != ep.proxy_put_signal_comp_map->end()) {
        if (!is_write_comp) iter->second.first = op;
        iter->second.second += progress_count;
    } else {
        iter = ep.proxy_put_signal_comp_map
                   ->insert(std::make_pair(map_key, std::make_pair(op, progress_count)))
                   .first;
    }

    if (!iter->second.second) {
        if (is_write_comp) {
            op = iter->second.first;
        }

        libfabric_state->op_queue[ep.domain_index]->putToRecv(op,
                                                              NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK);
        ep.proxy_put_signal_comp_map->erase(iter);
    }

out:
    return status;
}

static int nvshmemt_libfabric_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    uint64_t completed;
    int ep_start_idx;
    int ep_end_idx;
    int status = 0;
    bool all_quieted;

    if (qp_index == NVSHMEMX_QP_HOST) {
        ep_start_idx = 0;
        ep_end_idx = libfabric_state->num_host_domains;
    } else {
        ep_start_idx = libfabric_state->num_host_domains;
        ep_end_idx = libfabric_state->eps.size();
    }

    if ((use_staged_atomics)
#ifdef NVSHMEM_USE_GDRCOPY
        || (use_gdrcopy == true)
#endif
    ) {
        for (;;) {
            all_quieted = true;
            for (int i = ep_start_idx; i < ep_end_idx; i++) {
                const nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[i]);
                completed = fi_cntr_read(ep.counter) + ep.completed_staged_atomics;
                if (ep.submitted_ops != completed) all_quieted = false;
            }
            if (all_quieted) break;

            /* FI_PROGRESS_MANUAL requires calling progress on every endpoint */
            if (nvshmemt_libfabric_progress(tcurr)) {
                status = NVSHMEMX_ERROR_INTERNAL;
                break;
            }
        }
    } else {
        for (int i = ep_start_idx; i < ep_end_idx; i++) {
            const nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[i]);
            status =
                fi_cntr_wait(ep.counter, ep.submitted_ops, NVSHMEMT_LIBFABRIC_QUIET_TIMEOUT_MS);
            if (status) {
                /* note - Status is negative for this function in error cases but
                 * fi_strerror only accepts positive values.
                 */
                NVSHMEMI_ERROR_PRINT("Error in quiet operation (%d): %s.\n", status,
                                     fi_strerror(status * -1));
                status = NVSHMEMX_ERROR_INTERNAL;
            }
        }
    }

    return status;
}

static int nvshmemt_libfabric_fence(struct nvshmem_transport *tcurr, int pe, int qp_index,
                                    int is_multi) {
    int status = nvshmemt_libfabric_quiet(tcurr, pe, qp_index);

    return status;
}

static int nvshmemt_libfabric_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("libfabric show info not implemented");
    return 0;
}

static int nvshmemt_libfabric_rma_impl(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                       rma_memdesc_t *remote, rma_memdesc_t *local,
                                       rma_bytesdesc_t bytesdesc, int qp_index, uint32_t *imm_data,
                                       nvshmemt_libfabric_endpoint_t &ep) {
    nvshmemt_libfabric_mem_handle_ep_t *remote_handle, *local_handle = NULL;
    void *local_mr_desc = NULL;
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    struct iovec p_op_l_iov;
    struct fi_msg_rma p_op_msg;
    struct fi_rma_iov p_op_r_iov;
    size_t op_size;
    uint64_t num_retries = 0;
    int status = 0;
    int target_ep, ep_idx, domain_idx;
    void *context = NULL;

    memset(&p_op_l_iov, 0, sizeof(struct iovec));
    memset(&p_op_msg, 0, sizeof(struct fi_msg_rma));
    memset(&p_op_r_iov, 0, sizeof(struct fi_rma_iov));

    ep_idx = ep.ep_index;
    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep_idx;

    if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
        nvshmemt_libfabric_gdr_op_ctx_t *gdr_ctx;
        do {
            status = libfabric_state->op_queue[domain_idx]->getNextSends((void **)(&gdr_ctx), 1);
        } while (try_again(tcurr, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_GET_NEXT_SENDS));
        NVSHMEMI_NULL_ERROR_JMP(gdr_ctx, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "Unable to get context buffer for put request.\n");
        context = &gdr_ctx->ofi_context;

        /* local->handle may be NULL for small operations (P ops) sent by value/inline */
        if (likely(local->handle != NULL)) {
            local_handle = &((nvshmemt_libfabric_mem_handle_t *)local->handle)->hdls[domain_idx];
            local_mr_desc = local_handle->local_desc;
        }
    }

    remote_handle = &((nvshmemt_libfabric_mem_handle_t *)remote->handle)->hdls[domain_idx];
    op_size = bytesdesc.elembytes * bytesdesc.nelems;

    if (verb.desc == NVSHMEMI_OP_P) {
        assert(!imm_data);  // Write w/ imm not suppored with NVSHMEMI_OP_P on Libfabric transport
        if (libfabric_state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            nvshmemt_libfabric_gdr_op_ctx_t *p_buf =
                container_of(context, nvshmemt_libfabric_gdr_op_ctx_t, ofi_context);
            num_retries = 0;
            do {
                p_buf->p_op.value = *(uint64_t *)local->ptr;
                status = fi_write(ep.endpoint, &p_buf->p_op.value, op_size,
                                  fi_mr_desc(libfabric_state->mrs[domain_idx]), target_ep,
                                  (uintptr_t)remote->ptr, remote_handle->key, context);
            } while (try_again(tcurr, &status, &num_retries,
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_EFA));
        } else {
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
                               NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_P_NON_EFA));
        }
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        uintptr_t remote_addr;
        if (libfabric_state->prov_infos[domain_idx]->domain_attr->mr_mode & FI_MR_VIRT_ADDR)
            remote_addr = (uintptr_t)remote->ptr;
        else
            remote_addr = (uintptr_t)remote->offset;
        do {
            if (imm_data) {
                status = fi_writedata(ep.endpoint, local->ptr, op_size, local_mr_desc, *imm_data,
                                      target_ep, remote_addr, remote_handle->key, context);
            } else
                status = fi_write(ep.endpoint, local->ptr, op_size, local_mr_desc, target_ep,
                                  remote_addr, remote_handle->key, context);
        } while (try_again(tcurr, &status, &num_retries,
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_PUT));
    } else if (verb.desc == NVSHMEMI_OP_G || verb.desc == NVSHMEMI_OP_GET) {
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
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_RMA_IMPL_OP_GET));
    } else {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "Invalid RMA operation specified.\n");
    }

    if (status) goto out;  // Status set by try_again
    ep.submitted_ops++;

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an RMA operation.\n");
    }

    return status;
}

static int nvshmemt_libfabric_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                                  rma_memdesc_t *remote, rma_memdesc_t *local,
                                  rma_bytesdesc_t bytesdesc, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    int ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);
    return nvshmemt_libfabric_rma_impl(tcurr, pe, verb, remote, local, bytesdesc, qp_index, NULL,
                                       ep);
}

static int nvshmemt_libfabric_gdr_amo(struct nvshmem_transport *transport, int pe, void *curetptr,
                                      amo_verb_t verb, amo_memdesc_t *remote,
                                      amo_bytesdesc_t bytesdesc, int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
    nvshmemt_libfabric_gdr_op_ctx_t *amo;
    uint64_t num_retries = 0;
    int target_ep, ep_idx, domain_idx;
    int status = 0;

    ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);
    domain_idx = ep.domain_index;
    target_ep = pe * libfabric_state->eps.size() + ep_idx;

    do {
        status = libfabric_state->op_queue[domain_idx]->getNextSends((void **)(&amo), 1);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_GET_NEXT_SENDS));
    NVSHMEMI_NULL_ERROR_JMP(amo, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to retrieve AMO operation.");

    amo->send_amo.op = verb.desc;
    amo->send_amo.target_addr = remote->remote_memdesc.ptr;
    amo->send_amo.ret_addr = remote->retptr;
    amo->send_amo.retflag = remote->retflag;
    amo->send_amo.swap_add = remote->val;
    amo->send_amo.size = bytesdesc.elembytes;
    amo->send_amo.src_pe = transport->my_pe;
    amo->type = NVSHMEMT_LIBFABRIC_SEND;
    amo->send_amo.comp = remote->cmp;

    num_retries = 0;
    do {
        status =
            fi_send(ep.endpoint, (void *)amo, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                    fi_mr_desc(libfabric_state->mrs[domain_idx]), target_ep, &amo->ofi_context);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_AMO_SEND));

    if (status) {
        NVSHMEMI_ERROR_PRINT("Received an error when trying to post an AMO operation.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
    } else {
        ep.submitted_ops += 2;
    }

out:
    return status;
}

static int nvshmemt_libfabric_amo(struct nvshmem_transport *transport, int pe, void *curetptr,
                                  amo_verb_t verb, amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc,
                                  int qp_index) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
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
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_AMO_ATOMICMSG));

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
                                         void *curetptr, amo_verb_t verb, amo_memdesc_t *remote,
                                         amo_bytesdesc_t bytesdesc, int qp_index,
                                         uint32_t sequence_count, uint16_t num_writes,
                                         nvshmemt_libfabric_endpoint_t &ep) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;
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
        status = libfabric_state->op_queue[domain_idx]->getNextSends((void **)(&context), 1);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_GET_NEXT_SENDS));
    NVSHMEMI_NULL_ERROR_JMP(context, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to retrieve signal operation buffer.");

    signal = (nvshmemt_libfabric_gdr_signal_op_t *)context;
    signal->type = NVSHMEMT_LIBFABRIC_MATCH;
    signal->op = verb.desc;
    signal->sequence_count = sequence_count;
    signal->target_addr = remote->remote_memdesc.ptr;
    signal->sig_val = remote->val;
    signal->num_writes = num_writes;
    signal->src_pe = transport->my_pe;

    num_retries = 0;
    do {
        status =
            fi_send(ep.endpoint, (void *)signal, sizeof(nvshmemt_libfabric_gdr_signal_op_t),
                    fi_mr_desc(libfabric_state->mrs[domain_idx]), target_ep, &context->ofi_context);
    } while (try_again(transport, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_GDR_SIGNAL_SEND));

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
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    uint32_t sequence_count = 0;
    int status = 0;

    int ep_idx = get_next_ep(libfabric_state, qp_index);
    nvshmemt_libfabric_endpoint_t &ep = *(libfabric_state->eps[ep_idx]);

    /* Get sequence number for this put-signal, with retry */
    uint64_t num_retries = 0;
    do {
        int32_t seq_num = ep.put_signal_seq_counter.next_seq_num();
        if (seq_num < 0) {
            status = -EAGAIN;
        } else {
            sequence_count = seq_num;
            status = 0;
        }
    } while (try_again(tcurr, &status, &num_retries,
                       NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_PUT_SIGNAL_UNORDERED_SEQ));

    if (unlikely(status)) {
        NVSHMEMI_ERROR_PRINT("Error in nvshmemt_put_signal_unordered while waiting for category\n");
        goto out;
    }

    assert(write_remote.size() == write_local.size() &&
           write_local.size() == write_bytes_desc.size());
    for (size_t i = 0; i < write_remote.size(); i++) {
        status =
            nvshmemt_libfabric_rma_impl(tcurr, pe, write_verb, &write_remote[i], &write_local[i],
                                        write_bytes_desc[i], qp_index, &sequence_count, ep);
        if (unlikely(status)) {
            NVSHMEMI_ERROR_PRINT(
                "Error in nvshmemt_put_signal_unordered, could not submit write #%lu\n", i);
            goto out;
        }
    }

    assert(use_staged_atomics == true);
    status =
        nvshmemt_libfabric_gdr_signal(tcurr, pe, NULL, sig_verb, sig_target, sig_bytes_desc,
                                      qp_index, sequence_count, (uint16_t)write_remote.size(), ep);
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
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)tcurr->state;
    uint64_t num_retries = 0;
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
                           NVSHMEMT_LIBFABRIC_TRY_AGAIN_CALL_SITE_ENFORCE_CST));

        libfabric_state->eps[ep_idx]->submitted_ops++;

        /* If try_again errors out, need to break for upper-layer to abort */
        if (unlikely(status != 0)) break;
    }

    return status;
}

static int nvshmemt_libfabric_release_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                                 nvshmem_transport_t t) {
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;
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
    nvshmemt_libfabric_state_t *libfabric_state = (nvshmemt_libfabric_state_t *)t->state;
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
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error registering memory region: %s\n",
                                  fi_strerror(status * -1));

            status =
                fi_mr_bind(fabric_handle->hdls[i].mr, &libfabric_state->eps[i]->endpoint->fid, 0);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error binding MR to EP %zu: %s\n", i, fi_strerror(status * -1));

            status = fi_mr_enable(fabric_handle->hdls[i].mr);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Error enabling MR: %s\n",
                                  fi_strerror(status * -1));

            fabric_handle->hdls[i].key = fi_mr_key(fabric_handle->hdls[i].mr);
            fabric_handle->hdls[i].local_desc = fi_mr_desc(fabric_handle->hdls[i].mr);
        } else {
            struct fid_mr *mr;

            status = fi_mr_regattr(libfabric_state->domains[i], &mr_attr, 0, &mr);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Error registering memory region: %s\n",
                                  fi_strerror(status * -1));

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
                                             struct nvshmem_transport_pe_info *peer_info,
                                             nvshmem_transport_t t) {
    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return 0;
}

static int nvshmemt_libfabric_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                                int num_selected_devs, int *out_qp_indices,
                                                int num_qps) {
    nvshmemt_libfabric_state_t *state = (nvshmemt_libfabric_state_t *)t->state;
    std::vector<nvshmemt_libfabric_ep_name_t> all_ep_names;
    std::vector<nvshmemt_libfabric_ep_name_t> local_ep_names;
    struct fi_info *current_info;
    struct fid_fabric *fabric;
    struct fid_domain *domain;
    struct fid_av *address;
    struct fid_mr *mr;
    struct fi_av_attr av_attr;
    struct fi_cq_attr cq_attr;
    struct fi_cntr_attr cntr_attr;
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

    memset(&cntr_attr, 0, sizeof(struct fi_cntr_attr));
    cntr_attr.events = FI_CNTR_EVENTS_COMP;
    cntr_attr.wait_obj = FI_WAIT_UNSPEC;

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

    /* Create resources for each domain */
    for (size_t i = 0; i < state->prov_infos.size(); i++) {
        INFO(state->log_level, "Selected provider %s, fabric %s, nic %s, hmem %s domain %zu/%zu",
             state->prov_infos[i]->fabric_attr->prov_name, state->prov_infos[i]->fabric_attr->name,
             state->prov_infos[i]->nic->device_attr->name,
             state->prov_infos[i]->caps & FI_HMEM ? "yes" : "no", i + 1, state->prov_infos.size());

        if (state->prov_infos[i]->ep_attr->max_msg_size < t->max_op_len)
            t->max_op_len = state->prov_infos[i]->ep_attr->max_msg_size;

        status = fi_fabric(state->prov_infos[i]->fabric_attr, &fabric, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to allocate fabric: %d: %s\n", status,
                              fi_strerror(status * -1));
        state->fabrics.push_back(fabric);

        status = fi_domain(fabric, state->prov_infos[i], &domain, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to allocate domain: %d: %s\n", status,
                              fi_strerror(status * -1));
        state->domains.push_back(domain);

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            num_sends = state->prov_infos[i]->tx_attr->size;
            num_recvs = state->prov_infos[i]->rx_attr->size;
            elem_size = sizeof(nvshmemt_libfabric_gdr_op_ctx_t);
            num_recvs_per_ep = num_recvs;

            state->recv_buf.push_back(calloc(num_sends + num_recvs, elem_size));
            NVSHMEMI_NULL_ERROR_JMP(state->recv_buf[i], status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                    "Unable to allocate EFA msg buffer.\n");
            state->send_buf.push_back((char *)state->recv_buf[i] + (elem_size * num_recvs));

            status = fi_mr_reg(domain, state->recv_buf[i], (num_sends + num_recvs) * elem_size,
                               FI_SEND | FI_RECV | FI_WRITE, 0, 0, 0, &mr, NULL);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Failed to register EFA msg buffer: %d: %s\n", status,
                                  fi_strerror(status * -1));
            state->mrs.push_back(mr);

            state->op_queue.emplace_back(
                std::unique_ptr<threadSafeOpQueue>(new threadSafeOpQueue()));
            NVSHMEMI_NULL_ERROR_JMP(state->op_queue[i], status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                    "Unable to alloc thread-safe op queue struct.\n");
            state->op_queue.back()->putToSendBulk((char *)state->send_buf[i], elem_size, num_sends);
        }

        status = fi_av_open(domain, &av_attr, &address, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to allocate address vector: %d: %s\n", status,
                              fi_strerror(status * -1));
        state->addresses.push_back(address);

        /* Create endpoint resources */
        state->eps.emplace_back(
            std::unique_ptr<nvshmemt_libfabric_endpoint_t>(new nvshmemt_libfabric_endpoint_t()));
        NVSHMEMI_NULL_ERROR_JMP(state->eps[i], status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to alloc nvshmemt_libfabric_endpoint_t struct.\n");
        state->eps[i]->domain_index = i;
        state->eps[i]->ep_index = i;

        /* Initialize per-endpoint proxy_put_signal_comp_map */
        state->eps[i]->proxy_put_signal_comp_map =
            new std::unordered_map<uint64_t, std::pair<nvshmemt_libfabric_gdr_op_ctx_t *, int>>();

        state->eps[i]->put_signal_seq_counter.reset();
        state->eps[i]->completed_staged_atomics = 0;

        status = fi_cq_open(domain, &cq_attr, &state->eps[i]->cq, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to open completion queue for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        status = fi_cntr_open(domain, &cntr_attr, &state->eps[i]->counter, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to open counter for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        status = fi_endpoint(domain, state->prov_infos[i], &state->eps[i]->endpoint, NULL);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to allocate endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

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
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "Unable to set endpoint CUDA API status: %d: %s\n", status,
                                      fi_strerror(status * -1));
            }
        }

        /* Bind resources to EP */
        status = fi_ep_bind(state->eps[i]->endpoint, &state->addresses[i]->fid, 0);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to bind endpoint to address vector: %d: %s\n", status,
                              fi_strerror(status * -1));

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
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to bind endpoint to completion queue: %d: %s\n", status,
                              fi_strerror(status * -1));

        flags = FI_READ | FI_WRITE;
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) flags |= FI_SEND;
#endif
        if (use_staged_atomics) flags |= FI_SEND;

        status = fi_ep_bind(state->eps[i]->endpoint, &state->eps[i]->counter->fid, flags);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to bind endpoint to completion counter: %d: %s\n", status,
                              fi_strerror(status * -1));

        status = fi_enable(state->eps[i]->endpoint);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to enable endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));

        if (state->provider == NVSHMEMT_LIBFABRIC_PROVIDER_EFA) {
            nvshmemt_libfabric_gdr_op_ctx_t *op =
                (nvshmemt_libfabric_gdr_op_ctx_t *)state->recv_buf[i];
            for (size_t j = 0; j < num_recvs_per_ep; j++, op++) {
                assert(op != NULL);
                status = fi_recv(state->eps[i]->endpoint, op, NVSHMEM_STAGED_AMO_WIREDATA_SIZE,
                                 fi_mr_desc(state->mrs[i]), FI_ADDR_UNSPEC, &op->ofi_context);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "Unable to post recv to ep. Error: %d: %s\n", status,
                                      fi_strerror(status * -1));
            }
        }

        status =
            fi_getname(&state->eps[i]->endpoint->fid, local_ep_names[i].name.data(), &ep_namelen);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to get name for endpoint: %d: %s\n", status,
                              fi_strerror(status * -1));
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
            if (status < 1) {
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to insert ep names in address vector: %d: %s\n", status,
                                   fi_strerror(status * -1));
            }

            status = NVSHMEMX_SUCCESS;
        }
    }

    /* Exchange a pre-registered write w/imm target for staged_amo acks */
    if (use_staged_atomics) {
        state->remote_addr_staged_amo_ack = (void **)calloc(sizeof(void *), t->n_pes);
        NVSHMEMI_NULL_ERROR_JMP(state->remote_addr_staged_amo_ack, status,
                                NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate remote address array for staged atomic ack.\n");

        state->rkey_staged_amo_ack =
            (uint64_t *)calloc(sizeof(uint64_t), t->n_pes * state->domains.size());
        NVSHMEMI_NULL_ERROR_JMP(state->rkey_staged_amo_ack, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                                out, "Unable to allocate rkey array for staged atomic ack.\n");

        status = cudaMalloc(&state->remote_addr_staged_amo_ack[t->my_pe], sizeof(int));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to allocate CUDA memory for staged atomic ack.\n");

        for (size_t i = 0; i < state->domains.size(); i++) {
            status = fi_mr_reg(state->domains[i], state->remote_addr_staged_amo_ack[t->my_pe],
                               sizeof(int), FI_REMOTE_WRITE, 0, 0, 0, &mr, NULL);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Failed to register EFA msg buffer: %d: %s\n", status,
                                  fi_strerror(status * -1));
            state->rkey_staged_amo_ack[t->my_pe * state->domains.size() + i] = fi_mr_key(mr);
            state->mr_staged_amo_acks.push_back(mr);
        }

        status = t->boot_handle->allgather(&state->remote_addr_staged_amo_ack[t->my_pe],
                                           state->remote_addr_staged_amo_ack, sizeof(void *),
                                           t->boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to gather remote addresses.\n");

        status = t->boot_handle->allgather(
            &state->rkey_staged_amo_ack[t->my_pe * state->domains.size()],
            state->rkey_staged_amo_ack, sizeof(uint64_t) * state->domains.size(), t->boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to gather remote keys.\n");
    }

out:
    if (status != 0) {
        if (state->remote_addr_staged_amo_ack) {
            if (state->remote_addr_staged_amo_ack[t->my_pe])
                cudaFree(state->remote_addr_staged_amo_ack[t->my_pe]);
            free(state->remote_addr_staged_amo_ack);
        }
        if (state->rkey_staged_amo_ack) free(state->rkey_staged_amo_ack);
        for (size_t i = 0; i < state->mr_staged_amo_acks.size(); i++)
            fi_close(&state->mr_staged_amo_acks[i]->fid);
        for (size_t i = 0; i < state->eps.size(); i++) {
            if (state->eps[i]->proxy_put_signal_comp_map)
                delete state->eps[i]->proxy_put_signal_comp_map;
            if (state->eps[i]->endpoint) {
                fi_close(&state->eps[i]->endpoint->fid);
                state->eps[i]->endpoint = NULL;
            }
            if (state->eps[i]->cq) {
                fi_close(&state->eps[i]->cq->fid);
                state->eps[i]->cq = NULL;
            }
            if (state->eps[i]->counter) {
                fi_close(&state->eps[i]->counter->fid);
                state->eps[i]->counter = NULL;
            }
        }
        state->eps.clear();
        state->op_queue.clear();
    }

out_already_connected:
    return status;
}

static int nvshmemt_libfabric_finalize(nvshmem_transport_t transport) {
    nvshmemt_libfabric_state_t *libfabric_state;
    int status;

    assert(transport);

    libfabric_state = (nvshmemt_libfabric_state_t *)transport->state;

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

    for (size_t i = 0; i < libfabric_state->eps.size(); i++) {
        if (libfabric_state->eps[i]->proxy_put_signal_comp_map)
            delete libfabric_state->eps[i]->proxy_put_signal_comp_map;
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
        if (libfabric_state->eps[i]->counter) {
            status = fi_close(&libfabric_state->eps[i]->counter->fid);
            if (status) {
                NVSHMEMI_WARN_PRINT("Unable to close fabric counter: %d: %s\n", status,
                                    fi_strerror(status * -1));
            }
        }
    }

    if (libfabric_state->remote_addr_staged_amo_ack) {
        if (libfabric_state->remote_addr_staged_amo_ack[transport->my_pe])
            cudaFree(libfabric_state->remote_addr_staged_amo_ack[transport->my_pe]);
        free(libfabric_state->remote_addr_staged_amo_ack);
    }
    if (libfabric_state->rkey_staged_amo_ack) free(libfabric_state->rkey_staged_amo_ack);
    for (size_t i = 0; i < libfabric_state->mr_staged_amo_acks.size(); i++) {
        status = fi_close(&libfabric_state->mr_staged_amo_acks[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close staged atomic ack MR: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }

    for (size_t i = 0; i < libfabric_state->mrs.size(); i++) {
        status = fi_close(&libfabric_state->mrs[i]->fid);
        if (status) {
            NVSHMEMI_WARN_PRINT("Unable to close fabric MR: %d: %s\n", status,
                                fi_strerror(status * -1));
        }
    }
    for (size_t i = 0; i < libfabric_state->recv_buf.size(); i++)
        free(libfabric_state->recv_buf[i]);

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

    free(libfabric_state);
    free(transport);

    return 0;
}

static int nvshmemi_libfabric_init_state(nvshmem_transport_t t, nvshmemt_libfabric_state_t *state) {
    struct fi_info *all_infos, *current_info;
    size_t num_fabrics_returned = 0;
    int num_devices = 0;
    int status = 0;

    auto deleter = [](fi_info* p){ fi_freeinfo(p); };
    std::unique_ptr<fi_info, decltype(deleter)> hints{ fi_allocinfo(), deleter };
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

    if (use_staged_atomics) {
        hints->mode |= FI_CONTEXT2;
    }

    /* Ensure manual progress mode until auto progress is implemented */
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;

    /* Be thread safe at the level of the endpoint completion context. */
    hints->domain_attr->threading = FI_THREAD_SAFE;

    /* Require completion RMA completion at target for correctness of quiet */
    hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;

    hints->ep_attr->type = FI_EP_RDM;  // Reliable datagrams

    status = fi_getinfo(FI_VERSION(NVSHMEMT_LIBFABRIC_MAJ_VER, NVSHMEMT_LIBFABRIC_MIN_VER), NULL,
                        NULL, 0, hints.get(), &all_infos);

    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "No providers matched fi_getinfo query: %d: %s\n", status,
                          fi_strerror(status * -1));
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
    struct nvshmemi_options_s options;
    int status = 0;

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

    libfabric_state = (nvshmemt_libfabric_state_t *)calloc(1, sizeof(*libfabric_state));
    NVSHMEMI_NULL_ERROR_JMP(libfabric_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for libfabric transport state.");
    libfabric_state->table = table;
    transport->state = libfabric_state;

    transport->host_ops.can_reach_peer = nvshmemt_libfabric_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_libfabric_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_libfabric_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_libfabric_release_mem_handle;
    transport->host_ops.rma = nvshmemt_libfabric_rma;
    transport->host_ops.fence = nvshmemt_libfabric_fence;
    transport->host_ops.quiet = nvshmemt_libfabric_quiet;
    transport->host_ops.finalize = nvshmemt_libfabric_finalize;
    transport->host_ops.show_info = nvshmemt_libfabric_show_info;
    transport->host_ops.progress = nvshmemt_libfabric_progress;
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
        use_staged_atomics = true;
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
