/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <assert.h>
#include <stdint.h>  // IWYU pragma: keep
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <atomic>
#include <array>
#include <deque>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <memory>
#include "rdma/fabric.h"

// IWYU pragma: no_include <bits/stdint-uintn.h>

#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif

#define NVSHMEMT_LIBFABRIC_MAJ_VER 1
#define NVSHMEMT_LIBFABRIC_MIN_VER 15

#define NVSHMEMT_LIBFABRIC_DOMAIN_LEN 32
#define NVSHMEMT_LIBFABRIC_PROVIDER_LEN 32
#define NVSHMEMT_LIBFABRIC_EP_LEN 128

#define NVSHMEMT_LIBFABRIC_QUIET_TIMEOUT_MS 20

/* Maximum size of inject data. Currently
 * the max size we will use is one element
 * of a given type. Making it 16 bytes in the
 * case of complex number support. */
#ifdef NVSHMEM_COMPLEX_SUPPORT
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 16
#else
#define NVSHMEMT_LIBFABRIC_INJECT_BYTES 8
#endif

#define NVSHMEMT_LIBFABRIC_MAX_RETRIES (1ULL << 20)

#ifndef container_of
#define container_of(ptr, type, field) ((type *)((char *)ptr - offsetof(type, field)))
#endif

typedef struct {
    std::array<char, NVSHMEMT_LIBFABRIC_DOMAIN_LEN> name;
} nvshmemt_libfabric_domain_name_t;

typedef struct {
    std::array<char, NVSHMEMT_LIBFABRIC_EP_LEN> name;
} nvshmemt_libfabric_ep_name_t;

struct nvshmemt_libfabric_gdr_op_ctx;
typedef struct nvshmemt_libfabric_gdr_op_ctx nvshmemt_libfabric_gdr_op_ctx_t;

#define NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT 28
#define NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK \
    ((1U << NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT) - 1)

/**
 * The last sequence number is reserved for atomic-only operations.
 * This will not be returned by the sequence counter.
 */
#define NVSHMEM_STAGED_AMO_SEQ_NUM NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK

/**
 * Type used for tracking sequence numbers for put-signal operations
 *
 * A sequence number is associated with each put-signal operation. This type
 * manages allocation of sequence numbers and return of sequence numbers via
 * acks from the remote side.
 */
struct nvshmemt_libfabric_endpoint_seq_counter_t {
    constexpr static uint32_t num_sequence_bits = NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_SHIFT;

    /**
     * Sequence counter is composed of:
     *
     * |----------<num_sequence_bits>----------|
     * | <num_category_bits> <num_index_bits>  |
     */
    constexpr static uint32_t num_category_bits = 1;
    constexpr static uint32_t num_categories = (1U << num_category_bits);

    constexpr static uint32_t num_index_bits = (num_sequence_bits - num_category_bits);

    constexpr static uint32_t index_mask = ((1U << num_index_bits) - 1);
    constexpr static uint32_t category_mask = (1U << num_index_bits);

    constexpr static uint32_t sequence_mask = NVSHMEM_STAGED_AMO_PUT_SIGNAL_SEQ_CNTR_BIT_MASK;

    /**
     * The "category" is the bits before the index bit(s). Returns the category
     * bits shifted to the right.
     */
    constexpr static uint32_t get_category(uint32_t seq_num) {
        return ((seq_num & category_mask) >> num_index_bits);
    }

    /**
     * The "index" is the bits after the category bit(s)
     */
    constexpr static uint32_t get_index(uint32_t seq_num) { return seq_num & index_mask; }

    /* ------------------------------ */
    /* Member variables */

    uint32_t sequence_counter;
    std::array<uint32_t, num_categories> pending_acks;

    /**
     * Reset counter and pending acks to zero
     */
    void reset() {
        sequence_counter = 0;
        pending_acks.fill(0);
    }

    /**
     * Obtain the next sequence number.
     *
     * @return -1 if no sequence number available
     */
    int32_t next_seq_num() {
        /* Skip this sequence number if reserved */
        if (sequence_counter == NVSHMEM_STAGED_AMO_SEQ_NUM) {
            sequence_counter = (sequence_counter + 1) & sequence_mask;
        }

        uint32_t seq_num = sequence_counter;

        uint32_t category = get_category(seq_num);

        /* For first sequence number of category, check if category is available */
        if (get_index(seq_num) == 0) {
            if (pending_acks[category] != 0) {
                /* No sequence number available */
                return -1;
            }
        }

        /* Can't have more outstanding acks than sequence numbers in the category */
        assert(pending_acks[category] <= index_mask);
        /* Increment pending acks */
        ++pending_acks[category];

        /* Increment sequence counter */
        sequence_counter = (sequence_counter + 1) & sequence_mask;
        return seq_num;
    }

    /**
     * Mark a previously issued seq_num as complete, decremeting the pending
     * acks counter for the category
     */
    void return_acked_seq_num(uint32_t seq_num) {
        assert(seq_num != NVSHMEM_STAGED_AMO_SEQ_NUM);

        uint32_t category = get_category(seq_num);

        assert(pending_acks[category] > 0);
        --pending_acks[category];
    }
};

typedef enum {
    NVSHMEMT_LIBFABRIC_SEND,
    NVSHMEMT_LIBFABRIC_ACK,
    NVSHMEMT_LIBFABRIC_MATCH,
} nvshmemt_libfabric_recv_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK,
    NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK,
} nvshmemt_libfabric_recv_type_t;

typedef struct {
    struct fid_ep *endpoint;
    struct fid_cq *cq;
    struct fid_cntr *counter;
    uint64_t submitted_ops;
    uint64_t completed_staged_atomics;
    nvshmemt_libfabric_endpoint_seq_counter_t put_signal_seq_counter;
    std::unordered_map<uint64_t, std::pair<nvshmemt_libfabric_gdr_op_ctx_t *, int>>
        *proxy_put_signal_comp_map;
    int domain_index;
    int ep_index;
} nvshmemt_libfabric_endpoint_t;

typedef struct nvshmemt_libfabric_gdr_send_p_op {
    uint64_t value;
} nvshmemt_libfabric_gdr_send_p_op_t;

typedef struct nvshmemt_libfabric_gdr_send_amo_op {
    nvshmemi_amo_t op;
    void *target_addr;
    void *ret_addr;
    union {
        uint64_t retflag;
        uint32_t sequence_count;
    };
    uint64_t swap_add;
    uint64_t comp;
    uint32_t size;
    int src_pe;
} nvshmemt_libfabric_gdr_send_amo_op_t;

typedef struct nvshmemt_libfabric_gdr_ret_amo_op {
    void *ret_addr;
    g_elem_t elem;
} nvshmemt_libfabric_gdr_ret_amo_op_t;

struct nvshmemt_libfabric_gdr_op_ctx {
    nvshmemt_libfabric_recv_t type;
    int ep_index;
    union {
        nvshmemt_libfabric_gdr_send_p_op_t p_op;
        nvshmemt_libfabric_gdr_send_amo_op_t send_amo;
        nvshmemt_libfabric_gdr_ret_amo_op_t ret_amo;
    };
    struct fi_context2 ofi_context;
    fi_addr_t src_addr;
};

typedef enum {
    NVSHMEMT_LIBFABRIC_PROVIDER_VERBS = 0,
    NVSHMEMT_LIBFABRIC_PROVIDER_SLINGSHOT,
    NVSHMEMT_LIBFABRIC_PROVIDER_EFA
} nvshmemt_libfabric_provider;

typedef enum {
    NVSHMEMT_LIBFABRIC_CONTEXT_P_OP = 0,
    NVSHMEMT_LIBFABRIC_CONTEXT_SEND_AMO,
    NVSHMEMT_LIBFABRIC_CONTEXT_RECV_AMO
} nvshemmt_libfabric_context_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_IMM_PUT_SIGNAL_SEQ = 0,
    NVSHMEMT_LIBFABRIC_IMM_STAGED_ATOMIC_ACK,
} nvshmemt_libfabric_imm_cq_data_hdr_t;

class threadSafeOpQueue {
   private:
    std::mutex send_mutex;
    std::mutex ack_recv_mutex;
    std::mutex other_recv_mutex;
    std::vector<void *> send;
    std::deque<void *> ack_recv;
    std::deque<void *> other_recv;

   public:
    threadSafeOpQueue() = default;
    threadSafeOpQueue(const threadSafeOpQueue &) = delete;
    threadSafeOpQueue &operator=(const threadSafeOpQueue &) = delete;
    threadSafeOpQueue(threadSafeOpQueue &&) = delete;
    threadSafeOpQueue &operator=(threadSafeOpQueue &&) = delete;

    int getNextSends(void **elems, size_t num_elems = 1) {
        const std::lock_guard<std::mutex> lg{send_mutex};
        if (send.size() < num_elems) {
            for (size_t i = 0; i < num_elems; i++) {
                elems[i] = NULL;
            }
            return -EAGAIN;
        }
        for (size_t i = 0; i < num_elems; i++) {
            elems[i] = send.back();
            send.pop_back();
            assert(elems[i] != NULL);
        }
        return 0;
    }

    int getNextAmoOps(nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2],
                      nvshmemt_libfabric_gdr_op_ctx_t **recv_elem,
                      nvshmemt_libfabric_recv_type_t recv_type) {
        int status = 0;
        int num_sends = 0;

        if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK) {
            const std::lock_guard<std::mutex> lg{other_recv_mutex};
            if (other_recv.empty()) {
                *recv_elem = NULL;
                return 0;
            }
            *recv_elem = (nvshmemt_libfabric_gdr_op_ctx_t *)other_recv.front();
            if ((&((*recv_elem)->send_amo))->op > NVSHMEMI_AMO_END_OF_NONFETCH) {
                num_sends = 2;
            } else {
                num_sends = 1;
            }
            status = getNextSends((void **)send_elems, num_sends);
            if (status == -EAGAIN) {
                *recv_elem = NULL;
                return -EAGAIN;
            }
            assert(recv_elem != NULL);
            for (int i = 0; i < num_sends; i++) {
                assert(send_elems[i] != NULL);
            }
            other_recv.pop_front();
            return 0;
        } else if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK) {
            const std::lock_guard<std::mutex> lg{ack_recv_mutex};
            if (ack_recv.empty()) {
                *recv_elem = NULL;
                return 0;
            }
            *recv_elem = (nvshmemt_libfabric_gdr_op_ctx_t *)ack_recv.front();
            assert(*recv_elem != NULL);
            ack_recv.pop_front();
            return 0;
        } else {
            fprintf(stderr, "getNextAmoOps: invalid recv_type: %d\n", recv_type);
            assert(false);
            return -EINVAL;
        }
    }

    void putToSend(void *elem) {
        const std::lock_guard<std::mutex> lg{send_mutex};
        send.push_back(elem);
        return;
    }

    void putToSendBulk(char *elem, size_t elem_size, size_t num_elems) {
        const std::lock_guard<std::mutex> lg{send_mutex};
        for (size_t i = 0; i < num_elems; i++) {
            send.push_back(elem);
            elem = elem + elem_size;
        }
        return;
    }

    void *getNextRecv(nvshmemt_libfabric_recv_type_t recv_type) {
        void *elem = NULL;
        if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK) {
            const std::lock_guard<std::mutex> lg{ack_recv_mutex};
            if (ack_recv.empty()) {
                return NULL;
            }

            elem = ack_recv.front();
            ack_recv.pop_front();
            return elem;
        } else {
            const std::lock_guard<std::mutex> lg{other_recv_mutex};
            if (other_recv.empty()) {
                return NULL;
            }
            elem = other_recv.front();
            other_recv.pop_front();
            return elem;
        }
    }

    void putToRecv(void *elem, nvshmemt_libfabric_recv_type_t recv_type) {
        if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK) {
            const std::lock_guard<std::mutex> lg{ack_recv_mutex};
            ack_recv.push_back(elem);
        } else if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK) {
            const std::lock_guard<std::mutex> lg{other_recv_mutex};
            other_recv.push_back(elem);
        } else {
            fprintf(stderr, "putToRecv: invalid recv_type: %d\n", recv_type);
            assert(false);
            return;
        }
    }
};

/*
 * Each index of the vectors contain a domain-specific resource. Host domain resources are first,
 * proceeded by proxy domain resources. The number of each domain type is specified by
 * num_host_domains and num_proxy_domains. These are assigned in the beginning of connect_endpoints.
 * Upon completion of connect_endpoints, devices.size() == (num_host_domains + num_proxy_domains)
 *
 * Currently there is a 1-to-1 relationship between endpoints and domains. However, that may change
 * in the future. That is, it may be the case that eps.size() != devices.size(). The domain index
 * of an endpoint is stored directly in nvshmemt_libfabric_endpoint_t (domain_index).
 */
typedef struct {
    struct fi_info *all_prov_info;
    std::vector<struct fi_info *> prov_infos;
    std::vector<struct fid_fabric *> fabrics;
    std::vector<struct fid_domain *> domains;
    std::vector<struct fid_av *> addresses;
    std::vector<std::unique_ptr<nvshmemt_libfabric_endpoint_t>> eps;

    /* local_mr is used only for consistency ops. */
    std::vector<struct fid_mr *> local_mrs;
    std::vector<uint64_t> local_mr_keys;
    std::vector<void *> local_mr_descs;
    void *local_mem_ptr;

    std::vector<nvshmemt_libfabric_domain_name_t> domain_names;
    nvshmemt_libfabric_provider provider;
    int log_level;
    struct nvshmemi_cuda_fn_table *table;
    struct transport_mem_handle_info_cache *cache;

    /* Required for multi-rail */
    int num_host_domains;
    int num_proxy_domains;
    int num_selected_devs;
    int max_nic_per_pe;
    std::atomic<uint32_t> proxy_ep_cntr;

    /* Required for staged_amo */
    std::vector<std::unique_ptr<threadSafeOpQueue>> op_queue;
    std::vector<void *> send_buf;
    std::vector<void *> recv_buf;
    std::vector<struct fid_mr *> mrs;
    std::vector<struct fid_mr *> mr_staged_amo_acks;
    void **remote_addr_staged_amo_ack;
    uint64_t *rkey_staged_amo_ack;
} nvshmemt_libfabric_state_t;

typedef struct {
    struct fid_mr *mr;
    uint64_t key;
    void *local_desc;
} nvshmemt_libfabric_mem_handle_ep_t;

typedef struct {
    size_t gdr_mapping_size;
    void *ptr;
    void *cpu_ptr;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_mh_t mh;
    void *cpu_ptr_base;
#endif
} nvshmemt_libfabric_memhandle_info_t;

struct nvshmemt_libfabric_mem_handle_base_t {
    void *buf;
};

struct nvshmemt_libfabric_mem_handle_t : nvshmemt_libfabric_mem_handle_base_t {
    static constexpr size_t MAX_SIZE = sizeof(nvshmem_mem_handle_t);
    static constexpr size_t BASE_SIZE = sizeof(nvshmemt_libfabric_mem_handle_base_t);
    static constexpr size_t NUM_HDLS =
        (MAX_SIZE - BASE_SIZE) / sizeof(nvshmemt_libfabric_mem_handle_ep_t);

    /* Each domain needs 1 handle */
    std::array<nvshmemt_libfabric_mem_handle_ep_t, NUM_HDLS> hdls;

    /* Constrained by the size of nvshmem_mem_handle_t */
};

static constexpr size_t NVSHMEMT_LIBFABRIC_MAX_DOMAINS_PER_PE =
    nvshmemt_libfabric_mem_handle_t::NUM_HDLS;

typedef struct nvshmemt_libfabric_mem_handle_t nvshmemt_libfabric_mem_handle_t;
static_assert(sizeof(nvshmemt_libfabric_mem_handle_t) <= nvshmemt_libfabric_mem_handle_t::MAX_SIZE);

/* Wire data for put-signal gdr staged atomics
 * 32 bytes
 * | 4 type | 2 op | 2 num_writes | 8 signal | 8 target_addr | 4 sequence_count | 4 resv
 */
typedef struct nvshmemt_libfabric_gdr_signal_op {
    nvshmemt_libfabric_recv_t type; /* Must be first */
    uint16_t op;
    uint16_t num_writes;
    uint64_t sig_val;
    void *target_addr;
    uint32_t sequence_count;
    uint32_t src_pe;
} nvshmemt_libfabric_gdr_signal_op_t;
/*  EFA's inline send size is 32 bytes */
static_assert(sizeof(nvshmemt_libfabric_gdr_signal_op_t) == 32);
