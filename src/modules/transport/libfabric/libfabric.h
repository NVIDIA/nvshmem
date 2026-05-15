/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdint.h>  // IWYU pragma: keep
#include <stdio.h>
#include <cstdlib>
#include <stddef.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <atomic>
#include <array>
#include <deque>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <memory>
#include <cuda_runtime.h>
#include "rdma/fabric.h"

// IWYU pragma: no_include <bits/stdint-uintn.h>

#include "non_abi/nvshmem_build_options.h"
#include "non_abi/nvshmemx_error.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "internal/host_transport/transport.h"

#if defined(NVSHMEM_X86_64)
#include <immintrin.h>  // IWYU pragma: keep
#define NVSHMEMT_LIBFABRIC_CPU_RELAX() _mm_pause()
#elif defined(NVSHMEM_AARCH64)
#define NVSHMEMT_LIBFABRIC_CPU_RELAX() asm volatile("yield" ::: "memory")
#elif defined(NVSHMEM_PPC64LE)
#define NVSHMEMT_LIBFABRIC_CPU_RELAX() asm volatile("or 27,27,27" ::: "memory")
#else
#define NVSHMEMT_LIBFABRIC_CPU_RELAX() asm volatile("" ::: "memory")
#endif

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

static constexpr size_t nvshmemt_libfabric_signal_queue_capacity = 1024;

/**
 * Type used for tracking sequence numbers for put-signal operations
 *
 * A sequence number is associated with each put-signal operation. This type
 * manages allocation of sequence numbers and return of sequence numbers via
 * acks from the remote side.
 */
struct nvshmemt_libfabric_endpoint_seq_counter_t {
    constexpr static uint32_t num_sequence_bits = 16;
    constexpr static uint32_t bit_shift = num_sequence_bits;
    constexpr static uint32_t bit_mask = (1U << bit_shift) - 1;

    /**
     * The last sequence number is reserved for atomic-only operations.
     * This will not be returned by the sequence counter.
     */
    constexpr static uint32_t last_seq_num = bit_mask;

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

    /**
     * Frequency at which we send an ack for puts (without signal). For puts-only,
     * we don't need the ack for every message for semantic reasons. We only need
     * an occasional ack to handle sequence number overflow correctly.
     */
    constexpr static uint32_t put_ack_freq = 64;

    /* Assert that index_mask is large enough to simplify some ranged ack return
       logic. */
    static_assert((index_mask + 1) >= (2 * put_ack_freq),
                  "Number of indexes should be >= 2 * put_ack_freq");
    constexpr static uint32_t category_mask = (1U << num_index_bits);

    constexpr static uint32_t sequence_mask = bit_mask;

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
    uint32_t put_count;

    /**
     * Default constructor - initializes counter to zero
     */
    nvshmemt_libfabric_endpoint_seq_counter_t() {
        reset();
    }

    /**
     * Reset counter and pending acks to zero
     */
    void reset() {
        sequence_counter = 0;
        pending_acks.fill(0);
        put_count = 0;
    }

    /**
     * If seq_num == last_seq_num, increment by 1
     */
    static constexpr uint32_t seq_num_wrapup(uint32_t seq_num) {
        if (seq_num == last_seq_num) {
            return (seq_num + 1) & sequence_mask;
        } else {
            return seq_num;
        }
    }

    /**
     * If seq_num == last_seq_num, decrement by 1
     */
    static constexpr uint32_t seq_num_wrapdown(uint32_t seq_num) {
        if (seq_num == last_seq_num) {
            return (seq_num - 1) & sequence_mask;
        } else {
            return seq_num;
        }
    }

    /**
     * Obtain the next sequence number.
     *
     * @return -1 if no sequence number available
     */
    int32_t next_seq_num() {
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
        sequence_counter = seq_num_wrapup(sequence_counter + 1);
        return seq_num;
    }

    /**
     * Mark a range of sequence numbers as complete, resulting from reciving an
     * ack. The sequence range ends with end_seq. Decrement pending_acks by
     * count, ending at end_seq. Distributes the count across categories based
     * on end_seq position.
     *
     * The wraparound case is also handled.
     *
     * This code assumes the sequence range spans at most two categories. This
     * will be true as long as the index space is sufficiently larger than the
     * put ack frequency, as static asserted above.
     */
    void return_acked_range(uint32_t end_seq, uint32_t count) {
        assert(count > 0);
        assert(end_seq != last_seq_num);

        uint32_t end_category = get_category(end_seq);
        uint32_t end_index = get_index(end_seq);

        /* Note: in the wraparound case, the (start_seq, end_seq) range will include last_seq_num,
           which is not used. The logic below handles this correctly, as long as `start_category`
           is correct (which is true as long as the index space is sufficiently large that we can
           only span two categories, as static-asserted above.) */

        if (end_index >= count - 1) {
            /* All sequence numbers within the same category */
            assert(pending_acks[end_category] >= count);
            pending_acks[end_category] -= count;
        } else {
            /* Sequence numbers span two categories */
            uint32_t count_in_end_cat = end_index + 1;
            uint32_t count_in_start_cat = count - count_in_end_cat;
            uint32_t start_category = (end_category - 1) & (num_categories - 1);
            assert(pending_acks[start_category] >= count_in_start_cat);
            assert(pending_acks[end_category] >= count_in_end_cat);
            pending_acks[start_category] -= count_in_start_cat;
            pending_acks[end_category] -= count_in_end_cat;
        }
    }
};

typedef enum {
    NVSHMEMT_LIBFABRIC_SEND,
    NVSHMEMT_LIBFABRIC_ACK,
    NVSHMEMT_LIBFABRIC_MATCH,
    NVSHMEMT_LIBFABRIC_RMA,
    NVSHMEMT_LIBFABRIC_AMO_ACK_SEND,
} nvshmemt_libfabric_recv_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK,
    NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK,
} nvshmemt_libfabric_recv_type_t;

typedef struct {
    struct fid_ep *endpoint;
    struct fid_cq *cq;
    uint64_t submitted_ops;
    uint64_t completed_ops;
    uint64_t completed_staged_atomics;
    int domain_index;
    int ep_index;
    int qp_index;
} nvshmemt_libfabric_endpoint_t;

// Entry for signal operations (put-signal, atomic)
struct nvshmemt_libfabric_signal_comp_entry {
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    int progress_count;
};

// Entry for puts that need acknowledgment
struct nvshmemt_libfabric_put_ack_entry {
    fi_addr_t src_addr;
    int ep_index;
    uint8_t put_count;
};

using nvshmemt_libfabric_comp_entry_t =
    std::variant<nvshmemt_libfabric_signal_comp_entry, nvshmemt_libfabric_put_ack_entry>;

/**
 * Per-PE flat array + overflow map for signal completion entries.
 * Fast path: O(1) direct index by (seq % window). Covers normal operation.
 * Slow path: falls back to std::unordered_map on slot collision.
 */
struct signal_seq_map {
    static constexpr size_t window = 64;

    struct slot {
        nvshmemt_libfabric_comp_entry_t entry;
        uint32_t seq;
        bool occupied;
    };

    std::array<slot, window> slots{};
    std::unordered_map<uint32_t, nvshmemt_libfabric_comp_entry_t> overflow;

    nvshmemt_libfabric_comp_entry_t *find(uint32_t seq) {
        slot &s = slots[seq % window];
        if (s.occupied && s.seq == seq) return &s.entry;
        auto it = overflow.find(seq);
        return (it != overflow.end()) ? &it->second : nullptr;
    }

    std::pair<nvshmemt_libfabric_comp_entry_t *, bool> insert(
        uint32_t seq, const nvshmemt_libfabric_comp_entry_t &e) {
        slot &s = slots[seq % window];

        if (!s.occupied) {
            s.entry = e;
            s.seq = seq;
            s.occupied = true;
            return {&s.entry, true};
        }

        if (s.seq == seq) {
            return {&s.entry, false};
        }

        auto [it, inserted] = overflow.try_emplace(seq, e);
        return {&it->second, inserted};
    }

    void erase(uint32_t seq) {
        slot &s = slots[seq % window];
        if (s.occupied && s.seq == seq) {
            s.occupied = false;
        } else {
            overflow.erase(seq);
        }
    }
};

struct nvshmemt_libfabric_signal_state_t {
    std::vector<nvshmemt_libfabric_endpoint_seq_counter_t> put_signal_seq_counter;
    std::vector<signal_seq_map> proxy_put_signal_comp_map;
    std::vector<uint32_t> next_expected_seq;

    void clear() {
        put_signal_seq_counter.clear();
        proxy_put_signal_comp_map.clear();
        next_expected_seq.clear();
    }
};

typedef struct nvshmemt_libfabric_gdr_send_p_op {
    uint64_t value;
} nvshmemt_libfabric_gdr_send_p_op_t;

typedef struct nvshmemt_libfabric_gdr_send_amo_op {
    nvshmemi_amo_t op;  /* high bit (NVSHMEMI_AMO_FLOAT_BIT) encodes float type */
    void *target_addr;
    void *ret_addr;
    union {
        uint64_t retflag;
        struct {
            uint16_t sequence_count;
            uint8_t preceding_put_count;
            uint8_t reserved;
        };
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
    NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT,
    NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT_WITH_ACK_REQ,
} nvshmemt_libfabric_imm_cq_data_hdr_t;

typedef enum {
    NVSHMEMT_LIBFABRIC_IMM_STAGED_ATOMIC_ACK,
    NVSHMEMT_LIBFABRIC_IMM_STANDALONE_PUT_ACK,
} nvshmemt_libfabric_ack_t;

class threadSafeOpQueue {
   private:
    struct conditional_mutex {
       private:
        std::mutex mtx;
        bool should_lock;

       public:
        explicit conditional_mutex(bool locking_required) : should_lock(locking_required) {}
        void lock() {
            if (should_lock) mtx.lock();
        }
        void unlock() noexcept {
            if (should_lock) mtx.unlock();
        }
    };

    conditional_mutex send_mutex;
    conditional_mutex ack_recv_mutex;
    conditional_mutex amo_recv_mutex;
    std::vector<nvshmemt_libfabric_gdr_op_ctx_t *> send;
    std::deque<nvshmemt_libfabric_gdr_op_ctx_t *> ack_recv;
    std::deque<nvshmemt_libfabric_gdr_op_ctx_t *> amo_recv;

   public:
    explicit threadSafeOpQueue(bool locking_required)
        : send_mutex(locking_required),
          ack_recv_mutex(locking_required),
          amo_recv_mutex(locking_required) {}

    threadSafeOpQueue(const threadSafeOpQueue &) = delete;
    threadSafeOpQueue &operator=(const threadSafeOpQueue &) = delete;
    threadSafeOpQueue(threadSafeOpQueue &&) = delete;
    threadSafeOpQueue &operator=(threadSafeOpQueue &&) = delete;

    int getNextSends(nvshmemt_libfabric_gdr_op_ctx_t **elems, size_t num_elems = 1) {
        const std::lock_guard<conditional_mutex> lg{send_mutex};
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
            const std::lock_guard<conditional_mutex> lg{amo_recv_mutex};
            if (amo_recv.empty()) {
                *recv_elem = NULL;
                return 0;
            }
            *recv_elem = amo_recv.front();
            nvshmemi_amo_t amo_op = static_cast<nvshmemi_amo_t>(
                static_cast<int>((*recv_elem)->send_amo.op) & ~NVSHMEMI_AMO_FLOAT_BIT);
            if (amo_op > NVSHMEMI_AMO_END_OF_NONFETCH) {
                num_sends = 2;
            } else {
                num_sends = 1;
            }
            status = getNextSends(send_elems, num_sends);
            if (status == -EAGAIN) {
                *recv_elem = NULL;
                return -EAGAIN;
            }
            assert(recv_elem != NULL);
            for (int i = 0; i < num_sends; i++) {
                assert(send_elems[i] != NULL);
            }
            amo_recv.pop_front();
            return 0;
        } else if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK) {
            const std::lock_guard<conditional_mutex> lg{ack_recv_mutex};
            if (ack_recv.empty()) {
                *recv_elem = NULL;
                return 0;
            }
            *recv_elem = ack_recv.front();
            assert(*recv_elem != NULL);
            ack_recv.pop_front();
            return 0;
        } else {
            fprintf(stderr, "getNextAmoOps: invalid recv_type: %d\n", recv_type);
            assert(false);
            return -EINVAL;
        }
    }

    void putToSend(nvshmemt_libfabric_gdr_op_ctx_t *elem) {
        const std::lock_guard<conditional_mutex> lg{send_mutex};
        send.push_back(elem);
    }

    void putToSendBulk(nvshmemt_libfabric_gdr_op_ctx_t *elem, size_t num_elems) {
        const std::lock_guard<conditional_mutex> lg{send_mutex};
        for (size_t i = 0; i < num_elems; ++i, ++elem) {
            send.push_back(elem);
        }
    }

    void putToRecv(nvshmemt_libfabric_gdr_op_ctx_t *elem,
                   nvshmemt_libfabric_recv_type_t recv_type) {
        if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_ACK) {
            const std::lock_guard<conditional_mutex> lg{ack_recv_mutex};
            ack_recv.push_back(elem);
        } else if (recv_type == NVSHMEMT_LIBFABRIC_RECV_TYPE_NOT_ACK) {
            const std::lock_guard<conditional_mutex> lg{amo_recv_mutex};
            amo_recv.push_back(elem);
        } else {
            fprintf(stderr, "putToRecv: invalid recv_type: %d\n", recv_type);
            assert(false);
        }
    }
};

struct cuda_device_deleter {
    void operator()(void *p) const noexcept {
        if (p) cudaFree(p);
    }
};
using cuda_device_ptr = std::unique_ptr<void, cuda_device_deleter>;

struct signal_delivery_work_entry {
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2];
    uint16_t sequence_count;
    uint8_t preceding_put_count;
};

struct signal_delivery_done_entry {
    nvshmemt_libfabric_gdr_op_ctx_t *op;
    nvshmemt_libfabric_gdr_op_ctx_t *send_elems[2];
    uint16_t sequence_count;
    int src_pe;
    fi_addr_t src_addr;
    int ep_index;
    bool is_fetch_amo;
    uint64_t old_value;
    uint64_t ret_flags;
    void *ret_addr;
    uint8_t preceding_put_count;
};

template <typename T, size_t Capacity = nvshmemt_libfabric_signal_queue_capacity>
class SPSCRing {
    std::array<T, Capacity> ring{};
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};

   public:
    bool push(const T &entry) {
        size_t h = head.load(std::memory_order_relaxed);
        size_t next = (h + 1) % Capacity;
        if (next == tail.load(std::memory_order_acquire)) return false;
        ring[h] = entry;
        head.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T &entry) {
        size_t t = tail.load(std::memory_order_relaxed);
        if (t == head.load(std::memory_order_acquire)) return false;
        entry = ring[t];
        tail.store((t + 1) % Capacity, std::memory_order_release);
        return true;
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
struct nvshmemt_libfabric_state_t {
    /* Copy and move are implicitly deleted by std::mutex/std::unique_ptr members. */

    struct fi_info *all_prov_info = nullptr;
    std::vector<struct fi_info *> prov_infos;
    std::vector<struct fid_fabric *> fabrics;
    std::vector<struct fid_domain *> domains;
    std::vector<struct fid_av *> addresses;
    std::vector<std::unique_ptr<nvshmemt_libfabric_endpoint_t>> eps;

    /* local_mr is used only for consistency ops. */
    std::vector<struct fid_mr *> local_mrs;
    std::vector<uint64_t> local_mr_keys;
    std::vector<void *> local_mr_descs;
    void *local_mem_ptr = nullptr;

    std::vector<nvshmemt_libfabric_domain_name_t> domain_names;
    nvshmemt_libfabric_provider provider{};
    int log_level = 0;
    struct nvshmemi_cuda_fn_table *table = nullptr;
    struct transport_mem_handle_info_cache *cache = nullptr;

    /* Required for multi-rail */
    int num_host_domains = 0;
    int num_proxy_domains = 0;
    int num_selected_devs = 0;
    int max_nic_per_pe = 0;
    uint32_t proxy_ep_cntr = 0;
    /* EP chosen for the FI_MORE-deferred op so batched ops use the same EP.
     * Slot 0 = NVSHMEMX_QP_HOST, slot 1 = proxy. Each slot is SPSC.
     * -1 = no pending batched ops. */
    std::array<int, 2> pending_batch_ep;

    /* Required for staged_amo */
    std::vector<std::unique_ptr<threadSafeOpQueue>> op_queue;
    std::vector<std::vector<nvshmemt_libfabric_gdr_op_ctx_t>> recv_buf;
    std::vector<struct fid_mr *> mrs;

    /* Signal ordering state */
    nvshmemt_libfabric_signal_state_t host_signal_state;
    nvshmemt_libfabric_signal_state_t proxy_signal_state;

    /* Max ops per progress iteration */
    int proxy_request_batch_max = 0;

    /* Signal delivery thread. */
    pthread_t signal_delivery_thread{};
    std::atomic<int> signal_delivery_stop{0};
    std::atomic<int> signal_delivery_status{0};
    nvshmem_transport_t signal_delivery_transport = nullptr;
    std::atomic<int> signal_delivery_futex{0};
    std::atomic_flag signal_progress_lock = ATOMIC_FLAG_INIT;
    /* signal_work_queue is SPSC (single consumer: delivery thread), but two
     * threads can push (put_signal_completion and gdr_process_amos).
     * Push-side serialization is provided by signal_work_queue_lock. */
    std::atomic_flag signal_work_queue_lock = ATOMIC_FLAG_INIT;
    SPSCRing<signal_delivery_done_entry> signal_done_queue;
    SPSCRing<signal_delivery_work_entry> signal_work_queue;

    /* Misc state management */
    bool use_staged_atomics = false;
    bool use_auto_progress = false;
};

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
 * | 4 type | 1 op | 1 elem_size | 2 num_writes | 8 signal | 8 target_addr | 2 sequence_count
 * | 1 preceding_put_count | 1 reserved | 4 src_pe
 */
typedef struct nvshmemt_libfabric_gdr_signal_op {
    nvshmemt_libfabric_recv_t type; /* Must be first */
    uint8_t op;
    uint8_t elem_size;
    uint16_t num_writes;
    uint64_t sig_val;
    void *target_addr;
    uint16_t sequence_count;
    uint8_t  preceding_put_count;
    uint8_t  reserved;
    uint32_t src_pe;
} nvshmemt_libfabric_gdr_signal_op_t;
/*  EFA's inline send size is 32 bytes */
static_assert(sizeof(nvshmemt_libfabric_gdr_signal_op_t) == 32);
/* This type is nested in nvshmemt_libfabric_gdr_op_ctx_t, so make sure it fits */
static_assert(sizeof(nvshmemt_libfabric_gdr_signal_op_t) <=
              offsetof(nvshmemt_libfabric_gdr_op_ctx_t, ofi_context),
              "Must fit within nvshmemt_libfabric_gdr_op_ctx_t");

/* Wire data for AMO ack sent via fi_send
 * | 4 type | 4 ack_type | 2 sequence_count | 1 put_count
 */
typedef struct nvshmemt_libfabric_gdr_amo_ack_op {
    nvshmemt_libfabric_recv_t type; /* Must be first */
    nvshmemt_libfabric_ack_t ack_type;
    uint16_t sequence_count;
    uint8_t put_count;
} nvshmemt_libfabric_gdr_amo_ack_op_t;
static_assert(sizeof(nvshmemt_libfabric_gdr_amo_ack_op_t) <= 32,
              "Must fit within EFA's inline send limit of 32 bytes");
/* This type is nested in nvshmemt_libfabric_gdr_op_ctx_t, so make sure it fits */
static_assert(sizeof(nvshmemt_libfabric_gdr_amo_ack_op_t) <=
              offsetof(nvshmemt_libfabric_gdr_op_ctx_t, ofi_context),
              "Must fit within nvshmemt_libfabric_gdr_op_ctx_t");
