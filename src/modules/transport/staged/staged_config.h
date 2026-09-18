/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_STAGED_CONFIG_H
#define NVSHMEM_STAGED_CONFIG_H

#include <stddef.h>
#include <stdint.h>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/host_transport/transport.h"
#include "staged_control.h"

#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

namespace nvshmemi {

/* Staged transport resource limits. */
inline constexpr int STAGED_MAX_PIPELINE_DEPTH = 16;
/* Pending API operations, independent of the active PUT's wire pipeline depth. */
inline constexpr size_t STAGED_OPERATION_CAPACITY = 128;
inline constexpr int STAGED_CQ_DEPTH = 256;
inline constexpr int STAGED_SQ_DEPTH = 128;
inline constexpr int STAGED_RQ_DEPTH = 128;
inline constexpr int STAGED_GRH_HOP_LIMIT = 255;
inline constexpr int STAGED_MAX_HCA_LIST = 16;
inline constexpr size_t STAGED_MAX_BOUNCE_BYTES = UINT32_MAX;

enum class staged_copy_policy_t { STREAM, GDRCOPY };

/* ── message types for control channel (Send/Recv) ─────────────── */
enum class staged_ctrl_op_t : uint32_t {
    CHUNK_DONE = 2, /* a chunk has been RDMA-written to bounce */
    GET_REQ = 3,    /* GET request */
    RESPONSE = 4,   /* operation response */
    AMO = 5,        /* AMO request */
};

inline constexpr uint32_t STAGED_CTRL_AMO_FETCH = 1u << 31;
inline constexpr uint32_t STAGED_CTRL_AMO_OP_MASK = 0xffffu;

struct staged_ctrl_msg_t {
    staged_ctrl_op_t op;
    uint32_t flags;      /* for AMO: verb | float | is_fetch */
    uint64_t addr;       /* target virtual address */
    uint64_t bytes;      /* data length */
    uint64_t value;      /* AMO value */
    uint64_t cmp;        /* AMO compare value */
    uint64_t request_id; /* request/response matching */
    uint64_t reply_addr; /* requester bounce address for GET responses */
    uint32_t reply_rkey; /* requester bounce rkey for GET responses */
    int32_t status;      /* response status */
};
struct staged_response_t {
    uint64_t value;
    int status;
};

struct staged_rma_op_t {
    rma_verb_t verb;
    rma_memdesc_t remote;
    rma_memdesc_t local;
    rma_bytesdesc_t bytesdesc;
};

struct staged_amo_op_t {
    amo_verb_t verb;
    amo_memdesc_t target;
    amo_bytesdesc_t bytesdesc;
};

struct staged_client_op_t {
    staged_client_op_t(int pe, staged_rma_op_t operation)
        : pe(pe), operation(std::move(operation)) {}
    staged_client_op_t(int pe, staged_amo_op_t operation)
        : pe(pe), operation(std::move(operation)) {}

    int pe;
    std::variant<staged_rma_op_t, staged_amo_op_t> operation;
};

/* ── endpoint handle exchanged through bootstrap ────────────────── */
struct staged_ep_handle_t {
    uint32_t qpn;         /* QP number */
    uint32_t rkey;        /* rkey for bounce buffer MR */
    uint64_t bounce_addr; /* remote bounce buffer address */
    union ibv_gid gid;    /* local GID for GRH and native-IB path selection */
    uint16_t lid;
};

/* Owns one RC QP and every resource whose lifetime is tied to it. */
class staged_qp_t {
   public:
    staged_qp_t(struct ibv_context* context, struct ibv_pd* pd, void* server_bounce);
    ~staged_qp_t();

    staged_qp_t(const staged_qp_t&) = delete;
    staged_qp_t& operator=(const staged_qp_t&) = delete;
    staged_qp_t(staged_qp_t&&) = delete;
    staged_qp_t& operator=(staged_qp_t&&) = delete;

    int initialize(int ib_port, uint16_t pkey_index, size_t index);
    int post_receive(size_t index);

    struct ibv_qp* qp() const { return qp_; }
    struct ibv_cq* recv_cq() const { return recv_cq_; }
    void* server_bounce() const { return server_bounce_; }
    staged_ctrl_msg_t& ctrl_message(size_t index) { return ctrl_recv_bufs_.at(index); }
    staged_ep_handle_t& remote_endpoint() { return remote_ep_; }
    const staged_ep_handle_t& remote_endpoint() const { return remote_ep_; }

    template <typename Operation>
    decltype(auto) serialize_send(Operation&& operation) {
        std::lock_guard<std::mutex> lock(send_mutex_);
        return std::forward<Operation>(operation)(qp_, send_cq_);
    }

   private:
    struct ibv_context* context_;
    struct ibv_pd* pd_;
    struct ibv_qp* qp_ = nullptr;
    struct ibv_cq* send_cq_ = nullptr;
    struct ibv_cq* recv_cq_ = nullptr;
    void* server_bounce_;
    std::array<staged_ctrl_msg_t, STAGED_RQ_DEPTH> ctrl_recv_bufs_{};
    struct ibv_mr* ctrl_recv_mr_ = nullptr;
    staged_ep_handle_t remote_ep_{};
    std::mutex send_mutex_;
};

/* Local cache entry for an installed GPU CPU mapping. */
struct staged_mem_handle_info_t {
    void* ptr = nullptr; /* registered GPU VA used as the lookup key */
    size_t size = 0;
#ifdef NVSHMEM_USE_GDRCOPY
    nvshmemt_gpu_cpu_mapping cpu_mapping{};
#endif
};

/* Owns verbs resources separately from the internal QP collection. */
class staged_rdma_state_t {
   public:
    ~staged_rdma_state_t();

    staged_rdma_state_t() = default;
    staged_rdma_state_t(const staged_rdma_state_t&) = delete;
    staged_rdma_state_t& operator=(const staged_rdma_state_t&) = delete;
    staged_rdma_state_t(staged_rdma_state_t&&) = delete;
    staged_rdma_state_t& operator=(staged_rdma_state_t&&) = delete;

    struct ibv_context* context() const { return ib_.context; }
    void set_context(struct ibv_context* context) { ib_.context = context; }
    struct ibv_pd* protection_domain() const { return ib_.protection_domain; }
    void set_protection_domain(struct ibv_pd* protection_domain) {
        ib_.protection_domain = protection_domain;
    }
    struct ibv_mr* bounce_mr() const { return ib_.bounce_mr; }
    void* client_bounce() const { return ib_.client_bounce; }
    void* bounce_region() const { return ib_.bounce_region; }
    size_t bounce_region_size() const { return ib_.bounce_region_size; }
    void set_bounce_storage(void* region, size_t size, void* client_bounce,
                            struct ibv_mr* memory_region) {
        ib_.bounce_region = region;
        ib_.bounce_region_size = size;
        ib_.client_bounce = client_bounce;
        ib_.bounce_mr = memory_region;
    }

    int ib_port() const { return ib_.port; }
    void set_ib_port(int port) { ib_.port = port; }
    uint8_t local_gid_index() const { return ib_.local_gid_index; }
    void set_local_gid_index(uint8_t index) { ib_.local_gid_index = index; }
    uint16_t local_lid() const { return ib_.local_lid; }
    void set_local_lid(uint16_t lid) { ib_.local_lid = lid; }
    enum ibv_mtu path_mtu() const { return ib_.path_mtu; }
    void set_path_mtu(enum ibv_mtu mtu) { ib_.path_mtu = mtu; }
    const union ibv_gid& local_gid() const { return ib_.local_gid; }
    void set_local_gid(const union ibv_gid& gid) { ib_.local_gid = gid; }

    void configure_qps(int num_pes, int qps_per_pe) {
        qps_.num_pes = num_pes;
        qps_.qps_per_pe = qps_per_pe;
    }
    int num_pes() const { return qps_.num_pes; }
    int qps_per_pe() const { return qps_.qps_per_pe; }
    bool qps_empty() const { return qps_.qps.empty(); }
    void reserve_qps(size_t count) { qps_.qps.reserve(count); }
    void add_qp(std::unique_ptr<staged_qp_t> qp) { qps_.qps.emplace_back(std::move(qp)); }
    staged_qp_t& qp_at(size_t index) { return *qps_.qps.at(index); }

    size_t total_qps() const {
        return static_cast<size_t>(qps_.qps_per_pe) * static_cast<size_t>(qps_.num_pes);
    }

    staged_qp_t& qp(int qp_slot, int pe) {
        return qp_at(static_cast<size_t>(qp_slot) * static_cast<size_t>(qps_.num_pes) + pe);
    }

   private:
    struct ib_resources_t {
        struct ibv_context* context = nullptr;
        struct ibv_pd* protection_domain = nullptr;
        struct ibv_mr* bounce_mr = nullptr;
        int port = 0;
        uint8_t local_gid_index = 0;
        uint16_t local_lid = 0;
        enum ibv_mtu path_mtu = IBV_MTU_4096;
        union ibv_gid local_gid{};
        void* client_bounce = nullptr;
        void* bounce_region = nullptr;
        size_t bounce_region_size = 0;
    };

    struct qp_resources_t {
        int num_pes = 0;
        int qps_per_pe = 0;
        /* [qps_per_pe][num_pes], QP-major. */
        std::vector<std::unique_ptr<staged_qp_t>> qps;
    };

    ib_resources_t ib_;
    qp_resources_t qps_;
};

/* CUDA resources used by the local staging path. */
struct staged_cuda_state_t {
    int device = -1;
    staged_copy_policy_t copy_policy = staged_copy_policy_t::STREAM;
    CUgreenCtx green_ctx = nullptr;
    cudaStream_t client_stream = nullptr;
    cudaStream_t server_stream = nullptr;
    std::array<cudaEvent_t, STAGED_MAX_PIPELINE_DEPTH> client_events{};
    std::mutex copy_mutex;

    ~staged_cuda_state_t();

    staged_cuda_state_t() = default;
    staged_cuda_state_t(const staged_cuda_state_t&) = delete;
    staged_cuda_state_t& operator=(const staged_cuda_state_t&) = delete;
    staged_cuda_state_t(staged_cuda_state_t&&) = delete;
    staged_cuda_state_t& operator=(staged_cuda_state_t&&) = delete;
};

using staged_operation_state_t =
    staged_operation_queue_t<staged_client_op_t, STAGED_OPERATION_CAPACITY>;
using staged_response_state_t =
    staged_response_table_t<staged_response_t, STAGED_MAX_PIPELINE_DEPTH>;

/* Coordinates one worker group's shutdown and failure propagation. */
class staged_worker_state_t {
   public:
    staged_worker_state_t(staged_operation_state_t& operations, staged_response_state_t& responses);

    staged_worker_state_t(const staged_worker_state_t&) = delete;
    staged_worker_state_t& operator=(const staged_worker_state_t&) = delete;
    staged_worker_state_t(staged_worker_state_t&&) = delete;
    staged_worker_state_t& operator=(staged_worker_state_t&&) = delete;

    void fail(int status);
    void request_stop();
    bool stopping() const { return stop_requested_.load(std::memory_order_relaxed); }

   private:
    staged_operation_state_t& operations_;
    staged_response_state_t& responses_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<int> first_error_{0};
};

/* Owns one worker thread and guarantees stop-before-join on every destruction path. */
class staged_worker_thread_t {
   public:
    template <typename Function>
    staged_worker_thread_t(staged_worker_state_t& state, Function&& function)
        : state_(state), thread_(std::forward<Function>(function)) {}

    ~staged_worker_thread_t();

    staged_worker_thread_t(const staged_worker_thread_t&) = delete;
    staged_worker_thread_t& operator=(const staged_worker_thread_t&) = delete;
    staged_worker_thread_t(staged_worker_thread_t&&) = delete;
    staged_worker_thread_t& operator=(staged_worker_thread_t&&) = delete;

   private:
    friend class staged_worker_threads_t;

    void join();

    staged_worker_state_t& state_;
    std::thread thread_;
};

/* Starts both workers on construction; member RAII stops and joins them on destruction. */
class staged_worker_threads_t {
   public:
    template <typename ServerFunction, typename ClientFunction>
    staged_worker_threads_t(staged_worker_state_t& state, ServerFunction&& server_function,
                            ClientFunction&& client_function)
        : state_(state),
          server_thread_(state_, [this, function = std::forward<ServerFunction>(
                                            server_function)]() mutable { function(startup_); }),
          client_thread_(state_, [this, function = std::forward<ClientFunction>(
                                            client_function)]() mutable { function(startup_); }) {}

    ~staged_worker_threads_t();
    int wait_for_start() { return startup_.wait(); }

    staged_worker_threads_t(const staged_worker_threads_t&) = delete;
    staged_worker_threads_t& operator=(const staged_worker_threads_t&) = delete;
    staged_worker_threads_t(staged_worker_threads_t&&) = delete;
    staged_worker_threads_t& operator=(staged_worker_threads_t&&) = delete;

   private:
    staged_worker_state_t& state_;
    /* Constructed before, and destroyed after, both worker owners. */
    staged_startup_latch_t startup_{2};
    staged_worker_thread_t server_thread_;
    staged_worker_thread_t client_thread_;
};

/* Registered GPU mappings and memory-handle lifetime tracking. */
struct staged_memory_state_t {
    std::mutex amo_mutex;
    std::mutex gpu_cpu_mapping_mutex;
    std::shared_mutex mem_handle_mutex;
    std::unordered_set<staged_mem_handle_info_t*> mem_handle_infos;
    std::unordered_set<staged_mem_handle_info_t*> retired_mem_handle_infos;
#ifdef NVSHMEM_USE_GDRCOPY
    nvshmemt_gpu_cpu_mapping_state gpu_cpu_mapping_state{};
#endif
};

/* ── transport state ───────────────────────────────────────────── */
struct transport_staged_state_t {
    size_t bounce_bytes = 0;
    int pipeline_depth = 0;
    struct nvshmemi_options_s options{};
    int log_level = 0;
    staged_rdma_state_t rdma;
    staged_cuda_state_t cuda;
    staged_operation_state_t operations;
    staged_response_state_t responses;
    staged_worker_state_t workers;
    staged_memory_state_t memory;
    std::unique_ptr<staged_worker_threads_t> worker_threads;

    transport_staged_state_t();

    transport_staged_state_t(const transport_staged_state_t&) = delete;
    transport_staged_state_t& operator=(const transport_staged_state_t&) = delete;
};

}  // namespace nvshmemi

#endif /* NVSHMEM_STAGED_CONFIG_H */
