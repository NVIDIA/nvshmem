/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nccl.h>

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 7)
#include <nccl_device.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>

#define THREADS_PER_BLOCK 128
#define MSG_WORDS 128

struct buffer_layout {
    size_t nvshmem;
    size_t nccl;
    size_t signals;
    size_t total;
};

#define NCCL_CALL(stmt)                                                      \
    do {                                                                     \
        ncclResult_t nccl_status = (stmt);                                   \
        if (nccl_status != ncclSuccess) {                                    \
            fprintf(stderr, "[%s:%d] NCCL failed: %s\n", __FILE__, __LINE__, \
                    ncclGetErrorString(nccl_status));                        \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define NVSHMEM_CALL(stmt)                                                                 \
    do {                                                                                   \
        int nvshmem_status = (stmt);                                                       \
        if (nvshmem_status != 0) {                                                         \
            fprintf(stderr, "[%s:%d] NVSHMEM failed with status %d\n", __FILE__, __LINE__, \
                    nvshmem_status);                                                       \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#if !defined(CUDA_CHECK)
#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t cuda_status = (stmt);                                    \
        if (cuda_status != cudaSuccess) {                                    \
            fprintf(stderr, "[%s:%d] CUDA failed: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(cuda_status));                        \
            exit(1);                                                         \
        }                                                                    \
    } while (0)
#endif

static void init_nvshmem_and_select_device() {
    nvshmem_init();

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int dev_count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(mype_node / npes_per_gpu));
    nvshmem_barrier_all();
}

static void log_status(int mype, int npes, const char *fmt, ...) {
    printf("[interop_nccl pid=%ld pe=%d/%d] ", (long)getpid(), mype, npes - 1);

    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    printf("\n");
    fflush(stdout);
}

static unsigned char make_payload_byte(int pe) { return (unsigned char)(((uint32_t)pe % 255) + 1); }

static uint64_t make_payload(int pe) {
    uint64_t byte = (uint64_t)make_payload_byte(pe);
    return byte * 0x0101010101010101ULL;
}

static __host__ __device__ size_t word_offset_to_bytes(size_t word_offset) {
    return word_offset * sizeof(uint64_t);
}

static __host__ __device__ int next_peer_for_pe(int pe, int npes) { return (pe + 1) % npes; }

static __host__ __device__ int prev_peer_for_pe(int pe, int npes) { return (pe + npes - 1) % npes; }

static __host__ __device__ size_t pe_slot(int pe, size_t msg_words) {
    return (size_t)pe * msg_words;
}

static __host__ __device__ size_t signal_slot(int pe) { return (size_t)pe; }

static buffer_layout make_layout(int npes, size_t msg_words) {
    buffer_layout layout = {};
    size_t payload_words = (size_t)npes * msg_words;

    layout.nvshmem = 0;
    layout.nccl = payload_words;
    layout.signals = 2 * payload_words;
    layout.total = layout.signals + (size_t)npes;

    return layout;
}

__global__ void nccl_nvshmem_kernel(uint64_t *buffer, buffer_layout layout, size_t msg_words,
                                    int mype, int npes, int peer, int prev, bool peer_is_local,
                                    bool prev_is_local, bool use_gin, ncclDevComm_t dev_comm,
                                    ncclWindow_t window) {
    ncclCoopCta coop;
    ncclTeam world_team = ncclTeamWorld(dev_comm);

    size_t slot = pe_slot(mype, msg_words);
    uint64_t *nvshmem_src = buffer + layout.nvshmem + slot;
    uint64_t *nccl_src = buffer + layout.nccl + slot;
    size_t nccl_offset = word_offset_to_bytes(layout.nccl + slot);

    if (peer_is_local) {
        uint64_t *nccl_local_dst = (uint64_t *)ncclGetPeerPointer(window, nccl_offset, peer);
        auto local_dst = [=] __device__(int) -> uint64_t * { return nccl_local_dst; };

        ncclLocalCopy(coop, nccl_src, local_dst, 1, msg_words);
    } else {
        size_t signal_offset = word_offset_to_bytes(layout.signals + signal_slot(mype));
        ncclGin gin(dev_comm, 0);

        gin.put(world_team, peer, window, nccl_offset, window, nccl_offset,
                msg_words * sizeof(uint64_t), ncclGin_StrongVASignalAdd{window, signal_offset, 1},
                ncclGin_None{}, coop);
        gin.flush(coop);
    }

    nvshmemx_putmem_nbi_block(nvshmem_src, nvshmem_src, msg_words * sizeof(uint64_t), peer);
    nvshmem_quiet();

    if (!prev_is_local) {
        size_t wait_signal_offset = word_offset_to_bytes(layout.signals + signal_slot(prev));
        ncclGin gin(dev_comm, 0);

        gin.waitSignal(coop, window, wait_signal_offset, 1);
    }
}

static void init_buffer(uint64_t *buffer, buffer_layout layout, int mype, size_t msg_words) {
    unsigned char payload_byte = make_payload_byte(mype);
    size_t bytes = msg_words * sizeof(uint64_t);

    CUDA_CHECK(cudaMemset(buffer, 0, layout.total * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(buffer + layout.nvshmem + pe_slot(mype, msg_words), payload_byte, bytes));
    CUDA_CHECK(cudaMemset(buffer + layout.nccl + pe_slot(mype, msg_words), payload_byte, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void launch_interop(uint64_t *buffer, buffer_layout layout, size_t msg_words, int mype,
                           int npes, int peer, int prev, bool peer_is_local, bool prev_is_local,
                           bool use_gin, ncclDevComm_t dev_comm, ncclWindow_t window) {
    nccl_nvshmem_kernel<<<1, THREADS_PER_BLOCK>>>(buffer, layout, msg_words, mype, npes, peer, prev,
                                                  peer_is_local, prev_is_local, use_gin, dev_comm,
                                                  window);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static int verify_slot(const uint64_t *buffer, size_t base, size_t msg_words, int source_pe,
                       const char *label, int mype) {
    size_t slot = base + pe_slot(source_pe, msg_words);

    for (size_t i = 0; i < msg_words; i++) {
        uint64_t expected = make_payload(source_pe);
        uint64_t found = buffer[slot + i];
        if (found != expected) {
            fprintf(stderr,
                    "PE %d %s mismatch from PE %d at word %zu: found 0x%016lx expected "
                    "0x%016lx\n",
                    mype, label, source_pe, i, (unsigned long)found, (unsigned long)expected);
            return -1;
        }
    }

    return 0;
}

static int verify_buffer(const uint64_t *buffer, buffer_layout layout, size_t msg_words, int mype,
                         int prev) {
    int status = 0;

    status |= verify_slot(buffer, layout.nvshmem, msg_words, mype, "NVSHMEM self", mype);
    status |= verify_slot(buffer, layout.nvshmem, msg_words, prev, "NVSHMEM ring", mype);
    status |= verify_slot(buffer, layout.nccl, msg_words, mype, "NCCL self", mype);
    status |= verify_slot(buffer, layout.nccl, msg_words, prev, "NCCL ring", mype);

    return status;
}

static int allreduce_int_max(int value) {
    int result = 0;
    int *tmp = (int *)nvshmem_malloc(2 * sizeof(int));
    if (tmp == nullptr) {
        fprintf(stderr, "nvshmem_malloc failed for allreduce scratch\n");
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(tmp, &value, sizeof(int), cudaMemcpyDefault));
    NVSHMEM_CALL(nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, tmp + 1, tmp, 1));
    CUDA_CHECK(cudaMemcpy(&result, tmp + 1, sizeof(int), cudaMemcpyDefault));
    nvshmem_free(tmp);

    return result;
}

int main() {
    int status = 0;
    int mype = 0;
    int npes = 0;
    ncclUniqueId unique_id = {};
    ncclUniqueId *scratch = nullptr;
    ncclComm_t nccl_comm = nullptr;
    ncclWindow_t window = nullptr;
    ncclDevComm_t dev_comm = {};
    bool dev_comm_created = false;
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    ncclTeam_t world_team = {};
    ncclTeam_t lsa_team = {};
    buffer_layout layout = {};
    size_t buffer_bytes = 0;
    bool use_gin = false;
    bool peer_is_local = false;
    bool prev_is_local = false;
    bool gin_needed = false;
    int peer = 0;
    int prev = 0;
    bool ran_test = false;
    uint64_t *buffer = nullptr;
    uint64_t *host_buffer = nullptr;

    init_nvshmem_and_select_device();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    peer = next_peer_for_pe(mype, npes);
    prev = prev_peer_for_pe(mype, npes);
    log_status(mype, npes, "NVSHMEM initialized: peer=%d prev=%d", peer, prev);

    scratch = (ncclUniqueId *)nvshmem_malloc(sizeof(ncclUniqueId));
    if (scratch == nullptr) {
        fprintf(stderr, "nvshmem_malloc failed for NCCL unique ID scratch\n");
        status = -1;
        goto out;
    }

    if (mype == 0) {
        log_status(mype, npes, "creating NCCL unique ID");
        NCCL_CALL(ncclGetUniqueId(&unique_id));
        CUDA_CHECK(cudaMemcpy(scratch, &unique_id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    NVSHMEM_CALL(
        nvshmem_broadcastmem(NVSHMEM_TEAM_WORLD, scratch, scratch, sizeof(ncclUniqueId), 0));
    CUDA_CHECK(cudaMemcpy(&unique_id, scratch, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    log_status(mype, npes, "received NCCL unique ID");

    log_status(mype, npes, "creating NCCL communicator");
    NCCL_CALL(ncclCommInitRank(&nccl_comm, npes, unique_id, mype));
    log_status(mype, npes, "created NCCL communicator");

    NCCL_CALL(ncclCommQueryProperties(nccl_comm, &props));
    log_status(mype, npes, "NCCL properties: deviceApiSupport=%d ginType=%d",
               props.deviceApiSupport, (int)props.ginType);
    if (!props.deviceApiSupport) {
        if (mype == 0) {
            log_status(mype, npes, "NCCL runtime reports no device API support. Skipping.");
        }
        status = -1;
        goto out;
    }
    use_gin = props.ginType != NCCL_GIN_TYPE_NONE;

    lsa_team = ncclTeamLsa(nccl_comm);
    world_team = ncclTeamWorld(nccl_comm);
    log_status(mype, npes, "NCCL teams: world_rank=%d world_nRanks=%d lsa_rank=%d lsa_nRanks=%d",
               world_team.rank, world_team.nRanks, lsa_team.rank, lsa_team.nRanks);
    if (world_team.nRanks != npes) {
        if (mype == 0) {
            log_status(mype, npes, "NCCL world team does not match NVSHMEM PE count. Skipping.");
        }
        status = -1;
        goto out;
    }

    peer_is_local = ncclTeamRankIsMember(lsa_team, world_team, peer);
    prev_is_local = ncclTeamRankIsMember(lsa_team, world_team, prev);
    gin_needed = allreduce_int_max((!peer_is_local || !prev_is_local) ? 1 : 0) != 0;
    log_status(mype, npes,
               "selected ring path: peer=%d peer_is_local=%d prev=%d prev_is_local=%d "
               "gin_available=%d gin_needed=%d",
               peer, peer_is_local, prev, prev_is_local, use_gin, gin_needed);
    if (allreduce_int_max(((peer_is_local || use_gin) && (prev_is_local || use_gin)) ? 0 : 1)) {
        if (mype == 0) {
            log_status(mype, npes,
                       "Need either LSA locality or NCCL GIN for every ring edge. Skipping.");
        }
        status = -1;
        goto out;
    }

    layout = make_layout(npes, MSG_WORDS);
    buffer_bytes = layout.total * sizeof(uint64_t);
    buffer = (uint64_t *)nvshmem_align(NCCL_WIN_REQUIRED_ALIGNMENT, buffer_bytes);
    if (buffer == nullptr) {
        fprintf(stderr, "nvshmem_align failed for %zu bytes\n", buffer_bytes);
        status = -1;
        goto out;
    }
    log_status(mype, npes, "allocated symmetric buffer: ptr=%p bytes=%zu", (void *)buffer,
               buffer_bytes);

    host_buffer = (uint64_t *)calloc(layout.total, sizeof(uint64_t));
    if (host_buffer == nullptr) {
        fprintf(stderr, "calloc failed for host verification buffer\n");
        status = -1;
        goto out;
    }
    init_buffer(buffer, layout, mype, MSG_WORDS);
    nvshmem_barrier_all();
    log_status(mype, npes, "initialized test buffer");

    NCCL_CALL(
        ncclCommWindowRegister(nccl_comm, buffer, buffer_bytes, &window, NCCL_WIN_COLL_SYMMETRIC));
    log_status(mype, npes, "registered NCCL window: window=%p bytes=%zu", (void *)window,
               buffer_bytes);
    if (gin_needed) {
        reqs.ginForceEnable = true;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        reqs.ginStrongSignalsRequired = true;
        reqs.ginVaSignalsRequired = true;
    }
    log_status(mype, npes, "creating NCCL device communicator: gin_required=%d", gin_needed);
    NCCL_CALL(ncclDevCommCreate(nccl_comm, &reqs, &dev_comm));
    dev_comm_created = true;
    use_gin = use_gin && dev_comm.ginConnectionCount > 0;
    log_status(mype, npes, "created NCCL device communicator: ginConnectionCount=%d use_gin=%d",
               dev_comm.ginConnectionCount, use_gin);
    if (gin_needed && allreduce_int_max(use_gin ? 0 : 1)) {
        if (mype == 0) {
            log_status(mype, npes, "NCCL GIN could not be enabled for at least one PE. Skipping.");
        }
        status = -1;
        goto out;
    }

    log_status(mype, npes, "launching interop kernel");
    if (world_team.nRanks != npes ||
        ((!peer_is_local || !prev_is_local) && (!use_gin || dev_comm.ginConnectionCount == 0))) {
        log_status(mype, npes, "Invalid setup for the test");
        status = -1;
        goto out;
    }
    launch_interop(buffer, layout, MSG_WORDS, mype, npes, peer, prev, peer_is_local, prev_is_local,
                   use_gin, dev_comm, window);
    ran_test = true;
    log_status(mype, npes, "interop kernel completed");
    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(host_buffer, buffer, buffer_bytes, cudaMemcpyDeviceToHost));
    status |= verify_buffer(host_buffer, layout, MSG_WORDS, mype, prev);
    log_status(mype, npes, "verification %s", status == 0 ? "passed" : "failed");

out:
    if (dev_comm_created) {
        log_status(mype, npes, "destroying NCCL device communicator");
        NCCL_CALL(ncclDevCommDestroy(nccl_comm, &dev_comm));
    }
    if (window != nullptr) {
        log_status(mype, npes, "deregistering NCCL window");
        NCCL_CALL(ncclCommWindowDeregister(nccl_comm, window));
    }
    if (nccl_comm != nullptr) {
        log_status(mype, npes, "destroying NCCL communicator");
        NCCL_CALL(ncclCommDestroy(nccl_comm));
    }
    if (host_buffer != nullptr) free(host_buffer);
    if (buffer != nullptr) nvshmem_free(buffer);
    if (scratch != nullptr) nvshmem_free(scratch);
    if (status == 0 && ran_test && mype == 0) {
        log_status(mype, npes, "Interop test passed");
    }
    nvshmem_finalize();
    return status;
}

#else

#include <cstdio>

int main() {
    std::printf("Skipping interop_nccl: requires NCCL >= 2.30.7 Device API/GIN headers\n");
    return 0;
}

#endif
