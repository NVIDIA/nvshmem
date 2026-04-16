/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cassert>
#include <vector>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstdarg>
#include <array>
#include <memory>
#include "nccl.h"   // for ncclDataType_t, ncclResult_t
#include <dlfcn.h>  // for dlsym, dlopen, RTLD...

#define NCCL_CHECK(cmd)                                                   \
    do {                                                                  \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess) {                                           \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   nccl_ftable.get_error_string(r));                      \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

#define NVSHMEM_CHECK(stmt)                                                  \
    do {                                                                     \
        int result = (stmt);                                                 \
        if (0 != result) {                                                   \
            fprintf(stderr, "[%s:%d] NVSHMEM failed\n", __FILE__, __LINE__); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

#define CHECK(statement)                                                  \
    do {                                                                  \
        if ((statement)) {                                                \
            fprintf(stderr, "[%s:%d] Test failed\n", __FILE__, __LINE__); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

#define INFO(...) printf(__VA_ARGS__)

#define ALLOC_DEFAULT (1024 * 1024)

struct ftable {
    ncclResult_t (*get_version)(int *version);
    const char *(*get_error_string)(ncclResult_t result);
    ncclResult_t (*get_unique_id)(ncclUniqueId *uniqueId);
    ncclResult_t (*comm_init_rank)(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t (*comm_destroy)(ncclComm_t comm);
    ncclResult_t (*comm_register)(ncclComm_t comm, void *buff, size_t size, void **handle);
    ncclResult_t (*comm_deregister)(ncclComm_t comm, void *comm_handle);
    ncclResult_t (*allgather)(const void *sendbuf, void *recvbuff, size_t sendcount,
                              ncclDataType_t dtype, ncclComm_t comm, cudaStream_t stream);
};

struct ftable nccl_ftable = {};

int nccl_ftable_init(struct ftable *nccl_ftable, void **nccl_handle) {
    int nccl_major, nccl_minor;
    int nccl_build_major;
    int nccl_build_version, nccl_version;

    *nccl_handle = NULL;
    *nccl_handle = dlopen("libnccl.so.2", RTLD_LAZY);
    if (!*nccl_handle) {
        INFO("NCCL library not found in LD_LIBRARY_PATH. Please set up path correctly...\n");
        goto fn_out;
    }

    nccl_build_version = NCCL_VERSION_CODE;
    LOAD_SYM(*nccl_handle, "ncclGetVersion", nccl_ftable->get_version);
    nccl_ftable->get_version(&nccl_version);
    if (nccl_version > 10000) {
        nccl_major = nccl_version / 10000;
    } else {
        nccl_major = nccl_version / 1000;
    }

    nccl_minor = nccl_version / 100 - nccl_major * 100;
    if (nccl_build_version > 10000) {
        nccl_build_major = nccl_build_version / 10000;
    } else {
        nccl_build_major = nccl_build_version / 1000;
    }

    if (nccl_major != nccl_build_major) {
        INFO(
            "NCCL library major version (%d) is different than the"
            " version (%d) with which NVSHMEM was built, skipping use...\n",
            nccl_major, nccl_build_major);
        goto fn_out;
    }

    if (nccl_minor <= 18) {
        INFO(
            "NCCL library verson (%d.%d) is lower than minimum version (2.19) for this test "
            "support. Skipping this test\n",
            nccl_major, nccl_minor);
        goto fn_out;
    }

    LOAD_SYM(*nccl_handle, "GetErrorString", nccl_ftable->get_error_string);
    LOAD_SYM(*nccl_handle, "ncclCommRegister", nccl_ftable->comm_register);
    LOAD_SYM(*nccl_handle, "ncclCommDeregister", nccl_ftable->comm_deregister);
    LOAD_SYM(*nccl_handle, "ncclGetUniqueId", nccl_ftable->get_unique_id);
    LOAD_SYM(*nccl_handle, "ncclCommInitRank", nccl_ftable->comm_init_rank);
    LOAD_SYM(*nccl_handle, "ncclCommDestroy", nccl_ftable->comm_destroy);
    LOAD_SYM(*nccl_handle, "ncclAllGather", nccl_ftable->allgather);

    return 0;

fn_out:
    return -1;
};

int main(int argc, char *argv[]) {
    int mype, npes, mype_node;
    void *comm_buf, *scratch_buf, *comm_handle1, *comm_handle2, *shmem_target_buf, *nccl_target_buf;
    void *shmem_hbuf, *nccl_hbuf;
    ncclUniqueId id;
    ncclComm_t nccl_comm;
    void *nccl_lib_handle;
    uint64_t size = ALLOC_DEFAULT;

    /* Initialize NVSHMEM */
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));
    nvshmem_sync_all();

    /* Allocate scratchpad for uniqueId */
    scratch_buf = nvshmem_malloc(sizeof(ncclUniqueId));

    /* Allocate comm buffer to be used by both NCCL and NVSHMEM */
    comm_buf = nvshmem_malloc(size * sizeof(char));
    shmem_target_buf = nvshmem_malloc(size * npes * sizeof(char));
    nccl_target_buf = nvshmem_malloc(size * npes * sizeof(char));
    shmem_hbuf = calloc(size * npes, sizeof(char));
    nccl_hbuf = calloc(size * npes, sizeof(char));
    for (int i = 0; i < size * npes; i++) {
        *((uint8_t *)(shmem_hbuf) + i) = i;
    }

    CUDA_CHECK(cudaMemcpy(comm_buf, shmem_hbuf, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Load NCCL using dlopen */
    CHECK(nccl_ftable_init(&nccl_ftable, &nccl_lib_handle));

    /* Initialize NCCL */
    if (mype == 0) {
        NCCL_CHECK(nccl_ftable.get_unique_id(&id));
        CUDA_CHECK(cudaMemcpy(scratch_buf, &id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    NVSHMEM_CHECK(nvshmem_broadcastmem(NVSHMEM_TEAM_WORLD, scratch_buf, scratch_buf,
                                       sizeof(ncclUniqueId), 0));
    CUDA_CHECK(cudaMemcpy(&id, scratch_buf, sizeof(ncclUniqueId), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    NCCL_CHECK(nccl_ftable.comm_init_rank(&nccl_comm, npes, id, mype));

    /* Register symmetric memory into NCCL */
    NCCL_CHECK(nccl_ftable.comm_register(nccl_comm, comm_buf, size, &comm_handle1));
    NCCL_CHECK(nccl_ftable.comm_register(nccl_comm, nccl_target_buf, size * npes, &comm_handle2));

    /* Perform on-stream collective using NVSHMEM */
    NVSHMEM_CHECK(nvshmem_fcollectmem(NVSHMEM_TEAM_WORLD, shmem_target_buf, comm_buf, size));

    /* Perform on-stream collective using NCCL */
    NCCL_CHECK(nccl_ftable.allgather(comm_buf, nccl_target_buf, size, ncclUint8, nccl_comm,
                                     (cudaStream_t)0));

    /* Verify */
    CUDA_CHECK(cudaMemcpy(shmem_hbuf, shmem_target_buf, size * npes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nccl_hbuf, nccl_target_buf, size * npes, cudaMemcpyDeviceToHost));
    CHECK(memcmp(nccl_hbuf, shmem_hbuf, size * npes));

    /* Destroy and deregister buffers */
    NCCL_CHECK(nccl_ftable.comm_deregister(nccl_comm, comm_handle1));
    NCCL_CHECK(nccl_ftable.comm_deregister(nccl_comm, comm_handle2));

    /* Finalize NCCL and NVSHMEM */
    NCCL_CHECK(nccl_ftable.comm_destroy(nccl_comm));
    nvshmem_free(scratch_buf);
    nvshmem_free(comm_buf);
    nvshmem_free(nccl_target_buf);
    nvshmem_free(shmem_target_buf);
    free(shmem_hbuf);
    free(nccl_hbuf);
    nvshmem_finalize();
    return 0;
}
