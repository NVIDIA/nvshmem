/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        CUresult status_ = (call);                                                          \
        if (status_ != CUDA_SUCCESS) {                                                      \
            const char *message_ = nullptr;                                                 \
            cuGetErrorString(status_, &message_);                                           \
            std::fprintf(stderr, "%s failed: %s\\n", #call, message_);                    \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

#define CUDART_CHECK(call)                                                                   \
    do {                                                                                     \
        cudaError_t status_ = (call);                                                       \
        if (status_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "%s failed: %s\\n", #call, cudaGetErrorString(status_)); \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <cubin>\\n", argv[0]);
        return 2;
    }

    nvshmem_init();

    int device_count = 0;
    CUDART_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "no CUDA devices available\\n");
        return 1;
    }
    const int node_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    const int node_npes = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    CUDART_CHECK(cudaSetDevice(node_pe / ((node_npes + device_count - 1) / device_count)));

    CUmodule module = nullptr;
    CUDA_CHECK(cuModuleLoad(&module, argv[1]));
    const int init_status = nvshmemx_cumodule_init(module);
    if (init_status != 0) {
        std::fprintf(stderr, "nvshmemx_cumodule_init failed: %d\\n", init_status);
        cuModuleUnload(module);
        return 1;
    }

    CUfunction kernel = nullptr;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, "nvshmem_fortran_bitcode_put"));

    auto *dest = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    auto *source = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    if (dest == nullptr || source == nullptr) {
        std::fprintf(stderr, "nvshmem_malloc failed\\n");
        nvshmem_free(dest);
        nvshmem_free(source);
        nvshmemx_cumodule_finalize(module);
        cuModuleUnload(module);
        nvshmem_finalize();
        return 1;
    }

    constexpr int expected = 42;
    long long nelems = 1;
    int pe = nvshmem_my_pe();
    void *args[] = {&dest, &source, &nelems, &pe};
    CUDART_CHECK(cudaMemcpy(source, &expected, sizeof(expected), cudaMemcpyHostToDevice));
    CUDART_CHECK(cudaMemset(dest, 0, sizeof(*dest)));
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    nvshmemx_barrier_all_on_stream(nullptr);
    CUDART_CHECK(cudaDeviceSynchronize());

    int received = 0;
    CUDART_CHECK(cudaMemcpy(&received, dest, sizeof(received), cudaMemcpyDeviceToHost));
    nvshmem_free(dest);
    nvshmem_free(source);
    const int finalize_status = nvshmemx_cumodule_finalize(module);
    CUDA_CHECK(cuModuleUnload(module));
    nvshmem_finalize();

    if (finalize_status != 0) {
        std::fprintf(stderr, "nvshmemx_cumodule_finalize failed: %d\\n", finalize_status);
        return 1;
    }
    if (received != expected) {
        std::fprintf(stderr, "put result mismatch: expected %d, received %d\\n", expected, received);
        return 1;
    }

    std::puts("PASS: CUDA Fortran cubin loaded, initialized, and executed an NVSHMEM put");
    return 0;
}
