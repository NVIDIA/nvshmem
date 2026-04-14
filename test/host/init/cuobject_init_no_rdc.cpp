/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Non-RDC cuModule init test.
 *
 * Validates that a fatbin compiled WITHOUT -rdc=true (with all device functions
 * inlined) can be loaded via cuModuleLoadData and initialized with
 * nvshmemx_cumodule_init.  The kernel exercises nvshmem_int_p and
 * nvshmem_barrier_all to verify that device state was correctly populated.
 */
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include "utils.h"

struct NvshmemInitGuard {
    int *argc;
    char ***argv;
    NvshmemInitGuard(int *c, char ***v) : argc(c), argv(v) { init_wrapper(argc, argv); }
    ~NvshmemInitGuard() { finalize_wrapper(); }
};

struct CUModuleGuard {
    CUmodule mod = nullptr;
    bool initialized = false;

    CUModuleGuard(const std::vector<char> &fatbin) {
        CUresult res = cuModuleLoadData(&mod, fatbin.data());
        if (res != CUDA_SUCCESS) {
            const char *err;
            cuGetErrorString(res, &err);
            fprintf(stderr, "cuModuleLoadData failed: %s\n", err);
            return;
        }
        int status = nvshmemx_cumodule_init(mod);
        if (status) {
            fprintf(stderr, "nvshmemx_cumodule_init failed: %d\n", status);
        } else {
            initialized = true;
        }
    }

    ~CUModuleGuard() {
        if (initialized) nvshmemx_cumodule_finalize(mod);
        if (mod) cuModuleUnload(mod);
    }
};

static std::string get_exe_dir() {
    char buf[1024];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len < 0) return {};
    std::string_view exe(buf, static_cast<size_t>(len));
    auto slash = exe.find_last_of('/');
    return slash == std::string_view::npos ? std::string(".") : std::string(exe.substr(0, slash));
}

static std::vector<char> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return {};
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (!f.read(buf.data(), size)) return {};
    return buf;
}

int main(int argc, char **argv) {
    NvshmemInitGuard nvshmem(&argc, &argv);
    int mype = nvshmem_my_pe();

    /* Set CUDA device based on node-local PE rank; init_wrapper only calls
     * select_device() in the non-MPI/UID fallback path. */
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        fprintf(stderr, "[PE %d] no CUDA devices available\n", mype);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(mype_node / ((npes_node + dev_count - 1) / dev_count)));

    /* Load the non-RDC fatbin */
    std::string fatbin_path = get_exe_dir() + "/kernel_nvshmem_no_rdc.fatbin";
    auto fatbin = read_file(fatbin_path);
    if (fatbin.empty()) {
        fprintf(stderr, "[PE %d] Failed to read %s\n", mype, fatbin_path.c_str());
        return 1;
    }
    printf("[PE %d] Successfully loaded %s\n", mype, fatbin_path.c_str());

    /* Load module and register with NVSHMEM */
    CUModuleGuard module(fatbin);
    if (!module.initialized) {
        fprintf(stderr, "[PE %d] Failed to load/init cumodule\n", mype);
        return 1;
    }
    printf("[PE %d] Successfully initialized non-RDC cumodule\n", mype);

    CUfunction function;
    CUresult res = cuModuleGetFunction(&function, module.mod, "kernel_nvshmem");
    if (res != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString(res, &err);
        fprintf(stderr, "[PE %d] cuModuleGetFunction failed: %s\n", mype, err);
        return 1;
    }

    /* Allocate symmetric memory */
    auto dest = std::unique_ptr<int, decltype(&nvshmem_free)>((int *)nvshmem_malloc(sizeof(int)),
                                                              nvshmem_free);
    if (!dest) {
        fprintf(stderr, "[PE %d] nvshmem_malloc failed\n", mype);
        return 1;
    }

    /* Run the kernel */
    int *dest_ptr = dest.get();
    void *args[] = {&dest_ptr};
    nvshmemx_barrier_all_on_stream(0);
    res = cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString(res, &err);
        fprintf(stderr, "[PE %d] cuLaunchKernel failed: %s\n", mype, err);
        return 1;
    }
    nvshmemx_barrier_all_on_stream(0);

    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        fprintf(stderr, "[PE %d] cudaDeviceSynchronize failed: %s\n", mype,
                cudaGetErrorString(sync));
        return 1;
    }

    /* Verify result */
    int received = 0;
    const cudaError_t rc = cudaMemcpy(&received, dest.get(), sizeof(int), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        fprintf(stderr, "[PE %d] cudaMemcpy failed: %s\n", mype, cudaGetErrorString(rc));
        return 1;
    }

    if (received != 3 * mype + 14) {
        fprintf(stderr, "[PE %d] error: expected = %d, received = %d\n", mype, 3 * mype + 14,
                received);
        return 1;
    }

    printf("[PE %d] PASSED\n", mype);
    return 0;
}
