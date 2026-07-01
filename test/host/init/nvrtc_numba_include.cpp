/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace {

const char *numba_include_kernel =
    R"(
#define __NVSHMEM_NUMBA_SUPPORT__ 1
#include <nvshmem.h>
#include <nvshmemx.h>
extern "C" __global__ void nvshmem_numba_include_kernel() {}
)";

void print_compile_log(nvrtcProgram prog) {
    size_t compile_log_size = 0;
    nvrtcGetProgramLogSize(prog, &compile_log_size);
    if (compile_log_size <= 1) return;

    std::vector<char> compile_log(compile_log_size);
    nvrtcGetProgramLog(prog, compile_log.data());
    fprintf(stderr, "Compilation log.\n%s", compile_log.data());
}

}  // namespace

int main() {
    const char *cuda_home = getenv("CUDA_HOME");
    if (!cuda_home) {
        fprintf(stderr, "This test requires CUDA_HOME to be set in the environment.\n");
        return 1;
    }

    const char *nvshmem_prefix = getenv("NVSHMEM_PREFIX");
    if (!nvshmem_prefix) {
        fprintf(stderr, "This test requires NVSHMEM_PREFIX to be set in the environment.\n");
        return 1;
    }

    std::string cuda_include_arg(cuda_home);
    std::string cuda_cccl_include_arg(cuda_home);
    std::string nvshmem_include_arg(nvshmem_prefix);

    cuda_include_arg.append("/include");
    cuda_cccl_include_arg.append("/include/cccl");
    nvshmem_include_arg.append("/include");

    /* Match Numba/NVRTC behavior: do not use -default-device to paper over
     * unannotated functions in headers included by nvshmem4py kernels. */
    const char *compile_opts[] = {"-std=c++17",
                                  "-arch",
                                  "compute_80",
                                  "-rdc",
                                  "true",
                                  "-I",
                                  cuda_include_arg.c_str(),
                                  "-I",
                                  cuda_cccl_include_arg.c_str(),
                                  "-I",
                                  nvshmem_include_arg.c_str()};

    nvrtcProgram numba_include_prog;
    nvrtcResult result = nvrtcCreateProgram(&numba_include_prog, numba_include_kernel,
                                            "nvshmem_numba_include.cu", 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create program with error %s\n", nvrtcGetErrorString(result));
        return 1;
    }

    result = nvrtcCompileProgram(numba_include_prog, sizeof(compile_opts) / sizeof(compile_opts[0]),
                                 compile_opts);
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to compile program with error %s\n", nvrtcGetErrorString(result));
        print_compile_log(numba_include_prog);
        nvrtcDestroyProgram(&numba_include_prog);
        return 1;
    }

    nvrtcDestroyProgram(&numba_include_prog);
    return 0;
}
