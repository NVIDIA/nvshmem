/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvrtc.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <nvshmem.h>

const char *nvshmem_kernel =
    "                                                                                        \n\
extern \"C\" __global__ void nvshmem_put_quiet_kernel(int *src, int *dest, size_t num_elems) \n\
{                                                                                            \n\
    int pe;                                                                                  \n\
                                                                                             \n\
    pe = (nvshmem_my_pe() + 1) % nvshmem_n_pes();                                            \n\
    nvshmem_int_put_nbi(src, dest, num_elems, pe);                                           \n\
    nvshmem_quiet();                                                                         \n\
}                                                                                            \n";

int main() {
    const char *cuda_home, *nvshmem_prefix;
    std::string cuda_include_arg;
    std::string nvshmem_include_arg;
    nvrtcResult result;
    nvrtcProgram nvshmem_kern;
    size_t compile_log_size, ptx_size;
    char *compile_log, *ptx;

    cuda_home = getenv("CUDA_HOME");
    if (!cuda_home) {
        fprintf(stderr, "This test requires CUDA_HOME to be set in the environment.\n");
        return 1;
    }

    nvshmem_prefix = getenv("NVSHMEM_PREFIX");
    if (!nvshmem_prefix) {
        fprintf(stderr, "this test requires NVSHMEM_PREFIX to be set in the environment.\n");
        return 1;
    }

    cuda_include_arg.assign(cuda_home);
    nvshmem_include_arg.assign(nvshmem_prefix);

    cuda_include_arg.append("/include");
    nvshmem_include_arg.append("/include");

    const char *compile_opts[] = {"-std=c++11", "-default-device",
                                  "-arch",      "compute_70",
                                  "-rdc",       "true",
                                  "-I",         cuda_include_arg.c_str(),
                                  "-I",         nvshmem_include_arg.c_str(),
                                  "-include",   "nvshmem.h"};

    result = nvrtcCreateProgram(&nvshmem_kern, nvshmem_kernel, "put_quiet.cu", 0, NULL, NULL);
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create program with error %s\n", nvrtcGetErrorString(result));
        return 1;
    }

    result = nvrtcCompileProgram(nvshmem_kern, 12, compile_opts);
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to compile program with error %s\n", nvrtcGetErrorString(result));
        result = nvrtcGetProgramLogSize(nvshmem_kern, &compile_log_size);
        if (compile_log_size > 0) {
            compile_log = (char *)malloc(compile_log_size);
            if (compile_log) {
                nvrtcGetProgramLog(nvshmem_kern, compile_log);
                fprintf(stderr, "Compilation log. \n%s", compile_log);
            }
        }
        return 1;
    }
}
