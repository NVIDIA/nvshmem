/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nvrtc.h>
#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <nvJitLink.h>
#include <vector>
#include <iostream>
#include <fstream>

#define NVJITLINK_CHECK(linker, call)                                                        \
    do {                                                                                     \
        nvJitLinkResult res = (call);                                                        \
        if (res != NVJITLINK_SUCCESS) {                                                      \
            std::cerr << "NVJITLink error: " << res << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                          \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

// This kernel just includes NVSHMEM
// If it passes when run against the NVSHMEM LTOIR library
// we know NVSHMEM is NVRTC/Numba/LTO-IR compliant
const char *include_test_kernel =
    "\
#define __NVSHMEM_NUMBA_SUPPORT__ 1 \n\
#include<nvshmem.h>                 \n\
#include<nvshmemx.h>               \n";

const char *nvshmem_kernel =
    "\
    #define __NVSHMEM_NUMBA_SUPPORT__ 1 \n\
    extern \"C\" {\n\
    // Need forward declarations of the NVSHMEM functions so we can generate the LTO-IR          \n\
    // They are satisfied at link time against the LTO kernel with the headers                   \n\
    __device__ void nvshmem_int_put_nbi(int * src, int * dest, size_t num_elems, int pe);        \n\
    __device__ void nvshmem_quiet(void);                                                         \n\
    __device__ int nvshmem_my_pe(void);                                                          \n\
    __device__ int nvshmem_n_pes(void);                                                          \n\
    }                                                                                             \n\
     __global__ void nvshmem_put_quiet_kernel(int *src, int *dest, size_t num_elems)             \n\
    {                                                                                            \n\
        int pe;                                                                                  \n\
                                                                                                 \n\
        pe = (nvshmem_my_pe() + 1) % nvshmem_n_pes();                                            \n\
        nvshmem_int_put_nbi(src, dest, num_elems, pe);                                           \n\
        nvshmem_quiet();                                                                         \n\
    }                                                                                            \n\
\n";

int main() {
    const char *cuda_home, *nvshmem_prefix;
    std::string cuda_include_arg;
    std::string nvshmem_include_arg;
    nvrtcResult result1, result2;
    nvrtcProgram nvshmem_kern, include_test_kern;
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

    const char *compile_opts_header_only[] = {"-std=c++11",
                                              "-default-device",
                                              "-arch",
                                              "sm_70",
                                              "-rdc",
                                              "true",
                                              "-dlto",
                                              "-I",
                                              cuda_include_arg.c_str(),
                                              "-I",
                                              nvshmem_include_arg.c_str()};

    const char *compile_opts[] = {
        "-std=c++11", "-default-device",       "-rdc", "true", "-arch", "sm_70", "-dlto",
        "-I",         cuda_include_arg.c_str()};

    /**
     * Compile NVSHMEM Kernel
     */

    result1 = nvrtcCreateProgram(&nvshmem_kern, nvshmem_kernel, "put_quiet.cu", 0, NULL, NULL);
    if (result1 != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create program with error %s\n", nvrtcGetErrorString(result1));
        return 1;
    }

    result1 = nvrtcCompileProgram(nvshmem_kern, sizeof(compile_opts) / sizeof(compile_opts[0]),
                                  compile_opts);
    if (result1 != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to compile nvshmem kernel with error %s\n",
                nvrtcGetErrorString(result1));
        result1 = nvrtcGetProgramLogSize(nvshmem_kern, &compile_log_size);
        if (compile_log_size > 0) {
            compile_log = (char *)malloc(compile_log_size);
            if (compile_log) {
                nvrtcGetProgramLog(nvshmem_kern, compile_log);
                fprintf(stderr, "Compilation log. \n%s", compile_log);
            }
        }
        return 1;
    }

    /**
     * Compile include kernel as header-only library
     */
    result2 = nvrtcCreateProgram(&include_test_kern, include_test_kernel, "include_nvshmem.cu", 0,
                                 NULL, NULL);
    if (result2 != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create program with error %s\n", nvrtcGetErrorString(result2));
        return 1;
    }

    result2 = nvrtcCompileProgram(
        include_test_kern, sizeof(compile_opts_header_only) / sizeof(compile_opts_header_only[0]),
        compile_opts_header_only);
    if (result2 != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to compile nvshmem include program with error %s\n",
                nvrtcGetErrorString(result2));
        result2 = nvrtcGetProgramLogSize(include_test_kern, &compile_log_size);
        if (compile_log_size > 0) {
            compile_log = (char *)malloc(compile_log_size);
            if (compile_log) {
                nvrtcGetProgramLog(include_test_kern, compile_log);
                fprintf(stderr, "Compilation log. \n%s", compile_log);
            }
        }
        return 1;
    }

    // Get LTO IR for nvshmem_kernel
    size_t prog1_size;
    nvrtcGetLTOIRSize(nvshmem_kern, &prog1_size);
    std::vector<char> lto_ir1(prog1_size);
    nvrtcGetLTOIR(nvshmem_kern, lto_ir1.data());

    // Get LTO IR for include_test_kernel
    size_t prog2_size;
    nvrtcGetLTOIRSize(include_test_kern, &prog2_size);
    std::vector<char> lto_ir2(prog2_size);
    nvrtcGetLTOIR(include_test_kern, lto_ir2.data());
    std::cout << "NVSHMEM kernel size " << prog1_size << "\n";
    std::cout << "NVSHMEM header-only library size " << prog2_size << "\n";
    std::ofstream ostrm("header_only.ll", std::ios::binary);
    ostrm.write(lto_ir2.data(), prog2_size);
    // Create NVJitLink linker
    nvJitLinkHandle linker;
    const char *link_opts[] = {"-arch=sm_70", "-lto", "-verbose"};
    NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker, 2, link_opts));

    // Add both LTO IRs to linker
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, lto_ir2.data(),
                                             prog2_size, "include_test_kernel"));
    NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, lto_ir1.data(),
                                             prog1_size, "nvshmem_kernel"));
    // Finalize and get cubin
    size_t cubin_size;
    NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));

    size_t log_size = 0;
    nvJitLinkGetErrorLogSize(linker, &log_size);

    if (log_size > 1) {
        std::vector<char> link_log(log_size);
        nvJitLinkGetErrorLog(linker, link_log.data());
        std::cout << "Linker log:\n" << link_log.data() << "\n";
    }

    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));

    // Cleanup
    NVJITLINK_CHECK(linker, nvJitLinkDestroy(&linker));
    nvrtcDestroyProgram(&nvshmem_kern);
    nvrtcDestroyProgram(&include_test_kern);

    std::cout << "Linking successful. Final CUBIN size: " << cubin_size << " bytes\n";
    return 0;
}
