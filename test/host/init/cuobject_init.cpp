/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdio>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <libgen.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <nvshmem.h>
#include <nvshmemx.h>
#include "utils.h"

#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            status = -1;                                                          \
            goto out;                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

struct test_args_s {
    bool use_culibrary;
};

class TestCUObject {
   public:
    explicit TestCUObject(std::vector<char> &fatbin_buffer, std::string kernel_name)
        : fatbin_(fatbin_buffer), name_(kernel_name) {}
    virtual ~TestCUObject() = default;
    virtual int initialize(CUfunction *function) = 0;
    virtual int finalize(void) = 0;

   protected:
    std::vector<char> fatbin_;
    std::string name_;
};

class TestCUModule : public TestCUObject {
   public:
    explicit TestCUModule(std::vector<char> &fatbin, std::string kernel) noexcept
        : TestCUObject(fatbin, kernel) {}
    ~TestCUModule() = default;
    int initialize(CUfunction *function) {
        int status = -1;
        /** Load result **/
        CU_CHECK(cuModuleLoadData(&mod_, fatbin_.data()));
        status = nvshmemx_cumodule_init(mod_);
        if (status) {
            fprintf(stderr, "error: nvshmemx_cumodule_init failed.\n");
            return -1;
        }
        CU_CHECK(cuModuleGetFunction(function, mod_, name_.c_str()));
        printf("Successfully loaded fatbin using cumodule\n");
        return (status);
    }

    int finalize(void) {
        /** Cleanup **/
        int status = nvshmemx_cumodule_finalize(mod_);
        if (status) {
            fprintf(stderr, "error: nvshmemx_cumodule_finalize failed.\n");
        }
        CU_CHECK(cuModuleUnload(mod_));
        return (status);
    }

   private:
    CUmodule mod_;
};

class TestCULibrary : public TestCUObject {
   public:
    explicit TestCULibrary(std::vector<char> &fatbin, std::string kernel) noexcept
        : TestCUObject(fatbin, kernel) {}
    ~TestCULibrary() = default;
    int initialize(CUfunction *function) {
#if CUDA_VERSION > 12000
        CUkernel kernel;
        int status = -1;
        /** Load result using default JIT and library options **/
        CU_CHECK(cuLibraryLoadData(&lib_, fatbin_.data(), NULL, NULL, 0, NULL, NULL, 0));
        status = nvshmemx_culibrary_init(lib_);
        if (status) {
            fprintf(stderr, "error: nvshmemx_culibrary_init failed.\n");
            return -1;
        }
        CU_CHECK(cuLibraryGetKernel(&kernel, lib_, name_.c_str()));
        CU_CHECK(cuKernelGetFunction(function, kernel));
        printf("Successfully loaded fatbin using culibrary\n");
        return (status);
#else
        return (-1); /* Unsupported */
#endif
    }

    int finalize(void) {
#if CUDA_VERSION > 12000
        /** Cleanup **/
        int status = nvshmemx_culibrary_finalize(lib_);
        if (status) {
            fprintf(stderr, "error: nvshmemx_culibrary_finalize failed.\n");
        }
        CU_CHECK(cuLibraryUnload(lib_));
        return (status);
#else
        return (-1); /* Unsupported */
#endif
    }

   private:
#if CUDA_VERSION > 12000
    CUlibrary lib_;
#else
    void *lib_;
#endif
};

// Default: Check for CUDA version and if on older CUDA, use module else use library
static void detect_module_or_library_support(struct test_args_s *args) {
    int cuda_driver_version = -1;
    char *use_module = NULL;
    __attribute__((unused)) int status = -1;
    (*args).use_culibrary = false;
    CUDA_RUNTIME_CHECK(cudaDriverGetVersion(&cuda_driver_version));
    if (cuda_driver_version >= 12000) {
        (*args).use_culibrary = true;
    } else {
        return;
    }

    use_module = getenv("NVSHMEMTEST_USE_CUMODULE");
    if (use_module) (*args).use_culibrary = false;

out:
    return;
}

int main(int argc, char **argv) {
    int mype;
    int status = 0;
    TestCUObject *t = nullptr;
    CUfunction function;
    const char *kernel_name = "_Z14kernel_nvshmemPi";

    struct test_args_s test_args = {0};

    init_wrapper(&argc, &argv);
    detect_module_or_library_support(&test_args);

    int *destination = (int *)nvshmem_malloc(sizeof(int));
    void *args[] = {&destination};
    int received = 0;

    mype = nvshmem_my_pe();

    char exe_path[1000];
    size_t count = readlink("/proc/self/exe", exe_path, 1000);
    exe_path[count] = '\0';

    char *exe_dir = dirname(exe_path);
    char fatbin_path[1000];
    strcpy(fatbin_path, exe_dir);
    strcat(fatbin_path, "/kernel_nvshmem.fatbin");

    std::ifstream file(fatbin_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> fatbin_buffer(size);
    if (file.read(fatbin_buffer.data(), size)) {
        printf("Successfully loaded file %s\n", fatbin_path);
    } else {
        printf("Failed to read file %s\n", fatbin_path);
        status = 1;
        goto cleanup;
    }

    /** Load Result **/
    nvshmemx_barrier_all_on_stream(0);
    if (!test_args.use_culibrary)
        t = new TestCUModule(fatbin_buffer, kernel_name);
    else
        t = new TestCULibrary(fatbin_buffer, kernel_name);
    t->initialize(&function);

    /** Run **/
    CU_CHECK(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, args, 0));
    nvshmemx_barrier_all_on_stream(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&received, destination, sizeof(int), cudaMemcpyDeviceToHost));
    nvshmem_free(destination);
    if (received != 3 * mype + 14) {
        printf("error: expected = %d, received = %d\n", 3 * mype + 14, received);
        status = 1;
        goto cleanup;
    }

cleanup:
    /** Cleanup **/
    if (t != nullptr) t->finalize();
    if (t != nullptr) delete t;

    finalize_wrapper();

    if (!status) {
        printf("PASSED\n");
    }
    return status;
}
