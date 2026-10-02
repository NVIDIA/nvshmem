/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "nvml.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"
#include "utils.h"
#include "coll_common.h"
#include "reduce_common.h"

#define GRAN 512 * 1024 * 1024
#define COLL_NELEMS 4096
#define MEM_ALLOC_TYPE CU_MEM_ALLOCATION_TYPE_PINNED
#define MEM_ALLOC_LOCATION_TYPE CU_MEM_LOCATION_TYPE_DEVICE

#define DATATYPE_T int
#define DATATYPE_NAME int
#define DO_REDUCE_TEST(TYPENAME, TYPE, OP)                                                       \
    do {                                                                                         \
        init_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)source,   \
                                                                         nelems);                \
        cudaStreamSynchronize(stream);                                                           \
        nvshmem_barrier(team);                                                                   \
        nvshmem_##TYPENAME##_##OP##_reduce(team, (TYPE *)dest, (const TYPE *)source, nelems);    \
        validate_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest, \
                                                                             nelems);            \
        reset_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest,    \
                                                                          nelems);               \
        cudaStreamSynchronize(stream);                                                           \
    } while (0);

#define DO_REDUCE_ON_STREAM_TEST(TYPENAME, TYPE, OP)                                             \
    do {                                                                                         \
        init_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)source,   \
                                                                         nelems);                \
        nvshmemx_barrier_on_stream(team, stream);                                                \
        nvshmemx_##TYPENAME##_##OP##_reduce_on_stream(team, (TYPE *)dest, (const TYPE *)source,  \
                                                      nelems, stream);                           \
        validate_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest, \
                                                                             nelems);            \
        reset_##TYPENAME##_##OP##_reduce_data_kernel<<<1, 1, 0, stream>>>(team, (TYPE *)dest,    \
                                                                          nelems);               \
        cudaStreamSynchronize(stream);                                                           \
    } while (0);

namespace {

int check_cuda_driver(CUresult result, const char *call) {
    if (result == CUDA_SUCCESS) {
        return 0;
    }

    const char *str = nullptr;
    CUresult ret = cuGetErrorString(result, &str);
    if (ret == CUDA_ERROR_INVALID_VALUE || str == nullptr) {
        str = "Unknown error";
    }
    ERROR_PRINT("%s failed with %s \n", call, str);
    return -1;
}

int check_cuda_runtime(cudaError_t result, const char *call) {
    if (result == cudaSuccess) {
        return 0;
    }

    ERROR_PRINT("%s failed with %s \n", call, cudaGetErrorString(result));
    return -1;
}

struct NvshmemGuard {
    bool initialized = false;

    NvshmemGuard(int *argc, char ***argv) {
        init_wrapper(argc, argv);
        initialized = true;
    }

    ~NvshmemGuard() {
        if (initialized) {
            finalize_wrapper();
        }
    }

    NvshmemGuard(const NvshmemGuard &) = delete;
    NvshmemGuard &operator=(const NvshmemGuard &) = delete;
};

class CudaStreamGuard {
   public:
    CudaStreamGuard() = default;
    ~CudaStreamGuard() { reset(); }

    CudaStreamGuard(const CudaStreamGuard &) = delete;
    CudaStreamGuard &operator=(const CudaStreamGuard &) = delete;

    int create() {
        return check_cuda_runtime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                                  "cudaStreamCreateWithFlags");
    }

    cudaStream_t get() const { return stream_; }

   private:
    void reset() {
        if (stream_ != nullptr) {
            check_cuda_runtime(cudaStreamDestroy(stream_), "cudaStreamDestroy");
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

class SymmetricMapping {
   public:
    SymmetricMapping() = default;
    ~SymmetricMapping() { reset(); }

    SymmetricMapping(const SymmetricMapping &) = delete;
    SymmetricMapping &operator=(const SymmetricMapping &) = delete;
    SymmetricMapping(SymmetricMapping &&) = delete;
    SymmetricMapping &operator=(SymmetricMapping &&) = delete;

    int register_symmetric(void *user_buf, size_t size) {
        int status = unregister();
        if (status) {
            return status;
        }

        void *mapping = nvshmemx_buffer_register_symmetric(user_buf, size, 0);
        if (mapping == nullptr) {
            return -1;
        }

        mapping_ = mapping;
        size_ = size;
        registered_ = true;
        return 0;
    }

    int register_at_preferred(void *user_buf, size_t size, void *preferred_addr) {
        int status = unregister();
        if (status) {
            return status;
        }

        void *mapping = nvshmemx_buffer_register_symmetric_at_preferred_address(user_buf, size,
                                                                                preferred_addr, 0);
        if (mapping == nullptr) {
            return -1;
        }

        mapping_ = mapping;
        size_ = size;
        registered_ = true;
        return 0;
    }

    int unregister() {
        if (!registered_) {
            return 0;
        }

        int status = nvshmemx_buffer_unregister_symmetric(mapping_, size_);
        if (status) {
            return status;
        }

        registered_ = false;
        return 0;
    }

    void *addr() const { return mapping_; }

    void reset() {
        if (!registered_) {
            return;
        }

        int status = nvshmemx_buffer_unregister_symmetric(mapping_, size_);
        if (status) {
            ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed during cleanup \n");
        }
        registered_ = false;
    }

   private:
    void *mapping_ = nullptr;
    size_t size_ = 0;
    bool registered_ = false;
};

class UserBuffer {
   public:
    UserBuffer() = default;
    ~UserBuffer() { reset(); }

    UserBuffer(const UserBuffer &) = delete;
    UserBuffer &operator=(const UserBuffer &) = delete;
    UserBuffer(UserBuffer &&) = delete;
    UserBuffer &operator=(UserBuffer &&) = delete;

    int allocate(size_t size, const CUmemAllocationProp &prop) {
        CUmemAccessDesc access_desc = {};
        access_desc.location.id = prop.location.id;
        access_desc.location.type = prop.location.type;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        size_ = size;

        int status = check_cuda_driver(cuMemCreate(&alloc_handle_, size_, &prop, 0), "cuMemCreate");
        if (status) {
            reset();
            return status;
        }
        alloc_handle_valid_ = true;

        status = check_cuda_driver(
            cuMemAddressReserve((CUdeviceptr *)&ptr_, size_, 0, (CUdeviceptr)NULL, 0),
            "cuMemAddressReserve");
        if (status) {
            reset();
            return status;
        }
        address_reserved_ = true;

        status =
            check_cuda_driver(cuMemMap((CUdeviceptr)ptr_, size_, 0, alloc_handle_, 0), "cuMemMap");
        if (status) {
            reset();
            return status;
        }
        mapped_ = true;

        status = check_cuda_driver(cuMemSetAccess((CUdeviceptr)ptr_, size_, &access_desc, 1),
                                   "cuMemSetAccess");
        if (status) {
            reset();
            return status;
        }

        return 0;
    }

    int register_symmetric() { return symmetric_mapping_.register_symmetric(ptr_, size_); }
    int register_symmetric_at_preferred(void *preferred_addr) {
        return symmetric_mapping_.register_at_preferred(ptr_, size_, preferred_addr);
    }
    int unregister_symmetric() { return symmetric_mapping_.unregister(); }

    void *user_ptr() const { return ptr_; }
    void *mapped_ptr() const { return symmetric_mapping_.addr(); }
    size_t size() const { return size_; }

   private:
    void reset() {
        symmetric_mapping_.reset();

        if (mapped_) {
            check_cuda_driver(cuMemUnmap((CUdeviceptr)ptr_, size_), "cuMemUnmap");
            mapped_ = false;
        }
        if (address_reserved_) {
            check_cuda_driver(cuMemAddressFree((CUdeviceptr)ptr_, size_), "cuMemAddressFree");
            address_reserved_ = false;
        }
        if (alloc_handle_valid_) {
            check_cuda_driver(cuMemRelease(alloc_handle_), "cuMemRelease");
            alloc_handle_valid_ = false;
        }

        ptr_ = nullptr;
        size_ = 0;
    }

    void *ptr_ = nullptr;
    size_t size_ = 0;
    CUmemGenericAllocationHandle alloc_handle_ = {};
    bool alloc_handle_valid_ = false;
    bool address_reserved_ = false;
    bool mapped_ = false;
    SymmetricMapping symmetric_mapping_;
};

int test_physical_allocation_reclaimed(const CUmemAllocationProp &prop, size_t size,
                                       unsigned int iterations, int mype) {
    size_t free_before = 0;
    size_t free_after = 0;
    size_t total = 0;
    void *preferred_addr = nullptr;

    auto run_iteration = [&]() -> int {
        {
            UserBuffer buffer{};
            int status = buffer.allocate(size, prop);
            if (status) {
                return status;
            }

            if (preferred_addr == nullptr) {
                status = buffer.register_symmetric();
            } else {
                status = buffer.register_symmetric_at_preferred(preferred_addr);
            }
            if (status) {
                ERROR_PRINT("symmetric registration failed during reclaim test \n");
                return status;
            }

            if (preferred_addr == nullptr) {
                preferred_addr = buffer.mapped_ptr();
            } else if (buffer.mapped_ptr() != preferred_addr) {
                ERROR_PRINT("preferred symmetric address changed from %p to %p \n", preferred_addr,
                            buffer.mapped_ptr());
                return -1;
            }

            status = buffer.unregister_symmetric();
            if (status) {
                ERROR_PRINT("symmetric unregistration failed during reclaim test \n");
                return status;
            }
        }

        return 0;
    };

    int status = run_iteration();
    if (status) {
        return status;
    }

    nvshmem_barrier_all();
    status = check_cuda_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (status) {
        return status;
    }
    status = check_cuda_runtime(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo");
    if (status) {
        return status;
    }

    for (unsigned int i = 0; i < iterations; ++i) {
        status = run_iteration();
        if (status) {
            return status;
        }

        nvshmem_barrier_all();
        status = check_cuda_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (status) {
            return status;
        }
        status = check_cuda_runtime(cudaMemGetInfo(&free_after, &total), "cudaMemGetInfo");
        if (status) {
            return status;
        }

        const size_t unreclaimed = free_before > free_after ? free_before - free_after : 0;
        printf(
            "PE %d reclaim iteration %u: baseline free=%zu MiB, current free=%zu MiB, "
            "unreclaimed=%zu MiB\n",
            mype, i + 1, free_before / (1024 * 1024), free_after / (1024 * 1024),
            unreclaimed / (1024 * 1024));
        fflush(stdout);

        if (free_after < free_before) {
            ERROR_PRINT("%zu MiB remains allocated after reclaim iteration %u\n",
                        unreclaimed / (1024 * 1024), i + 1);
            return -1;
        }
    }

    return 0;
}

// Agree on failures before returning so every PE follows the same collective cleanup path.
int collective_status(int *status_buffer, int local_status) {
    nvshmem_int_p(status_buffer, local_status ? EXIT_FAILURE : EXIT_SUCCESS, nvshmem_my_pe());
    nvshmem_quiet();
    const int reduce_status =
        nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, status_buffer + 1, status_buffer, 1);
    int failed = EXIT_SUCCESS;
    const int copy_status = check_cuda_runtime(
        cudaMemcpy(&failed, status_buffer + 1, sizeof(failed), cudaMemcpyDeviceToHost),
        "cudaMemcpy");
    // Every PE must consume the result before the next reduction reuses its storage.
    nvshmem_barrier_all();
    return reduce_status || copy_status || failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

int test_rejected_registration(const CUmemAllocationProp &prop, size_t size, int dev_count,
                               int mype) {
    std::unique_ptr<int, decltype(&nvshmem_free)> status_buffer(
        static_cast<int *>(nvshmem_malloc(2 * sizeof(int))), nvshmem_free);
    if (!status_buffer) {
        ERROR_PRINT("failed to allocate validation status buffer\n");
        return EXIT_FAILURE;
    }
    const int npes = nvshmem_n_pes();

    UserBuffer existing{};
    int status = collective_status(status_buffer.get(), existing.allocate(size, prop));
    if (!status) {
        status = collective_status(status_buffer.get(), existing.register_symmetric());
    }
    if (status) {
        ERROR_PRINT("failed to prepare validation regression\n");
        return status;
    }

    auto reject = [&](const char *name, void *candidate, size_t candidate_size, int flags) {
        void *mapping = nvshmemx_buffer_register_symmetric(candidate, candidate_size, flags);
        if (mapping != nullptr) {
            ERROR_PRINT("PE %d: %s unexpectedly registered\n", mype, name);
        }
        int status = collective_status(status_buffer.get(), mapping != nullptr);
        if (status) {
            return status;
        }

        const int next = (mype + 1) % npes;
        const int previous = (mype + npes - 1) % npes;
        nvshmem_int_p(static_cast<int *>(existing.mapped_ptr()), mype + 1, next);
        nvshmem_quiet();
        nvshmem_barrier_all();
        int value = 0;
        const int copy_status = check_cuda_runtime(
            cudaMemcpy(&value, existing.mapped_ptr(), sizeof(value), cudaMemcpyDeviceToHost),
            "cudaMemcpy");
        if (copy_status || value != previous + 1) {
            ERROR_PRINT("PE %d: existing registration broken after %s\n", mype, name);
        }
        status = collective_status(status_buffer.get(), copy_status || value != previous + 1);
        if (status) {
            return status;
        }
        if (mype == 0) {
            printf("Rejected %s on one PE; existing registration remains usable\n", name);
        }
        return EXIT_SUCCESS;
    };

    void *valid = existing.user_ptr();
    if (reject("zero size", valid, mype == 0 ? 0 : size, 0) ||
        reject("unaligned size", valid, mype == 0 ? size - 1 : size, 0) ||
        reject("invalid pointer", mype == 0 ? nullptr : valid, size, 0) ||
        reject("unsupported flags", valid, size, mype == 0 ? 1 : 0)) {
        return EXIT_FAILURE;
    }

    {
        UserBuffer undersized{};
        status = collective_status(status_buffer.get(),
                                   undersized.allocate(mype == 0 ? size : 2 * size, prop));
        if (status) {
            return status;
        }
        status = reject("CUDA map beyond allocation", undersized.user_ptr(), 2 * size, 0);
        if (status) {
            return status;
        }

        UserBuffer hole{};
        if (collective_status(status_buffer.get(), hole.allocate(2 * size, prop)) ||
            collective_status(status_buffer.get(), hole.register_symmetric()) ||
            collective_status(status_buffer.get(), hole.unregister_symmetric())) {
            return EXIT_FAILURE;
        }
        void *preferred = hole.mapped_ptr();
        status = reject("CUDA map failure in a reused hole", undersized.user_ptr(), 2 * size, 0);
        if (status) {
            return status;
        }
        if (collective_status(status_buffer.get(),
                              hole.register_symmetric_at_preferred(preferred)) ||
            collective_status(status_buffer.get(), hole.mapped_ptr() != preferred) ||
            collective_status(status_buffer.get(), hole.unregister_symmetric())) {
            ERROR_PRINT("failed mapping did not restore the reusable address range\n");
            return EXIT_FAILURE;
        }
    }

    const int can_test_wrong_gpu =
        prop.location.type == CU_MEM_LOCATION_TYPE_DEVICE && dev_count > 1;
    nvshmem_barrier_all();
    if (mype == 0) {
        for (int pe = 0; pe < npes; ++pe) {
            nvshmem_int_p(static_cast<int *>(existing.mapped_ptr()), can_test_wrong_gpu, pe);
        }
        nvshmem_quiet();
    }
    nvshmem_barrier_all();
    int root_can_test = 0;
    status = collective_status(
        status_buffer.get(),
        check_cuda_runtime(cudaMemcpy(&root_can_test, existing.mapped_ptr(), sizeof(root_can_test),
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy"));
    if (status) {
        return status;
    }
    if (root_can_test) {
        UserBuffer wrong_device{};
        int allocation_status = 0;
        if (mype == 0) {
            CUmemAllocationProp wrong_prop = prop;
            wrong_prop.location.id = (prop.location.id + 1) % dev_count;
            allocation_status = wrong_device.allocate(size, wrong_prop);
        }
        status = collective_status(status_buffer.get(), allocation_status);
        if (status) {
            return status;
        }
        status = reject("wrong GPU", mype == 0 ? wrong_device.user_ptr() : valid, size, 0);
        if (status) {
            return status;
        }
    }

    UserBuffer subsequent{};
    if (collective_status(status_buffer.get(), subsequent.allocate(size, prop)) ||
        collective_status(status_buffer.get(), subsequent.register_symmetric()) ||
        collective_status(status_buffer.get(), subsequent.unregister_symmetric()) ||
        collective_status(status_buffer.get(), existing.unregister_symmetric())) {
        ERROR_PRINT("heap could not register/unregister after rejected inputs\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int check_collective_errors(uint64_t *errs) {
    int status =
        check_cuda_runtime(cudaMemcpyFromSymbol(errs, errs_d, sizeof(unsigned long long int), 0),
                           "cudaMemcpyFromSymbol");
    if (status) {
        return status;
    }

    if (*errs) {
        printf("Validation errors found\n");
        return static_cast<int>(*errs);
    }
    return 0;
}

}  // namespace

int test_allocation_at_preferred_address(void *user_buf, size_t size, void *mmaped_addr,
                                         bool should_match_preferred_addr) {
    int status = 0;
    SymmetricMapping mapping;

    // register with preferred address
    status = mapping.register_at_preferred(user_buf, size, mmaped_addr);
    if (status) {
        ERROR_PRINT("shmem_mmap of user buffer at preferred address failed \n");
        return status;
    }
    // A free preferred address must be honored; an occupied one must use another address.
    if ((mapping.addr() == mmaped_addr) != should_match_preferred_addr) {
        ERROR_PRINT(
            "Unexpected preferred-address result: requested %p, mapped %p, expected match "
            "%d\n",
            mmaped_addr, mapping.addr(), should_match_preferred_addr);
        status = -1;
    }

    int unregister_status = mapping.unregister();
    if (unregister_status) {
        ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
        if (!status) {
            status = unregister_status;
        }
    }
    return status;
}

int main(int argc, char **argv) {
    typedef DATATYPE_T dtype_t;

    uint64_t errs = 0;
    int status = 0;
    int mype;
    size_t size;
    char size_string[100];
    CUmemAllocationProp prop = {};
    int dev_count, dev_id, npes_per_gpu;
    size_t granularity = GRAN;
    unsigned int iter = 0;
    uint32_t lsize;
    unsigned int seed = 0;
    bool enable_egm = false;
    int numa_id;
    CUdevice my_dev;
    int cuda_drv_version;
    int bufId;
    int free_start_idx, free_end_idx;
    dtype_t *source, *dest;
    size_t nelems;
    nvshmem_team_t team = NVSHMEM_TEAM_WORLD;

    read_args(argc, argv);
    enable_egm = use_egm;
    size = ((_max_size - 1) / (GRAN) + 1) * (GRAN);
    size = _min_iters * size;
    snprintf(size_string, sizeof(size_string), "%zu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        return status;
    }

    NvshmemGuard nvshmem(&argc, &argv);

    srand(1);
    mype = nvshmem_my_pe();
    iter = _min_iters;
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    status = check_cuda_runtime(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount");
    if (status) {
        return status;
    }
    npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    dev_id = (mype_node / npes_per_gpu);
    status = check_cuda_runtime(cudaSetDevice(dev_id), "cudaSetDevice");
    if (status) {
        return status;
    }
    status = check_cuda_driver(cuDeviceGet(&my_dev, dev_id), "cuDeviceGet");
    if (status) {
        return status;
    }
    status = check_cuda_runtime(cudaDriverGetVersion(&cuda_drv_version), "cudaDriverGetVersion");
    if (status) {
        return status;
    }
    prop.type = MEM_ALLOC_TYPE;
    prop.location.type = MEM_ALLOC_LOCATION_TYPE;
    prop.location.id = dev_id;
    if (enable_egm) {
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        status = check_cuda_driver(
            cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev),
            "cuDeviceGetAttribute");
        if (status) {
            return status;
        }
        prop.location.id = numa_id;
    } else {
        prop.allocFlags.gpuDirectRDMACapable = 1;
    }

    prop.requestedHandleTypes =
        (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (is_mnnvl_supported(dev_id)) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    }
    // override if user specified mem handle type
    if (_mem_handle_type == MEM_TYPE_FABRIC) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    } else if (_mem_handle_type == MEM_TYPE_POSIX_FD) {
        prop.requestedHandleTypes =
            (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }
    status = check_cuda_driver(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
    if (status) {
        return status;
    }

    size_t reclaim_test_size = ((_max_size - 1) / granularity + 1) * granularity;
    status = test_rejected_registration(prop, reclaim_test_size, dev_count, mype);
    if (status) {
        return status;
    }
    status = test_physical_allocation_reclaimed(prop, reclaim_test_size, _repeat, mype);
    if (status) {
        return status;
    }

    std::vector<UserBuffer> buffers(iter);

    if (!mype) {
        DEBUG_PRINT("creating and mmapping %d buffers ", iter);
    }
    for (unsigned int i = 0; i < iter; i++) {
        if (i == 0) {
            // Exercise a physical mapping smaller than the NVSHMEM heap granularity.
            lsize = granularity;
        } else {
            seed = i;
            lsize = rand_r(&seed) % (_max_size - _min_size + 1) + _min_size;
            lsize = ((lsize - 1) / granularity + 1) * granularity;
        }

        UserBuffer &buffer = buffers[i];
        status = buffer.allocate(lsize, prop);
        if (status) {
            return status;
        }

        status = buffer.register_symmetric();
        if (status) {
            ERROR_PRINT("shmem_mmap failed \n");
            return status;
        }

        if (use_egm) {
            memset(buffer.mapped_ptr(), 0, lsize);
        } else {
            status = check_cuda_runtime(cudaMemset(buffer.mapped_ptr(), 0, lsize), "cudaMemset");
            if (status) {
                return status;
            }
        }
    }

    // test allocation when preferred address is unavailable
    // allocation should succeed but should not be at preferred address
    if (iter > 2) {
        status = test_allocation_at_preferred_address(buffers[0].user_ptr(), buffers[0].size(),
                                                      buffers[1].mapped_ptr(), false);
        if (status) {
            ERROR_PRINT("test_allocation_at_preferred_address failed \n");
            return status;
        }
    }

    for (size_t r = 0; r < _repeat; r++) {
        free_start_idx = 0;
        free_end_idx = iter - 1;

        // test heap usage to verify mmap correctness
        CudaStreamGuard stream_guard;
        status = stream_guard.create();
        if (status) {
            return status;
        }
        cudaStream_t stream = stream_guard.get();

        bufId = r % iter;
        // Limiting the size of collective to limit test run time
        if (COLL_NELEMS * sizeof(dtype_t) > buffers[bufId].size()) {
            nelems = buffers[bufId].size() / sizeof(dtype_t);
        } else {
            nelems = COLL_NELEMS;
        }
        nelems = nelems / 2;  // split the buffer into source and dest

        source = (dtype_t *)buffers[bufId].mapped_ptr();
        dest = (dtype_t *)(buffers[bufId].mapped_ptr()) + nelems;

        DO_REDUCE_ON_STREAM_TEST(int, dtype_t, sum);
        DO_REDUCE_TEST(int, dtype_t, sum);
        status = check_collective_errors(&errs);
        if (status) {
            return status;
        }
        fflush(stdout);
        nvshmem_barrier_all();

        // check if preferred allocation within unmapped region works
        if (iter > 2) {
            // create a hole by unmapping buf id =0,1
            for (int i = 0; i < 2; i++) {
                status = buffers[i].unregister_symmetric();
                if (status) {
                    ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
                    return status;
                }
            }

            // register with buf id 1 as preferred offset
            status = test_allocation_at_preferred_address(buffers[1].user_ptr(), buffers[1].size(),
                                                          buffers[1].mapped_ptr(), true);
            if (status) {
                ERROR_PRINT(
                    "test_allocation_at_preferred_address within unmapped region failed \n");
                return status;
            }

            // restore buf 0,1
            for (int i = 0; i < 2; i++) {
                status = buffers[i].register_symmetric();
                if (status) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    return status;
                }
            }
        }
        if (!mype) {
            DEBUG_PRINT("freeing buffers with index %d to %d \n", free_start_idx, free_end_idx);
        }
        for (int i = free_start_idx; i <= free_end_idx; i++) {
            status = buffers[i].unregister_symmetric();
            if (status) {
                ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
                return status;
            }
        }

        // check if preferred allocation in empty heap region
        if (iter > 2) {
            status = test_allocation_at_preferred_address(buffers[1].user_ptr(), buffers[1].size(),
                                                          buffers[1].mapped_ptr(), true);
            if (status) {
                ERROR_PRINT("test_allocation_at_preferred_address failed \n");
                return status;
            }
        }

        if (!mype) {
            DEBUG_PRINT("re-allocating buffers with index %d to %d \n", free_start_idx,
                        free_end_idx);
        }
        if (r % 2) {
            for (int i = free_end_idx; i >= free_start_idx; i--) {
                status = buffers[i].register_symmetric();
                if (status) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    return status;
                }
                if (use_egm) {
                    memset(buffers[i].mapped_ptr(), 0, buffers[i].size());
                } else {
                    status = check_cuda_runtime(
                        cudaMemset(buffers[i].mapped_ptr(), 0, buffers[i].size()), "cudaMemset");
                    if (status) {
                        return status;
                    }
                }
            }
        } else {
            for (int i = free_start_idx; i <= free_end_idx; i++) {
                status = buffers[i].register_symmetric();
                if (status) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    return status;
                }
                if (use_egm) {
                    memset(buffers[i].mapped_ptr(), 0, buffers[i].size());
                } else {
                    status = check_cuda_runtime(
                        cudaMemset(buffers[i].mapped_ptr(), 0, buffers[i].size()), "cudaMemset");
                    if (status) {
                        return status;
                    }
                }
            }
        }
        if (!mype) {
            DEBUG_PRINT("done repetition %d \n", r);
        }
    }

    fflush(stdout);

    if (!mype) {
        DEBUG_PRINT("[binsize %d] done testing \n", b);
    }

    return status;
}
