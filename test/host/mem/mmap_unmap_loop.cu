/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <iostream>
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
    if (result == CUDA_SUCCESS) return 0;

    const char *str = nullptr;
    CUresult ret = cuGetErrorString(result, &str);
    if (ret == CUDA_ERROR_INVALID_VALUE || str == nullptr) str = "Unknown error";
    ERROR_PRINT("%s failed with %s \n", call, str);
    return -1;
}

int check_cuda_runtime(cudaError_t result, const char *call) {
    if (result == cudaSuccess) return 0;

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
        if (initialized) finalize_wrapper();
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
        if (status) return status;

        void *mapping = nvshmemx_buffer_register_symmetric(user_buf, size, 0);
        if (mapping == nullptr) return -1;

        mapping_ = mapping;
        size_ = size;
        registered_ = true;
        return 0;
    }

    int register_at_preferred(void *user_buf, size_t size, void *preferred_addr) {
        int status = unregister();
        if (status) return status;

        void *mapping = nvshmemx_buffer_register_symmetric_at_preferred_address(user_buf, size,
                                                                                preferred_addr, 0);
        if (mapping == nullptr) return -1;

        mapping_ = mapping;
        size_ = size;
        registered_ = true;
        return 0;
    }

    int unregister() {
        if (!registered_) return 0;

        int status = nvshmemx_buffer_unregister_symmetric(mapping_, size_);
        if (status) return status;

        registered_ = false;
        return 0;
    }

    void *addr() const { return mapping_; }

    void reset() {
        if (!registered_) return;

        int status = nvshmemx_buffer_unregister_symmetric(mapping_, size_);
        if (status) ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed during cleanup \n");
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

int check_collective_errors(uint64_t *errs) {
    int status =
        check_cuda_runtime(cudaMemcpyFromSymbol(errs, errs_d, sizeof(unsigned long long int), 0),
                           "cudaMemcpyFromSymbol");
    if (status) return status;

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
    // this should allocate at preferred address
    if (should_match_preferred_addr && (mapping.addr() != mmaped_addr)) {
        ERROR_PRINT("Could not allocate at preferred address %p %p\n", mapping.addr(), mmaped_addr);
        status = -1;
    }

    int unregister_status = mapping.unregister();
    if (unregister_status) {
        ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
        if (!status) status = unregister_status;
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
    if (status) return status;
    npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    dev_id = (mype_node / npes_per_gpu);
    status = check_cuda_runtime(cudaSetDevice(dev_id), "cudaSetDevice");
    if (status) return status;
    status = check_cuda_driver(cuDeviceGet(&my_dev, dev_id), "cuDeviceGet");
    if (status) return status;
    status = check_cuda_runtime(cudaDriverGetVersion(&cuda_drv_version), "cudaDriverGetVersion");
    if (status) return status;
    prop.type = MEM_ALLOC_TYPE;
    prop.location.type = MEM_ALLOC_LOCATION_TYPE;
    prop.location.id = dev_id;
    if (enable_egm) {
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        status = check_cuda_driver(
            cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev),
            "cuDeviceGetAttribute");
        if (status) return status;
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

    std::vector<UserBuffer> buffers(iter);

    if (!mype) DEBUG_PRINT("creating and mmapping %d buffers ", iter);
    for (unsigned int i = 0; i < iter; i++) {
        seed = i;
        lsize = rand_r(&seed) % (_max_size - _min_size + 1) + _min_size;
        lsize = ((lsize - 1) / granularity + 1) * granularity;

        UserBuffer &buffer = buffers[i];
        status = buffer.allocate(lsize, prop);
        if (status) return status;

        status = buffer.register_symmetric();
        if (status) {
            ERROR_PRINT("shmem_mmap failed \n");
            return status;
        }

        if (use_egm) {
            memset(buffer.mapped_ptr(), 0, lsize);
        } else {
            status = check_cuda_runtime(cudaMemset(buffer.mapped_ptr(), 0, lsize), "cudaMemset");
            if (status) return status;
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
        if (status) return status;
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
        if (status) return status;
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
        if (!mype)
            DEBUG_PRINT("freeing buffers with index %d to %d \n", free_start_idx, free_end_idx);
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

        if (!mype)
            DEBUG_PRINT("re-allocating buffers with index %d to %d \n", free_start_idx,
                        free_end_idx);
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
                    if (status) return status;
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
                    if (status) return status;
                }
            }
        }
        if (!mype) DEBUG_PRINT("done repetition %d \n", r);
    }

    fflush(stdout);

    if (!mype) DEBUG_PRINT("[binsize %d] done testing \n", b);

    return status;
}
