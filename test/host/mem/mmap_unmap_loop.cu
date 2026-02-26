/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <unordered_map>
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

void *createUserBuffer(size_t size, CUmemAllocationProp &prop) {
    void *bufAddr;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle userAllocHandle;

    CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));
    CU_CHECK(
        cuMemSetAccess((CUdeviceptr)bufAddr, size, (const CUmemAccessDesc *)&accessDescriptor, 1));
    return bufAddr;
}

void releaseUserBuf(void *ptr, size_t size) {
    CUmemGenericAllocationHandle memHandle;
    CU_CHECK(cuMemRetainAllocationHandle(&memHandle, ptr));
    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(memHandle));
}

int test_allocation_at_preferred_address(void *user_buf, size_t size, void *mmaped_addr, bool should_match_preferred_addr) {
    void *mmaped_buf_pref;
    // register with preferred address
    mmaped_buf_pref = (void *)nvshmemx_buffer_register_symmetric_at_preferred_address(
        user_buf, size, mmaped_addr, 0);
    if (!mmaped_buf_pref) {
        ERROR_PRINT("shmem_mmap of user buffer at preferred address failed \n");
        return -1;
    }
    // this should allocate at preferred address
    if (should_match_preferred_addr && (mmaped_buf_pref != mmaped_addr)) {
        ERROR_PRINT("Could not allocate at preferred address %p %p\n", mmaped_buf_pref, mmaped_addr);
        return -1;
    }
    if (nvshmemx_buffer_unregister_symmetric(mmaped_buf_pref, size)) {
        ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    typedef DATATYPE_T dtype_t;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);
    uint64_t errs;
    int status = 0;
    int mype;
    size_t size;
    void **buffer;
    char size_string[100];
    CUmemAllocationProp prop = {};
    int dev_count, dev_id, npes_per_gpu;
    std::unordered_map<void *, void *> mmaped_buf;
    std::unordered_map<void *, size_t> bufSize;
    size_t granularity = GRAN;
    unsigned int iter = 0;
    uint32_t lsize;
    unsigned int seed = 0;
    bool enable_egm = false;
    int numa_id;
    CUdevice my_dev;
    int cuda_drv_version;
    int bufId;
    dtype_t *source, *dest;
    size_t nelems;
    enable_egm = use_egm;
    cudaStream_t stream;
    nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
    size = ((_max_size - 1) / (GRAN) + 1) * (GRAN);
    size = _min_iters * size;
    sprintf(size_string, "%zu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        ERROR_PRINT("setenv failed \n");
        goto out;
    }

    srand(1);
    mype = nvshmem_my_pe();
    iter = _min_iters;
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    dev_id = (mype_node / npes_per_gpu);
    CUDA_CHECK(cudaSetDevice(dev_id));
    CU_CHECK(cuDeviceGet(&my_dev, dev_id));
    CUDA_CHECK(cudaDriverGetVersion(&cuda_drv_version));
    prop.type = MEM_ALLOC_TYPE;
    prop.location.type = MEM_ALLOC_LOCATION_TYPE;
    prop.location.id = dev_id;
    if (enable_egm) {
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        CU_CHECK(cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev));
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

    buffer = (void **)calloc(iter, sizeof(void *));
    if (!buffer) {
        ERROR_PRINT("malloc failed \n");
        goto out;
    }
    int free_start_idx, free_end_idx;
    if (!mype) DEBUG_PRINT("creating and mmapping %d buffers ", iter);
    for (unsigned int i = 0; i < iter; i++) {
        seed = i;
        lsize = rand_r(&seed) % (_max_size - _min_size + 1) + _min_size;
        lsize = ((lsize - 1) / granularity + 1) * granularity;
        buffer[i] = createUserBuffer(lsize, prop);

        mmaped_buf[buffer[i]] = (void *)nvshmemx_buffer_register_symmetric(buffer[i], lsize, 0);
        if (!mmaped_buf[buffer[i]]) {
            ERROR_PRINT("shmem_mmap failed \n");
            goto out;
        }
        bufSize[buffer[i]] = lsize;
        if (use_egm) {
            memset(mmaped_buf[buffer[i]], 0, lsize);
        } else {
            CUDA_CHECK(cudaMemset(mmaped_buf[buffer[i]], 0, lsize));
        }
    }

    // test allocation when preferred address is unavailable
    // allocation should succeed but should not be at preferred address
    if (iter > 2) {
        status = test_allocation_at_preferred_address(buffer[0], bufSize[buffer[0]], mmaped_buf[buffer[1]],
                                                      false);
        if (status) {
            ERROR_PRINT("test_allocation_at_preferred_address failed \n");
            goto out;
        }
    }

    for (size_t r = 0; r < _repeat; r++) {
        free_start_idx = 0;
        free_end_idx = iter - 1;

        // test heap usage to verify mmap correctness
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bufId = r % iter;
        // Limiting the size of collective to limit test run time
        if (COLL_NELEMS * sizeof(dtype_t) > bufSize[buffer[bufId]]) {
            nelems = bufSize[buffer[bufId]] / sizeof(dtype_t);
        } else {
            nelems = COLL_NELEMS;
        }
        nelems = nelems / 2;  // split the buffer into source and dest

        source = (dtype_t *)mmaped_buf[buffer[bufId]];
        dest = (dtype_t *)(mmaped_buf[buffer[bufId]]) + nelems;

        DO_REDUCE_ON_STREAM_TEST(int, dtype_t, sum);
        DO_REDUCE_TEST(int, dtype_t, sum);
        COLL_CHECK_ERRS_D();
        fflush(stdout);
        nvshmem_barrier_all();

        // check if preferred allocation within unmapped region works
        if (iter > 2) {
            // create a hole by unmapping buf id =0,1
            for (int i = 0; i < 2; i++) {
                status =
                    nvshmemx_buffer_unregister_symmetric(mmaped_buf[buffer[i]], bufSize[buffer[i]]);
                if (status) {
                    ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
                    goto out;
                }
            }

            // register with buf id 1 as preferred offset
            status = test_allocation_at_preferred_address(buffer[1], bufSize[buffer[1]], mmaped_buf[buffer[1]], true);
            if (status) {
                ERROR_PRINT("test_allocation_at_preferred_address within unmapped region failed \n");
                goto out;
            }

            // restore buf 0,1
            for (int i = 0; i < 2; i++) {
                mmaped_buf[buffer[i]] =
                    (void *)nvshmemx_buffer_register_symmetric(buffer[i], bufSize[buffer[i]], 0);
                if (!mmaped_buf[buffer[i]]) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    goto out;
                }
            }
        }
        if (!mype)
            DEBUG_PRINT("freeing buffers with index %d to %d \n", free_start_idx, free_end_idx);
        for (int i = free_start_idx; i <= free_end_idx; i++) {
            status =
                nvshmemx_buffer_unregister_symmetric(mmaped_buf[buffer[i]], bufSize[buffer[i]]);
            if (status) {
                ERROR_PRINT("nvshmemx_buffer_unregister_symmetric failed \n");
                goto out;
            }
        }

        // check if preferred allocation in empty heap region
        if (iter > 2) {
            status =
                test_allocation_at_preferred_address(buffer[1], bufSize[buffer[1]], mmaped_buf[buffer[1]], true);
            if (status) {
                ERROR_PRINT("test_allocation_at_preferred_address failed \n");
                goto out;
            }
        }

        if (!mype)
            DEBUG_PRINT("re-allocating buffers with index %d to %d \n", free_start_idx,
                        free_end_idx);
        if (r % 2) {
            for (int i = free_end_idx; i >= free_start_idx; i--) {
                mmaped_buf[buffer[i]] =
                    (void *)nvshmemx_buffer_register_symmetric(buffer[i], bufSize[buffer[i]], 0);
                if (!mmaped_buf[buffer[i]]) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    goto out;
                }
                if (use_egm) {
                    memset(mmaped_buf[buffer[i]], 0, bufSize[buffer[i]]);
                } else {
                    CUDA_CHECK(cudaMemset(mmaped_buf[buffer[i]], 0, bufSize[buffer[i]]));
                }
            }
        } else {
            for (int i = free_start_idx; i <= free_end_idx; i++) {
                mmaped_buf[buffer[i]] =
                    (char *)nvshmemx_buffer_register_symmetric(buffer[i], bufSize[buffer[i]], 0);
                if (!mmaped_buf[buffer[i]]) {
                    ERROR_PRINT("shmem_mmap failed \n");
                    goto out;
                }
                if (use_egm) {
                    memset(mmaped_buf[buffer[i]], 0, bufSize[buffer[i]]);
                } else {
                    CUDA_CHECK(cudaMemset(mmaped_buf[buffer[i]], 0, bufSize[buffer[i]]));
                }
            }
        }
        if (!mype) DEBUG_PRINT("done repetition %d \n", r);
    }

    // free all buffers
    for (unsigned int i = 0; i < iter; i++) {
        nvshmemx_buffer_unregister_symmetric(mmaped_buf[buffer[i]], bufSize[buffer[i]]);
        mmaped_buf.erase(buffer[i]);
        releaseUserBuf(buffer[i], bufSize[buffer[i]]);
        bufSize.erase(buffer[i]);
    }
    fflush(stdout);

    free(buffer);
    if (!mype) DEBUG_PRINT("[binsize %d] done testing \n", b);

    finalize_wrapper();
out:
    return status;
}
