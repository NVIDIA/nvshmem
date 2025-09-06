/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

/* This example performs user buffer registration and
 * unmap using NVSHMEM buffer register/unregister API. The API requires user
 * buffer to be VMM based or EGM based
 */

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"

#define GRANULARITY 536870912UL
#define COLL_NELEMS 4096

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#undef CU_CHECK
#define CU_CHECK(stmt)                                                                  \
    do {                                                                                \
        CUresult result = (stmt);                                                       \
        const char *str;                                                                \
        if (CUDA_SUCCESS != result) {                                                   \
            CUresult ret = cuGetErrorString(result, &str);                              \
            if (ret == CUDA_ERROR_INVALID_VALUE) str = "Unknown error";                 \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, str); \
            exit(-1);                                                                   \
        }                                                                               \
        assert(CUDA_SUCCESS == result);                                                 \
    } while (0)

__global__ void init_data_kernel(float *source, size_t nelems) {
    for (int i = 0; i < nelems; ++i) {
        source[i] = (float)i;
    }
}

void *createUserBuffer(size_t size, CUmemAllocationProp &prop) {
    void *bufAddr = nullptr;

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

int main(int argc, char **argv) {
    nvshmem_init();
    int status = 0;
    int mype, npes;
    int npes_node, mype_node;
    const size_t size = GRANULARITY;
    void *buffer;
    void *mmaped_buffer;
    CUmemAllocationProp prop = {};
    int dev_id;
    float *source, *dest, *dest_h;
    size_t nelems;
    cudaStream_t stream;
    nvshmem_team_t team = NVSHMEM_TEAM_WORLD;

    mype = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    npes = nvshmem_n_pes();
    dev_id = mype_node % npes_node;
    CUDA_CHECK(cudaSetDevice(dev_id));

    if (!mype) printf("creating and mmapping buffer of size: %lu\n", size);
    // Allocation of user buffer is local
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev_id;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    prop.requestedHandleTypes =
        (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

    buffer = createUserBuffer(size, prop);
    if (!buffer) {
        fprintf(stderr, "Failed to create user buffer \n");
        status = 1;
        goto out;
    }
    mmaped_buffer = (void *)nvshmemx_buffer_register_symmetric(buffer, size, 0);
    if (!mmaped_buffer) {
        fprintf(stderr, "shmem_mmap failed \n");
        status = 1;
        goto out;
    }
    CUDA_CHECK(cudaMemset(mmaped_buffer, 0, size));

    // test heap usage to verify mmap correctness
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    nelems = COLL_NELEMS;
    nelems = nelems / 2;  // split the buffer into source and dest

    source = (float *)mmaped_buffer;
    dest = (float *)(mmaped_buffer) + nelems;
    dest_h = (float *)malloc(nelems * sizeof(float));

    init_data_kernel<<<1, 1, 0, stream>>>((float *)source, nelems);
    nvshmemx_barrier_on_stream(team, stream);
    nvshmemx_float_sum_reduce_on_stream(team, (float *)dest, (const float *)source, nelems, stream);
    cudaStreamSynchronize(stream);

    CUDA_CHECK(cudaMemcpy(dest_h, dest, nelems * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nelems; i++) {
        if (dest_h[i] != (float)i * npes) {
            printf("PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, dest_h[i], i,
                   (float)i * npes);
            status = -1;
        }
    }
    if (!status) {
        fprintf(stderr, "No errors found\n");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // free all buffers
    status = nvshmemx_buffer_unregister_symmetric(mmaped_buffer, size);
    if (status) {
        fprintf(stderr, "nvshmemx_buffer_unregister_symmetric failed\n");
    }
    free(dest_h);
    nvshmem_finalize();
    releaseUserBuf(buffer, size);
out:
    return status;
}
