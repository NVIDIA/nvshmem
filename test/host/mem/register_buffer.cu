/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "cuda_runtime.h"
#include "utils.h"

#define BUFFER_COUNT 128

#define LARGE_BUFFER_SIZE 4294967296ULL
#define BUFFER_OFFSET_SIZE 3221225472ULL

struct buffer_registration_arg {
    int *buffer;
    int start_offset;
    int retval;
    int device_id;
};

__global__ void device_side_large_buffer_test(char *src, char *dest, int pe) {
    if (threadIdx.x == 0) {
        nvshmem_char_put(dest, src, 1, pe);
        nvshmem_barrier_all();
    }
}

__global__ void device_side_registered_host_buffer_test(int *src, int *dest, int buffer_count,
                                                        int stride, int pe) {
    int idx = threadIdx.x * stride;
    if (idx < buffer_count) {
        nvshmem_int32_put(&dest[idx], &src[idx], 1, pe);
    }
    return;
}

void *nvshmem_register_buffers_offset(void *in) {
    struct buffer_registration_arg *info = (struct buffer_registration_arg *)in;

    for (int i = info->start_offset; i < BUFFER_COUNT; i += 2) {
        info->retval = nvshmemx_buffer_register(&info->buffer[i], sizeof(int));
        if (info->retval) {
            break;
        }
    }
    return NULL;
}

void *nvshmem_register_buffers_offset_set_device(void *in) {
    struct buffer_registration_arg *info = (struct buffer_registration_arg *)in;
    CUDA_CHECK(cudaSetDevice(info->device_id));

    return nvshmem_register_buffers_offset(in);
}

void *nvshmem_unregister_buffers_offset(void *in) {
    struct buffer_registration_arg *info = (struct buffer_registration_arg *)in;

    for (int i = info->start_offset; i < BUFFER_COUNT; i += 2) {
        info->retval = nvshmemx_buffer_unregister(&info->buffer[i]);
        if (info->retval) {
            break;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    struct buffer_registration_arg reg_arg_local, reg_arg_thread;
    int *local_array = NULL;
    int *remote_array = NULL;
    int *remote_array_h = NULL;
    char *large_device_buffer = NULL;
    char *remote_target = NULL;
    char host_target_buffer;
    int status = 0;
    int mype, npes, mype_node, target_pe;
    int provided, requested;
    pthread_t registration_thread;
    int dev_count;

    requested = NVSHMEM_THREAD_SERIALIZED;
    nvshmem_init_thread(requested, &provided);
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int npes_per_gpu = (npes_node + dev_count - 1) / dev_count;
    CUDA_CHECK(cudaSetDevice(mype_node / npes_per_gpu));

    if (provided != requested) {
        status = -1;
        fprintf(stderr,
                "Unable to run this test. We require support for thread "
                "level %d, but nvshmem_init returned %d.\n",
                NVSHMEM_THREAD_SERIALIZED, provided);
        goto out;
    }

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    nvshmem_barrier_all();

    remote_array = (int *)nvshmem_malloc(sizeof(int) * npes * BUFFER_COUNT);
    if (!remote_array) {
        status = -1;
        fprintf(stderr, "Unable to allocate nvshmem memory.\n");
        goto out;
    }
    local_array = (int *)calloc(BUFFER_COUNT, sizeof(int));
    if (!local_array) {
        status = -1;
        fprintf(stderr, "Unable to allocate system heap memory.\n");
        goto out;
    }
    remote_array_h = (int *)calloc(npes * BUFFER_COUNT, sizeof(int));
    if (!remote_array_h) {
        status = -1;
        fprintf(stderr, "Unable to allocate nvshmem memory.\n");
        goto out;
    }

    /* set local memory value to */
    for (int j = 0; j < BUFFER_COUNT; j++) {
        local_array[j] = mype + (1000 * j);
    }

    reg_arg_thread.buffer = local_array;
    reg_arg_thread.start_offset = 0;
    reg_arg_thread.retval = 0;
    reg_arg_thread.device_id = mype_node / npes_per_gpu;

    reg_arg_local.buffer = local_array;
    reg_arg_local.start_offset = 1;
    reg_arg_local.retval = 0;
    reg_arg_local.device_id = mype_node / npes_per_gpu;

    /* Testing the thread safety of our registration function. */
    pthread_create(&registration_thread, NULL, nvshmem_register_buffers_offset_set_device,
                   &reg_arg_thread);
    nvshmem_register_buffers_offset(&reg_arg_local);
    pthread_join(registration_thread, NULL);
    if (reg_arg_local.retval != 0 || reg_arg_thread.retval != 0) {
        fprintf(stderr,
                "Unable to register memory with nvshmem. error_local: %d error_thread: %d\n",
                reg_arg_local.retval, reg_arg_thread.retval);
        goto out;
    }

    nvshmem_barrier_all();
    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }

        for (int j = 0; j < BUFFER_COUNT; j++) {
            nvshmem_putmem((void *)&remote_array[mype * BUFFER_COUNT + j], (void *)&local_array[j],
                           sizeof(int), i);
        }
    }

    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(remote_array_h, remote_array, (sizeof(int) * BUFFER_COUNT * npes),
                          cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();

    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }

        for (int j = 0; j < BUFFER_COUNT; j++) {
            if (remote_array_h[i * BUFFER_COUNT + j] != i + 1000 * j) {
                fprintf(stderr, "Got a buffer value mismatch. expected %d, got %d.\n", i + 1000 * j,
                        remote_array_h[i * BUFFER_COUNT + j]);
                status = -1;
            }
        }
    }

    nvshmem_unregister_buffers_offset(&reg_arg_local);
    if (reg_arg_local.retval != 0) {
        fprintf(stderr, "Unable to unregister the buffers.\n");
    }

    /* Testing that we only unregistered the odd memory buffers and we can still do I/O to the even
     * ones. */
    memset(local_array, 0, sizeof(int) * BUFFER_COUNT);
    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }
        for (int j = 0; j < BUFFER_COUNT; j += 2) {
            nvshmem_int32_put(&remote_array[mype * BUFFER_COUNT + j], &local_array[j], 1, i);
        }
    }

    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(remote_array_h, remote_array, (sizeof(int) * BUFFER_COUNT * npes),
                          cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();

    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }
        for (int j = 0; j < BUFFER_COUNT; j += 2) {
            if (remote_array_h[i * BUFFER_COUNT + j] != 0) {
                fprintf(stderr, "Got a buffer value mismatch. expected %d, got %d.\n", 0,
                        remote_array_h[i * BUFFER_COUNT + j]);
                status = -1;
            }
        }
    }

    for (int j = 0; j < BUFFER_COUNT; j++) {
        local_array[j] = mype + (1000 * j);
    }

    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }
        device_side_registered_host_buffer_test<<<1, 64>>>(
            local_array, &remote_array[mype * BUFFER_COUNT], BUFFER_COUNT, 2, i);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(remote_array_h, remote_array, (sizeof(int) * BUFFER_COUNT * npes),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < npes; i++) {
        if (i == mype) {
            continue;
        }
        for (int j = 0; j < BUFFER_COUNT; j += 2) {
            if (remote_array_h[i * BUFFER_COUNT + j] != i + 1000 * j) {
                fprintf(stderr, "Got a buffer value mismatch. expected %d, got %d.\n", i + 1000 * j,
                        remote_array_h[i * BUFFER_COUNT + j]);
                status = -1;
            }
        }
    }

    CUDA_CHECK(cudaMalloc((void **)&large_device_buffer, LARGE_BUFFER_SIZE));
    status = nvshmemx_buffer_register(large_device_buffer, LARGE_BUFFER_SIZE);
    if (status != NVSHMEMX_SUCCESS) {
        goto out;
    }

    CUDA_CHECK(cudaMemset((void *)large_device_buffer, 'b', sizeof(char)));
    CUDA_CHECK(cudaMemset((void *)(large_device_buffer + BUFFER_OFFSET_SIZE), 'd', sizeof(char)));

    remote_target = (char *)nvshmem_calloc(1, sizeof(char));
    if (!remote_target) {
        status = -1;
        fprintf(stderr, "Unable to allocate memory for remote buffer.\n");
        goto out;
    }

    target_pe = (mype + 1) % npes;
    nvshmem_char_put(remote_target, large_device_buffer, 1, target_pe);
    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(&host_target_buffer, remote_target, 1, cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();
    if (host_target_buffer != 'b') {
        fprintf(stderr, "Received an invalid buffer handle. Got %d. Expected %d.\n",
                host_target_buffer, 'b');
        status = -1;
        goto out;
    }
    nvshmem_barrier_all();
    nvshmem_char_put(remote_target, (large_device_buffer + BUFFER_OFFSET_SIZE), 1, target_pe);
    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(&host_target_buffer, remote_target, 1, cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();
    if (host_target_buffer != 'd') {
        fprintf(stderr, "Received an invalid buffer handle. Got %d. Expected %d.\n",
                host_target_buffer, 'd');
        status = -1;
        goto out;
    }

    nvshmem_barrier_all();
    device_side_large_buffer_test<<<1, 1>>>(large_device_buffer, remote_target, target_pe);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&host_target_buffer, remote_target, 1, cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();
    if (host_target_buffer != 'b') {
        fprintf(stderr, "Received an invalid buffer handle. Got %d. Expected %d.\n",
                host_target_buffer, 'b');
        status = -1;
        goto out;
    }
    nvshmem_barrier_all();
    device_side_large_buffer_test<<<1, 1>>>((large_device_buffer + BUFFER_OFFSET_SIZE),
                                            remote_target, target_pe);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&host_target_buffer, remote_target, 1, cudaMemcpyDeviceToHost));
    nvshmem_barrier_all();
    if (host_target_buffer != 'd') {
        fprintf(stderr, "Received an invalid buffer handle. Got %d. Expected %d.\n",
                host_target_buffer, 'd');
        status = -1;
        goto out;
    }

out:
    if (large_device_buffer) {
        cudaFree(large_device_buffer);
    }

    if (remote_target) {
        nvshmem_free(remote_target);
    }

    if (remote_array) {
        nvshmem_free(remote_array);
    }

    if (remote_array_h) {
        free(remote_array_h);
    }

    if (local_array) {
        free(local_array);
    }

    nvshmem_finalize();
    return status;
}
