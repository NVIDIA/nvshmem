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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <getopt.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include "utils.h"

int bw(void *data_d, void *data_d_local, int sizeBytes, int pe, int iter, int skip,
       putget_issue_t iss, dir_t dir, cudaStream_t strm, cudaEvent_t sev, cudaEvent_t eev,
       float *ms, float *us) {
    int status = 0;
    int peer = !pe;
    struct timeval start, stop;

    if (iss.type == ON_STREAM) {
        if (dir.type == WRITE) {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) {
                    nvshmemx_quiet_on_stream(strm);
                    CUDA_CHECK(cudaStreamSynchronize(strm));
                    CUDA_CHECK(cudaEventRecord(sev, strm));
                }
                nvshmemx_putmem_nbi_on_stream((void *)data_d, (void *)data_d_local, sizeBytes, peer,
                                              strm);
            }
        } else {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) {
                    nvshmemx_quiet_on_stream(strm);
                    CUDA_CHECK(cudaStreamSynchronize(strm));
                    CUDA_CHECK(cudaEventRecord(sev, strm));
                }
                nvshmemx_getmem_nbi_on_stream((void *)data_d_local, (void *)data_d, sizeBytes, peer,
                                              strm);
            }
        }
        nvshmemx_quiet_on_stream(strm);
        CUDA_CHECK(cudaStreamSynchronize(strm));
        CUDA_CHECK(cudaEventRecord(eev, strm));
        CUDA_CHECK(cudaEventSynchronize(eev));
        CUDA_CHECK(cudaEventElapsedTime(ms, sev, eev));
    } else {
        if (dir.type == WRITE) {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) {
                    nvshmem_quiet();
                    gettimeofday(&start, NULL);
                }
                nvshmem_putmem_nbi((void *)data_d, (void *)data_d_local, sizeBytes, peer);
            }
        } else {
            for (int i = 0; i < (iter + skip); i++) {
                if (i == skip) {
                    nvshmem_quiet();
                    gettimeofday(&start, NULL);
                }
                nvshmem_getmem_nbi((void *)data_d_local, (void *)data_d, sizeBytes, peer);
            }
        }
        nvshmem_quiet();
        gettimeofday(&stop, NULL);
        *us = (stop.tv_usec - start.tv_usec) + (stop.tv_sec - start.tv_sec) * 1000000;
    }

    return status;
}

int main(int argc, char *argv[]) {
    int status = 0;
    int mype, npes;
    char *data_d = NULL, *data_d_local = NULL;

    read_args(argc, argv);

    int iter = iters;
    int skip = warmup_iters;
    uint64_t *size_array = NULL;
    double *bandwidth_array = NULL;
    int num_entries;
    int i;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        status = -1;
        goto finalize;
    }

    num_entries = floor(log2((float)max_size)) - floor(log2((float)min_size)) + 1;
    size_array = (uint64_t *)calloc(sizeof(uint64_t), num_entries);
    if (!size_array) {
        status = -1;
        goto finalize;
    }

    bandwidth_array = (double *)calloc(sizeof(double), num_entries);
    if (!bandwidth_array) {
        status = -1;
        goto finalize;
    }
    if (use_mmap) {
        data_d = (char *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        data_d_local = (char *)allocate_mmap_buffer(max_size, mem_handle_type, use_egm, true);
        DEBUG_PRINT("Allocated mmap buffer\n");
    } else {
        data_d = (char *)nvshmem_malloc(max_size);
        data_d_local = (char *)nvshmem_malloc(max_size);
        DEBUG_PRINT("Allocated nvshmem malloc buffer\n");
        CUDA_CHECK(cudaMemset(data_d, 0, max_size));
        CUDA_CHECK(cudaMemset(data_d_local, 0, max_size));
    }

    cudaStream_t strm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        float ms, us;
        cudaEvent_t sev, eev;
        CUDA_CHECK(cudaEventCreate(&sev));
        CUDA_CHECK(cudaEventCreate(&eev));
        i = 0;
        for (int size = min_size; size <= max_size; size *= step_factor) {
            size_array[i] = size;
            bw(data_d, data_d_local, size, mype, iter, skip, putget_issue, dir, strm, sev, eev, &ms,
               &us);

            if (putget_issue.type == ON_STREAM) {
                bandwidth_array[i] = ((float)iter * (float)size) / ((ms / 1000) * B_TO_GB);
            } else {
                bandwidth_array[i] = ((float)iter * (float)size) / ((us / 1000000) * B_TO_GB);
            }
            i++;
        }

        print_table_basic("Bandwidth", "None", "size (Bytes)", "Bandwidth", "GB", '+', size_array,
                          bandwidth_array, i);
        CUDA_CHECK(cudaEventDestroy(sev));
        CUDA_CHECK(cudaEventDestroy(eev));

        nvshmem_barrier_all();
    } else {
        nvshmem_barrier_all();
    }

finalize:
    CUDA_CHECK(cudaStreamDestroy(strm));

    if (data_d) {
        if (use_mmap) {
            free_mmap_buffer(data_d);
        } else {
            nvshmem_free(data_d);
        }
    }
    if (size_array) free(size_array);
    if (bandwidth_array) free(bandwidth_array);

    if (data_d_local) {
        if (use_mmap) {
            free_mmap_buffer(data_d_local);
        } else {
            nvshmem_free(data_d_local);
        }
    }

    finalize_wrapper();

    return status;
}
