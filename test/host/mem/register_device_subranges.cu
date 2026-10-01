/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "cuda_runtime.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

static unsigned char pattern(size_t index, int pe) {
    /* A period distinct from the page size makes page-sized shifts visible. */
    return static_cast<unsigned char>(1 + (index + 37 * static_cast<size_t>(pe)) % 251);
}

static int count_fds() {
    DIR *dir = opendir("/proc/self/fd");
    if (dir == NULL) {
        return -1;
    }
    int count = 0;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] != '.') {
            count++;
        }
    }
    closedir(dir);
    return count;
}

int main(int argc, char **argv) {
    int status = EXIT_SUCCESS;
    init_wrapper(&argc, &argv);
    const int mype = nvshmem_my_pe();
    const int npes = nvshmem_n_pes();
    if (npes < 2) {
        ERROR_EXIT("This test requires at least two PEs.\n");
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        ERROR_EXIT("Unable to query host page size.\n");
    }
    const size_t page = static_cast<size_t>(page_size);
    const size_t capacity = 4 * page;
    const size_t offsets[] = {0, 1, page - 1, page, page + 17};
    const size_t lengths[] = {1, page - 1, page + 37};
    char *allocation = NULL;
    CUDA_CHECK(cudaMalloc(&allocation, capacity + page - 1));
    const uintptr_t address = reinterpret_cast<uintptr_t>(allocation);
    const size_t padding = (page - address % page) % page;
    char *source = allocation + padding;
    if (reinterpret_cast<uintptr_t>(source) % page != 0) {
        ERROR_EXIT("Device subrange base is not page aligned.\n");
    }
    char *target = static_cast<char *>(nvshmem_malloc(capacity));
    if (target == NULL) {
        ERROR_EXIT("Unable to allocate symmetric target.\n");
    }
    std::vector<unsigned char> source_data(capacity);
    for (size_t i = 0; i < capacity; i++) {
        source_data[i] = pattern(i, mype);
    }
    CUDA_CHECK(cudaMemcpy(source, source_data.data(), capacity, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned char> received(capacity);
    const int previous_pe = (mype + npes - 1) % npes;

    int baseline_fds = -1;
    /* Warm up every range before measuring: providers may lazily create cache resources. */
    for (int round = 0; round < 4; round++) {
        for (size_t offset : offsets) {
            for (size_t length : lengths) {
                if (nvshmemx_buffer_register(source + offset, length) != NVSHMEMX_SUCCESS) {
                    ERROR_EXIT("Unable to register device subrange.\n");
                }
                CUDA_CHECK(cudaMemset(target, 0, capacity));
                CUDA_CHECK(cudaDeviceSynchronize());
                nvshmem_barrier_all();
                nvshmem_putmem(target + offset, source + offset, length, (mype + 1) % npes);
                nvshmem_quiet();
                nvshmem_barrier_all();
                CUDA_CHECK(cudaMemcpy(received.data(), target, capacity, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < capacity; i++) {
                    const unsigned char value =
                        i >= offset && i - offset < length ? pattern(i, previous_pe) : 0;
                    if (received[i] != value) {
                        ERROR_PRINT("PE %d: Device subrange transfer corrupted data.\n", mype);
                        status = EXIT_FAILURE;
                        break;
                    }
                }
                if (nvshmemx_buffer_unregister(source + offset) != NVSHMEMX_SUCCESS) {
                    ERROR_EXIT("Unable to unregister device subrange.\n");
                }
                nvshmem_barrier_all();
            }
        }
        const int open_fds = count_fds();
        if (open_fds < 0) {
            ERROR_PRINT("PE %d: Unable to count open file descriptors.\n", mype);
            status = EXIT_FAILURE;
        } else if (baseline_fds < 0) {
            baseline_fds = open_fds;
        } else if (open_fds != baseline_fds) {
            ERROR_PRINT("PE %d: File descriptor count changed after registration cycles.\n", mype);
            status = EXIT_FAILURE;
        }
    }
    if (status == EXIT_SUCCESS) {
        printf("PE %d: PASS device subranges (60 registrations, open fds stable at %d)\n", mype,
               baseline_fds);
    }
    CUDA_CHECK(cudaFree(allocation));
    nvshmem_free(target);
    finalize_wrapper();
    return status;
}
