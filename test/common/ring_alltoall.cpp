/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <sys/types.h>
#include <unistd.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "data_check.h"
#include "utils.h"
#include "ring_alltoall.h"

#define _RING_ 1
#define _ALL_TO_ALL_ 1
#define MiB 1048576LU

template <typename T>
extern void launch_alltoall(void *src, void *dest, size_t len, int mype, int npes,
                            cudaStream_t cstrm);
template <typename T>
extern void launch_ring(void *src, void *dest, size_t len, int nextpe, int prevpe,
                        cudaStream_t cstrm);

#define MALLOC_ALIGNMENT ((size_t)512U)
const size_t alignbytes = MALLOC_ALIGNMENT;
int dev_count;
void *src_d = NULL, *dest_d = NULL;
cudaDeviceProp prop;
int iter = ITER;
size_t max_msg_size = MAX_MSG_SIZE;
int disp;
bool local_cuda_malloc;
bool dest_is_local;
cudaStream_t strm;

int setup(bool is_scalar, int ldisp, size_t max_size, uint64_t max_iter, bool local_dest, int *argc,
          char ***argv) {
    int status = 0;
    size_t size;
    iter = max_iter;
    max_msg_size = max_size;
    const char *scalar_test_num_iter = getenv("NVSHMEMTEST_SCALAR_NUM_ITERS");
    const char *vector_test_use_whole_mem = getenv("NVSHMEMTEST_USE_FULL_MEM");
    const char *vector_test_num_iter = getenv("NVSHMEMTEST_VECTOR_NUM_ITERS");
    const char *local_buffer = getenv("NVSHMEMTEST_LOCAL_CUDA_BUFFER");
    const char *heap_size = getenv("NVSHMEM_SYMMETRIC_SIZE");

    init_wrapper(argc, argv);  // Initializes and selects the GPU device

    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes = nvshmem_n_pes();
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    disp = ldisp;
    CUDA_CHECK(cudaStreamCreateWithFlags(&strm, cudaStreamNonBlocking));

    /* use cuda or nvshmem memory for local cuda allocations. */
    dest_is_local = local_dest;
    if (local_buffer != NULL && !strncasecmp(local_buffer, "1", 2)) {
        DEBUG_PRINT("Using CUDA memory for local buffer.\n");
        local_cuda_malloc = true;
    } else {
        local_cuda_malloc = false;
    }

    if (is_scalar) {
        if (scalar_test_num_iter != NULL) {
            iter = atoi(scalar_test_num_iter);
            if (iter < 0) {
                iter = ITER;
            }
        }
    } else {
        if (vector_test_num_iter != NULL) {
            iter = atoi(vector_test_num_iter);
            if (iter < 0) {
                iter = ITER;
            }
        }
        if (vector_test_use_whole_mem != NULL && !strncasecmp(vector_test_use_whole_mem, "1", 2) &&
            heap_size != NULL) {
            size = atoll(heap_size);

            if (size < (MiB * 16)) {
                fprintf(stderr, "Heap memory is too small to run this test.\n");
                goto out;
            }
            max_msg_size = size - (2 * alignbytes);
            max_msg_size = max_msg_size / (2 * disp * iter * npes);
            assert(max_msg_size > (max_msg_size % alignbytes));
            max_msg_size -= (max_msg_size % alignbytes);
        }
    }

    DEBUG_PRINT("setup %zu\n", max_msg_size * disp * iter * npes);
    if (local_cuda_malloc == true) {
        if (dest_is_local) {
            CUDA_CHECK(cudaMalloc(&dest_d, max_msg_size * disp * iter * npes));
            if (use_mmap) {
                src_d = allocate_mmap_buffer(max_msg_size * disp * iter * npes, _mem_handle_type,
                                             use_egm);
                DEBUG_PRINT("Allocating mmaped buffer for src_d\n");
            } else {
                src_d = nvshmem_malloc(max_msg_size * disp * iter * npes);
            }
        } else {
            CUDA_CHECK(cudaMalloc(&src_d, max_msg_size * disp * iter * npes));
            if (use_mmap) {
                dest_d = allocate_mmap_buffer(max_msg_size * disp * iter * npes, _mem_handle_type,
                                              use_egm);
                DEBUG_PRINT("Allocating mmaped buffer for dest_d\n");
            } else {
                dest_d = nvshmem_malloc(max_msg_size * disp * iter * npes);
            }
        }
    } else {
        if (use_mmap) {
            src_d =
                allocate_mmap_buffer(max_msg_size * disp * iter * npes, _mem_handle_type, use_egm);
            dest_d =
                allocate_mmap_buffer(max_msg_size * disp * iter * npes, _mem_handle_type, use_egm);
            DEBUG_PRINT("Allocating mmaped buffer for src_d and dest_d\n");
        } else {
            src_d = nvshmem_malloc(max_msg_size * disp * iter * npes);
            dest_d = nvshmem_malloc(max_msg_size * disp * iter * npes);
        }
    }
    if (use_egm) {
        memset(src_d, 0, max_msg_size * disp * iter * npes);
        memset(dest_d, 0, max_msg_size * disp * iter * npes);
    } else {
        CUDA_CHECK(cudaMemset(src_d, 0, max_msg_size * disp * iter * npes));
        CUDA_CHECK(cudaMemset(dest_d, 0, max_msg_size * disp * iter * npes));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    nvshmem_barrier_all();

    if (mype == 0) {
        DEBUG_PRINT("size \t ring-validation \t alltoall-validation \n");
        fflush(stdout);
    }

out:
    return status;
}

template <typename T>
int test(launch_alltoall_ptr_t launch_alltoall, launch_ring_ptr_t launch_ring) {
    int status = 0;
    size_t size;
    int nextpe, prevpe, i;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    for (size = sizeof(T); size <= max_msg_size; size *= 2) {
#ifdef _RING_
        nvshmem_barrier_all();

        // check ring
        status =
            init_data_ring<T>((T *)src_d, size, disp, iter, mype, npes, &nextpe, &prevpe, 0, strm);
        if (status) {
            ERROR_PRINT("data initialization failed for size %zu \n", size);
            status = -1;
            goto out;
        }

        nvshmem_barrier_all();

        DEBUG_PRINT(
            "[%d] launch_ring starting dest next %d prev %d size %zu max_msg_size %zu iter %d\n",
            mype, nextpe, prevpe, size, max_msg_size, iter);
        for (i = 0; i < iter; i++) {
            launch_ring((T *)((char *)src_d + size * disp * i),
                        (T *)((char *)dest_d + size * disp * i), size / sizeof(T), nextpe, prevpe,
                        strm);

            CUDA_CHECK(cudaGetLastError());
            if (cudaStreamSynchronize(strm)) {
                ERROR_EXIT("[%d] cudaStreamSynchronize failed \n", mype);
            }
        }
        DEBUG_PRINT("[%d] launch_ring returned \n", mype);
        nvshmem_barrier_all();

        status = check_data_ring<T>((T *)dest_d, strm);
        if (status) {
            ERROR_PRINT("[%d] data validation failed for size %zu \n", mype, size);
            status = -1;
            goto out;
        }
        if (!mype) DEBUG_PRINT("%zu \t\t ring passed \t\t ", size);
#endif
#ifdef _ALL_TO_ALL_
        nvshmem_barrier_all();

        if (!mype) DEBUG_PRINT("%zu \t\t passed \t\t ", size);

        // check alltoall
        status = init_data_alltoall<T>((T *)src_d, size, disp, iter, mype, npes, 0, strm);
        if (status) {
            ERROR_PRINT("data initialization failed for size %zu \n", size);
            status = -1;
            goto out;
        }

        nvshmem_barrier_all();

        DEBUG_PRINT("[%d] launch_alltoall starting \n", mype);
        for (int i = 0; i < iter; i++) {
            launch_alltoall((T *)((char *)src_d + size * disp * npes * i),
                            (T *)((char *)dest_d + size * disp * npes * i), size / sizeof(T), mype,
                            npes, strm);

            CUDA_CHECK(cudaGetLastError());
            if (cudaStreamSynchronize(strm)) {
                ERROR_EXIT("[%d] cudaStreamSynchronize failed \n", mype);
            }
        }
        DEBUG_PRINT("[%d] launch_alltoall returned \n", mype);

        nvshmem_barrier_all();

        status = check_data_alltoall<T>((T *)dest_d, strm);
        if (status) {
            ERROR_PRINT("[%d] data validation failed for size %zu \n", mype, size);
            status = -1;
            goto out;
        }

        nvshmem_barrier_all();
#endif
        if (!mype) fprintf(stderr, "passed %ld bytes\n", size);
    }

out:
    return status;
}

template int test<char>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<unsigned char>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<short>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<unsigned short>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<int>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<unsigned int>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<long long int>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<unsigned long long int>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<float>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<double>(launch_alltoall_ptr_t, launch_ring_ptr_t);
template int test<uint64_t>(launch_alltoall_ptr_t, launch_ring_ptr_t);

void cleanup() {
#ifdef _NVSHMEM_DEBUG
    int mype = nvshmem_my_pe();
#endif
    DEBUG_PRINT("[%d] cleanup \n", mype);
    if (src_d) {
        if (local_cuda_malloc && (!dest_is_local)) {
            cudaFree(src_d);
        } else {
            if (use_mmap) {
                free_mmap_buffer(src_d);
            } else {
                nvshmem_free(src_d);
            }
        }
    }
    if (dest_d) {
        if (local_cuda_malloc && dest_is_local) {
            cudaFree(dest_d);
        } else {
            if (use_mmap) {
                free_mmap_buffer(dest_d);
            } else {
                nvshmem_free(dest_d);
            }
        }
    }

    finalize_wrapper();
}
