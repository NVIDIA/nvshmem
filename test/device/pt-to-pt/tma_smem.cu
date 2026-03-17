/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

/*
 * Test: TMA Shared Memory Management APIs
 *
 * Validates that nvshmemx_ask_smem and nvshmemx_give_smem work correctly,
 * and that a put operation succeeds after giving shared memory to NVSHMEM.
 */

#include <stdio.h>
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#define NUM_ELEMS 256

#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[PE %d][%s:%d] CUDA error: %s\n", nvshmem_my_pe(),   \
                    __FILE__, __LINE__, cudaGetErrorString(result));               \
        }                                                                         \
    } while (0)

/* Test 1: Verify nvshmemx_ask_smem returns valid sizes */
__global__ void test_ask_smem(int *results) {
    if (threadIdx.x == 0) {
        results[0] = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        results[1] = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
        results[2] = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    }
}

/* Test 2: Give smem and perform a put */
__global__ void test_give_smem_and_put(int *send_data, int *recv_data, int num_elems, int mype,
                                       int npes) {
    extern __shared__ char nvshmem_smem[];

    if (threadIdx.x == 0) {
        nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elems) {
        send_data[tid] = mype * 100 + tid;
    }
    __syncthreads();

    int peer = (mype + 1) % npes;

    if (threadIdx.x == 0) {
        printf("[PE %d][kernel] putting %d elems to peer %d, send_data=%p recv_data=%p\n",
               mype, num_elems, peer, send_data, recv_data);
        printf("[PE %d][kernel] send_data[0]=%d send_data[1]=%d send_data[255]=%d\n",
               mype, send_data[0], send_data[1], send_data[num_elems - 1]);
    }
    __syncthreads();

    nvshmemx_int_put_nbi_block(recv_data, send_data, num_elems, peer);
    nvshmem_quiet();

    if (threadIdx.x == 0) {
        printf("[PE %d][kernel] put+quiet done\n", mype);
    }
}

/* Test 2b: Same as test 2 but without give_smem (control test) */
__global__ void test_put_no_give_smem(int *send_data, int *recv_data, int num_elems, int mype,
                                      int npes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elems) {
        send_data[tid] = mype * 100 + tid;
    }
    __syncthreads();

    int peer = (mype + 1) % npes;
    nvshmemx_int_put_nbi_block(recv_data, send_data, num_elems, peer);
    nvshmem_quiet();
}

/* Test 3: Multiple blocks each give their own smem */
__global__ void test_multiblock_give_smem(int *send_data, int *recv_data, int elems_per_block,
                                          int mype, int npes) {
    extern __shared__ char nvshmem_smem[];

    if (threadIdx.x == 0) {
        nvshmemx_give_smem(nvshmem_smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM));
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    send_data[tid] = mype * 1000 + tid;
    __syncthreads();

    int peer = (mype + 1) % npes;
    int offset = blockIdx.x * blockDim.x;
    nvshmemx_int_put_nbi_block(recv_data + offset, send_data + offset, blockDim.x, peer);
    nvshmem_quiet();
}

int main(int argc, char **argv) {
    int status = 0;
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    printf("[PE %d] npes=%d\n", mype, npes);

    /* ============ Test 1: ask_smem returns valid sizes ============ */
    {
        /* Host-side test */
        int host_recommended = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        int host_minimum = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
        int host_barriers = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);

        if (host_recommended <= 0 || host_minimum <= 0 || host_barriers <= 0) {
            printf("[PE %d] FAIL: ask_smem returned non-positive value on host\n", mype);
            status = 1;
            goto out;
        }
        if (host_recommended < host_minimum || host_minimum < host_barriers) {
            printf("[PE %d] FAIL: ask_smem size ordering violated: recommended=%d minimum=%d "
                   "barriers=%d\n",
                   mype, host_recommended, host_minimum, host_barriers);
            status = 1;
            goto out;
        }

        /* Device-side test */
        int *d_results;
        cudaMalloc(&d_results, 3 * sizeof(int));
        test_ask_smem<<<1, 1>>>(d_results);
        cudaDeviceSynchronize();

        int h_results[3];
        cudaMemcpy(h_results, d_results, 3 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_results);

        if (h_results[0] != host_recommended || h_results[1] != host_minimum ||
            h_results[2] != host_barriers) {
            printf("[PE %d] FAIL: ask_smem host/device mismatch: "
                   "host=(%d,%d,%d) device=(%d,%d,%d)\n",
                   mype, host_recommended, host_minimum, host_barriers, h_results[0], h_results[1],
                   h_results[2]);
            status = 1;
            goto out;
        }
        printf("[PE %d] PASS: ask_smem (recommended=%d, minimum=%d, barriers=%d)\n", mype,
               host_recommended, host_minimum, host_barriers);
    }

    /* ============ Test 2a: control - put WITHOUT give_smem, no dynamic smem ============ */
    {
        int *send_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        assert(send_data && recv_data);
        cudaMemset(recv_data, 0, sizeof(int) * NUM_ELEMS);

        printf("[PE %d] Test 2a: control put (no smem), send=%p recv=%p\n", mype, send_data,
               recv_data);
        nvshmem_barrier_all();

        test_put_no_give_smem<<<1, NUM_ELEMS>>>(send_data, recv_data, NUM_ELEMS, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        int *host = new int[NUM_ELEMS];
        cudaMemcpy(host, recv_data, sizeof(int) * NUM_ELEMS, cudaMemcpyDeviceToHost);
        int prev_pe = (mype - 1 + npes) % npes;
        bool pass = true;
        for (int i = 0; i < NUM_ELEMS; i++) {
            int expected = prev_pe * 100 + i;
            if (host[i] != expected) {
                printf("[PE %d] FAIL: control put at %d: got %d, expected %d\n", mype, i, host[i],
                       expected);
                pass = false;
                break;
            }
        }
        if (pass)
            printf("[PE %d] PASS: control put (no smem)\n", mype);
        else
            status = 1;

        delete[] host;
        nvshmem_free(send_data);
        nvshmem_free(recv_data);
    }

    /* ============ Test 2b: give_smem + single-block put ============ */
    {
        int *send_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        assert(send_data && recv_data);
        cudaMemset(recv_data, 0, sizeof(int) * NUM_ELEMS);

        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        printf("[PE %d] Test 2b: give_smem put, smem_size=%d, send=%p recv=%p\n", mype, smem_size,
               send_data, recv_data);

        if (smem_size > 48 * 1024) {
            cudaError_t attr_err = cudaFuncSetAttribute(
                test_give_smem_and_put, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
            printf("[PE %d] cudaFuncSetAttribute(%d) = %s\n", mype, smem_size,
                   cudaGetErrorString(attr_err));
        }
        nvshmem_barrier_all();

        test_give_smem_and_put<<<1, NUM_ELEMS, smem_size>>>(send_data, recv_data, NUM_ELEMS, mype,
                                                            npes);
        cudaError_t launch_err = cudaGetLastError();
        printf("[PE %d] kernel launch = %s\n", mype, cudaGetErrorString(launch_err));
        cudaError_t sync_err = cudaDeviceSynchronize();
        printf("[PE %d] cudaDeviceSynchronize = %s\n", mype, cudaGetErrorString(sync_err));
        nvshmem_barrier_all();

        int *host = new int[NUM_ELEMS];
        cudaMemcpy(host, recv_data, sizeof(int) * NUM_ELEMS, cudaMemcpyDeviceToHost);
        int prev_pe = (mype - 1 + npes) % npes;
        bool pass = true;
        int num_wrong = 0;
        for (int i = 0; i < NUM_ELEMS; i++) {
            int expected = prev_pe * 100 + i;
            if (host[i] != expected) {
                if (num_wrong < 5) {
                    printf("[PE %d] FAIL: single-block put at %d: got %d, expected %d\n", mype, i,
                           host[i], expected);
                }
                num_wrong++;
                pass = false;
            }
        }
        if (!pass)
            printf("[PE %d] total wrong: %d / %d\n", mype, num_wrong, NUM_ELEMS);
        if (pass)
            printf("[PE %d] PASS: give_smem + single-block put\n", mype);
        else
            status = 1;

        delete[] host;
        nvshmem_free(send_data);
        nvshmem_free(recv_data);
    }

    /* ============ Test 3: give_smem + multi-block put ============ */
    {
        int num_blocks = 4;
        int threads_per_block = 64;
        int total_elems = num_blocks * threads_per_block;

        int *send_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        int *recv_data = (int *)nvshmem_malloc(sizeof(int) * total_elems);
        assert(send_data && recv_data);
        cudaMemset(recv_data, 0, sizeof(int) * total_elems);

        int smem_size = nvshmemx_ask_smem(NVSHMEMX_SMEM_MINIMUM);
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(test_multiblock_give_smem,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        nvshmem_barrier_all();

        test_multiblock_give_smem<<<num_blocks, threads_per_block, smem_size>>>(
            send_data, recv_data, threads_per_block, mype, npes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        int *host = new int[total_elems];
        cudaMemcpy(host, recv_data, sizeof(int) * total_elems, cudaMemcpyDeviceToHost);
        int prev_pe = (mype - 1 + npes) % npes;
        bool pass = true;
        for (int i = 0; i < total_elems; i++) {
            int expected = prev_pe * 1000 + i;
            if (host[i] != expected) {
                printf("[PE %d] FAIL: multi-block put at %d: got %d, expected %d\n", mype, i,
                       host[i], expected);
                pass = false;
                break;
            }
        }
        if (pass)
            printf("[PE %d] PASS: give_smem + multi-block put\n", mype);
        else
            status = 1;

        delete[] host;
        nvshmem_free(send_data);
        nvshmem_free(recv_data);
    }

out:
    finalize_wrapper();
    return status;
}
