/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#undef CUDA_RUNTIME_CHECK
#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#if CUDART_VERSION < 12010
#define CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED 132
#endif

__device__ long errors_d, mc_teams_d;

__global__ void check_mc_ptr(void *v_h, void *v_d, void *arr_d, int narr) {
    errors_d = 0;
    mc_teams_d = 0;

    nvshmem_team_t *arr = (nvshmem_team_t *)arr_d;
    int me = nvshmem_my_pe();
    for (int i = 0; i < narr; i++) {
        void *mc_ptr = nvshmemx_mc_ptr(arr[i], v_d);
        if (mc_ptr == NULL) {
            printf("[%d] Device expected non-NULL mc ptr for %p\n", me, v_d);
            ++errors_d;
        } else {
            mc_teams_d++;
        }
    }

    for (int i = 0; i < narr; i++) {
        void *mc_ptr = (int *)nvshmemx_mc_ptr(arr[i], v_h);
        if (mc_ptr != NULL) {
            printf("[%d] Device expected NULL for %p\n", me, v_h);
            ++errors_d;
        } else {
            mc_teams_d++;
        }
    }

    return;
}

__global__ void check_mc_ptr_no_support(void *v_d) {
    errors_d = 0;
    mc_teams_d = 0;

    int me = nvshmem_my_pe();
    void *mc_ptr = nvshmemx_mc_ptr(NVSHMEM_TEAM_MC_SHARED, v_d);
    if (mc_ptr != NULL) {
        printf("[%d] Device expected NULL mc ptr for %p when NVLS is unavailable.\n", me, v_d);
        ++errors_d;
    } else {
        mc_teams_d++;
    }
    return;
}

static bool is_mc_platform(void) {
    CUdevice current_dev;
    int cuda_dev = -1, cuda_driver_version = -1, mc_support = 0;
    CUDA_RUNTIME_CHECK(cudaDriverGetVersion(&cuda_driver_version));
    if (cuda_driver_version < 12010) {
        return false;
    }

    CUDA_RUNTIME_CHECK(cudaGetDevice(&cuda_dev));
    CU_CHECK(cuDeviceGet(&current_dev, cuda_dev));
    CU_CHECK(cuDeviceGetAttribute(
        &mc_support, static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED),
        current_dev));

    return (mc_support != 0);
}

static bool nvls_disabled_by_env(void) {
    const char *disable_nvls = getenv("NVSHMEM_DISABLE_NVLS");
    return disable_nvls != NULL && strcmp(disable_nvls, "0") != 0;
}

int main(int argc, char **argv) {
    long errors_h = 0, errors = 0;
    long mc_teams_h = 0, mc_teams = 0;
    nvshmem_team_t *arr = nullptr;
    int team_count = 0;
    void *arr_d = NULL;

    read_args(argc, argv);
    init_wrapper(&argc, &argv);

    int me = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    void *v_d;
    int dev_id;
    if (use_mmap) {
        v_d = (int *)allocate_mmap_buffer(sizeof(int), _mem_handle_type, use_egm);
    } else {
        v_d = nvshmem_malloc(sizeof(int));
    }
    void *v_h = malloc(sizeof(int));

    // Skip for 1 PE
    if (npes == 1) {
        printf(
            "Only one processing element (PE) is available. Multicast test require min of 2 or "
            "more PEs\n");
        finalize_wrapper();
        return (0);
    }

    if (nvls_disabled_by_env() || !is_mc_platform()) {
        void *mc_ptr = nvshmemx_mc_ptr(NVSHMEM_TEAM_MC_SHARED, v_d);
        if (mc_ptr != NULL) {
            printf("[%d] Host expected NULL mc ptr for %p when NVLS is unavailable.\n", me, v_d);
            ++errors;
        }
        check_mc_ptr_no_support<<<1, 1>>>(v_d);

        CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

        nvshmem_barrier_all();
        goto out;
    }

    // conditionalize teams as per platform
    CUDA_CHECK(cudaGetDevice(&dev_id));
    if (!is_mnnvl_supported(dev_id)) {
        // NVLS enabled team = NVSHMEM_TEAM_NODE, NVSHMEM_TEAM_MC_SHARED on NVLD (single node)
        team_count = 2;
        arr = (nvshmem_team_t *)malloc(sizeof(nvshmem_team_t) * team_count);
        arr[0] = NVSHMEMX_TEAM_NODE;
        arr[1] = NVSHMEM_TEAM_MC_SHARED;
    } else {
        // NVLS enabled team = NVSHMEM_TEAM_MC_SHARED on NVLD (single & multi node)
        team_count = 1;
        arr = (nvshmem_team_t *)malloc(sizeof(nvshmem_team_t));
        arr[0] = NVSHMEM_TEAM_MC_SHARED;
    }

    for (int i = 0; i < team_count; i++) {
        void *mc_ptr = nvshmemx_mc_ptr(arr[i], v_d);
        if (mc_ptr == NULL) {
            printf("[%d] Host expected non-NULL mc ptr for %p\n", me, v_d);
            ++errors;
        } else {
            mc_teams++;
        }
    }

    for (int i = 0; i < team_count; i++) {
        void *mc_ptr = (int *)nvshmemx_mc_ptr(arr[i], v_h);
        if (mc_ptr != NULL) {
            printf("[%d] Host expected NULL for %p\n", me, v_h);
            ++errors;
        } else {
            mc_teams++;
        }
    }

    CUDA_CHECK(cudaMalloc((void **)&arr_d, sizeof(nvshmem_team_t) * team_count));
    CUDA_CHECK(cudaMemcpy(arr_d, arr, sizeof(nvshmem_team_t) * team_count, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    check_mc_ptr<<<1, 1>>>(v_h, v_d, arr_d, team_count);

    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

    nvshmem_barrier_all();

    CUDA_CHECK(
        cudaMemcpyFromSymbol(&mc_teams_h, mc_teams_d, sizeof(long), 0, cudaMemcpyDeviceToHost));

out:
    CUDA_CHECK(cudaMemcpyFromSymbol(&errors_h, errors_d, sizeof(long), 0, cudaMemcpyDeviceToHost));
    errors += errors_h;

    CUDA_CHECK(cudaFree(arr_d));
    finalize_wrapper();
    free(arr);
    return errors != 0 || (mc_teams != mc_teams_h);
}
