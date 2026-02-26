/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See License.txt for license information
 */

#include "perf_utils.h"
#include <stdlib.h>

double *d_latency = NULL;
double *d_avg_time = NULL;
double *latency = NULL;
double *avg_time = NULL;
__device__ int clockrate;

void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table) {
    void **tables;
    int i, dev_property;
    int dev_count;

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(
        cudaDeviceGetAttribute(&dev_property, cudaDevAttrUnifiedAddressing, mype_node % dev_count));
    assert(dev_property == 1);

    assert(num_tables >= 1);
    assert(num_entries_per_table >= 1);
    CUDA_CHECK(cudaHostAlloc(table_mem, num_tables * sizeof(void *), cudaHostAllocMapped));
    tables = *table_mem;

    /* Just allocate an array of 8 byte values. The user can decide if they want to use double or
     * uint64_t */
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(
            cudaHostAlloc(&tables[i], num_entries_per_table * sizeof(double), cudaHostAllocMapped));
        memset(tables[i], 0, num_entries_per_table * sizeof(double));
    }
}

void free_tables(void **tables, int num_tables) {
    int i;
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaFreeHost(tables[i]));
    }
    CUDA_CHECK(cudaFreeHost(tables));
}

void print_table_v1(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double *value, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    /* Used for automated test output. It outputs the data in a non human-friendly format. */
    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, value[i], plus_minus, units);
            }
        }
    } else {
        printf("+------------------------+----------------------+\n");
        printf("| %-22s | %-20s |\n", job_name, subjob_name);
        printf("+------------------------+----------------------+\n");
        printf("| %-22s | %10s %-9s |\n", var_name, output_var, units);
        printf("+------------------------+----------------------+\n");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("| %-22.1lu | %-20.6lf |\n", size[i], value[i]);
                printf("+------------------------+----------------------+\n");
            }
        }
    }
    printf("\n\n");
}
