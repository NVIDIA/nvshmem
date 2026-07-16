/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COLL_TEST_H
#define COLL_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <assert.h>
#include <unistd.h>
#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>

#define MAX_SKIP 16
#define MAX_ITERS 128
#define BARRIER_MAX_ITERS 1000
#define FCOLLECT_MAX_ITERS 1024
#define BARRIER_MAX_SKIP 10

extern int coll_max_iters;

#define call_shmem_broadcast(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                              \
        nvshmem_##TYPENAME##_broadcast(team, d_dest, d_source, num_elems, root);      \
    } while (0)

#define call_shmem_fcollect(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_fcollect(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_alltoall(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root) \
    do {                                                                             \
        nvshmem_##TYPENAME##_alltoall(team, d_dest, d_source, num_elems);            \
    } while (0)

#define call_shmem_broadcast_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                       stream)                                                  \
    do {                                                                                        \
        nvshmemx_##TYPENAME##_broadcast_on_stream(team, d_dest, d_source, num_elems, root,      \
                                                  stream);                                      \
    } while (0)

#define call_shmem_fcollect_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_fcollect_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

#define call_shmem_alltoall_on_stream(TYPENAME, TYPE, team, d_dest, d_source, num_elems, root, \
                                      stream)                                                  \
    do {                                                                                       \
        nvshmemx_##TYPENAME##_alltoall_on_stream(team, d_dest, d_source, num_elems, stream);   \
    } while (0)

#define RUN_COLL_ON_STREAM(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, npes, \
                           root, stream, size_array, latency_array)                              \
    do {                                                                                         \
        if (use_graph) {                                                                         \
            RUN_COLL_ON_STREAM_GRAPH(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest,     \
                                     h_dest, npes, root, stream, size_array, latency_array);     \
        } else {                                                                                 \
            RUN_COLL_ON_STREAM_STREAM(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest,    \
                                      h_dest, npes, root, stream, size_array, latency_array);    \
        }                                                                                        \
    } while (0)

/* */
#define RUN_COLL_ON_STREAM_STREAM(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, \
                                  npes, root, stream, size_array, latency_array)                  \
    do {                                                                                          \
        int array_index = 0;                                                                      \
        size_t min_elems, max_elems;                                                              \
        if (strcmp(#coll, "broadcast") == 0) {                                                    \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                     \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                     \
        } else {                                                                                  \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));            \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));            \
        }                                                                                         \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {    \
            float latency = 0;                                                                    \
            cudaEvent_t t_start, t_stop;                                                          \
            CUDA_CHECK(cudaEventCreate(&t_start));                                                \
            CUDA_CHECK(cudaEventCreate(&t_stop));                                                 \
            auto lat_idx_array = latency_array[array_index];                                      \
            nvshmemx_barrier_all_on_stream(stream);                                               \
            for (int iter = 0; iter < warmup_iters; iter++) {                                     \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,         \
                                              d_source, num_elems, root, stream);                 \
            }                                                                                     \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
            nvshmemx_barrier_all_on_stream(stream);                                               \
            for (size_t repetition = 0; repetition < repetitions; repetition++) {                 \
                double repetition_sum = 0.0;                                                      \
                for (size_t iter = 0; iter < iters; iter++) {                                     \
                    CUDA_CHECK(cudaEventRecord(t_start, stream));                                 \
                    call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,     \
                                                  d_source, num_elems, root, stream);             \
                    CUDA_CHECK(cudaEventRecord(t_stop, stream));                                  \
                    CUDA_CHECK(cudaStreamSynchronize(stream));                                    \
                    CUDA_CHECK(cudaEventElapsedTime(&latency, t_start, t_stop));                  \
                    lat_idx_array[iter] = latency * 1e+3;                                         \
                    repetition_sum += lat_idx_array[iter];                                        \
                }                                                                                 \
                perf_stats_add(latency_stats[array_index], repetition_sum / (double)iters);       \
            }                                                                                     \
            CUDA_CHECK(cudaEventDestroy(t_start));                                                \
            CUDA_CHECK(cudaEventDestroy(t_stop));                                                 \
            const char *op_name = #coll;                                                          \
            size_array[array_index] =                                                             \
                calculate_collective_size(op_name, num_elems, sizeof(TYPE), npes);                \
            array_index++;                                                                        \
        }                                                                                         \
    } while (0)

#define RUN_COLL_ON_STREAM_GRAPH(coll, COLL, TYPENAME, TYPE, d_source, h_source, d_dest, h_dest, \
                                 npes, root, stream, size_array, latency_array)                  \
    do {                                                                                         \
        int array_index = 0;                                                                     \
        cudaEvent_t t_start, t_stop;                                                             \
        CUDA_CHECK(cudaEventCreate(&t_start));                                                   \
        CUDA_CHECK(cudaEventCreate(&t_stop));                                                    \
        size_t min_elems, max_elems;                                                             \
        if (strcmp(#coll, "broadcast") == 0) {                                                   \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                    \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                    \
        } else {                                                                                 \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));           \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));           \
        }                                                                                        \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {   \
            float ms = 0.0f;                                                                     \
            auto lat_idx_array = latency_array[array_index];                                     \
            for (size_t iter = 0; iter < warmup_iters; iter++)                                   \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
            nvshmemx_barrier_all_on_stream(stream);                                              \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            /* Start graph capture */                                                            \
            cudaGraph_t graph;                                                                   \
            cudaGraphExec_t graph_instance;                                                      \
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));             \
            for (size_t iter = 0; iter < iters; iter++) {                                        \
                call_shmem_##coll##_on_stream(TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, d_dest,        \
                                              d_source, num_elems, root, stream);                \
            }                                                                                    \
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));                                    \
            CUDA_CHECK(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));             \
            nvshmemx_barrier_all_on_stream(stream);                                              \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
                                                                                                 \
            for (size_t repetition = 0; repetition < repetitions; repetition++) {                \
                CUDA_CHECK(cudaEventRecord(t_start, stream));                                    \
                CUDA_CHECK(cudaGraphLaunch(graph_instance, stream));                             \
                CUDA_CHECK(cudaEventRecord(t_stop, stream));                                     \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                       \
                CUDA_CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));                          \
                lat_idx_array[0] = (ms * 1e+3) / (double)iters;                                  \
                perf_stats_add(latency_stats[array_index], lat_idx_array[0]);                    \
            }                                                                                    \
            /* Cleanup */                                                                        \
            CUDA_CHECK(cudaGraphExecDestroy(graph_instance));                                    \
            CUDA_CHECK(cudaGraphDestroy(graph));                                                 \
            const char *op_name = #coll;                                                         \
            size_array[array_index] =                                                            \
                calculate_collective_size(op_name, num_elems, sizeof(TYPE), npes);               \
            array_index++;                                                                       \
        }                                                                                        \
    } while (0)

#define RUN_RDXN(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array, latency_array, \
                 stream)                                                                      \
    do {                                                                                      \
        if (use_graph) {                                                                      \
            RUN_RDXN_GRAPH(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array,      \
                           latency_array, stream);                                            \
        } else {                                                                              \
            RUN_RDXN_STREAM(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array,     \
                            latency_array, stream);                                           \
        }                                                                                     \
    } while (0)

#define RUN_RDXN_GRAPH(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array,             \
                       latency_array, stream)                                                    \
    do {                                                                                         \
        cudaEvent_t start_event, stop_event;                                                     \
        CUDA_CHECK(cudaEventCreate(&start_event));                                               \
        CUDA_CHECK(cudaEventCreate(&stop_event));                                                \
        float ms = 0.0f;                                                                         \
        if (strcmp(#coll, "reduce") == 0) {                                                      \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                    \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                    \
        } else {                                                                                 \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));           \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));           \
        }                                                                                        \
        int idx = 0;                                                                             \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {   \
            for (size_t iter = 0; iter < warmup_iters; iter++)                                   \
                nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                                 \
                    team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);            \
            nvshmemx_barrier_all_on_stream(stream);                                              \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            cudaGraph_t graph;                                                                   \
            cudaGraphExec_t graph_instance;                                                      \
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));             \
            for (size_t iter = 0; iter < iters; iter++) {                                        \
                nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                                 \
                    team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);            \
            }                                                                                    \
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));                                    \
            CUDA_CHECK(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));             \
            nvshmemx_barrier_all_on_stream(stream);                                              \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            for (size_t repetition = 0; repetition < repetitions; repetition++) {                \
                CUDA_CHECK(cudaEventRecord(start_event, stream));                                \
                CUDA_CHECK(cudaGraphLaunch(graph_instance, stream));                             \
                CUDA_CHECK(cudaEventRecord(stop_event, stream));                                 \
                nvshmemx_barrier_all_on_stream(stream);                                          \
                CUDA_CHECK(cudaStreamSynchronize(stream));                                       \
                CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));                  \
                latency_array[idx][0] = (ms * 1e+3) / (double)iters;                             \
                perf_stats_add(latency_stats[idx], latency_array[idx][0]);                       \
            }                                                                                    \
            /* Cleanup */                                                                        \
            CUDA_CHECK(cudaGraphExecDestroy(graph_instance));                                    \
            CUDA_CHECK(cudaGraphDestroy(graph));                                                 \
            const char *op_name = #coll;                                                         \
            size_array[idx] = calculate_collective_size(op_name, num_elems, sizeof(TYPE), npes); \
            idx++;                                                                               \
        }                                                                                        \
        CUDA_CHECK(cudaEventDestroy(start_event));                                               \
        CUDA_CHECK(cudaEventDestroy(stop_event));                                                \
    } while (0)

#define RUN_RDXN_STREAM(coll, TYPENAME, TYPE, OP, team, d_source, d_dest, size_array,            \
                        latency_array, stream)                                                   \
    do {                                                                                         \
        cudaEvent_t start_event, stop_event;                                                     \
        CUDA_CHECK(cudaEventCreate(&start_event));                                               \
        CUDA_CHECK(cudaEventCreate(&stop_event));                                                \
        float ms = 0.0f;                                                                         \
        if (strcmp(#coll, "reduce") == 0) {                                                      \
            min_elems = max(static_cast<size_t>(1), min_size / sizeof(TYPE));                    \
            max_elems = max(static_cast<size_t>(1), max_size / sizeof(TYPE));                    \
        } else {                                                                                 \
            min_elems = max(static_cast<size_t>(1), min_size / (npes * sizeof(TYPE)));           \
            max_elems = max(static_cast<size_t>(1), max_size / (npes * sizeof(TYPE)));           \
        }                                                                                        \
        int idx = 0;                                                                             \
        for (size_t num_elems = min_elems; num_elems <= max_elems; num_elems *= step_factor) {   \
            nvshmemx_barrier_all_on_stream(stream);                                              \
            for (size_t iter = 0; iter < warmup_iters; iter++)                                   \
                nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                                 \
                    team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);            \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                           \
            for (size_t repetition = 0; repetition < repetitions; repetition++) {                \
                double repetition_sum = 0.0;                                                     \
                for (size_t iter = 0; iter < iters; iter++) {                                    \
                    CUDA_CHECK(cudaEventRecord(start_event, stream));                            \
                    nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                             \
                        team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems, stream);        \
                    CUDA_CHECK(cudaEventRecord(stop_event, stream));                             \
                    CUDA_CHECK(cudaStreamSynchronize(stream));                                   \
                    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));              \
                    latency_array[idx][iter] = ms * 1e+3;                                        \
                    repetition_sum += latency_array[idx][iter];                                  \
                }                                                                                \
                perf_stats_add(latency_stats[idx], repetition_sum / (double)iters);              \
            }                                                                                    \
            const char *op_name = #coll;                                                         \
            size_array[idx] = calculate_collective_size(op_name, num_elems, sizeof(TYPE), npes); \
            idx++;                                                                               \
        }                                                                                        \
        CUDA_CHECK(cudaEventDestroy(start_event));                                               \
        CUDA_CHECK(cudaEventDestroy(stop_event));                                                \
    } while (0)

#define RUN_RDXN_BITWISE_DATATYPE(coll, TYPENAME, TYPE, team, d_source, d_dest, num_elems, stream, \
                                  size_array, latency_array)                                       \
    switch (reduce_op.type) {                                                                      \
        case NVSHMEM_SUM:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, sum, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MIN:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, min, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MAX:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, max, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_PROD:                                                                         \
            RUN_RDXN(coll, TYPENAME, TYPE, prod, team, d_source, d_dest, size_array,               \
                     latency_array, stream);                                                       \
            break;                                                                                 \
        case NVSHMEM_AND:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, and, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_OR:                                                                           \
            RUN_RDXN(coll, TYPENAME, TYPE, or, team, d_source, d_dest, size_array, latency_array,  \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_XOR:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, xor, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        default:                                                                                   \
            printf("Invalid reduction operation\n");                                               \
            break;                                                                                 \
    }

#define RUN_RDXN_DATATYPE(coll, TYPENAME, TYPE, team, d_source, d_dest, num_elems, stream,         \
                          size_array, latency_array)                                               \
    switch (reduce_op.type) {                                                                      \
        case NVSHMEM_SUM:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, sum, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MIN:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, min, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_MAX:                                                                          \
            RUN_RDXN(coll, TYPENAME, TYPE, max, team, d_source, d_dest, size_array, latency_array, \
                     stream);                                                                      \
            break;                                                                                 \
        case NVSHMEM_PROD:                                                                         \
            RUN_RDXN(coll, TYPENAME, TYPE, prod, team, d_source, d_dest, size_array,               \
                     latency_array, stream);                                                       \
            break;                                                                                 \
        default:                                                                                   \
            printf("Invalid reduction operation\n");                                               \
            break;                                                                                 \
    }
#endif /*COLL_TEST_H*/
