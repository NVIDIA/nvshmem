/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"

#if !defined(NVSHMEM_CFT_HANDLES_SUPPORT)
#error "putmem_signal_counted.cu must only be registered when CFT handles are enabled"
#endif

static constexpr int kThreads = 128;
static constexpr size_t kSharedBytes = 256;
static constexpr size_t kCountedWriteAlignment = 16;
static constexpr size_t kCountedCounterAlignment = 256;
static constexpr size_t kCountedCounterStride = kCountedCounterAlignment / sizeof(uint64_t);

enum class Source { Shared };

// A nonzero PE- and offset-dependent pattern exposes missing or misaddressed staging chunks.
__host__ __device__ static constexpr unsigned char payload_byte(int producer_pe, size_t offset) {
    constexpr size_t kProducerPatternStride = 67;
    constexpr size_t kOffsetPatternStride = 31;
    constexpr size_t kNonzeroByteValueCount = 255;
    // Coprime strides vary the pattern by PE and offset; adding one excludes zero.
    const size_t pattern =
        static_cast<size_t>(producer_pe) * kProducerPatternStride + offset * kOffsetPatternStride;
    return static_cast<unsigned char>(1 + pattern % kNonzeroByteValueCount);
}

__global__ void reset_load_counter(uint64_t *counter, uint64_t *observed) {
    nvshmemx_signal_counted_reset(counter);
    __syncthreads();
    observed[threadIdx.x] = nvshmemx_signal_counted_load(counter);
}

__global__ void initialize_payload(unsigned char *source, size_t bytes, int producer_pe) {
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        source[i] = payload_byte(producer_pe, i);
    }
}

__global__ void counted_put(unsigned char *destination, const unsigned char *source, size_t bytes,
                            uint64_t *counter, int pe, int *statuses, Source source_kind,
                            size_t donation) {
    extern __shared__ __align__(16) unsigned char smem[];
    if (donation) nvshmemx_give_smem(smem, donation);
    const void *put_source = source;
    if (source_kind == Source::Shared) {
        unsigned char *payload = smem + nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
        for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) payload[i] = source[i];
        put_source = payload;
    }
    __syncthreads();
    statuses[threadIdx.x] =
        nvshmemx_putmem_signal_counted_nbi_block(destination, put_source, bytes, counter, pe);
    __syncthreads();
    if (donation) nvshmemx_release_smem();
}

__global__ void wait_and_validate(const unsigned char *destination, size_t bytes,
                                  const uint64_t *counter, uint64_t expected_counter_value,
                                  int producer_pe, size_t source_offset, int *errors) {
    nvshmemx_signal_counted_wait_until(counter, expected_counter_value);
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        if (destination[i] != payload_byte(producer_pe, source_offset + i)) atomicAdd(errors, 1);
    }
}

__global__ void wait_and_validate_producers(const unsigned char *destination, size_t bytes,
                                            const uint64_t *counter,
                                            uint64_t expected_counter_value, int *errors) {
    nvshmemx_signal_counted_wait_until(counter, expected_counter_value);
    for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
        int producer_pe = static_cast<int>(i / 16 + 1);
        size_t source_offset = i % 16;
        if (destination[i] != payload_byte(producer_pe, source_offset)) atomicAdd(errors, 1);
    }
}

__global__ void ordinary_cft_put(unsigned char *destination, const unsigned char *source,
                                 size_t bytes, int pe) {
    extern __shared__ __align__(16) unsigned char smem[];
    nvshmemx_give_smem(smem, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED));
    nvshmemx_putmem_block(destination, source, bytes, pe);
    nvshmem_quiet();
    nvshmemx_release_smem();
}

enum invalid_case {
    INVALID_PE,
    INVALID_DESTINATION,
    INVALID_DEST_ALIGNMENT,
    INVALID_COUNTER_8B_ALIGNMENT,
    INVALID_COUNTER_CFT_ALIGNMENT,
    INVALID_SIZE,
    INVALID_SOURCE_SPACE,
    MISSING_DONATION
};

__global__ void invalid_counted_put(unsigned char *destination, const unsigned char *source,
                                    size_t bytes, uint64_t *counter, int npes, int which,
                                    int *statuses) {
    extern __shared__ __align__(16) unsigned char smem[];
    void *dst = destination;
    uint64_t *count = counter;
    size_t count_bytes = bytes;
    alignas(16) uint4 local_source = {};
    const void *put_source = source;
    int pe = 0;
    if (which == INVALID_PE) pe = npes;
    if (which == INVALID_DESTINATION) dst = statuses;
    if (which == INVALID_DEST_ALIGNMENT) dst = destination + 1;
    if (which == INVALID_COUNTER_8B_ALIGNMENT)
        count = reinterpret_cast<uint64_t *>(reinterpret_cast<unsigned char *>(counter) + 4);
    if (which == INVALID_COUNTER_CFT_ALIGNMENT)
        count = reinterpret_cast<uint64_t *>(reinterpret_cast<unsigned char *>(counter) + 8);
    if (which == INVALID_SIZE) count_bytes = bytes - 1;
    if (which == INVALID_SOURCE_SPACE) put_source = &local_source;
    if (which == MISSING_DONATION) put_source = smem;
    statuses[threadIdx.x] =
        nvshmemx_putmem_signal_counted_nbi_block(dst, put_source, count_bytes, count, pe);
}

static int check_uniform_status(int *statuses, int expected_status) {
    std::vector<int> host(kThreads);
    CUDA_CHECK(
        cudaMemcpy(host.data(), statuses, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < kThreads; i++) {
        if (host[i] != expected_status) {
            fprintf(stderr, "PE %d: status[%d]=%d, expected %d\n", nvshmem_my_pe(), i, host[i],
                    expected_status);
            return 1;
        }
    }
    return 0;
}

static int run_put_case(unsigned char *destination, unsigned char *source, uint64_t *counter,
                        int *statuses, int *errors, size_t bytes, int pe, Source source_kind,
                        int expected_status, uint64_t counter_value_before_put,
                        int expected_producer_pe, size_t expected_source_offset = 0) {
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(*errors)));
    CUDA_CHECK(cudaMemset(destination, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    size_t donation = (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    size_t dynamic_smem = donation + bytes;
    counted_put<<<1, kThreads, dynamic_smem>>>(destination, source, bytes, counter, pe, statuses,
                                               source_kind, donation);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int failed = check_uniform_status(statuses, expected_status);
    if (expected_status == NVSHMEMX_SUCCESS && failed == 0) {
        const uint64_t expected_counter_value = counter_value_before_put + bytes;
        wait_and_validate<<<1, kThreads>>>(destination, bytes, counter, expected_counter_value,
                                           expected_producer_pe, expected_source_offset, errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    int payload_errors = 0;
    CUDA_CHECK(cudaMemcpy(&payload_errors, errors, sizeof(payload_errors), cudaMemcpyDeviceToHost));
    return failed + payload_errors;
}

struct counted_test_context {
    int mype;
    int npes;
    int peer;
    int previous;
    unsigned char *destination;
    unsigned char *source;
    uint64_t *counters;
    int *case_failures;
    int *statuses;
    int *errors;
    int valid_status = NVSHMEMX_ERROR_NOT_SUPPORTED;
    bool counted_available = false;
};

static int report_case(const counted_test_context &test, const char *name, int local_failures) {
    CUDA_CHECK(cudaMemcpy(&test.case_failures[0], &local_failures, sizeof(local_failures),
                          cudaMemcpyHostToDevice));
    int status = nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, &test.case_failures[1],
                                        &test.case_failures[0], 1);
    if (status != 0) {
        if (test.mype == 0)
            fprintf(stderr, "putmem_signal_counted: result reduction failed: %d\n", status);
        exit(1);
    }
    int global_failures = 0;
    CUDA_CHECK(cudaMemcpy(&global_failures, &test.case_failures[1], sizeof(global_failures),
                          cudaMemcpyDeviceToHost));
    if (test.mype == 0) {
        if (global_failures == 0)
            printf("putmem_signal_counted: %s: SUCCESS\n", name);
        else
            printf("putmem_signal_counted: %s: FAILURE (%d errors)\n", name, global_failures);
        fflush(stdout);
    }
    return global_failures;
}

static int run_counter_reset_load_case(const counted_test_context &test, uint64_t *observed) {
    // Every participating thread must observe zero after resetting the counter.
    nvshmemx_signal_counted_reset(&test.counters[0]);
    nvshmemx_signal_counted_reset(&test.counters[kCountedCounterStride]);
    nvshmem_barrier_all();

    reset_load_counter<<<1, kThreads>>>(&test.counters[0], observed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> observed_host(kThreads);
    CUDA_CHECK(cudaMemcpy(observed_host.data(), observed, observed_host.size() * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    int failed = 0;
    for (uint64_t value : observed_host) failed += value != 0;
    return failed;
}

static int probe_counted_backend(counted_test_context *test, bool expect_not_supported) {
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int counted_ops_supported = 0;
#if CUDA_VERSION >= 13030
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device));
    CU_CHECK(cuDeviceGetAttribute(&counted_ops_supported,
                                  CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED,
                                  cu_device));
#endif
    test->valid_status = (expect_not_supported || properties.major < 10 || !counted_ops_supported)
                             ? NVSHMEMX_ERROR_NOT_SUPPORTED
                             : NVSHMEMX_SUCCESS;

    // Probe backend availability through the direct shared-memory source path.
    nvshmemx_signal_counted_reset(&test->counters[0]);
    CUDA_CHECK(cudaMemset(test->destination, 0, kCountedWriteAlignment));
    nvshmem_barrier_all();
    const size_t donation = nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY);
    counted_put<<<1, kThreads, donation + kCountedWriteAlignment>>>(
        test->destination, test->source, kCountedWriteAlignment, &test->counters[0], test->peer,
        test->statuses, Source::Shared, donation);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> probe_statuses(kThreads);
    CUDA_CHECK(cudaMemcpy(probe_statuses.data(), test->statuses,
                          probe_statuses.size() * sizeof(int), cudaMemcpyDeviceToHost));
    int backend_status = probe_statuses[0];
    test->counted_available = backend_status == NVSHMEMX_SUCCESS;
    int failed = check_uniform_status(test->statuses, test->valid_status);
    nvshmem_barrier_all();
    nvshmemx_signal_counted_reset(&test->counters[0]);
    nvshmem_barrier_all();
    return failed;
}

static int run_ring_cases(const counted_test_context &test) {
    int failed = 0;
    int case_failures = 0;
    // A shared source is sent directly to the next PE without global-source staging.
    nvshmemx_signal_counted_reset(&test.counters[0]);
    CUDA_CHECK(cudaMemset(test.destination, 0, 16));
    nvshmem_barrier_all();
    int isolation_failed =
        run_put_case(test.destination, test.source, &test.counters[0], test.statuses, test.errors,
                     16, test.peer, Source::Shared, test.valid_status, 0, test.previous);
    failed += report_case(test, "ring shared source (16 bytes)", isolation_failed);

    // A larger shared source still bypasses the internal global-source staging buffers.
    nvshmemx_signal_counted_reset(&test.counters[0]);
    nvshmem_barrier_all();
    case_failures =
        run_put_case(test.destination, test.source, &test.counters[0], test.statuses, test.errors,
                     kSharedBytes, test.peer, Source::Shared, test.valid_status, 0, test.previous);
    failed += report_case(test, "ring shared source (256 bytes)", case_failures);

    // Distinct 256-byte-aligned counters advance independently for separate payloads.
    nvshmemx_signal_counted_reset(&test.counters[0]);
    nvshmemx_signal_counted_reset(&test.counters[kCountedCounterStride]);
    nvshmem_barrier_all();
    case_failures =
        run_put_case(test.destination, test.source, &test.counters[0], test.statuses, test.errors,
                     16, test.peer, Source::Shared, test.valid_status, 0, test.previous);
    case_failures +=
        run_put_case(test.destination + 16, test.source + 16, &test.counters[kCountedCounterStride],
                     test.statuses, test.errors, 16, test.peer, Source::Shared, test.valid_status,
                     0, test.previous, 16);
    failed += report_case(test, "distinct counters", case_failures);

    // Consecutive puts to separate payload regions accumulate on the same counter.
    CUDA_CHECK(cudaMemset(test.destination, 0, 32));
    nvshmemx_signal_counted_reset(&test.counters[0]);
    nvshmem_barrier_all();
    case_failures =
        run_put_case(test.destination, test.source, &test.counters[0], test.statuses, test.errors,
                     16, test.peer, Source::Shared, test.valid_status, 0, test.previous);
    case_failures += run_put_case(test.destination + 16, test.source + 16, &test.counters[0],
                                  test.statuses, test.errors, 16, test.peer, Source::Shared,
                                  test.valid_status, 16, test.previous, 16);
    failed += report_case(test, "same counter, separate payloads", case_failures);

    if (test.counted_available) {
        // Counter epochs and waits remain correct across unsigned 64-bit wraparound.
        uint64_t counter_value_before_wrap = UINT64_MAX - 15;
        CUDA_CHECK(cudaMemcpy(&test.counters[0], &counter_value_before_wrap,
                              sizeof(counter_value_before_wrap), cudaMemcpyHostToDevice));
        nvshmem_barrier_all();
        case_failures = run_put_case(test.destination, test.source, &test.counters[0],
                                     test.statuses, test.errors, 32, test.peer, Source::Shared,
                                     NVSHMEMX_SUCCESS, counter_value_before_wrap, test.previous);
        failed += report_case(test, "counter wraparound", case_failures);
    }

    return failed;
}

static int run_fan_in_case(const counted_test_context &test) {
    if (!test.counted_available) return 0;

    // All producers write disjoint payloads and contribute to one receiver-local counter.
    int failed = 0;
    CUDA_CHECK(cudaMemset(test.errors, 0, sizeof(*test.errors)));
    size_t fan_in_bytes = (size_t)(test.npes - 1) * 16;
    nvshmemx_signal_counted_reset(&test.counters[0]);
    CUDA_CHECK(cudaMemset(test.destination, 0, fan_in_bytes));
    nvshmem_barrier_all();
    if (test.mype != 0) {
        size_t donation = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        counted_put<<<1, kThreads, donation>>>(test.destination + (size_t)(test.mype - 1) * 16,
                                               test.source, 16, &test.counters[0], 0, test.statuses,
                                               Source::Shared, donation);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        failed += check_uniform_status(test.statuses, NVSHMEMX_SUCCESS);
    }
    if (test.mype == 0) {
        wait_and_validate_producers<<<1, kThreads>>>(test.destination, fan_in_bytes,
                                                     &test.counters[0], fan_in_bytes, test.errors);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<unsigned char> producer_data(fan_in_bytes);
        CUDA_CHECK(cudaMemcpy(producer_data.data(), test.destination, producer_data.size(),
                              cudaMemcpyDeviceToHost));
        for (int producer = 1; producer < test.npes; producer++) {
            for (size_t byte = 0; byte < 16; byte++) {
                size_t offset = (size_t)(producer - 1) * 16 + byte;
                unsigned char expected = payload_byte(producer, byte);
                if (producer_data[offset] != expected) {
                    fprintf(stderr, "PE 0: fan-in producer %d byte %zu = %u, expected %u\n",
                            producer, byte, (unsigned int)producer_data[offset],
                            (unsigned int)expected);
                    break;
                }
            }
        }
    }
    nvshmem_barrier_all();
    int payload_errors = 0;
    CUDA_CHECK(
        cudaMemcpy(&payload_errors, test.errors, sizeof(payload_errors), cudaMemcpyDeviceToHost));
    failed += payload_errors;
    return report_case(test, "fan-in", failed);
}

static int run_invalid_argument_cases(const counted_test_context &test) {
    if (!test.counted_available) return 0;

    int failed = 0;
    int case_failures = 0;
    const std::array<const char *, 8> invalid_case_names = {
        {"invalid PE", "invalid destination", "misaligned destination",
         "counter not 8-byte aligned", "counter not 256-byte aligned", "invalid size",
         "invalid source address space", "missing shared-memory donation"}};
    CUDA_CHECK(cudaMemset(test.destination, 0, 64));
    nvshmemx_signal_counted_reset(&test.counters[0]);
    // Validate the documented argument and missing-registration status mappings.
    for (int which = INVALID_PE; which <= MISSING_DONATION; which++) {
        invalid_counted_put<<<1, kThreads, kCountedWriteAlignment>>>(
            test.destination, test.source, 16, &test.counters[0], test.npes, which, test.statuses);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int expected =
            which == MISSING_DONATION ? NVSHMEMX_ERROR_NOT_SUPPORTED : NVSHMEMX_ERROR_INVALID_VALUE;
        case_failures = check_uniform_status(test.statuses, expected);
        failed += report_case(test, invalid_case_names[which], case_failures);
    }
    // A registration smaller than the barriers-only region is also unsupported.
    size_t short_donation = (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY) - 16;
    counted_put<<<1, kThreads, short_donation + 2 * kCountedWriteAlignment>>>(
        test.destination, test.source, 16, &test.counters[0], 0, test.statuses, Source::Shared,
        short_donation);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    case_failures = check_uniform_status(test.statuses, NVSHMEMX_ERROR_NOT_SUPPORTED);
    failed += report_case(test, "undersized shared-memory donation", case_failures);
    nvshmem_barrier_all();

    // Rejected operations must leave the counter and destination untouched.
    uint64_t unchanged_counter = 1;
    CUDA_CHECK(cudaMemcpy(&unchanged_counter, &test.counters[0], sizeof(unchanged_counter),
                          cudaMemcpyDeviceToHost));
    case_failures = unchanged_counter != 0;
    std::vector<unsigned char> unchanged_destination(64);
    CUDA_CHECK(cudaMemcpy(unchanged_destination.data(), test.destination,
                          unchanged_destination.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : unchanged_destination) case_failures += value != 0;
    nvshmem_barrier_all();
    failed += report_case(test, "rejected operations preserve state", case_failures);
    return failed;
}

static int run_ordinary_cft_sanity_case(const counted_test_context &test) {
    // Counted-endpoint setup must not regress ordinary CFT puts.
    ordinary_cft_put<<<1, kThreads, nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED)>>>(
        test.destination, test.source, 16, test.peer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    unsigned char ordinary_value = 0;
    CUDA_CHECK(cudaMemcpy(&ordinary_value, test.destination, 1, cudaMemcpyDeviceToHost));
    return ordinary_value != payload_byte(test.previous, 0);
}

int main(int argc, char **argv) {
    bool expect_not_supported = getenv("NVSHMEMTEST_COUNTED_EXPECT_NOT_SUPPORTED") != nullptr;
    init_wrapper(&argc, &argv);

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    int previous = (mype + npes - 1) % npes;
    constexpr size_t max_bytes = size_t{1} << 16;
    unsigned char *destination = (unsigned char *)nvshmem_align(16, max_bytes * (size_t)npes);
    unsigned char *source = (unsigned char *)nvshmem_align(16, max_bytes);
    uint64_t *counters =
        (uint64_t *)nvshmem_align(kCountedCounterAlignment, 2 * kCountedCounterAlignment);
    int *case_failures = static_cast<int *>(nvshmem_malloc(2 * sizeof(int)));
    int *statuses = nullptr;
    uint64_t *observed = nullptr;
    int *errors = nullptr;
    CUDA_CHECK(cudaMalloc(&statuses, kThreads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&observed, kThreads * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&errors, sizeof(int)));
    counted_test_context test{mype,   npes,     peer,          previous, destination,
                              source, counters, case_failures, statuses, errors};

    int recommended_smem = nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
    if (recommended_smem > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(counted_put, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        recommended_smem));
        CUDA_CHECK(cudaFuncSetAttribute(
            ordinary_cft_put, cudaFuncAttributeMaxDynamicSharedMemorySize, recommended_smem));
    }
    CUDA_CHECK(cudaMemset(errors, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(destination, 0, max_bytes * (size_t)npes));
    initialize_payload<<<1, kThreads>>>(source, max_bytes, mype);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int failed = 0;
    failed += report_case(test, "counter reset/load", run_counter_reset_load_case(test, observed));
    failed += report_case(test, "backend capability probe",
                          probe_counted_backend(&test, expect_not_supported));
    if (npes < 2) {
        if (mype == 0)
            printf("putmem_signal_counted: remote cases: SKIPPED (requires at least 2 PEs)\n");
    } else {
        failed += run_ring_cases(test);
        failed += run_fan_in_case(test);
    }
    failed += run_invalid_argument_cases(test);
    failed += report_case(test, "ordinary CFT compatibility", run_ordinary_cft_sanity_case(test));

    if (mype == 0) {
        printf("putmem_signal_counted: overall: %s (%d errors)\n", failed ? "FAILURE" : "SUCCESS",
               failed);
        fflush(stdout);
    }

    cudaFree(errors);
    cudaFree(observed);
    cudaFree(statuses);
    nvshmem_free(case_failures);
    nvshmem_free(counters);
    nvshmem_free(source);
    nvshmem_free(destination);
    finalize_wrapper();
    return failed != 0;
}
