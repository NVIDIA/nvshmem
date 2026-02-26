/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include <stdio.h>
#include <assert.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <time.h>
#include "utils.h"
#include <stdlib.h>
#include <dlfcn.h>
#include <cstdint>
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host/nvshmemi_bootstrap_library.h"
#include "non_abi/nvshmemx_error.h"

#define API_STATUS_INTERNAL(expr, code_block, ...) \
    do {                                           \
        if ((expr)) {                              \
            fprintf(stderr, __VA_ARGS__);          \
            do {                                   \
                code_block;                        \
            } while (0);                           \
        }                                          \
    } while (0)

/** Test Specific Macros & Helper preprocessor directive. Use NVSHMEM_TEST_ prefix */
#define NVSHMEM_TEST_API_CHECK(expr, code, ...) API_STATUS_INTERNAL(expr, code, __VA_ARGS__)
#define NVSHMEM_TEST_API_STATUS(status, expected, code, ...) \
    API_STATUS_INTERNAL((status != expected), code, __VA_ARGS__)

#define NVSHMEM_TEST_GET_SYMBOL(lib_handle, name, var, status)                                   \
    do {                                                                                         \
        void **var_ptr = (void **)&(var);                                                        \
        void *tmp = (void *)dlsym(lib_handle, name);                                             \
        NVSHMEMI_NULL_ERROR_JMP(tmp, status, -1, out,                                            \
                                "Bootstrap failed to get symbol '%s'\n\t%s\n", name, dlerror()); \
        *var_ptr = tmp;                                                                          \
    } while (0)

#define NVSHMEM_TEST_BOOTSTRAP_ABI_VERSION            \
    (NVSHMEM_BOOTSTRAP_PLUGIN_MAJOR_VERSION * 10000 + \
     NVSHMEM_BOOTSTRAP_PLUGIN_MINOR_VERSION * 100 + NVSHMEM_BOOTSTRAP_PLUGIN_PATCH_VERSION)

#ifndef nvshmem_test_unlikely
#define nvshmem_test_unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#define NVSHMEM_TEST_LOG_DEBUG(fmt, ...)                                                 \
    do {                                                                                 \
        if (!testing::performance::mode && testing::log_verbose > TEST_LOG_LEVEL_INFO) { \
            fprintf(stdout, "[DEBUG][%s:%d] ", __FILE__, __LINE__);                      \
            fprintf(stdout, fmt, __VA_ARGS__);                                           \
        }                                                                                \
    } while (0)

#define NVSHMEM_TEST_LOG_INFO(fmt, ...)                                                  \
    do {                                                                                 \
        if (!testing::performance::mode && testing::log_verbose > TEST_LOG_LEVEL_NONE) { \
            fprintf(stdout, "[INFO][%s:%d] ", __FILE__, __LINE__);                       \
            fprintf(stdout, fmt, __VA_ARGS__);                                           \
        }                                                                                \
    } while (0)

#define NVSHMEM_TEST_LOG_INFO_RANK(pg_rank, fmt, ...)                      \
    do {                                                                   \
        if (!testing::performance::mode && testing::log_rank == pg_rank && \
            testing::log_verbose >= TEST_LOG_LEVEL_INFO) {                 \
            fprintf(stdout, "[INFO][%s:%d] ", __FILE__, __LINE__);         \
            fprintf(stdout, fmt, __VA_ARGS__);                             \
        }                                                                  \
    } while (0)

#define NVSHMEM_TEST_LOG_DEBUG_RANK(pg_rank, fmt, ...)                     \
    do {                                                                   \
        if (!testing::performance::mode && testing::log_rank == pg_rank && \
            testing::log_verbose >= TEST_LOG_LEVEL_DEBUG) {                \
            fprintf(stdout, fmt, __VA_ARGS__);                             \
        }                                                                  \
    } while (0)

namespace testing {

// Common to all namespaces
static std::unordered_map<std::string, int> bstrap_dict = {
    {"MPI", BOOTSTRAP_MPI},       {"SHMEM", BOOTSTRAP_SHMEM}, {"PMI", BOOTSTRAP_PMI},
    {"PMI2", BOOTSTRAP_PMI},      {"PMI-2", BOOTSTRAP_PMI},   {"PMIX", BOOTSTRAP_PMI},
    {"PLUGIN", BOOTSTRAP_PLUGIN}, {"UID", BOOTSTRAP_UID}};

enum {
    TEST_ALLTOALL = 0,
    TEST_ALLGATHER,
    TEST_BARRIER,
    TEST_SHOWINFO,
    TEST_ALL,
    TEST_MAX,
};

static std::unordered_map<std::string, int> coll_dict = {{"ALLTOALL", TEST_ALLTOALL},
                                                         {"ALLGATHER", TEST_ALLGATHER},
                                                         {"BARRIER", TEST_BARRIER},
                                                         {"ALL", TEST_ALL},
                                                         {"SHOWINFO", TEST_SHOWINFO}};

enum {
    TEST_LOG_LEVEL_NONE = 0, /* Prints no messages */
    TEST_LOG_LEVEL_INFO,     /* Prints test summary & test stages */
    TEST_LOG_LEVEL_DEBUG,    /* Prints chatty messages */
    TEST_LOG_LEVEL_MAX,
};

enum {
    TEST_BOOTSTRAP_SUCCESS = 0,
    TEST_BOOTSTRAP_INTERNAL_ERROR = -1,
    TEST_BOOTSTRAP_DATA_MISMATCH_ERROR = -2,
    TEST_BOOTSTRAP_UNSUPPORTED_COLL
};

static std::unordered_map<std::string, int> log_dict = {
    {"NONE", TEST_LOG_LEVEL_NONE}, {"INFO", TEST_LOG_LEVEL_INFO}, {"DEBUG", TEST_LOG_LEVEL_DEBUG}};

// Message size range for collectives
static std::vector<int> msg_len = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

// Identify the rank to log the results
static int log_rank = 0;

// Log verbosity used to filter out the msg buffer logs that can be chatty
#define TEST_BOOTSTRAP_DEFAULT_LOG_VERBOSITY TEST_LOG_LEVEL_NONE
static int log_verbose = TEST_BOOTSTRAP_DEFAULT_LOG_VERBOSITY;

// Performance diagnostics infra
namespace performance {

typedef struct test_perf_args_s {
    test_perf_args_s(long int a, long int b, int c)
        : elapsed_nsec(a), elapsed_count(b), xfer_size(c) {}
    long int elapsed_nsec;
    long int elapsed_count;
    int xfer_size;
} test_perf_args_t;

static bool mode = false;
static std::unordered_map<int, std::vector<test_perf_args_t>> results;

// In-memory host-side profiling macros
#define TEST_BOOTSTRAP_SEC_TO_NSEC 1000000000ULL
static void inline test_bootstrap_start(struct timespec *start) {
    clock_gettime(CLOCK_MONOTONIC, start);
    return;
}
static void inline test_bootstrap_get_elapsed_time(int coll_type, struct timespec *start,
                                                   struct timespec *end, long int count,
                                                   int msg_size) {
    clock_gettime(CLOCK_MONOTONIC, end);
    long int tdiff = ((end->tv_nsec + end->tv_sec * TEST_BOOTSTRAP_SEC_TO_NSEC) -
                      (start->tv_nsec + start->tv_sec * TEST_BOOTSTRAP_SEC_TO_NSEC));
    if (results.find(coll_type) == results.end()) {
        results.insert(std::make_pair(coll_type, std::vector<test_perf_args_t>()));
        results[coll_type].push_back(test_perf_args_t(tdiff, count, msg_size));
    } else {
        results[coll_type].push_back(test_perf_args_t(tdiff, count, msg_size));
    }
    return;
}

static void inline print_results(std::stringstream &summary_str) {
    // Iterate for every collective and collect a perf dict per msg size
    // {0: [], 1: [], 2: [], ... 1048576: []}
    for (auto &pair : results) {
        summary_str << "Bootstrap perf coll: " << pair.first << "\n";
        std::unordered_map<int, std::vector<test_perf_args_t>> msg_dict;
        std::for_each(pair.second.begin(), pair.second.end(), [&](test_perf_args_t item) {
            if (msg_dict.find(item.xfer_size) == msg_dict.end()) {
                msg_dict.insert(std::make_pair(item.xfer_size, std::vector<test_perf_args_t>()));
                msg_dict[item.xfer_size].push_back(
                    test_perf_args_t(item.elapsed_nsec, item.elapsed_count, item.xfer_size));
            } else {
                msg_dict[item.xfer_size].push_back(
                    test_perf_args_t(item.elapsed_nsec, item.elapsed_count, item.xfer_size));
            }
        });

        // Iterate over perf dict and summarize the latency statistics
        for (auto &pair : msg_dict) {
            long int min_nsec = LONG_MAX, max_nsec = LONG_MIN, avg_nsec = 0, avg_count = 0;
            int index = 0;
            std::for_each(pair.second.begin(), pair.second.end(), [&](test_perf_args_t item) {
                // summary_str << "[" << index++ << "]: " << item.elapsed_nsec << "," <<
                // item.elapsed_count << "\n"; compute the min compute the max compute the avg
                min_nsec = std::min(min_nsec, (item.elapsed_nsec / item.elapsed_count));
                max_nsec = std::max(max_nsec, (item.elapsed_nsec / item.elapsed_count));
                avg_nsec += item.elapsed_nsec;
                avg_count += item.elapsed_count;
            });

            avg_nsec /= (avg_count);
            summary_str << "Bootstrap perf msg size: " << pair.first
                        << " bytes min(nsec): " << min_nsec << " max(nsec): " << max_nsec
                        << " avg(nsec): " << avg_nsec << "\n";
        }
    }
}

}  // namespace performance

// Default test configurations (useful when debugging a particular bootstrap)
#define TEST_BOOTSTRAP_COLL_DEFAULT_ITERATIONS 10
#define TEST_BOOTSTRAP_TEST_DEFAULT_ITERATIONS 1

namespace memory {

static void test_memory_print_buffer(void *buf, int length, int pg_rank, int pg_size) {
    NVSHMEM_TEST_LOG_DEBUG_RANK(pg_rank, "[Rank%d][START]: ", pg_rank);
    for (auto i = 0; i < (pg_size * length); i++) {
        if (i != 0 && i % 16 == 0) {
            NVSHMEM_TEST_LOG_DEBUG_RANK(pg_rank, "\n[Rank%d][0x%02x]: ", pg_rank, i);
        }

        NVSHMEM_TEST_LOG_DEBUG_RANK(pg_rank, "%02x\t", *((char *)(buf) + i));
    }

    NVSHMEM_TEST_LOG_DEBUG_RANK(pg_rank, "[Rank%d][END]\n", pg_rank);
}

static int test_memory_setup_buffer(int pg_rank, int pg_size, bool is_inplace,
                                    std::vector<int> msg_lens, void ***send_bufs,
                                    void ***recv_bufs) {
    // Allocate and prepare the buffers
    (*send_bufs) = (void **)calloc(msg_lens.size(), sizeof(void *));
    for (auto i = 0U; i < msg_lens.size(); i++) {
        (*send_bufs)[i] = malloc(msg_lens[i] * pg_size * sizeof(char));
        memset((*send_bufs)[i], pg_rank, sizeof(char) * pg_size * msg_lens[i]);
        test_memory_print_buffer((*send_bufs)[i], msg_lens[i], pg_rank, pg_size);
    }

    (*recv_bufs) = (void **)calloc(msg_lens.size(), sizeof(void *));
    for (auto i = 0U; i < msg_lens.size(); i++) {
        if (is_inplace)
            (*recv_bufs)[i] = (*send_bufs)[i];
        else {
            (*recv_bufs)[i] = malloc(msg_lens[i] * pg_size * sizeof(char));
            memset((*recv_bufs)[i], 0, sizeof(char) * pg_size * msg_lens[i]);
        }
    }

    return 0;
}

static int test_memory_free_buffer(int n_bufs, void **send_bufs, void **recv_bufs) {
    for (auto i = 0; i < n_bufs; i++) {
        if (send_bufs[i] == recv_bufs[i]) {
            free(send_bufs[i]);
        } else {
            free(send_bufs[i]);
            free(recv_bufs[i]);
        }
    }

    free(send_bufs);
    free(recv_bufs);
    return 0;
}

static int test_memory_validate_buffer(int type, void *send_buf, void *recv_buf, int pg_rank,
                                       int pg_size, int length) {
    int status = 0;
    auto it = std::find_if(
        coll_dict.begin(), coll_dict.end(),
        [&type](const std::pair<std::string, int> &tuple) { return (tuple.second == type); });
    switch (type) {
        case TEST_ALLGATHER:
            // Since all elements of a send_buf are identical per rank, allgather and alltoall have
            // the same correctness logic
        case TEST_ALLTOALL: {
            test_memory_print_buffer(recv_buf, length, pg_rank, pg_size);
            for (auto i = 0; i < (length * pg_size); i++) {
                int exp_rank = (i / length);
                if (*((char *)recv_buf + i) != (char)exp_rank) {
                    status = TEST_BOOTSTRAP_DATA_MISMATCH_ERROR;
                    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, error,
                                          "Data mismatch observed for collective = %s Offset = %d "
                                          "Expected = %c, Actual = %c\n",
                                          it->first.c_str(), i, exp_rank, *((char *)recv_buf + i));
                }
            }
        } break;
        default:
            status = TEST_BOOTSTRAP_UNSUPPORTED_COLL;
    }

error:
    return (status);
}

}  // namespace memory

// Bootstrap specific test functions
namespace bootstrap {

static bool inline test_bootstrap_inplace_supported(char *plugin_lib) {
    /* MPI bootstrap supports inplace operations */
    return (strstr(plugin_lib, "mpi") != NULL);
}

static void _test_bootstrap_fini_helper(void *plugin_hdl, char *plugin_name) {
    if (plugin_hdl != nullptr) {
        dlclose(plugin_hdl);
        plugin_hdl = nullptr;
    }

    if (plugin_name != nullptr) {
        free(plugin_name);
        plugin_name = nullptr;
    }
}

static int _test_bootstrap_init_helper(char *plugin, void **plugin_hdl, char **plugin_name) {
    int status = 0;

    dlerror(); /* Clear any existing error */
    *plugin_name = strdup(plugin);
    *plugin_hdl = dlopen(plugin, RTLD_NOW);
    NVSHMEMI_NULL_ERROR_JMP(*plugin_hdl, status, TEST_BOOTSTRAP_INTERNAL_ERROR, error,
                            "Bootstrap unable to load '%s'\n\t%s\n", plugin, dlerror());

    dlerror(); /* Clear any existing error */
    goto out;

error:
    _test_bootstrap_fini_helper(*plugin_hdl, *plugin_name);
out:
    return status;
}

static int test_bootstrap_preinit(char *plugin, bootstrap_handle_t *handle) {
    int status = 0;
    void *plugin_hdl = NULL;
    char *plugin_name = NULL;
    int (*bootstrap_plugin_preinitops)(bootstrap_handle_t * handle, int nvshmem_version);
    status = _test_bootstrap_init_helper(plugin, &plugin_hdl, &plugin_name);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap library dlopen failed for %s\n", plugin);
    NVSHMEM_TEST_GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_pre_init",
                            bootstrap_plugin_preinitops, status);
    status = bootstrap_plugin_preinitops(handle, NVSHMEM_TEST_BOOTSTRAP_ABI_VERSION);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap plugin preinit failed for '%s'\n", plugin);
    goto out;
error:
    _test_bootstrap_fini_helper(plugin_hdl, plugin_name);
out:
    return (status);
}

static int test_bootstrap_init(char *plugin, void *arg, bootstrap_handle_t *handle) {
    int status = 0;
    void *plugin_hdl = NULL;
    char *plugin_name = NULL;
    int (*bootstrap_plugin_initops)(void *arg, bootstrap_handle_t *handle, int nvshmem_version);
    status = _test_bootstrap_init_helper(plugin, &plugin_hdl, &plugin_name);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap library dlopen failed for %s\n", plugin);
    NVSHMEM_TEST_GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_init", bootstrap_plugin_initops,
                            status);
    status = bootstrap_plugin_initops(arg, handle, NVSHMEM_TEST_BOOTSTRAP_ABI_VERSION);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap plugin init failed for '%s'\n", plugin);
    goto out;
error:
    _test_bootstrap_fini_helper(plugin_hdl, plugin_name);
out:
    return (status);
}

static int test_bootstrap_showinfo(char *plugin_lib, void *plugin_args, int iter,
                                   bootstrap_handle_t *bstrap_handle) {
    int status = 0;
    // Initialize the bootstrap
    status = testing::bootstrap::test_bootstrap_init(plugin_lib, plugin_args, bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Init Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Initialization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    for (auto i = 0L; i < iter; i++) {
        status = bstrap_handle->show_info(bstrap_handle, BOOTSTRAP_OPTIONS_STYLE_INFO);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                              "Bootstrap ShowInfo Failed\n");
    }

    // Finalize the bootstrap
cleanup:
    status |= bstrap_handle->finalize(bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Fini Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Finalization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);
return_status:
    return (status);
}

static int test_bootstrap_barrier(char *plugin_lib, void *plugin_args, int iter,
                                  bootstrap_handle_t *bstrap_handle) {
    int status = 0;
    struct timespec start, end;
    // Initialize the bootstrap
    status = testing::bootstrap::test_bootstrap_init(plugin_lib, plugin_args, bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Init Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Initialization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    for (auto i = 0L; i < iter; i++) {
        testing::performance::test_bootstrap_start(&start);
        status = bstrap_handle->barrier(bstrap_handle);
        testing::performance::test_bootstrap_get_elapsed_time(TEST_BARRIER, &start, &end, 1, 0);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                              "Bootstrap Barrier Failed\n");
    }
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Barrier Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    // Finalize the bootstrap
cleanup:
    status |= bstrap_handle->finalize(bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Fini Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Finalization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);
return_status:
    return (status);
}

static int test_bootstrap_alltoall(char *plugin_lib, void *plugin_args, long int iter,
                                   bootstrap_handle_t *bstrap_handle) {
    int status = 0;
    struct timespec start, end;
    void **send_buf = NULL, **recv_buf = NULL;

    // Initialize the bootstrap
    status = testing::bootstrap::test_bootstrap_init(plugin_lib, plugin_args, bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Init Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Initialization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    // Test for inplace, if supported
    if (test_bootstrap_inplace_supported(plugin_lib)) {
        bool inplace = true;
        // Setup the send/recv buffer with different message sizes
        status = testing::memory::test_memory_setup_buffer(
            bstrap_handle->pg_rank, bstrap_handle->pg_size, inplace, msg_len, &send_buf, &recv_buf);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                              "Bootstrap Send/Recv Buffer Alloc Failed");

        for (auto k = 0U; k < msg_len.size(); k++) {
            for (auto i = 0L; i < iter; i++) {
                testing::performance::test_bootstrap_start(&start);
                status =
                    bstrap_handle->alltoall(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
                testing::performance::test_bootstrap_get_elapsed_time(TEST_ALLTOALL, &start, &end,
                                                                      1, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, inplace_cleanup,
                                      "Bootstrap AlltoAll Failed\n");
                if (nvshmem_test_unlikely(!testing::performance::mode)) {
                    status = testing::memory::test_memory_validate_buffer(
                        TEST_ALLTOALL, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                        bstrap_handle->pg_size, msg_len[k]);
                    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, inplace_cleanup,
                                          "Data correctness failed\n");
                }
                NVSHMEM_TEST_LOG_INFO_RANK(
                    bstrap_handle->pg_rank, "AlltoAll Done on Rank %d Size %d Msg %d\n",
                    bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);
            }
        }

    inplace_cleanup:
        status |= testing::memory::test_memory_free_buffer(msg_len.size(), send_buf, recv_buf);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                              "Bootstrap Send/Recv Buffer Free Failed");
    }

    // Setup the send/recv buffer with different message sizes
    status = testing::memory::test_memory_setup_buffer(bstrap_handle->pg_rank,
                                                       bstrap_handle->pg_size, false /*inplace */,
                                                       msg_len, &send_buf, &recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Alloc Failed");

    for (auto k = 0U; k < msg_len.size(); k++) {
        for (auto i = 0L; i < iter; i++) {
            testing::performance::test_bootstrap_start(&start);
            status = bstrap_handle->alltoall(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
            testing::performance::test_bootstrap_get_elapsed_time(TEST_ALLTOALL, &start, &end, 1,
                                                                  msg_len[k]);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap AlltoAll Failed\n");
            if (nvshmem_test_unlikely(!testing::performance::mode)) {
                status = testing::memory::test_memory_validate_buffer(
                    TEST_ALLTOALL, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                    bstrap_handle->pg_size, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                      "Data correctness failed\n");
            }
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank,
                                       "AlltoAll Done on Rank %d Size %d Msg %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);
        }
    }

cleanup:
    status |= testing::memory::test_memory_free_buffer(msg_len.size(), send_buf, recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Free Failed");

finalize_only:
    // Finalize the bootstrap
    status |= bstrap_handle->finalize(bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Fini Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Finalization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);
return_status:
    return (status);
}

static int test_bootstrap_allgather(char *plugin_lib, void *plugin_args, int iter,
                                    bootstrap_handle_t *bstrap_handle) {
    int status = 0;
    struct timespec start, end;
    void **send_buf = NULL, **recv_buf = NULL;

    // Initialize the bootstrap
    status = testing::bootstrap::test_bootstrap_init(plugin_lib, plugin_args, bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Init Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Initialization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    // Test for inplace is supported
    if (test_bootstrap_inplace_supported(plugin_lib)) {
        bool inplace = true;
        // Setup the send/recv buffer with different message sizes
        status = testing::memory::test_memory_setup_buffer(
            bstrap_handle->pg_rank, bstrap_handle->pg_size, inplace, msg_len, &send_buf, &recv_buf);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                              "Bootstrap Send/Recv Buffer Alloc Failed");
        for (auto k = 0U; k < msg_len.size(); k++) {
            for (auto i = 0L; i < iter; i++) {
                testing::performance::test_bootstrap_start(&start);
                status =
                    bstrap_handle->allgather(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
                testing::performance::test_bootstrap_get_elapsed_time(TEST_ALLGATHER, &start, &end,
                                                                      1, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, inplace_cleanup,
                                      "Bootstrap Allgather Failed\n");
                if (nvshmem_test_unlikely(!testing::performance::mode)) {
                    status = testing::memory::test_memory_validate_buffer(
                        TEST_ALLGATHER, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                        bstrap_handle->pg_size, msg_len[k]);
                    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, inplace_cleanup,
                                          "Data correctness failed\n");
                }
                NVSHMEM_TEST_LOG_INFO_RANK(
                    bstrap_handle->pg_rank, "Allgather Done on Rank %d Size %d Msg %d\n",
                    bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);
            }
        }

    inplace_cleanup:
        status |= testing::memory::test_memory_free_buffer(msg_len.size(), send_buf, recv_buf);
        NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                              "Bootstrap Send/Recv Buffer Free Failed");
    }

    // Setup the send/recv buffer with different message sizes
    status =
        testing::memory::test_memory_setup_buffer(bstrap_handle->pg_rank, bstrap_handle->pg_size,
                                                  false /*inplace*/, msg_len, &send_buf, &recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Alloc Failed");
    for (auto k = 0U; k < msg_len.size(); k++) {
        for (auto i = 0L; i < iter; i++) {
            testing::performance::test_bootstrap_start(&start);
            status = bstrap_handle->allgather(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
            testing::performance::test_bootstrap_get_elapsed_time(TEST_ALLGATHER, &start, &end, 1,
                                                                  msg_len[k]);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap Allgather Failed\n");
            if (nvshmem_test_unlikely(!testing::performance::mode)) {
                status = testing::memory::test_memory_validate_buffer(
                    TEST_ALLGATHER, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                    bstrap_handle->pg_size, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                      "Data correctness failed\n");
            }
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank,
                                       "Allgather Done on Rank %d Size %d Msg %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);
        }
    }

cleanup:
    status |= testing::memory::test_memory_free_buffer(msg_len.size(), send_buf, recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Free Failed");

finalize_only:
    // Finalize the bootstrap
    status |= bstrap_handle->finalize(bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Fini Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Finalization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);
return_status:
    return (status);
}

static int test_bootstrap_catchall(char *plugin_lib, void *plugin_args, int iter,
                                   bootstrap_handle_t *bstrap_handle) {
    int status = 0;
    struct timespec start, end;
    void **send_buf = NULL, **recv_buf = NULL;

    // Initialize the bootstrap
    status = testing::bootstrap::test_bootstrap_init(plugin_lib, plugin_args, bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Init Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Initialization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);

    // Setup the send/recv buffer with different message sizes
    status = testing::memory::test_memory_setup_buffer(
        bstrap_handle->pg_rank, bstrap_handle->pg_size, false, msg_len, &send_buf, &recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Alloc Failed");

    for (auto k = 0U; k < msg_len.size(); k++) {
        for (auto i = 0L; i < iter; i++) {
            testing::performance::test_bootstrap_start(&start);
            /* allgather */
            status = bstrap_handle->allgather(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap Allgather Failed\n");
            if (nvshmem_test_unlikely(!testing::performance::mode)) {
                status = testing::memory::test_memory_validate_buffer(
                    TEST_ALLGATHER, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                    bstrap_handle->pg_size, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                      "Data correctness failed\n");
            }
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank,
                                       "Allgather Done on Rank %d Size %d Msg %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);

            /* alltoall */
            status = bstrap_handle->alltoall(send_buf[k], recv_buf[k], msg_len[k], bstrap_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap AlltoAll Failed\n");
            if (nvshmem_test_unlikely(!testing::performance::mode)) {
                status = testing::memory::test_memory_validate_buffer(
                    TEST_ALLTOALL, send_buf[k], recv_buf[k], bstrap_handle->pg_rank,
                    bstrap_handle->pg_size, msg_len[k]);
                NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                      "Data correctness failed\n");
            }
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank,
                                       "AlltoAll Done on Rank %d Size %d Msg %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size, msg_len[k]);

            /* barrier */
            status = bstrap_handle->barrier(bstrap_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap Barrier Failed\n");
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Barrier Done on Rank %d Size %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size);

            status = bstrap_handle->show_info(bstrap_handle, BOOTSTRAP_OPTIONS_STYLE_INFO);
            NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, cleanup,
                                  "Bootstrap ShowInfo Failed\n");
            NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank,
                                       "Show Info Done on Rank %d Size %d\n",
                                       bstrap_handle->pg_rank, bstrap_handle->pg_size);
            testing::performance::test_bootstrap_get_elapsed_time(TEST_ALL, &start, &end, 1,
                                                                  msg_len[k]);
        }
    }

cleanup:
    status |= testing::memory::test_memory_free_buffer(msg_len.size(), send_buf, recv_buf);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, finalize_only,
                          "Bootstrap Send/Recv Buffer Free Failed");

finalize_only:
    // Finalize the bootstrap
    status |= bstrap_handle->finalize(bstrap_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, TEST_BOOTSTRAP_INTERNAL_ERROR, return_status,
                          "Bootstrap Fini Failed\n");
    NVSHMEM_TEST_LOG_INFO_RANK(bstrap_handle->pg_rank, "Finalization Done on Rank %d Size %d\n",
                               bstrap_handle->pg_rank, bstrap_handle->pg_size);
return_status:
    return (status);
}

}  // namespace bootstrap

// Framework specific test functions
namespace framework {

typedef struct {
    long int test_iter;
    std::string coll;
    std::string type;
    long int coll_iter;
} test_args_t;

static int test_bootstrap_all(char *lib, int ctype, void *args,
                              long int citer = TEST_BOOTSTRAP_COLL_DEFAULT_ITERATIONS,
                              bootstrap_handle_t *bstrap_handle = nullptr) {
    switch (ctype) {
        case TEST_ALLTOALL:
            return bootstrap::test_bootstrap_alltoall(lib, args, citer, bstrap_handle);
        case TEST_ALLGATHER:
            return bootstrap::test_bootstrap_allgather(lib, args, citer, bstrap_handle);
        case TEST_BARRIER:
            return bootstrap::test_bootstrap_barrier(lib, args, citer, bstrap_handle);
        case TEST_SHOWINFO:
            return bootstrap::test_bootstrap_showinfo(lib, args, citer, bstrap_handle);
        case TEST_ALL:
            return bootstrap::test_bootstrap_catchall(lib, args, citer, bstrap_handle);
        default:
            return (TEST_BOOTSTRAP_UNSUPPORTED_COLL);
    }

    return (TEST_BOOTSTRAP_INTERNAL_ERROR);
}

int test_bootstrap(test_args_t *args) {
    int status = 0;
    char *bootstrap_type = getenv("NVSHMEM_BOOTSTRAP");
    if (bootstrap_type) {
        args->type = bootstrap_type;
    } else {
        // Default bootstrap: pmi
        args->type = "PMI";
    }

    bootstrap_handle_t bstrap_handle = {};
    nvshmemx_uniqueid_t *uid = nullptr;
    int is_initialized = 0;

    // Test for type = MPI, SHMEM, PMI, PMI2, PMIX, PLUGIN
    NVSHMEM_TEST_API_CHECK(bstrap_dict.find(args->type) == bstrap_dict.end(),
                           {
                               status = TEST_BOOTSTRAP_INTERNAL_ERROR;
                               return (status);
                           },
                           "Unsupported bootstrap method: %s\n", args->type.c_str());

    // Test for coll = ALLGATHER, ALLTOALL, BARRIER
    NVSHMEM_TEST_API_CHECK(coll_dict.find(args->coll) == coll_dict.end(),
                           {
                               status = TEST_BOOTSTRAP_INTERNAL_ERROR;
                               return (status);
                           },
                           "Unsupported collective: %s\n", args->coll.c_str());

    for (auto i = 0; i < args->test_iter; i++) {
        NVSHMEM_TEST_LOG_INFO("Running Test Iteration [%d/%ld] \n", i, args->test_iter);
        switch (bstrap_dict[args->type]) {
            case BOOTSTRAP_MPI: {
#ifdef NVSHMEMTEST_MPI_SUPPORT
                bootstrap_attr_t type_args;
                MPI_Comm comm = MPI_COMM_WORLD;
                type_args.mpi_comm = &comm;
                char *plugin_lib = getenv("NVSHMEM_BOOTSTRAP_MPI_PLUGIN");
                if (plugin_lib) {
                    status |= test_bootstrap_all(plugin_lib, coll_dict[args->coll],
                                                 (void *)(type_args.mpi_comm), args->coll_iter,
                                                 &bstrap_handle);
                } else {
                    // Default: nvshmem_bootstrap_mpi.so
                    const char *default_lib = "nvshmem_bootstrap_mpi.so";
                    status |=
                        test_bootstrap_all(const_cast<char *>(default_lib), coll_dict[args->coll],
                                           NULL, args->coll_iter, &bstrap_handle);
                }
#else
                /* Throw an error if MPI is not compiled and MPI bootstrap is selected */
                status |= TEST_BOOTSTRAP_INTERNAL_ERROR;
#endif
            } break;
            case BOOTSTRAP_SHMEM: {
                bootstrap_attr_t type_args;
                type_args.initialize_shmem = 1;
                char *plugin_lib = getenv("NVSHMEM_BOOTSTRAP_SHMEM_PLUGIN");
                if (plugin_lib) {
                    status |=
                        test_bootstrap_all(plugin_lib, coll_dict[args->coll], (void *)(&type_args),
                                           args->coll_iter, &bstrap_handle);
                } else {
                    // Default: nvshmem_bootstrap_shmem.so
                    const char *default_lib = "nvshmem_bootstrap_shmem.so";
                    status |=
                        test_bootstrap_all(const_cast<char *>(default_lib), coll_dict[args->coll],
                                           NULL, args->coll_iter, &bstrap_handle);
                }
            } break;
            case BOOTSTRAP_PMI: {
                char *pmi_types = getenv("NVSHMEM_BOOTSTRAP_PMI");
                char *pmi_plugin_lib = NULL, *pmi2_plugin_lib = NULL, *pmix_plugin_lib = NULL;
                if (pmi_types) {
                    if (strcasecmp(pmi_types, "PMI") == 0) {
                        pmi_plugin_lib = getenv("NVSHMEM_BOOTSTRAP_PMI_PLUGIN");
                        const char *default_lib = "nvshmem_bootstrap_pmi.so";
                        pmi_plugin_lib = (pmi_plugin_lib == NULL) ? const_cast<char *>(default_lib)
                                                                  : (pmi_plugin_lib);
                    } else if (strcasecmp(pmi_types, "PMI-2") == 0 ||
                               strcasecmp(pmi_types, "PMI2") == 0) {
                        pmi2_plugin_lib = getenv("NVSHMEM_BOOTSTRAP_PMI2_PLUGIN");
                        const char *default_lib = "nvshmem_bootstrap_pmi2.so";
                        pmi2_plugin_lib = (pmi2_plugin_lib == NULL)
                                              ? const_cast<char *>(default_lib)
                                              : (pmi2_plugin_lib);
                    } else if (strcasecmp(pmi_types, "PMIX") == 0) {
                        pmix_plugin_lib = getenv("NVSHMEM_BOOTSTRAP_PMIX_PLUGIN");
                        const char *default_lib = "nvshmem_bootstrap_pmix.so";
                        pmix_plugin_lib = (pmix_plugin_lib == NULL)
                                              ? const_cast<char *>(default_lib)
                                              : (pmix_plugin_lib);
                    } else {
                        fprintf(stderr, "Invalid NVSHMEM_BOOTSTRAP_PMI value: %s\n", pmi_types);
                        status |= TEST_BOOTSTRAP_INTERNAL_ERROR;
                        return (status);
                    }
                } else {
                    // Default: PMI and PLUGIN: nvshmem_bootstrap_pmi.so
                    pmi_plugin_lib = getenv("NVSHMEM_BOOTSTRAP_PMI_PLUGIN");
                    const char *default_lib = "nvshmem_bootstrap_pmi.so";
                    pmi_plugin_lib =
                        (pmi_plugin_lib == NULL) ? const_cast<char *>(default_lib) : pmi_plugin_lib;
                }

                if (pmi_plugin_lib) {
                    status |= test_bootstrap_all(pmi_plugin_lib, coll_dict[args->coll], NULL,
                                                 args->coll_iter, &bstrap_handle);
                } else if (pmi2_plugin_lib) {
                    status |= test_bootstrap_all(pmi2_plugin_lib, coll_dict[args->coll], NULL,
                                                 args->coll_iter, &bstrap_handle);
                } else if (pmix_plugin_lib) {
                    status |= test_bootstrap_all(pmix_plugin_lib, coll_dict[args->coll], NULL,
                                                 args->coll_iter, &bstrap_handle);
                } else {
                    // Default: nvshmem_bootstrap_pmi.so
                    const char *default_lib = "nvshmem_bootstrap_pmi.so";
                    status |=
                        test_bootstrap_all(const_cast<char *>(default_lib), coll_dict[args->coll],
                                           NULL, args->coll_iter, &bstrap_handle);
                }
            } break;
            case BOOTSTRAP_PLUGIN: {
                char *plugin_lib = getenv("NVSHMEM_BOOTSTRAP_PLUGIN");
                if (plugin_lib) {
                    status |= test_bootstrap_all(plugin_lib, coll_dict[args->coll], NULL,
                                                 args->coll_iter, &bstrap_handle);
                } else {
                    fprintf(stderr, "Missing plugin lib path in NVSHMEM_BOOTSTRAP_PLUGIN\n");
                    status |= TEST_BOOTSTRAP_INTERNAL_ERROR;
                    return (status);
                }
            } break;
            case BOOTSTRAP_UID: {
                bootstrap_attr_t type_args = {};
                nvshmemx_uniqueid_args_t uid_args = NVSHMEMX_UNIQUEID_ARGS_INITIALIZER;
                uid = (typeof(uid))calloc(1, sizeof(nvshmemx_uniqueid_t));
                (*uid) = NVSHMEMX_UNIQUEID_INITIALIZER;
                char *plugin_lib = getenv("NVSHMEM_BOOTSTRAP_UID_PLUGIN");
                const char *default_lib = "nvshmem_bootstrap_uid.so";
                int root = 0; /* Use the first rank to be the root */
                status = testing::bootstrap::test_bootstrap_preinit(
                    plugin_lib == NULL ? const_cast<char *>(default_lib) : plugin_lib,
                    &bstrap_handle);
                NVSHMEMI_NZ_ERROR_JMP(status, -1, return_status, "Bootstrap Pre Init Failed\n");

#ifdef NVSHMEMTEST_MPI_SUPPORT
                MPI_Initialized(&is_initialized);
                if (!is_initialized) {
                    MPI_Init(NULL, NULL);
                    is_initialized = 1;
                }

                MPI_Comm_rank(MPI_COMM_WORLD, &(uid_args.myrank));
                MPI_Comm_size(MPI_COMM_WORLD, &(uid_args.nranks));
#endif
                if (uid_args.myrank == root) {
                    status = bstrap_handle.pre_init_ops->get_unique_id((void *)uid);
                    NVSHMEMI_NZ_ERROR_JMP(status, -1, return_status,
                                          "Bootstrap Get UniqueID Failed\n");
                }

#ifdef NVSHMEMTEST_MPI_SUPPORT
                MPI_Bcast(uid, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, root, MPI_COMM_WORLD);
#endif
                uid_args.id = uid;
                type_args.uid_args = (void *)&(uid_args);
                if (plugin_lib) {
                    status |= test_bootstrap_all(plugin_lib, coll_dict[args->coll],
                                                 (void *)(type_args.uid_args), args->coll_iter,
                                                 &bstrap_handle);
                } else {
                    // Default: nvshmem_bootstrap_uid.so
                    status |= test_bootstrap_all(
                        const_cast<char *>(default_lib), coll_dict[args->coll],
                        (void *)(type_args.uid_args), args->coll_iter, &bstrap_handle);
                }
            } break;
            default:
                status |= TEST_BOOTSTRAP_INTERNAL_ERROR;
                return (status);
        }
    }

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (is_initialized) {
        MPI_Finalize();
    }
#endif

return_status:
    if (uid) {
        free(uid);
    }

    return (status);
}

void reset(void) {
    // Reset any global data-structures, incase the test needs to be repeated
    testing::performance::mode = false;
    testing::performance::results.clear();
}

void print_summary(int status, framework::test_args_t *args) {
    std::stringstream summary_str;
    summary_str << "Bootstrap test " << ((status == 0) ? "PASSED" : "FAILED") << " Summary\n";
    summary_str << "Bootstrap Type: " << args->type << "\n";
    summary_str << "Bootstrap Coll: " << args->coll << "\n";
    summary_str << "Bootstrap Test Iterations: " << std::to_string(args->test_iter) << "\n";
    summary_str << "Bootstrap Coll Iterations: " << std::to_string(args->coll_iter) << "\n";
    summary_str << "Bootstrap Perf Mode Status: "
                << ((testing::performance::mode) ? "Enabled" : "Disabled") << "\n";

    if (testing::performance::mode) {
        testing::performance::print_results(summary_str);
    }

    fprintf(stdout, "%s", summary_str.str().c_str());
    return;
}

void print_env(void) {
    fprintf(stdout, "List of envs (key-value) supported are summarized\n");
    fprintf(stdout, "-------------------------------------------------\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_PMI=<PMI, PMI-2, PMIX>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP=<MPI, PMI, SHMEM, plugin, UID>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_PLUGIN=<user define plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_MPI_PLUGIN=<{/path/to/}mpi plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_UID_PLUGIN=<{/path/to/}uid plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_SHMEM_PLUGIN=<{/path/to/}shmem plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_PMI_PLUGIN=<{/path/to/}pmi plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_PMI2_PLUGIN=<{/path/to/}pmi2 plugin lib .so>\n");
    fprintf(stdout, "\tNVSHMEM_BOOTSTRAP_PMIX_PLUGIN=<{/path/to/}pmix plugin lib .so>\n");
    fprintf(
        stdout,
        "Refer to https://docs.nvidia.com/nvshmem/api/gen/env.html#bootstrap-options for usage\n");
    return;
}

void print_help(void) {
    fprintf(stdout, "Bootstrap Collective Benchmark\n");
    fprintf(stdout, "-------------------------------\n");
    fprintf(stdout, "./bootstrap_coll [OPTIONS]\n");
    fprintf(stdout, "Options: \n");
    fprintf(stdout,
            "\t\tUse NVSHMEM_BOOTSTRAP_* to select type, plugin lib. Default is PMI @ "
            "nvshmem_bootstrap_pmi.so\n");
    fprintf(stdout, "\t\t-n Number of Iterations to run a test. Default is 1, if not specified\n");
    fprintf(stdout,
            "\t\t-m Number of Iterations to repeat a collective per test. Default is 10, if "
            "not specified\n");
    fprintf(stdout,
            "\t\t-c Test Case Name <ALLTOALL/ALLGATHER/BARRIER/SHOWINFO>. Default is ALL, if not "
            "specified\n");
    fprintf(stdout,
            "\t\t-p Enable Performance Mode. Emits the results to console log. Default is "
            "disabled\n");
    fprintf(stdout,
            "\t\t-r Identify the rank to log the DEBUG messages. Default is rank 0, if not "
            "specified\n");
    fprintf(stdout,
            "\t\t-s Message size for the collective. Default is sweep from 1B to 2KB in steps of "
            "2^k\n");
    fprintf(stdout,
            "\t\t-l Log level for this test <NONE/INFO/DEBUG>. Default is NONE, if not "
            "specified\n");
    fprintf(stdout, "\t\t-h Help mode\n");
    print_env();
    return;
}

void default_args(test_args_t *args) {
    // Default bootstrap is PMI
    (*args).type = "PMI";
    // Default test iteration is 1
    (*args).test_iter = TEST_BOOTSTRAP_TEST_DEFAULT_ITERATIONS;
    // Default coll iteration is 1
    (*args).coll_iter = TEST_BOOTSTRAP_COLL_DEFAULT_ITERATIONS;
    // Default collective is ALL
    (*args).coll = "ALL";
    // Default performance mode is disabled
    testing::performance::mode = false;
    // Default verbosity mode is none
    testing::log_verbose = TEST_BOOTSTRAP_DEFAULT_LOG_VERBOSITY;
}

}  // namespace framework

}  // namespace testing

int main(int argc, char *argv[]) {
    // getopts
    int option = 0;
    testing::framework::test_args_t args = {0, "", "", 0};
    int rc = 0;

    // setup defaults
    testing::framework::default_args(&args);
    while ((option = getopt(argc, argv, "n:m:s:c:r:l:hp")) != -1) {
        switch (option) {
            case 'n': {
                args.test_iter = atoll(optarg);
            } break;
            case 'm': {
                args.coll_iter = atoll(optarg);
            } break;
            case 'h': {
                testing::framework::print_help();
                exit(1);
            } break;
            case 'c': {
                args.coll = optarg;
            } break;
            case 'p': {
                testing::performance::mode = true;
                testing::log_verbose =
                    testing::TEST_LOG_LEVEL_NONE; /* No logging as it would interfer
                                                     in perf collection */
            } break;
            case 'r': {
                testing::log_rank = atoi(optarg);
            } break;
            case 's': {
                testing::msg_len.clear();
                testing::msg_len.push_back(atoi(optarg));
            } break;
            case 'l': {
                std::string log_level = optarg;
                if (testing::log_dict.find(log_level) != testing::log_dict.end()) {
                    testing::log_verbose = testing::log_dict[log_level];
                } else {
                    fprintf(stderr, "Unsupported optarg. See usage\n");
                    testing::framework::print_help();
                    exit(1);
                }
            } break;
            case '?':
            default: {
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                return 1;
            } break;
        }
    }

    rc = testing::framework::test_bootstrap(&args);
    testing::framework::print_summary(rc, &args);
    testing::framework::reset();
    return (rc);
}
