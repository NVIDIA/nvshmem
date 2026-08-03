/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils.h"
#include <array>
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include <stdio.h>
#include <errno.h>
#include <cmath>
#include <string.h>

double *d_latency = NULL;
double *d_avg_time = NULL;
double *latency = NULL;
double *avg_time = NULL;
int mype = 0;
int npes = 0;
int use_mpi = 0;
int use_shmem = 0;
int use_uid = 0;
int use_cubin = 0;

CUmodule mymodule = NULL;

void print_device_uuid_and_peer(int pe, int peer) {
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::array<char, 40> uuid_str{};
    const auto &bytes = prop.uuid.bytes;
    std::snprintf(uuid_str.data(), uuid_str.size(),
                  "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  static_cast<unsigned char>(bytes[0]), static_cast<unsigned char>(bytes[1]),
                  static_cast<unsigned char>(bytes[2]), static_cast<unsigned char>(bytes[3]),
                  static_cast<unsigned char>(bytes[4]), static_cast<unsigned char>(bytes[5]),
                  static_cast<unsigned char>(bytes[6]), static_cast<unsigned char>(bytes[7]),
                  static_cast<unsigned char>(bytes[8]), static_cast<unsigned char>(bytes[9]),
                  static_cast<unsigned char>(bytes[10]), static_cast<unsigned char>(bytes[11]),
                  static_cast<unsigned char>(bytes[12]), static_cast<unsigned char>(bytes[13]),
                  static_cast<unsigned char>(bytes[14]), static_cast<unsigned char>(bytes[15]));
    std::fprintf(stdout, "PE %d: GPU %d, UUID: GPU-%s, peer: %d\n", pe, dev, uuid_str.data(), peer);
    std::fflush(stdout);
}

void init_cumodule(const char *str) {
    int init_error = 0;

    char exe_path[1000];
    size_t count = readlink("/proc/self/exe", exe_path, 1000);
    exe_path[count] = '\0';

    char *exe_dir = dirname(exe_path);
    char cubin_default_path[1000];
    strcpy(cubin_default_path, exe_dir);
    strcat(cubin_default_path, "/");
    strcat(cubin_default_path, str);

    // base name of the test without the extension (e.g. shmem_put_latency)
    char base_name[512];
    strncpy(base_name, str, sizeof(base_name));
    base_name[sizeof(base_name) - 1] = '\0';
    char *dot = strrchr(base_name, '.');
    if (dot && strcmp(dot, ".cubin") == 0) {
        *dot = '\0';
    }
    char cubin_bc_path[1000];
    snprintf(cubin_bc_path, sizeof(cubin_bc_path), "%s/%s_bc.cubin", exe_dir,
             base_name);  // produces: path/to/shmem_put_latency_bc.cubin

    char cubin_ltoir_path[1000];
    snprintf(cubin_ltoir_path, sizeof(cubin_ltoir_path), "%s/%s_ltoir.cubin", exe_dir,
             base_name);  // produces: path/to/shmem_put_latency_ltoir.cubin

    char *selected_path;

    if (use_cubin == NVSHMEM_CUBIN_BC) {
        selected_path = cubin_bc_path;
    } else if (use_cubin == NVSHMEM_CUBIN_LTOIR) {
        selected_path = cubin_ltoir_path;
    }

    // check if that specific .cubin with a suffix exists
    if (use_cubin == NVSHMEM_CUBIN_BC || use_cubin == NVSHMEM_CUBIN_LTOIR) {
        if (access(selected_path, R_OK) != 0) {
            fprintf(
                stderr,
                "Requested NVSHMEM_TEST_CUBIN_LIBRARY=%d [0=libnvshmem.a, 1=libnvshmem_device.bc, "
                "2=libnvshmem_device.ltoir.fatbin] but cubin not found: %s\n",
                use_cubin, selected_path);
            exit(-1);
        }
    } else {
        fprintf(stderr, "Invalid NVSHMEM_TEST_CUBIN_LIBRARY value: %d. Expected 1 or 2.\n",
                use_cubin);
        exit(-1);
    }

    printf("CUBIN Selected: %s\n", selected_path);
    CU_CHECK(cuModuleLoad(&mymodule, selected_path));
    init_error = nvshmemx_cumodule_init(mymodule);
    if (init_error) {
        ERROR_PRINT("cumodule_init failed \n");
        assert(false);
    }
}

void init_test_case_kernel(CUfunction *kernel, const char *kernel_name) {
    CU_CHECK(cuModuleGetFunction(kernel, mymodule, kernel_name));
}

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "unistd.h"

void *nvshmemi_shmem_handle = NULL;
struct nvshmemi_shmem_fn_table shmem_fn_table = {0};
static uint64_t nvshmemiu_getHostHash() {
    char hostname[1024];
    uint64_t result = 5381;
    int status = 0;

    status = gethostname(hostname, 1024);
    if (status) ERROR_EXIT("gethostname failed \n");

    for (int c = 0; c < 1024 && hostname[c] != '\0'; c++) {
        result = ((result << 5) + result) + hostname[c];
    }

    return result;
}

/* This is a special function that is a WAR for a bug in OSHMEM
implementation. OSHMEM erroneosly sets the context on device 0 during
shmem_init. Hence before nvshmem_init() is called, device must be
set correctly */
void select_device_shmem() {
    uint64_t host;
    uint64_t *hosts;
    long *pSync;
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;
    int mype, n_pes;

    mype = shmem_fn_table.fn_shmem_my_pe();
    n_pes = shmem_fn_table.fn_shmem_n_pes();
    mype_node = 0;

    host = nvshmemiu_getHostHash();
    hosts = (uint64_t *)shmem_fn_table.fn_shmem_malloc(sizeof(uint64_t) * (n_pes + 1));
    hosts[0] = host;

    pSync = (long *)shmem_fn_table.fn_shmem_malloc(SHMEM_COLLECT_SYNC_SIZE * sizeof(long));

    shmem_fn_table.fn_shmem_fcollect64(hosts + 1, hosts, 1, 0, 0, n_pes, pSync);
    for (int i = 0; i < n_pes; i++) {
        if (i == mype) break;
        if (hosts[i + 1] == host) mype_node++;
    }

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stdout, "mype: %d mype_node: %d device name: %s bus id: %d \n", mype, mype_node,
            prop.name, prop.pciBusID);
}
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
void *nvshmemi_mpi_handle = NULL;
struct nvshmemi_mpi_fn_table mpi_fn_table = {0};
MPI_Comm MPI_COMM_WORLD_PLACEHOLDER;
MPI_Datatype MPI_UINT8_T_PLACEHOLDER;
MPI_Datatype *mpi_uint8_ptr;

int nvshmemi_load_mpi() {
    nvshmemi_mpi_handle = dlopen("libmpi.so.40", RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    if (nvshmemi_mpi_handle == NULL) {
        // Print the error number and description from errno.
        fprintf(stderr, "dlopen failed: errno = %d, description = %s\n", errno, strerror(errno));

        // Additionally, print the error message from dlerror for more specific information.
        const char *dlerror_msg = dlerror();
        if (dlerror_msg) {
            fprintf(stderr, "dlerror: %s\n", dlerror_msg);
        }
        fprintf(stderr,
                "Unable to dlopen libmpi.so.40."
                "Please add it to your LD_LIBRARY_PATH or run without"
                " NVSHMEMTEST_USE_MPI_LAUNCHER.\n");
        return -1;
    }
    MPI_LOAD_SYM(MPI_Init);
    MPI_LOAD_SYM(MPI_Bcast);
    MPI_LOAD_SYM(MPI_Comm_rank);
    MPI_LOAD_SYM(MPI_Comm_size);
    MPI_LOAD_SYM(MPI_Comm_split_type);
    MPI_LOAD_SYM(MPI_Comm_free);
    MPI_LOAD_SYM(MPI_Finalize);

    return 0;
}

int nvshmemi_dlclose_mpi() {
    int status;

    status = dlclose(nvshmemi_mpi_handle);
    if (status) {
        fprintf(stderr, "unable to dlclose MPI.\n");
        return -1;
    }
    return 0;
}
#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
int nvshmemi_load_shmem() {
    nvshmemi_shmem_handle = dlopen("liboshmem.so.40", RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    if (nvshmemi_shmem_handle == NULL) {
        // Print the error number and description from errno.
        fprintf(stderr, "dlopen failed: errno = %d, description = %s\n", errno, strerror(errno));

        // Additionally, print the error message from dlerror for more specific information.
        const char *dlerror_msg = dlerror();
        if (dlerror_msg) {
            fprintf(stderr, "dlerror: %s\n", dlerror_msg);
        }
        fprintf(stderr,
                "Unable to dlopen liboshmem.so.40."
                "Please add it to your LD_LIBRARY_PATH or run without"
                " NVSHMEMTEST_USE_SHMEM_LAUNCHER.\n");
        return -1;
    }
    SHMEM_LOAD_SYM(shmem_init);
    SHMEM_LOAD_SYM(shmem_my_pe);
    SHMEM_LOAD_SYM(shmem_n_pes);
    SHMEM_LOAD_SYM(shmem_fcollect64);
    SHMEM_LOAD_SYM(shmem_malloc);
    SHMEM_LOAD_SYM(shmem_free);
    SHMEM_LOAD_SYM(shmem_finalize);

    return 0;
}

int nvshmemi_dlclose_shmem() {
    int status;

    status = dlclose(nvshmemi_shmem_handle);
    if (status) {
        fprintf(stderr, "unable to dlclose shmem.\n");
        return -1;
    }
    return 0;
}
#endif

void select_device() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;
    int mype;

    mype = nvshmem_my_pe();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stdout, "mype: %d mype_node: %d device name: %s bus id: %d \n", mype, mype_node,
            prop.name, prop.pciBusID);
}

static void check_for_cumodule_tests() {
    char *test_mode = getenv("NVSHMEM_TEST_CUBIN_LIBRARY");
    if (test_mode) {
        use_cubin = atoi(test_mode);
    }
    if (use_cubin == 1) {
        printf("LLVM-IR Bitcode Library Testing Method Chosen.\n");
    } else if (use_cubin == 2) {
        printf("LTOIR Library Testing Method Chosen.\n");
    }
}

static void print_read_args_summary() {
    if (nvshmem_my_pe() != 0) return;
    printf("[PE 0] Runtime options after parsing command line arguments \n");
    printf(
        "min_size: %zu, max_size: %zu, step_factor: %zu, iterations: %zu, warmup iterations: %zu, "
        "number of ctas: %zu, threads per cta: %zu "
        "stride: %zu, datatype: %s, reduce_op: %s, threadgroup_scope: %s, atomic_op: %s, dir: %s, "
        "report_msgrate: %d, bidirectional: %d, putget_issue :%s, use_graph: %d, use_mmap: %d, "
        "mem_handle_type: %zu, use_egm: %d, use_smem: %d\n",
        min_size, max_size, step_factor, iters, warmup_iters, num_blocks, threads_per_block, stride,
        datatype.name.c_str(), reduce_op.name.c_str(), threadgroup_scope.name.c_str(),
        test_amo.name.c_str(), dir.name.c_str(), report_msgrate, bidirectional,
        putget_issue.name.c_str(), use_graph, use_mmap, mem_handle_type, use_egm, use_smem);
    if (repetitions_requested) printf("repetitions: %zu\n", repetitions);
    printf(
        "Note: Above is full list of options, any given test will use only a subset of these "
        "variables.\n");
}

void init_wrapper(int *c, char ***v) {
    check_for_cumodule_tests();
#ifdef NVSHMEMTEST_MPI_SUPPORT
    {
        char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        if (value) use_mpi = atoi(value);
        char *uid_value = getenv("NVSHMEMTEST_USE_UID_BOOTSTRAP");
        if (uid_value) use_uid = atoi(uid_value);
    }
#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    {
        char *value = getenv("NVSHMEMTEST_USE_SHMEM_LAUNCHER");
        if (value) use_shmem = atoi(value);
    }
#endif

#ifdef NVSHMEMTEST_MPI_SUPPORT
    int status;
    int rank, nranks;

    if (use_mpi || use_uid) {
        status = nvshmemi_load_mpi();
        if (status) exit(-1);

        mpi_fn_table.fn_MPI_Init(c, v);

        MPI_COMM_WORLD_PLACEHOLDER = (MPI_Comm)dlsym(nvshmemi_mpi_handle, "ompi_mpi_comm_world");
        MPI_UINT8_T_PLACEHOLDER = (MPI_Datatype)dlsym(nvshmemi_mpi_handle, "ompi_mpi_uint8_t");

        mpi_fn_table.fn_MPI_Comm_rank(MPI_COMM_WORLD_PLACEHOLDER, &rank);
        mpi_fn_table.fn_MPI_Comm_size(MPI_COMM_WORLD_PLACEHOLDER, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
    }
    if (use_mpi || use_uid) {
        // Select device before NVSHMEM init so state->device_id is set correctly.
        // nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) is unavailable pre-init, so derive
        // node-local rank from a shared-memory communicator split.
        MPI_Comm node_comm;
        int local_rank, dev_count;
        MPI_Info info_null = (MPI_Info)dlsym(nvshmemi_mpi_handle, "ompi_mpi_info_null");
        mpi_fn_table.fn_MPI_Comm_split_type(MPI_COMM_WORLD_PLACEHOLDER, MPI_COMM_TYPE_SHARED, rank,
                                            info_null, &node_comm);
        mpi_fn_table.fn_MPI_Comm_rank(node_comm, &local_rank);
        mpi_fn_table.fn_MPI_Comm_free(&node_comm);
        CUDA_CHECK(cudaGetDeviceCount(&dev_count));
        if (dev_count <= 0) ERROR_EXIT("No CUDA devices available\n");
        CUDA_CHECK(cudaSetDevice(local_rank % dev_count));
    }
    if (use_mpi) {
        MPI_Comm mpi_comm = MPI_COMM_WORLD_PLACEHOLDER;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        nvshmem_barrier_all();
        print_read_args_summary();
        return;
    } else if (use_uid) {
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
        if (rank == 0) {
            nvshmemx_get_uniqueid(&id);
        }

        mpi_fn_table.fn_MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T_PLACEHOLDER, 0,
                                  MPI_COMM_WORLD_PLACEHOLDER);
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        nvshmem_barrier_all();
        print_read_args_summary();
        return;
    }
#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) {
        status = nvshmemi_load_shmem();
        if (status) exit(-1);

        shmem_fn_table.fn_shmem_init();
        mype = shmem_fn_table.fn_shmem_my_pe();
        npes = shmem_fn_table.fn_shmem_n_pes();
        DEBUG_PRINT("SHMEM: [%d of %d] hello SHMEM world! \n", mype, npes);

        latency = (double *)shmem_fn_table.fn_shmem_malloc(sizeof(double));
        if (!latency) ERROR_EXIT("(shmem_malloc) failed \n");

        avg_time = (double *)shmem_fn_table.fn_shmem_malloc(sizeof(double));
        if (!avg_time) ERROR_EXIT("(shmem_malloc) failed \n");

        select_device_shmem();

        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

        nvshmem_barrier_all();
        print_read_args_summary();
        return;
    }
#endif

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    select_device();

    nvshmem_barrier_all();
    print_read_args_summary();
    d_latency = (double *)nvshmem_malloc(sizeof(double));
    if (!d_latency) ERROR_EXIT("nvshmem_malloc failed \n");

    d_avg_time = (double *)nvshmem_malloc(sizeof(double));
    if (!d_avg_time) ERROR_EXIT("nvshmem_malloc failed \n");

    DEBUG_PRINT("end of init \n");
    return;
}

void finalize_wrapper() {
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_fn_table.fn_shmem_free(latency);
        shmem_fn_table.fn_shmem_free(avg_time);
    }
#endif

#if !defined(NVSHMEMTEST_SHMEM_SUPPORT) && !defined(NVSHMEMTEST_MPI_SUPPORT)
    if (!use_mpi && !use_shmem) {
        nvshmem_free(d_latency);
        nvshmem_free(d_avg_time);
    }
#endif
    nvshmem_finalize();

#ifdef NVSHMEMTEST_MPI_SUPPORT
    if (use_mpi || use_uid) {
        mpi_fn_table.fn_MPI_Finalize();
        nvshmemi_dlclose_mpi();
    }
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_fn_table.fn_shmem_finalize();
        // Calling dlclose will cause oshrun to segfault at termination.
        // nvshmemi_dlclose_shmem();
    }
#endif
}

void datatype_parse(const char *optarg, datatype_t *datatype) {
    if (!strcmp(optarg, "int")) {
        datatype->type = NVSHMEM_INT;
        datatype->size = sizeof(int);
        datatype->name = "int";
    } else if (!strcmp(optarg, "long")) {
        datatype->type = NVSHMEM_LONG;
        datatype->size = sizeof(long);
        datatype->name = "long";
    } else if (!strcmp(optarg, "longlong")) {
        datatype->type = NVSHMEM_LONGLONG;
        datatype->size = sizeof(long long);
        datatype->name = "longlong";
    } else if (!strcmp(optarg, "ulonglong")) {
        datatype->type = NVSHMEM_ULONGLONG;
        datatype->size = sizeof(unsigned long long);
        datatype->name = "ulonglong";
    } else if (!strcmp(optarg, "size")) {
        datatype->type = NVSHMEM_SIZE;
        datatype->size = sizeof(size_t);
        datatype->name = "size";
    } else if (!strcmp(optarg, "ptrdiff")) {
        datatype->type = NVSHMEM_PTRDIFF;
        datatype->size = sizeof(ptrdiff_t);
        datatype->name = "ptrdiff";
    } else if (strstr(optarg, "float")) {
        datatype->type = NVSHMEM_FLOAT;
        datatype->size = sizeof(float);
        datatype->name = "float";
    } else if (!strcmp(optarg, "double")) {
        datatype->type = NVSHMEM_DOUBLE;
        datatype->size = sizeof(double);
        datatype->name = "double";
    } else if (!strcmp(optarg, "uint")) {
        datatype->type = NVSHMEM_UINT;
        datatype->size = sizeof(unsigned int);
        datatype->name = "uint";
    } else if (!strcmp(optarg, "int32")) {
        datatype->type = NVSHMEM_INT32;
        datatype->size = sizeof(int32_t);
        datatype->name = "int32";
    } else if (!strcmp(optarg, "int64")) {
        datatype->type = NVSHMEM_INT64;
        datatype->size = sizeof(int64_t);
        datatype->name = "int64";
    } else if (!strcmp(optarg, "uint32")) {
        datatype->type = NVSHMEM_UINT32;
        datatype->size = sizeof(int32_t);
        datatype->name = "uint32";
    } else if (!strcmp(optarg, "uint64")) {
        datatype->type = NVSHMEM_UINT64;
        datatype->size = sizeof(uint64_t);
        datatype->name = "uint64";
    } else if (strstr(optarg, "fp16")) {
        datatype->type = NVSHMEM_FP16;
        datatype->size = sizeof(half);
        datatype->name = "fp16";
    } else if (!strcmp(optarg, "bf16")) {
        datatype->type = NVSHMEM_BF16;
        datatype->size = sizeof(__nv_bfloat16);
        datatype->name = "bf16";
    }
}

static void reduce_op_parse(char *str, reduce_op_t *reduce_op) {
    if (!strcmp(optarg, "min")) {
        reduce_op->type = NVSHMEM_MIN;
        reduce_op->name = "min";
    } else if (!strcmp(optarg, "max")) {
        reduce_op->type = NVSHMEM_MAX;
        reduce_op->name = "max";
    } else if (!strcmp(optarg, "sum")) {
        reduce_op->type = NVSHMEM_SUM;
        reduce_op->name = "sum";
    } else if (!strcmp(optarg, "prod")) {
        reduce_op->type = NVSHMEM_PROD;
        reduce_op->name = "prod";
    } else if (!strcmp(optarg, "and")) {
        reduce_op->type = NVSHMEM_AND;
        reduce_op->name = "and";
    } else if (!strcmp(optarg, "or")) {
        reduce_op->type = NVSHMEM_OR;
        reduce_op->name = "or";
    } else if (!strcmp(optarg, "xor")) {
        reduce_op->type = NVSHMEM_XOR;
        reduce_op->name = "xor";
    }
}

void atomic_op_parse(char *str, amo_t *amo) {
    size_t string_length = strnlen(str, 20);

    if (strncmp(str, "inc", string_length) == 0) {
        amo->type = AMO_INC;
        amo->name = "inc";
    } else if (strncmp(str, "fetch_inc", string_length) == 0) {
        amo->type = AMO_FETCH_INC;
        amo->name = "fetch_inc";
    } else if (strncmp(str, "set", string_length) == 0) {
        amo->type = AMO_SET;
        amo->name = "set";
    } else if (strncmp(str, "add", string_length) == 0) {
        amo->type = AMO_ADD;
        amo->name = "add";
    } else if (strncmp(str, "fetch_add", string_length) == 0) {
        amo->type = AMO_FETCH_ADD;
        amo->name = "fetch_add";
    } else if (strncmp(str, "and", string_length) == 0) {
        amo->type = AMO_AND;
        amo->name = "and";
    } else if (strncmp(str, "fetch_and", string_length) == 0) {
        amo->type = AMO_FETCH_AND;
        amo->name = "fetch_and";
    } else if (strncmp(str, "or", string_length) == 0) {
        amo->type = AMO_OR;
        amo->name = "or";
    } else if (strncmp(str, "fetch_or", string_length) == 0) {
        amo->type = AMO_FETCH_OR;
        amo->name = "fetch_or";
    } else if (strncmp(str, "xor", string_length) == 0) {
        amo->type = AMO_XOR;
        amo->name = "xor";
    } else if (strncmp(str, "fetch_xor", string_length) == 0) {
        amo->type = AMO_FETCH_XOR;
        amo->name = "fetch_xor";
    } else if (strncmp(str, "swap", string_length) == 0) {
        amo->type = AMO_SWAP;
        amo->name = "swap";
    } else if (strncmp(str, "compare_swap", string_length) == 0) {
        amo->type = AMO_COMPARE_SWAP;
        amo->name = "compare_swap";
    } else {
        amo->type = AMO_ACK;
        amo->name = "ack";
    }
}

/* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
static inline int atol_scaled(const char *str, size_t *out) {
    int scale, n;
    double p = -1.0;
    char f;
    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k':
            case 'K':
                scale = 10;
                break;
            case 'm':
            case 'M':
                scale = 20;
                break;
            case 'g':
            case 'G':
                scale = 30;
                break;
            case 't':
            case 'T':
                scale = 40;
                break;
            default:
                return 1;
        }
    } else if (p < 0) {
        return 1;
    } else
        scale = 0;

    *out = (size_t)ceil(p * (1lu << scale));
    return 0;
}

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

void get_coll_info(double *algBw, double *busBw, const char *job_name, double usec, int npes,
                   uint64_t size) {
    double factor;
    // convert to seconds
    double sec = usec / 1.0E6;

    if (strcmp(job_name, "reduction") == 0 || strcmp(job_name, "reduction_on_stream") == 0 ||
        strcmp(job_name, "device_reduction") == 0) {
        factor = ((double)2 * (npes - 1)) / ((double)(npes));
    } else if (strcmp(job_name, "broadcast") == 0 || strcmp(job_name, "broadcast_on_stream") == 0 ||
               strcmp(job_name, "bcast_device") == 0) {
        factor = 1;
    } else if (strcmp(job_name, "alltoall") == 0 || strcmp(job_name, "alltoall_on_stream") == 0 ||
               strcmp(job_name, "alltoall_device") == 0 || strcmp(job_name, "fcollect") == 0 ||
               strcmp(job_name, "fcollect_on_stream") == 0 ||
               strcmp(job_name, "fcollect_device") == 0 || strcmp(job_name, "reducescatter") == 0 ||
               strcmp(job_name, "reducescatter_on_stream") == 0 ||
               strcmp(job_name, "device_reducescatter") == 0) {
        factor = ((double)(npes - 1)) / ((double)(npes));
    } else {
        printf("Job Name %s bandwidth factor not set. Using 1 values for bw.\n", job_name);
        *algBw = 1;
        *busBw = 1;
        return;
    }

    *algBw = (double)(size) / 1.0E9 / sec;
    *busBw = *algBw * factor;
}

uint64_t calculate_collective_size(const char *coll_name, uint64_t num_elems, uint64_t type_size,
                                   int npes) {
    if (!coll_name) {
        printf("WARNING: NULL collective operation name, using default size calculation\n");
        return num_elems * type_size;
    }

    uint64_t size = num_elems * type_size;

    // Handle collective operations that scale with npes
    if (strstr(coll_name, "alltoall") || strstr(coll_name, "fcollect") ||
        strstr(coll_name, "reducescatter")) {
        size *= npes;
    }

    DEBUG_PRINT("Collective: %s, Elements: %lu, Type size: %lu, PEs: %d, Total size: %lu\n",
                coll_name, num_elems, type_size, npes, size);

    return size;
}

double calculate_msgrate(size_t messages_per_iteration, size_t iterations, float milliseconds) {
    return static_cast<double>(messages_per_iteration) * iterations / (milliseconds * MS_TO_S);
}

tuple<double, double, double> get_latency_metrics(double *values, int num_values) {
    double min, max, sum;
    int i = 0;
    min = max = values[0];
    sum = 0.0;
    int num_zeroes = 0;

    while (i < num_values) {
        if (values[i] == 0) {
            i++;
            num_zeroes++;
            continue;
        }

        auto v = values[i];
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
        sum += v;
        i++;
    }
    // Don't count if they are zero
    double avg = (double)sum / (num_values - num_zeroes);
    return make_tuple(avg, min, max);
}

void perf_stats_add(perf_stats_t &stats, double value) {
    ++stats.count;
    if (stats.count == 1) {
        stats.mean = value;
        stats.m2 = 0.0;
        stats.min = value;
        stats.max = value;
        return;
    }

    const double delta = value - stats.mean;
    stats.mean += delta / static_cast<double>(stats.count);
    stats.m2 += delta * (value - stats.mean);
    stats.min = std::min(stats.min, value);
    stats.max = std::max(stats.max, value);
}

double perf_stats_stddev(const perf_stats_t *stats) {
    const auto &state = *stats;
    return state.count > 1 ? std::sqrt(state.m2 / static_cast<double>(state.count - 1))
                           : std::numeric_limits<double>::quiet_NaN();
}

template <typename T>
static void append_column(std::ostringstream &builder, const T &value, int width) {
    builder << std::left;
    if constexpr (std::is_same_v<T, double>) builder << std::fixed << std::setprecision(6);
    builder << std::setw(width) << value << "  ";
}

static void append_stddev_column(std::ostringstream &builder, const perf_stats_t &stats,
                                 int width) {
    if (stats.count > 1) {
        append_column(builder, perf_stats_stddev(&stats), width);
    } else {
        append_column(builder, "NA", width);
    }
}

static void append_table_title(std::ostringstream &builder, std::string_view job_name) {
    builder << '#' << std::right << std::setw(10) << job_name << '\n';
}

static void append_perf_result(std::ostringstream &builder, std::string_view job_name,
                               std::string_view subjob_name, uint64_t size,
                               std::string_view output_var, std::string_view units, char plus_minus,
                               double mean, const perf_stats_t *stats) {
    if (stats == nullptr) {
        // Emit the legacy machine-readable row without repetition statistics.
        builder << "&&&& PERF " << job_name << "___" << subjob_name << "___size__" << size << "___"
                << output_var << ' ' << std::fixed << std::setprecision(6) << mean << ' '
                << plus_minus << units << '\n';
    } else {
        // Emit the machine-readable row with statistics across timed repetitions.
        builder << "&&&& PERF_STATS " << job_name << "___" << subjob_name << "___size__" << size
                << "___" << output_var << ' ' << plus_minus << units << " mean=" << std::fixed
                << std::setprecision(6) << mean << " stddev=";
        if (stats->count > 1) {
            builder << perf_stats_stddev(stats);
        } else {
            builder << "NA";
        }
        builder << " min=" << stats->min << " max=" << stats->max << " repetitions=" << stats->count
                << '\n';
    }
}

static bool print_machine_table(std::string_view job_name, std::string_view subjob_name,
                                std::string_view output_var, std::string_view units,
                                char plus_minus, const uint64_t *sizes, const double *values,
                                const perf_stats_t *stats, int num_entries) {
    const char *machine_readable_output = std::getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (machine_readable_output == nullptr || std::atoi(machine_readable_output) == 0) return false;

    std::ostringstream builder;
    builder << job_name << '\n';
    for (int i = 0; i < num_entries; ++i) {
        const perf_stats_t *entry_stats = stats != nullptr ? &stats[i] : nullptr;
        const double value = values != nullptr ? values[i] : 0.0;
        if (sizes[i] == 0 || (entry_stats != nullptr ? entry_stats->count == 0 : value == 0.0))
            continue;
        const double mean = entry_stats != nullptr ? entry_stats->mean : value;
        append_perf_result(builder, job_name, subjob_name, sizes[i], output_var, units, plus_minus,
                           mean, entry_stats);
    }
    std::fputs(builder.str().c_str(), stdout);
    return true;
}

static void print_basic_table_impl(const char *job_name, const char *subjob_name,
                                   const char *output_var, const char *units, char plus_minus,
                                   uint64_t *sizes, const double *values, const perf_stats_t *stats,
                                   int num_entries) {
    if (print_machine_table(job_name, subjob_name, output_var, units, plus_minus, sizes, values,
                            stats, num_entries))
        return;

    const bool show_stats = stats != nullptr;
    const std::string mean_header = std::string{output_var} + " (" + units + ')';
    std::ostringstream builder;
    append_table_title(builder, job_name);
    append_column(builder, "size(B)", 10);
    append_column(builder, "scope", 8);
    if (show_stats) {
        append_column(builder, mean_header, 16);
        append_column(builder, std::string{"stddev ("} + units + ')', 16);
        append_column(builder, std::string{"min ("} + units + ')', 16);
        append_column(builder, std::string{"max ("} + units + ')', 16);
        builder << std::left << std::setw(12) << "repetitions" << '\n';
    } else {
        builder << std::left << std::setw(16) << mean_header << '\n';
    }

    for (int i = 0; i < num_entries; ++i) {
        const perf_stats_t *entry_stats = show_stats ? &stats[i] : nullptr;
        const double value = values != nullptr ? values[i] : 0.0;
        if (sizes[i] == 0 || (entry_stats != nullptr ? entry_stats->count == 0 : value == 0.0))
            continue;
        append_column(builder, sizes[i], 10);
        append_column(builder, subjob_name, 8);
        const double mean = entry_stats != nullptr ? entry_stats->mean : value;
        if (show_stats) {
            append_column(builder, mean, 16);
            append_stddev_column(builder, *entry_stats, 16);
            append_column(builder, entry_stats->min, 16);
            append_column(builder, entry_stats->max, 16);
            builder << std::left << std::setw(12) << entry_stats->count << '\n';
        } else {
            builder << std::left << std::fixed << std::setprecision(6) << std::setw(16) << mean
                    << '\n';
        }
    }
    std::fputs(builder.str().c_str(), stdout);
}

static void print_device_collective_table_impl(const char *job_name, const char *subjob_name,
                                               const char *output_var, const char *units,
                                               char plus_minus, uint64_t *sizes,
                                               const double *values, const perf_stats_t *stats,
                                               int num_entries) {
    if (print_machine_table(job_name, subjob_name, output_var, units, plus_minus, sizes, values,
                            stats, num_entries))
        return;

    const bool show_stats = stats != nullptr;
    const std::string_view job{job_name};
    const std::string_view subjob{subjob_name};
    const size_t last_delimiter = subjob.rfind('-');
    const std::string_view type_and_operation = subjob.substr(0, last_delimiter);
    const std::string_view scope =
        last_delimiter == std::string_view::npos ? "" : subjob.substr(last_delimiter + 1);
    const size_t first_delimiter = type_and_operation.find('-');
    const std::string_view datatype_name = type_and_operation.substr(0, first_delimiter);
    const std::string_view operation = first_delimiter == std::string_view::npos
                                           ? ""
                                           : type_and_operation.substr(first_delimiter + 1);
    int datatype_size = 4;
    if (subjob.find("64-bit") != std::string_view::npos) {
        datatype_size = 8;
    } else if (subjob.find("32-bit") == std::string_view::npos) {
        datatype_t parsed = {NVSHMEM_INT, 4, "int"};
        const std::string datatype{datatype_name};
        datatype_parse(datatype.c_str(), &parsed);
        datatype_size = parsed.size;
    }

    const bool reduction = job == "device_reduction" || job == "device_reducescatter";
    const std::string_view type = reduction ? datatype_name : type_and_operation;
    const int npes = nvshmem_n_pes();
    std::ostringstream builder;
    append_table_title(builder, job);
    append_column(builder, "size(B)", 10);
    append_column(builder, "count", 8);
    append_column(builder, "type", 8);
    if (reduction) append_column(builder, "redop", 8);
    append_column(builder, "scope", 8);
    append_column(builder, "latency(us)", 16);
    if (show_stats) {
        append_column(builder, "stddev(us)", 16);
        append_column(builder, "min_lat(us)", 16);
        append_column(builder, "max_lat(us)", 16);
        append_column(builder, "repetitions", 12);
    }
    append_column(builder, "algbw(GB/s)", 12);
    builder << std::left << std::setw(12) << "busbw(GB/s)" << '\n';

    for (int i = 0; i < num_entries; ++i) {
        const perf_stats_t *entry_stats = show_stats ? &stats[i] : nullptr;
        const double value = values != nullptr ? values[i] : 0.0;
        if (sizes[i] == 0 || (entry_stats != nullptr ? entry_stats->count == 0 : value == 0.0))
            continue;
        const double mean = entry_stats != nullptr ? entry_stats->mean : value;
        double algbw = 0.0;
        double busbw = 0.0;
        get_coll_info(&algbw, &busbw, job_name, mean, npes, sizes[i]);
        append_column(builder, sizes[i], 10);
        append_column(builder, sizes[i] / datatype_size, 8);
        append_column(builder, type, 8);
        if (reduction) append_column(builder, operation, 8);
        append_column(builder, scope, 8);
        append_column(builder, mean, 16);
        if (show_stats) {
            append_stddev_column(builder, *entry_stats, 16);
            append_column(builder, entry_stats->min, 16);
            append_column(builder, entry_stats->max, 16);
            append_column(builder, entry_stats->count, 12);
        }
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(12) << algbw
                << "  ";
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(12) << busbw
                << '\n';
    }
    std::fputs(builder.str().c_str(), stdout);
}

static void print_host_collective_table_impl(const char *job_name, const char *subjob_name,
                                             const char *output_var, const char *units,
                                             char plus_minus, uint64_t *sizes, double **values,
                                             size_t num_iters, const perf_stats_t *stats,
                                             int num_entries) {
    const bool show_stats = stats != nullptr;
    const std::string_view job{job_name};
    const std::string_view subjob{subjob_name};

    const char *machine_readable_output = std::getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (machine_readable_output != nullptr && std::atoi(machine_readable_output) != 0) {
        std::ostringstream builder;
        builder << job << '\n';
        for (int i = 0; i < num_entries; ++i) {
            const perf_stats_t *entry_stats = show_stats ? &stats[i] : nullptr;
            if (sizes[i] == 0 || (show_stats ? entry_stats->count == 0 : values[i][0] == 0.0))
                continue;
            double mean = entry_stats != nullptr ? entry_stats->mean : 0.0;
            if (!show_stats) {
                double min = 0.0;
                double max = 0.0;
                tie(mean, min, max) = get_latency_metrics(values[i], num_iters);
            }
            append_perf_result(builder, job, subjob, sizes[i], output_var, units, plus_minus, mean,
                               entry_stats);
        }
        std::fputs(builder.str().c_str(), stdout);
        return;
    }

    const size_t delimiter = subjob.find('-');
    const std::string_view type = subjob.substr(0, delimiter);
    const std::string_view operation =
        delimiter == std::string_view::npos ? "None" : subjob.substr(delimiter + 1);
    datatype_t parsed = {NVSHMEM_INT, 4, "int"};
    const std::string datatype{type};
    datatype_parse(datatype.c_str(), &parsed);

    const bool reduction = job == "reduction_on_stream" || job == "reducescatter_on_stream";
    const int npes = nvshmem_n_pes();
    std::ostringstream builder;
    append_table_title(builder, job);
    append_column(builder, "size(B)", 10);
    append_column(builder, "count", 8);
    append_column(builder, "type", 8);
    if (reduction) append_column(builder, "redop", 8);
    append_column(builder, "latency(us)", 16);
    if (show_stats) append_column(builder, "stddev(us)", 16);
    append_column(builder, "min_lat(us)", 16);
    append_column(builder, "max_lat(us)", 16);
    if (show_stats) append_column(builder, "repetitions", 12);
    append_column(builder, "algbw(GB/s)", 12);
    builder << std::left << std::setw(12) << "busbw(GB/s)" << '\n';

    for (int i = 0; i < num_entries; ++i) {
        const perf_stats_t *entry_stats = show_stats ? &stats[i] : nullptr;
        if (sizes[i] == 0 || (show_stats ? entry_stats->count == 0 : values[i][0] == 0.0)) continue;
        double mean = 0.0;
        double min = 0.0;
        double max = 0.0;
        if (show_stats) {
            mean = entry_stats->mean;
            min = entry_stats->min;
            max = entry_stats->max;
        } else {
            tie(mean, min, max) = get_latency_metrics(values[i], num_iters);
        }
        double algbw = 0.0;
        double busbw = 0.0;
        get_coll_info(&algbw, &busbw, job_name, mean, npes, sizes[i]);
        append_column(builder, sizes[i], 10);
        append_column(builder, sizes[i] / parsed.size, 8);
        append_column(builder, parsed.name, 8);
        if (reduction) append_column(builder, operation, 8);
        append_column(builder, mean, 16);
        if (show_stats) append_stddev_column(builder, *entry_stats, 16);
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(16) << min << "  ";
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(16) << max << "  ";
        if (show_stats) append_column(builder, entry_stats->count, 12);
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(12) << algbw
                << "  ";
        builder << std::left << std::fixed << std::setprecision(3) << std::setw(12) << busbw
                << '\n';
    }
    std::fputs(builder.str().c_str(), stdout);
}

void print_basic_table(const char *job_name, const char *subjob_name, const char *output_var,
                       const char *units, const char plus_minus, uint64_t *size, double *value,
                       int num_entries, const perf_stats_t *stats) {
    print_basic_table_impl(job_name, subjob_name, output_var, units, plus_minus, size, value,
                           repetitions_requested ? stats : nullptr, num_entries);
}

void print_device_collective_table(const char *job_name, const char *subjob_name,
                                   const char *output_var, const char *units, const char plus_minus,
                                   uint64_t *size, double *value, int num_entries,
                                   const perf_stats_t *stats) {
    print_device_collective_table_impl(job_name, subjob_name, output_var, units, plus_minus, size,
                                       value, repetitions_requested ? stats : nullptr, num_entries);
}

void print_host_collective_table(const char *job_name, const char *subjob_name,
                                 const char *output_var, const char *units, const char plus_minus,
                                 uint64_t *size, double **values, int num_entries, size_t num_iters,
                                 const perf_stats_t *stats) {
    print_host_collective_table_impl(job_name, subjob_name, output_var, units, plus_minus, size,
                                     values, num_iters, repetitions_requested ? stats : nullptr,
                                     num_entries);
}

size_t min_size = 4;
size_t max_size = min_size * 1024 * 1024;
size_t num_blocks = 32;
size_t threads_per_block = 256;
size_t iters = 10;
size_t warmup_iters = 5;
size_t repetitions = 1;
bool repetitions_requested = false;
size_t step_factor = 2;
size_t max_size_log = 1;
size_t stride = 1;
size_t mem_handle_type = MEM_TYPE_AUTO;
bool bidirectional = false;
bool report_msgrate = false;
bool use_graph = false;
bool use_mmap = false;
bool use_egm = false;
bool use_smem = true;

datatype_t datatype = {NVSHMEM_INT, 4, "int"};
reduce_op_t reduce_op = {NVSHMEM_SUM, "sum"};
threadgroup_scope_t threadgroup_scope = {NVSHMEM_ALL_SCOPES, "all_scopes"};
amo_t test_amo = {AMO_INC, "inc"};
putget_issue_t putget_issue = {ON_STREAM, "on_stream"};
dir_t dir = {WRITE, "write"};

// tracks mmap addr -> {user buff, size, handle}
std::unordered_map<void *, std::tuple<void *, size_t, CUmemGenericAllocationHandle>> mmaped_buffers;

void *nvml_handle = nullptr;
struct nvml_function_table nvml_ftable;
const char *env_value = nullptr;

static bool parse_bool_arg(const char *option, const char *arg, bool default_value) {
    int value = 0;
    std::string_view sv{arg};

    const auto result = std::from_chars(sv.data(), sv.data() + sv.size(), value);
    if (result.ec == std::errc{} && result.ptr == sv.data() + sv.size()) {
        return value != 0;
    }

    fprintf(stderr, "Warning: invalid %s='%s'; defaulting to %s\n", option, arg,
            default_value ? "enabled" : "disabled");
    return default_value;
}

void read_args(int argc, char **argv) {
    int c;
    static struct option long_options[] = {{"bidir", no_argument, 0, 0},
                                           {"msgrate", no_argument, 0, 0},
                                           {"cudagraph", no_argument, 0, 0},
                                           {"dir", required_argument, 0, 0},
                                           {"issue", required_argument, 0, 0},
                                           {"mmap", no_argument, 0, 0},
                                           {"egm", no_argument, 0, 0},
                                           {"use_smem", required_argument, 0, 0},
                                           {"help", no_argument, 0, 'h'},
                                           {"min_size", required_argument, 0, 'b'},
                                           {"max_size", required_argument, 0, 'e'},
                                           {"step", required_argument, 0, 'f'},
                                           {"iters", required_argument, 0, 'n'},
                                           {"warmup_iters", required_argument, 0, 'w'},
                                           {"repetitions", required_argument, 0, 'r'},
                                           {"ctas", required_argument, 0, 'c'},
                                           {"threads_per_cta", required_argument, 0, 't'},
                                           {"datatype", required_argument, 0, 'd'},
                                           {"reduce_op", required_argument, 0, 'o'},
                                           {"scope", required_argument, 0, 's'},
                                           {"atomic_op", required_argument, 0, 'a'},
                                           {"stride", required_argument, 0, 'i'},
                                           {"mem_handle_type", required_argument, 0, 'm'},
                                           {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;
    while ((c = getopt_long(argc, argv, "hb:e:f:n:w:r:c:t:d:o:s:a:i:m:", long_options,
                            &option_index)) != -1) {
        switch (c) {
            case 'h':
                printf(
                    "Accepted arguments: \n"
                    "-b, --min_size <minbytes> \n"
                    "-e, --max_size <maxbytes> \n"
                    "-f, --step <step factor for message sizes> \n"
                    "-n, --iters <number of iterations> \n"
                    "-w, --warmup_iters <number of warmup iterations> \n"
                    "-r, --repetitions <number of timed repetitions> \n"
                    "-c, --ctas <number of CTAs to launch> (used in some device pt-to-pt tests) \n"
                    "-t, --threads_per_cta <number of threads per block> (used in some device "
                    "pt-to-pt tests) \n"
                    "-d, --datatype: "
                    "<int, int32_t, uint32_t, int64_t, uint64_t, long, longlong, ulonglong, size, "
                    "ptrdiff, "
                    "float, double, fp16, bf16> \n"
                    "-o, --reduce_op <min, max, sum, prod, and, or, xor> \n"
                    "-s, --scope <thread, warp, block, all> \n"
                    "-i, --stride stride between elements \n"
                    "-a, --atomic_op <inc, add, and, or, xor, set, swap, fetch_<inc, add, and, or, "
                    "xor>, compare_swap> \n"
                    "--bidir: run bidirectional test \n"
                    "--msgrate: report logical operation rate in bandwidth tests "
                    "(MMPS for messaging, MOPS for stores)\n"
                    "--dir: <read, write> (whether to run put or get operations) \n"
                    "--issue: <on_stream, host> (applicable in some host pt-to-pt tests) \n"
                    "--mmap (Use mmaped buffer) \n"
                    "--egm (Use EGM memory for mmaped buffer) \n"
                    "--use_smem <0|1> (Enable shared-memory registration in TMA-capable tests) \n"
                    "-m, --mem_handle_type: <0:auto, 1:posix_fd, 2:fabric> (for mmaped buffer) \n"
                    "--cudagraph (Use CUDA graph to amortize launch overhead) \n");
                exit(0);
            case 0:
                if (strcmp(long_options[option_index].name, "bidir") == 0) {
                    bidirectional = true;
                } else if (strcmp(long_options[option_index].name, "msgrate") == 0) {
                    report_msgrate = true;
                } else if (strcmp(long_options[option_index].name, "dir") == 0) {
                    if (strcmp(optarg, "read") == 0) {
                        dir.type = READ;
                        dir.name = "read";
                    } else {
                        dir.type = WRITE;
                        dir.name = "write";
                    }
                } else if (strcmp(long_options[option_index].name, "issue") == 0) {
                    if (strcmp(optarg, "on_stream") == 0) {
                        putget_issue.type = ON_STREAM;
                        putget_issue.name = "on_stream";
                    } else {
                        putget_issue.type = HOST;
                        putget_issue.name = "host";
                    }
                } else if (strcmp(long_options[option_index].name, "cudagraph") == 0) {
                    use_graph = true;
                } else if (strcmp(long_options[option_index].name, "mmap") == 0) {
                    use_mmap = true;
                } else if (strcmp(long_options[option_index].name, "egm") == 0) {
                    use_egm = true;
                } else if (strcmp(long_options[option_index].name, "use_smem") == 0) {
                    use_smem = parse_bool_arg("--use_smem", optarg, true);
                }
                break;
            case 'b':
                atol_scaled(optarg, &min_size);
                break;
            case 'e':
                atol_scaled(optarg, &max_size);
                break;
            case 'f':
                atol_scaled(optarg, &step_factor);
                break;
            case 'n':
                atol_scaled(optarg, &iters);
                break;
            case 'w':
                atol_scaled(optarg, &warmup_iters);
                break;
            case 'r':
                if (atol_scaled(optarg, &repetitions) || repetitions == 0) {
                    fprintf(stderr, "--repetitions must be an integer greater than zero\n");
                    exit(EXIT_FAILURE);
                }
                repetitions_requested = true;
                break;
            case 'c':
                atol_scaled(optarg, &num_blocks);
                break;
            case 't':
                atol_scaled(optarg, &threads_per_block);
                break;
            case 'm':
                atol_scaled(optarg, &mem_handle_type);
                break;
            case 'i':
                atol_scaled(optarg, &stride);
                break;
            case 'd':
                datatype_parse(optarg, &datatype);
                break;
            case 'o':
                reduce_op_parse(optarg, &reduce_op);
                break;
            case 's':
                if (!strcmp(optarg, "thread")) {
                    threadgroup_scope.type = NVSHMEM_THREAD;
                    threadgroup_scope.name = "thread";
                } else if (!strcmp(optarg, "warp")) {
                    threadgroup_scope.type = NVSHMEM_WARP;
                    threadgroup_scope.name = "warp";
                } else if (!strcmp(optarg, "block")) {
                    threadgroup_scope.type = NVSHMEM_BLOCK;
                    threadgroup_scope.name = "block";
                }
                break;
            case 'a':
                atomic_op_parse(optarg, &test_amo);
                break;
            case '?':
                if (optopt == 'c')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                return;
            default:
                abort();
        }
    }
    max_size_log = 1;
    size_t tmp = max_size;
    while (tmp) {
        max_size_log += 1;
        tmp >>= 1;
    }

    assert(min_size <= max_size);
}

#define LOAD_SYM(handle, symbol, funcptr, optional, ret)        \
    do {                                                        \
        void **cast = (void **)&funcptr;                        \
        void *tmp = dlsym(handle, symbol);                      \
        *cast = tmp;                                            \
        if (*cast == NULL && !optional) {                       \
            NVSHMEMI_ERROR_PRINT("Retrieve %s failed", symbol); \
            ret = NVSHMEMX_ERROR_INTERNAL;                      \
        }                                                       \
    } while (0)

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    int status = 0;
    char path[1024];
    env_value = (const char *)getenv("NVSHMEM_CUDA_PATH");
    if (!env_value)
        snprintf(path, 1024, "%s", "libnvidia-ml.so.1");
    else
        snprintf(path, 1024, "%s/%s", env_value, "libnvidia-ml.so.1");

    *nvml_handle = dlopen(path, RTLD_NOW);
    if (!(*nvml_handle)) {
        DEBUG_PRINT("NVML library not found. %s", path);
        status = -1;
    } else {
        DEBUG_PRINT("NVML library found. %s", path);
        LOAD_SYM(*nvml_handle, "nvmlInit", nvml_ftable->nvmlInit, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlShutdown", nvml_ftable->nvmlShutdown, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetHandleByPciBusId",
                 nvml_ftable->nvmlDeviceGetHandleByPciBusId, 0, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetP2PStatus", nvml_ftable->nvmlDeviceGetP2PStatus, 0,
                 status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetGpuFabricInfoV",
                 nvml_ftable->nvmlDeviceGetGpuFabricInfoV, 1, status);
        LOAD_SYM(*nvml_handle, "nvmlDeviceGetFieldValues", nvml_ftable->nvmlDeviceGetFieldValues, 0,
                 status);
    }

    if (status != 0) {
        nvshmemi_nvml_ftable_fini(nvml_ftable, nvml_handle);
    }
    return status;
}

void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    if (*nvml_handle) {
        dlclose(*nvml_handle);
        *nvml_handle = NULL;
        memset(nvml_ftable, 0, sizeof(*nvml_ftable));
    }
}

bool is_mnnvl_supported(int dev_id) {
    nvmlGpuFabricInfoV_t fabricInfo = {};
    const unsigned char zero[NVML_GPU_FABRIC_UUID_LEN] = {0};
    cudaDeviceProp prop;
    char pcie_bdf[50] = {0};
    int nbytes = 0;
    int attr;
    nvmlReturn_t nvml_status;
    nvmlDevice_t local_device;
    CUdevice my_dev;
    int cuda_drv_version;
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    CUDA_CHECK(cudaDriverGetVersion(&cuda_drv_version));
    CU_CHECK(cuDeviceGet(&my_dev, dev_id));

    /* start NVML Library */
    if (nvshmemi_nvml_ftable_init(&nvml_ftable, &nvml_handle) != 0) {
        DEBUG_PRINT("Unable to open NVML library, disabling MNNVL\n");
        return false;
    }

    nvml_status = nvml_ftable.nvmlInit();
    if (nvml_status != NVML_SUCCESS) {
        DEBUG_PRINT("Unable to initialize NVML library, disabling MNNVL. NVML error: %d\n",
                    nvml_status);
        return false;
    }

    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    nbytes =
        snprintf(pcie_bdf, 50, "%x:%x:%x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (nbytes < 0 || nbytes > 50) {
        DEBUG_PRINT("Unable to set device pcie bdf for our local device, disabling MNNVL\n");
        return false;
    }

    bool disable_mnnvl = false;
    const char *env_value = (const char *)getenv("NVSHMEM_DISABLE_MNNVL");
    if (env_value && (env_value[0] == '0' || env_value[0] == 'N' || env_value[0] == 'n' ||
                      env_value[0] == 'F' || env_value[0] == 'f')) {
        disable_mnnvl = true;
    } else if (env_value) {
        disable_mnnvl = true;
    }

    if (cuda_drv_version >= 12040 && prop.major >= 9 && !disable_mnnvl) {
        nvml_status = nvml_ftable.nvmlDeviceGetHandleByPciBusId(pcie_bdf, &local_device);
        if (nvml_status != NVML_SUCCESS) {
            DEBUG_PRINT("nvmlDeviceGetHandleByPciBusId failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

        /* Some platforms with older driver may not support this API, so bypass MNNVL discovery */
        if (nvml_ftable.nvmlDeviceGetGpuFabricInfoV == NULL) {
            DEBUG_PRINT("nvmlDeviceGetGpuFabricInfoV not found, MNNVL not supported\n");
            return false;
        }

        nvml_status = nvml_ftable.nvmlDeviceGetGpuFabricInfoV(local_device, &fabricInfo);
        if (nvml_status != NVML_SUCCESS) {
            DEBUG_PRINT("nvmlDeviceGetGpuFabricInfoV failed %d, disabling MNNVL\n", nvml_status);
            return false;
        }

        CU_CHECK(
            cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, my_dev));
        if (attr <= 0) {
            DEBUG_PRINT("CUDA EGM fabric not supported\n");
            return false;
        }

        if (fabricInfo.state < NVML_GPU_FABRIC_STATE_COMPLETED ||
            memcmp(fabricInfo.clusterUuid, zero, NVML_GPU_FABRIC_UUID_LEN) == 0) {
            DEBUG_PRINT("MNNVL not supported\n");
            return false;
        }
    } else {
        DEBUG_PRINT("MNNVL disabled\n");
        return false;
    }

    nvml_status = nvml_ftable.nvmlShutdown();
    if (nvml_status != NVML_SUCCESS) {
        DEBUG_PRINT("Unable to stop NVML library in NVSHMEM. NVML error: %d\n", nvml_status);
        // is this a fatal error?
        return false;
    }
    nvshmemi_nvml_ftable_fini(&nvml_ftable, &nvml_handle);
    return true;
}

void *allocate_mmap_buffer(size_t size, int mem_fabric_handle_type, bool use_egm, bool reset_zero) {
    mype = nvshmem_my_pe();
    if (!mype) DEBUG_PRINT("allocating mmap buffer\n");
    CUmemAllocationProp prop = {};
    int dev_id, numa_id;
    size_t granularity = MEM_GRANULARITY;
    int cuda_drv_version;
    CUdevice my_dev;
    CUDA_CHECK(cudaDriverGetVersion(&cuda_drv_version));
    // Application should set the device id before calling this function
    // same as nvshmem_malloc()
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CU_CHECK(cuDeviceGet(&my_dev, dev_id));
    prop.location.id = dev_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (use_egm) {
        if (!mype) DEBUG_PRINT("using EGM memory\n");
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        CU_CHECK(cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, my_dev));
        prop.location.id = numa_id;
    } else {
        prop.allocFlags.gpuDirectRDMACapable = 1;
    }

    prop.requestedHandleTypes =
        (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if ((mem_handle_type == MEM_TYPE_AUTO) && is_mnnvl_supported(dev_id)) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    }
    // override if user specified mem handle type
    if (mem_handle_type == MEM_TYPE_FABRIC) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    } else if (mem_handle_type == MEM_TYPE_POSIX_FD) {
        prop.requestedHandleTypes =
            (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }

    // pad size to be multiple of granularity
    size = ((size + granularity - 1) / granularity) * granularity;
    // printf("Allocating mmap buffer: %d, %d size:%lu\n",mem_handle_type, use_egm, size);
    if (!mype) DEBUG_PRINT("padding buffer size to %lu\n", size);
    void *bufAddr, *mmapedAddr;

    CUmemAccessDesc accessDescriptor[2];
    accessDescriptor[0].location.id = prop.location.id;
    accessDescriptor[0].location.type = prop.location.type;
    accessDescriptor[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (use_egm) {
        // accessDescriptor[0] contains host permissions
        accessDescriptor[1].location.id = dev_id;
        accessDescriptor[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDescriptor[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CUmemGenericAllocationHandle userAllocHandle;

    CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));

    if (use_egm) {
        CU_CHECK(cuMemSetAccess((CUdeviceptr)bufAddr, size, &accessDescriptor[0], 2));
    } else {
        CU_CHECK(cuMemSetAccess((CUdeviceptr)bufAddr, size, &accessDescriptor[0], 1));
    }

    mmapedAddr = (void *)nvshmemx_buffer_register_symmetric(bufAddr, size, 0);
    mmaped_buffers[mmapedAddr] = std::make_tuple(bufAddr, size, userAllocHandle);
    if (reset_zero && mmapedAddr) {
        if (use_egm) {
            memset(mmapedAddr, 0, size);
        } else {
            CUDA_CHECK(cudaMemset(mmapedAddr, 0, size));
        }
    }
    return mmapedAddr;
}

size_t pad_up(size_t size) {
    return ((size + MEM_GRANULARITY - 1) / MEM_GRANULARITY) * MEM_GRANULARITY;
}

void free_mmap_buffer(void *ptr) {
    if (mmaped_buffers.count(ptr) == 0) {
        ERROR_PRINT("mmaped buffer not found %p\n", ptr);
        exit(1);
    }
    void *bufAddr = std::get<0>(mmaped_buffers[ptr]);
    size_t size = std::get<1>(mmaped_buffers[ptr]);
    nvshmemx_buffer_unregister_symmetric(ptr, size);
    // free the user buffer
    CU_CHECK(cuMemUnmap((CUdeviceptr)bufAddr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)bufAddr, size));
    CU_CHECK(cuMemRelease(std::get<2>(mmaped_buffers[ptr])));
    mmaped_buffers.erase(ptr);
}
