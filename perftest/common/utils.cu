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

#include "utils.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include <stdio.h>
#include <errno.h>
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
bool use_cubin = false;

CUmodule mymodule = NULL;

void init_cumodule(const char *str) {
    int init_error = 0;

    char exe_path[1000];
    size_t count = readlink("/proc/self/exe", exe_path, 1000);
    exe_path[count] = '\0';

    char *exe_dir = dirname(exe_path);
    char cubin_path[1000];
    strcpy(cubin_path, exe_dir);
    strcat(cubin_path, "/");
    strcat(cubin_path, str);
    printf("CUBIN Selected: %s\n", cubin_path);
    CU_CHECK(cuModuleLoad(&mymodule, cubin_path));
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
    if (use_cubin) {
        printf("Bitcode Library Testing Method Chosen.\n");
    }
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
    if (use_mpi) {
        MPI_Comm mpi_comm = MPI_COMM_WORLD_PLACEHOLDER;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        select_device();
        nvshmem_barrier_all();

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
        select_device();
        nvshmem_barrier_all();
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
        return;
    }
#endif

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    select_device();

    nvshmem_barrier_all();
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

void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries) {
    bool machine_readable = false;
    char buffer[256] = {0};
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, value[i], plus_minus, units);
            }
        }
    } else {
        printf("#%10s\n", job_name);
        snprintf(buffer, 256, "%s (%s)", output_var, units);
        printf("%-10s  %-8s  %-16s\n", "size(B)", "scope", buffer);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("%-10lu  %-8s  %-16.6lf", size[i], subjob_name, value[i]);
                printf("\n");
            }
        }
    }
}

void print_table_v1(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double *value, int num_entries) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    int npes = nvshmem_n_pes();
    double avg, algbw, busbw;

    const char *tokens[3] = { };
    const char *delim = "-";
    char copy[strlen(subjob_name) + 1];
    strcpy(copy, subjob_name);
    const char *token = strtok(copy, delim);

    int token_count = 0;
    while (token != NULL && token_count < 3) {
        tokens[token_count] = token;
        token = strtok(NULL, delim);
        token_count++;
    }

    // Set default values for missing tokens
    for (i = token_count; i < 3; i++) {
        tokens[i] = "";
    }

    int datatype_size = 4;
    if (strstr(subjob_name, "32-bit")) {
        datatype_size = 4;
    } else if (strstr(subjob_name, "64-bit")) {
        datatype_size = 8;
    } else {
        datatype_t datatype = {NVSHMEM_INT, 4, "int"};
        datatype_parse(tokens[0], &datatype);
        datatype_size = datatype.size;
    }

    /* Used for automated test output. It outputs the data in a non human-friendly format. */
    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, value[i], plus_minus, units);
            }
        }
    } else if (strcmp(job_name, "device_reduction") == 0 ||
               strcmp(job_name, "device_reducescatter") == 0) {
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-8s  %-8s  %-16s  %-12s  %-12s\n", "size(B)", "count", "type",
               "redop", "scope", "latency(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                avg = value[i];
                get_coll_info(&algbw, &busbw, job_name, avg, npes, size[i]);
                printf("%-10lu  %-8lu  %-8s  %-8s  %-8s  %-16.6lf  %-12.3lf  %-12.3lf", size[i],
                       (size[i] / datatype_size), tokens[0], tokens[1], tokens[2], avg, algbw,
                       busbw);
                printf("\n");
            }
        }

    } else {
        // recombine first two tokens of subjob_name
        char type[50];  // setting size based on strlen caused buffer overflow
        strcpy(type, subjob_name);
        char *last_delim = strrchr(type, '-');
        if (last_delim != NULL) *last_delim = '\0';
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-8s  %-16s  %-12s  %-12s\n", "size(B)", "count", "type",
               "scope", "latency(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            if (size[i] != 0 && value[i] != 0.00) {
                avg = value[i];
                get_coll_info(&algbw, &busbw, job_name, avg, npes, size[i]);
                printf("%-10lu  %-8lu  %-8s  %-8s  %-16.6lf  %-12.3lf  %-12.3lf", size[i],
                       (size[i] / datatype_size), type, tokens[2], avg, algbw, busbw);
                printf("\n");
            }
        }
    }
}

void print_table_v2(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double **values, int num_entries, size_t num_iters) {
    bool machine_readable = false;
    char *env_value = getenv("NVSHMEM_MACHINE_READABLE_OUTPUT");
    if (env_value) machine_readable = atoi(env_value);
    int i;

    int npes = nvshmem_n_pes();
    double avg, min, max, algbw, busbw = 0;

    /* Used for automated test output. It outputs the data in a non human-friendly format. */
    if (machine_readable) {
        printf("%s\n", job_name);
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            tie(avg, min, max) = get_latency_metrics(value, num_iters);
            if (size[i] != 0 && value[0] != 0.00) {
                printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name,
                       size[i], output_var, avg, plus_minus, units);
            }
        }
    } else if (strcmp(job_name, "reduction_on_stream") == 0 ||
               strcmp(job_name, "reducescatter_on_stream") == 0) {
        /* Splits subjob_name into data type and operation name */
        char **tokens = (char **)malloc(2 * sizeof(char *));
        const char *delim = "-";
        char copy[strlen(subjob_name) + 1];
        strcpy(copy, subjob_name);
        char *token = strtok(copy, delim);
        if (token != NULL) {
            tokens[0] = strdup(token);
            token = strtok(NULL, delim);
            if (token != NULL) {
                tokens[1] = strdup(token);
            } else {
                tokens[1] = strdup("None");
            }
        } else {
            tokens[0] = strdup("None");
            tokens[1] = strdup("None");
        }
        datatype_t datatype = {NVSHMEM_INT, 4, "int"};
        datatype_parse(tokens[0], &datatype);
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-8s  %-16s  %-16s  %-16s  %-12s  %-12s\n", "size(B)", "count",
               "type", "redop", "latency(us)", "min_lat(us)", "max_lat(us)", "algbw(GB/s)",
               "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            if (size[i] != 0 && value[0] != 0.00) {
                tie(avg, min, max) = get_latency_metrics(value, num_iters);
                get_coll_info(&algbw, &busbw, job_name, avg, npes, size[i]);
                printf(
                    "%-10.1lu  %-8lu  %-8s  %-8s  %-16.6lf  %-16.3lf  %-16.3lf  %-12.3lf  %-12.3lf",
                    size[i], size[i] / datatype.size, datatype.name.c_str(), tokens[1], avg, min,
                    max, algbw, busbw);
                printf("\n");
            }
        }
    } else {
        datatype_t datatype = {NVSHMEM_INT, 4, "int"};
        char copy[strlen(subjob_name) + 1];
        strcpy(copy, subjob_name);
        datatype_parse(copy, &datatype);
        printf("#%10s\n", job_name);
        printf("%-10s  %-8s  %-8s  %-16s  %-16s  %-16s  %-12s  %-12s\n", "size(B)", "count", "type",
               "latency(us)", "min_lat(us)", "max_lat(us)", "algbw(GB/s)", "busbw(GB/s)");
        for (i = 0; i < num_entries; i++) {
            auto value = values[i];
            if (size[i] != 0 && value[0] != 0.00) {
                tie(avg, min, max) = get_latency_metrics(value, num_iters);
                get_coll_info(&algbw, &busbw, job_name, avg, npes, size[i]);
                printf("%-10.1lu  %-8lu  %-8s  %-16.6lf  %-16.3lf  %-16.3lf  %-12.3lf  %-12.3lf",
                       size[i], size[i] / datatype.size, datatype.name.c_str(), avg, min, max,
                       algbw, busbw);
                printf("\n");
            }
        }
    }
}

size_t min_size = 4;
size_t max_size = min_size * 1024 * 1024;
size_t num_blocks = 32;
size_t threads_per_block = 256;
size_t iters = 10;
size_t warmup_iters = 5;
size_t step_factor = 2;
size_t max_size_log = 1;
size_t stride = 1;
size_t mem_handle_type = MEM_TYPE_AUTO;
bool bidirectional = false;
bool report_msgrate = false;
bool use_graph = false;
bool use_mmap = false;
bool use_egm = false;

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

void read_args(int argc, char **argv) {
    int c;
    static struct option long_options[] = {{"bidir", no_argument, 0, 0},
                                           {"report_msgrate", no_argument, 0, 0},
                                           {"cudagraph", no_argument, 0, 0},
                                           {"dir", required_argument, 0, 0},
                                           {"issue", required_argument, 0, 0},
                                           {"mmap", no_argument, 0, 0},
                                           {"egm", no_argument, 0, 0},
                                           {"help", no_argument, 0, 'h'},
                                           {"min_size", required_argument, 0, 'b'},
                                           {"max_size", required_argument, 0, 'e'},
                                           {"step", required_argument, 0, 'f'},
                                           {"iters", required_argument, 0, 'n'},
                                           {"warmup_iters", required_argument, 0, 'w'},
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
    while ((c = getopt_long(argc, argv, "hb:e:f:n:w:c:t:d:o:s:a:i:m:", long_options,
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
                    "--msgrate: report message rate (MMPs)\n"
                    "--dir: <read, write> (whether to run put or get operations) \n"
                    "--issue: <on_stream, host> (applicable in some host pt-to-pt tests) \n"
                    "--mmap (Use mmaped buffer) \n"
                    "--egm (Use EGM memory for mmaped buffer) \n"
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

    printf("Runtime options after parsing command line arguments \n");
    printf(
        "min_size: %zu, max_size: %zu, step_factor: %zu, iterations: %zu, warmup iterations: %zu, "
        "number of ctas: %zu, threads per cta: %zu "
        "stride: %zu, datatype: %s, reduce_op: %s, threadgroup_scope: %s, atomic_op: %s, dir: %s, "
        "report_msgrate: %d, bidirectional: %d, putget_issue :%s, use_graph: %d, use_mmap: %d, "
        "mem_handle_type: %zu, use_egm: %d\n",
        min_size, max_size, step_factor, iters, warmup_iters, num_blocks, threads_per_block, stride,
        datatype.name.c_str(), reduce_op.name.c_str(), threadgroup_scope.name.c_str(),
        test_amo.name.c_str(), dir.name.c_str(), report_msgrate, bidirectional,
        putget_issue.name.c_str(), use_graph, use_mmap, mem_handle_type, use_egm);
    printf(
        "Note: Above is full list of options, any given test will use only a subset of these "
        "variables.\n");
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
        DEBUG_PRINT("Unable to initialize NVML library, disabling MNNVL\n");
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
        DEBUG_PRINT("Unable to stop NVML library in NVSHMEM.");
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

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUmemGenericAllocationHandle userAllocHandle;

    CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
    CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
    CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));
    CU_CHECK(
        cuMemSetAccess((CUdeviceptr)bufAddr, size, (const CUmemAccessDesc *)&accessDescriptor, 1));

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
