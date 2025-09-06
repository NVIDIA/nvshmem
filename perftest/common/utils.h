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

#ifndef UTILS
#define UTILS
#include <dlfcn.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <libgen.h>
#include <cuda.h>
#include <string>
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "nvshmem.h"
#include "nvshmemx.h"

#define NVSHMEMI_TEST_STRINGIFY(x) #x

#define MEM_TYPE_AUTO 0
#define MEM_TYPE_POSIX_FD 1
#define MEM_TYPE_FABRIC 2
#define MEM_GRANULARITY 536870912  // 512MB
#define CUMODULE_LOAD(CUMODULE, CUMODULE_PATH, ERROR) \
    CU_CHECK(cuModuleLoad(&CUMODULE, CUMODULE_PATH)); \
    ERROR = nvshmemx_cumodule_init(CUMODULE);

#if CUDART_VERSION < 12040
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED (CUdevice_attribute)128
#define CU_MEM_HANDLE_TYPE_FABRIC (CUmemAllocationHandleType)0x8
#define CU_CTX_SYNC_MEMOPS 0x80
#endif

#if CUDART_VERSION < 12020
#define CU_MEM_LOCATION_TYPE_HOST_NUMA (CUmemLocationType)0x3
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID (CUdevice_attribute)134
#endif

void init_test_case_kernel(CUfunction *kernel, const char *kernel_name);

#ifdef NVSHMEMTEST_MPI_SUPPORT

#define MPI_LOAD_SYM(fn_name)                                                          \
    mpi_fn_table.fn_##fn_name = (fnptr_##fn_name)dlsym(nvshmemi_mpi_handle, #fn_name); \
    if (mpi_fn_table.fn_##fn_name == NULL) {                                           \
        fprintf(stderr, "Unable to load MPI symbol" #fn_name "\n");                    \
        return -1;                                                                     \
    }

typedef int (*fnptr_MPI_Init)(int *argc, char ***argv);
typedef int (*fnptr_MPI_Bcast)(void *buffer, int count, MPI_Datatype datatype, int root,
                               MPI_Comm comm);
typedef int (*fnptr_MPI_Comm_rank)(MPI_Comm comm, int *rank);
typedef int (*fnptr_MPI_Comm_size)(MPI_Comm comm, int *size);
typedef int (*fnptr_MPI_Finalize)(void);
struct nvshmemi_mpi_fn_table {
    fnptr_MPI_Init fn_MPI_Init;
    fnptr_MPI_Bcast fn_MPI_Bcast;
    fnptr_MPI_Comm_rank fn_MPI_Comm_rank;
    fnptr_MPI_Comm_size fn_MPI_Comm_size;
    fnptr_MPI_Finalize fn_MPI_Finalize;
};

extern void *nvshmemi_mpi_handle;
extern struct nvshmemi_mpi_fn_table mpi_fn_table;

#endif

#ifdef NVSHMEMTEST_SHMEM_SUPPORT

#define SHMEM_LOAD_SYM(fn_name)                                                            \
    shmem_fn_table.fn_##fn_name = (fnptr_##fn_name)dlsym(nvshmemi_shmem_handle, #fn_name); \
    if (shmem_fn_table.fn_##fn_name == NULL) {                                             \
        fprintf(stderr, "Unable to load SHMEM symbol" #fn_name "\n");                      \
        return -1;                                                                         \
    }

typedef void (*fnptr_shmem_init)(void);
typedef int (*fnptr_shmem_my_pe)(void);
typedef int (*fnptr_shmem_n_pes)(void);
typedef void (*fnptr_shmem_fcollect64)(void *target, const void *source, size_t nlong, int PE_start,
                                       int logPE_stride, int PE_size, long *pSync);
typedef void *(*fnptr_shmem_malloc)(size_t size);
typedef void (*fnptr_shmem_free)(void *ptr);
typedef void (*fnptr_shmem_finalize)(void);

struct nvshmemi_shmem_fn_table {
    fnptr_shmem_init fn_shmem_init;
    fnptr_shmem_my_pe fn_shmem_my_pe;
    fnptr_shmem_n_pes fn_shmem_n_pes;
    fnptr_shmem_fcollect64 fn_shmem_fcollect64;
    fnptr_shmem_malloc fn_shmem_malloc;
    fnptr_shmem_free fn_shmem_free;
    fnptr_shmem_finalize fn_shmem_finalize;
};

extern void *nvshmemi_shmem_handle;
extern struct nvshmemi_shmem_fn_table shmem_fn_table;

#endif

using namespace std;

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#define CU_CHECK(stmt)                                                                     \
    do {                                                                                   \
        CUresult result = (stmt);                                                          \
        char str[1024];                                                                    \
        if (CUDA_SUCCESS != result) {                                                      \
            CUresult ret = cuGetErrorString(result, (const char **)&str);                  \
            fprintf(stderr, "[%s:%d] cuda failed with (%d) %s \n", __FILE__, __LINE__,     \
                    (int)result, (ret != CUDA_SUCCESS) ? "cuGetErrorString failed" : str); \
            exit(-1);                                                                      \
        }                                                                                  \
        assert(CUDA_SUCCESS == result);                                                    \
    } while (0)

#define ERROR_EXIT(...)                                                  \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        exit(-1);                                                        \
    } while (0)

#define ERROR_PRINT(...)                                                 \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
    } while (0)

#undef WARN_PRINT
#define WARN_PRINT(...)                                                  \
    do {                                                                 \
        fprintf(stdout, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stdout, __VA_ARGS__);                                    \
    } while (0)

#ifdef _NVSHMEM_DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__);
#else
#define DEBUG_PRINT(...) \
    do {                 \
    } while (0)
#endif

#define MS_TO_S 1000
#define B_TO_GB (1000 * 1000 * 1000)

enum NVSHMEM_DATATYPE_T {
    NVSHMEM_INT = 0,
    NVSHMEM_LONG,
    NVSHMEM_LONGLONG,
    NVSHMEM_ULONGLONG,
    NVSHMEM_SIZE,
    NVSHMEM_PTRDIFF,
    NVSHMEM_FLOAT,
    NVSHMEM_DOUBLE,
    NVSHMEM_UINT,
    NVSHMEM_INT32,
    NVSHMEM_UINT32,
    NVSHMEM_INT64,
    NVSHMEM_UINT64,
    NVSHMEM_FP16,
    NVSHMEM_BF16
};

enum NVSHMEM_REDUCE_OP_T {
    NVSHMEM_MIN = 0,
    NVSHMEM_MAX,
    NVSHMEM_SUM,
    NVSHMEM_PROD,
    NVSHMEM_AND,
    NVSHMEM_OR,
    NVSHMEM_XOR
};

enum NVSHMEM_THREADGROUP_SCOPE_T {
    NVSHMEM_THREAD = 0,
    NVSHMEM_WARP,
    NVSHMEM_BLOCK,
    NVSHMEM_ALL_SCOPES
};

enum AMO_T {
    AMO_INC = 0,
    AMO_FETCH_INC,
    AMO_SET,
    AMO_ADD,
    AMO_FETCH_ADD,
    AMO_AND,
    AMO_FETCH_AND,
    AMO_OR,
    AMO_FETCH_OR,
    AMO_XOR,
    AMO_FETCH_XOR,
    AMO_SWAP,
    AMO_COMPARE_SWAP,
    AMO_ACK
};

enum PUTGET_ISSUE_T { ON_STREAM = 0, HOST };

enum DIR_T { WRITE = 0, READ };

extern size_t min_size;
extern size_t max_size;
extern size_t num_blocks;
extern size_t threads_per_block;
extern size_t iters;
extern size_t warmup_iters;
extern size_t step_factor;
extern size_t max_size_log;
extern size_t stride;
extern size_t mem_handle_type;
extern bool bidirectional;

extern void *nvml_handle;
extern struct nvml_function_table nvml_ftable;
extern const char *env_value;

struct datatype_t {
    NVSHMEM_DATATYPE_T type;
    size_t size;
    string name;
};
extern datatype_t datatype;
struct reduce_op_t {
    NVSHMEM_REDUCE_OP_T type;
    string name;
};
extern reduce_op_t reduce_op;
struct threadgroup_scope_t {
    NVSHMEM_THREADGROUP_SCOPE_T type;
    string name;
};
extern threadgroup_scope_t threadgroup_scope;

struct amo_t {
    AMO_T type;
    string name;
};
extern amo_t test_amo;

struct putget_issue_t {
    PUTGET_ISSUE_T type;
    string name;
};
extern putget_issue_t putget_issue;

struct dir_t {
    DIR_T type;
    string name;
};
extern dir_t dir;

extern bool use_graph;
extern bool use_mmap;
extern bool use_egm;

extern bool use_cubin;

void init_cumodule(const char *str);
void init_wrapper(int *c, char ***v);
void finalize_wrapper();
void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table);
void free_tables(void **tables, int num_tables);
uint64_t calculate_collective_size(const char *coll_name, uint64_t num_elems, uint64_t type_size,
                                   int npes);
void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries);
void print_table_v1(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double *value, int num_entries);
void print_table_v2(const char *job_name, const char *subjob_name, const char *var_name,
                    const char *output_var, const char *units, const char plus_minus,
                    uint64_t *size, double **value, int num_entries, size_t num_iters);
void read_args(int argc, char **argv);
void *allocate_mmap_buffer(size_t size, int mem_handle_type, bool use_egm = false,
                           bool reset_zero = false);
void free_mmap_buffer(void *ptr);
size_t pad_up(size_t size);

/* Copied from CUDA 12.4 NVML header. */
#if ((NVML_API_VERSION < 12) || (CUDA_VERSION < 12040))

#ifndef NVML_GPU_FABRIC_STATE_COMPLETED
#define NVML_GPU_FABRIC_STATE_COMPLETED 3
#endif

#ifndef nvmlGpuFabricInfo_v2
#define nvmlGpuFabricInfo_v2 (unsigned int)(sizeof(nvmlGpuFabricInfo_v2_t) | (2 << 24U))
#endif

#ifndef NVML_GPU_FABRIC_UUID_LEN
#define NVML_GPU_FABRIC_UUID_LEN 16
#endif

typedef unsigned char nvmlGpuFabricState_t;
typedef struct {
    unsigned int version;  //!< Structure version identifier (set to \ref nvmlGpuFabricInfo_v2)
    unsigned char
        clusterUuid[NVML_GPU_FABRIC_UUID_LEN];  //!< Uuid of the cluster to which this GPU belongs
    nvmlReturn_t
        status;  //!< Error status, if any. Must be checked only if state returns "complete".
    unsigned int cliqueId;       //!< ID of the fabric clique to which this GPU belongs
    nvmlGpuFabricState_t state;  //!< Current state of GPU registration process
    unsigned int healthMask;     //!< GPU Fabric health Status Mask
} nvmlGpuFabricInfo_v2_t;

typedef nvmlGpuFabricInfo_v2_t nvmlGpuFabricInfoV_t;
#endif
/* end NVML Header defs. */

struct nvml_function_table {
    nvmlReturn_t (*nvmlInit)(void);
    nvmlReturn_t (*nvmlShutdown)(void);
    nvmlReturn_t (*nvmlDeviceGetHandleByPciBusId)(const char *pciBusId, nvmlDevice_t *device);
    nvmlReturn_t (*nvmlDeviceGetP2PStatus)(nvmlDevice_t device1, nvmlDevice_t device2,
                                           nvmlGpuP2PCapsIndex_enum caps,
                                           nvmlGpuP2PStatus_t *p2pStatus);
    nvmlReturn_t (*nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info);
};

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle);
bool is_mnnvl_supported(int dev_id);
#endif
