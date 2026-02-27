/*
 * Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef UTILS
#define UTILS

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <libgen.h>
#include <stdio.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <stddef.h>
#include <sstream>
#include <vector>
#include <inttypes.h>
#include <assert.h>
#include <dlfcn.h>
#include "test-simple-pmi/test_pmi_internal.h"
#ifdef NVSHMEMTEST_MPI_SUPPORT
#include "mpi.h"
#endif
#ifdef NVSHMEMTEST_SHMEM_SUPPORT
#include "shmem.h"
#include "shmemx.h"
#endif
#include "nvshmem.h"
#include "nvshmemx.h"
#include "device_host/nvshmem_common.cuh"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#define NVSHMEMI_TEST_STRINGIFY(x) #x

#define NVSHPRI_half "%f"
#define NVSHPRI_bfloat16 "%f"
#define NVSHPRI_float "%0.2f"
#define NVSHPRI_double "%0.2f"
#define NVSHPRI_char "%hhd"
#define NVSHPRI_schar "%hhd"
#define NVSHPRI_short "%hd"
#define NVSHPRI_int "%d"
#define NVSHPRI_long "%ld"
#define NVSHPRI_longlong "%lld"
#define NVSHPRI_uchar "%hhu"
#define NVSHPRI_ushort "%hu"
#define NVSHPRI_uint "%u"
#define NVSHPRI_ulong "%lu"
#define NVSHPRI_ulonglong "%llu"
#define NVSHPRI_int8 "%" PRIi8
#define NVSHPRI_int16 "%" PRIi16
#define NVSHPRI_int32 "%" PRIi32
#define NVSHPRI_int64 "%" PRIi64
#define NVSHPRI_uint8 "%" PRIu8
#define NVSHPRI_uint16 "%" PRIu16
#define NVSHPRI_uint32 "%" PRIu32
#define NVSHPRI_uint64 "%" PRIu64
#define NVSHPRI_size "%zu"
#define NVSHPRI_ptrdiff "%zu"
#define NVSHPRI_bool "%s"
#define NVSHPRI_string "\"%s\""
#define NVSHPRI_cutlass_half_t "%f"
#define NVSHPRI_cutlass_bfloat16_t "%f"

#define MEM_TYPE_AUTO 0
#define MEM_TYPE_POSIX_FD 1
#define MEM_TYPE_FABRIC 2

#define MEM_GRANULARITY 536870912  // 512MB

#ifdef __CUDACC__
#if CUDA_VERSION < 12020
static __device__ bool printed_error = false;
#define CU_MEM_LOCATION_TYPE_HOST_NUMA (CUmemLocationType)0x3
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID (CUdevice_attribute)134
#endif

#if CUDART_VERSION < 12040
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED (CUdevice_attribute)128
#define CU_MEM_HANDLE_TYPE_FABRIC (CUmemAllocationHandleType)0x8
#define CU_CTX_SYNC_MEMOPS 0x80
#endif

#if CUTLASS_ENABLED == 1
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#endif

enum NVSHMEM_CUBIN_LIBRARY_T {
    NVSHMEM_CUBIN_NONE = 0,
    NVSHMEM_CUBIN_BC = 1,
    NVSHMEM_CUBIN_LTOIR = 2
};

enum NVSHMEM_THREADGROUP_SCOPE_T {
    NVSHMEM_THREAD = 0,
    NVSHMEM_WARP,
    NVSHMEM_WARPGROUP,
    NVSHMEM_BLOCK,
    NVSHMEM_ALL_SCOPES
};

struct threadgroup_scope_t {
    NVSHMEM_THREADGROUP_SCOPE_T type;
    std::string name;
};
extern threadgroup_scope_t threadgroup_scope;

template <typename T>
static __device__ T assign(const unsigned long val) {
    return val;
}
template <>
__device__ half assign<half>(const unsigned long val) {
    return __ull2half_rn(val);
}

#if CUTLASS_ENABLED == 1
template <>
__device__ cutlass::half_t assign<cutlass::half_t>(const unsigned long val) {
    return cutlass::half_t(__ull2half_rn(val));
}

template <>
__device__ cutlass::bfloat16_t assign<cutlass::bfloat16_t>(const unsigned long val) {
    return cutlass::bfloat16_t(__ull2bfloat16_rn(val));
}
#endif

template <>
__device__ __nv_bfloat16 assign<__nv_bfloat16>(const unsigned long val) {
#if CUDA_VERSION < 12020
    if (!printed_error) {
        printf("Tests built before cuda version 12.2 cannot support __nv_bfloat16.\n");
        printf("Errors expected.\n");
        printed_error = true;
    }

    return __double2bfloat16((double)val);
#else
    return __ull2bfloat16_rn(val);
#endif
}
#endif

#define NVSHMEMTEST_ERRSTR_FORMAT_1(TYPENAME, SC)                          \
    "error: found = " NVSHPRI_##TYPENAME " expected = " NVSHPRI_##TYPENAME \
        ", index = %llu, nelems = %llu type = " #TYPENAME ", scope = " #SC ", team = %d\n"

#define NVSHMEMTEST_ERRSTR_FORMAT_2(TYPENAME, OP, SC)                                    \
    "error: found = " NVSHPRI_##TYPENAME " expected = " NVSHPRI_##TYPENAME               \
        ", index = %llu, nelems = %llu type = " #TYPENAME ", op = " #OP ", scope = " #SC \
        ", team = %d\n"

template <typename TYPE>
static __device__ void print_err(TYPE found, TYPE expected, size_t idx, size_t nelems,
                                 nvshmem_team_t team, const char *print_formatter) {
    printf(print_formatter, found, expected, idx, nelems, team);
}
template <>
void __device__ print_err<half>(half found, half expected, size_t idx, size_t nelems,
                                nvshmem_team_t team, const char *print_formatter) {
    printf(print_formatter, __half2float(found), __half2float(expected), idx, nelems, team);
}
template <>
void __device__ print_err<__nv_bfloat16>(__nv_bfloat16 found, __nv_bfloat16 expected, size_t idx,
                                         size_t nelems, nvshmem_team_t team,
                                         const char *print_formatter) {
    printf(print_formatter, __bfloat162float(found), __bfloat162float(expected), idx, nelems, team);
}

#if CUTLASS_ENABLED == 1
template <>
void __device__ print_err<cutlass::half_t>(cutlass::half_t found, cutlass::half_t expected,
                                           size_t idx, size_t nelems, nvshmem_team_t team,
                                           const char *print_formatter) {
    printf(print_formatter, float(found), float(expected), idx, nelems, team);
}

template <>
void __device__ print_err<cutlass::bfloat16_t>(cutlass::bfloat16_t found,
                                               cutlass::bfloat16_t expected, size_t idx,
                                               size_t nelems, nvshmem_team_t team,
                                               const char *print_formatter) {
    printf(print_formatter, float(found), float(expected), idx, nelems, team);
}
#endif

void init_test_case_kernel(CUfunction *kernel, const char *kernel_name);

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

#undef CU_CHECK
#define CU_CHECK(stmt)                                                                  \
    do {                                                                                \
        CUresult result = (stmt);                                                       \
        const char *str;                                                                \
        if (CUDA_SUCCESS != result) {                                                   \
            CUresult ret = cuGetErrorString(result, &str);                              \
            if (ret == CUDA_ERROR_INVALID_VALUE) str = "Unknown error";                 \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, str); \
            exit(-1);                                                                   \
        }                                                                               \
        assert(CUDA_SUCCESS == result);                                                 \
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

#define TILE_CHECK_ERRS()                                                                      \
    do {                                                                                       \
        cudaMemcpyFromSymbol(&tile_errs, tile_errs_d, sizeof(int), 0);                         \
        if (tile_errs) {                                                                       \
            printf("NVSHMEM Error :%d , %s\n", tile_errs, get_err_string(tile_errs).c_str());  \
            return tile_errs;                                                                  \
        }                                                                                      \
    } while (0)

#define TOSTRING(_var) ((std::to_string(_var)).c_str())
#define MAX_ELEMS (1 * 1024 * 1024)


extern int mype, mype_node;
extern int npes, npes_node;
extern int use_cubin;
extern size_t _min_size;
extern size_t _max_size;
extern size_t _min_iters;
extern size_t _max_iters;
extern size_t _step_factor;
extern size_t _repeat;
extern bool use_egm;
extern bool use_mmap;
extern size_t _mem_handle_type;
extern bool _only_p2p;

extern void *nvml_handle;
extern struct nvml_function_table nvml_ftable;
extern const char *env_value;

extern __device__ int tile_errs_d;
extern int tile_errs;

void init_cumodule(const char *str);
void init_wrapper(int *c, char ***v);
void finalize_wrapper();
void select_device();
void read_args(int argc, char **argv);
void *allocate_mmap_buffer(size_t size, int mem_handle_type, bool use_egm = false,
                           bool reset_zero = false);
void free_mmap_buffer(void *ptr);
size_t pad_up(size_t size);
std::string get_err_string(int errs);

#define CUMODULE_LOAD(CUMODULE, CUMODULE_PATH, ERROR) \
    CU_CHECK(cuModuleLoad(&CUMODULE, CUMODULE_PATH)); \
    ERROR = nvshmemx_cumodule_init(CUMODULE);

#ifdef __CUDACC__
__device__ inline int nvshmtest_thread_id_in_warp() {
    int myIdx;
    asm volatile("mov.u32  %0, %%laneid;" : "=r"(myIdx));
    return myIdx;
}

__device__ inline int nvshmtest_warp_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : warpSize;
}

__device__ inline void nvshmtest_warp_sync() { __syncwarp(); }

__device__ inline int nvshmtest_thread_id_in_block() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ inline int nvshmtest_warpgroup_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < 4 * warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : 4 * warpSize;
}

// TODO: add warpgroup sync
__device__ inline void nvshmtest_warpgroup_sync() { __syncwarp(); }

__device__ inline int nvshmtest_thread_id_in_warpgroup() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) %
           nvshmtest_warpgroup_size();
}

__device__ inline int nvshmtest_block_size() { return (blockDim.x * blockDim.y * blockDim.z); }

__device__ inline void nvshmtest_block_sync() { __syncthreads(); }

__device__ inline int nvshmtest_thread_id_in_thread() { return 0; }

__device__ inline int nvshmtest_thread_size() { return 1; }

__device__ inline void nvshmtest_thread_sync() {}

#endif /* __CUDA_ARCH__ */

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
    nvmlReturn_t (*nvmlDeviceGetFieldValues)(nvmlDevice_t device, unsigned int count, nvmlFieldValue_t *values);
};

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle);
bool is_mnnvl_supported(int dev_id);
#endif
