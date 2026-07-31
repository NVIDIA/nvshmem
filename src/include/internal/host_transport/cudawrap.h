/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVSHMEM_CUDAWRAP_H
#define NVSHMEM_CUDAWRAP_H

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDART_VERSION < 12040
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED 128
#define CU_MEM_HANDLE_TYPE_FABRIC (CUmemAllocationHandleType)0x8
#define CU_CTX_SYNC_MEMOPS 0x80
#endif

#if CUDART_VERSION < 12020
#define CU_MEM_LOCATION_TYPE_HOST_NUMA (CUmemLocationType)0x3
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID (CUdevice_attribute)134
#endif

#if CUDART_VERSION < 12010
typedef CUresult(CUDAAPI *PFN_cuCtxSetFlags_v12010)(int flags);
#endif

#if CUDART_VERSION < 12080
#define CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE 0x1
#endif

#if CUDART_VERSION < 12010
#define CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED 132
typedef enum CUmulticastGranularity_flags_enum {
    CU_MULTICAST_GRANULARITY_MINIMUM = 0x0,
    CU_MULTICAST_GRANULARITY_RECOMMENDED = 0x1
} CUmulticastGranularity_flags;
typedef struct CUmulticastObjectProp_st {
    unsigned int numDevices;
    size_t size;
    unsigned long long handleTypes;
    unsigned long long flags;
} CUmulticastObjectProp_v1;
typedef CUmulticastObjectProp_v1 CUmulticastObjectProp;
typedef CUresult(CUDAAPI *PFN_cuMulticastCreate_v12010)(CUmemGenericAllocationHandle *mcHandle,
                                                        const CUmulticastObjectProp *prop);
typedef CUresult(CUDAAPI *PFN_cuMulticastBindMem_v12010)(CUmemGenericAllocationHandle mcHandle,
                                                         size_t mcOffset,
                                                         CUmemGenericAllocationHandle memHandle,
                                                         size_t memOffset, size_t size,
                                                         unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuMulticastAddDevice_v12010)(CUmemGenericAllocationHandle mcHandle,
                                                           CUdevice dev);
typedef CUresult(CUDAAPI *PFN_cuMulticastUnbind_v12010)(CUmemGenericAllocationHandle mcHandle,
                                                        CUdevice dev, size_t mcOffset, size_t size);
typedef CUresult(CUDAAPI *PFN_cuMulticastGetGranularity_v12010)(
    size_t *granularity, const CUmulticastObjectProp *prop, CUmulticastGranularity_flags option);
#endif

#if CUDART_VERSION < 13030

/** An ID that represents a logical endpoint */
typedef uint32_t CUlogicalEndpointId;

/** Logical endpoint type */
typedef enum CUlogicalEndpointType_enum {
    CU_LOGICAL_ENDPOINT_TYPE_INVALID,
    CU_LOGICAL_ENDPOINT_TYPE_UNICAST,
    CU_LOGICAL_ENDPOINT_TYPE_MULTICAST
} CUlogicalEndpointType;

/** IPC handle types that can be requested/queried for a given logical endpoint */
typedef enum CUlogicalEndpointIpcHandleType_enum {
    CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE,
    CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC
} CUlogicalEndpointIpcHandleType;

/** Fabric handle for a logical endpoint */
typedef struct CUlogicalEndpointFabricHandle_st {
    unsigned char data[CU_IPC_HANDLE_SIZE];
} CUlogicalEndpointFabricHandle;

/** Properties of a logical endpoint */
typedef struct CUlogicalEndpointProp_struct {
    CUlogicalEndpointType type;
    union {
        struct {
            CUdevice device;
        } unicast;
        struct {
            unsigned int numDevices;
        } multicast;
    };
    unsigned long long size;
    CUlogicalEndpointIpcHandleType ipcHandleTypes;
    unsigned int flags;  // Must be zero
} CUlogicalEndpointProp;

#define CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED (CUdevice_attribute)153
#define CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_MULTICAST_SUPPORTED (CUdevice_attribute)154
#define CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_COUNTED_OPS_SUPPORTED (CUdevice_attribute)155
#define CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS 0x1u

typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointIdReserve_v13030)(CUlogicalEndpointId *leId,
                                                                 cuuint32_t count);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointIdRelease_v13030)(CUlogicalEndpointId leId,
                                                                 cuuint32_t count);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointCreate_v13030)(CUlogicalEndpointId leId,
                                                              const CUlogicalEndpointProp *prop);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointAddDevice_v13030)(CUlogicalEndpointId leId,
                                                                 CUdevice dev);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointDestroy_v13030)(CUlogicalEndpointId leId);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointBindAddr_v13030)(CUlogicalEndpointId leId,
                                                                CUdevice dev, cuuint64_t offset,
                                                                void *ptr, cuuint64_t size,
                                                                unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointBindMem_v13030)(
    CUlogicalEndpointId leId, CUdevice dev, cuuint64_t offset,
    CUmemGenericAllocationHandle memHandle, cuuint64_t memOffset, cuuint64_t size,
    unsigned long long flags);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointUnbind_v13030)(CUlogicalEndpointId leId,
                                                              CUdevice dev, cuuint64_t offset,
                                                              cuuint64_t size);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointExport_v13030)(
    void *handle, const CUlogicalEndpointId leId, CUlogicalEndpointIpcHandleType handleType);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointImport_v13030)(
    CUlogicalEndpointId leId, const void *handle, CUlogicalEndpointIpcHandleType handleType);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointQuery_v13030)(CUlogicalEndpointId leId,
                                                             cuuint32_t count, int *queryStatus);
typedef CUresult(CUDAAPI *PFN_cuLogicalEndpointGetLimits_v13030)(cuuint64_t *bindAlignment,
                                                                 cuuint64_t *maxSize,
                                                                 const CUlogicalEndpointProp *prop);

#endif

#include <cudaTypedefs.h>

#define DEFINE_SYM(symbol, version) PFN_##symbol##_v##version pfn_##symbol;
struct nvshmemi_cuda_fn_table {
    DEFINE_SYM(cuCtxGetDevice, 2000)
    DEFINE_SYM(cuCtxSynchronize, 2000)
    DEFINE_SYM(cuDeviceGet, 2000)
    DEFINE_SYM(cuDeviceGetAttribute, 2000)
    DEFINE_SYM(cuPointerGetAttribute, 4000)
    DEFINE_SYM(cuPointerSetAttribute, 6000)
    DEFINE_SYM(cuModuleGetGlobal, 3020)
    DEFINE_SYM(cuGetErrorString, 6000)
    DEFINE_SYM(cuGetErrorName, 6000)
    DEFINE_SYM(cuCtxSetCurrent, 4000)
    DEFINE_SYM(cuDevicePrimaryCtxRetain, 7000)
    DEFINE_SYM(cuCtxGetCurrent, 4000)
    DEFINE_SYM(cuCtxGetFlags, 7000)
    DEFINE_SYM(cuCtxSetFlags, 12010)
    DEFINE_SYM(cuFlushGPUDirectRDMAWrites, 11030)     // DMA-BUF support
    DEFINE_SYM(cuMemGetHandleForAddressRange, 11070)  // DMA-BUF support
    DEFINE_SYM(cuMemCreate, 10020)
    DEFINE_SYM(cuMemAddressReserve, 10020)
    DEFINE_SYM(cuMemAddressFree, 10020)
    DEFINE_SYM(cuMemMap, 10020)
    DEFINE_SYM(cuMemGetAllocationGranularity, 10020)
    DEFINE_SYM(cuMemGetAllocationPropertiesFromHandle, 10020)
    DEFINE_SYM(cuMemImportFromShareableHandle, 10020)
    DEFINE_SYM(cuMemExportToShareableHandle, 10020)
    DEFINE_SYM(cuMemRelease, 10020)
    DEFINE_SYM(cuMemSetAccess, 10020)
    DEFINE_SYM(cuMemGetAccess, 10020)
    DEFINE_SYM(cuMemUnmap, 10020)
    DEFINE_SYM(cuMulticastCreate, 12010)
    DEFINE_SYM(cuMulticastAddDevice, 12010)
    DEFINE_SYM(cuMulticastBindMem, 12010)
    DEFINE_SYM(cuMulticastUnbind, 12010)
    DEFINE_SYM(cuMulticastGetGranularity, 12010)
    DEFINE_SYM(cuStreamWriteValue64, 11070)
    DEFINE_SYM(cuStreamWaitValue64, 11070)
    DEFINE_SYM(cuMemRetainAllocationHandle, 11000)
    DEFINE_SYM(cuLibraryGetGlobal, 12000)
    DEFINE_SYM(cuGetExportTable, 3000)
    /* CUDA Driver functions loaded with dlsym() */
    DEFINE_SYM(cuInit, 2000)
    DEFINE_SYM(cuGetProcAddress, 11030)
    DEFINE_SYM(cuLogicalEndpointIdReserve, 13030)
    DEFINE_SYM(cuLogicalEndpointIdRelease, 13030)
    DEFINE_SYM(cuLogicalEndpointCreate, 13030)
    DEFINE_SYM(cuLogicalEndpointAddDevice, 13030)
    DEFINE_SYM(cuLogicalEndpointDestroy, 13030)
    DEFINE_SYM(cuLogicalEndpointBindAddr, 13030)
    DEFINE_SYM(cuLogicalEndpointBindMem, 13030)
    DEFINE_SYM(cuLogicalEndpointUnbind, 13030)
    DEFINE_SYM(cuLogicalEndpointExport, 13030)
    DEFINE_SYM(cuLogicalEndpointImport, 13030)
    DEFINE_SYM(cuLogicalEndpointQuery, 13030)
    DEFINE_SYM(cuLogicalEndpointGetLimits, 13030)
};
#undef DEFINE_SYM

#define CUPFN(table, symbol) table->pfn_##symbol

static inline void nvshmemi_cu_get_error_info(struct nvshmemi_cuda_fn_table *table, int err_code,
                                              const char **err_name, const char **err_desc) {
    CUresult err = static_cast<CUresult>(err_code);
    if (err_name) *err_name = "UNKNOWN";
    if (err_desc) *err_desc = "Unknown";
    if (table) {
        if (table->pfn_cuGetErrorName && err_name) {
            (void)table->pfn_cuGetErrorName(err, err_name);
        }
        if (table->pfn_cuGetErrorString && err_desc) {
            (void)table->pfn_cuGetErrorString(err, err_desc);
        }
    }
}

#define NVSHMEMI_CU_ERROR_LOG(table, status)                                        \
    do {                                                                            \
        const char *cu_err_name = "UNKNOWN";                                        \
        const char *cu_err_desc = "Unknown";                                        \
        nvshmemi_cu_get_error_info(table, status, &cu_err_name, &cu_err_desc);      \
        fprintf(stderr, "Status = %s. Description = %s", cu_err_name, cu_err_desc); \
    } while (false)

#define NVSHMEMI_CU_NE_ERROR_JMP(table, status, expected, err, label, ...) \
    do {                                                                   \
        if ((status) != (expected)) {                                      \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                \
            fprintf(stderr, __VA_ARGS__);                                  \
            fprintf(stderr, " ");                                          \
            NVSHMEMI_CU_ERROR_LOG(table, status);                          \
            fprintf(stderr, "\n");                                         \
            status = err;                                                  \
            goto label;                                                    \
        }                                                                  \
    } while (false)

#define NVSHMEMI_CU_NZ_ERROR_JMP(table, status, err, label, ...) \
    do {                                                         \
        if ((status) != CUDA_SUCCESS) {                          \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);      \
            fprintf(stderr, __VA_ARGS__);                        \
            fprintf(stderr, " ");                                \
            NVSHMEMI_CU_ERROR_LOG(table, status);                \
            fprintf(stderr, "\n");                               \
            status = err;                                        \
            goto label;                                          \
        }                                                        \
    } while (false)

#define NVSHMEMI_CU_NZ_EXIT(table, status, ...)             \
    do {                                                    \
        if ((status) != CUDA_SUCCESS) {                     \
            fprintf(stderr, "%s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                   \
            fprintf(stderr, " ");                           \
            NVSHMEMI_CU_ERROR_LOG(table, status);           \
            fprintf(stderr, "\n");                          \
            exit(-1);                                       \
        }                                                   \
    } while (false)

// Check CUDA PFN driver calls
#define CUCHECKNORETURN(table, cmd)                                                             \
    do {                                                                                        \
        CUresult err = table->pfn_##cmd;                                                        \
        if (err != CUDA_SUCCESS) {                                                              \
            const char *err_name = "UNKNOWN";                                                   \
            const char *err_desc = "Unknown";                                                   \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc);                       \
            fprintf(stderr, "Cuda failure. Status = %s. Description = %s", err_name, err_desc); \
        }                                                                                       \
        assert(err == CUDA_SUCCESS);                                                            \
    } while (false)

#define CUASSERTAPIAVAILABLE(table, cmd) \
    do {                                 \
        if (table->pfn_##cmd == NULL) {  \
            assert(false);               \
        }                                \
    } while (false)

// Check CUDA PFN driver calls
#define CUCHECK(table, cmd)                                                                \
    do {                                                                                   \
        CUresult err = table->pfn_##cmd;                                                   \
        if (err != CUDA_SUCCESS) {                                                         \
            const char *err_name = "UNKNOWN";                                              \
            const char *err_desc = "Unknown";                                              \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc);                  \
            fprintf(stderr, "%s:%d Cuda failure. Status = %s. Description = %s", __FILE__, \
                    __LINE__, err_name, err_desc);                                         \
            return NVSHMEMX_ERROR_INTERNAL;                                                \
        }                                                                                  \
    } while (false)

#define CUCHECKGOTO(table, cmd, res, label)                                                \
    do {                                                                                   \
        CUresult err = table->pfn_##cmd;                                                   \
        if (err != CUDA_SUCCESS) {                                                         \
            const char *err_name = "UNKNOWN";                                              \
            const char *err_desc = "Unknown";                                              \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc);                  \
            fprintf(stderr, "%s:%d Cuda failure. Status = %s. Description = %s", __FILE__, \
                    __LINE__, err_name, err_desc);                                         \
            res = NVSHMEMX_ERROR_INTERNAL;                                                 \
            goto label;                                                                    \
        }                                                                                  \
    } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(table, cmd)                                                          \
    do {                                                                                   \
        CUresult err = table->pfn_##cmd;                                                   \
        if (err != CUDA_SUCCESS) {                                                         \
            const char *err_name = "UNKNOWN";                                              \
            const char *err_desc = "Unknown";                                              \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc);                  \
            fprintf(stderr, "%s:%d Cuda failure. Status = %s. Description = %s", __FILE__, \
                    __LINE__, err_name, err_desc);                                         \
        }                                                                                  \
    } while (false)

// Report failure but clear error and continue
#define CUCHECKIGNORE_NO_PRINT(table, cmd)                                \
    do {                                                                  \
        CUresult err = table->pfn_##cmd;                                  \
        if (err != CUDA_SUCCESS) {                                        \
            const char *err_name = "UNKNOWN";                             \
            const char *err_desc = "Unknown";                             \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc); \
        }                                                                 \
    } while (false)

#define CUCHECKTHREAD(table, cmd, args)                                                      \
    do {                                                                                     \
        CUresult err = table->pfn_##cmd;                                                     \
        if (err != CUDA_SUCCESS) {                                                           \
            const char *err_name = "UNKNOWN";                                                \
            const char *err_desc = "Unknown";                                                \
            nvshmemi_cu_get_error_info(table, err, &err_name, &err_desc);                    \
            fprintf(stderr, "%s:%d -> %s (%s) [Async thread]", __FILE__, __LINE__, err_name, \
                    err_desc);                                                               \
            args->ret = NVSHMEMX_ERROR_INTERNAL;                                             \
            return args;                                                                     \
        }                                                                                    \
    } while (0)

#endif
