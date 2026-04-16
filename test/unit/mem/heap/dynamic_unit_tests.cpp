/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

// Declarations of functions and structs should be included, definitions we create on our own

#include <string.h>
#include <cstdint>
#include "sockets.h"
#include "util.h"
#include "debug.h"
#include "nvmlwrap.h"
#include "nvshmemi_types.h"
#include "nvml.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "host/nvshmem_coll_api.h"
#include "internal/host/nvshmemi_nvls_rsc.hpp"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/transport.h"
#include "internal/host/nvshmemi_mem_transport.hpp"
#include "nvshmemi_symmetric_heap.hpp"

using namespace std;
using namespace testing;

namespace nvls {
class Mock_nvshmemi_nvls_rsc : public nvshmemi_nvls_rsc {
   public:
    MOCK_METHOD1(subscribe_group, int(CUmemGenericAllocationHandle *mc_handle));
    MOCK_METHOD2(export_group, int(uint64_t mem_size, char *shareable_handle));
    MOCK_METHOD3(import_group, int(char *shareable_handle, CUmemGenericAllocationHandle *mc_handle,
                                   uint64_t mem_size));
    MOCK_METHOD3(unbind_group_mem,
                 int(CUmemGenericAllocationHandle *mc_handle, off_t mc_offset, size_t mem_size));
    MOCK_METHOD5(bind_group_mem,
                 int(CUmemGenericAllocationHandle *mc_handle, nvshmem_mem_handle_t *mem_handle,
                     size_t mem_size, off_t mem_offset, off_t mc_offset));
    MOCK_METHOD4(map_group_mem, int(CUmemGenericAllocationHandle *mc_handle, size_t mem_size,
                                    off_t mem_offset, off_t mc_offset));
    MOCK_METHOD2(unmap_group_mem, int(off_t mc_offset, size_t mem_size));
};
}  // namespace nvls

using namespace nvls;

nvshmemi_symmetric_heap *nv_symheap;
nvshmemi_team_t **nvshmemi_team_pool;
bootstrap_handle_t nvshmemi_boot_handle;
struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms = NULL;
int nvshmem_debug_level;
int nvshmem_nvtx_options;
long nvshmemi_max_teams;

uintptr_t addr_val = 0;
void *addr = &addr_val;
nvshmemi_state_t *nvshmemi_state;
nvshmemi_device_host_state_t nvshmemi_device_state;
nvshmemi_options_s nvshmemi_options;

nvshmemi_state_t state;

void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...) {
    if (nvshmem_debug_level <= NVSHMEM_LOG_NONE) {
        return;
    }
}

class Mock_general_global_functions {
   public:
    MOCK_METHOD2(nvshmemi_team_translate_pe_to_team_world_wrap,
                 int(nvshmemi_team_t *src_team, int src_pe));
    MOCK_METHOD0(nvshmemi_update_device_state, int());
    MOCK_METHOD0(nvshmemi_check_state_and_init, int());
    MOCK_METHOD0(nvshmemi_barrier_all, void());
    MOCK_METHOD3(ipcOpenSocket, int(ipcHandle *&handle, pid_t a, pid_t b));
    MOCK_METHOD4(ipcSendFd, int(ipcHandle *handle, const int fd, pid_t process, pid_t a));
    MOCK_METHOD2(ipcRecvFd, int(ipcHandle *handle, int *fd));
    MOCK_METHOD1(ipcCloseSocket, int(ipcHandle *handle));
    MOCK_METHOD3(shared_memory_create,
                 int(const char *name, size_t sz, nvshmemi_shared_memory_info *info));
    MOCK_METHOD0(nvshmemi_get_teams_mem_requirement, size_t());
    MOCK_METHOD2(cudaMalloc, cudaError_t(void **devPtr, size_t size));
    MOCK_METHOD3(cudaHostRegister, cudaError_t(void *ptr, size_t size, unsigned int flags));
    MOCK_METHOD3(cudaHostGetDevicePointer,
                 cudaError_t(void **pDevice, void *pHost, unsigned int flags));
    MOCK_METHOD1(cudaFree, cudaError_t(void *devPtr));
    MOCK_METHOD2(cudaIpcGetMemHandle, cudaError_t(cudaIpcMemHandle_t *handle, void *devPtr));
    MOCK_METHOD3(cudaIpcOpenMemHandle,
                 cudaError_t(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags));
    MOCK_METHOD4(cudaMemcpy,
                 cudaError_t(void *dst, const void *src, size_t count, cudaMemcpyKind kind));
    MOCK_METHOD0(cudaDeviceSynchronize, cudaError_t(void));
    MOCK_METHOD1(cudaIpcCloseMemHandle, cudaError_t(void *devPtr));
    MOCK_METHOD2(shared_memory_close, void(char *shm_name, nvshmemi_shared_memory_info *info));
    MOCK_METHOD1(cudaHostUnregister, cudaError_t(void *ptr));
    MOCK_METHOD1(cudaGetErrorString, const char *(cudaError_t error));
    MOCK_METHOD3(shared_memory_open,
                 int(const char *name, size_t sz, nvshmemi_shared_memory_info *info));
    MOCK_METHOD1(nvshmem_barrier, int(nvshmem_team_t team));
    MOCK_METHOD1(nvshmemi_team_support_nvls, bool(nvshmemi_team_t *team));
    MOCK_METHOD3(nvshmemi_team_translate_pe,
                 int(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team));
    MOCK_METHOD5(nvshmemx_char_put_nbi_on_stream,
                 void(char *dest, const char *source, size_t nelems, int pe, cudaStream_t stream));
    MOCK_METHOD2(nvshmemi_team_get_psync, long *(nvshmemi_team_t *team, nvshmemi_team_op_t op));
};

struct Mock_bootstrap_handle {
   public:
    MOCK_METHOD4(allgather, int(const void *sendbuf, void *recvbuf, int bytes,
                                struct bootstrap_handle *handle));
    MOCK_METHOD1(barrier, int(struct bootstrap_handle *handle));
};

struct Mock_nvshmemi_cuda_function_table {
   public:
    MOCK_METHOD3(pfn_cuMemGetAllocationGranularity,
                 CUresult(size_t *granularity, const CUmemAllocationProp *prop,
                          CUmemAllocationGranularity_flags option));
    MOCK_METHOD4(pfn_cuMemCreate,
                 CUresult(CUmemGenericAllocationHandle *handle, size_t size,
                          const CUmemAllocationProp *prop, unsigned long long flags));
    MOCK_METHOD5(pfn_cuMemMap,
                 CUresult(CUdeviceptr ptr, size_t size, size_t offset,
                          CUmemGenericAllocationHandle handle, unsigned long long flags));
    MOCK_METHOD2(pfn_cuMemUnmap, CUresult(CUdeviceptr ptr, size_t size));
    MOCK_METHOD4(pfn_cuMemSetAccess,
                 CUresult(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count));
    MOCK_METHOD5(pfn_cuMemAddressReserve, CUresult(CUdeviceptr *ptr, size_t size, size_t alignment,
                                                   CUdeviceptr addr, unsigned long long flags));
    MOCK_METHOD2(pfn_cuMemAddressFree, CUresult(CUdeviceptr ptr, size_t size));
    MOCK_METHOD1(pfn_cuMemRelease, CUresult(CUmemGenericAllocationHandle handle));
    MOCK_METHOD4(pfn_cuMemExportToShareableHandle,
                 CUresult(void *shareableHandle, CUmemGenericAllocationHandle handle,
                          CUmemAllocationHandleType handleType, unsigned long long flags));
    MOCK_METHOD3(pfn_cuPointerSetAttribute,
                 CUresult(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr));
    MOCK_METHOD1(pfn_cuCtxGetDevice, CUresult(CUdevice *device));
    MOCK_METHOD3(pfn_cuMemImportFromShareableHandle,
                 CUresult(CUmemGenericAllocationHandle *handle, void *osHandle,
                          CUmemAllocationHandleType shHandleType));
};

class Mock_mspace : public mspace {
   public:
    explicit Mock_mspace(void *base, size_t capacity) {}
    ~Mock_mspace() = default;
    MOCK_METHOD2(add_new_chunk, void(void *base, size_t capacity));
    MOCK_METHOD1(track_large_chunks, int(int enable));
    MOCK_METHOD1(allocate, void *(size_t bytes));
    MOCK_METHOD1(deallocate, void(void *mem));
    MOCK_METHOD2(allocate_zeroed, void *(size_t n_elements, size_t elem_size));
    MOCK_METHOD2(allocate_aligned, void *(size_t alignment, size_t bytes));
    MOCK_METHOD2(checkInuse, bool(void *ptr, size_t size));
};

Mock_general_global_functions *mggf;
nvshmemi_mem_p2p_transport *mnvmpt;
nvshmemi_mem_remote_transport *mnvmrt;
Mock_bootstrap_handle *mbh;
Mock_nvshmemi_cuda_function_table *mnvcft;
Mock_mspace *mmspace;
Mock_nvshmemi_nvls_rsc *mnvls;

int nvshmemi_team_translate_pe_to_team_world_wrap(nvshmemi_team_t *src_team, int src_pe) {
    return mggf->nvshmemi_team_translate_pe_to_team_world_wrap(src_team, src_pe);
}
int nvshmemi_update_device_state() { return mggf->nvshmemi_update_device_state(); }
int nvshmemi_check_state_and_init() { return mggf->nvshmemi_check_state_and_init(); }
void nvshmemi_barrier_all() { return mggf->nvshmemi_barrier_all(); }
int ipcOpenSocket(ipcHandle *&handle, pid_t a, pid_t b) {
    return mggf->ipcOpenSocket(handle, a, b);
}
int ipcSendFd(ipcHandle *handle, int fd, pid_t process, pid_t a) {
    return mggf->ipcSendFd(handle, fd, process, a);
}
int ipcRecvFd(ipcHandle *handle, int *fd) { return mggf->ipcRecvFd(handle, fd); }
int ipcCloseSocket(ipcHandle *handle) { return mggf->ipcCloseSocket(handle); }
int shared_memory_create(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    return mggf->shared_memory_create(name, sz, info);
}
size_t nvshmemi_get_teams_mem_requirement() { return mggf->nvshmemi_get_teams_mem_requirement(); }
cudaError_t cudaMalloc(void **devPtr, size_t size) { return mggf->cudaMalloc(devPtr, size); }
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags) {
    return mggf->cudaHostRegister(ptr, size, flags);
}
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags) {
    return mggf->cudaHostGetDevicePointer(pDevice, pHost, flags);
}
cudaError_t cudaFree(void *devPtr) { return mggf->cudaFree(devPtr); }
cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
    return mggf->cudaIpcGetMemHandle(handle, devPtr);
}
cudaError_t cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
    return mggf->cudaIpcOpenMemHandle(devPtr, handle, flags);
}
cudaError_t cudaIpcCloseMemHandle(void *devPtr) { return mggf->cudaIpcCloseMemHandle(devPtr); }
void shared_memory_close(char *shm_name, nvshmemi_shared_memory_info *info) {
    return mggf->shared_memory_close(shm_name, info);
}
cudaError_t cudaHostUnregister(void *ptr) { return mggf->cudaHostUnregister(ptr); }
const char *cudaGetErrorString(cudaError_t error) { return mggf->cudaGetErrorString(error); }

cudaError_t cudaMemcpy(void *dst, const void *src, size_t nelems, cudaMemcpyKind kind) {
    return mggf->cudaMemcpy(dst, src, nelems, kind);
}

cudaError_t cudaDeviceSynchronize(void) { return mggf->cudaDeviceSynchronize(); }

#ifdef __cplusplus
extern "C" {
#endif

void nvshmemx_char_put_nbi_on_stream(char *dest, const char *source, size_t nelems, int pe,
                                     cudaStream_t stream) {
    return mggf->nvshmemx_char_put_nbi_on_stream(dest, source, nelems, pe, stream);
}

#ifdef __cplusplus
}
#endif

int shared_memory_open(const char *name, size_t sz, nvshmemi_shared_memory_info *info) {
    return mggf->shared_memory_open(name, sz, info);
}
int nvshmem_barrier(nvshmem_team_t team) { return 0; }
bool nvshmemi_team_support_nvls(nvshmemi_team_t *team) {
    return mggf->nvshmemi_team_support_nvls(team);
}
int nvshmemi_team_translate_pe(nvshmemi_team_t *src_team, int src_pe, nvshmemi_team_t *dest_team) {
    return mggf->nvshmemi_team_translate_pe(src_team, src_pe, dest_team);
}

long *nvshmemi_team_get_psync(nvshmemi_team_t *team, nvshmemi_team_op_t op) {
    return mggf->nvshmemi_team_get_psync(team, op);
}

mspace::mspace(void *base, size_t capacity) { Mock_mspace(base, capacity); }
void mspace::add_new_chunk(void *base, size_t capacity) {
    return mmspace->add_new_chunk(base, capacity);
}
int mspace::track_large_chunks(int enable) { return mmspace->track_large_chunks(enable); }
void *mspace::allocate(size_t bytes) { return mmspace->allocate(bytes); }
void mspace::deallocate(void *mem) { return mmspace->deallocate(mem); }
void *mspace::allocate_zeroed(size_t n_elements, size_t elem_size) {
    return mmspace->allocate_zeroed(n_elements, elem_size);
}
void *mspace::allocate_aligned(size_t alignment, size_t bytes) {
    return mmspace->allocate_aligned(alignment, bytes);
}
bool mspace::checkInuse(void *ptr, size_t size) { return mmspace->checkInuse(ptr, size); }

int allgather(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle) {
    return mbh->allgather(sendbuf, recvbuf, bytes, handle);
}

int barrier(struct bootstrap_handle *handle) { return mbh->barrier(handle); }

CUresult pfn_cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop,
                                           CUmemAllocationGranularity_flags option) {
    return mnvcft->pfn_cuMemGetAllocationGranularity(granularity, prop, option);
}
CUresult pfn_cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                         const CUmemAllocationProp *prop, unsigned long long flags) {
    return mnvcft->pfn_cuMemCreate(handle, size, prop, flags);
}
CUresult pfn_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                      CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return mnvcft->pfn_cuMemMap(ptr, size, offset, handle, flags);
}
CUresult pfn_cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc,
                            size_t count) {
    return mnvcft->pfn_cuMemSetAccess(ptr, size, desc, count);
}
CUresult pfn_cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                 unsigned long long flags) {
    return mnvcft->pfn_cuMemAddressReserve(ptr, size, alignment, addr, flags);
}
CUresult pfn_cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    return mnvcft->pfn_cuMemAddressFree(ptr, size);
}
CUresult pfn_cuMemUnmap(CUdeviceptr ptr, size_t size) { return mnvcft->pfn_cuMemUnmap(ptr, size); }
CUresult pfn_cuMemRelease(CUmemGenericAllocationHandle handle) {
    return mnvcft->pfn_cuMemRelease(handle);
}
CUresult pfn_cuMemExportToShareableHandle(void *shareableHandle,
                                          CUmemGenericAllocationHandle handle,
                                          CUmemAllocationHandleType handleType,
                                          unsigned long long flags) {
    return mnvcft->pfn_cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags);
}
CUresult pfn_cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                                   CUdeviceptr ptr) {
    return mnvcft->pfn_cuPointerSetAttribute(value, attribute, ptr);
}

CUresult pfn_cuCtxGetDevice(CUdevice *device) { return mnvcft->pfn_cuCtxGetDevice(device); }

CUresult pfn_cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle,
                                            CUmemAllocationHandleType shHandleType) {
    return mnvcft->pfn_cuMemImportFromShareableHandle(handle, osHandle, shHandleType);
}

int nvls::nvshmemi_nvls_rsc::unbind_group_mem(CUmemGenericAllocationHandle *mc_handle,
                                              off_t mc_offset, size_t mem_size) {
    return mnvls->unbind_group_mem(mc_handle, mc_offset, mem_size);
}
int nvls::nvshmemi_nvls_rsc::subscribe_group(CUmemGenericAllocationHandle *mc_handle) {
    return mnvls->subscribe_group(mc_handle);
}
int nvls::nvshmemi_nvls_rsc::import_group(char *shareable_handle,
                                          CUmemGenericAllocationHandle *mc_handle,
                                          uint64_t mem_size) {
    return mnvls->import_group(shareable_handle, mc_handle, mem_size);
}
int nvls::nvshmemi_nvls_rsc::bind_group_mem(CUmemGenericAllocationHandle *mc_handle,
                                            nvshmem_mem_handle_t *mem_handle, size_t mem_size,
                                            off_t mem_offset, off_t mc_offset) {
    return mnvls->bind_group_mem(mc_handle, mem_handle, mem_size, mem_offset, mc_offset);
}
int nvls::nvshmemi_nvls_rsc::map_group_mem(CUmemGenericAllocationHandle *mc_handle, size_t mem_size,
                                           off_t mem_offset, off_t mc_offset) {
    return mnvls->map_group_mem(mc_handle, mem_size, mem_offset, mc_offset);
}
int nvls::nvshmemi_nvls_rsc::unmap_group_mem(off_t mc_offset, size_t mem_size) {
    return mnvls->unmap_group_mem(mc_offset, mem_size);
}
int nvls::nvshmemi_nvls_rsc::export_group(uint64_t mem_size, char *shareable_handle) {
    return mnvls->export_group(mem_size, shareable_handle);
}

static bootstrap_handle_t initialize_bootstrap_handle() {
    bootstrap_handle_t new_bootstrap_handle = {.version = 0,
                                               .pg_rank = 0,
                                               .pg_size = 0,
                                               .mype_node = 0,
                                               .npes_node = 0,
                                               .allgather = &allgather,
                                               .alltoall = NULL,
                                               .barrier = &barrier,
                                               .global_exit = NULL,
                                               .finalize = NULL,
                                               .show_info = NULL,
                                               .pre_init_ops = NULL,
                                               .comm_state = NULL};
    return new_bootstrap_handle;
}

static nvshmemi_cuda_fn_table initialize_CUDA_fn_table() {
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = {
        .pfn_cuCtxGetDevice = &pfn_cuCtxGetDevice,
        .pfn_cuCtxSynchronize = NULL,
        .pfn_cuDeviceGet = NULL,
        .pfn_cuDeviceGetAttribute = NULL,
        .pfn_cuPointerSetAttribute = &pfn_cuPointerSetAttribute,
        .pfn_cuModuleGetGlobal = NULL,
        .pfn_cuGetErrorString = NULL,
        .pfn_cuGetErrorName = NULL,
        .pfn_cuCtxSetCurrent = NULL,
        .pfn_cuDevicePrimaryCtxRetain = NULL,
        .pfn_cuCtxGetCurrent = NULL,
        .pfn_cuCtxGetFlags = NULL,
        .pfn_cuCtxSetFlags = NULL,
        .pfn_cuFlushGPUDirectRDMAWrites = NULL,
        .pfn_cuMemGetHandleForAddressRange = NULL,
        .pfn_cuMemCreate = &pfn_cuMemCreate,
        .pfn_cuMemAddressReserve = &pfn_cuMemAddressReserve,
        .pfn_cuMemAddressFree = &pfn_cuMemAddressFree,
        .pfn_cuMemMap = &pfn_cuMemMap,
        .pfn_cuMemGetAllocationGranularity = &pfn_cuMemGetAllocationGranularity,
        .pfn_cuMemImportFromShareableHandle = &pfn_cuMemImportFromShareableHandle,
        .pfn_cuMemExportToShareableHandle = &pfn_cuMemExportToShareableHandle,
        .pfn_cuMemRelease = &pfn_cuMemRelease,
        .pfn_cuMemSetAccess = &pfn_cuMemSetAccess,
        .pfn_cuMemUnmap = &pfn_cuMemUnmap,
        .pfn_cuMulticastCreate = NULL,
        .pfn_cuMulticastAddDevice = NULL,
        .pfn_cuMulticastBindMem = NULL,
        .pfn_cuMulticastUnbind = NULL,
        .pfn_cuMulticastGetGranularity = NULL,
        .pfn_cuStreamWriteValue64 = NULL,
        .pfn_cuStreamWaitValue64 = NULL,
        .pfn_cuInit = NULL,
        .pfn_cuGetProcAddress = NULL};
    return new_cuda_fn_table;
}

TEST(map_heap_chunk, tested_with_heapAllocate) {
    state.mype = 0;
    state.npes = 1;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 1;
    state.transport_map[1] = 1;
    state.num_initialized_transports = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 1;
    new_tcurr.cap[1] = 1;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    size_t size = 10;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;
    mbh = new Mock_bootstrap_handle();

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;
    mnvcft = new Mock_nvshmemi_cuda_function_table();

    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mnvmrt = nvshmemi_mem_remote_transport::get_instance();

    int value = 42;
    void *mspace_ptr = &value;

    mmspace = new Mock_mspace(mspace_ptr, size);

    mggf = new Mock_general_global_functions();

    // Mocked function applied during constructor creation
    EXPECT_CALL(*mnvmpt, get_mem_handle_type()).WillOnce(Return(CU_MEM_HANDLE_TYPE_WIN32));

    // is_symmetric()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(DoAll(Assign(&state.npes, 2), Return(0)));

    // allocate_virtual_memory_from_mspace()
    EXPECT_CALL(*mmspace, allocate(_)).WillRepeatedly(Return(nullptr));
    //

    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*mmspace, add_new_chunk(_, _)).WillOnce(Return());
    // allocate_physical_memory_to_heap()
    EXPECT_CALL(*mnvcft, pfn_cuMemCreate(_, _, _, 0)).WillOnce(Return(CUDA_SUCCESS));
    EXPECT_CALL(*mnvcft, pfn_cuMemMap(_, _, _, _, 0))
        .WillOnce(Return(CUDA_SUCCESS))
        .WillOnce(Return(CUDA_SUCCESS));
    EXPECT_CALL(*mnvcft, pfn_cuMemSetAccess(_, _, _, 1))
        .WillOnce(Return(CUDA_SUCCESS))
        .WillOnce(Return(CUDA_SUCCESS));

    // import_memory()
    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));
    EXPECT_CALL(*mnvcft, pfn_cuMemImportFromShareableHandle(_, _, _))
        .WillOnce(Return(CUDA_SUCCESS));
    EXPECT_CALL(*mnvcft, pfn_cuMemRelease(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mbh, barrier(_)).WillOnce(Return(0));
    EXPECT_CALL(*mggf, nvshmemi_update_device_state()).WillOnce(Return(0));

    nvshmemi_symmetric_heap_vidmem_dynamic_vmm *heap =
        new nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    mnvmpt->set_mem_granularity_(*heap, 1);
    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_peer_heap_base_p2p_(*heap);

    void *ptr = heap->heap_malloc(size);
    EXPECT_EQ(ptr, nullptr);

    delete heap;
    delete mnvcft;
    delete mbh;
    delete mggf;
    delete mmspace;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    delete mnvmrt;
    delete mnvmpt;
    return result;
}
