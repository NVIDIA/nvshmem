#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <gmock/gmock-more-actions.h>
#include <vector>
#include <cstdint>

// Declarations of functions and structs should be included, definitions we create on our own

#include <string.h>
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
    MOCK_METHOD1(cudaIpcCloseMemHandle, cudaError_t(void *devPtr));
    MOCK_METHOD4(cudaMemcpy,
                 cudaError_t(void *dst, const void *src, size_t count, cudaMemcpyKind kind));
    MOCK_METHOD0(cudaDeviceSynchronize, cudaError_t(void));

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
        .pfn_cuCtxGetDevice = NULL,
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
        .pfn_cuMemImportFromShareableHandle = NULL,
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

ACTION_P3(set_heap_size, mnvmpt, heap, size) { mnvmpt->set_heap_size_(*heap, size); }

TEST(setup_symmetric_heap, get_transport_mem_handle_successful_no_paths) {
    state.mype = 0;
    state.npes = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, 0, _, _)).WillOnce(Return(0));
    //

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    int reserve_heap_value = heap->reserve_heap();

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    nvshmem_mem_handle *get_transport_mem_handles_result =
        heap->get_transport_mem_handle(address, len, pe, transport_idx);
    EXPECT_FALSE(get_transport_mem_handles_result == nullptr);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}

TEST(setup_symmetric_heap, allgather_peer_base_fail) {
    state.mype = 0;
    state.npes = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(7));
    //

    // cleanup_symmetric_heap()
    EXPECT_CALL(*mggf, cudaFree(_)).WillRepeatedly(Return(cudaSuccess));
    //

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    int reserve_heap_value = heap->reserve_heap();

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 7);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}

TEST(setup_symmetric_heap, get_mem_handle_addr_offset_valid_idx) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(1, sizeof(int));
    new_tcurr.cap[0] = 0;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes_node, 1), Return(0)));

    // destructor
    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 1;

    int reserve_heap_value = heap->reserve_heap();

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    size_t get_mem_handle_addr_offset_result = heap->get_mem_handle_addr_offset(address);
    EXPECT_TRUE(get_mem_handle_addr_offset_result > 140000000000000);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}

TEST(setup_symmetric_heap, register_heap_memory_handle_fail) {
    state.mype = 0;
    state.npes = 1;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(1, sizeof(int));
    new_tcurr.cap[0] = 0;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(1));

    EXPECT_CALL(*mggf, cudaFree(_)).WillRepeatedly(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    nvshmemi_device_state.enable_rail_opt = 1;

    int reserve_heap_value = heap->reserve_heap();

    size_t setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 7);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}

TEST(setup_symmetric_heap, map_heap_memory_forEachIf) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 0;
    new_tcurr.cap[1] = 1;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes, 2), Assign(&state.npes_node, 1), Return(0)));

    EXPECT_CALL(*mggf, cudaIpcOpenMemHandle(_, _, _)).WillOnce(Return(cudaSuccess));

    // destructor
    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 0;

    int reserve_heap_value = heap->reserve_heap();

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    size_t get_mem_handle_addr_offset_result = heap->get_mem_handle_addr_offset(address);
    EXPECT_TRUE(get_mem_handle_addr_offset_result == 4);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}
TEST(setup_symmetric_heap, map_heap_chunk_cudaIpcFail) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 0;
    new_tcurr.cap[1] = 1;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_function_table();
    mggf = new Mock_general_global_functions();

    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr, capacity);

    // MOCKED FUNCTION FOR RESERVE_HEAP

    // set_heap_size
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return((size_t)5));

    // allocate_heap_memory()
    int heap_base = 1;
    void *heap_base_ptr = &heap_base;
    size_t heap_size = 10;

    EXPECT_CALL(*mggf, cudaMalloc(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(heap_base_ptr), Return(cudaSuccess)));
    //

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(0));
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes, 2), Assign(&state.npes_node, 1), Return(0)));

    EXPECT_CALL(*mggf, cudaIpcOpenMemHandle(_, _, _)).WillOnce(Return(cudaErrorInvalidValue));

    // destructor
    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    nvshmemi_device_state.enable_rail_opt = 0;

    int reserve_heap_value = heap->reserve_heap();

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    delete heap;
    delete mbh;
    delete mnvcft;
    delete mmspace;
    delete mggf;
}

TEST(setup_symmetric_heap, register_heap_chunk_if_if) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 1;
    new_tcurr.cap[1] = 1;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    new_bootstrap_handle.mype_node = 0;
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mbh = new Mock_bootstrap_handle();

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _))
        .WillOnce(Return(0))
        .WillOnce(DoAll(Assign(&state.npes, 2), Return(0)));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes_node, 1), Return(0)));

    nvshmemi_symmetric_heap_sysmem_static_shm *heap =
        new nvshmemi_symmetric_heap_sysmem_static_shm(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 0;
    nvshmemi_options.SYMMETRIC_SIZE = 1;

    mnvmpt->set_heap_size_(*heap, 1);
    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_mem_granularity_(*heap, 1);

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    delete heap;
    delete mbh;
}

TEST(setup_symmetric_heap, register_heap_chunk_else_else) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 1;
    new_tcurr.cap[1] = 1;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    new_bootstrap_handle.mype_node = 0;
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mbh = new Mock_bootstrap_handle();

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _))
        .WillOnce(Return(0))
        .WillOnce(DoAll(Assign(&state.npes, 2), Return(0)))
        .WillOnce(DoAll(Assign(&state.transports[0]->cap[0], 0), Assign(&state.npes, 1), Return(0)))
        .WillOnce(DoAll(Assign(&state.npes, 1), Return(0)));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes_node, 1), Return(0)))
        .WillOnce(DoAll(Assign(&nvshmemi_device_state.enable_rail_opt, 0), Return(0)));

    nvshmemi_symmetric_heap_sysmem_static_shm *heap =
        new nvshmemi_symmetric_heap_sysmem_static_shm(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 1;
    nvshmemi_options.SYMMETRIC_SIZE = 1;

    mnvmpt->set_heap_size_(*heap, 1);
    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_mem_granularity_(*heap, 1);

    int setup_result = heap->setup_symmetric_heap();
    int setup_result_2 = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);
    EXPECT_EQ(setup_result_2, 0);

    delete heap;
    delete mbh;
}

TEST(setup_symmetric_heap, shm_reg_heap_mem_handle) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 0;
    new_tcurr.cap[1] = 0;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    new_bootstrap_handle.mype_node = 0;
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mbh = new Mock_bootstrap_handle();

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _))
        .WillOnce(Return(0))
        .WillOnce(DoAll(Assign(&state.npes, 2), Return(0)));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes_node, 1), Return(0)));
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(0));

    nvshmemi_symmetric_heap_sysmem_static_shm *heap =
        new nvshmemi_symmetric_heap_sysmem_static_shm(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 0;
    nvshmemi_options.SYMMETRIC_SIZE = 1;

    mnvmpt->set_heap_size_(*heap, 1);
    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_mem_granularity_(*heap, 1);

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    delete heap;
    delete mbh;
}

TEST(setup_symmetric_heap, shm_reg_heap_mem_handle_optEq1s) {
    state.mype = 0;
    state.npes = 1;
    state.npes_node = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    new_tcurr.cap = (int *)calloc(2, sizeof(int));
    new_tcurr.cap[0] = 0;
    new_tcurr.cap[1] = 0;
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    state.transport_map = (int *)calloc(2, sizeof(int));
    state.transport_map[0] = 0;
    state.transport_map[1] = 1;

    size_t size = 1;
    size_t count = 1;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    new_bootstrap_handle.mype_node = 0;
    nvshmemi_boot_handle = new_bootstrap_handle;

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    int value = 42;
    void *ptr = &value;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mbh = new Mock_bootstrap_handle();

    // MOCKED FUNCTIONS FOR SETUP_SYMMETRIC_HEAP

    // allgather_peer_base() + register_heap_chunk()
    EXPECT_CALL(*mbh, allgather(_, _, _, _))
        .WillOnce(Return(0))
        .WillOnce(DoAll(Assign(&state.npes, 2), Return(0)));
    //

    // register_heap_memory
    // register_heap_chunk
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _))
        .WillOnce(DoAll(Assign(&state.npes_node, 1), Return(0)));
    EXPECT_CALL(*mnvmrt, register_mem_handle(_, _, _, _, _)).WillOnce(Return(0));

    nvshmemi_symmetric_heap_sysmem_static_shm *heap =
        new nvshmemi_symmetric_heap_sysmem_static_shm(&state);

    int address_value = 0;
    void *address = &address_value;
    size_t len_val = 10;
    size_t *len = &len_val;
    int pe = 1;
    int transport_idx = 10;

    nvshmemi_device_state.enable_rail_opt = 1;
    nvshmemi_options.SYMMETRIC_SIZE = 1;

    mnvmpt->set_heap_size_(*heap, 1);
    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_mem_granularity_(*heap, 1);

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);

    delete heap;
    delete mbh;
}

TEST(cleanup_symmetric_heap, pinned_no_paths) {
    state.npes = 0;

    mnvmrt = nvshmemi_mem_remote_transport::get_instance();  // perhaps no transport object is
                                                             // needed

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    int cleanup_result = heap->cleanup_symmetric_heap();
    EXPECT_EQ(cleanup_result, 0);

    delete heap;
}

TEST(cleanup_symmetric_heap, first_for_each) {
    state.npes = 1;
    state.mype = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 0;

    struct nvshmem_transport new_tcurr = {};
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mggf = new Mock_general_global_functions();

    // setup_symmetric_heap calls for handles push back
    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));
    EXPECT_CALL(*mnvmrt, gather_mem_handles(_, _, _, _)).WillOnce(Return(0));

    // cleanup_symmetric_heap calls
    EXPECT_CALL(*mnvmrt, release_mem_handles(_, _)).WillOnce(Return(0));
    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_mem_granularity_(*heap, 821);
    mnvmpt->set_heap_size_(*heap, 1);

    int setup_result = heap->setup_symmetric_heap();
    EXPECT_EQ(setup_result, 0);
    int cleanup_result = heap->cleanup_symmetric_heap();
    EXPECT_EQ(cleanup_result, 0);

    delete heap;
    delete mbh;
    delete mggf;
}

TEST(cleanup_symmetric_heap, second_for_each_if) {
    state.npes = 1;
    state.mype = 0;
    state.num_initialized_transports = 1;
    state.transport_bitmap = 0;

    struct nvshmem_transport new_tcurr = {};
    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));
    state.transports[0] = &new_tcurr;

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;

    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    mnvmrt = nvshmemi_mem_remote_transport::get_instance();
    mbh = new Mock_bootstrap_handle();
    mggf = new Mock_general_global_functions();

    // setup_symmetric_heap calls for handles push back
    EXPECT_CALL(*mbh, allgather(_, _, _, _))
        .WillOnce(Return(0))
        .WillOnce(DoAll(Assign(&state.mype, 1), Return(7)));

    // cleanup_symmetric_heap calls
    EXPECT_CALL(*mggf, cudaIpcCloseMemHandle(_)).WillOnce(Return(cudaErrorCudartUnloading));
    EXPECT_CALL(*mggf, cudaFree(_)).WillOnce(Return(cudaSuccess));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    mnvmpt->set_heap_base_(*heap);
    mnvmpt->set_heap_size_(*heap, 1);

    int cleanup_final_result = heap->setup_symmetric_heap();
    EXPECT_EQ(cleanup_final_result, 7);

    delete heap;
    delete mbh;
    delete mggf;
}

TEST(reserve_heap, pinned_no_errors) {
    state.npes = 1;
    state.mype = 0;

    mggf = new Mock_general_global_functions();

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;
    mnvcft = new Mock_nvshmemi_cuda_function_table();

    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    int val1 = 42;
    void *ptr_for_mspace = &val1;
    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr_for_mspace, capacity);

    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return(capacity));
    EXPECT_CALL(*mggf, cudaMalloc(_, _)).WillOnce(Return(cudaSuccess));
    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _)).WillOnce(Return(CUDA_SUCCESS));
    EXPECT_CALL(*mmspace, track_large_chunks(_)).WillOnce(Return(0));

    nvshmemi_symmetric_heap_vidmem_static_pinned *heap =
        new nvshmemi_symmetric_heap_vidmem_static_pinned(&state);

    mnvmpt->set_heap_base_(*heap);

    int reserve_heap_result = heap->reserve_heap();

    EXPECT_EQ(reserve_heap_result, 0);

    delete heap;
    delete mmspace;
    delete mggf;
    delete mnvcft;
}

TEST(reserve_heap, shm_no_errors) {
    state.npes = 1;
    state.mype = 0;

    mggf = new Mock_general_global_functions();

    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    nvshmemi_boot_handle = new_bootstrap_handle;
    nvshmemi_boot_handle.mype_node = 0;
    mbh = new Mock_bootstrap_handle();

    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_fn_table();
    nvshmemi_cuda_syms = &new_cuda_fn_table;
    mnvcft = new Mock_nvshmemi_cuda_function_table();

    mnvmpt = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    int val1 = 42;
    void *ptr_for_mspace = &val1;
    size_t capacity = 10;
    mmspace = new Mock_mspace(ptr_for_mspace, capacity);

    nvshmemi_shared_memory_info_t mock_mem_info = {};
    void *ptr = malloc(1 * sizeof(int));
    mock_mem_info.addr = ptr;
    EXPECT_CALL(*mggf, nvshmemi_get_teams_mem_requirement()).WillOnce(Return(2));
    EXPECT_CALL(*mggf, shared_memory_create(_, _, _))
        .WillOnce(DoAll(SetArgPointee<2>(mock_mem_info), Return(0)));
    EXPECT_CALL(*mbh, barrier(_))
        .WillOnce(DoAll(Assign(&nvshmemi_boot_handle.mype_node, 1), Return(0)))
        .WillOnce(Return(0));

    EXPECT_CALL(*mggf, cudaHostRegister(mock_mem_info.addr, _, _)).WillOnce(Return(cudaSuccess));
    EXPECT_CALL(*mggf, cudaHostGetDevicePointer(_, _, 0)).WillOnce(Return(cudaSuccess));

    EXPECT_CALL(*mnvcft, pfn_cuPointerSetAttribute(_, _, _))
        .WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    EXPECT_CALL(*mggf, cudaHostUnregister(mock_mem_info.addr)).WillOnce(Return(cudaSuccess));
    EXPECT_CALL(*mggf, shared_memory_close(_, _)).WillOnce(Return());

    nvshmemi_symmetric_heap_sysmem_static_shm *heap =
        new nvshmemi_symmetric_heap_sysmem_static_shm(&state);

    EXPECT_CALL(*mggf, shared_memory_open(_, _, _))
        .WillOnce(DoAll(set_heap_size(mnvmpt, heap, 4), Assign(&nvshmemi_boot_handle.mype_node, 0),
                        Return(0)));

    mnvmpt->set_mem_granularity_(*heap, 4);
    mnvmpt->set_heap_base_(*heap);

    int reserve_heap_result = heap->reserve_heap();

    EXPECT_EQ(reserve_heap_result, 2);

    delete heap;
    delete mmspace;
    delete mggf;
    delete mnvcft;
    delete mbh;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    delete mnvmrt;
    delete mnvmpt;
    return result;
}
