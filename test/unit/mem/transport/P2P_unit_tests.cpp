/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

// Declarations of functions and structs should be included, definitions we create on our own
#include "nvshmemi_mem_transport.hpp"
#include "internal/host/nvshmemi_symmetric_heap.hpp"
#include "util.h"
#include "debug.h"
#include "nvmlwrap.h"
#include "nvshmemi_types.h"
#include "nvshmem_internal.h"
#include "cuda_runtime_api.h"
#include "nvml.h"
#include "cuda.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"

using namespace std;
using namespace testing;

nvshmemi_mem_p2p_transport *nvshmemi_mem_p2p_transport::p2p_objref_ = nullptr;
// Structs necessary to make sure that the linker finds references
bootstrap_handle_t nvshmemi_boot_handle;
struct nvshmemi_cuda_fn_table *nvshmemi_cuda_syms = NULL;  // 0 init
int nvshmem_debug_level;
int nvshmemi_cuda_driver_version;
nvshmemi_state_t state;
nvshmemi_options_s nvshmemi_options;

/*-------------------------------------------------Wrapper Function Declarations For
 * Mocking--------------------------------------------------------*/
char *nvshmemu_hexdump(void *ptr, size_t len);
cudaError_t cudaGetDeviceCount(int *count);
int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
int nvshmemi_update_device_state();

nvmlReturn_t nvmlInit();
nvmlReturn_t nvmlShutdown();

CUresult pfn_cuDeviceGet(CUdevice *device, int ordinal);
CUresult pfn_cuCtxGetDevice(CUdevice *device);
CUresult pfn_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...) {
    if (nvshmem_debug_level <= NVSHMEM_LOG_NONE) {
        return;
    }
}

nvshmemi_symmetric_heap_vidmem_dynamic_vmm::~nvshmemi_symmetric_heap_vidmem_dynamic_vmm() {}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::reserve_heap() { return 0; }
int nvshmemi_symmetric_heap::map_heap_range_by_size(void *buf, size_t size) { return 1; }
int nvshmemi_symmetric_heap::update_heap_handle_cache(void *buf, size_t size) { return 1; }
int nvshmemi_symmetric_heap::register_heap_memory(nvshmem_mem_handle_t *input, void *buffer,
                                                  size_t size) {
    return 1;
}
int nvshmemi_symmetric_heap_dynamic::register_heap_memory(nvshmem_mem_handle_t *mem_handle_in,
                                                          void *buf, size_t size) {
    return 1;
}
void *nvshmemi_symmetric_heap::heap_allocate(size_t size, size_t count, size_t alignment,
                                             int type) {
    void *ptr = NULL;
    return ptr;
}
void nvshmemi_symmetric_heap::heap_deallocate(void *ptr) { return; }
void *nvshmemi_symmetric_heap::heap_malloc(size_t size) {
    void *ptr = NULL;
    return ptr;
}
void *nvshmemi_symmetric_heap::heap_calloc(size_t size, size_t count) {
    void *ptr = NULL;
    return ptr;
}
void *nvshmemi_symmetric_heap::heap_align(size_t size, size_t alignment) {
    void *ptr = NULL;
    return ptr;
}
int nvshmemi_symmetric_heap_dynamic::setup_mspace() { return 1; }
void nvshmemi_symmetric_heap::update_idx_in_handle(void *addr, size_t size) { return; }
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::setup_symmetric_heap() { return 0; }
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::cleanup_symmetric_heap() { return 0; }
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_create_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return 0;
};
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_bind_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_map_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return 0;
}
void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_unmap_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return;
}
void nvshmemi_symmetric_heap_vidmem_dynamic_vmm::nvls_unbind_heap_memory_by_team(
    nvshmemi_team_t *team) {
    return;
}
void *nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_symmetric_memory(size_t size,
                                                                            size_t count,
                                                                            size_t alignment,
                                                                            int type) {
    void *ptr = NULL;
    return ptr;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::map_heap_range_by_pe(int pe_id, int transport_idx,
                                                                     char *buf = nullptr,
                                                                     size_t size = 0) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::exchange_heap_memory_handle(
    nvshmem_mem_handle_t *local_handles) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::import_memory(nvshmem_mem_handle_t *mem_handle,
                                                              void **buf, size_t length) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::release_memory(void *buf, size_t size = 0) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::allocate_physical_memory_to_heap(size_t size) {
    return 0;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::export_memory(nvshmem_mem_handle_t *mem_handle,
                                                              nvshmem_mem_handle_t *mem_handle_in) {
    return 0;
}

class Mock_nvshmemi_mem_p2p_transport {
   public:
    MOCK_METHOD2(nvshmemu_hexdump, char *(void *ptr, size_t len));
    MOCK_METHOD0(getpid, __pid_t());
    MOCK_METHOD2(nvshmemi_nvml_ftable_init,
                 int(struct nvml_function_table *nvml_ftable, void **nvml_handle));
    MOCK_METHOD2(nvshmemi_nvml_ftable_fini,
                 void(struct nvml_function_table *nvml_ftable, void **nvml_handle));
    MOCK_METHOD1(cudaGetDeviceCount, cudaError_t(int *count));
    MOCK_METHOD2(cudaGetDeviceProperties, cudaError_t(cudaDeviceProp *prop, int device));
    MOCK_METHOD0(nvshmemi_update_device_state, int());
};

struct Mock_bootstrap_handle {
   public:
    MOCK_METHOD4(allgather, int(const void *sendbuf, void *recvbuf, int bytes,
                                struct bootstrap_handle *handle));
};

class Mock_nvml_function_table {
   public:
    MOCK_METHOD0(nvmlInit, nvmlReturn_t());
    MOCK_METHOD2(nvmlDeviceGetHandleByPciBusId,
                 nvmlReturn_t(const char *pciBusId, nvmlDevice_t *device));
    MOCK_METHOD0(nvmlShutdown, nvmlReturn_t());
    MOCK_METHOD2(nvmlDeviceGetGpuFabricInfoV,
                 nvmlReturn_t(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info));
};

class Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm
    : public nvshmemi_symmetric_heap_vidmem_dynamic_vmm {
   public:
    explicit Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(nvshmemi_state_t *state)
        : nvshmemi_symmetric_heap_vidmem_dynamic_vmm(state) {}
    ~Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm() = default;
    MOCK_METHOD2(map_heap_range_by_size, int(void *buf, size_t size));
    MOCK_METHOD2(update_heap_handle_cache, int(void *buf, size_t size));
    MOCK_METHOD4(map_heap_range_by_pe, int(int pe_id, int transport_idx, char *buf, size_t size));
    MOCK_METHOD0(reserve_heap, int());
    MOCK_METHOD4(allocate_symmetric_memory,
                 void *(size_t size, size_t count, size_t alignment, int type));
    MOCK_METHOD3(export_memory, int(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length));
    MOCK_METHOD0(setup_symmetric_heap, int(void));
    MOCK_METHOD0(cleanup_symmetric_heap, int(void));
    MOCK_METHOD1(exchange_heap_memory_handle, int(nvshmem_mem_handle_t *local_handles));
    MOCK_METHOD2(export_memory,
                 int(nvshmem_mem_handle_t *mem_handle, nvshmem_mem_handle_t *mem_handle_in));
    MOCK_METHOD3(import_memory, int(nvshmem_mem_handle_t *mem_handle, void **buf, size_t length));
    MOCK_METHOD2(release_memory, int(void *buf, size_t size));
    MOCK_METHOD0(get_state, nvshmemi_state_t *());
    MOCK_METHOD3(register_heap_memory, int(nvshmem_mem_handle_t *input, void *buffer, size_t size));
    MOCK_METHOD4(heap_allocate, void *(size_t size, size_t count, size_t alignment, int type));
    MOCK_METHOD1(heap_deallocate, void(void *ptr));
    MOCK_METHOD1(heap_malloc, void *(size_t size));
    MOCK_METHOD2(heap_calloc, void *(size_t size, size_t count));
    MOCK_METHOD2(heap_align, void *(size_t size, size_t alignment));
    MOCK_METHOD0(setup_mspace, int());
    MOCK_METHOD2(update_idx_in_handle, void(void *addr, size_t size));
    MOCK_METHOD1(nvls_create_heap_memory_by_team, int(nvshmemi_team_t *team));
    MOCK_METHOD1(nvls_bind_heap_memory_by_team, int(nvshmemi_team_t *team));
    MOCK_METHOD1(nvls_map_heap_memory_by_team, int(nvshmemi_team_t *team));
    MOCK_METHOD1(nvls_unmap_heap_memory_by_team, void(nvshmemi_team_t *team));
    MOCK_METHOD1(nvls_unbind_heap_memory_by_team, void(nvshmemi_team_t *team));
    MOCK_METHOD1(allocate_physical_memory_to_heap, int(size_t size));
};

struct Mock_nvshmemi_cuda_fn_table {
   public:
    MOCK_METHOD1(pfn_cuCtxGetDevice, CUresult(CUdevice *device));
    MOCK_METHOD2(pfn_cuDeviceGet, CUresult(CUdevice *device, int ordinal));
    MOCK_METHOD3(pfn_cuDeviceGetAttribute,
                 CUresult(int *pi, CUdevice_attribute attrib, CUdevice dev));
};

Mock_nvshmemi_mem_p2p_transport *mpt;
Mock_bootstrap_handle *mbh;
Mock_nvml_function_table *mnft = nullptr;
Mock_nvshmemi_cuda_fn_table *mnvcft;

/*-------Mock_nvshmemi_mem_p2p_transport functions that provide a level of abstraction for the
 * tester functions--------------*/

char *nvshmemu_hexdump(void *ptr, size_t len) { return mpt->nvshmemu_hexdump(ptr, len); }
cudaError_t cudaGetDeviceCount(int *count) { return mpt->cudaGetDeviceCount(count); }
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    return;
}
int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    return mpt->nvshmemi_nvml_ftable_init(nvml_ftable, nvml_handle);
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return mpt->cudaGetDeviceProperties(prop, device);
}
int nvshmemi_update_device_state() { return mpt->nvshmemi_update_device_state(); }

/*-----------------------Mocked NVML functions that allow for unit testing of
 * mem_transport.cpp------------------------------*/
nvmlReturn_t nvmlInit() { return mnft->nvmlInit_v2(); }
nvmlReturn_t nvmlShutdown() { return mnft->nvmlShutdown(); }
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    return mnft->nvmlDeviceGetHandleByPciBusId(pciBusId, device);
}
nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info) {
    return mnft->nvmlDeviceGetGpuFabricInfoV(device, info);
}
/*-----------------------Mocked CUDA functions that allow for unit testing of
 * mem_transport.cpp------------------------------*/

CUresult pfn_cuCtxGetDevice(CUdevice *device) { return mnvcft->pfn_cuCtxGetDevice(device); }
CUresult pfn_cuDeviceGet(CUdevice *device, int ordinal) {
    return mnvcft->pfn_cuDeviceGet(device, ordinal);
}
CUresult pfn_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    return mnvcft->pfn_cuDeviceGetAttribute(pi, attrib, dev);
}

/*-----------------------Mocked Bootstrap functions that allow for unit testing of
 * mem_transport.cpp--------------------------*/

int allgather(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle) {
    return mbh->allgather(sendbuf, recvbuf, bytes, handle);
}
/*-------------------------------------------------Helper
 * functions--------------------------------------------------------*/

/*
Static function that initializes a new nvml_function_table with
function pointers pointing to mocked functions found in this file
*/
static nvml_function_table initialize_NVML_func_table() {
    struct nvml_function_table new_nvml_ftable = {
        .nvmlInit_v2 = &nvmlInit,
        .nvmlShutdown = &nvmlShutdown,
        .nvmlDeviceGetHandleByPciBusId = &nvmlDeviceGetHandleByPciBusId,
        .nvmlDeviceGetP2PStatus = NULL,
        .nvmlDeviceGetGpuFabricInfoV = &nvmlDeviceGetGpuFabricInfoV};

    return new_nvml_ftable;
}

static nvshmemi_cuda_fn_table initialize_CUDA_func_table() {
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = {
        .pfn_cuCtxGetDevice = &pfn_cuCtxGetDevice,
        .pfn_cuCtxSynchronize = NULL,
        .pfn_cuDeviceGet = &pfn_cuDeviceGet,
        .pfn_cuDeviceGetAttribute = &pfn_cuDeviceGetAttribute,
        .pfn_cuPointerSetAttribute = NULL,
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
        .pfn_cuMemCreate = NULL,
        .pfn_cuMemAddressReserve = NULL,
        .pfn_cuMemAddressFree = NULL,
        .pfn_cuMemMap = NULL,
        .pfn_cuMemGetAllocationGranularity = NULL,
        .pfn_cuMemImportFromShareableHandle = NULL,
        .pfn_cuMemExportToShareableHandle = NULL,
        .pfn_cuMemRelease = NULL,
        .pfn_cuMemSetAccess = NULL,
        .pfn_cuMemUnmap = NULL,
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

static bootstrap_handle_t initialize_bootstrap_handle() {
    bootstrap_handle_t new_bootstrap_handle = {.version = 0,
                                               .pg_rank = 0,
                                               .pg_size = 0,
                                               .mype_node = 0,
                                               .npes_node = 0,
                                               .allgather = &allgather,
                                               .alltoall = NULL,
                                               .barrier = NULL,
                                               .global_exit = NULL,
                                               .finalize = NULL,
                                               .show_info = NULL,
                                               .pre_init_ops = NULL,
                                               .comm_state = NULL};

    return new_bootstrap_handle;
}

static void deleteGlobalPtrs(string &abcd) {
    if (abcd[0] == 'a') {
        delete mpt;
    }

    if (abcd[1] == 'b') {
        delete mbh;
    }

    if (abcd[2] == 'c') {
        delete mnft;
    }

    if (abcd[3] == 'd') {
        delete mnvcft;
    }
}

TEST(constructor_destructor_test, mem_transport_constructor_P2P_no_branches) {
    // Arrange
    state.mype = 0;
    state.npes = 100;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _)).WillOnce(DoAll(Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _)).WillOnce(Return(cudaSuccess));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_constructor_P2P_ftable_init_fail) {
    state.mype = 0;
    state.npes = 0;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(1)));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    EXPECT_EQ(0, transport->is_initialized());

    string abcd = "abc-";
    delete transport;
    deleteGlobalPtrs(abcd);
}

// nvml_status set to 1
TEST(constructor_destructor_test, mem_transport_constructor_nvmlInit_fail) {
    state.mype = 0;
    state.npes = 0;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    // mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_ERROR_INVALID_ARGUMENT));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    EXPECT_EQ(0, transport->is_initialized());

    string abcd = "abc-";
    delete transport;
    deleteGlobalPtrs(abcd);
}

// Status set to 1
TEST(constructor_destructor_test, mem_transport_cudaGetDeviceCountFail) {
    state.mype = 0;
    state.npes = 100;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaErrorInvalidValue)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Assert

    EXPECT_EQ(0, transport->is_initialized());

    string abcd = "abc-";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_cuCtxGetDevice_fail) {
    state.mype = 0;
    state.npes = 100;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_ERROR_INVALID_VALUE));  // 2

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(0, transport->is_initialized());

    delete transport;

    string abcd = "abcd";
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_cuDeviceGetFail) {
    // Arrange
    state.mype = 0;
    state.npes = 100;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _)).WillOnce(Return(CUDA_ERROR_INVALID_VALUE));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(0, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_cuDeviceGetDeviceProperties) {
    // Arrange
    state.mype = 0;
    state.npes = 10;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _)).WillRepeatedly(Return(cudaSuccess));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_cuDeviceGetDevicePropertiesFail) {
    // Arrange
    state.mype = 0;
    state.npes = 10;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillRepeatedly(Return(cudaErrorCudartUnloading));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_discoverNvmlDeviceProps) {
    // Arrange
    state.mype = 0;
    state.npes = 10;

    nvshmemi_cuda_driver_version = 12040;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    struct cudaDeviceProp prop;
    prop.major = 9;

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetHandleByPciBusId(_, _)).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetGpuFabricInfoV(_, _)).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGetAttribute(_, _, _)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(prop), Return(cudaSuccess)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_NvmlDeviceProps_PciBusIdFail) {
    // Arrange
    state.mype = 0;
    state.npes = 1;

    nvshmemi_cuda_driver_version = 12040;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    struct cudaDeviceProp prop;
    prop.major = 9;

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetHandleByPciBusId(_, _))
        .WillOnce(Return(NVML_ERROR_INSUFFICIENT_SIZE));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(prop), Return(cudaSuccess)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_NvmlDeviceProps_InfoVNull) {
    // Arrange
    state.mype = 0;
    state.npes = 10;

    nvshmemi_cuda_driver_version = 12040;

    struct nvml_function_table new_nvml_ftable = {
        .nvmlInit_v2 = &nvmlInit,
        .nvmlShutdown = &nvmlShutdown,
        .nvmlDeviceGetHandleByPciBusId = &nvmlDeviceGetHandleByPciBusId,
        .nvmlDeviceGetP2PStatus = NULL,
        .nvmlDeviceGetGpuFabricInfoV = NULL};
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();
    struct cudaDeviceProp prop;
    prop.major = 9;

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetHandleByPciBusId(_, _))
        .WillOnce(Return(NVML_ERROR_INSUFFICIENT_SIZE));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(prop), Return(cudaSuccess)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_NvmlDeviceProps_InfoVFail) {
    // Arrange
    state.mype = 0;
    state.npes = 10;

    nvshmemi_cuda_driver_version = 12040;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    struct cudaDeviceProp prop;
    prop.major = 9;

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetHandleByPciBusId(_, _)).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetGpuFabricInfoV(_, _))
        .WillOnce(Return(NVML_ERROR_INSUFFICIENT_SIZE));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(CUDA_SUCCESS), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(prop), Return(cudaSuccess)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act

    // Assert
    EXPECT_EQ(1, transport->is_initialized());

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

TEST(constructor_destructor_test, mem_transport_NvmlDeviceProps_MNNVLSupport) {
    state.mype = 0;
    state.npes = 1;

    nvshmemi_cuda_driver_version = 12040;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    struct cudaDeviceProp prop;
    prop.major = 9;

    nvmlGpuFabricState_t st = 4;
    nvmlGpuFabricInfoV_t nvgfi;

    nvgfi.state = st;
    nvgfi.clusterUuid[0] = 'A';

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetHandleByPciBusId(_, _)).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlDeviceGetGpuFabricInfoV(_, _))
        .WillOnce(DoAll(SetArgPointee<1>(nvgfi), Return(NVML_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGetAttribute(_, _, _))
        .WillOnce(DoAll(SetArgPointee<0>(1), Assign(&state.npes, 0), Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(prop), Return(cudaSuccess)));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillOnce(Return(0)).WillOnce(Return(0));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);
    int result = transport->is_initialized();

    // Act

    // Assert
    EXPECT_EQ(result, 1);

    delete transport;
    string abcd = "abcd";
    deleteGlobalPtrs(abcd);
}

TEST(create_proc_map_tests, successful) {
    // Arrange
    state.mype = 0;
    state.npes = 100;

    nvshmemi_cuda_driver_version = 12039;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _)).WillOnce(DoAll(Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _)).WillOnce(Return(cudaSuccess));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(NVSHMEMX_SUCCESS));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act
    int proc_map_status = transport->create_proc_map(*mnvsh);

    // Assert
    EXPECT_EQ(proc_map_status, 0);

    delete transport;
    string abcd = "abcd";
    deleteGlobalPtrs(abcd);
}

TEST(create_proc_map_tests, allgatherFailure) {
    // Arrange
    state.mype = 0;
    state.npes = 100;

    nvshmemi_cuda_driver_version = 12039;

    struct nvml_function_table new_nvml_ftable = initialize_NVML_func_table();
    struct nvshmemi_cuda_fn_table new_cuda_fn_table = initialize_CUDA_func_table();
    bootstrap_handle_t new_bootstrap_handle = initialize_bootstrap_handle();

    nvshmemi_mem_p2p_transport *transport;
    nvshmemi_cuda_syms = &new_cuda_fn_table;

    mpt = new Mock_nvshmemi_mem_p2p_transport();
    mnft = new Mock_nvml_function_table();
    mbh = new Mock_bootstrap_handle();
    mnvcft = new Mock_nvshmemi_cuda_fn_table();

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);

    EXPECT_CALL(*mpt, nvshmemi_nvml_ftable_init(_, _))
        .WillOnce(DoAll(SetArgPointee<0>(dynamic_cast<nvml_function_table &>(new_nvml_ftable)),
                        Assign(&nvshmemi_boot_handle, new_bootstrap_handle), Return(0)));

    EXPECT_CALL(*mnft, nvmlInit()).WillOnce(Return(NVML_SUCCESS));

    EXPECT_CALL(*mnft, nvmlShutdown()).WillRepeatedly(Return(NVML_SUCCESS));

    EXPECT_CALL(*mpt, cudaGetDeviceCount(_))
        .WillOnce(DoAll(SetArgPointee<0>(1), Return(cudaSuccess)));

    EXPECT_CALL(*mnvcft, pfn_cuCtxGetDevice(_)).WillOnce(Return(CUDA_SUCCESS));

    EXPECT_CALL(*mnvcft, pfn_cuDeviceGet(_, _)).WillOnce(DoAll(Return(CUDA_SUCCESS)));

    EXPECT_CALL(*mpt, cudaGetDeviceProperties(_, _)).WillOnce(Return(cudaSuccess));

    EXPECT_CALL(*mbh, allgather(_, _, _, _)).WillRepeatedly(Return(NVSHMEMX_ERROR_INVALID_VALUE));

    transport = nvshmemi_mem_p2p_transport::get_instance(state.mype, state.npes);

    // Act
    int proc_map_status = transport->create_proc_map(*mnvsh);

    // Assert
    EXPECT_EQ(proc_map_status, 7);

    string abcd = "abcd";
    delete transport;
    deleteGlobalPtrs(abcd);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
