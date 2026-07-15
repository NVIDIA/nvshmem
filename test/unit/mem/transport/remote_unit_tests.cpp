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

nvshmemi_mem_remote_transport *nvshmemi_mem_remote_transport::remote_objref_ = nullptr;
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
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle);
int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);
int nvshmemi_update_device_state();

nvmlReturn_t nvmlInit();
nvmlReturn_t nvmlShutdown();
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device);
nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info);

CUresult pfn_cuDeviceGet(CUdevice *device, int ordinal);
CUresult pfn_cuCtxGetDevice(CUdevice *device);
CUresult pfn_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);

int allgather(const void *sendbuf, void *recvbuf, int bytes, struct bootstrap_handle *handle);

int add_device_remote_mem_handles(struct nvshmem_transport *transport, int transport_stride,
                                  nvshmem_mem_handle_t *mem_handles, uint64_t heap_offset,
                                  size_t size);
int release_mem_handle(nvshmem_mem_handle_t *mem_handle, struct nvshmem_transport *transport);

nvshmemi_symmetric_heap_vidmem_dynamic_vmm::~nvshmemi_symmetric_heap_vidmem_dynamic_vmm() {}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::reserve_heap() { return 0; }
int nvshmemi_symmetric_heap::map_heap_range_by_size(void *buf, size_t size) { return 1; }
int nvshmemi_symmetric_heap::update_heap_handle_cache(void *buf, size_t size) { return 1; }
int nvshmemi_symmetric_heap::register_heap_memory(nvshmem_mem_handle_t *input, void *buffer,
                                                  size_t size) {
    return 1;
}
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::register_heap_memory(
    nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t size) {
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
int nvshmemi_symmetric_heap_vidmem_dynamic_vmm::setup_mspace() { return 1; }
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

class Mock_nvshmemi_mem_remote_transport {
   public:
    MOCK_METHOD2(nvshmemu_hexdump, char *(void *ptr, size_t len));
    MOCK_METHOD0(getpid, __pid_t());
    MOCK_METHOD1(malloc, void *(size_t __size));                  // no need to mock stdlib apis
    MOCK_METHOD2(calloc, void *(size_t __nmemb, size_t __size));  // no need to mock stdlib apis
    MOCK_METHOD2(nvshmemi_nvml_ftable_init,
                 int(struct nvml_function_table *nvml_ftable, void **nvml_handle));
    MOCK_METHOD2(nvshmemi_nvml_ftable_fini,
                 void(struct nvml_function_table *nvml_ftable, void **nvml_handle));
    MOCK_METHOD1(cudaGetDeviceCount, cudaError_t(int *count));
    MOCK_METHOD2(cudaGetDeviceProperties, cudaError_t(cudaDeviceProp *prop, int device));
    MOCK_METHOD3(memcmp, int(const void *__s1, const void *__s2, size_t __n));
    MOCK_METHOD3(memset, void *(void *__s, int __c, size_t __n));  // no need to mock stdlib apis
    MOCK_METHOD0(nvshmemi_update_device_state, int());
    MOCK_METHOD1(is_mem_handle_null, int(nvshmem_mem_handle_t *handle));
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
    void set_handle(const std::vector<std::vector<nvshmem_mem_handle>> &handles) {
        this->remote_handles_ = handles;
    }
};

class Mock_nvshmemi_cuda_fn_table {
   public:
    MOCK_METHOD1(pfn_cuCtxGetDevice, CUresult(CUdevice *device));
    MOCK_METHOD2(pfn_cuDeviceGet, CUresult(CUdevice *device, int ordinal));
    MOCK_METHOD3(pfn_cuDeviceGetAttribute,
                 CUresult(int *pi, CUdevice_attribute attrib, CUdevice dev));
};

class Mock_nvshmem_transport_host_ops {
   public:
    MOCK_METHOD5(add_device_remote_mem_handles,
                 int(struct nvshmem_transport *transport, int transport_stride,
                     nvshmem_mem_handle_t *mem_handles, uint64_t heap_offset, size_t size));
    MOCK_METHOD2(release_mem_handle,
                 int(nvshmem_mem_handle_t *mem_handle, struct nvshmem_transport *transport));
};

Mock_nvshmemi_mem_remote_transport *mrt;
Mock_nvshmem_transport_host_ops *mntho;

void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...) {
    if (nvshmem_debug_level <= NVSHMEM_LOG_NONE) {
        return;
    }
}

/*-------------------------------------------------Helper
 * functions--------------------------------------------------------*/

nvshmem_transport_host_ops initialize_nvshmem_transport_host_ops() {
    struct nvshmem_transport_host_ops new_nvshmem_transport_host_ops = {
        .can_reach_peer = NULL,
        .connect_endpoints = NULL,
        .get_mem_handle = NULL,
        .release_mem_handle = &release_mem_handle,
        .finalize = NULL,
        .show_info = NULL,
        .progress = NULL,
        .rma = NULL,
        .amo = NULL,
        .fence = NULL,
        .quiet = NULL,
        .put_signal = NULL,
        .enforce_cst = NULL,
        .enforce_cst_at_target = NULL,
        .add_device_remote_mem_handles = &add_device_remote_mem_handles};

    return new_nvshmem_transport_host_ops;
}

void deleteGlobalPtrs(string &abcd) {
    if (abcd[0] == 'a') {
        delete mrt;
    }
    if (abcd[1] == 'b') {
        delete mntho;
    }
}

/*-------Mock_nvshmemi_mem_p2p_transport functions that provide a level of abstraction for the
 * tester functions--------------*/

char *nvshmemu_hexdump(void *ptr, size_t len) { return mrt->nvshmemu_hexdump(ptr, len); }
cudaError_t cudaGetDeviceCount(int *count) { return mrt->cudaGetDeviceCount(count); }
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    return mrt->nvshmemi_nvml_ftable_fini(nvml_ftable, nvml_handle);
}
int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle) {
    return mrt->nvshmemi_nvml_ftable_init(nvml_ftable, nvml_handle);
}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return mrt->cudaGetDeviceProperties(prop, device);
}
int nvshmemi_update_device_state() { return mrt->nvshmemi_update_device_state(); }
int is_mem_handle_null(nvshmem_mem_handle_t *handle) { return mrt->is_mem_handle_null(handle); }

/*-------Mock_nvshmem_transport_host_ops functions that provide a level of abstraction for the
 * tester functions--------------*/
int add_device_remote_mem_handles(struct nvshmem_transport *transport, int transport_stride,
                                  nvshmem_mem_handle_t *mem_handles, uint64_t heap_offset,
                                  size_t size) {
    return mntho->add_device_remote_mem_handles(transport, transport_stride, mem_handles,
                                                heap_offset, size);
}
int release_mem_handle(nvshmem_mem_handle_t *mem_handle, struct nvshmem_transport *transport) {
    return mntho->release_mem_handle(mem_handle, transport);
}

TEST(gather_mem_handles_test, successful_noForEach) {
    nvshmem_transport_t new_tcurr;
    nvshmem_mem_handle_t *handles;

    state.num_initialized_transports = 0;
    state.transport_bitmap = 0;

    size_t size = 10;
    uint64_t offset = 10;
    bool ext_alloc = false;

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    int status = transport->gather_mem_handles(*mnvsh, offset, size, ext_alloc);

    EXPECT_EQ(status, 0);
}

TEST(gather_mem_handles_test, successful_forEach_noIf) {
    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t *handles;

    state.num_initialized_transports = 1;
    state.transport_bitmap = 0;

    state.transports = (nvshmem_transport_t *)malloc(1 * sizeof(nvshmem_transport_t *));

    state.transports[0] = &new_tcurr;
    size_t size = 10;
    uint64_t offset = 10;
    bool ext_alloc = false;
    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    int status = transport->gather_mem_handles(*mnvsh, offset, size, ext_alloc);

    EXPECT_EQ(status, 0);
}

TEST(gather_mem_handles_test, successful_forEach_if) {
    mrt = new Mock_nvshmemi_mem_remote_transport();
    mntho = new Mock_nvshmem_transport_host_ops();
    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {};
    new_tcurr.host_ops.add_device_remote_mem_handles = &add_device_remote_mem_handles;

    std::vector<std::vector<nvshmem_mem_handle>> handle_;
    std::vector<nvshmem_mem_handle> innerVector;
    innerVector.push_back(handles);
    handle_.push_back(innerVector);

    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));

    state.transports[0] = &new_tcurr;
    size_t size = 10;
    uint64_t offset = 10;
    bool ext_alloc = false;
    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    mnvsh->set_handle(handle_);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    EXPECT_CALL(*mntho, add_device_remote_mem_handles(_, _, _, _, _)).WillOnce(Return(0));

    EXPECT_CALL(*mrt, nvshmemi_update_device_state()).WillOnce(Return(0));

    int status = transport->gather_mem_handles(*mnvsh, offset, size, ext_alloc);
    EXPECT_EQ(status, 0);

    delete mrt;
    delete mntho;
}

TEST(gather_mem_handles_test, fail_forEach_if_first) {
    mntho = new Mock_nvshmem_transport_host_ops();
    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {};
    new_tcurr.host_ops.add_device_remote_mem_handles = &add_device_remote_mem_handles;

    std::vector<std::vector<nvshmem_mem_handle>> handle_;
    std::vector<nvshmem_mem_handle> innerVector;
    innerVector.push_back(handles);
    handle_.push_back(innerVector);

    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));

    state.transports[0] = &new_tcurr;
    size_t size = 10;
    uint64_t offset = 10;
    bool ext_alloc = false;
    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    mnvsh->set_handle(handle_);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    EXPECT_CALL(*mntho, add_device_remote_mem_handles(_, _, _, _, _))
        .WillOnce(Return(NVSHMEMX_ERROR_INTERNAL));

    int status = transport->gather_mem_handles(*mnvsh, offset, size, ext_alloc);
    EXPECT_EQ(status, 7);

    delete mntho;
}

TEST(gather_mem_handles_test, fail_forEach_if_second) {
    mrt = new Mock_nvshmemi_mem_remote_transport();
    mntho = new Mock_nvshmem_transport_host_ops();
    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {};
    new_tcurr.host_ops.add_device_remote_mem_handles = &add_device_remote_mem_handles;

    std::vector<std::vector<nvshmem_mem_handle>> handle_;
    std::vector<nvshmem_mem_handle> innerVector;
    innerVector.push_back(handles);
    handle_.push_back(innerVector);

    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    state.transports = (struct nvshmem_transport **)calloc(1, sizeof(struct nvshmem_transport *));

    state.transports[0] = &new_tcurr;
    size_t size = 10;
    uint64_t offset = 10;
    bool ext_alloc = false;
    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    mnvsh->set_handle(handle_);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    EXPECT_CALL(*mntho, add_device_remote_mem_handles(_, _, _, _, _)).WillOnce(Return(0));

    EXPECT_CALL(*mrt, nvshmemi_update_device_state()).WillOnce(Return(NVSHMEMX_ERROR_INTERNAL));

    int status = transport->gather_mem_handles(*mnvsh, offset, size, ext_alloc);
    EXPECT_EQ(status, 7);

    delete mrt;
    delete mntho;
}

TEST(release_mem_handle_test, successful_noPaths) {
    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {};

    state.transports[0] = &new_tcurr;

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    int status = transport->release_mem_handles(&handles, *mnvsh);

    EXPECT_EQ(0, status);
}

TEST(release_mem_handle_test, successful_for_noIf) {
    mntho = new Mock_nvshmem_transport_host_ops();

    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {};
    struct nvshmem_transport_host_ops new_host_ops = initialize_nvshmem_transport_host_ops();

    new_tcurr.host_ops = new_host_ops;
    state.transports[0] = &new_tcurr;

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    int status = transport->release_mem_handles(&handles, *mnvsh);

    EXPECT_EQ(0, status);
    delete mntho;
}

TEST(release_mem_handle_test, failure_for_if) {
    mntho = new Mock_nvshmem_transport_host_ops();

    state.num_initialized_transports = 1;
    state.transport_bitmap = 1;

    struct nvshmem_transport new_tcurr = {};
    nvshmem_mem_handle_t handles = {0};

    handles.reserved[0] = 1;

    struct nvshmem_transport_host_ops new_host_ops = initialize_nvshmem_transport_host_ops();

    new_tcurr.host_ops = new_host_ops;
    state.transports[0] = &new_tcurr;

    Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm *mnvsh =
        new Mock_nvshmemi_symmetric_heap_vidmem_dynamic_vmm(&state);
    nvshmemi_mem_remote_transport *transport = nvshmemi_mem_remote_transport::get_instance();

    EXPECT_CALL(*mntho, release_mem_handle(_, _)).WillOnce(Return(7));
    int status = transport->release_mem_handles(&handles, *mnvsh);

    EXPECT_EQ(7, status);
    delete mntho;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
