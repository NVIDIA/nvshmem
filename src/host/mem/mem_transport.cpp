/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>                                                          // for CUdevice
#include <cuda_runtime.h>                                                  // for cudaG...
#include <driver_types.h>                                                  // for cudaD...
#include <ext/alloc_traits.h>                                              // for __all...
#include <nvml.h>                                                          // for NVML_...
#include <stdint.h>                                                        // for uint64_t
#include <stdio.h>                                                         // for NULL
#include <stdlib.h>                                                        // for malloc
#include <string.h>                                                        // for memcmp
#include <unistd.h>                                                        // for pid_t
#include <map>                                                             // for map
#include <memory>                                                          // for alloc...
#include <vector>                                                          // for vector
#include "internal/host_transport/cudawrap.h"                              // for CUPFN
#include "non_abi/nvshmemx_error.h"                                        // for NVSHM...
#include "internal/host/debug.h"                                           // for INFO
#include "internal/host/nvshmem_internal.h"                                // for nvshm...
#include "internal/host/nvmlwrap.h"                                        // for nvmlG...
#include "internal/host/nvshmemi_handle_table.hpp"                         // for nvshm...
#include "internal/host/nvshmemi_mem_transport.hpp"                        // for nvshm...
#include "internal/host/nvshmemi_transport_view.hpp"                       // for nvshm...
#include "internal/host/nvshmemi_types.h"                                  // for nvshm...
#include "internal/host/util.h"                                            // for NVSHM...
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshm...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for nvshm...
#include "internal/host_transport/transport.h"                             // for nvshm...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHM...

namespace {

constexpr size_t nvshmemi_num_cuda_clique_types = 4;
static_assert(nvshmemi_num_cuda_clique_types <= 8,
              "valid_types must have one bit for each CUDA clique type");

struct nvshmemi_cuda_clique_record {
    CUuuid cluster_uuid{};  // Fabric cluster containing this PE's CUDA device.
    // Clique ID indexed by CUcliqueType; an entry is meaningful only when its valid_types bit is
    // set.
    unsigned int clique_ids[nvshmemi_num_cuda_clique_types]{};
    uint8_t valid_types = 0;  // Bitmask of clique types returned by cuDeviceGetCliqueInfo.
    uint8_t query_valid = 0;  // Whether all CUDA clique queries completed with consistent results.
};

const char *nvshmemi_cuda_clique_type_name(size_t type) {
    switch (static_cast<CUcliqueType>(type)) {
        case CU_CLIQUE_TYPE_UNICAST_POINTER:
            return "unicast pointer";
        case CU_CLIQUE_TYPE_MULTICAST_POINTER:
            return "multicast pointer";
        case CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT:
            return "unicast logical endpoint";
        case CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT:
            return "multicast logical endpoint";
        default:
            return "unknown";
    }
}

bool nvshmemi_cuda_clique_records_match(const nvshmemi_cuda_clique_record &lhs,
                                        const nvshmemi_cuda_clique_record &rhs, size_t type) {
    const uint32_t type_bit = 1u << type;
    return lhs.query_valid && rhs.query_valid && (lhs.valid_types & type_bit) &&
           (rhs.valid_types & type_bit) &&
           memcmp(&lhs.cluster_uuid, &rhs.cluster_uuid, sizeof(CUuuid)) == 0 &&
           lhs.clique_ids[type] == rhs.clique_ids[type];
}

int nvshmemi_discover_cuda_cliques(
    CUdevice device, int mype, int npes,
    std::array<std::vector<uint8_t>, nvshmemi_num_cuda_clique_types> &connected_pes,
    nvshmemi_clique_discovery_mode &mode) {
    nvshmemi_cuda_clique_record local_record{};
    std::vector<nvshmemi_cuda_clique_record> peer_records(npes);
    mode = nvshmemi_clique_discovery_mode::LEGACY_NVML;

    if (!nvshmemi_options.DISABLE_MNNVL && nvshmemi_cuda_syms != nullptr &&
        CUPFN(nvshmemi_cuda_syms, cuDeviceGetFabricClusterUuid) != nullptr &&
        CUPFN(nvshmemi_cuda_syms, cuDeviceGetCliqueCount) != nullptr &&
        CUPFN(nvshmemi_cuda_syms, cuDeviceGetCliqueInfo) != nullptr) {
        size_t count = 0;
        CUresult cuda_status = CUPFN(
            nvshmemi_cuda_syms, cuDeviceGetFabricClusterUuid(&local_record.cluster_uuid, device));
        if (cuda_status == CUDA_SUCCESS) {
            cuda_status = CUPFN(nvshmemi_cuda_syms, cuDeviceGetCliqueCount(&count, device));
        }

        std::vector<CUcliqueInfo> clique_info;
        if (cuda_status == CUDA_SUCCESS) {
            try {
                clique_info.resize(count);
            } catch (const std::bad_alloc &) {
                cuda_status = CUDA_ERROR_OUT_OF_MEMORY;
            }
        }

        if (cuda_status == CUDA_SUCCESS) {
            size_t returned_count = count;
            cuda_status = CUPFN(nvshmemi_cuda_syms,
                                cuDeviceGetCliqueInfo(clique_info.data(), &returned_count, device));
            if (cuda_status == CUDA_SUCCESS && returned_count <= count) {
                bool record_valid = true;
                for (size_t i = 0; i < returned_count; i++) {
                    const int type = static_cast<int>(clique_info[i].type);
                    if (type < 0 || type >= static_cast<int>(nvshmemi_num_cuda_clique_types)) {
                        record_valid = false;
                        break;
                    }

                    const uint32_t type_bit = 1u << type;
                    if ((local_record.valid_types & type_bit) &&
                        local_record.clique_ids[type] != clique_info[i].id) {
                        record_valid = false;
                        break;
                    }
                    local_record.valid_types |= type_bit;
                    local_record.clique_ids[type] = clique_info[i].id;
                }
                local_record.query_valid = record_valid;
            }
        }
    }

    int status =
        nvshmemi_boot_handle.allgather(&local_record, peer_records.data(),
                                       sizeof(nvshmemi_cuda_clique_record), &nvshmemi_boot_handle);
    if (status != 0) return status;

    if (!std::all_of(
            peer_records.begin(), peer_records.end(),
            [](const nvshmemi_cuda_clique_record &record) { return record.query_valid != 0; })) {
        INFO(NVSHMEM_MEM,
             "CUDA fabric clique discovery is unavailable on at least one PE; using legacy NVML "
             "discovery");
        return 0;
    }

    mode = nvshmemi_clique_discovery_mode::CUDA_CLIQUE_API;
    const auto &my_record = peer_records.at(mype);
    for (size_t type = 0; type < nvshmemi_num_cuda_clique_types; type++) {
        for (int pe = 0; pe < npes; pe++) {
            connected_pes[type][pe] =
                nvshmemi_cuda_clique_records_match(my_record, peer_records[pe], type);
        }
        if (my_record.valid_types & (1u << type)) {
            INFO(NVSHMEM_MEM, "CUDA fabric clique: type=%s id=%u",
                 nvshmemi_cuda_clique_type_name(type), my_record.clique_ids[type]);
        }
    }
    INFO(NVSHMEM_MEM, "CUDA fabric clique discovery is active");
    return 0;
}

}  // namespace

// Static member variable definitions (only the ones not defined elsewhere)
void *nvshmemi_mem_p2p_transport::nvml_handle_ = nullptr;
struct nvml_function_table nvshmemi_mem_p2p_transport::nvml_ftable_;

/**
 * nvshmemi_mem_p2p_transport specific functions
 */

// Static wrapper function for atexit that matches the expected signature
void nvshmemi_mem_p2p_transport::nvshmemi_nvml_ftable_fini_wrapper(void) {
    nvshmemi_nvml_ftable_fini(&nvml_ftable_, &nvml_handle_);
}

void nvshmemi_mem_p2p_transport::print_mem_handle(nvshmem_mem_handle_t *handle, int mype) {
    char *hex = nvshmemu_hexdump(handle, sizeof(CUipcMemHandle));
    INFO(NVSHMEM_INIT, "[%d] cuIpcOpenMemHandle fromhandle 0x%s", mype, hex);
    NVSHMEMU_HOST_PTR_FREE(hex);
}

int nvshmemi_mem_p2p_transport::create_proc_map(int npes,
                                                const nvshmemi_transport_view &transports) {
    pid_t pid = getpid();
    int status = 0;
    if (proc_map_.size() > 0) {
        return 0;
    }
    std::vector<pid_t> peer_pids(npes);
    status = nvshmemi_boot_handle.allgather((void *)&pid, (void *)peer_pids.data(), sizeof(pid_t),
                                            &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of pids failed \n");

    NVSHMEMU_FOR_EACH(pe, npes) {
        NVSHMEMU_FOR_EACH_IF(j, transports.num_transports(),
                             transports.active_has_cap(j, pe, NVSHMEM_TRANSPORT_CAP_MAP),
                             { proc_map_[peer_pids[pe]] = pe; });
    }

    INFO(NVSHMEM_MEM, "I am connected to %lu p2p processes (including myself)", proc_map_.size());
out:
    return (status);
}

nvshmemi_mem_p2p_transport::nvshmemi_mem_p2p_transport(int mype, int npes) {
    int status = 0;
    int nvml_status = 0;
    int device_id = -1;
    int ndev;
    int nbytes = 0;
    CUdevice cudevice;
    CUdevice *cudev = NULL;
    char pcie_bdf[NVSHMEM_PCIE_BDF_BUFFER_LEN] = {0};
    bool *peer_error_status = NULL;

    errored_on_initialization_ =
        true; /* By default, p2p is not initialized, so some features may be disabled */

    nvshmemi_nvls_connected_pes_.resize(npes, 0);  // this is a bitmap
    nvshmemi_nvl_connected_pes_.resize(npes, 0);
    nvshmemi_handle_accessible_pes_.resize(npes, 0);
    for (auto &connected_pes : cuda_clique_connected_pes_) connected_pes.resize(npes, 0);
    cudaDeviceProp prop;
    int flag = false;
    nvmlDevice_t local_device;
    nvmlGpuFabricInfoV_t fabricInfo = {}, fabricInfo1 = {}, fabricInfo2 = {};
    nvmlPlatformInfo_t platformInfo = {}, platformInfo1 = {}, platformInfo2 = {};
    const unsigned char zero[NVML_GPU_FABRIC_UUID_LEN] = {0};
    nvmlGpuFabricInfoV_t *pe_fabricInfo = nullptr;
    std::vector<nvmlPlatformInfo_t> pe_platformInfo;
    fabricInfo.version = nvmlGpuFabricInfo_v2;
    fabricInfo1.version = nvmlGpuFabricInfo_v2;
    fabricInfo2.version = nvmlGpuFabricInfo_v2;

    /* start NVML Library */
    if (nvml_handle_ == nullptr) {
        nvml_status = nvshmemi_nvml_ftable_init(&nvml_ftable_, &nvml_handle_);
        if (nvml_status != NVML_SUCCESS) {
            status = NVSHMEMX_ERROR_INTERNAL;
            INFO(NVSHMEM_MEM, "Unable to open NVML. Some features will be disabled. %d",
                 nvml_status);
            goto out;
        }

        nvml_status = nvml_ftable_.nvmlInit();
        if (nvml_status != NVML_SUCCESS) {
            status = NVSHMEMX_ERROR_INTERNAL;
            INFO(NVSHMEM_MEM, "Unable to initialize NVML. Some features will be disabled. %d",
                 nvml_status);
            goto out;
        }
        atexit(nvshmemi_nvml_ftable_fini_wrapper);
    }

    /* Discover cudevice instance and device ID */
    status = cudaGetDeviceCount(&ndev);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaGetDeviceCount failed \n");

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&cudevice));
    NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                             "cuCtxGetDevice failed \n");

    cudev = (CUdevice *)std::malloc(sizeof(CUdevice) * ndev);
    NVSHMEMI_NULL_ERROR_JMP(cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "cudev array allocation failed \n");

    NVSHMEMU_FOR_EACH(i, ndev) {
        status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet(&cudev[i], i));
        NVSHMEMI_CU_NE_ERROR_JMP(nvshmemi_cuda_syms, status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL,
                                 out, "cuDeviceGet failed \n");
        if (cudev[i] == cudevice) {
            device_id = i;
            cudaDeviceProp prop;
            status = cudaGetDeviceProperties(&prop, i);
            NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaGetDeviceProperties failed \n");
            nbytes = snprintf(pcie_bdf, NVSHMEM_PCIE_BDF_BUFFER_LEN, "%x:%x:%x.0", prop.pciDomainID,
                              prop.pciBusID, prop.pciDeviceID);
            if (nbytes < 0 || nbytes > NVSHMEM_PCIE_BDF_BUFFER_LEN) {
                status = NVSHMEMX_ERROR_INTERNAL;
                NVSHMEMI_ERROR_JMP(nbytes, NVSHMEMX_ERROR_INTERNAL, out,
                                   "Unable to set device pcie bdf for our local device.\n");
            }
        }
    }

    status = nvshmemi_discover_cuda_cliques(cudevice, mype, npes, cuda_clique_connected_pes_,
                                            clique_discovery_mode_);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of CUDA fabric clique information failed\n");

    /* For the assigned device_id, discover nvmlDevice properties */
    cudaGetDeviceProperties(&prop, device_id);
    if (nvshmemi_cuda_driver_version >= 12040 && prop.major >= 9 &&
        !nvshmemi_options.DISABLE_MNNVL) {
        nvml_status = nvml_ftable_.nvmlDeviceGetHandleByPciBusId(pcie_bdf, &local_device);
        NVSHMEMI_CHECK_ERROR_JMP(nvml_status != NVML_SUCCESS, status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "nvmlDeviceGetHandleByPciBusId failed with NVML error %d\n",
                                 nvml_status);

        /* Some platforms with older driver may not support this API, so bypass MNNVL discovery */
        if (nvml_ftable_.nvmlDeviceGetGpuFabricInfoV == NULL) {
            INFO(NVSHMEM_INIT,
                 "nvmlDeviceGetGpuFabricInfoV not found. Detection of MNNVL environment will not "
                 "be attempted\n");
            status |= NVSHMEMX_SUCCESS;
            goto out;
        }

        nvml_status = nvml_ftable_.nvmlDeviceGetGpuFabricInfoV(local_device, &fabricInfo);
        NVSHMEMI_CHECK_ERROR_JMP(nvml_status != NVML_SUCCESS, status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "nvmlDeviceGetGpuFabricInfoV() failed... Detection of MNNVL "
                                 "environment will not be attempted. NVML error: %d",
                                 nvml_status);

        pe_fabricInfo = (nvmlGpuFabricInfoV_t *)std::malloc(sizeof(nvmlGpuFabricInfoV_t) * npes);
        NVSHMEMI_NULL_ERROR_JMP(pe_fabricInfo, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "pe_fabricInfo array allocation failed\n");

        if (nvshmemi_options.MNNVL_OVERRIDE_MC_CLIQUE_ID) {
            // Override the cliqueId to use rackIDs to determine NVLink domain
            if (nvml_ftable_.nvmlDeviceGetPlatformInfo == nullptr) {
                NVSHMEMI_ERROR_PRINT(
                    "nvmlDeviceGetPlatformInfo not found. Override of cliqueId will not be "
                    "attempted\n");
                status = NVSHMEMX_ERROR_INTERNAL;
                goto out;
            }

            pe_platformInfo.resize(npes);

            platformInfo.version = nvmlPlatformInfo_v2;
            nvml_status = nvml_ftable_.nvmlDeviceGetPlatformInfo(local_device, &platformInfo);
            NVSHMEMI_CHECK_ERROR_JMP(nvml_status != NVML_SUCCESS, status, NVSHMEMX_ERROR_INTERNAL,
                                     out, "nvmlDeviceGetPlatformInfo failed with NVML error %d\n",
                                     nvml_status);

            pe_platformInfo[mype] = platformInfo;

            status = nvshmemi_boot_handle.allgather(
                (void *)&platformInfo, (void *)pe_platformInfo.data(), sizeof(nvmlPlatformInfo_t),
                &nvshmemi_boot_handle);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "allgather of pe_platformInfo failed \n");
            platformInfo1 = pe_platformInfo[mype];
        }

        pe_fabricInfo[mype] = fabricInfo;
        status =
            nvshmemi_boot_handle.allgather((void *)&fabricInfo, (void *)pe_fabricInfo,
                                           sizeof(nvmlGpuFabricInfoV_t), &nvshmemi_boot_handle);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "allgather of pe_fabricInfo failed \n");

        nvshmemi_has_mnnvl_fabric_ = 1;
        CUPFN(nvshmemi_cuda_syms,
              cuDeviceGetAttribute(
                  &flag,
                  static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED),
                  device_id));
        if (!flag) nvshmemi_has_mnnvl_fabric_ = 0;

        fabricInfo1 = pe_fabricInfo[mype];
        if (fabricInfo1.state < NVML_GPU_FABRIC_STATE_COMPLETED ||
            memcmp(fabricInfo1.clusterUuid, zero, NVML_GPU_FABRIC_UUID_LEN) == 0)
            nvshmemi_has_mnnvl_fabric_ = 0;

        nvshmemi_mem_handle_type_ =
            (nvshmemi_has_mnnvl_fabric_ && flag)
                ? static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_FABRIC)
                : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        /* if OVERRIDE_CLIQUE_ID is enabled, then only PEs with same rackID will be considered
         * as nvls_connected_pes, others will not be considered.
         */
        for (int i = 0; i < npes && nvshmemi_has_mnnvl_fabric_; i++) {
            fabricInfo2 = pe_fabricInfo[i];
            if ((fabricInfo2.state == NVML_GPU_FABRIC_STATE_COMPLETED) &&
                (memcmp(fabricInfo1.clusterUuid, fabricInfo2.clusterUuid,
                        NVML_GPU_FABRIC_UUID_LEN) == 0) &&
                (fabricInfo1.cliqueId == fabricInfo2.cliqueId)) {
                // setup nvl_connected_pes initially to include all PEs
                // that are connected via NVL. If there are VA mapping restrictions,
                // then this will updated.
                nvshmemi_nvl_connected_pes_[i] = 1;

                nvshmemi_handle_accessible_pes_[i] = 1;

                if (nvshmemi_options.MNNVL_OVERRIDE_MC_CLIQUE_ID) {
                    // group PEs with same rackID in multicast domain (a subset of
                    // nvl_connected_pes)
                    platformInfo2 = pe_platformInfo[i];
                    if (memcmp(platformInfo1.chassisSerialNumber, platformInfo2.chassisSerialNumber,
                               sizeof(platformInfo1.chassisSerialNumber)) == 0) {
                        nvshmemi_nvls_connected_pes_[i] = 1;
                    }
                } else {
                    // track nvl_connected_pes
                    nvshmemi_nvls_connected_pes_[i] = 1;
                }
            }
        }

        if (nvshmemi_has_mnnvl_fabric_ && has_cuda_clique_info()) {
            nvshmemi_nvl_connected_pes_ = cuda_clique_connected_pes_.at(
                static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_POINTER));
            nvshmemi_handle_accessible_pes_ = cuda_clique_connected_pes_.at(
                static_cast<size_t>(CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT));
        }

        if (nvshmemi_has_mnnvl_fabric_) {
            INFO(NVSHMEM_MEM, "Multi-node NVLink is supported and enabled on this platform");
        }
    }

    if (nvshmemi_options.CUMEM_HANDLE_TYPE_provided) {
        if (strcmp_case_insensitive(nvshmemi_options.CUMEM_HANDLE_TYPE, "FABRIC") == 0)
            nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_FABRIC;
        else if (strcmp_case_insensitive(nvshmemi_options.CUMEM_HANDLE_TYPE, "ANY") == 0)
            nvshmemi_mem_handle_type_ =
                (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC |
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        else
            nvshmemi_mem_handle_type_ = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }

    if (nvshmemi_mem_handle_type_ == CU_MEM_HANDLE_TYPE_FABRIC) {
        INFO(NVSHMEM_MEM, "Symmetric Memory Heap Handle Type: Fabric Handle\n");
    } else if (nvshmemi_mem_handle_type_ ==
               (CU_MEM_HANDLE_TYPE_FABRIC | CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
        INFO(NVSHMEM_MEM,
             "Symmetric Memory Heap Handle Type: ANY "
             "(effective: Fabric when MNNVL active, POSIX otherwise)\n");
    } else {
        INFO(NVSHMEM_MEM, "Symmetric Memory Heap Handle Type: POSIX File Descriptor\n");
    }
out:
    if (status == 0) errored_on_initialization_ = false;

    NVSHMEMU_HOST_PTR_FREE(cudev);
    if ((status || nvml_status) && nvml_ftable_.nvmlShutdown != NULL) {
        nvml_status = nvml_ftable_.nvmlShutdown();
        if (nvml_status != NVML_SUCCESS) {
            INFO(NVSHMEM_MEM, "Unable to stop NVML library in NVSHMEM. NVML error: %d",
                 nvml_status);
        }
        nvshmemi_nvml_ftable_fini(&nvml_ftable_, &nvml_handle_);
        if (status)
            INFO(NVSHMEM_MEM,
                 "Unable to intialize mem p2p transport (likely non-fatal). status = %d\n", status);
    }

    NVSHMEMU_HOST_PTR_FREE(pe_fabricInfo);
    /* Successful initialization on my PE and peer PEs
       This is to avoid a case where some PEs on some node are P2P reachable and some PEs on some
       node are not, causing asymmetry
    */
    peer_error_status = (bool *)std::calloc(npes, sizeof(*peer_error_status));
    nvshmemi_boot_handle.allgather((void *)(&errored_on_initialization_), peer_error_status,
                                   sizeof(bool), &nvshmemi_boot_handle);
    NVSHMEMU_FOR_EACH(i, npes) {
        if (static_cast<int>(i) != mype && peer_error_status[i] != errored_on_initialization_) {
            errored_on_initialization_ = true;
            break;
        }
    }

    NVSHMEMU_HOST_PTR_FREE(peer_error_status);
}

int nvshmemi_mem_p2p_transport::get_num_p2p_connected_pes(int npes_node) {
    return std::max(npes_node,
                    static_cast<int>(std::count(nvshmemi_nvl_connected_pes_.begin(),
                                                nvshmemi_nvl_connected_pes_.end(), uint8_t{1})));
}

nvshmemi_mem_p2p_transport::~nvshmemi_mem_p2p_transport() {
    proc_map_.clear();
    if (p2p_objref_ != nullptr) p2p_objref_ = nullptr;
}

/*
 * nvshmemi_mem_remote_transport specific functions
 */
int nvshmemi_mem_remote_transport::gather_mem_handles(const nvshmemi_transport_view &transports,
                                                      nvshmem_mem_handle_t *handle_data,
                                                      uint64_t heap_offset, size_t size) {
    int status = 0;

    NVSHMEMU_FOR_EACH(i, transports.num_transports()) {
        nvshmem_transport_t tcurr = transports.transport(i);
        if (transports.is_active(i) && transports.supports_add_device_remote_mem(i)) {
            status = tcurr->host_ops.add_device_remote_mem_handles(
                tcurr, transports.num_transports(), handle_data, heap_offset, size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "add_device_remote_mem_handles failed \n");

            status = nvshmemi_update_device_state();
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "nvshmemi_update_device_state() failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_mem_remote_transport::register_mem_handle(nvshmem_mem_handle_t *local_handles,
                                                       int transport_idx, void *buf, size_t size,
                                                       const nvshmemi_transport_view &transports) {
    if (!transports.supports_get_mem(transport_idx)) return 0;
    nvshmem_transport_t current = transports.transport(transport_idx);
    return current->host_ops.get_mem_handle((nvshmem_mem_handle_t *)(local_handles + transport_idx),
                                            buf, size, current, false);
}

int nvshmemi_mem_remote_transport::release_mem_handles(nvshmem_mem_handle_t *handles,
                                                       const nvshmemi_transport_view &transports) {
    int first_status = NVSHMEMX_SUCCESS;
    NVSHMEMU_FOR_EACH_IF(i, transports.num_transports(),
                         transports.is_active(i) && transports.supports_release_mem(i), {
                             if (!is_mem_handle_null(&handles[i])) {
                                 int status = transports.transport(i)->host_ops.release_mem_handle(
                                     &handles[i], transports.transport(i));
                                 if (status == NVSHMEMX_SUCCESS) {
                                     memset(&handles[i], 0, sizeof(handles[i]));
                                 } else {
                                     if (first_status == NVSHMEMX_SUCCESS) first_status = status;
                                     NVSHMEMI_ERROR_PRINT(
                                         "transport %llu failed to release memory handle "
                                         "(status=%d)",
                                         static_cast<unsigned long long>(i), status);
                                 }
                             }
                         });
    return first_status;
}
