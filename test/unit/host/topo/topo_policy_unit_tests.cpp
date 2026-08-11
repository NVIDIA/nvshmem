/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "bootstrap_host_transport/env_defs_internal.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host/debug.h"
#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/util.h"
#include "internal/host_transport/cudawrap.h"
#include "internal/host_transport/transport.h"
#include "topo.h"

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kLocalHostHash = 0x1234;

nvshmemi_state_t test_state;
nvshmemi_cuda_fn_table test_cuda_syms;
std::vector<std::string> cuda_device_bus_ids;
std::vector<std::string> allgather_bus_ids;
std::string test_pci_bus_root;
std::string test_nvidia_root;

std::string remap_sysfs_path(const char *path) {
    const std::string input(path);
    const std::string pci_bus_prefix = "/sys/class/pci_bus";
    const std::string nvidia_prefix = "/sys/bus/pci/drivers/nvidia";

    if (input == nvidia_prefix) {
        return test_nvidia_root;
    }

    if (input.compare(0, pci_bus_prefix.size(), pci_bus_prefix) == 0) {
        return test_pci_bus_root + input.substr(pci_bus_prefix.size());
    }

    return input;
}

std::string make_temp_dir() {
    char tmpl[] = "/tmp/nvshmem_topo_unit.XXXXXX";
    char *dir = mkdtemp(tmpl);
    EXPECT_NE(dir, nullptr);
    return dir ? std::string(dir) : std::string("/tmp");
}

std::string pci_bus_for_bdf(const std::string &bdf) { return bdf.substr(0, 7); }

void write_numa_node(const fs::path &path, int numa_node) {
    std::ofstream file(path / "numa_node");
    file << numa_node << "\n";
}

nvshmem_debug_log_level debug_level_from_env() {
    const char *debug = getenv("NVSHMEM_DEBUG");
    if (!debug) {
        return NVSHMEM_LOG_NONE;
    }
    if (strcmp_case_insensitive(debug, "WARN") == 0) {
        return NVSHMEM_LOG_WARN;
    }
    if (strcmp_case_insensitive(debug, "INFO") == 0) {
        return NVSHMEM_LOG_INFO;
    }
    if (strcmp_case_insensitive(debug, "TRACE") == 0) {
        return NVSHMEM_LOG_TRACE;
    }
    return NVSHMEM_LOG_INFO;
}

bool debug_subsys_enabled(unsigned long flags) {
    const char *debug_subsys = getenv("NVSHMEM_DEBUG_SUBSYS");
    if (!debug_subsys || debug_subsys[0] == '\0') {
        return (flags & NVSHMEM_INIT) != 0;
    }

    std::string subsys_list(debug_subsys);
    size_t start = 0;
    while (start <= subsys_list.size()) {
        size_t end = subsys_list.find(',', start);
        std::string subsys =
            subsys_list.substr(start, end == std::string::npos ? std::string::npos : end - start);

        if (strcmp_case_insensitive(subsys.c_str(), "ALL") == 0) {
            return true;
        }
        if ((flags & NVSHMEM_INIT) && strcmp_case_insensitive(subsys.c_str(), "INIT") == 0) {
            return true;
        }
        if ((flags & NVSHMEM_TOPO) && strcmp_case_insensitive(subsys.c_str(), "TOPO") == 0) {
            return true;
        }

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return false;
}

class TopoPolicyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        root = make_temp_dir();
        pci_bus_root = fs::path(root) / "pci_bus";
        nvidia_root = fs::path(root) / "nvidia";
        fs::create_directories(pci_bus_root);
        fs::create_directories(nvidia_root);
        test_pci_bus_root = pci_bus_root.string();
        test_nvidia_root = nvidia_root.string();

        std::memset(&test_state, 0, sizeof(test_state));
        std::memset(&test_cuda_syms, 0, sizeof(test_cuda_syms));
        std::memset(&nvshmemi_options, 0, sizeof(nvshmemi_options));
        nvshmemi_state = &test_state;
        nvshmemi_cuda_syms = &test_cuda_syms;
        nvshmemi_boot_handle.allgather = mock_allgather;
        test_cuda_syms.pfn_cuCtxGetDevice = mock_cu_ctx_get_device;
        nvshmemi_options.NETDEVS_POLICY = "AUTO";
        nvshmemi_options.ENABLE_NIC_PE_MAPPING = false;
        cuda_device_bus_ids.clear();
        allgather_bus_ids.clear();
    }

    void TearDown() override {
        fs::remove_all(root);
        test_pci_bus_root.clear();
        test_nvidia_root.clear();
        pe_info.clear();
        device_paths.clear();
        device_path_ptrs.clear();
    }

    void set_state(int npes, int npes_node, int mype, int device_id) {
        pe_info.assign(npes, {});
        for (int i = 0; i < npes; i++) {
            pe_info[i].pe = i;
            pe_info[i].hostHash = kLocalHostHash;
        }

        test_state.npes = npes;
        test_state.npes_node = npes_node;
        test_state.mype = mype;
        test_state.mype_node = mype;
        test_state.device_id = device_id;
        test_state.pe_info = pe_info.data();
        test_state.are_nics_ll128_compliant = true;
    }

    std::string create_pci_device(const std::string &bdf, int numa_node = 0) {
        fs::path bus_device = pci_bus_root / pci_bus_for_bdf(bdf) / "device";
        fs::path pci_device = bus_device / bdf;
        fs::create_directories(pci_device);
        write_numa_node(pci_device, numa_node);
        return fs::canonical(pci_device).string();
    }

    void add_sysfs_gpu(const std::string &bdf) {
        fs::create_directories(nvidia_root / bdf);
        (void)create_pci_device(bdf);
    }

    nvshmem_transport make_transport(const std::vector<std::string> &dev_bdfs,
                                     int fixed_numa_node = -1) {
        device_paths.clear();
        device_path_ptrs.clear();
        for (size_t i = 0; i < dev_bdfs.size(); i++) {
            int numa_node = fixed_numa_node >= 0 ? fixed_numa_node : static_cast<int>(i);
            device_paths.push_back(create_pci_device(dev_bdfs[i], numa_node));
        }
        for (std::string &path : device_paths) {
            device_path_ptrs.push_back(const_cast<char *>(path.c_str()));
        }

        nvshmem_transport transport = {};
        transport.n_devices = static_cast<int>(device_path_ptrs.size());
        transport.device_pci_paths = device_path_ptrs.data();
        return transport;
    }

    int select_devices(nvshmem_transport *transport, int max_devices, std::vector<int> *selected) {
        selected->assign(max_devices, -1);
        return nvshmemi_get_devices_by_distance(selected->data(), max_devices, transport);
    }

    static CUresult CUDAAPI mock_cu_ctx_get_device(CUdevice *device) {
        *device = test_state.device_id;
        return CUDA_SUCCESS;
    }

    static int mock_allgather(const void * /*sendbuf*/, void *recvbuf, int bytes,
                              bootstrap_handle * /*handle*/) {
        char *recv = static_cast<char *>(recvbuf);
        for (size_t i = 0; i < allgather_bus_ids.size(); i++) {
            std::memset(recv + i * bytes, 0, bytes);
            std::strncpy(recv + i * bytes, allgather_bus_ids[i].c_str(), bytes - 1);
        }
        return 0;
    }

    std::string root;
    fs::path pci_bus_root;
    fs::path nvidia_root;
    std::vector<nvshmem_transport_pe_info_t> pe_info;
    std::vector<std::string> device_paths;
    std::vector<char *> device_path_ptrs;
};

}  // namespace

nvshmemi_state_t *nvshmemi_state = nullptr;
bootstrap_handle_t nvshmemi_boot_handle = {};
nvshmemi_device_host_state_t nvshmemi_device_state = {};
nvshmemi_team_t **nvshmemi_team_pool = nullptr;
nvshmemi_options_s nvshmemi_options = {};
nvshmemi_cuda_fn_table *nvshmemi_cuda_syms = nullptr;

int nvshmem_debug_level = 0;
uint64_t nvshmem_debug_mask = 0;
pthread_mutex_t nvshmem_debug_output_lock = PTHREAD_MUTEX_INITIALIZER;
FILE *nvshmem_debug_file = nullptr;

void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags, const char *filefunc,
                       int line, const char *fmt, ...) {
    if (level > debug_level_from_env() || !debug_subsys_enabled(flags)) {
        return;
    }

    FILE *out = nvshmem_debug_file ? nvshmem_debug_file : stderr;
    fprintf(out, "%s:%d: ", filefunc, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
}

uint64_t nvshmemu_getHostHash() { return kLocalHostHash; }

extern "C" DIR *nvshmemi_test_opendir(const char *path) {
    std::string mapped_path = remap_sysfs_path(path);
    return opendir(mapped_path.c_str());
}

extern "C" char *nvshmemi_test_realpath(const char *path, char *resolved_path) {
    std::string mapped_path = remap_sysfs_path(path);
    return realpath(mapped_path.c_str(), resolved_path);
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (device < 0 || device >= static_cast<int>(cuda_device_bus_ids.size())) {
        return cudaErrorInvalidDevice;
    }

    std::strncpy(pciBusId, cuda_device_bus_ids[device].c_str(), len - 1);
    pciBusId[len - 1] = '\0';
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    std::memset(prop, 0, sizeof(*prop));
    prop->pciDomainID = 0;
    prop->pciBusID = device + 1;
    prop->pciDeviceID = 0;
    return cudaSuccess;
}

TEST_F(TopoPolicyTest, AutoUsesLocalPeScopeAndDoesNotRequireGpuSysfs) {
    nvshmemi_options.NETDEVS_POLICY = "AUTO";
    set_state(/*npes=*/2, /*npes_node=*/2, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:01:00.0", "0000:02:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const std::string &bdf : cuda_device_bus_ids) {
        create_pci_device(bdf);
    }

    fs::remove_all(nvidia_root);
    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 2);

    nvshmem_transport transport = make_transport({"0000:01:00.1", "0000:02:00.1"});
    std::vector<int> selected;
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], 0);
}

TEST_F(TopoPolicyTest, ExternalSharingPcieSwitchNicExclusiveUsesAllNodeLocalGpus) {
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:01:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const char *bdf : {"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"}) {
        add_sysfs_gpu(bdf);
    }

    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 4);

    nvshmem_transport transport =
        make_transport({"0000:01:00.1", "0000:02:00.1", "0000:03:00.1", "0000:04:00.1"});
    std::vector<int> selected;
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], 0);
}

TEST_F(TopoPolicyTest, PartialParticipationChangesOnlyPolicyEntityCount) {
    set_state(/*npes=*/2, /*npes_node=*/2, /*mype=*/0, /*device_id=*/0);
    for (const char *bdf : {"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"}) {
        add_sysfs_gpu(bdf);
    }

    nvshmemi_options.NETDEVS_POLICY = "AUTO";
    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 2);

    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 4);
}

TEST_F(TopoPolicyTest,
       ExternalSharingPcieSwitchNicExclusiveSortsSysfsBdfsBeforeChoosingCurrentGpu) {
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:03:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;

    for (const char *bdf : {"0000:04:00.0", "0000:01:00.0", "0000:03:00.0", "0000:02:00.0"}) {
        add_sysfs_gpu(bdf);
    }

    nvshmem_transport transport =
        make_transport({"0000:01:00.1", "0000:02:00.1", "0000:03:00.1", "0000:04:00.1"});
    std::vector<int> selected;
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], 2);
}

TEST_F(TopoPolicyTest, InvalidPolicyFallsBackToAuto) {
    nvshmemi_options.NETDEVS_POLICY = "bad-policy";
    set_state(/*npes=*/2, /*npes_node=*/2, /*mype=*/1, /*device_id=*/1);
    cuda_device_bus_ids = {"0000:01:00.0", "0000:02:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const std::string &bdf : cuda_device_bus_ids) {
        create_pci_device(bdf);
    }
    fs::remove_all(nvidia_root);

    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 2);

    nvshmem_transport transport = make_transport({"0000:01:00.1", "0000:02:00.1"});
    std::vector<int> selected;
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], 1);
}

TEST_F(TopoPolicyTest, ExternalSharingPcieSwitchNicExclusiveFailsWhenGpuSysfsIsUnavailable) {
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:01:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    fs::remove_all(nvidia_root);

    nvshmem_transport transport = make_transport({"0000:01:00.1"});
    std::vector<int> selected;
    EXPECT_NE(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
}

TEST_F(TopoPolicyTest, ExternalSharingPcieSwitchNicExclusiveSharesNicsWhenFewerNicsThanGpus) {
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:02:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const char *bdf : {"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"}) {
        add_sysfs_gpu(bdf);
    }

    nvshmem_transport transport = make_transport({"0000:10:00.1", "0000:11:00.1"},
                                                 /*fixed_numa_node=*/0);

    std::vector<int> selected_exclusive;
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected_exclusive), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected_exclusive.size(), 1u);
    EXPECT_EQ(selected_exclusive[0], 1);
}

TEST_F(TopoPolicyTest,
       ExternalSharingPcieSwitchNicExclusiveCanChangeAssignmentWhenNonParticipatingGpusShareNics) {
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:02:00.0"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const char *bdf : {"0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"}) {
        add_sysfs_gpu(bdf);
    }

    nvshmem_transport transport =
        make_transport({"0000:10:00.1", "0000:11:00.1", "0000:12:00.1", "0000:13:00.1"},
                       /*fixed_numa_node=*/0);

    std::vector<int> selected_auto;
    nvshmemi_options.NETDEVS_POLICY = "AUTO";
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected_auto), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected_auto.size(), 1u);
    EXPECT_EQ(selected_auto[0], 0);

    std::vector<int> selected_exclusive;
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected_exclusive), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected_exclusive.size(), 1u);
    EXPECT_EQ(selected_exclusive[0], 2);

    EXPECT_EQ(nvshmemi_get_netdevs_policy_entity_count(&test_state), 4);
}

TEST_F(
    TopoPolicyTest,
    ExternalSharingPcieSwitchNicExclusiveKeepsOptimalNicWhenOnlyLessOptimalAlternativeAvoidsOverlap) {
    nvshmemi_options.NETDEVS_POLICY = "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
    set_state(/*npes=*/1, /*npes_node=*/1, /*mype=*/0, /*device_id=*/0);
    cuda_device_bus_ids = {"0000:01:00.2"};
    allgather_bus_ids = cuda_device_bus_ids;
    for (const char *bdf : {"0000:01:00.0", "0000:01:00.2"}) {
        add_sysfs_gpu(bdf);
    }

    nvshmem_transport transport = make_transport({"0000:01:00.1", "0000:02:00.1"});

    std::vector<int> selected;
    EXPECT_EQ(select_devices(&transport, /*max_devices=*/1, &selected), NVSHMEMX_SUCCESS);
    ASSERT_EQ(selected.size(), 1u);
    EXPECT_EQ(selected[0], 0);
    EXPECT_TRUE(test_state.are_nics_ll128_compliant);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
