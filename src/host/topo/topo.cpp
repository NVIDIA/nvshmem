/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topo.h"
#include <ctype.h>                                   // for tolower
#include <cuda.h>                                    // for CUDA_SUCCESS
#include <cuda_runtime.h>                            // for cudaDevice...
#include <driver_types.h>                            // for cudaDevice...
#include <algorithm>                                 // for sort
#include <array>                                     // for array
#include <dirent.h>                                  // for opendir, readdir
#include <limits.h>                                  // for PATH_MAX
#include <sched.h>                                   // for cpu_set_t, sched_setaffinity
#include <stdio.h>                                   // for NULL, fclose
#include <stdlib.h>                                  // for free, calloc
#include <string.h>                                  // for strlen
#include <strings.h>                                 // for strcasecmp
#include <list>                                      // for _List_iter...
#include <vector>                                    // for vector
#include <cerrno>                                    // errno
#include <cstddef>                                   // std::size_t
#include <cstdint>                                   // uint32_t
#include <fstream>                                   // std::ifstream
#include <iterator>                                  // std::data, std::istreambuf_iterator
#include <limits>                                    // std::numeric_limits
#include <string>                                    // std::string
#include <string_view>                               // std::string_view
#include "non_abi/nvshmemx_error.h"                  // for NVSHMEMX_E...
#include "internal/host/debug.h"                     // for INFO, NVSH...
#include "internal/host/nvshmem_internal.h"          // for nvshmemi_s...
#include "internal/host/nvshmemi_mem_transport.hpp"  // for nvshm...
#include "internal/host/nvshmemi_types.h"            // for nvshmemi_state
#include "internal/host/util.h"                      // for nvshmemu_getHostHash
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_...
#include "internal/host_transport/cudawrap.h"                              // for CUPFN, nvs...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_o...
#include "internal/host_transport/nvshmemi_transport_defines.h"            // for pcie_id_t
#include "internal/host_transport/transport.h"                             // for nvshmem_tr...

#define MAX_BUSID_SIZE 16
#define MAXPATHSIZE 1024

bool nvshmemi_is_mpg_run = 0;

enum pe_device_assignment {
    PE_DEVICE_NOT_ASSIGNED = -1,
    PE_DEVICE_NO_OPTIMAL_ASSIGNMENT = -2,
};

/* Enumeration of possible PCIe paths and sister arrays for perf characteristics and string
 * representations */
enum pci_distance {
    PATH_PIX = 0,
    PATH_PXB = 1,
    PATH_PHB = 2,
    PATH_NODE = 3,
    PATH_SYS = 4,
    PATH_COUNT = 5
};
static const int pci_distance_perf[PATH_COUNT] = {4, 4, 3, 2, 1};
static const char *pci_distance_string[PATH_COUNT] = {"PIX", "PXB", "PHB", "NODE", "SYS"};

#define NVIDIA_DRIVER_PATH "/sys/bus/pci/drivers/nvidia"

enum netdevs_policy {
    NETDEVS_POLICY_AUTO,
    NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE,
};

static int get_cuda_bus_id(int cuda_dev, char *bus_id) {
    int status = NVSHMEMX_SUCCESS;
    cudaError_t err;

    err = cudaDeviceGetPCIBusId(bus_id, MAX_BUSID_SIZE, cuda_dev);
    if (err != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("cudaDeviceGetPCIBusId failed with error: %d \n", err);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    return status;
}

static int get_numa_id(char *path) {
    char npath[PATH_MAX];
    snprintf(npath, PATH_MAX, "%s/numa_node", path);
    npath[PATH_MAX - 1] = '\0';

    int numaId = -1;
    FILE *file = fopen(npath, "r");
    if (file == NULL) return -1;
    if (fscanf(file, "%d", &numaId) == EOF) {
        fclose(file);
        return -1;
    }
    fclose(file);

    return numaId;
}

static int get_device_path(char *bus_id, char **path) {
    int status = NVSHMEMX_SUCCESS;
    char pathname[MAXPATHSIZE + 1];
    char *cuda_rpath;
    char bus_path[] = "/sys/class/pci_bus/0000:00/device";

    for (int i = 0; i < 16; i++) bus_id[i] = tolower(bus_id[i]);
    memcpy(bus_path + sizeof("/sys/class/pci_bus/") - 1, bus_id, sizeof("0000:00") - 1);

    cuda_rpath = realpath(bus_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(cuda_rpath, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "realpath failed \n");

    strncpy(pathname, cuda_rpath, MAXPATHSIZE);
    strncpy(pathname + strlen(pathname), "/", MAXPATHSIZE - strlen(pathname));
    strncpy(pathname + strlen(pathname), bus_id, MAXPATHSIZE - strlen(pathname));
    free(cuda_rpath);

    *path = realpath(pathname, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static int is_pci_addr(const char *name) {
    // Match XXXX:XX:XX.X pattern
    return strlen(name) == 12 && name[4] == ':' && name[7] == ':' && name[10] == '.';
}

static int get_nvidia_gpu_count(void) {
    DIR *dir = opendir(NVIDIA_DRIVER_PATH);
    if (!dir) return 0;
    int count = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (is_pci_addr(ent->d_name)) count++;
    }
    closedir(dir);
    return count;
}

static enum netdevs_policy get_netdevs_policy(void) {
    if (strcasecmp(nvshmemi_options.NETDEVS_POLICY, "AUTO") == 0) {
        return NETDEVS_POLICY_AUTO;
    }

    if (strcasecmp(nvshmemi_options.NETDEVS_POLICY,
                   "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE") == 0) {
        return NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE;
    }

    NVSHMEMI_WARN_PRINT("Invalid NVSHMEM_NETDEVS_POLICY value '%s'. Using AUTO.\n",
                        nvshmemi_options.NETDEVS_POLICY);
    return NETDEVS_POLICY_AUTO;
}

static const char *get_netdevs_policy_name(enum netdevs_policy policy) {
    switch (policy) {
        case NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE:
            return "EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE";
        case NETDEVS_POLICY_AUTO:
            return "AUTO";
        default:
            return "UNKNOWN";
    }
}

int nvshmemi_get_netdevs_policy_entity_count(nvshmemi_state_t *state) {
    if (get_netdevs_policy() == NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE) {
        int gpu_count = get_nvidia_gpu_count();
        if (gpu_count > 0) return gpu_count;
        if (state && state->npes_node > 0) return state->npes_node;
        return 1;
    }

    if (!state || state->npes_node <= 0) return 1;
    return state->npes_node;
}

static int get_all_physical_gpu_paths_and_index(int cuda_device_id, char ***cuda_device_paths,
                                                int *out_ngpus, int *out_mygpu_index) {
    int status = NVSHMEMX_SUCCESS;
    char my_bus_id[MAX_BUSID_SIZE];
    DIR *nvidia_dir = NULL;
    std::vector<std::array<char, MAX_BUSID_SIZE>> gpu_bus_ids;

    status = get_cuda_bus_id(cuda_device_id, my_bus_id);
    if (status != NVSHMEMX_SUCCESS) return status;
    for (int k = 0; k < MAX_BUSID_SIZE; k++)
        my_bus_id[k] = tolower(my_bus_id[k]);

    nvidia_dir = opendir(NVIDIA_DRIVER_PATH);
    if (!nvidia_dir) {
        NVSHMEMI_ERROR_PRINT("Failed to open " NVIDIA_DRIVER_PATH "\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    *cuda_device_paths = NULL;
    *out_ngpus = 0;
    *out_mygpu_index = -1;
    struct dirent *ent;
    while ((ent = readdir(nvidia_dir)) != NULL) {
        if (!is_pci_addr(ent->d_name)) continue;
        std::array<char, MAX_BUSID_SIZE> bus_id = {};
        strncpy(bus_id.data(), ent->d_name, MAX_BUSID_SIZE - 1);
        for (int k = 0; k < MAX_BUSID_SIZE; k++)
            bus_id[k] = tolower(bus_id[k]);
        gpu_bus_ids.push_back(bus_id);
    }
    closedir(nvidia_dir);

    std::sort(gpu_bus_ids.begin(), gpu_bus_ids.end(),
              [](const std::array<char, MAX_BUSID_SIZE> &lhs,
                 const std::array<char, MAX_BUSID_SIZE> &rhs) {
                  return strncmp(lhs.data(), rhs.data(), MAX_BUSID_SIZE) < 0;
              });

    *out_ngpus = gpu_bus_ids.size();
    if (*out_ngpus <= 0) {
        NVSHMEMI_ERROR_PRINT("No NVIDIA GPUs found in " NVIDIA_DRIVER_PATH "\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    *cuda_device_paths = (char **)calloc(*out_ngpus, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(*cuda_device_paths, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for GPU/NIC Mapping.\n");

    for (int gpu_id = 0; gpu_id < *out_ngpus; gpu_id++) {
        status = get_device_path(gpu_bus_ids[gpu_id].data(), &((*cuda_device_paths)[gpu_id]));
        if (status != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("get cuda path failed\n");
            goto out;
        }

        if (strncmp(my_bus_id, gpu_bus_ids[gpu_id].data(), MAX_BUSID_SIZE) == 0)
            *out_mygpu_index = gpu_id;
    }

    if (*out_mygpu_index < 0) {
        NVSHMEMI_ERROR_PRINT("Could not find current GPU in sysfs\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    if (status) {
        if (*cuda_device_paths) {
            for (int i = 0; i < *out_ngpus; i++) {
                if ((*cuda_device_paths)[i]) free((*cuda_device_paths)[i]);
            }
            free(*cuda_device_paths);
            *cuda_device_paths = NULL;
        }
        *out_ngpus = 0;
    }
    return status;
}

static enum pci_distance get_pci_distance(char *cuda_path, char *mlx_path) {
    int score = 0;
    int depth = 0;
    int same = 1;
    size_t i;
    for (i = 0; i < strlen(cuda_path); i++) {
        if (cuda_path[i] != mlx_path[i]) same = 0;
        if (cuda_path[i] == '/') {
            depth++;
            if (same == 1) score++;
        }
    }
    if (score <= 3) {
        /* Split the former PATH_SOC distance into PATH_NODE and PATH_SYS based on numaId */
        int numaId1 = get_numa_id(cuda_path);
        int numaId2 = get_numa_id(mlx_path);
        return ((numaId1 == numaId2) ? PATH_NODE : PATH_SYS);
    }
    if (score == 4) return PATH_PHB;
    if (score == depth - 1) return PATH_PIX;
    return PATH_PXB;
}

typedef struct nvshmemi_path_pair_info {
    int entity_idx;
    int dev_idx;
    enum pci_distance pcie_distance;
} nvshmemi_path_pair_info_t;

static void free_entity_paths(char **entity_paths, int n_entities) {
    if (!entity_paths) return;

    for (int i = 0; i < n_entities; i++) {
        if (entity_paths[i]) free(entity_paths[i]);
    }
    free(entity_paths);
}

static int collect_local_pe_paths(char ***entity_paths, int *n_entities, int *my_entity_index) {
    int status = NVSHMEMX_ERROR_INTERNAL;
    int mype = nvshmemi_state->mype;
    int n_pes = nvshmemi_state->npes;
    int n_pes_node = nvshmemi_state->npes_node;
    CUdevice gpu_device_id;

    struct gpu_info {
        char gpu_bus_id[MAX_BUSID_SIZE];
    } gpu_info, *gpu_info_all = NULL;

    *entity_paths = NULL;
    *n_entities = 0;
    *my_entity_index = -1;

    status = CUPFN(nvshmemi_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    gpu_info_all = (struct gpu_info *)calloc(n_pes, sizeof(struct gpu_info));
    NVSHMEMI_NULL_ERROR_JMP(gpu_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "gpu_info_all allocation failed \n");

    *entity_paths = (char **)calloc(n_pes_node, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(*entity_paths, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");

    status = get_cuda_bus_id(gpu_device_id, gpu_info.gpu_bus_id);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get cuda busid failed \n");

    status = nvshmemi_boot_handle.allgather((void *)&gpu_info, (void *)gpu_info_all,
                                            sizeof(struct gpu_info), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of gpu_info failed \n");

    for (int i = 0; i < n_pes; i++) {
        if (nvshmemi_state->pe_info[i].hostHash != nvshmemi_state->pe_info[mype].hostHash) {
            continue;
        }

        status = get_device_path(gpu_info_all[i].gpu_bus_id, &((*entity_paths)[*n_entities]));
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get cuda path failed \n");

        if (i == mype) {
            *my_entity_index = *n_entities;
        }

        (*n_entities)++;
        if (*n_entities == n_pes_node) {
            break;
        }
    }

    if (*n_entities != n_pes_node || *my_entity_index == -1) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Number of PEs found doesn't match the PE node count.\n");
    }

    status = NVSHMEMX_SUCCESS;

out:
    if (gpu_info_all) free(gpu_info_all);
    if (status) {
        free_entity_paths(*entity_paths, *n_entities);
        *entity_paths = NULL;
        *n_entities = 0;
    }
    return status;
}

static int collect_all_physical_gpu_paths(char ***entity_paths, int *n_entities,
                                          int *my_entity_index) {
    return get_all_physical_gpu_paths_and_index(nvshmemi_state->device_id, entity_paths,
                                                n_entities, my_entity_index);
}

static int select_devices_by_distance(int *device_arr, int max_dev_per_entity,
                                      struct nvshmem_transport *tcurr, char **entity_paths,
                                      int n_entities, int my_entity_index,
                                      const char *entity_name, enum netdevs_policy policy) {
    struct dev_info {
        char *dev_path;
        int use_count;
    } *dev_info_all = NULL;

    std::list<nvshmemi_path_pair_info_t> entity_dev_pairs;
    std::list<nvshmemi_path_pair_info_t>::iterator pairs_iter;

    int ndev = tcurr->n_devices;

    int *entity_selected_devices = NULL;
    enum pci_distance *entity_device_distance = NULL;
    int *used_devs = NULL;

    int mydev_index = -1;
    int i, dev_id, entity_id, entity_pair_index;
    int devices_assigned = 0;
    int my_entity_device_count = 0;
    int status = NVSHMEMX_ERROR_INTERNAL;
    int my_entity_array_index = my_entity_index * max_dev_per_entity;

    if (ndev <= 0) {
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "transport devices (setup_connections) failed \n");
    }

    if (policy != NETDEVS_POLICY_AUTO &&
        policy != NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unsupported NVSHMEM_NETDEVS_POLICY value %d.\n", policy);
    }

    /* Allocate data structures start */
    /* Array of dev_info structures of size # of local NICs */
    dev_info_all = (struct dev_info *)calloc(ndev, sizeof(struct dev_info));
    NVSHMEMI_NULL_ERROR_JMP(dev_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "dev_info_all allocation failed \n");

    used_devs = (int *)calloc(ndev, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(used_devs, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");
    /* Allocate data structures end */

    entity_selected_devices = (int *)calloc(n_entities * max_dev_per_entity, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(entity_selected_devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for NIC Mapping.\n");
    for (entity_id = 0; entity_id < n_entities; entity_id++) {
        for (dev_id = 0; dev_id < max_dev_per_entity; dev_id++) {
            entity_selected_devices[entity_id * max_dev_per_entity + dev_id] = -1;
        }
    }

    entity_device_distance =
        (enum pci_distance *)calloc(n_entities * max_dev_per_entity, sizeof(enum pci_distance));
    NVSHMEMI_NULL_ERROR_JMP(entity_device_distance, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for NIC Mapping.\n");
    for (entity_id = 0; entity_id < n_entities; entity_id++) {
        for (dev_id = 0; dev_id < max_dev_per_entity; dev_id++) {
            entity_device_distance[entity_id * max_dev_per_entity + dev_id] = PATH_SYS;
        }
    }

    for (i = 0; i < ndev; i++) {
        dev_info_all[i].dev_path = tcurr->device_pci_paths[i];
        NVSHMEMI_NULL_ERROR_JMP(dev_info_all[i].dev_path, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "get device path failed \n");
    }

    /* Get path distances start */
    /* construct a n_entities * ndev array of distance measurements */
    for (entity_id = 0; entity_id < n_entities; entity_id++) {
        for (dev_id = 0; dev_id < ndev; dev_id++) {
            enum pci_distance distance_compare;
            distance_compare =
                get_pci_distance(entity_paths[entity_id], dev_info_all[dev_id].dev_path);
            if (unlikely(entity_dev_pairs.empty())) {
                entity_dev_pairs.push_front({entity_id, dev_id, distance_compare});
            } else {
                for (pairs_iter = entity_dev_pairs.begin(); pairs_iter != entity_dev_pairs.end();
                     pairs_iter++) {
                    if (distance_compare < (*pairs_iter).pcie_distance) {
                        break;
                    }
                }
                INFO(NVSHMEM_TOPO, "%s %d: %s dev %d: %s distance: %d\n", entity_name,
                     entity_id, entity_paths[entity_id], dev_id, dev_info_all[dev_id].dev_path,
                     distance_compare);
                entity_dev_pairs.insert(pairs_iter, {entity_id, dev_id, distance_compare});
            }
        }
    }
    /* Get path distances end */

    /* loop one, do initial assignments of NIC(s) to each entity */
    for (pairs_iter = entity_dev_pairs.begin(); pairs_iter != entity_dev_pairs.end();
         pairs_iter++) {
        bool need_more_assignments = 0;
        int entity_base_index = (*pairs_iter).entity_idx * max_dev_per_entity;
        /* skip pairs where the entity already has a partner in the first loop */
        for (entity_pair_index = 0; entity_pair_index < max_dev_per_entity; entity_pair_index++)
            if (entity_selected_devices[entity_base_index + entity_pair_index] ==
                PE_DEVICE_NOT_ASSIGNED) {
                need_more_assignments = 1;
                break;
            }

        if (!need_more_assignments) {
            continue;
        }

        if (pci_distance_perf[(*pairs_iter).pcie_distance] <
            pci_distance_perf[entity_device_distance[entity_base_index]]) {
            /* This NIC and all subsequent ones are less optimal than the already selected NICs
             * They can be safely ignored and we assign -2 to indicate that there are no more
             * optimal NICs for this entity.
             */
            for (; entity_pair_index < max_dev_per_entity; entity_pair_index++) {
                entity_selected_devices[entity_base_index + entity_pair_index] =
                    PE_DEVICE_NO_OPTIMAL_ASSIGNMENT;
                /* While not technically assigned, we need to account for these NICs to make
                 * forward progress.
                 */
                devices_assigned++;
            }
        } else {
            /* This NIC is optimal for this entity. */
            INFO(NVSHMEM_TOPO, "Pairing %s %d with device %d at distance %d\n", entity_name,
                 (*pairs_iter).entity_idx, (*pairs_iter).dev_idx, (*pairs_iter).pcie_distance);
            entity_selected_devices[entity_base_index + entity_pair_index] =
                (*pairs_iter).dev_idx;
            entity_device_distance[entity_base_index + entity_pair_index] =
                (*pairs_iter).pcie_distance;
            used_devs[(*pairs_iter).dev_idx]++;
            devices_assigned++;
        }

        if (devices_assigned == n_entities * max_dev_per_entity) {
            break;
        }
    }

    /* loop two, load balance the NICs. */
    for (entity_id = 0; entity_id < n_entities; entity_id++) {
        for (dev_id = 0; dev_id < max_dev_per_entity; dev_id++) {
            int entity_pair_idx = entity_id * max_dev_per_entity + dev_id;
            int nic_density;
            if (entity_selected_devices[entity_pair_idx] < 0) {
                continue;
            }
            nic_density = used_devs[entity_selected_devices[entity_pair_idx]];

            /* Can't find a less populated NIC if ours is only assigned to one entity. */
            if (nic_density < 2) {
                continue;
            }

            /* Calculate entity index from nic_id. Each entity gets max_dev_per_entity assigned
             * to it. If there are 8 NICs and 4 entities, the nic -> entity mapping looks like
             * nic_id:  0   1   2   3   4   5   6   7
             * entity:  0   0   1   1   2   2   3   3
             */
            int entity_idx =
                (entity_pair_idx - (entity_pair_idx % max_dev_per_entity)) / max_dev_per_entity;
            for (pairs_iter = entity_dev_pairs.begin(); pairs_iter != entity_dev_pairs.end();
                 pairs_iter++) {
                /* Never change for a less optimal NIC. */

                if ((*pairs_iter).entity_idx != entity_idx) {
                    continue;
                }

                if (pci_distance_perf[(*pairs_iter).pcie_distance] <
                    pci_distance_perf[entity_device_distance[entity_pair_idx]]) {
                    break;
                }

                if ((nic_density - used_devs[(*pairs_iter).dev_idx]) >= 2) {
                    INFO(NVSHMEM_TOPO, "Re-Pairing %s %d with device %d at distance %d\n",
                         entity_name, (*pairs_iter).entity_idx, (*pairs_iter).dev_idx,
                         (*pairs_iter).pcie_distance);
                    used_devs[entity_selected_devices[entity_pair_idx]]--;
                    used_devs[(*pairs_iter).dev_idx]++;
                    nic_density = used_devs[(*pairs_iter).dev_idx];
                    entity_selected_devices[entity_pair_idx] = (*pairs_iter).dev_idx;
                    entity_device_distance[entity_pair_idx] = (*pairs_iter).pcie_distance;
                    if (nic_density < 2) {
                        break;
                    }
                }
            }
        }
    }

    for (entity_pair_index = 0; entity_pair_index < max_dev_per_entity; entity_pair_index++) {
        if (entity_selected_devices[my_entity_array_index + entity_pair_index] >= 0) {
            mydev_index = entity_selected_devices[my_entity_array_index + entity_pair_index];
            device_arr[entity_pair_index] = mydev_index;
            my_entity_device_count++;
            INFO(NVSHMEM_TOPO, "Our %s selected device %d, shared by %d %ss.\n", entity_name,
                 mydev_index, used_devs[mydev_index], entity_name);
        }
    }

    if (my_entity_device_count == 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "No NICs were assigned to our %s.\n", entity_name);
    }

    /* No need to report this in a loop - All Devices will have the same perf characteristics. */
    if (pci_distance_perf[entity_device_distance[my_entity_array_index]] <
        pci_distance_perf[PATH_PIX]) {
        nvshmemi_state->are_nics_ll128_compliant = false;
        INFO(NVSHMEM_TOPO,
             "Our %s is connected to a NIC with pci distance %s."
             " this will provide less than optimal performance.\n",
             entity_name, pci_distance_string[entity_device_distance[my_entity_array_index]]);
    }

    status = NVSHMEMX_SUCCESS;

out:
    if (dev_info_all) {
        free(dev_info_all);
    }

    entity_dev_pairs.clear();

    if (entity_selected_devices) {
        free(entity_selected_devices);
    }

    if (used_devs) {
        free(used_devs);
    }

    if (entity_device_distance) {
        free(entity_device_distance);
    }

    return status;
}

int nvshmemi_get_devices_by_distance(int *device_arr, int max_dev_per_pe,
                                     struct nvshmem_transport *tcurr) {
    char **entity_paths = NULL;
    int n_entities = 0;
    int my_entity_index = -1;
    int status;
    const char *entity_name;
    enum netdevs_policy policy = get_netdevs_policy();

    if (policy == NETDEVS_POLICY_EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE) {
        entity_name = "GPU";
        status = collect_all_physical_gpu_paths(&entity_paths, &n_entities, &my_entity_index);
    } else {
        entity_name = "PE";
        status = collect_local_pe_paths(&entity_paths, &n_entities, &my_entity_index);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "failed to collect topology assignment entities\n");

    INFO(NVSHMEM_TOPO,
         "NVSHMEM_NETDEVS_POLICY=%s assignment_scope=%s n_entities=%d "
         "my_entity_index=%d max_devices_per_entity=%d\n",
         get_netdevs_policy_name(policy), entity_name, n_entities, my_entity_index,
         max_dev_per_pe);

    status = select_devices_by_distance(device_arr, max_dev_per_pe, tcurr, entity_paths, n_entities,
                                        my_entity_index, entity_name, policy);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "failed to select devices by distance\n");

out:
    free_entity_paths(entity_paths, n_entities);
    return status;
}

int nvshmemi_build_transport_map(nvshmemi_state_t *state) {
    int status = 0;
    int *local_map = NULL;

    if (state->transport_map != NULL) {
        free(state->transport_map);
        state->transport_map = NULL;
    }

    state->transport_map = (int *)calloc(state->npes * state->npes, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(state->transport_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "access map allocation failed \n");

    local_map = (int *)calloc(state->npes, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(local_map, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "access map allocation failed \n");

    state->transport_bitmap = 0;

    for (int i = 0; i < state->npes; i++) {
        int reach_any = 0;

        for (int j = 0; j < state->num_initialized_transports; j++) {
            int reach = 0;

            if (!state->transports[j]) {
                continue;
            }

            status = state->transports[j]->host_ops.can_reach_peer(&reach, &state->pe_info[i],
                                                                   state->transports[j]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "can reach peer failed \n");
            INFO(NVSHMEM_TOPO, "[%d] reach %d to peer %d over transport %d", state->mype, reach, i,
                 j);

            state->transports[j]->cap[i] = reach;
            reach_any |= reach;

            if (reach) {
                int m = 1 << j;
                local_map[i] |= m;
                /* Add transport to the bitmap if this is the first PE to use it. */
                if ((state->transport_bitmap & m) == 0) {
                    state->transport_bitmap |= m;
                }
            }
        }

        if ((!reach_any) && (!nvshmemi_options.BYPASS_ACCESSIBILITY_CHECK)) {
            status = NVSHMEMX_ERROR_NOT_SUPPORTED;
            fprintf(stderr, "%s:%d: [GPU %d] Peer GPU %d is not accessible, exiting ... \n",
                    __FILE__, __LINE__, state->mype, i);
            goto out;
        }
    }
    INFO(NVSHMEM_TOPO, "[%d] transport bitmap: %x", state->mype, state->transport_bitmap);

    status = nvshmemi_boot_handle.allgather((void *)local_map, (void *)state->transport_map,
                                            sizeof(int) * state->npes, &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ipc handles failed \n");

out:
    if (local_map) free(local_map);
    if (status) {
        if (state->transport_map) free(state->transport_map);
    }
    return status;
}

int nvshmemi_get_pcie_attrs(pcie_id_t *pcie_id, int devid) {
    int status = 0;
    cudaDeviceProp prop;

    status = cudaGetDeviceProperties(&prop, devid);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaDeviceGetAttribute failed \n");
    pcie_id->dev_id = prop.pciDeviceID;
    pcie_id->bus_id = prop.pciBusID;
    pcie_id->domain_id = prop.pciDomainID;

out:
    return status;
}

int nvshmemi_detect_same_device(nvshmemi_state_t *state) {
    int status = NVSHMEMX_SUCCESS;
    nvshmem_transport_pe_info_t my_info;
    cudaDeviceProp prop;

    my_info.pe = state->mype;
    status = nvshmemi_get_pcie_attrs(&my_info.pcie_id, state->device_id);
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "getPcieAttrs failed \n");

    my_info.hostHash = nvshmemu_getHostHash();
    cudaGetDeviceProperties(&prop, state->device_id);
    my_info.gpu_uuid = prop.uuid;

    // TODO: move this to a topo init function as it is reused in other functions in topo that
    // follow
    state->pe_info =
        (nvshmem_transport_pe_info_t *)malloc(sizeof(nvshmem_transport_pe_info_t) * state->npes);
    NVSHMEMI_NULL_ERROR_JMP(state->pe_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "topo init info allocation failed \n");
    status =
        nvshmemi_boot_handle.allgather((void *)&my_info, (void *)state->pe_info,
                                       sizeof(nvshmem_transport_pe_info_t), &nvshmemi_boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ipc handles failed \n");

    for (int i = 0; i < state->npes; i++) {
        (state->pe_info + i)->pe = i;
        if (i == state->mype) continue;

        status = (((state->pe_info + i)->hostHash == my_info.hostHash) &&
                  ((state->pe_info + i)->pcie_id.dev_id == my_info.pcie_id.dev_id) &&
                  ((state->pe_info + i)->pcie_id.bus_id == my_info.pcie_id.bus_id) &&
                  ((state->pe_info + i)->pcie_id.domain_id == my_info.pcie_id.domain_id));
        if (status) {
            INFO(NVSHMEM_INIT, "More than 1 PE per GPU detected. This is an MPG run.\n");
#if defined(NVSHMEM_PPC64LE)
            NVSHMEMI_ERROR_EXIT("MPG support is currently not available on P9 platforms");
#endif
            nvshmemi_is_mpg_run = 1;
            status = NVSHMEMX_SUCCESS;
        }
    }

out:
    if (status) {
        state->cucontext = NULL;
        if (state->pe_info) free(state->pe_info);
    }
    return status;
}

typedef enum {
    NVSHMEMI_CPU_AFFINITY_AUTO,
    NVSHMEMI_CPU_AFFINITY_OFF,
} nvshmemi_cpu_affinity_mode_t;

static nvshmemi_cpu_affinity_mode_t get_cpu_affinity_mode() {
    const char *mode = nvshmemi_options.CPU_AFFINITY;

    if (mode == nullptr || strcasecmp(mode, "AUTO") == 0) {
        return NVSHMEMI_CPU_AFFINITY_AUTO;
    }

    if (strcasecmp(mode, "OFF") == 0) {
        return NVSHMEMI_CPU_AFFINITY_OFF;
    }

    NVSHMEMI_WARN_PRINT("Invalid NVSHMEM_CPU_AFFINITY value '%s'. Using AUTO.", mode);
    return NVSHMEMI_CPU_AFFINITY_AUTO;
}

static int parse_cpumap_mask(std::string_view token, uint32_t *mask) {
    constexpr int cpumap_base = 16;
    constexpr std::string_view hex_digits = "0123456789abcdefABCDEF";

    if (token.empty() || token.find_first_not_of(hex_digits) != std::string_view::npos) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    std::string token_string{std::data(token), token.size()};
    errno = 0;
    unsigned long value = strtoul(token_string.c_str(), nullptr, cpumap_base);
    if (errno != 0 || value > std::numeric_limits<uint32_t>::max()) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    *mask = static_cast<uint32_t>(value);
    return NVSHMEMX_SUCCESS;
}

/* Parse hex cpumap string (e.g. "0000ffff,0000ffff") into cpu_set_t.
 * Mirrors NCCL's ncclStrToCpuset. */
static int cpumap_to_cpuset(std::string_view map_str, cpu_set_t *set) {
    constexpr int cpus_per_mask = 32;
    static_assert(CPU_SETSIZE % cpus_per_mask == 0, "CPU_SETSIZE must align with cpumap masks");
    constexpr int mask_count = CPU_SETSIZE / cpus_per_mask;
    std::array<uint32_t, mask_count> masks = {};
    int m = mask_count;

    std::size_t start = 0;
    while (start < map_str.size()) {
        if (m == 0) {
            INFO(NVSHMEM_INIT, "cpumap contains more than %d %d-bit masks; CPU_SETSIZE is %d.\n",
                 mask_count, cpus_per_mask, CPU_SETSIZE);
            return NVSHMEMX_ERROR_INTERNAL;
        }

        std::size_t end = map_str.find(',', start);
        std::string_view token_view =
            map_str.substr(start, end == std::string_view::npos ? std::string_view::npos
                                                                : end - start);
        uint32_t parsed_mask;
        int status = parse_cpumap_mask(token_view, &parsed_mask);
        if (status != NVSHMEMX_SUCCESS) {
            return status;
        }
        masks[--m] = parsed_mask;

        if (end == std::string_view::npos) break;
        start = end + 1;
    }

    CPU_ZERO(set);
    for (int a = 0; (a + m) < mask_count; a++) {
        for (int i = 0; i < cpus_per_mask; i++) {
            if (masks[a + m] & (1U << i)) {
                CPU_SET(i + a * cpus_per_mask, set);
            }
        }
    }

    return NVSHMEMX_SUCCESS;
}

static int set_cpu_affinity(nvshmemi_state_t *state) {
    CUdevice cudev;
    cpu_set_t cur_set, numa_set, final_set;
    int numa_id = -1;
    int status;

    status = CUPFN(nvshmemi_cuda_syms, cuDeviceGet)(&cudev, state->device_id);
    if (status != CUDA_SUCCESS) {
        INFO(NVSHMEM_INIT, "cuDeviceGet failed: %d.\n", status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = CUPFN(nvshmemi_cuda_syms,
                   cuDeviceGetAttribute)(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cudev);
    if (status != CUDA_SUCCESS || numa_id < 0) {
        INFO(NVSHMEM_INIT, "cuDeviceGetAttribute failed: %d (numa_id: %d).\n", status, numa_id);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Get current process affinity */
    status = sched_getaffinity(0, sizeof(cur_set), &cur_set);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "sched_getaffinity failed: %d.\n", status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Read cpumap for this NUMA node */
    std::array<char, PATH_MAX> cpumap_path{};
    int written = snprintf(std::data(cpumap_path), cpumap_path.size(),
                           "/sys/devices/system/node/node%d/cpumap", numa_id);
    if (written < 0 || static_cast<std::size_t>(written) >= cpumap_path.size()) {
        INFO(NVSHMEM_INIT, "cpumap path is too long for NUMA node %d.\n", numa_id);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    std::ifstream cpumap_file{std::data(cpumap_path)};
    if (!cpumap_file) {
        INFO(NVSHMEM_INIT, "unable to open cpumap path: %s.\n", std::data(cpumap_path));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    std::string map_str{std::istreambuf_iterator<char>(cpumap_file),
                        std::istreambuf_iterator<char>()};

    if (map_str.empty()) {
        INFO(NVSHMEM_INIT, "cpumap path is empty.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    /* Strip newline if present */
    if (map_str.back() == '\n') map_str.pop_back();

    status = cpumap_to_cpuset(map_str, &numa_set);
    if (status != NVSHMEMX_SUCCESS) {
        INFO(NVSHMEM_INIT, "failed to parse cpumap path: %s.\n", std::data(cpumap_path));
        return status;
    }
    CPU_AND(&final_set, &cur_set, &numa_set);

    if (CPU_COUNT(&final_set) == 0) {
        INFO(NVSHMEM_INIT, "target cpuset is empty.\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = sched_setaffinity(0, sizeof(final_set), &final_set);
    if (status != 0) {
        INFO(NVSHMEM_INIT, "sched_setaffinity failed: %d.\n", status);
        return NVSHMEMX_ERROR_INTERNAL;
    }

    INFO(NVSHMEM_INIT, "PE %d pinned to NUMA node %d (%d CPUs) for GPU %d",
         nvshmemi_boot_handle.pg_rank, numa_id, CPU_COUNT(&final_set), state->device_id);

    return NVSHMEMX_SUCCESS;
}

void nvshmemi_apply_cpu_affinity(nvshmemi_state_t *state) {
    if (get_cpu_affinity_mode() == NVSHMEMI_CPU_AFFINITY_OFF) {
        return;
    }

    if (set_cpu_affinity(state) != NVSHMEMX_SUCCESS) {
        INFO(NVSHMEM_INIT, "Failed to set CPU affinity - skipping.\n");
    }
}
