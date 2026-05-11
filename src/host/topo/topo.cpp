/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "topo.h"
#include <ctype.h>                                   // for tolower
#include <cuda.h>                                    // for CUDA_SUCCESS
#include <cuda_runtime.h>                            // for cudaDevice...
#include <driver_types.h>                            // for cudaDevice...
#include <limits.h>                                  // for PATH_MAX
#include <stdio.h>                                   // for NULL, fclose
#include <stdlib.h>                                  // for free, calloc
#include <string.h>                                  // for strlen
#include <list>                                      // for _List_iter...
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

static int select_devices_by_distance(int *device_arr, int max_dev_per_entity,
                                      struct nvshmem_transport *tcurr, char **entity_paths,
                                      int n_entities, int my_entity_index,
                                      const char *entity_name) {
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

            /* Calculate entity index from nic_id. Each entity gets max_dev_per_entity assigned to
             * it. If there are 8 NICs and 4 entities, the nic -> entity mapping looks like
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
    const char *entity_name = "PE";

    status = collect_local_pe_paths(&entity_paths, &n_entities, &my_entity_index);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "failed to collect topology assignment entities\n");

    status = select_devices_by_distance(device_arr, max_dev_per_pe, tcurr, entity_paths, n_entities,
                                        my_entity_index, entity_name);
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
