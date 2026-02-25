/*
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "topo.h"
#include <ctype.h>                                   // for tolower
#include <cuda.h>                                    // for CUDA_SUCCESS
#include <cuda_runtime.h>                            // for cudaDevice...
#include <driver_types.h>                            // for cudaDevice...
#include <dirent.h>                                  // for opendir, readdir
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

#define NVIDIA_DRIVER_PATH "/sys/bus/pci/drivers/nvidia"

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

int get_nvidia_gpu_count(void) {
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

static int get_gpu_paths_and_index(int cuda_device_id, char **cuda_device_paths,
                                   int *out_mygpu_index) {
    int status = NVSHMEMX_SUCCESS;
    char my_bus_id[MAX_BUSID_SIZE];
    DIR *nvidia_dir = NULL;

    status = get_cuda_bus_id(cuda_device_id, my_bus_id);
    if (status != NVSHMEMX_SUCCESS) return status;
    for (int k = 0; k < MAX_BUSID_SIZE; k++)
        my_bus_id[k] = tolower(my_bus_id[k]);

    nvidia_dir = opendir(NVIDIA_DRIVER_PATH);
    if (!nvidia_dir) {
        NVSHMEMI_ERROR_PRINT("Failed to open " NVIDIA_DRIVER_PATH "\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    int gpu_id = 0;
    *out_mygpu_index = -1;
    struct dirent *ent;
    while ((ent = readdir(nvidia_dir)) != NULL) {
        if (!is_pci_addr(ent->d_name)) continue;
        char bus_id[MAX_BUSID_SIZE];
        strncpy(bus_id, ent->d_name, MAX_BUSID_SIZE - 1);
        bus_id[MAX_BUSID_SIZE - 1] = '\0';

        status = get_device_path(bus_id, &cuda_device_paths[gpu_id]);
        if (status != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("get cuda path failed\n");
            closedir(nvidia_dir);
            return status;
        }

        if (strncmp(my_bus_id, bus_id, MAX_BUSID_SIZE) == 0)
            *out_mygpu_index = gpu_id;

        gpu_id++;
    }
    closedir(nvidia_dir);

    if (*out_mygpu_index < 0) {
        NVSHMEMI_ERROR_PRINT("Could not find current GPU in sysfs\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return NVSHMEMX_SUCCESS;
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
    int gpu_idx;
    int dev_idx;
    enum pci_distance pcie_distance;
} nvshmemi_path_pair_info_t;

int nvshmemi_get_devices_by_distance(int *device_arr, int max_dev_per_pe,
                                     struct nvshmem_transport *tcurr) {
    struct dev_info {
        char *dev_path;
        int use_count;
    } *dev_info_all = NULL;

    struct gpu_info {
        char gpu_bus_id[MAX_BUSID_SIZE];
    } gpu_info, *gpu_info_all = NULL;

    std::list<nvshmemi_path_pair_info_t> gpu_dev_pairs;
    std::list<nvshmemi_path_pair_info_t>::iterator pairs_iter;

    int ndev = tcurr->n_devices;
    int mype = nvshmemi_state->mype;
    int n_pes = nvshmemi_state->npes;
    int n_pes_node = nvshmemi_state->npes_node;

    char **cuda_device_paths = NULL;
    int *gpu_selected_devices = NULL;
    enum pci_distance *gpu_device_distance = NULL;
    int *used_devs = NULL;
    int n_gpus_node = 0;

    int mygpu_index = -1, mydev_index = -1;
    int i, dev_id, gpu_id, gpu_pair_index;
    int devices_assigned = 0;
    int mygpu_device_count = 0;
    int status = NVSHMEMX_ERROR_INTERNAL;
    int mygpu_array_index;

    if (ndev <= 0) {
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "transport devices (setup_connections) failed \n");
    }

    /* Allocate data structures start */
    /* Array of dev_info structures of size # of local NICs */
    dev_info_all = (struct dev_info *)calloc(ndev, sizeof(struct dev_info));
    NVSHMEMI_NULL_ERROR_JMP(dev_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "dev_info_all allocation failed \n");

    /* Array of GPU bus IDs of size n_pes*/
    gpu_info_all = (struct gpu_info *)calloc(n_pes, sizeof(struct gpu_info));
    NVSHMEMI_NULL_ERROR_JMP(gpu_info_all, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "gpu_info_all allocation failed \n");

    used_devs = (int *)calloc(ndev, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(used_devs, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for PE/NIC Mapping.\n");
    /* Allocate data structures end */

    /* Gather GPU and NIC paths start */
    n_gpus_node = get_nvidia_gpu_count();
    if (n_gpus_node <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "No NVIDIA GPUs found in " NVIDIA_DRIVER_PATH "\n");
    }

    cuda_device_paths = (char **)calloc(n_gpus_node, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(cuda_device_paths, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for GPU/NIC Mapping.\n");

    status = get_gpu_paths_and_index(nvshmemi_state->device_id, cuda_device_paths, &mygpu_index);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "get_gpu_paths_and_index failed\n");
    mygpu_array_index = mygpu_index * max_dev_per_pe;

    /* Allocate GPU-based arrays */
    gpu_selected_devices = (int *)calloc(n_gpus_node * max_dev_per_pe, sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(gpu_selected_devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for GPU/NIC Mapping.\n");
    for (gpu_id = 0; gpu_id < n_gpus_node; gpu_id++) {
        for (dev_id = 0; dev_id < max_dev_per_pe; dev_id++) {
            gpu_selected_devices[gpu_id * max_dev_per_pe + dev_id] = -1;
        }
    }

    gpu_device_distance =
        (enum pci_distance *)calloc(n_gpus_node * max_dev_per_pe, sizeof(enum pci_distance));
    NVSHMEMI_NULL_ERROR_JMP(gpu_device_distance, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate memory for GPU/NIC Mapping.\n");
    for (gpu_id = 0; gpu_id < n_gpus_node; gpu_id++) {
        for (dev_id = 0; dev_id < max_dev_per_pe; dev_id++) {
            gpu_device_distance[gpu_id * max_dev_per_pe + dev_id] = PATH_SYS;
        }
    }

    for (i = 0; i < ndev; i++) {
        dev_info_all[i].dev_path = tcurr->device_pci_paths[i];
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get device path failed \n");
    }
    /* Gather GPU and NIC paths end */

    /* Get path distances start */
    /* construct a n_gpus_node * ndev array of distance measurements */
    for (gpu_id = 0; gpu_id < n_gpus_node; gpu_id++) {
        for (dev_id = 0; dev_id < ndev; dev_id++) {
            enum pci_distance distance_compare;
            distance_compare =
                get_pci_distance(cuda_device_paths[gpu_id], dev_info_all[dev_id].dev_path);
            if (unlikely(gpu_dev_pairs.empty())) {
                gpu_dev_pairs.push_front({gpu_id, dev_id, distance_compare});
            } else {
                for (pairs_iter = gpu_dev_pairs.begin(); pairs_iter != gpu_dev_pairs.end();
                     pairs_iter++) {
                    if (distance_compare < (*pairs_iter).pcie_distance) {
                        break;
                    }
                }
                INFO(NVSHMEM_TOPO, "GPU %d: %s dev %d: %s distance: %d\n", gpu_id,
                     cuda_device_paths[gpu_id], dev_id, dev_info_all[dev_id].dev_path,
                     distance_compare);
                gpu_dev_pairs.insert(pairs_iter, {gpu_id, dev_id, distance_compare});
            }
        }
    }
    /* Get path distances end */

    /* loop one, do initial assignments of NIC(s) to each GPU */
    for (pairs_iter = gpu_dev_pairs.begin(); pairs_iter != gpu_dev_pairs.end(); pairs_iter++) {
        bool need_more_assignments = 0;
        int gpu_base_index = (*pairs_iter).gpu_idx * max_dev_per_pe;
        /* skip pairs where the GPU already has a partner in the first loop */
        for (gpu_pair_index = 0; gpu_pair_index < max_dev_per_pe; gpu_pair_index++)
            if (gpu_selected_devices[gpu_base_index + gpu_pair_index] == PE_DEVICE_NOT_ASSIGNED) {
                need_more_assignments = 1;
                break;
            }

        if (!need_more_assignments) {
            continue;
        }

        if (pci_distance_perf[(*pairs_iter).pcie_distance] <
            pci_distance_perf[gpu_device_distance[gpu_base_index]]) {
            /* This NIC and all subsequent ones are less optimal than the already selected NICs
             * They can be safely ignored and we assign -2 to indicate that there are no more
             * optimal NICs for this GPU.
             */
            for (; gpu_pair_index < max_dev_per_pe; gpu_pair_index++) {
                gpu_selected_devices[gpu_base_index + gpu_pair_index] =
                    PE_DEVICE_NO_OPTIMAL_ASSIGNMENT;
                /* While not technically assigned, we need to account for these NICs to make
                 * forward progress.
                 */
                devices_assigned++;
            }
        } else {
            /* This NIC is optimal for this GPU. */
            INFO(NVSHMEM_TOPO, "Pairing GPU %d with device %d at distance %d\n",
                 (*pairs_iter).gpu_idx, (*pairs_iter).dev_idx, (*pairs_iter).pcie_distance);
            gpu_selected_devices[gpu_base_index + gpu_pair_index] = (*pairs_iter).dev_idx;
            gpu_device_distance[gpu_base_index + gpu_pair_index] = (*pairs_iter).pcie_distance;
            used_devs[(*pairs_iter).dev_idx]++;
            devices_assigned++;
        }

        if (devices_assigned == n_gpus_node * max_dev_per_pe) {
            break;
        }
    }

    /* loop two, load balance the NICs. */
    for (gpu_id = 0; gpu_id < n_gpus_node; gpu_id++) {
        for (dev_id = 0; dev_id < max_dev_per_pe; dev_id++) {
            int gpu_pair_idx = gpu_id * max_dev_per_pe + dev_id;
            int nic_density;
            if (gpu_selected_devices[gpu_pair_idx] < 0) {
                continue;
            }
            nic_density = used_devs[gpu_selected_devices[gpu_pair_idx]];

            /* Can't find a less populated NIC if ours is only assigned to 1 gpu. */
            if (nic_density < 2) {
                continue;
            }

            /* Calculate GPU Index from nic_id. Each GPU gets max_dev_per_pe assigned to them.
             * If there are 8 NIC's and 4 GPU's, the nic -> GPU mapping looks like
             * nic_id:  0   1   2   3   4   5   6   7
             * gpu_idx:  0   0   1   1   2   2   3   3
             */
            int gpu_idx = (gpu_pair_idx - (gpu_pair_idx % max_dev_per_pe)) / max_dev_per_pe;
            for (pairs_iter = gpu_dev_pairs.begin(); pairs_iter != gpu_dev_pairs.end();
                 pairs_iter++) {
                /* Never change for a less optimal NIC. */

                if ((*pairs_iter).gpu_idx != gpu_idx) {
                    continue;
                }

                if (pci_distance_perf[(*pairs_iter).pcie_distance] <
                    pci_distance_perf[gpu_device_distance[gpu_pair_idx]]) {
                    break;
                }

                if ((nic_density - used_devs[(*pairs_iter).dev_idx]) >= 2) {
                    INFO(NVSHMEM_TOPO, "Re-Pairing GPU %d with device %d at distance %d\n",
                         (*pairs_iter).gpu_idx, (*pairs_iter).dev_idx, (*pairs_iter).pcie_distance);
                    used_devs[gpu_selected_devices[gpu_pair_idx]]--;
                    used_devs[(*pairs_iter).dev_idx]++;
                    nic_density = used_devs[(*pairs_iter).dev_idx];
                    gpu_selected_devices[gpu_pair_idx] = (*pairs_iter).dev_idx;
                    gpu_device_distance[gpu_pair_idx] = (*pairs_iter).pcie_distance;
                    if (nic_density < 2) {
                        break;
                    }
                }
            }
        }

        for (gpu_pair_index = 0; gpu_pair_index < max_dev_per_pe; gpu_pair_index++) {
            if (gpu_selected_devices[mygpu_array_index + gpu_pair_index] >= 0) {
                mydev_index = gpu_selected_devices[mygpu_array_index + gpu_pair_index];
                device_arr[gpu_pair_index] = mydev_index;
                mygpu_device_count++;
                INFO(NVSHMEM_TOPO, "Our GPU is sharing its NIC at index %d with %d other GPUs.\n",
                     used_devs[mydev_index], mygpu_device_count);
            }
        }

        if (mygpu_device_count == 0) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "No NICs were assigned to our GPU.\n");
        }

        /* No need to report this in a loop - All Devices will have the same perf characteristics.
         */
        if (pci_distance_perf[gpu_device_distance[mygpu_array_index]] < pci_distance_perf[PATH_PIX]) {
            nvshmemi_state->are_nics_ll128_compliant = false;
            INFO(NVSHMEM_TOPO,
                 "Our GPU is connected to a NIC with pci distance %s."
                 "this will provide less than optimal performance.\n",
                 pci_distance_string[gpu_device_distance[mygpu_array_index]]);
        }
    }

    status = NVSHMEMX_SUCCESS;

out:
    if (dev_info_all) {
        free(dev_info_all);
    }

    if (gpu_info_all) {
        free(gpu_info_all);
    }

    if (cuda_device_paths) {
        for (i = 0; i < n_gpus_node; i++) {
            if (cuda_device_paths[i]) {
                free(cuda_device_paths[i]);
            }
        }
        free(cuda_device_paths);
    }

    gpu_dev_pairs.clear();

    if (gpu_selected_devices) {
        free(gpu_selected_devices);
    }

    if (used_devs) {
        free(used_devs);
    }

    if (gpu_device_distance) {
        free(gpu_device_distance);
    }

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
