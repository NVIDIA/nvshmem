/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>                                   // for assert
#include <dlfcn.h>                                    // for dlclose, dlerror
#include <stdint.h>                                   // for SIZE_MAX
#include <stdio.h>                                    // for snprintf, NULL
#include <mutex>                                      // for std::once_flag, std::call_once
#include <stdlib.h>                                   // for calloc
#include <vector>                                     // for std::vector
#include <strings.h>                                  // for strncasecmp
#include "device_host/nvshmem_types.h"                // for nvshmemi_devi...
#include "device_host/nvshmem_common.cuh"             // for nvshmemi_devi...
#include "non_abi/nvshmemx_error.h"                   // for NVSHMEMI_ERRO...
#include "internal/host/debug.h"                      // for INFO, NVSHMEM...
#include "internal/host/nvshmem_internal.h"           // for nvshmemi_loca...
#include "internal/common/error_codes_internal.h"     // for NVSHMEMI_INTE...
#include "internal/host/nvshmemi_symmetric_heap.hpp"  // for nvshmemi_symm...
#include "internal/host/nvshmemi_types.h"             // for nvshmemi_state_t
#include "internal/host/util.h"                       // for nvshmemi_options
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for nvshmemi_boot...
#include "bootstrap_host_transport/env_defs_internal.h"                    // for nvshmemi_opti...
#include "internal/host_transport/transport.h"                             // for nvshmem_trans...
#include "non_abi/nvshmem_build_options.h"                                 // for NVSHMEM_IBGDA...
#include "non_abi/nvshmem_version.h"                                       // for NVSHMEM_TRANS...
#include "topo.h"                                                          // for nvshmemi_get_...

#define TRANSPORT_STRING_MAX_LENGTH 16
#define NVSHMEM_TRANSPORT_COUNT 6

static void *transport_lib = nullptr;
#ifdef NVSHMEM_IBGDA_SUPPORT
static void *transport_lib_IBGDA = nullptr;
#endif

static std::once_flag transport_lib_atexit_flag;

static void nvshmemi_transport_lib_fini_wrapper(void) {
    if (transport_lib) {
        dlclose(transport_lib);
        transport_lib = nullptr;
    }
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (transport_lib_IBGDA) {
        dlclose(transport_lib_IBGDA);
        transport_lib_IBGDA = nullptr;
    }
#endif
}

static const char *nvshmemi_device_assignment_mode_name(
    nvshmem_transport_device_assignment_mode_t assignment_mode) {
    switch (assignment_mode) {
        case NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_HCA_LIST:
            return "NVSHMEM_HCA_LIST";
        case NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_HCA_PE_MAPPING:
            return "NVSHMEM_HCA_PE_MAPPING";
        case NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_DEFAULT:
        default:
            return "DEFAULT";
    }
}

static int nvshmemi_get_device_block(int device_count, int entity_count, int entity_index,
                                     int *start, int *count) {
    if (!start || !count) return NVSHMEMX_ERROR_INVALID_VALUE;

    *start = -1;
    *count = 0;

    if (device_count <= 0 || entity_count <= 0 || entity_index < 0 ||
        entity_index >= entity_count) {
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    int block_count = device_count / entity_count;
    int remainder = device_count % entity_count;
    *start = entity_index * block_count + (entity_index < remainder ? entity_index : remainder);
    *count = block_count + (entity_index < remainder ? 1 : 0);

    return NVSHMEMX_SUCCESS;
}

int nvshmemi_transport_show_info(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    for (int i = 0; i < state->num_initialized_transports; ++i) {
        transports[i]->host_ops.show_info(transports[i], TRANSPORT_OPTIONS_STYLE_INFO);
    }
    return status;
}

int nvshmemi_transport_init(nvshmemi_state_t *state) {
    int status = 0;
    int index = 0;
    nvshmem_transport_t *transports = NULL;
    nvshmemi_transport_init_fn init_fn;
    const int transport_object_file_len = 100;
    char transport_object_file[transport_object_file_len];
    const char *transport_name = nullptr;
    nvshmem_local_buf_cache_t *tmp_cache_ptr = NULL;

    std::call_once(transport_lib_atexit_flag,
                   []() { atexit(nvshmemi_transport_lib_fini_wrapper); });

    if (nvshmemi_options.IBGDA_ENABLE_MULTI_PORT_provided) {
        WARN(
            "NVSHMEM_IBGDA_ENABLE_MULTI_PORT is deprecated; use "
            "NVSHMEM_ENABLE_MULTI_PORT instead.\n");
    }
    if (!nvshmemi_options.ENABLE_MULTI_PORT_provided &&
        nvshmemi_options.IBGDA_ENABLE_MULTI_PORT_provided) {
        nvshmemi_options.ENABLE_MULTI_PORT = nvshmemi_options.IBGDA_ENABLE_MULTI_PORT;
    } else if (nvshmemi_options.ENABLE_MULTI_PORT_provided &&
               nvshmemi_options.IBGDA_ENABLE_MULTI_PORT_provided &&
               nvshmemi_options.ENABLE_MULTI_PORT != nvshmemi_options.IBGDA_ENABLE_MULTI_PORT) {
        WARN(
            "NVSHMEM_ENABLE_MULTI_PORT and NVSHMEM_IBGDA_ENABLE_MULTI_PORT disagree; using "
            "NVSHMEM_ENABLE_MULTI_PORT.\n");
    }
    nvshmemi_options.IBGDA_ENABLE_MULTI_PORT = nvshmemi_options.ENABLE_MULTI_PORT;

    const char *multi_port_source =
        nvshmemi_options.ENABLE_MULTI_PORT_provided
            ? "NVSHMEM_ENABLE_MULTI_PORT"
            : (nvshmemi_options.IBGDA_ENABLE_MULTI_PORT_provided ? "NVSHMEM_IBGDA_ENABLE_MULTI_PORT"
                                                                 : "default");
    INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_MULTI_PORT = %d (source: %s)",
         nvshmemi_options.ENABLE_MULTI_PORT, multi_port_source);

    if (!state->transports)
        state->transports =
            (nvshmem_transport_t *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(nvshmem_transport_t));

    transports = (nvshmem_transport_t *)state->transports;

    if (!nvshmemi_options.DISABLE_P2P) {
        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                              "Unable to allocate transport mem cache.\n");
        status = nvshmemt_p2p_init(&transports[index]);
        if (!status) {
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            transports[index]->heap_base = state->heap_obj->get_base();
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->log2_cumem_granularity =
                nvshmemi_state->heap_obj->get_log2_cumem_granularity();
            transports[index]->cache_handle = tmp_cache_ptr;
            // No need for alias VA map for P2P transport
            transports[index]->alias_va_map = nullptr;
            transports[index]->egm_map = nullptr;
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            index++;
        } else {
            nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
            NVSHMEMI_ERROR_PRINT("init failed for transport: P2P");
            /* non-fatal error, so changing to a warning */
            status = 0;
        }
    } else {
        WARN("P2P access was disabled in the environment");
    }

#ifdef NVSHMEM_IBRC_SUPPORT
    if (!transport_name &&
        strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, "ibrc", TRANSPORT_STRING_MAX_LENGTH) == 0)
        transport_name = "ibrc";
#endif
#ifdef NVSHMEM_UCX_SUPPORT
    if (!transport_name &&
        strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, "ucx", TRANSPORT_STRING_MAX_LENGTH) == 0)
        transport_name = "ucx";
#endif
#ifdef NVSHMEM_IBDEVX_SUPPORT
    if (!transport_name &&
        strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, "ibdevx", TRANSPORT_STRING_MAX_LENGTH) == 0)
        transport_name = "ibdevx";
#endif
#ifdef NVSHMEM_LIBFABRIC_SUPPORT
    if (!transport_name && strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, "libfabric",
                                       TRANSPORT_STRING_MAX_LENGTH) == 0)
        transport_name = "libfabric";
#endif
#ifdef NVSHMEM_GPUNETIO_SUPPORT
    if (!transport_name && strncasecmp(nvshmemi_options.REMOTE_TRANSPORT, "gpunetio",
                                       TRANSPORT_STRING_MAX_LENGTH) == 0)
        transport_name = "gpunetio";
#endif

    if (transport_name) {
        INFO(NVSHMEM_INIT, "Selected remote transport: %s", transport_name);
        snprintf(transport_object_file, transport_object_file_len, "nvshmem_transport_%s.so.%d",
                 transport_name, NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);

        transport_lib = dlopen(transport_object_file, RTLD_NOW);
        if (transport_lib == NULL) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
        }
    }

    if (transport_lib) {
        init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib, "nvshmemt_init");
        if (!init_fn) {
            dlclose(transport_lib);
            transport_lib = NULL;
            WARN("Unable to get info from %s transport.\n", transport_object_file);
        }
    }

    if (transport_lib) {
        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                              "Unable to allocate transport mem cache.\n");

        status =
            init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        if (!status) {
            assert(NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(transports[index]->api_version) <=
                   NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(NVSHMEM_TRANSPORT_INTERFACE_VERSION));
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            if (nvshmemi_device_state.enable_rail_opt == 1) {
                transports[index]->heap_base = nvshmemi_state->heap_obj->get_global_base();
            } else {
                transports[index]->heap_base = state->heap_obj->get_base();
            }

            transports[index]->log2_cumem_granularity =
                nvshmemi_state->heap_obj->get_log2_cumem_granularity();
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->my_pe = nvshmemi_state->mype;
            transports[index]->n_pes = nvshmemi_state->npes;
            transports[index]->cache_handle = (void *)tmp_cache_ptr;
            transports[index]->alias_va_map =
                state->vmm_heap ? state->vmm_heap->get_alias_va_map() : nullptr;
            transports[index]->egm_map = state->vmm_heap ? state->vmm_heap->get_egm_map() : nullptr;
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
#ifdef NVSHMEM_GPUNETIO_SUPPORT
            if (strncasecmp(transport_name, "gpunetio", TRANSPORT_STRING_MAX_LENGTH) == 0 &&
                nvshmemi_options.GPUNETIO_ENABLE_GDAKI) {
                nvshmemi_gpunetio_get_device_state(&transports[index]->type_specific_shared_state);
                nvshmemi_device_state.selected_device_transport =
                    NVSHMEMI_DEVICE_TRANSPORT_TYPE_GPUNETIO_GDAKI;
                INFO(NVSHMEM_INIT, "GPUNetIO GDAKI enabled for device-side APIs over IB.");
            }
#endif
            index++;
        } else {
            nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
            dlclose(transport_lib);
            transport_lib = NULL;
            /* non-fatal error, so changing to a warning */
            INFO(NVSHMEM_TRANSPORT, "init failed for remote transport: %s",
                 nvshmemi_options.REMOTE_TRANSPORT);
            status = 0;
        }
    }

#ifdef NVSHMEM_GPUNETIO_SUPPORT
    if (nvshmemi_options.GPUNETIO_ENABLE_GDAKI &&
        (!transport_name ||
         strncasecmp(transport_name, "gpunetio", TRANSPORT_STRING_MAX_LENGTH) != 0)) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM_GPUNETIO_ENABLE_GDAKI=1 requires NVSHMEM_REMOTE_TRANSPORT=gpunetio.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
#endif

#if defined(NVSHMEM_IBGDA_SUPPORT) && defined(NVSHMEM_GPUNETIO_SUPPORT)
    if (nvshmemi_options.IB_ENABLE_IBGDA && nvshmemi_options.GPUNETIO_ENABLE_GDAKI) {
        NVSHMEMI_ERROR_PRINT(
            "IBGDA and GPUNetIO GDAKI cannot be enabled at the same time on the device side.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
#endif

#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_options.IB_ENABLE_IBGDA) {
        status = snprintf(transport_object_file, transport_object_file_len,
                          "nvshmem_transport_ibgda.so.%d", NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION);
        if (status < 0 || status > transport_object_file_len) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
            goto out;
        }
        transport_lib_IBGDA = dlopen(transport_object_file, RTLD_NOW);
        if (transport_lib_IBGDA == NULL) {
            WARN("Unable to open the %s transport. %s\n", transport_object_file, dlerror());
            goto out;
        }

        init_fn = (nvshmemi_transport_init_fn)dlsym(transport_lib_IBGDA, "nvshmemt_init");
        if (!init_fn) {
            dlclose(transport_lib_IBGDA);
            transport_lib_IBGDA = NULL;
            WARN("Unable to get info from %s transport.\n", transport_object_file);
            goto out;
        }

        status = nvshmemi_local_mem_cache_init(&tmp_cache_ptr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMI_INTERNAL_ERROR, out,
                              "Unable to allocate transport mem cache.\n");

        status =
            init_fn(&transports[index], nvshmemi_cuda_syms, NVSHMEM_TRANSPORT_INTERFACE_VERSION);
        if (!status) {
            assert(NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(transports[index]->api_version) <=
                   NVSHMEM_TRANSPORT_MAJOR_MINOR_VERSION(NVSHMEM_TRANSPORT_INTERFACE_VERSION));
            transports[index]->boot_handle = &nvshmemi_boot_handle;
            if (nvshmemi_device_state.enable_rail_opt == 1) {
                transports[index]->heap_base = nvshmemi_state->heap_obj->get_global_base();
            } else {
                transports[index]->heap_base = state->heap_obj->get_base();
            }
            transports[index]->log2_cumem_granularity =
                nvshmemi_state->heap_obj->get_log2_cumem_granularity();
            transports[index]->cap = (int *)calloc(state->npes, sizeof(int));
            transports[index]->index = index;
            transports[index]->my_pe = nvshmemi_state->mype;
            transports[index]->n_pes = nvshmemi_state->npes;
            transports[index]->cache_handle = (void *)tmp_cache_ptr;
            transports[index]->alias_va_map =
                state->vmm_heap ? state->vmm_heap->get_alias_va_map() : nullptr;
            transports[index]->egm_map = state->vmm_heap ? state->vmm_heap->get_egm_map() : nullptr;
            nvshmemi_ibgda_get_device_state(&transports[index]->type_specific_shared_state);
            if (transports[index]->max_op_len == 0) transports[index]->max_op_len = SIZE_MAX;
            state->atomic_host_endian_min_size = transports[index]->atomic_host_endian_min_size;
            nvshmemi_device_state.ibgda_is_initialized = true;
            nvshmemi_device_state.selected_device_transport = NVSHMEMI_DEVICE_TRANSPORT_TYPE_IBGDA;
            index++;
        } else {
            NVSHMEMI_ERROR_PRINT("init failed for transport: IBGDA");
            nvshmemi_local_mem_cache_fini(tmp_cache_ptr);
            dlclose(transport_lib_IBGDA);
            transport_lib_IBGDA = NULL;
            status = 0;
        }
    } else {
        INFO(NVSHMEM_INIT, "IBGDA Disabled by the environment.");
    }
#endif

    if (index == 0) {
        NVSHMEMI_ERROR_PRINT("Unable to initialize any transports. returning error.");
        status = NVSHMEMX_ERROR_INTERNAL;
    }
out:
    state->num_initialized_transports = index;
    if (status > 0) {
        for (int idx = 0; idx < index; idx++) {
            nvshmemi_local_mem_cache_fini(
                (nvshmem_local_buf_cache_t *)transports[idx]->cache_handle);
        }
    } else {
        if (transport_lib) {
            INFO(NVSHMEM_INIT, "Successfully initialized the transport: %s",
                 nvshmemi_options.REMOTE_TRANSPORT);
        }
#ifdef NVSHMEM_IBGDA_SUPPORT
        if (transport_lib_IBGDA) {
            INFO(NVSHMEM_INIT,
                 "Successfully initialized the transport: IBGDA. It will be used for device-side "
                 "APIs over IB.");
        }
#endif
    }
    return status;
}

int nvshmemi_transport_finalize_one(nvshmemi_state_t *state, int transport_id) {
    int status = 0;

    assert(state->transports);
    nvshmem_transport_t transport = state->transports[transport_id];
    if (transport->is_successfully_initialized) {
        if (transport->type == NVSHMEM_TRANSPORT_LIB_CODE_IBGDA) {
            nvshmemi_device_state.ibgda_is_initialized = false;
        }
        if (transport->cache_handle) {
            nvshmemi_transport_buffer_unregister_all(transport);
            nvshmemi_local_mem_cache_fini((nvshmem_local_buf_cache_t *)transport->cache_handle);
        }

        status = transport->host_ops.finalize(transport);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport finalize failed \n");
    }

out:
    status = nvshmemi_update_device_state();
    nvshmemi_state->transport_bitmap &= ~(1 << transport_id);
    state->transports[transport_id] = NULL;

    return status;
}

int nvshmemi_transport_finalize(nvshmemi_state_t *state) {
    INFO(NVSHMEM_INIT, "In nvshmemi_transport_finalize");
    int status = 0;

    if (!state->transports) return 0;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        nvshmemi_transport_finalize_one(state, i);
    }

    // Transport library handles will be dlclosed in the atexit handler

    return status;
}

int nvshmemi_setup_connections(nvshmemi_state_t *state) {
    int status = 0;

    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmem_transport_t tcurr;

    for (int i = 0; i < state->num_initialized_transports; i++) {
        int current_status = 0;
        if (!((state->transport_bitmap) & (1 << i))) continue;
        tcurr = transports[i];

        if (!(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED)) {
            continue;
        }

        bool explicit_hca_assignment =
            tcurr->device_assignment_mode != NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_DEFAULT;
        int assignment_entity_count = nvshmemi_get_netdevs_policy_entity_count(state);
        int max_devices_per_pe = tcurr->n_devices / assignment_entity_count;

        if (nvshmemi_options.ENABLE_NIC_PE_MAPPING && explicit_hca_assignment) {
            assignment_entity_count = state->npes_node > 0 ? state->npes_node : 1;
            max_devices_per_pe =
                (tcurr->n_devices + assignment_entity_count - 1) / assignment_entity_count;
        }

        if (max_devices_per_pe == 0) max_devices_per_pe = 1;

        std::vector<int> selected_devices(max_devices_per_pe, -1);
        int found_devices = 0;

        if (nvshmemi_options.ENABLE_NIC_PE_MAPPING && explicit_hca_assignment &&
            tcurr->n_devices <= 0) {
            NVSHMEMI_ERROR_JMP(current_status, NVSHMEMX_ERROR_INVALID_VALUE, handle_transport_error,
                               "%s resolved no HCA slots; cannot apply explicit HCA PE mapping.\n",
                               nvshmemi_device_assignment_mode_name(tcurr->device_assignment_mode));
        }

        // assumes symmetry of transport list at all PEs
        if (tcurr->n_devices == 0) {
            INFO(NVSHMEM_INIT, "Transport manages device selection internally.");
        } else if (tcurr->n_devices == 1) {
            /* return the index of the only available device. */
            selected_devices[0] = 0;
            found_devices++;
        } else if (nvshmemi_options.ENABLE_NIC_PE_MAPPING) {
            if (explicit_hca_assignment) {
                int entity_count = state->npes_node > 0 ? state->npes_node : 1;
                int entity_index = state->mype_node;
                int block_start = -1;
                int block_count = 0;

                if (entity_index < 0 || entity_index >= entity_count) {
                    NVSHMEMI_ERROR_JMP(
                        current_status, NVSHMEMX_ERROR_INVALID_VALUE, handle_transport_error,
                        "Invalid local PE index %d for explicit HCA assignment entity count %d.\n",
                        entity_index, entity_count);
                }

                bool use_block = nvshmemi_options.ENABLE_MULTI_PORT &&
                                 (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_MULTI_NIC_ENABLED) &&
                                 ((tcurr->device_assignment_mode ==
                                       NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_HCA_LIST &&
                                   tcurr->n_devices >= entity_count) ||
                                  (tcurr->device_assignment_mode ==
                                       NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_HCA_PE_MAPPING &&
                                   tcurr->n_devices > entity_count));
                if (use_block) {
                    int block_devices = tcurr->n_devices;
                    if (tcurr->device_assignment_mode ==
                        NVSHMEM_TRANSPORT_DEVICE_ASSIGNMENT_HCA_LIST) {
                        int ignored_devices = tcurr->n_devices % entity_count;
                        block_devices = tcurr->n_devices - ignored_devices;
                        if (ignored_devices > 0) {
                            WARN(
                                "%s has %d HCA slot(s), which does not divide cleanly across %d "
                                "local PE(s); ignoring the trailing %d slot(s).",
                                nvshmemi_device_assignment_mode_name(tcurr->device_assignment_mode),
                                tcurr->n_devices, entity_count, ignored_devices);
                        }
                    }
                    current_status = nvshmemi_get_device_block(
                        block_devices, entity_count, entity_index, &block_start, &block_count);
                    NVSHMEMI_NZ_ERROR_JMP(current_status, NVSHMEMX_ERROR_INVALID_VALUE,
                                          handle_transport_error,
                                          "Failed to select HCA BLOCK range.\n");
                    for (int j = 0; j < block_count; j++) {
                        selected_devices[j] = block_start + j;
                    }
                    found_devices = block_count;
                } else {
                    selected_devices[0] = entity_index % tcurr->n_devices;
                    found_devices = 1;
                }

                INFO(NVSHMEM_INIT, "%s selected %d logical HCA slot(s) for local PE %d",
                     nvshmemi_device_assignment_mode_name(tcurr->device_assignment_mode),
                     found_devices, entity_index);
                for (int j = 0; j < found_devices; j++) {
                    INFO(NVSHMEM_INIT, "%s connection slot %d uses dev_id = %d",
                         nvshmemi_device_assignment_mode_name(tcurr->device_assignment_mode), j,
                         selected_devices[j]);
                }
            } else {
                selected_devices[0] =
                    nvshmemi_state->mype_node % (tcurr->n_devices > 0 ? tcurr->n_devices : 1);
                INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_NIC_PE_MAPPING = 1, setting dev_id = %d",
                     selected_devices[0]);
                found_devices++;
            }
        } else {
            current_status = nvshmemi_get_devices_by_distance(selected_devices.data(),
                                                              max_devices_per_pe, tcurr);
            NVSHMEMI_NZ_ERROR_JMP(current_status, NVSHMEMX_ERROR_INTERNAL, handle_transport_error,
                                  "get devices by distance failed \n");
            for (int i = 0; i < max_devices_per_pe; i++) {
                if (selected_devices[i] == -1) {
                    break;
                }
                found_devices++;
                INFO(NVSHMEM_INIT,
                     "NVSHMEM_ENABLE_NIC_PE_MAPPING = 0, device %d setting dev_id = %d", i,
                     selected_devices[i]);
            }
        }

        /* setting n_devices to 0 is the transports way of
         * letting us know it's managing devices internally.
         */
        if (tcurr->n_devices > 0 && selected_devices[0] == -1) {
            NVSHMEMI_ERROR_JMP(current_status, NVSHMEMX_ERROR_INTERNAL, handle_transport_error,
                               "No devices selected.\n");
        }

        if (!nvshmemi_options.ENABLE_MULTI_PORT && found_devices > 1) {
            INFO(NVSHMEM_INIT,
                 "NVSHMEM_ENABLE_MULTI_PORT = 0; using dev_id = %d and ignoring %d additional "
                 "selected NIC(s).",
                 selected_devices[0], found_devices - 1);
            found_devices = 1;
        }

        current_status = tcurr->host_ops.connect_endpoints(
            tcurr, found_devices > 0 ? selected_devices.data() : NULL, found_devices, NULL, 0);
        NVSHMEMI_NZ_ERROR_JMP(current_status, NVSHMEMX_ERROR_INTERNAL, handle_transport_error,
                              "connect EPS failed \n");

    handle_transport_error:
        int barrier_stat = nvshmemi_boot_handle.barrier(&nvshmemi_boot_handle);

        if (barrier_stat != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT("barrier failed\n");

            if (current_status == NVSHMEMX_SUCCESS) {
                current_status = barrier_stat;
            }
        }

        if (current_status != NVSHMEMX_SUCCESS) {
            nvshmemi_transport_finalize_one(state, i);
            status |= current_status;
        } else {
            status |= nvshmemi_update_device_state();
        }
    }

    return status;
}
