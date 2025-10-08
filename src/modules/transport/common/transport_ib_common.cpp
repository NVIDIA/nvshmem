/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#include "transport_ib_common.h"
#include <assert.h>            // for assert
#include <cuda.h>              // for CUdeviceptr, CU_MEM_RA...
#include <cuda_runtime.h>      // for cudaGetLastError, cuda...
#include <dlfcn.h>             // for dlclose, dlopen, RTLD_...
#include <driver_types.h>      // for cudaPointerAttributes
#include <errno.h>             // for errno
#include <infiniband/verbs.h>  // for IBV_ACCESS_LOCAL_WRITE
#include <stdint.h>            // for uintptr_t, uint64_t
#include <string.h>            // for strerror
#include <unistd.h>            // for access, close, sysconf
#include "device_host_transport/nvshmem_constants.h"
#include "internal/host_transport/cudawrap.h"  // for nvshmemi_cuda_fn_table
#include "non_abi/nvshmemx_error.h"            // for NVSHMEMX_ERROR_INTERNAL
#include "non_abi/nvshmem_build_options.h"     // for NVSHMEM_USE_MLX5DV
#include "transport_common.h"                  // for LOAD_SYM, INFO, MAXPAT...

int nvshmemt_ib_common_nv_peer_mem_available() {
    if (access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/kernel/mm/memory_peers/nvidia-peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }
    if (access("/sys/module/nvidia_peermem/version", F_OK) == 0) {
        return NVSHMEMX_SUCCESS;
    }

    return NVSHMEMX_ERROR_INTERNAL;
}

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                      struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                      struct ibv_pd *pd, nvshmem_mem_handle_t *mem_handle,
                                      void *buf, size_t length, bool local_only,
                                      bool dmabuf_support, struct nvshmemi_cuda_fn_table *table,
                                      int log_level, bool relaxed_ordering, bool is_data_direct,
                                      void *alias_va_ptr) {
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    struct ibv_mr *mr = NULL;
    int status = 0;
    int ro_flag = 0;
    bool host_memory = false;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    cudaPointerAttributes attr;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        host_memory = true;
        status = 0;
        cudaGetLastError();
    } else if (attr.type != cudaMemoryTypeDevice) {
        host_memory = true;
    }

#if defined(HAVE_IBV_ACCESS_RELAXED_ORDERING)
#if HAVE_IBV_ACCESS_RELAXED_ORDERING == 1
    // IBV_ACCESS_RELAXED_ORDERING has been introduced to rdma-core since v28.0.
    if (relaxed_ordering) {
        ro_flag = IBV_ACCESS_RELAXED_ORDERING;
    }
#endif
#endif

    if (ftable->reg_dmabuf_mr != NULL && !host_memory && dmabuf_support &&
        CUPFN(table, cuMemGetHandleForAddressRange)) {
        size_t page_size = sysconf(_SC_PAGESIZE);
        size_t size_aligned;
        int handle_flag = is_data_direct ? CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE : 0;
        CUdeviceptr p;
        p = (CUdeviceptr)((uintptr_t)buf & ~(page_size - 1));
        size_aligned =
            ((length + (uintptr_t)buf - (uintptr_t)p + page_size - 1) / page_size) * page_size;

        CUCHECKGOTO(table,
                    cuMemGetHandleForAddressRange(&handle->fd, (CUdeviceptr)p, size_aligned,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, handle_flag),
                    status, out);
        if (is_data_direct) {
            NVSHMEMI_NULL_ERROR_JMP(mlx5dv_ftable, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                    "mlx5dv_ftable is NULL with data direct enabled\n");
            mr = mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr(
                pd, 0, size_aligned, (uint64_t)p, handle->fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_ATOMIC | ro_flag,
                MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
            if (mr == NULL) {
                close(handle->fd);
                goto reg_dmabuf_failure;
            }
            INFO(log_level, "mlx5dv_reg_dmabuf_mr handle %p mr %p", handle, mr);
        } else {
            mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                                           ro_flag);
            if (mr == NULL) {
                close(handle->fd);
                goto reg_dmabuf_failure;
            }
            INFO(log_level, "ibv_reg_dmabuf_mr handle %p mr %p", handle, mr);
        }
    } else {
    reg_dmabuf_failure:

        handle->fd = 0;
        if (alias_va_ptr) {
            mr = ftable->reg_mr_iova(pd, alias_va_ptr, length, (uint64_t)buf,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                                         ro_flag);
            INFO(log_level, "ibv_reg_mr_iova handle %p mr %p", handle, mr);
        } else {
            mr = ftable->reg_mr(pd, buf, length,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC | ro_flag);
            INFO(log_level, "ibv_reg_mr handle %p mr %p", handle, mr);
        }

        NVSHMEMI_NULL_ERROR_JMP(mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed. Reason: %s\n", strerror(errno));
    }

    handle->buf = buf;
    handle->lkey = mr->lkey;
    handle->rkey = mr->rkey;
    handle->mr = mr;
    handle->local_only = local_only;

out:
    return status;
}

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level) {
    int status = 0;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    INFO(log_level, "ibv_dereg_mr handle %p handle->mr %p", handle, handle->mr);
    if (handle->mr) {
        status = ftable->dereg_mr((struct ibv_mr *)handle->mr);
        if (handle->fd) close(handle->fd);
    }
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");

out:
    return status;
}

bool nvshmemt_mlx5dv_dmabuf_capable(ibv_context *context,
                                    struct nvshmemt_ibv_function_table *ftable,
                                    struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable) {
    int status = 0;
    int dev_fail = 0;
    struct ibv_pd *pd = ftable->alloc_pd(context);
    NVSHMEMI_NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_alloc_pd failed \n");

    if (mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr == NULL) {
        errno = EOPNOTSUPP;
    } else {
        mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                                     0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/,
                                                     0 /* mlx5 flags*/);
        // mlx5dv_reg_dmabuf_mr() will fail with EOPNOTSUPP/EPROTONOSUPPORT if not supported (EBADF
        // otherwise)
    }

    dev_fail |= (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);

    status = ftable->dealloc_pd(pd);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dealloc_pd failed \n");
    if (dev_fail) goto out;
    return true;
out:
    return false;
}

int nvshmemt_ib_common_check_poll_avail(nvshmem_transport_t tcurr, nvshmemt_ib_common_ep_ptr_t ep,
                                        nvshmemt_ib_wait_predicate_t wait_predicate) {
    int status = 0;
    uint32_t outstanding_count;
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)tcurr->state;
    struct nvshmemt_ib_common_ep *common_ep = (struct nvshmemt_ib_common_ep *)ep;

    assert(ib_state->qp_depth > 1);
    if (wait_predicate == NVSHMEMT_IB_COMMON_WAIT_ANY) {
        outstanding_count = (ib_state->qp_depth - 1);
    } else if (wait_predicate == NVSHMEMT_IB_COMMON_WAIT_TWO) {
        outstanding_count = (ib_state->qp_depth - 2);
    } else if (wait_predicate == NVSHMEMT_IB_COMMON_WAIT_ALL) {
        outstanding_count = 0;
    } else {
        outstanding_count = common_ep->head_op_id - common_ep->tail_op_id;
    }

    /* poll until space becomes available in local send qp */
    while (((common_ep->head_op_id - common_ep->tail_op_id) > outstanding_count)) {
        /* *second argument is a noop for now. */
        status = ib_state->ib_transport_ftable->progress(tcurr, true);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "progress_send failed, outstanding_count: %d\n", outstanding_count);
    }

    if (ib_state->ib_transport_ftable->progress_recv) {
        status = ib_state->ib_transport_ftable->progress_recv(tcurr, wait_predicate);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_recv failed \n");
    }

out:
    return status;
}

int nvshmemt_ib_common_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index) {
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)tcurr->state;
    nvshmemt_ib_common_ep_ptr_t ep;
    int status = 0;
    int n_pes = tcurr->n_pes;

    if (pe == NVSHMEMX_PE_ANY) {
        /* Loop over all PEs */
        for (int pe_idx = 0; pe_idx < n_pes; pe_idx++) {
            status = nvshmemt_ib_common_quiet(tcurr, pe_idx, qp_index);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "quiet failed for PE %d \n",
                                  pe_idx);
        }
        return status;
    }

    if (qp_index == NVSHMEMX_QP_DEFAULT) {
        /* Loop over all default QPs for this PE */
        int default_qp_count = ib_state->options->IB_NUM_RC_PER_DEVICE;
        for (int qp = 0; qp < default_qp_count; qp++) {
            ep = nvshmemt_ib_common_get_ep_from_qp_index(tcurr, qp, pe);
            if (ep) {
                status =
                    nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
            }
        }
    } else if (qp_index == NVSHMEMX_QP_ANY || qp_index == NVSHMEMX_QP_ALL) {
        /* Loop over all QPs for this PE */
        int total_qps = ib_state->next_qp_index;
        for (int qp = 0; qp < total_qps; qp++) {
            ep = nvshmemt_ib_common_get_ep_from_qp_index(tcurr, qp, pe);
            if (ep) {
                status =
                    nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
            }
        }
    } else {
        /* Single QP */
        ep = nvshmemt_ib_common_get_ep_from_qp_index(tcurr, qp_index, pe);
        if (ep) {
            status = nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
        }
    }

out:
    return status;
}

int nvshmemt_ib_common_fence(nvshmem_transport_t tcurr, int pe, int qp_index, int is_multi) {
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)tcurr->state;
    int status = 0;
    int n_pes = tcurr->n_pes;
    bool multiple_qps = false;

    if (pe == NVSHMEMX_PE_ANY) {
        /* Loop over all PEs */
        for (int pe_idx = 0; pe_idx < n_pes; pe_idx++) {
            status = nvshmemt_ib_common_fence(tcurr, pe_idx, qp_index, is_multi);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "fence failed for PE %d \n",
                                  pe_idx);
        }
        return status;
    }

    /* Check if this will result in fencing on more than one QP per PE */
    if (qp_index == NVSHMEMX_QP_DEFAULT) {
        multiple_qps = (ib_state->options->IB_NUM_RC_PER_DEVICE > 1);
    } else if (qp_index == NVSHMEMX_QP_ANY || qp_index == NVSHMEMX_QP_ALL) {
        multiple_qps = (ib_state->next_qp_index > 1);
    }

    if (multiple_qps || is_multi) {
        /* Call quiet to ensure all operations complete before fencing */
        status = nvshmemt_ib_common_quiet(tcurr, pe, qp_index);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "quiet failed \n");
    }

out:
    return status;
}

int nvshmemt_ib_common_setup_cst_loopback(int dev_id, nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)t->state;
    struct nvshmemt_ib_common_ep_handle cst_ep_handle;

    status = ib_state->ib_transport_ftable->ep_create(&ib_state->cst_ep, dev_id, t);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create cst failed \n");

    status = ib_state->ib_transport_ftable->ep_get_handle(&cst_ep_handle, ib_state->cst_ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

    status = ib_state->ib_transport_ftable->ep_connect(ib_state->cst_ep, &cst_ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");
out:
    return status;
}

int nvshmemt_ib_common_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                         int num_selected_devs, int *out_qp_indices, int num_qps) {
    /* transport side */
    struct nvshmemt_ib_common_ep_handle *local_ep_handles = NULL, *ep_handles = NULL;
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)t->state;
    int n_pes = t->n_pes;
    int status = 0;
    int first_ep_idx;
    bool is_initial_call = (ib_state->ep == NULL);

    /* Calculate ep_count and total_eps based on call type */
    int ep_count, total_eps, qps_to_create;

    if (is_initial_call) {
        /* Allow user to override IB_NUM_RC_PER_DEVICE if num_qps is provided */
        if (num_qps > 0) {
            ep_count = num_qps + 1; /* +1 for host EP */
        } else {
            ep_count = ib_state->options->IB_NUM_RC_PER_DEVICE + 1;
        }
        total_eps = n_pes * ep_count;
        qps_to_create = ep_count;
        ib_state->host_ep_index = NVSHMEMX_QP_HOST;
        ib_state->selected_dev_id = selected_dev_ids[0];
        ib_state->cur_ep_index = NVSHMEMX_QP_DEFAULT;
        /* The initial call to connect endpoints will increment this to the first needed index*/
        ib_state->next_qp_index = 0;
        ib_state->cur_default_qp_index = NVSHMEMX_QP_DEFAULT;
        ib_state->cur_any_qp_index = 0;

        /* Allocate ep array for initial call */
        ib_state->ep =
            (nvshmemt_ib_common_ep_ptr_t *)calloc(total_eps, sizeof(nvshmemt_ib_common_ep_ptr_t));
        NVSHMEMI_NULL_ERROR_JMP(ib_state->ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "failed allocating space for endpoints \n");
        ib_state->ep_count = total_eps;
    } else {
        assert(out_qp_indices != NULL);
        ep_count = num_qps;
        total_eps = n_pes * num_qps;
        qps_to_create = num_qps;

        /* Reallocate ep array for additional QPs */
        int new_total_eps = ib_state->ep_count + total_eps;
        nvshmemt_ib_common_ep_ptr_t *new_ep_array = (nvshmemt_ib_common_ep_ptr_t *)realloc(
            ib_state->ep, new_total_eps * sizeof(nvshmemt_ib_common_ep_ptr_t));
        if (!new_ep_array) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                               "Failed to reallocate ep array\n");
        }
        ib_state->ep = new_ep_array;
        ib_state->ep_count = new_total_eps;
    }

    /* Allocate handles */
    local_ep_handles = (struct nvshmemt_ib_common_ep_handle *)calloc(
        total_eps, sizeof(struct nvshmemt_ib_common_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(local_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for local ep handles \n");

    ep_handles = (struct nvshmemt_ib_common_ep_handle *)calloc(
        total_eps, sizeof(struct nvshmemt_ib_common_ep_handle));
    NVSHMEMI_NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "failed allocating space for ep handles \n");

    /* Create endpoints */
    first_ep_idx = ib_state->next_qp_index / n_pes;
    for (int i = first_ep_idx; i < first_ep_idx + qps_to_create; i++) {
        for (int j = 0; j < n_pes; j++) {
            int ep_idx = i * n_pes + j;
            int handle_idx = j * qps_to_create + (i - first_ep_idx);

            status = ib_state->ib_transport_ftable->ep_create(&ib_state->ep[ep_idx],
                                                              ib_state->selected_dev_id, t);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create ep failed \n");
            status = ib_state->ib_transport_ftable->ep_get_handle(&local_ep_handles[handle_idx],
                                                                  ib_state->ep[ep_idx]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport get ep handle failed \n");
        }
    }

    /* Exchange handles */
    status = t->boot_handle->alltoall((void *)local_ep_handles, (void *)ep_handles,
                                      sizeof(struct nvshmemt_ib_common_ep_handle) * qps_to_create,
                                      t->boot_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "allgather of ep handles failed \n");

    /* Connect endpoints */
    for (int i = first_ep_idx; i < first_ep_idx + qps_to_create; i++) {
        for (int j = 0; j < n_pes; j++) {
            int ep_idx = i * n_pes + j;
            int handle_idx = j * qps_to_create + (i - first_ep_idx);

            status = ib_state->ib_transport_ftable->ep_connect(ib_state->ep[ep_idx],
                                                               &ep_handles[handle_idx]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "transport create connect failed \n");
        }
    }

    /* Populate out_qp_indices array with QP numbers */
    if (out_qp_indices != NULL) {
        for (int i = 0; i < qps_to_create; i++) {
            out_qp_indices[i] = ib_state->next_qp_index;
            ib_state->next_qp_index += n_pes;
        }
    } else {
        ib_state->next_qp_index += qps_to_create * n_pes;
    }

out:
    if (status) {
        if (is_initial_call) {
            ib_state->selected_dev_id = -1;
            if (ib_state->ep) {
                free(ib_state->ep);
                ib_state->ep = NULL;
            }
        }
        if (local_ep_handles) free(local_ep_handles);
        if (ep_handles) free(ep_handles);
    }
    return status;
}

nvshmemt_ib_common_ep_ptr_t nvshmemt_ib_common_get_ep_from_qp_index(nvshmem_transport_t t,
                                                                    int qp_index, int pe_index) {
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)t->state;

    if (qp_index == NVSHMEMX_QP_HOST) {
        return ib_state->ep[ib_state->host_ep_index + pe_index];
    } else if (qp_index == NVSHMEMX_QP_DEFAULT) {
        /* Round robin over default QPs */
        int default_qp_count = ib_state->options->IB_NUM_RC_PER_DEVICE;
        int selected_qp = ib_state->cur_default_qp_index % (default_qp_count + 1);
        assert(ib_state->cur_default_qp_index != NVSHMEMX_QP_HOST);
        int next_qp_index = (ib_state->cur_default_qp_index + 1) % (default_qp_count + 1);
        if (next_qp_index == NVSHMEMX_QP_HOST) {
            next_qp_index = NVSHMEMX_QP_DEFAULT;
        }
        ib_state->cur_default_qp_index = next_qp_index;
        return ib_state->ep[selected_qp * t->n_pes + pe_index];
    } else if (qp_index == NVSHMEMX_QP_ANY) {
        /* Round robin over all QPs except host */
        /* QP indices are strided by the total number of PEs so we need to take that into account.
         */
        int total_qps = ib_state->next_qp_index / t->n_pes;
        int selected_qp = ib_state->cur_any_qp_index % total_qps;
        assert(ib_state->cur_any_qp_index != NVSHMEMX_QP_HOST);
        ib_state->cur_any_qp_index = (ib_state->cur_any_qp_index + 1) % total_qps;
        if (ib_state->cur_any_qp_index == NVSHMEMX_QP_HOST) {
            ib_state->cur_any_qp_index = NVSHMEMX_QP_DEFAULT;
        }
        return ib_state->ep[selected_qp * t->n_pes + pe_index];
    } else if (qp_index > NVSHMEMX_QP_DEFAULT && qp_index < ib_state->next_qp_index) {
        /* qp indices greater than the default qp index are strided by the total number of pes */
        return ib_state->ep[qp_index + pe_index];
    }

    return NULL;
}

int nvshmemt_ib_iface_get_mlx_path(ibv_device *dev, ibv_context *ctx, char **path,
                                   struct nvshmemt_ibv_function_table *ftable,
                                   struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                   bool *is_data_direct, int log_level) {
    int status;
    char device_path[MAXPATHSIZE];

    *is_data_direct = false;
#ifdef NVSHMEM_USE_MLX5DV
    if (mlx5dv_ftable->mlx5dv_internal_is_supported &&
        mlx5dv_ftable->mlx5dv_internal_is_supported(dev)) {
        snprintf(device_path, MAXPATHSIZE, "/sys");
        if ((nvshmemt_mlx5dv_dmabuf_capable(ctx, ftable, mlx5dv_ftable)) &&
            (!mlx5dv_ftable->mlx5dv_internal_get_data_direct_sysfs_path(ctx, device_path + 4,
                                                                        MAXPATHSIZE - 4))) {
            *is_data_direct = true;
            INFO(log_level, "directNIC features supported and enabled in device %s %s", dev->name,
                 device_path);
            status = NVSHMEMX_SUCCESS;
        }
    }
#endif
    if (!*is_data_direct) {
        status = snprintf(device_path, MAXPATHSIZE, "/sys/class/infiniband/%s/device",
                          (const char *)dev->name);
        if (status < 0 || status >= MAXPATHSIZE) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "Unable to fill in device name.\n");
        } else {
            status = NVSHMEMX_SUCCESS;
        }
    }

    *path = realpath(device_path, NULL);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level) {
    *ibv_handle = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (*ibv_handle == NULL) {
        INFO(log_level, "libibverbs not found on the system.");
        return -1;
    }

    LOAD_SYM(*ibv_handle, "ibv_fork_init", ftable->fork_init);
    LOAD_SYM(*ibv_handle, "ibv_create_ah", ftable->create_ah);
    LOAD_SYM(*ibv_handle, "ibv_get_device_list", ftable->get_device_list);
    LOAD_SYM(*ibv_handle, "ibv_get_device_name", ftable->get_device_name);
    LOAD_SYM(*ibv_handle, "ibv_open_device", ftable->open_device);
    LOAD_SYM(*ibv_handle, "ibv_close_device", ftable->close_device);
    LOAD_SYM(*ibv_handle, "ibv_query_port", ftable->query_port);
    LOAD_SYM(*ibv_handle, "ibv_query_device", ftable->query_device);
    LOAD_SYM(*ibv_handle, "ibv_alloc_pd", ftable->alloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr", ftable->reg_mr);
    LOAD_SYM(*ibv_handle, "ibv_reg_mr_iova", ftable->reg_mr_iova);
    LOAD_SYM(*ibv_handle, "ibv_reg_dmabuf_mr", ftable->reg_dmabuf_mr);
    LOAD_SYM(*ibv_handle, "ibv_dereg_mr", ftable->dereg_mr);
    LOAD_SYM(*ibv_handle, "ibv_create_cq", ftable->create_cq);
    LOAD_SYM(*ibv_handle, "ibv_create_qp", ftable->create_qp);
    LOAD_SYM(*ibv_handle, "ibv_create_srq", ftable->create_srq);
    LOAD_SYM(*ibv_handle, "ibv_modify_qp", ftable->modify_qp);
    LOAD_SYM(*ibv_handle, "ibv_query_gid", ftable->query_gid);
    LOAD_SYM(*ibv_handle, "ibv_dealloc_pd", ftable->dealloc_pd);
    LOAD_SYM(*ibv_handle, "ibv_destroy_qp", ftable->destroy_qp);
    LOAD_SYM(*ibv_handle, "ibv_destroy_cq", ftable->destroy_cq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_srq", ftable->destroy_srq);
    LOAD_SYM(*ibv_handle, "ibv_destroy_ah", ftable->destroy_ah);

    return 0;
}

int nvshmemt_mlx5dv_ftable_init(void **mlx5dv_handle, struct nvshmemt_mlx5dv_function_table *ftable,
                                int log_level) {
    *mlx5dv_handle = dlopen("libmlx5.so", RTLD_LAZY);
    if (*mlx5dv_handle == NULL) {
        *mlx5dv_handle = dlopen("libmlx5.so.1", RTLD_LAZY);
    }
    if (*mlx5dv_handle == NULL) {
        INFO(log_level, "Failed to open libmlx5.so[.1]");
        ftable->mlx5dv_internal_is_supported = NULL;
        ftable->mlx5dv_internal_get_data_direct_sysfs_path = NULL;
        ftable->mlx5dv_internal_reg_dmabuf_mr = NULL;
        return -1;
    }
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_is_supported", ftable->mlx5dv_internal_is_supported,
                     MLX5DV_VERSION);
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_get_data_direct_sysfs_path",
                     ftable->mlx5dv_internal_get_data_direct_sysfs_path, "MLX5_1.25");
    LOAD_SYM_VERSION(*mlx5dv_handle, "mlx5dv_reg_dmabuf_mr", ftable->mlx5dv_internal_reg_dmabuf_mr,
                     "MLX5_1.25");

    return 0;
}

void nvshmemt_ibv_ftable_fini(void **ibv_handle) {
    int status;

    if (ibv_handle) {
        status = dlclose(*ibv_handle);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to close libibverbs handle.");
        }
    }
}

void nvshmemt_mlx5dv_ftable_fini(void **mlx5dv_handle) {
    int status;

    if (mlx5dv_handle) {
        status = dlclose(*mlx5dv_handle);
        if (status) {
            NVSHMEMI_ERROR_PRINT("Unable to close libmlx5dv handle.");
        }
    }
}
