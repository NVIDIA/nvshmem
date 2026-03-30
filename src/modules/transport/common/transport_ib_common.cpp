/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

static void *ibv_lib_handle = nullptr;
static void *mlx5_lib_handle = nullptr;

/*
 * Dynamic GID detection for RoCE platforms
 * Adapted from NCCL: https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/359
 */

static sa_family_t env_ib_addr_family(int log_level, nvshmemi_options_s *options) {
    sa_family_t family = AF_INET;
    const char *env = options->IB_ADDR_FAMILY;
    if (env == NULL || strlen(env) == 0) {
        return family;
    }

    INFO(log_level, "NVSHMEM_IB_ADDR_FAMILY set by environment to %s", env);

    if (strcmp(env, "AF_INET") == 0) {
        family = AF_INET;
    } else if (strcmp(env, "AF_INET6") == 0) {
        family = AF_INET6;
    }

    return family;
}

static void *env_ib_addr_range(sa_family_t af, int *prefix_len, int log_level,
                               nvshmemi_options_s *options) {
    *prefix_len = 0;
    static struct in_addr addr;
    static struct in6_addr addr6;
    void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

    const char *env = options->IB_ADDR_RANGE;
    if (NULL == env || strlen(env) == 0) {
        return NULL;
    }

    INFO(log_level, "NVSHMEM_IB_ADDR_RANGE set by environment to %s", env);

    char addr_string[128] = {0};
    snprintf(addr_string, 128, "%s", env);
    char *addr_str_ptr = addr_string;
    char *slash_ptr = strstr(addr_string, "/");
    if (slash_ptr == NULL) {
        return NULL;
    }
    *slash_ptr = '\0';
    char *mask_str_ptr = slash_ptr + 1;

    if (inet_pton(af, addr_str_ptr, ret) == 0) {
        INFO(log_level, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address",
             addr_str_ptr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        return NULL;
    }

    *prefix_len = (int)strtol(mask_str_ptr, NULL, 10);
    if (af == AF_INET && *prefix_len > 32) {
        INFO(log_level, "IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
             *prefix_len, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        *prefix_len = 0;
        ret = NULL;
    } else if (af == AF_INET6 && *prefix_len > 128) {
        INFO(log_level, "IB: Ip address mask '%d' is invalid for family %s, ignoring mask",
             *prefix_len, (af == AF_INET) ? "AF_INET" : "AF_INET6");
        *prefix_len = 0;
        ret = NULL;
    }

    return ret;
}

static sa_family_t get_gid_addr_family(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    bool is_ipv4_mapped =
        ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
    bool is_ipv4_mapped_multicast =
        (a->s6_addr32[0] == htonl(0xff0e0000) &&
         ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
    return (is_ipv4_mapped || is_ipv4_mapped_multicast) ? AF_INET : AF_INET6;
}

static bool match_gid_addr_prefix(sa_family_t af, void *prefix, int prefix_len,
                                  union ibv_gid *gid) {
    struct in_addr *base = NULL;
    struct in6_addr *base6 = NULL;
    struct in6_addr *addr6 = NULL;

    if (af == AF_INET) {
        base = (struct in_addr *)prefix;
    } else {
        base6 = (struct in6_addr *)prefix;
    }
    addr6 = (struct in6_addr *)gid->raw;

    int i = 0;
    while (prefix_len > 0 && i < 4) {
        if (af == AF_INET) {
            int mask = NETMASK(prefix_len);
            if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
                break;
            }
            prefix_len = 0;
            break;
        } else {
            if (prefix_len >= 32) {
                if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
                    break;
                }
                prefix_len -= 32;
                ++i;
            } else {
                int mask = NETMASK(prefix_len);
                if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
                    break;
                }
                prefix_len = 0;
            }
        }
    }

    return (prefix_len == 0) ? true : false;
}

static bool configured_gid(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
    if (((a->s6_addr32[0] | trailer) == 0UL) ||
        ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
        return false;
    }
    return true;
}

static bool link_local_gid(union ibv_gid *gid) {
    const struct in6_addr *a = (struct in6_addr *)gid->raw;
    if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
        return true;
    }
    return false;
}

static bool valid_gid(union ibv_gid *gid) { return (configured_gid(gid) && !link_local_gid(gid)); }

int ib_roce_get_version_num(const char *deviceName, int portNum, int gidIndex, int *version) {
    char gidRoceVerStr[16] = {0};
    char roceTypePath[PATH_MAX] = {0};
    sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName,
            portNum, gidIndex);

    int fd = open(roceTypePath, O_RDONLY);
    if (fd == -1) {
        NVSHMEMI_WARN_PRINT("IB: open failed in ib_roce_get_version_num: %s", strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }
    int ret = read(fd, gidRoceVerStr, 15);
    close(fd);

    if (ret == -1) {
        // In containerized environments, read could return EINVAL if the GID index is not mapped to
        // the container sysfs. In this case return NVSHMEMX_SUCCESS and let the caller move to next
        // GID index.
        if (errno == EINVAL) return NVSHMEMX_SUCCESS;
        NVSHMEMI_WARN_PRINT("IB: read failed in ib_roce_get_version_num: %s", strerror(errno));
        return NVSHMEMX_ERROR_INTERNAL;
    }

    if (strlen(gidRoceVerStr)) {
        if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 ||
            strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
            *version = 1;
        } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
            *version = 2;
        }
    }

    return NVSHMEMX_SUCCESS;
}

static void update_gid_index(const struct nvshmemt_ibv_function_table *ftable,
                             struct ibv_context *context, uint8_t portNum, sa_family_t af,
                             void *prefix, int prefixlen, int roceVer, int gidIndexCandidate,
                             int *gidIndex) {
    union ibv_gid gid, gidCandidate;
    ftable->query_gid(context, portNum, *gidIndex, &gid);
    ftable->query_gid(context, portNum, gidIndexCandidate, &gidCandidate);

    sa_family_t usrFam = af;
    sa_family_t gidFam = get_gid_addr_family(&gid);
    sa_family_t gidCandidateFam = get_gid_addr_family(&gidCandidate);
    bool gidCandidateMatchSubnet = match_gid_addr_prefix(usrFam, prefix, prefixlen, &gidCandidate);

    if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
        *gidIndex = gidIndexCandidate;
    } else {
        if (gidCandidateFam != usrFam || !valid_gid(&gidCandidate) || !gidCandidateMatchSubnet) {
            return;
        }
        int usrRoceVer = roceVer;
        int gidRoceVerNum, gidRoceVerNumCandidate = -1;
        const char *deviceName = ftable->get_device_name(context->device);
        ib_roce_get_version_num(deviceName, portNum, *gidIndex, &gidRoceVerNum);
        ib_roce_get_version_num(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate);
        if ((gidRoceVerNum != gidRoceVerNumCandidate || !valid_gid(&gid)) &&
            gidRoceVerNumCandidate == usrRoceVer) {
            *gidIndex = gidIndexCandidate;
        }
    }
}

void ib_get_gid_index(const struct nvshmemt_ibv_function_table *ftable, struct ibv_context *context,
                      uint8_t portNum, int gidTblLen, int *gidIndex, int log_level,
                      nvshmemi_options_s *options) {
    *gidIndex = options->IB_GID_INDEX;
    if (*gidIndex >= 0) {
        return;
    }

    sa_family_t userAddrFamily = env_ib_addr_family(log_level, options);
    int userRoceVersion = options->IB_ROCE_VERSION_NUM;
    int prefixlen;
    void *prefix = env_ib_addr_range(userAddrFamily, &prefixlen, log_level, options);

    *gidIndex = 0;
    for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
        update_gid_index(ftable, context, portNum, userAddrFamily, prefix, prefixlen,
                         userRoceVersion, gidIndexNext, gidIndex);
    }
}

/* =============================================================================
 * End dynamic GID detection
 * ============================================================================= */

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
    struct ibv_mr *mr = nullptr;
    int status = 0;
    int ro_flag = 0;
    bool host_memory = false;

    assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE);

    cudaPointerAttributes attr;
    status = cudaPointerGetAttributes(&attr, buf);
    if (status != cudaSuccess) {
        NVSHMEMI_ERROR_PRINT("cudaPointerGetAttributes failed\n");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    if (attr.type != cudaMemoryTypeDevice) {
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

    if (ftable->reg_dmabuf_mr != nullptr && !host_memory && dmabuf_support &&
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
            if (mr == nullptr) {
                close(handle->fd);
                goto reg_dmabuf_failure;
            }
            INFO(log_level, "mlx5dv_reg_dmabuf_mr handle %p mr %p", handle, mr);
        } else {
            mr = ftable->reg_dmabuf_mr(pd, 0, size_aligned, (uint64_t)p, handle->fd,
                                       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC |
                                           ro_flag);
            if (mr == nullptr) {
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
                                    const struct nvshmemt_ibv_function_table *ftable,
                                    const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable) {
    int status = 0;
    int dev_fail = 0;
    struct ibv_pd *pd = ftable->alloc_pd(context);
    NVSHMEMI_NULL_ERROR_JMP(pd, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_alloc_pd failed \n");

    if (mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr == nullptr) {
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
        status = ib_state->ib_transport_ftable->progress(tcurr);
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
            ep = ib_state->ep[(qp + 1) * n_pes + pe];
            if (ep) {
                status =
                    nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
            }
        }
    } else if (qp_index == NVSHMEMX_QP_ANY || qp_index == NVSHMEMX_QP_ALL) {
        /* Loop over all QPs for this PE */
        int total_qps = ib_state->next_qp_index / n_pes;
        for (int qp = 1; qp < total_qps; qp++) {
            ep = ib_state->ep[qp * n_pes + pe];
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
                                         int /*num_selected_devs*/, int *out_qp_indices,
                                         int num_qps) {
    /* transport side */
    struct nvshmemt_ib_common_ep_handle *local_ep_handles = nullptr, *ep_handles = nullptr;
    nvshmemt_ib_common_state_t ib_state = (nvshmemt_ib_common_state_t)t->state;
    int n_pes = t->n_pes;
    int status = 0;
    int first_ep_idx;
    bool is_initial_call = (ib_state->ep == nullptr);

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
        assert(out_qp_indices != nullptr);
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
    if (out_qp_indices != nullptr) {
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
                ib_state->ep = nullptr;
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

    return nullptr;
}

int nvshmemt_ib_iface_get_mlx_path(ibv_device *dev, ibv_context *ctx, char **path,
                                   const struct nvshmemt_ibv_function_table *ftable,
                                   const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
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

    *path = realpath(device_path, nullptr);
    NVSHMEMI_NULL_ERROR_JMP(*path, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "realpath failed \n");

out:
    return status;
}

static void nvshmemt_ibv_ftable_fini_wrapper(void) {
    if (ibv_lib_handle) {
        dlclose(ibv_lib_handle);
        ibv_lib_handle = nullptr;
    }
}

static void nvshmemt_mlx5dv_ftable_fini_wrapper(void) {
    if (mlx5_lib_handle) {
        dlclose(mlx5_lib_handle);
        mlx5_lib_handle = nullptr;
    }
}

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level) {
    if (ibv_lib_handle != nullptr) {
        *ibv_handle = ibv_lib_handle;
    } else {
        *ibv_handle = dlopen("libibverbs.so.1", RTLD_LAZY);
        if (*ibv_handle == nullptr) {
            INFO(log_level, "libibverbs not found on the system.");
            return -1;
        }
        ibv_lib_handle = *ibv_handle;
        atexit(nvshmemt_ibv_ftable_fini_wrapper);
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
    if (mlx5_lib_handle != nullptr) {
        *mlx5dv_handle = mlx5_lib_handle;
    } else {
        *mlx5dv_handle = dlopen("libmlx5.so", RTLD_LAZY);
        if (*mlx5dv_handle == nullptr) {
            *mlx5dv_handle = dlopen("libmlx5.so.1", RTLD_LAZY);
        }
        if (*mlx5dv_handle == nullptr) {
            INFO(log_level, "Failed to open libmlx5.so[.1]");
            ftable->mlx5dv_internal_is_supported = nullptr;
            ftable->mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
            ftable->mlx5dv_internal_reg_dmabuf_mr = nullptr;
            return -1;
        }
        mlx5_lib_handle = *mlx5dv_handle;
        atexit(nvshmemt_mlx5dv_ftable_fini_wrapper);
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
    if (ibv_handle) {
        *ibv_handle = nullptr;
    }
}

void nvshmemt_mlx5dv_ftable_fini(void **mlx5dv_handle) {
    if (mlx5dv_handle) {
        *mlx5dv_handle = nullptr;
    }
}

bool nvshmemt_check_hca_prefix(const nvshmemi_options_s *options, const char *name) {
    bool device_supported = false;
    const char *hca_prefix = "^smi";
    if (options->HCA_PREFIX_provided) {
        hca_prefix = options->HCA_PREFIX;
    }

    if (hca_prefix[0] == '^') {
        // ignore first letter : "^"
        device_supported = strstr(name, &hca_prefix[1]) == nullptr;
    } else {
        device_supported = strstr(name, hca_prefix) != nullptr;
    }
    if (!device_supported) {
        NVSHMEMI_WARN_PRINT(
            "device %s is not supported (expected HCA interface: %s). Skipping...\n", name,
            hca_prefix);
    }

    return device_supported;
}

#ifdef NVSHMEM_USE_MLX5DV
int nvshmemt_ib_common_init_mlx5dv(void **mlx5dv_handle,
                                   struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                   bool disable_data_direct, int log_level) {
    if (!disable_data_direct) {
        if (nvshmemt_mlx5dv_ftable_init(mlx5dv_handle, mlx5dv_ftable, log_level)) {
            NVSHMEMI_WARN_PRINT("Unable to dlopen libmlx5dv. Disabling directNIC features.");
            mlx5dv_ftable->mlx5dv_internal_is_supported = nullptr;
            mlx5dv_ftable->mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
            mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr = nullptr;
        }
    } else {
        mlx5dv_ftable->mlx5dv_internal_is_supported = nullptr;
        mlx5dv_ftable->mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
        mlx5dv_ftable->mlx5dv_internal_reg_dmabuf_mr = nullptr;
        INFO(log_level, "directNIC features are disabled by NVSHMEM_DISABLE_DATA_DIRECT=1");
    }
    return 0;
}
#endif

int nvshmemt_ib_common_parse_hca_filter(struct nvshmemt_ib_hca_filter &filter,
                                        const struct nvshmemt_ib_common_state &state) {
    struct nvshmemi_options_s *options = state.options;
    int log_level = state.log_level;
    filter.hca_list_count = 0;
    filter.pe_hca_map_count = 0;
    filter.user_selection = 0;
    filter.exclude_list = 0;

    if (options->HCA_LIST_provided) {
        filter.user_selection = 1;
        filter.exclude_list = (options->HCA_LIST[0] == '^');
        filter.hca_list_count =
            nvshmemt_parse_hca_list(options->HCA_LIST, filter.hca_list, MAX_NUM_HCAS, log_level);
    }

    if (options->HCA_PE_MAPPING_provided) {
        if (filter.hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            filter.user_selection = 1;
            filter.pe_hca_map_count = nvshmemt_parse_hca_list(
                options->HCA_PE_MAPPING, filter.pe_hca_mapping, MAX_NUM_PES_PER_NODE, log_level);
        }
    }

    return 0;
}

void nvshmemt_ib_common_warn_missing_hcas(const struct nvshmemt_ib_hca_filter &filter) {
    if (filter.hca_list_count) {
        for (int j = 0; j < filter.hca_list_count; j++) {
            if (filter.hca_list[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "could not find user specified HCA name: %s port: %d, skipping\n",
                    filter.hca_list[j].name, filter.hca_list[j].port);
            }
        }
    } else if (filter.pe_hca_map_count) {
        for (int j = 0; j < filter.pe_hca_map_count; j++) {
            if (filter.pe_hca_mapping[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "could not find user specified HCA name: %s port: %d, skipping\n",
                    filter.pe_hca_mapping[j].name, filter.pe_hca_mapping[j].port);
            }
        }
    }
}

void nvshmemt_ib_common_log_device_assignment(const struct nvshmemt_ib_common_state &state) {
    INFO(state.log_level,
         "Begin - Ordered list of devices for assignment (after processing user provided env vars "
         "(if any))  - ");
    for (int i = 0; i < state.n_dev_ids; i++) {
        INFO(state.log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d",
             i, state.n_dev_ids, state.dev_ids[i], state.port_ids[i]);
    }
    INFO(state.log_level,
         "End - Ordered list of devices for assignment (after processing user provided env vars "
         "(if any))");
}

int nvshmemt_ib_common_check_dmabuf_support(bool &out_dmabuf_support,
                                            const struct nvshmemi_cuda_fn_table *table,
                                            bool ib_disable_dmabuf) {
    int status = 0;
    int flag;
    CUdevice gpu_device_id;

    out_dmabuf_support = false;

    if (!ib_disable_dmabuf) {
        status = CUPFN(table, cuCtxGetDevice(&gpu_device_id));
        if (status != CUDA_SUCCESS) {
            return NVSHMEMX_ERROR_INTERNAL;
        }
        status = CUPFN(table, cuDeviceGetAttribute(
                                  &flag, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                  gpu_device_id));
        if (status != CUDA_SUCCESS) {
            status = 0;
            cudaGetLastError();
        } else if (flag == 1) {
            out_dmabuf_support = true;
        }
    }

    if (!out_dmabuf_support) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }
    }

    return 0;
}

int nvshmemt_ib_common_discover_pci_paths(nvshmem_transport_t t,
                                          struct nvshmemt_ib_common_state &state,
                                          size_t device_struct_size,
                                          const struct nvshmemt_ibv_function_table *ftable,
                                          const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;

    transport->n_devices = state.n_dev_ids;
    transport->device_pci_paths = (char **)calloc(transport->n_devices, sizeof(char *));
    if (!transport->device_pci_paths) {
        NVSHMEMI_ERROR_PRINT("Unable to allocate paths for IB transport.");
        return NVSHMEMX_ERROR_INTERNAL;
    }
    for (int i = 0; i < transport->n_devices; i++) {
        int dev_id = state.dev_ids[i];
        struct nvshmemt_ib_common_device *device =
            (struct nvshmemt_ib_common_device *)((char *)state.devices +
                                                 dev_id * device_struct_size);
        status = nvshmemt_ib_iface_get_mlx_path(
            device->dev, device->context, &transport->device_pci_paths[i], ftable, mlx5dv_ftable,
            &device->data_direct, state.log_level);
        if (status) {
            NVSHMEMI_ERROR_PRINT("nvshmemt_ib_iface_get_mlx_path failed \n");
            status = NVSHMEMX_ERROR_INTERNAL;
            for (int j = 0; j < i; j++) {
                free(transport->device_pci_paths[j]);
            }
            free(transport->device_pci_paths);
            transport->device_pci_paths = nullptr;
            transport->n_devices = 0;
            break;
        }
    }

    return status;
}

int nvshmemt_ib_common_enumerate_devices(const struct nvshmemt_ibv_function_table *ftable,
                                         struct nvshmemt_ib_common_state &state,
                                         size_t device_struct_size,
                                         struct nvshmemt_ib_hca_filter &filter,
                                         struct ibv_device **dev_list, int num_devices) {
    void *devices = state.devices;
    int *dev_ids = state.dev_ids;
    int *port_ids = state.port_ids;
    struct nvshmemi_options_s *options = state.options;
    int log_level = state.log_level;
    int status = 0;
    int offset = 0;

    INFO(log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - ");
    for (int i = 0; i < num_devices; i++) {
        struct nvshmemt_ib_common_device *device =
            (struct nvshmemt_ib_common_device *)((char *)devices + i * device_struct_size);
        device->dev = dev_list[i];

        device->context = ftable->open_device(device->dev);
        if (!device->context) {
            INFO(log_level, "open_device failed for IB device at index %d", i);
            continue;
        }

        const char *name = ftable->get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");

        bool device_supported = nvshmemt_check_hca_prefix(options, name);

        if (!device_supported) {
            ftable->close_device(device->context);
            device->context = nullptr;
            continue;
        }

        status = ftable->query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        INFO(log_level,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d", i,
             num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (filter.hca_list_count) {
                allowed_device = filter.exclude_list;
                for (int j = 0; j < filter.hca_list_count; j++) {
                    if (!strcmp(filter.hca_list[j].name, name)) {
                        if (filter.hca_list[j].port == -1 || filter.hca_list[j].port == p) {
                            filter.hca_list[j].found = 1;
                            allowed_device = !filter.exclude_list;
                        }
                    }
                }
            } else if (filter.pe_hca_map_count) {
                allowed_device = 0;
                for (int j = 0; j < filter.pe_hca_map_count; j++) {
                    if (!strcmp(filter.pe_hca_mapping[j].name, name)) {
                        if (filter.pe_hca_mapping[j].port == -1 ||
                            filter.pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            filter.pe_hca_mapping[j].found = 1;
                            replicate_count = filter.pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            }

            status = ftable->query_port(device->context, p, &device->port_attr[p - 1]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "ibv_port_query failed \n");

            if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) ||
                (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND &&
                    device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET)) {
                if (filter.user_selection) {
                    NVSHMEMI_WARN_PRINT(
                        "found inactive port or port with non-IB link layer protocol, "
                        "skipping...\n");
                }
                continue;
            }

            ib_get_gid_index(ftable, device->context, p, device->port_attr[p - 1].gid_tbl_len,
                                &device->gid_info[p - 1].local_gid_index, log_level, options);
            status =
                ftable->query_gid(device->context, p, device->gid_info[p - 1].local_gid_index,
                                    &device->gid_info[p - 1].local_gid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "query_gid failed \n");

            if (!device->pd) {
                device->pd = ftable->alloc_pd(device->context);
                NVSHMEMI_NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "ibv_alloc_pd failed \n");
            }

            for (int k = 0; k < replicate_count; k++) {
                if (offset >= MAX_NUM_PES_PER_NODE) {
                    NVSHMEMI_ERROR_PRINT(
                        "Too many device/port entries (%d), exceeds MAX_NUM_PES_PER_NODE "
                        "(%d)\n",
                        offset + 1, MAX_NUM_PES_PER_NODE);
                    status = NVSHMEMX_ERROR_INTERNAL;
                    goto out;
                }
                dev_ids[offset] = i;
                port_ids[offset] = p;
                offset++;
            }

            device_used = 1;
        }

        if (!device_used) {
            if (device->pd) {
                status = ftable->dealloc_pd(device->pd);
                device->pd = nullptr;
            }
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dealloc_pd failed \n");

            if (device->context) {
                status = ftable->close_device(device->context);
                device->context = nullptr;
            }
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_close_device failed \n");
        }
    }
    INFO(log_level, "End - Enumerating IB devices in the system");

    state.n_dev_ids = offset;

    if (!state.n_dev_ids) {
        INFO(log_level, "no active IB device found, exiting");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    if (status) {
        for (int i = 0; i < num_devices; i++) {
            struct nvshmemt_ib_common_device *device =
                (struct nvshmemt_ib_common_device *)((char *)devices + i * device_struct_size);
            if (device->pd) {
                int rc = ftable->dealloc_pd(device->pd);
                if (rc) {
                    NVSHMEMI_ERROR_PRINT("ibv_dealloc_pd failed during cleanup for device %d\n", i);
                }
                device->pd = nullptr;
            }
            if (device->context) {
                int rc = ftable->close_device(device->context);
                if (rc) {
                    NVSHMEMI_ERROR_PRINT("ibv_close_device failed during cleanup for device %d\n",
                                         i);
                }
                device->context = nullptr;
            }
        }
        state.n_dev_ids = 0;
    }
    return status;
}
