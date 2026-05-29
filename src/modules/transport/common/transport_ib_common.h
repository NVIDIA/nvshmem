/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _TRANSPORT_IB_COMMON_H
#define _TRANSPORT_IB_COMMON_H
#include <arpa/inet.h>                                           // for inet...
#include <errno.h>                                               // for errno
#include <fcntl.h>                                               // for open
#include <limits.h>                                              // for PATH...
#include <netinet/in.h>                                          // for in6_...
#include <stdint.h>                                              // for uint...
#include <stdio.h>                                               // for NULL
#include <stdlib.h>                                              // for strtol
#include <string.h>                                              // for strlen
#include <sys/socket.h>                                          // for AF_INET
#include <sys/un.h>                                              // for sa_f...
#include <unistd.h>                                              // for close
#include "bootstrap_host_transport/env_defs_internal.h"          // for nvsh...
#include "infiniband/verbs.h"                                    // for ibv_gid
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvsh...
#include "non_abi/nvshmem_build_options.h"                       // for NVSH...
#include "non_abi/nvshmemx_error.h"                              // for NVSH...
#include "transport_common.h"                                    // for INFO
#include "transport_ib_common.h"                                 // lines 26-26

#ifdef NVSHMEM_USE_MLX5DV
#include <infiniband/mlx5dv.h>
#ifndef MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT
#define MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT 1
#endif
// MLX5DV Library versioning
#define MLX5DV_VERSION "MLX5_1.8"
#endif

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

#define NETMASK(bits) (htonl(0xffffffff << (32 - bits)))

#ifndef MAX_NUM_HCAS
#define MAX_NUM_HCAS 48
#endif

#ifndef MAX_NUM_PES_PER_NODE
#define MAX_NUM_PES_PER_NODE 32
#endif

typedef void *nvshmemt_ib_common_ep_ptr_t;

typedef enum {
    NVSHMEMT_IB_COMMON_WAIT_ANY = 0,
    NVSHMEMT_IB_COMMON_WAIT_TWO = 1,
    NVSHMEMT_IB_COMMON_WAIT_ALL = 2,
    NVSHMEMT_IB_COMMON_WAIT_NONE = 3
} nvshmemt_ib_wait_predicate_t;

/* Generic ep structure that should be embedded at the top of transport-specific ep structures */
struct nvshmemt_ib_common_ep {
    volatile uint64_t head_op_id;
    volatile uint64_t tail_op_id;
    nvshmem_transport_t transport;
};

#ifndef MAX_NUM_PORTS
#define MAX_NUM_PORTS 4
#endif

struct nvshmemt_ib_gid_info {
    uint8_t link_layer;
    union ibv_gid local_gid;
    int32_t local_gid_index;
};

/* Common device structure - start with basic IB fields */
struct nvshmemt_ib_common_device {
    struct ibv_device *dev;
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    struct nvshmemt_ib_gid_info gid_info[MAX_NUM_PORTS];
    bool data_direct;
};

struct nvshmemt_ib_common_ftable {
    int (*ep_create)(nvshmemt_ib_common_ep_ptr_t *ep, int dev_id, nvshmem_transport_t t);
    int (*ep_get_handle)(struct nvshmemt_ib_common_ep_handle *ep_handle,
                         nvshmemt_ib_common_ep_ptr_t ep);
    int (*ep_connect)(nvshmemt_ib_common_ep_ptr_t ep,
                      struct nvshmemt_ib_common_ep_handle *ep_handle);
    int (*progress)(nvshmem_transport_t tcurr);
    int (*progress_recv)(nvshmem_transport_t tcurr, nvshmemt_ib_wait_predicate_t wait_predicate);
};

struct nvshmemt_ib_common_ep_handle {
    uint32_t qpn;
    uint16_t lid;
    // GID routing
    uint64_t spn;
    uint64_t iid;
};

struct nvshmemt_ib_qp_path {
    uint16_t dlid;
    bool grh_required;
};

struct nvshmemt_ib_common_state {
    void *devices;
    int *dev_ids;
    int *port_ids;
    int n_dev_ids;
    int ep_count;
    int qp_depth;
    int srq_depth;
    int host_ep_index;
    int cur_ep_index;
    int selected_dev_id;
    int log_level;
    bool dmabuf_support;
    nvshmemt_ib_common_ep_ptr_t cst_ep;
    nvshmemt_ib_common_ep_ptr_t *ep;
    struct nvshmemi_options_s *options;
    struct nvshmemi_cuda_fn_table *table;
    struct nvshmemt_ib_common_ftable *ib_transport_ftable;
    struct transport_mem_handle_info_cache *cache;

    /* Dynamic QP management fields */
    int next_qp_index;        /* Next available QP index */
    int cur_default_qp_index; /* Round-robin counter for DEFAULT QPs */
    int cur_any_qp_index;     /* Round-robin counter for ANY QPs */
};

typedef struct nvshmemt_ib_common_state *nvshmemt_ib_common_state_t;

struct nvshmemt_ib_common_mem_handle {
    struct ibv_mr *mr;
    void *buf;
    int fd;
    uint32_t lkey;
    uint32_t rkey;
    bool local_only;
};

struct nvshmemt_ibv_function_table {
    int (*fork_init)(void);
    struct ibv_ah *(*create_ah)(struct ibv_pd *pd, struct ibv_ah_attr *ah_attr);
    struct ibv_device **(*get_device_list)(int *num_devices);
    const char *(*get_device_name)(struct ibv_device *device);
    struct ibv_context *(*open_device)(struct ibv_device *device);
    int (*close_device)(struct ibv_context *context);
    int (*query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
    int (*query_port)(struct ibv_context *context, uint8_t port_num,
                      struct ibv_port_attr *port_attr);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *context);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    struct ibv_mr *(*reg_mr_iova)(struct ibv_pd *pd, void *addr, size_t length, uint64_t hca_va,
                                  int access);
    struct ibv_mr *(*reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access);
    int (*dereg_mr)(struct ibv_mr *mr);
    struct ibv_cq *(*create_cq)(struct ibv_context *context, int cqe, void *cq_context,
                                struct ibv_comp_channel *channel, int comp_vector);
    struct ibv_qp *(*create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
    struct ibv_srq *(*create_srq)(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr);
    int (*dealloc_pd)(struct ibv_pd *pd);
    int (*modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
    int (*query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
    int (*destroy_qp)(struct ibv_qp *qp);
    int (*destroy_cq)(struct ibv_cq *cq);
    int (*destroy_srq)(struct ibv_srq *srq);
    int (*destroy_ah)(struct ibv_ah *ah);
};

struct nvshmemt_mlx5dv_function_table {
    bool (*mlx5dv_internal_is_supported)(struct ibv_device *device);
    int (*mlx5dv_internal_get_data_direct_sysfs_path)(struct ibv_context *context, char *buf,
                                                      size_t buf_len);
    /* DMA-BUF support */
    struct ibv_mr *(*mlx5dv_internal_reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset,
                                                    size_t length, uint64_t iova, int fd,
                                                    int access, int mlx5_access);
};

bool nvshmemt_mlx5dv_dmabuf_capable(ibv_context *context,
                                    const struct nvshmemt_ibv_function_table *ftable,
                                    const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable);

int nvshmemt_ib_iface_get_mlx_path(ibv_device *dev, ibv_context *ctx, char **path,
                                   const struct nvshmemt_ibv_function_table *ftable,
                                   const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                   bool *is_data_direct, int log_level);

int nvshmemt_ibv_ftable_init(void **ibv_handle, struct nvshmemt_ibv_function_table *ftable,
                             int log_level);
void nvshmemt_ibv_ftable_fini(void **ibv_handle);

int nvshmemt_mlx5dv_ftable_init(void **mlx5dv_handle, struct nvshmemt_mlx5dv_function_table *ftable,
                                int log_level);
void nvshmemt_mlx5dv_ftable_fini(void **mlx5dv_handle);

void nvshmemt_ib_common_sanitize_timeout(struct nvshmemi_options_s *options);
void nvshmemt_ib_common_sanitize_retry_cnt(struct nvshmemi_options_s *options);

int nvshmemt_ib_common_nv_peer_mem_available();

int nvshmemt_ib_common_reg_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                      struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                      struct ibv_pd *pd, nvshmem_mem_handle_t *mem_handle,
                                      void *buf, size_t length, bool local_only,
                                      bool dmabuf_support, struct nvshmemi_cuda_fn_table *table,
                                      int log_level, bool relaxed_ordering, bool is_data_direct,
                                      void *alias_va_ptr = NULL);

int nvshmemt_ib_common_release_mem_handle(struct nvshmemt_ibv_function_table *ftable,
                                          nvshmem_mem_handle_t *mem_handle, int log_level);

const char *nvshmemt_ib_common_link_layer_name(uint8_t link_layer);

int nvshmemt_ib_common_setup_cst_loopback(int dev_id, nvshmem_transport_t t);

int nvshmemt_ib_common_check_poll_avail(nvshmem_transport_t tcurr, nvshmemt_ib_common_ep_ptr_t ep,
                                        nvshmemt_ib_wait_predicate_t wait_predicate);

int nvshmemt_ib_common_quiet(struct nvshmem_transport *tcurr, int pe, int qp_index);

int nvshmemt_ib_common_fence(nvshmem_transport_t tcurr, int pe, int qp_index, int is_multi);

int nvshmemt_ib_common_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                         int num_selected_devs, int *out_qp_indices, int num_qps);

/* Helper function to get ep from qp index */
nvshmemt_ib_common_ep_ptr_t nvshmemt_ib_common_get_ep_from_qp_index(nvshmem_transport_t t,
                                                                    int qp_index, int pe_index);

/* Helper function to filter devices based on HCA_PREFIX */
bool nvshmemt_check_hca_prefix(const nvshmemi_options_s *options, const char *name);

#ifdef NVSHMEM_USE_MLX5DV
int nvshmemt_ib_common_init_mlx5dv(void **mlx5dv_handle,
                                   struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable,
                                   bool disable_data_direct, int log_level);
#endif

struct nvshmemt_ib_hca_filter {
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    int hca_list_count;
    int pe_hca_map_count;
    int user_selection;
    int exclude_list;
};

int nvshmemt_ib_common_parse_hca_filter(struct nvshmemt_ib_hca_filter &filter,
                                        const struct nvshmemt_ib_common_state &state);

void nvshmemt_ib_common_warn_missing_hcas(const struct nvshmemt_ib_hca_filter &filter);

void nvshmemt_ib_common_log_device_assignment(const struct nvshmemt_ib_common_state &state);

int nvshmemt_ib_common_check_dmabuf_support(bool &out_dmabuf_support,
                                            const struct nvshmemi_cuda_fn_table *table,
                                            bool ib_disable_dmabuf);

int nvshmemt_ib_common_discover_pci_paths(nvshmem_transport_t t,
                                          struct nvshmemt_ib_common_state &state,
                                          size_t device_struct_size,
                                          const struct nvshmemt_ibv_function_table *ftable,
                                          const struct nvshmemt_mlx5dv_function_table *mlx5dv_ftable);

int nvshmemt_ib_common_enumerate_devices(const struct nvshmemt_ibv_function_table *ftable,
                                         struct nvshmemt_ib_common_state &state,
                                         size_t device_struct_size,
                                         struct nvshmemt_ib_hca_filter &filter,
                                         struct ibv_device **dev_list, int num_devices);

/* The following code is for dynamic GID detection for RoCE platforms.
   It has been adapted from NCCL */
int ib_roce_get_version_num(const char *deviceName, int portNum, int gidIndex, int *version);
struct nvshmemt_ib_qp_path nvshmemt_ib_select_qp_path(const union ibv_gid *local_gid,
                                                      uint16_t local_lid, uint16_t remote_lid,
                                                      uint64_t remote_spn, uint64_t remote_iid);
void ib_get_gid_index(const struct nvshmemt_ibv_function_table *ftable, struct ibv_context *context,
                      uint8_t portNum, const struct ibv_port_attr *portAttr, int *gidIndex,
                      int log_level, nvshmemi_options_s *options);
#endif
