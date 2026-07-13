/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <assert.h>
#include <stdint.h>  // IWYU pragma: keep
// IWYU pragma: no_include <bits/stdint-uintn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// IWYU pragma: no_include <mm_malloc.h>
#include <string.h>
#include <unistd.h>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "internal/host_transport/cudawrap.h"
#include "bootstrap_host_transport/env_defs_internal.h"
#ifdef NVSHMEM_USE_GDRCOPY
#include "gdrapi.h"
#endif
#include "infiniband/verbs.h"
#include "non_abi/nvshmem_build_options.h"
#include "device_host_transport/nvshmem_constants.h"
#include "device_host_transport/nvshmem_common_transport.h"
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/host_transport/nvshmemi_transport_defines.h"
#include "non_abi/nvshmemx_error.h"
#include "non_abi/nvshmem_version.h"
#include "internal/host_transport/transport.h"
#include "transport_common.h"
#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif
#include "transport_ib_common.h"

#ifdef NVSHMEM_X86_64
#include <immintrin.h>  // IWYU pragma: keep
#endif
// IWYU pragma: no_include <xmmintrin.h>

#define IBRC_MAX_INLINE_SIZE 128

// Helper functions to access qp_depth and srq_depth from state
static inline int get_ibrc_qp_depth(nvshmemt_ib_common_state_t state) { return state->qp_depth; }

static inline int get_ibrc_srq_depth(nvshmemt_ib_common_state_t state) { return state->srq_depth; }

#define IBRC_SRQ_MASK(state) (get_ibrc_srq_depth(state) - 1)
#define IBRC_REQUEST_QUEUE_MASK(state) (get_ibrc_qp_depth(state) - 1)
#define IBRC_BUF_SIZE 64

#if defined(NVSHMEM_X86_64)
#define IBRC_CACHELINE 64
#elif defined(NVSHMEM_PPC64LE)
#define IBRC_CACHELINE 128
#elif defined(NVSHMEM_AARCH64)
#define IBRC_CACHELINE 64
#else
#error Unknown cache line size
#endif

#define MAX_NUM_HCAS 48
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32
#ifdef NVSHMEM_USE_GDRCOPY
#define BAR_READ_BUFSIZE (2 * 1024 * 1024)
#else
#define BAR_READ_BUFSIZE (sizeof(uint64_t))
#endif
#define IBRC_GRH_HOP_LIMIT 255

// Enum values are now defined in transport_ib_common.h

#ifdef NVSHMEM_USE_GDRCOPY
struct ibrc_gdrcopy_mapping {
    void *cpu_ptr_base = nullptr;
    gdr_mh_t mh{};
    bool pinned = false;
    bool mapped = false;
};
#endif

struct ibrc_request {
    struct ibv_send_wr sr;
    struct ibv_send_wr *bad_sr;
    struct ibv_sge sge;
};

struct ibrc_atomic_op {
    nvshmemi_amo_t op; /* high bit (NVSHMEMI_AMO_FLOAT_BIT) encodes float type */
    void *addr;
    void *retaddr;
    uint32_t retrkey;
    uint64_t retflag;
    uint32_t elembytes;
    uint64_t compare;
    uint64_t swap_add;
};

typedef struct ibrc_buf {
    struct ibv_recv_wr rwr;
    struct ibv_recv_wr *bad_rwr;
    struct ibv_sge sge;
    int qp_num;
    char buf[IBRC_BUF_SIZE];
} ibrc_buf_t;
ibrc_buf_t *bpool;
int bpool_size;
static std::vector<void *> bpool_free;
static std::deque<void *> bqueue_toprocess;

struct ibrc_device {
    struct nvshmemt_ib_common_device common_device;
    // bpool information
    struct ibv_srq *srq;
    int srq_posted;
    struct ibv_mr *bpool_mr;
    struct ibv_cq *recv_cq;
    struct ibv_cq *send_cq;
};

struct ibrc_ep {
    struct nvshmemt_ib_common_ep common_ep; /* Must be first for casting */
    int devid;
    int portid;
    struct ibv_qp *qp;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibrc_request *req;
};

typedef struct ibrc_mem_handle_info {
    struct ibv_mr *mr;
    void *ptr;
    size_t size;
    void *cpu_ptr;  // CPU-accessible pointer: set via GDRCopy map or directly for SYSMEM
#ifdef NVSHMEM_USE_GDRCOPY
    ibrc_gdrcopy_mapping gdrcopy;
#endif
} ibrc_mem_handle_info_t;
ibrc_mem_handle_info_t *dummy_local_mem;
pthread_mutex_t ibrc_mutex_recv_progress;
pthread_mutex_t ibrc_mutex_send_progress;

static std::map<unsigned int, long unsigned int> qp_map;
static uint64_t connected_qp_count;

static int use_ib_native_atomics = 1;
/* Maximum number of RDMA Read & Atomic operations that can be outstanding per QP */
static int nvshmemt_ibrc_max_rd_atomic = INT_MAX;
static bool use_gdrcopy = 0;
static std::atomic<bool> use_cpu_atomics{
    false};  // true when send-based atomics are possible (GDRCopy or SYSMEM)
static volatile uint64_t atomics_received = 0;
static volatile uint64_t atomics_processed = 0;
static volatile uint64_t atomics_issued = 0;
static volatile uint64_t atomics_completed = 0;
static volatile uint64_t atomics_acked = 0;
static bool is_egm = false;
#ifdef NVSHMEM_USE_GDRCOPY
static gdr_t gdr_desc;
static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;

static int nvshmemt_ibrc_release_gdrcopy_mapping(ibrc_mem_handle_info_t &handle_info) {
    int status = 0;
    int first_error = 0;
    auto &mapping = handle_info.gdrcopy;

    if (mapping.mapped) {
        status = gdrcopy_ftable.unmap(gdr_desc, mapping.mh, mapping.cpu_ptr_base, handle_info.size);
        if (status == 0) {
            mapping.mapped = false;
            mapping.cpu_ptr_base = nullptr;
        } else if (!first_error) {
            first_error = status;
        }
    }

    if (mapping.pinned) {
        status = gdrcopy_ftable.unpin_buffer(gdr_desc, mapping.mh);
        if (status == 0) {
            mapping.pinned = false;
            mapping.mh = {};
            if (first_error) {
                mapping.mapped = false;
                mapping.cpu_ptr_base = nullptr;
            }
        } else if (!first_error) {
            first_error = status;
        }
    }

    return first_error;
}
#endif

static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

static struct nvshmemt_mlx5dv_function_table mlx5dv_ftable;
static void *mlx5dv_handle;

int progress_send(nvshmemt_ib_common_state_t ibrc_state);
int progress_recv_wrapper(nvshmem_transport_t tcurr, nvshmemt_ib_wait_predicate_t wait_predicate);

// Allocate memory to be potentially ibv_reg_mr'd. This needs to be
// // allocated on separate pages as those pages will be marked DONTFORK
// // and if they are shared, that could cause a crash in a child process
static int nvshmemi_ib_malloc_debug(void **ptr, size_t size, int log_level, const char *filefunc,
                                    int line) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    void *p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return -1;
    memset(p, 0, size);
    *ptr = p;
    INFO(log_level, "%s:%d Ib Alloc Size %ld pointer %p", filefunc, line, size, *ptr);
    return 0;
}
#define nvshmemi_ib_malloc(...) nvshmemi_ib_malloc_debug(__VA_ARGS__, __FILE__, __LINE__)

ibrc_mem_handle_info_t *get_mem_handle_info(nvshmem_transport_t t, void *gpu_ptr) {
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;

    return (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get(t, ibrc_state->cache, gpu_ptr);
}

static const char *ibrc_wc_status_string(enum ibv_wc_status status) {
    switch (status) {
        case IBV_WC_SUCCESS:
            return "success";
        case IBV_WC_LOC_LEN_ERR:
            return "local length error";
        case IBV_WC_LOC_QP_OP_ERR:
            return "local QP operation error";
        case IBV_WC_LOC_PROT_ERR:
            return "local protection error";
        case IBV_WC_WR_FLUSH_ERR:
            return "work request flushed error";
        case IBV_WC_MW_BIND_ERR:
            return "memory window bind error";
        case IBV_WC_BAD_RESP_ERR:
            return "bad response error";
        case IBV_WC_LOC_ACCESS_ERR:
            return "local access error";
        case IBV_WC_REM_INV_REQ_ERR:
            return "remote invalid request error";
        case IBV_WC_REM_ACCESS_ERR:
            return "remote access error";
        case IBV_WC_REM_OP_ERR:
            return "remote operation error";
        case IBV_WC_RETRY_EXC_ERR:
            return "transport retry counter exceeded";
        case IBV_WC_RNR_RETRY_EXC_ERR:
            return "RNR retry counter exceeded";
        case IBV_WC_REM_ABORT_ERR:
            return "remote aborted error";
        case IBV_WC_GENERAL_ERR:
            return "general error";
        default:
            return "unknown completion error";
    }
}

inline int refill_srq(struct ibrc_device *device, nvshmemt_ib_common_state_t ibrc_state) {
    int status = 0;

    while ((device->srq_posted < get_ibrc_srq_depth(ibrc_state)) && !bpool_free.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bpool_free.back();

        buf->rwr.next = NULL;
        buf->rwr.wr_id = (uint64_t)buf;
        buf->rwr.sg_list = &(buf->sge);
        buf->rwr.num_sge = 1;

        buf->sge.addr = (uint64_t)buf->buf;
        buf->sge.length = IBRC_BUF_SIZE;
        buf->sge.lkey = device->bpool_mr->lkey;

        status = ibv_post_srq_recv(device->srq, &buf->rwr, &buf->bad_rwr);
        NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "ibv_post_srq_recv failed \n");

        bpool_free.pop_back();
        device->srq_posted++;
    }

out:
    return status;
}

int nvshmemt_ibrc_show_info(struct nvshmem_transport * /*transport*/, int /*style*/) {
    NVSHMEMI_ERROR_PRINT("ibrc show info not implemented");
    return 0;
}

int nvshmemt_ibrc_can_reach_peer(int *access, struct nvshmem_transport_pe_info * /*peer_info*/,
                                 nvshmem_transport_t /*t*/) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return status;
}

static int ep_create(void **ep_ptr, int devid, nvshmem_transport_t t) {
    int status = 0;
    struct ibrc_ep *ep = NULL;
    struct ibrc_ep **my_ep_ptr = (struct ibrc_ep **)ep_ptr;
    struct ibv_qp_init_attr init_attr;
    struct ibv_qp_attr attr;
    int flags;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;
    struct ibrc_device *device =
        ((struct ibrc_device *)ibrc_state->devices + ibrc_state->dev_ids[devid]);
    int portid = ibrc_state->port_ids[devid];
    const struct ibv_port_attr *port_attr = device->common_device.port_attr + (portid - 1);
    const int pkey_index = ibrc_state->options->IB_PKEY_INDEX;
    struct ibv_context *context = device->common_device.context;
    struct ibv_pd *pd = device->common_device.pd;

    if (pkey_index < 0 || pkey_index >= port_attr->pkey_tbl_len) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "Invalid NVSHMEM_IB_PKEY_INDEX %d for IBRC QP: expected 0 <= "
            "NVSHMEM_IB_PKEY_INDEX < pkey_tbl_len (%hu); pe %d device %s devid %d port %d "
            "link_layer %s lid %hu\n",
            pkey_index, port_attr->pkey_tbl_len, t->my_pe, device->common_device.dev->name,
            ibrc_state->dev_ids[devid], portid,
            nvshmemt_ib_common_link_layer_name(port_attr->link_layer), port_attr->lid);
    }

    // algining ep structure to prevent split tranactions when accessing head_op_id and
    // tail_op_id which can be used in inter-thread synchronization
    // TODO: use atomic variables instead to rely on language memory model guarantees
    status = posix_memalign((void **)&ep, IBRC_CACHELINE, sizeof(struct ibrc_ep));
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "ep allocation failed \n");
    memset((void *)ep, 0, sizeof(struct ibrc_ep));

    if (!device->send_cq) {
        device->send_cq =
            ftable.create_cq(context, device->common_device.device_attr.max_cqe, NULL, NULL, 0);
        NVSHMEMT_ERRNO_NULL_ERROR_JMP(device->send_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_create_cq failed \n");
    }
    assert(device->send_cq != NULL);
    ep->send_cq = device->send_cq;

    if (!device->srq) {
        struct ibv_srq_init_attr srq_init_attr;
        memset(&srq_init_attr, 0, sizeof(srq_init_attr));

        srq_init_attr.attr.max_wr = get_ibrc_srq_depth(ibrc_state);
        srq_init_attr.attr.max_sge = 1;

        device->srq = ftable.create_srq(pd, &srq_init_attr);
        NVSHMEMT_ERRNO_NULL_ERROR_JMP(device->srq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_create_srq failed \n");

        device->recv_cq = ftable.create_cq(context, get_ibrc_srq_depth(ibrc_state), NULL, NULL, 0);
        NVSHMEMT_ERRNO_NULL_ERROR_JMP(device->recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_create_cq failed \n");
    }
    assert(device->recv_cq != NULL);
    ep->recv_cq = device->recv_cq;

    memset(&init_attr, 0, sizeof(struct ibv_qp_init_attr));
    init_attr.srq = device->srq;
    init_attr.send_cq = ep->send_cq;
    init_attr.recv_cq = ep->recv_cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = get_ibrc_qp_depth(ibrc_state);
    init_attr.cap.max_recv_wr = 0;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 0;
    init_attr.cap.max_inline_data = IBRC_MAX_INLINE_SIZE;

    ep->qp = ftable.create_qp(pd, &init_attr);
    NVSHMEMT_ERRNO_NULL_ERROR_JMP(
        ep->qp, status, NVSHMEMX_ERROR_INTERNAL, out,
        "IBRC QP create failed: pe %d device %s devid %d port %d link_layer %s lid %u "
        "qp_type %d qp_depth %d srq_depth %d max_send_wr %u max_recv_wr %u "
        "max_send_sge %u max_recv_sge %u max_inline_data %u\n",
        t->my_pe, device->common_device.dev->name, ibrc_state->dev_ids[devid], portid,
        nvshmemt_ib_common_link_layer_name(device->common_device.port_attr[portid - 1].link_layer),
        device->common_device.port_attr[portid - 1].lid, init_attr.qp_type,
        get_ibrc_qp_depth(ibrc_state), get_ibrc_srq_depth(ibrc_state), init_attr.cap.max_send_wr,
        init_attr.cap.max_recv_wr, init_attr.cap.max_send_sge, init_attr.cap.max_recv_sge,
        init_attr.cap.max_inline_data);

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = static_cast<uint16_t>(pkey_index);
    attr.port_num = portid;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(
        status, NVSHMEMX_ERROR_INTERNAL, out,
        "IBRC QP modify RESET->INIT failed: pe %d device %s devid %d port %d "
        "link_layer %s lid %u local_qpn %u pkey_index %u access_flags %d flags %d "
        "\n",
        t->my_pe, device->common_device.dev->name, ibrc_state->dev_ids[devid], portid,
        nvshmemt_ib_common_link_layer_name(device->common_device.port_attr[portid - 1].link_layer),
        device->common_device.port_attr[portid - 1].lid, ep->qp->qp_num, attr.pkey_index,
        attr.qp_access_flags, flags);

    ep->req =
        (struct ibrc_request *)malloc(sizeof(struct ibrc_request) * get_ibrc_qp_depth(ibrc_state));
    NVSHMEMI_NULL_ERROR_JMP(ep->req, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "req allocation failed \n");

    /* Initialize common_ep fields */
    ep->common_ep.head_op_id = 0;
    ep->common_ep.tail_op_id = 0;
    ep->common_ep.transport = t;

    ep->devid = ibrc_state->dev_ids[devid];
    ep->portid = portid;

    // insert qp into map
    qp_map.insert(std::make_pair((unsigned int)ep->qp->qp_num, (long unsigned int)ep));

    *my_ep_ptr = ep;

out:
    if (status) {
        if (ep) {
            free(ep);
        }
    }
    return status;
}

static int ep_connect(struct ibrc_ep *ep, struct nvshmemt_ib_common_ep_handle *ep_handle) {
    int status = 0;
    struct ibv_qp_attr attr;
    int flags;
    int devid = ep->devid;
    int portid = ep->portid;
    nvshmem_transport_t t = (nvshmem_transport_t)ep->common_ep.transport;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);
    struct ibv_port_attr *port_attr = device->common_device.port_attr + (portid - 1);

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr->active_mtu;
    attr.dest_qp_num = ep_handle->qpn;
    attr.rq_psn = 0;

    auto set_grh_fields = [&]() {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.dgid.global.subnet_prefix = ep_handle->spn;
        attr.ah_attr.grh.dgid.global.interface_id = ep_handle->iid;
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.sgid_index = device->common_device.gid_info[portid - 1].local_gid_index;
        attr.ah_attr.grh.hop_limit = IBRC_GRH_HOP_LIMIT;
        attr.ah_attr.grh.traffic_class = ibrc_state->options->IB_TRAFFIC_CLASS;
    };

    if (port_attr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
        struct nvshmemt_ib_qp_path path = nvshmemt_ib_select_qp_path(
            &device->common_device.gid_info[portid - 1].local_gid, port_attr->lid, ep_handle->lid,
            ep_handle->spn, ep_handle->iid);
        attr.ah_attr.dlid = path.dlid;
        /* GRH is needed for GRH-only ports, cross-subnet IB, and ambiguous same-LID paths. */
        if (ibrc_state->options->IB_FORCE_GRH || nvshmemt_ib_common_port_requires_grh(port_attr) ||
            path.grh_required) {
            set_grh_fields();
        } else {
            attr.ah_attr.is_global = 0;
        }
    } else if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {
        status = ib_get_gid_index(&ftable, device->common_device.context, portid, port_attr,
                                  &device->common_device.gid_info[portid - 1].local_gid_index,
                                  ibrc_state->log_level, ibrc_state->options);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "IBRC failed to select GID index for device %s devid %d port %d\n",
                              device->common_device.dev->name, devid, portid);
        status = ftable.query_gid(device->common_device.context, portid,
                                  device->common_device.gid_info[portid - 1].local_gid_index,
                                  &device->common_device.gid_info[portid - 1].local_gid);
        NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "IBRC query_gid failed for device %s devid %d port %d\n",
                                    device->common_device.dev->name, devid, portid);
        set_grh_fields();
    }
    attr.max_dest_rd_atomic = nvshmemt_ibrc_max_rd_atomic;
    attr.min_rnr_timer = 12;
    attr.ah_attr.sl = ibrc_state->options->IB_SL;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = portid;
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
            IBV_QP_MIN_RNR_TIMER | IBV_QP_MAX_DEST_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    if (status) {
        if (attr.ah_attr.is_global) {
            NVSHMEMI_ERROR_JMP(
                status, NVSHMEMX_ERROR_INTERNAL, out,
                "IBRC QP modify INIT->RTR failed: pe %d device %s devid %d port %d link_layer %s "
                "local_qpn %u remote_qpn %u lid %u remote_lid %u remote_gid 0x%llx:0x%llx "
                "gid_index %d path_mtu %d rq_psn %u max_dest_rd_atomic %d min_rnr_timer %d "
                "av_is_global %d av_hop_limit %u sl %d traffic_class %d flags %d status %d (%s)\n",
                t->my_pe, device->common_device.dev->name, devid, portid,
                nvshmemt_ib_common_link_layer_name(port_attr->link_layer), ep->qp->qp_num,
                ep_handle->qpn, port_attr->lid, ep_handle->lid,
                (unsigned long long)attr.ah_attr.grh.dgid.global.subnet_prefix,
                (unsigned long long)attr.ah_attr.grh.dgid.global.interface_id,
                device->common_device.gid_info[portid - 1].local_gid_index, attr.path_mtu,
                attr.rq_psn, attr.max_dest_rd_atomic, attr.min_rnr_timer, attr.ah_attr.is_global,
                attr.ah_attr.grh.hop_limit, attr.ah_attr.sl, attr.ah_attr.grh.traffic_class, flags,
                status, strerror(status));
        } else {
            NVSHMEMI_ERROR_JMP(
                status, NVSHMEMX_ERROR_INTERNAL, out,
                "IBRC QP modify INIT->RTR failed: pe %d device %s devid %d port %d link_layer %s "
                "local_qpn %u remote_qpn %u lid %u remote_lid %u path_mtu %d rq_psn %u "
                "max_dest_rd_atomic %d min_rnr_timer %d av_is_global %d av_dlid %u sl %d "
                "flags %d status %d (%s)\n",
                t->my_pe, device->common_device.dev->name, devid, portid,
                nvshmemt_ib_common_link_layer_name(port_attr->link_layer), ep->qp->qp_num,
                ep_handle->qpn, port_attr->lid, ep_handle->lid, attr.path_mtu, attr.rq_psn,
                attr.max_dest_rd_atomic, attr.min_rnr_timer, attr.ah_attr.is_global,
                attr.ah_attr.dlid, attr.ah_attr.sl, flags, status, strerror(status));
        }
    }

    memset(&attr, 0, sizeof(struct ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    attr.timeout = ibrc_state->options->IB_TIMEOUT;
    attr.retry_cnt = ibrc_state->options->IB_RETRY_CNT;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = nvshmemt_ibrc_max_rd_atomic;
    flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_MAX_QP_RD_ATOMIC;

    status = ftable.modify_qp(ep->qp, &attr, flags);
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(
        status, NVSHMEMX_ERROR_INTERNAL, out,
        "IBRC QP modify RTR->RTS failed: pe %d device %s devid %d port %d local_qpn %u "
        "remote_qpn %u sq_psn %u timeout %d retry_cnt %d rnr_retry %d max_rd_atomic %d "
        "flags %d\n",
        t->my_pe, device->common_device.dev->name, devid, portid, ep->qp->qp_num, ep_handle->qpn,
        attr.sq_psn, attr.timeout, attr.retry_cnt, attr.rnr_retry, attr.max_rd_atomic, flags);

    // register and post receive buffer pool
    if (!device->bpool_mr) {
        device->bpool_mr = ftable.reg_mr(
            device->common_device.pd, bpool, bpool_size * sizeof(ibrc_buf_t),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        NVSHMEMI_NULL_ERROR_JMP(device->bpool_mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");

        assert(device->srq != NULL);

        status = refill_srq(device, ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_srq failed \n");
    }

    connected_qp_count++;
out:
    return status;
}

int ep_get_handle(struct nvshmemt_ib_common_ep_handle *ep_handle, struct ibrc_ep *ep) {
    int status = 0;
    nvshmem_transport_t t = (nvshmem_transport_t)ep->common_ep.transport;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + ep->devid);

    ep_handle->lid = device->common_device.port_attr[ep->portid - 1].lid;
    ep_handle->qpn = ep->qp->qp_num;
    /* Always store GID info so IB peers with GRH routing can exchange it. */
    ep_handle->spn = device->common_device.gid_info[ep->portid - 1].local_gid.global.subnet_prefix;
    ep_handle->iid = device->common_device.gid_info[ep->portid - 1].local_gid.global.interface_id;

    return status;
}

int nvshmemt_ibrc_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length,
                                 nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)transport->state;
    struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices +
                                  ibrc_state->dev_ids[ibrc_state->selected_dev_id]);
    std::unique_ptr<ibrc_mem_handle_info_t> handle_info;
    struct nvshmemt_ib_common_mem_handle *handle;
    bool is_sysmem = false;
    const auto cache_granularity = 1ULL << t->log2_cumem_granularity;
    auto *cached_ptr_end = static_cast<char *>(buf);

    /*
     * In cases where same physical memory has been mapped to multiple VAs (say VA1 and VA2)
     * e.g., user buffers mmapped into symmetric heap. Using VA2 for buffer registration
     * and gdrcopy.pin_buffer is unsupported in RM/nv-p2p (Ref nvbug 507809).
     * We need to use VA1 (first mapped address mapped) as a work around.
     * For an mmapped buffer, buf is VA2, we track the VA2->VA1 mapping during mmap call
     * alias_va_ptr will hold the VA1 address, if applicable
     */
    void *alias_va_ptr = NULL;
    if (transport->alias_va_map != NULL && transport->alias_va_map->count(buf)) {
        INFO(ibrc_state->log_level, "IBRC: alias va found for buf: %p, alias va: %p", buf,
             transport->alias_va_map->operator[](buf));
        alias_va_ptr = transport->alias_va_map->operator[](buf);
    }

    INFO(ibrc_state->log_level, "[%d] IBRC: device used %s, data_direct support: %d",
         transport->my_pe, device->common_device.dev->name, device->common_device.data_direct);
    status = nvshmemt_ib_common_reg_mem_handle(
        &ftable, &mlx5dv_ftable, device->common_device.pd, mem_handle, buf, length, local_only,
        ibrc_state->dmabuf_support, ibrc_state->table, ibrc_state->log_level,
        ibrc_state->options->IB_ENABLE_RELAXED_ORDERING, device->common_device.data_direct,
        alias_va_ptr);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory handle.");

    handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;

    if (!local_only) {
        handle_info = std::make_unique<ibrc_mem_handle_info_t>();

        handle_info->mr = handle->mr;
        handle_info->ptr = buf;
        handle_info->size = length;
    }

    /* For SYSMEM heap, the buffer is directly CPU-accessible without GDRCopy.
     * Set cpu_ptr so the send-based atomic path can use it, and skip GDRCopy
     * pin (which would fail on system memory). */
    if (!local_only && handle_info) {
        cudaPointerAttributes attrs;
        if (cudaPointerGetAttributes(&attrs, buf) == cudaSuccess &&
            attrs.type != cudaMemoryTypeDevice) {
            handle_info->cpu_ptr = buf;
            if (!use_cpu_atomics) {
                use_cpu_atomics = true;
                ibrc_state->ib_transport_ftable->progress_recv = progress_recv_wrapper;
            }
            is_sysmem = true;
        }
    }

#ifdef NVSHMEM_USE_GDRCOPY
    /* we track if the memory handle is EGM based so that GDRCOPY can be disabled*/
    is_egm = check_egm(buf, transport->egm_map);
    if (use_gdrcopy && !local_only && !is_egm && !is_sysmem) {
        void *gdr_buf = buf;

        // if applicable, alias_va_ptr (VA1) is only used for pin_buffer() and
        // computing the offset below off = gdr_buf - info.va
        // We use "buf" (VA2) everywhere else
        if (alias_va_ptr != NULL) {
            gdr_buf = alias_va_ptr;
        }

        auto &mapping = handle_info->gdrcopy;
        status = gdrcopy_ftable.pin_buffer(gdr_desc, reinterpret_cast<unsigned long>(gdr_buf),
                                           length, 0, 0, &mapping.mh);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy pin_buffer failed \n");
        mapping.pinned = true;

        status = gdrcopy_ftable.map(gdr_desc, mapping.mh, &mapping.cpu_ptr_base, length);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy map failed \n");
        mapping.mapped = true;

        gdr_info_t info;
        status = gdrcopy_ftable.get_info(gdr_desc, mapping.mh, &info);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy get_info failed \n");

        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        const auto off = reinterpret_cast<uintptr_t>(gdr_buf) - info.va;
        handle_info->cpu_ptr =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mapping.cpu_ptr_base) + off);
    }
#endif

    /* The memory handle cache is only used with GDRCopy.
     * Local memory is never used with GDRCopy so it doesn't need
     * to go into the cache.
     * This optimization allows us to greatly simplify the lookup of
     * mem handle info when using the dynamic heap.
     */
    if (!local_only) {
        // get_mem_handle is called on chunk size upto 2GB (MAX_HANDLE_LEN) but
        // is tracked at cumem granularity (typically 512 MB), so if buffer size
        // is > 512 MB, we need to split them to ensure entries beyond first 512 MB
        // are added to cache.
        auto *curr_ptr = static_cast<char *>(buf);
        do {
            if (!ibrc_state->cache) {
                status = nvshmemt_mem_handle_cache_init(t, &ibrc_state->cache);
                NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                      "Unable to initialize mem handle cache in IB transport.");
            }
            status =
                nvshmemt_mem_handle_cache_add(t, ibrc_state->cache, curr_ptr, handle_info.get());
            NVSHMEMI_NZ_ERROR_JMP(status, status, out,
                                  "Unable to cache mem handle in IB transport.");
            cached_ptr_end = curr_ptr + cache_granularity;
            curr_ptr += cache_granularity;
        } while (curr_ptr < static_cast<char *>(buf) + length);
    }

    if (!dummy_local_mem) {
        dummy_local_mem = (ibrc_mem_handle_info_t *)malloc(sizeof(ibrc_mem_handle_info_t));
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "dummy_local_mem allocation failed\n");

        nvshmemi_ib_malloc(&dummy_local_mem->ptr, sizeof(uint64_t), ibrc_state->log_level);
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem->ptr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "dummy_mem allocation failed\n");

        dummy_local_mem->mr =
            ftable.reg_mr(device->common_device.pd, dummy_local_mem->ptr, sizeof(uint64_t),
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
        NVSHMEMI_NULL_ERROR_JMP(dummy_local_mem->mr, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "mem registration failed \n");
    }
out:
    if (status) {
#ifdef NVSHMEM_USE_GDRCOPY
        if (handle_info) {
            (void)nvshmemt_ibrc_release_gdrcopy_mapping(*handle_info);
        }
#endif
        if (handle_info) {
            if (!local_only && ibrc_state->cache != nullptr) {
                for (auto *cached_ptr = static_cast<char *>(buf); cached_ptr < cached_ptr_end;
                     cached_ptr += cache_granularity) {
                    nvshmemt_mem_handle_cache_remove(t, ibrc_state->cache, cached_ptr);
                }
            }
        }
        nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, ibrc_state->log_level);
    } else if (handle_info) {
        (void)handle_info.release();
    }
    return status;
}

int nvshmemt_ibrc_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    struct nvshmemt_ib_common_mem_handle *handle;
    struct ibrc_mem_handle_info *handle_info = nullptr;
    nvshmemt_ib_common_state_t state;
    void *addr;
    int status = 0;
    std::unique_ptr<ibrc_mem_handle_info_t> handle_info_owner;

    state = (nvshmemt_ib_common_state_t)t->state;
    handle = (struct nvshmemt_ib_common_mem_handle *)mem_handle;
    addr = handle->buf;

    if (!handle->local_only) {
        handle_info =
            (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get(t, state->cache, addr);
    }

    status = nvshmemt_ib_common_release_mem_handle(&ftable, mem_handle, state->log_level);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Unable to dereg memory.\n");
    if (handle_info) {
        handle_info->mr = nullptr;
    }

    if (handle_info) {
#ifdef NVSHMEM_USE_GDRCOPY
        /* we track if the memory handle is EGM based so that GDRCOPY can be disabled*/
        is_egm = check_egm(addr, t->egm_map);

        if (use_gdrcopy && !is_egm) {
            status = nvshmemt_ibrc_release_gdrcopy_mapping(*handle_info);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy cleanup failed\n");
        }
#endif

        if (state->cache != nullptr) {
            auto *curr_ptr = static_cast<char *>(addr);
            const auto cache_granularity = 1ULL << t->log2_cumem_granularity;
            const auto *end = static_cast<char *>(addr) + handle_info->size;
            do {
                nvshmemt_mem_handle_cache_remove(t, state->cache, curr_ptr);
                curr_ptr += cache_granularity;
            } while (curr_ptr < end);
        }

        handle_info_owner.reset(handle_info);
    }
out:
    return status;
}

int nvshmemt_ibrc_finalize(nvshmem_transport_t transport) {
    int status = 0;
    size_t mem_handle_cache_size;
    nvshmemt_ib_common_state_t state;
    struct ibrc_mem_handle_info *handle_info, *previous_handle_info = nullptr;
    std::unique_ptr<ibrc_mem_handle_info_t> handle_info_owner;

    state = (nvshmemt_ib_common_state_t)transport->state;
    assert(state != nullptr);
    mem_handle_cache_size = nvshmemt_mem_handle_cache_get_size(state->cache);

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }
    if (state->ep) {
        for (int i = 0; i < state->ep_count; i++) {
            status = ftable.destroy_qp(((struct ibrc_ep *)state->ep[i])->qp);
            NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "ibv_destroy_qp failed \n");
        }
        free(state->ep);
    }

    if (state->cst_ep) {
        status = ftable.destroy_qp(((struct ibrc_ep *)state->cst_ep)->qp);
        NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "ibv_destroy_qp failed \n");
        free(state->cst_ep);
        state->cst_ep = NULL;
    }

    for (size_t i = 0; i < mem_handle_cache_size; i++) {
        handle_info =
            (struct ibrc_mem_handle_info *)nvshmemt_mem_handle_cache_get_by_idx(state->cache, i);
        if (handle_info && handle_info != previous_handle_info) {
#ifdef NVSHMEM_USE_GDRCOPY
            /* we track if the memory handle is EGM based so that GDRCOPY can be disabled*/
            is_egm = check_egm(handle_info->ptr, transport->egm_map);
            if (use_gdrcopy && !is_egm) {
                status = nvshmemt_ibrc_release_gdrcopy_mapping(*handle_info);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "gdrcopy cleanup failed\n");
            }
#endif
            if (handle_info->mr) {
                status = ftable.dereg_mr(handle_info->mr);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_dereg_mr failed \n");
                handle_info->mr = nullptr;
            }
            handle_info_owner.reset(handle_info);
        }
        previous_handle_info = handle_info;
    }

    nvshmemt_mem_handle_cache_fini(state->cache);

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
    }
#endif

    // clear qp map
    qp_map.clear();

    if (dummy_local_mem) {
        status = ftable.dereg_mr(dummy_local_mem->mr);
        NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dereg_mr failed \n");
        free(dummy_local_mem);
        dummy_local_mem = NULL;
    }

    if (bpool != NULL) {
        while (!bpool_free.empty()) bpool_free.pop_back();

        free(bpool);
    }
    bqueue_toprocess.clear();

    status = pthread_mutex_destroy(&ibrc_mutex_send_progress);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

    status = pthread_mutex_destroy(&ibrc_mutex_recv_progress);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_destroy failed\n");

#ifdef NVSHMEM_USE_GDRCOPY
    atomics_received = 0;
    atomics_processed = 0;
    atomics_issued = 0;
    atomics_completed = 0;
    atomics_acked = 0;
#endif
    connected_qp_count = 0;

    if (state->devices) {
        bool device_finalized[MAX_NUM_HCAS] = {};
        for (int i = 0; i < state->n_dev_ids; i++) {
            int dev_id = state->dev_ids[i];
            if (dev_id < 0 || dev_id >= MAX_NUM_HCAS || device_finalized[dev_id]) continue;
            device_finalized[dev_id] = true;

            if (((struct ibrc_device *)state->devices)[dev_id].bpool_mr) {
                status = ftable.dereg_mr(((struct ibrc_device *)state->devices)[dev_id].bpool_mr);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_dereg_mr failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].send_cq) {
                status = ftable.destroy_cq(((struct ibrc_device *)state->devices)[dev_id].send_cq);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_destroy_cq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].recv_cq) {
                status = ftable.destroy_cq(((struct ibrc_device *)state->devices)[dev_id].recv_cq);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_destroy_cq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].srq) {
                status = ftable.destroy_srq(((struct ibrc_device *)state->devices)[dev_id].srq);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_destroy_srq failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].common_device.pd) {
                status = ftable.dealloc_pd(
                    ((struct ibrc_device *)state->devices)[dev_id].common_device.pd);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_dealloc_pd failed \n");
            }
            if (((struct ibrc_device *)state->devices)[dev_id].common_device.context) {
                status = ftable.close_device(
                    ((struct ibrc_device *)state->devices)[dev_id].common_device.context);
                NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_close_device failed \n");
            }
        }
        free(state->devices);
    }
    if (state->dev_ids) {
        free(state->dev_ids);
    }
    if (state->port_ids) {
        free(state->port_ids);
    }
    if (state->options) {
        free(state->options);
    }
    free(state);

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    nvshmemt_ib_common_fini_mlx5dv(&mlx5dv_handle);

out:
    return status;
}

int poll_recv(nvshmemt_ib_common_state_t ibrc_state);

template <typename T>
int perform_gdrcopy_amo(struct ibrc_ep *ep, struct ibrc_atomic_op *op, void *ptr) {
    int status = 0;

    T old_value, new_value = {};
    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this
    // resolved
    // status = gdrcopy_ftable.copy_from_mapping(mh, &old_value, ptr, sizeof(T));
    // NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy from mapping
    // failed\n");
    // assert size is 64-bit or smaller, issued as single tansaction
    static_assert(sizeof(T) <= 8, "static_assert(sizeof(T) >= 8) failed");
    old_value = *((volatile T *)ptr);

    bool is_float = (op->op & NVSHMEMI_AMO_FLOAT_BIT) != 0;
    nvshmemi_amo_t amo_op = (nvshmemi_amo_t)(op->op & ~NVSHMEMI_AMO_FLOAT_BIT);

    switch (amo_op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            /* The static_cast is used to truncate the uint64_t value of swap_add back to its
             * original length */
            new_value = static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            if (is_float) {
                new_value = nvshmemt_float_atomic_add<T>(old_value, op->swap_add);
            } else {
                new_value = old_value + static_cast<T>(op->swap_add);
            }
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ static_cast<T>(op->swap_add);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == static_cast<T>(op->compare)) ? static_cast<T>(op->swap_add)
                                                                   : old_value;
            break;
        }
        case NVSHMEMI_AMO_FETCH: {
            new_value = old_value;
            break;
        }
        default: {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                               "RMA/AMO verb %d not implemented\n", op->op);
        }
    }

    // FIXME: gdrcopy causing duplicate copies for small transfers, using direct LD/ST until this
    // resolved status = gdrcopy_ftable.copy_to_mapping(mh, ptr, (void *)&new_value, sizeof(T));
    // NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr copy to mapping failed\n");
    *((volatile T *)ptr) = new_value;
    STORE_BARRIER();
    {
        nvshmem_transport_t t = (nvshmem_transport_t)ep->common_ep.transport;
        nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;
        struct ibv_send_wr *sr, **bad_sr;
        struct ibv_sge *sge;
        int op_id;
        nvshmemi_amo_t ack;
        g_elem_t ret;

        // wait for one send request to become avaialble on the ep
        assert(get_ibrc_qp_depth(ibrc_state) >= 1);
        uint32_t outstanding_count = (get_ibrc_qp_depth(ibrc_state) - 1);
        while ((ep->common_ep.head_op_id - ep->common_ep.tail_op_id) > outstanding_count) {
            status = progress_send(ibrc_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "progress_send failed, outstanding_count: %d\n",
                                  outstanding_count);

            // already in processing a recv request
            // only poll recv cq
            status = poll_recv(ibrc_state);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "poll_recv failed, outstanding_count: %d\n", outstanding_count);
        }

        op_id = ep->common_ep.head_op_id &
                IBRC_REQUEST_QUEUE_MASK(ibrc_state);  // ep->common_ep.head_op_id % ibrc_qp_depth
        ep->common_ep.head_op_id = ep->common_ep.head_op_id + 1;

        sr = &(ep->req + op_id)->sr;
        bad_sr = &(ep->req + op_id)->bad_sr;
        sge = &(ep->req + op_id)->sge;

        memset(sr, 0, sizeof(ibv_send_wr));
        if (amo_op > NVSHMEMI_AMO_END_OF_NONFETCH) {
            ret.data = ret.flag = 0;
            ret.data = old_value;
            ret.flag = op->retflag;

            sr->next = NULL;
            sr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_END_OF_NONFETCH;
            sr->num_sge = 1;
            sr->sg_list = sge;

            sr->imm_data = (uint32_t)NVSHMEMI_AMO_ACK;
            sr->wr.rdma.remote_addr = (uint64_t)op->retaddr;
            sr->wr.rdma.rkey = op->retrkey;
            sge->length = sizeof(g_elem_t);
            sge->addr = (uintptr_t)&ret;
            sge->lkey = 0;
        } else {
            ack = NVSHMEMI_AMO_ACK;

            sr->next = NULL;
            sr->opcode = IBV_WR_SEND;
            sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            sr->wr_id = NVSHMEMI_AMO_ACK;
            sr->num_sge = 1;
            sr->sg_list = sge;

            // dummy send
            sge->length = sizeof(nvshmemi_amo_t);
            sge->addr = (uintptr_t)&ack;
            sge->lkey = 0;
        }

        status = ibv_post_send(ep->qp, sr, bad_sr);
        NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                    "ibv_post_send failed \n");
    }

out:
    return status;
}

int poll_recv(nvshmemt_ib_common_state_t ibrc_state) {
    int status = 0;
    int n_devs = ibrc_state->n_dev_ids;

    // poll all CQs available
    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->recv_cq) continue;

        int ne = ibv_poll_cq(device->recv_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "ibv_poll_cq failed \n");
        } else if (ne) {
            assert(ne == 1);
            ibrc_buf_t *buf = (ibrc_buf_t *)wc.wr_id;
            if (wc.wc_flags & IBV_WC_WITH_IMM) {
                atomics_acked = atomics_acked + 1;
                TRACE(ibrc_state->log_level, "[%d] atomic acked : %lu \n", getpid(), atomics_acked);
                bpool_free.push_back((void *)buf);
            } else {
                struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
                if (op->op == NVSHMEMI_AMO_ACK) {
                    atomics_acked = atomics_acked + 1;
                    TRACE(ibrc_state->log_level, "[%d] atomic acked : %lu \n", getpid(),
                          atomics_acked);
                    bpool_free.push_back((void *)buf);
                } else {
                    buf->qp_num = wc.qp_num;
                    atomics_received = atomics_received + 1;
                    TRACE(ibrc_state->log_level, "[%d] atomic received, enqueued : %lu \n",
                          getpid(), atomics_received);
                    bqueue_toprocess.push_back((void *)buf);
                }
            }
            device->srq_posted--;
        }

        status = refill_srq(device, ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "refill_sqr failed \n");
    }

out:
    return status;
}

int process_recv(nvshmem_transport_t t, nvshmemt_ib_common_state_t ibrc_state) {
    int status = 0;

    if (!bqueue_toprocess.empty()) {
        ibrc_buf_t *buf = (ibrc_buf_t *)bqueue_toprocess.front();
        struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)buf->qp_num)->second;
        struct ibrc_atomic_op *op = (struct ibrc_atomic_op *)buf->buf;
        ibrc_mem_handle_info_t *mem_handle_info = get_mem_handle_info(t, (void *)op->addr);
        void *ptr = (void *)((uintptr_t)mem_handle_info->cpu_ptr +
                             ((uintptr_t)op->addr - (uintptr_t)mem_handle_info->ptr));

        switch (op->elembytes) {
            case 2:
                perform_gdrcopy_amo<uint16_t>(ep, op, ptr);
                break;
            case 4:
                perform_gdrcopy_amo<uint32_t>(ep, op, ptr);
                break;
            case 8:
                perform_gdrcopy_amo<uint64_t>(ep, op, ptr);
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "invalid element size encountered %u\n", op->elembytes);
        }
        atomics_processed = atomics_processed + 1;
        TRACE(ibrc_state->log_level, "[%d] atomic dequeued and processed : %lu \n", getpid(),
              atomics_processed);

        bqueue_toprocess.pop_front();
        bpool_free.push_back((void *)buf);
    }

out:
    return status;
}

int progress_recv(nvshmem_transport_t t, nvshmemt_ib_common_state_t ibrc_state,
                  nvshmemt_ib_wait_predicate_t wait_predicate) {
    int status = 0;
    uint32_t outstanding_atomics;

    pthread_mutex_lock(&ibrc_mutex_recv_progress);
    if (wait_predicate == NVSHMEMT_IB_COMMON_WAIT_ALL) {
        outstanding_atomics = 0;
    } else if (wait_predicate == NVSHMEMT_IB_COMMON_WAIT_TWO ||
               wait_predicate == NVSHMEMT_IB_COMMON_WAIT_ANY) {
        outstanding_atomics = (ibrc_state->srq_depth / (connected_qp_count + 1));
    } else {
        outstanding_atomics = atomics_issued - atomics_acked;
    }

    do {
        status = poll_recv(ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "poll recv failed \n");

        status = process_recv(t, ibrc_state);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "process recv failed \n");

    } while ((atomics_issued - atomics_acked) > outstanding_atomics);

out:
    pthread_mutex_unlock(&ibrc_mutex_recv_progress);
    return status;
}

int progress_send(nvshmemt_ib_common_state_t ibrc_state) {
    int status = 0;
    int n_devs = ibrc_state->n_dev_ids;

    pthread_mutex_lock(&ibrc_mutex_send_progress);

    for (int i = 0; i < n_devs; i++) {
        struct ibv_wc wc;
        int devid = ibrc_state->dev_ids[i];
        struct ibrc_device *device = ((struct ibrc_device *)ibrc_state->devices + devid);

        if (!device->send_cq) continue;

        int ne = ibv_poll_cq(device->send_cq, 1, &wc);
        if (ne < 0) {
            status = ne;
            NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "ibv_poll_cq failed \n");
        } else if (ne) {
            if (wc.status) {
                status = wc.status;
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                   "ibv_poll_cq failed, completion status: %d (%s)\n", wc.status,
                                   ibrc_wc_status_string(wc.status));
            }

            assert(ne == 1);
            if (wc.wr_id == NVSHMEMI_OP_AMO) {
                atomics_completed++;
                TRACE(ibrc_state->log_level, "[%d] atomic completed : %lu \n", getpid(),
                      atomics_completed);
            }

            struct ibrc_ep *ep = (struct ibrc_ep *)qp_map.find((unsigned int)wc.qp_num)->second;
            ep->common_ep.tail_op_id += ne;
        }
    }

out:
    pthread_mutex_unlock(&ibrc_mutex_send_progress);
    return status;
}

int nvshmemt_ibrc_progress(nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;

    status = progress_send(ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_send failed, \n");

    if (use_gdrcopy || use_cpu_atomics) {
        status = progress_recv(t, ibrc_state, NVSHMEMT_IB_COMMON_WAIT_NONE);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "progress_recv failed, \n");
    }

out:
    return status;
}

int nvshmemt_ibrc_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb,
                      rma_memdesc_t *remote, rma_memdesc_t *local, rma_bytesdesc_t bytesdesc,
                      int qp_index) {
    int status = 0;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)tcurr->state;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibrc_ep *ep;
    struct ibv_sge *sge;
    int op_id;

    ep = (struct ibrc_ep *)nvshmemt_ib_common_get_ep_from_qp_index(tcurr, qp_index, pe);

    status = nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->common_ep.head_op_id &
            IBRC_REQUEST_QUEUE_MASK(ibrc_state);  // ep->common_ep.head_op_id % ibrc_qp_depth

    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->wr_id = NVSHMEMI_OP_PUT;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->wr.rdma.remote_addr = (uint64_t)remote->ptr;
    assert(remote->handle);
    sr->wr.rdma.rkey = ((struct nvshmemt_ib_common_mem_handle *)remote->handle)->rkey;
    sge->length = bytesdesc.nelems * bytesdesc.elembytes;
    sge->addr = (uintptr_t)local->ptr;
    /* local->handle is unset for p operations since they are sent by value. */
    if (likely(local->handle != NULL)) {
        sge->lkey = ((struct nvshmemt_ib_common_mem_handle *)local->handle)->lkey;
    }
    if (verb.desc == NVSHMEMI_OP_P) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        sr->send_flags |= IBV_SEND_INLINE;
        TRACE(ibrc_state->log_level, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else if (verb.desc == NVSHMEMI_OP_GET || verb.desc == NVSHMEMI_OP_G) {
        sr->opcode = IBV_WR_RDMA_READ;
        TRACE(ibrc_state->log_level, "[GET] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        sr->opcode = IBV_WR_RDMA_WRITE;
        TRACE(ibrc_state->log_level, "[PUT] remote_addr %p addr %p rkey %d lkey %d length %x",
              (void *)sr->wr.rdma.remote_addr, (void *)sge->addr, sr->wr.rdma.rkey, sge->lkey,
              sge->length);
    } else {
        NVSHMEMI_ERROR_PRINT("RMA/AMO verb not implemented\n");
        exit(-1);
    }

    TRACE(ibrc_state->log_level, "[%d] ibrc post_send dest handle %p rkey %x src handle %p lkey %x",
          getpid(), remote->handle, sr->wr.rdma.rkey, local->handle, sge->lkey);
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->common_ep.head_op_id = ep->common_ep.head_op_id + 1;

    if (unlikely(!verb.is_nbi && verb.desc != NVSHMEMI_OP_P)) {
        nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL /*1*/);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");
    }
    // end of post send sr thourgh ep qp

out:
    return status;
}

int nvshmemt_ibrc_amo(struct nvshmem_transport *tcurr, int pe, void * /*curetptr*/, amo_verb_t verb,
                      amo_memdesc_t *remote, amo_bytesdesc_t bytesdesc, int qp_index) {
    int status = 0;
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)tcurr->state;
    struct ibrc_ep *ep;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    int op_id;
    struct ibrc_atomic_op op;

    ep = (struct ibrc_ep *)nvshmemt_ib_common_get_ep_from_qp_index(tcurr, qp_index, pe);

    status = nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ANY);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    op_id = ep->common_ep.head_op_id &
            IBRC_REQUEST_QUEUE_MASK(ibrc_state);  // ep->common_ep.head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    memset(sr, 0, sizeof(ibv_send_wr));
    memset(sge, 0, sizeof(ibv_sge));

    sr->num_sge = 1;
    sr->sg_list = sge;
    sr->wr_id = NVSHMEMI_OP_AMO;
    sr->next = NULL;

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_SIGNAL_ADD) {
            if (bytesdesc.elembytes == 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
                assert(remote->remote_memdesc.handle);
                sr->wr.atomic.rkey =
                    ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;
                sr->wr.atomic.compare_add = remote->val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        }
    }

    /* we track if the memory handle is EGM based so that GDRCOPY can be disabled*/
    is_egm = check_egm(remote->remote_memdesc.ptr, tcurr->egm_map);
    if (is_egm) {
        INFO(ibrc_state->log_level, "IBRC: buf: %p is egm, not using gdrcopy for atomics\n",
             remote->remote_memdesc.ptr);
    }
    // if gdrcopy or cpu-accessible memory is available, use send-based atomics
    // to guarantee atomicity across different ops
    if ((use_gdrcopy || use_cpu_atomics) && !is_egm) {
        ibrc_mem_handle_info_t *mem_handle_info;

        // assuming GDRCopy availability is uniform on all nodes
        op.op = (nvshmemi_amo_t)(verb.desc | (verb.is_float ? NVSHMEMI_AMO_FLOAT_BIT : 0));
        op.addr = remote->remote_memdesc.ptr;
        op.retaddr = remote->retptr;
        op.retflag = remote->retflag;
        op.compare = remote->cmp;
        op.swap_add = remote->val;
        op.elembytes = bytesdesc.elembytes;

        // send rkey info
        if (verb.desc > NVSHMEMI_AMO_END_OF_NONFETCH) {
            mem_handle_info = get_mem_handle_info(tcurr, remote->retptr);
            op.retrkey = mem_handle_info->mr->rkey;
        }

        sr->opcode = IBV_WR_SEND;
        sr->send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
        sge->length = sizeof(struct ibrc_atomic_op);
        assert(sge->length <= IBRC_BUF_SIZE);
        sge->addr = (uintptr_t)&op;
        sge->lkey = 0;

        atomics_issued = atomics_issued + 1;
        TRACE(ibrc_state->log_level, "[%d] atomic issued : %lu \n", getpid(), atomics_issued);
        goto post_op;
    }

    if (use_ib_native_atomics) {
        if (verb.desc == NVSHMEMI_AMO_ADD) {
            if (bytesdesc.elembytes == 8) {
                sr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
                sr->send_flags = IBV_SEND_SIGNALED;

                sr->wr.atomic.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
                assert(remote->remote_memdesc.handle);
                sr->wr.atomic.rkey =
                    ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;
                sr->wr.atomic.compare_add = remote->val;

                sge->length = bytesdesc.elembytes;
                sge->addr = (uintptr_t)dummy_local_mem->ptr;
                sge->lkey = dummy_local_mem->mr->lkey;
                goto post_op;
            }
        } else if (verb.desc == NVSHMEMI_AMO_SIGNAL || verb.desc == NVSHMEMI_AMO_SIGNAL_SET) {
            sr->opcode = IBV_WR_RDMA_WRITE;
            sr->send_flags = IBV_SEND_SIGNALED;
            sr->send_flags |= IBV_SEND_INLINE;

            sr->wr.rdma.remote_addr = (uint64_t)remote->remote_memdesc.ptr;
            assert(remote->remote_memdesc.handle);
            sr->wr.rdma.rkey =
                ((struct nvshmemt_ib_common_mem_handle *)remote->remote_memdesc.handle)->rkey;

            sge->length = bytesdesc.elembytes;
            sge->addr = (uintptr_t)&remote->val;
            sge->lkey = 0;
            goto post_op;
        }
    }

    NVSHMEMI_ERROR_EXIT("RMA/AMO verb %d not implemented\n", verb.desc);

post_op:
    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->common_ep.head_op_id = ep->common_ep.head_op_id + 1;

out:
    return status;
}

int nvshmemt_ibrc_enforce_cst_at_target(struct nvshmem_transport *tcurr) {
    int status = 0;
    ibrc_mem_handle_info_t *mem_handle_info;
    nvshmemt_ib_common_state_t state;

    state = (nvshmemt_ib_common_state_t)tcurr->state;

    // pick the last region that was inserted
    mem_handle_info =
        (ibrc_mem_handle_info_t *)nvshmemt_mem_handle_cache_get_by_idx(state->cache, 0);
    assert(mem_handle_info != NULL);

#ifdef NVSHMEM_USE_GDRCOPY
    /* we track if the memory handle is EGM based so that GDRCOPY can be disabled*/
    is_egm = check_egm(mem_handle_info->ptr, tcurr->egm_map);
    if (use_gdrcopy && !is_egm && mem_handle_info->gdrcopy.mapped) {
        int temp;
        status = gdrcopy_ftable.copy_from_mapping(mem_handle_info->gdrcopy.mh, &temp,
                                                  mem_handle_info->cpu_ptr, sizeof(int));
        if (status == 0) {
            return NVSHMEMX_SUCCESS;
        }
        INFO(state->log_level, "GDRCopy CST read failed (%d), falling back to RDMA read", status);
        status = 0;
    }
#endif

    struct ibrc_ep *ep = (struct ibrc_ep *)state->cst_ep;
    struct ibv_send_wr *sr, **bad_sr;
    struct ibv_sge *sge;
    int op_id;

    op_id = ep->common_ep.head_op_id &
            IBRC_REQUEST_QUEUE_MASK(state);  // ep->common_ep.head_op_id % ibrc_qp_depth
    sr = &(ep->req + op_id)->sr;
    bad_sr = &(ep->req + op_id)->bad_sr;
    sge = &(ep->req + op_id)->sge;

    sr->next = NULL;
    sr->send_flags = IBV_SEND_SIGNALED;
    sr->num_sge = 1;
    sr->sg_list = sge;

    sr->opcode = IBV_WR_RDMA_READ;
    sr->wr.rdma.remote_addr = (uint64_t)mem_handle_info->ptr;
    sr->wr.rdma.rkey = mem_handle_info->mr->rkey;

    sge->length = sizeof(int);
    sge->addr = (uintptr_t)mem_handle_info->ptr;
    sge->lkey = mem_handle_info->mr->lkey;

    status = ibv_post_send(ep->qp, sr, bad_sr);
    NVSHMEMT_ERRNO_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_post_send failed \n");

    ep->common_ep.head_op_id = ep->common_ep.head_op_id + 1;

    status = nvshmemt_ib_common_check_poll_avail(tcurr, ep, NVSHMEMT_IB_COMMON_WAIT_ALL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

out:
    return status;
}

int nvshmemt_ibrc_ep_create(struct ibrc_ep **ep, int devid, nvshmem_transport_t t,
                            nvshmemt_ib_common_state_t ibrc_state) {
    int status = 0;

    status = ep_create((void **)ep, devid, t);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_create failed\n");

    // setup loopback connection on the first device used.
    if (!ibrc_state->cst_ep) {
        status = nvshmemt_ib_common_setup_cst_loopback(devid, t);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cst setup failed \n");
    }

out:
    return status;
}

int nvshmemt_ibrc_ep_get_handle(struct nvshmemt_ib_common_ep_handle *ep_handle_ptr,
                                struct ibrc_ep *ep) {
    int status = 0;

    status = ep_get_handle(ep_handle_ptr, ep);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_get_handle failed \n");

out:
    return status;
}

int nvshmemt_ibrc_ep_destroy(struct ibrc_ep *ep) {
    int status = 0;

    status = nvshmemt_ib_common_check_poll_avail(ep->common_ep.transport,
                                                 (nvshmemt_ib_common_ep_ptr_t)ep,
                                                 NVSHMEMT_IB_COMMON_WAIT_ALL /*1*/);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "check_poll failed \n");

    // TODO: clean up qp, cq, etc.

out:
    return status;
}

int nvshmemt_ibrc_ep_connect(struct ibrc_ep *ep, struct nvshmemt_ib_common_ep_handle *ep_handle) {
    int status = 0;

    status = ep_connect(ep, ep_handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ep_connect failed \n");

out:
    return status;
}

// Wrapper functions to match the function pointer signatures
int nvshmemt_ibrc_ep_create_wrapper(nvshmemt_ib_common_ep_ptr_t *ep, int dev_id,
                                    struct nvshmem_transport *t) {
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)t->state;
    return nvshmemt_ibrc_ep_create((struct ibrc_ep **)ep, dev_id, t, ibrc_state);
}

int nvshmemt_ibrc_ep_get_handle_wrapper(struct nvshmemt_ib_common_ep_handle *ep_handle,
                                        nvshmemt_ib_common_ep_ptr_t ep) {
    return nvshmemt_ibrc_ep_get_handle(ep_handle, (struct ibrc_ep *)ep);
}

int nvshmemt_ibrc_ep_connect_wrapper(nvshmemt_ib_common_ep_ptr_t ep,
                                     struct nvshmemt_ib_common_ep_handle *ep_handle) {
    return nvshmemt_ibrc_ep_connect((struct ibrc_ep *)ep, ep_handle);
}

int progress_recv_wrapper(nvshmem_transport_t tcurr, nvshmemt_ib_wait_predicate_t wait_predicate) {
    nvshmemt_ib_common_state_t ibrc_state = (nvshmemt_ib_common_state_t)tcurr->state;
    return progress_recv(tcurr, ibrc_state, wait_predicate);
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    int status = 0;
    struct nvshmem_transport *transport = NULL;
    nvshmemt_ib_common_state_t ibrc_state = NULL;
    struct ibv_device **dev_list = NULL;
    int num_devices;
    struct nvshmemt_ib_hca_filter hca_filter;
    connected_qp_count = 0;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized =
        false; /* set it to true after everything has been successfully initialized */

    ibrc_state = (nvshmemt_ib_common_state_t)calloc(1, sizeof(struct nvshmemt_ib_common_state));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");

    /* set selected device ID to -1 to indicate none is selected. */
    ibrc_state->selected_dev_id = -1;
    transport->state = (void *)ibrc_state;

    ibrc_state->options = (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->options, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate options.");

    status = nvshmemi_env_options_init(ibrc_state->options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize transport options.");

    nvshmemt_ib_common_sanitize_timeout(ibrc_state->options);
    nvshmemt_ib_common_sanitize_retry_cnt(ibrc_state->options);

    ibrc_state->log_level = nvshmemt_common_get_log_level(ibrc_state->options);

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibrc_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping devx transport.");
    }

    nvshmemt_ib_common_init_mlx5dv(&mlx5dv_handle, &mlx5dv_ftable,
                                   ibrc_state->options->DISABLE_DATA_DIRECT, ibrc_state->log_level);

    ftable.fork_init();

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibrc_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibrc_device));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibrc_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibrc_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    status = pthread_mutex_init(&ibrc_mutex_send_progress, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

    status = pthread_mutex_init(&ibrc_mutex_recv_progress, NULL);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "pthread_mutex_init failed \n");

#ifdef NVSHMEM_USE_GDRCOPY
    if (ibrc_state->options->DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   ibrc_state->log_level);
    }
    if (use_gdrcopy) {
        use_cpu_atomics = true;
    }
#else
#endif

    ibrc_state->table = table;
    ibrc_state->ib_transport_ftable =
        (nvshmemt_ib_common_ftable *)calloc(1, sizeof(struct nvshmemt_ib_common_ftable));
    NVSHMEMI_NULL_ERROR_JMP(ibrc_state->ib_transport_ftable, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            out, "p2p state allocation failed \n");

    ibrc_state->ib_transport_ftable->ep_create = nvshmemt_ibrc_ep_create_wrapper;
    ibrc_state->ib_transport_ftable->ep_get_handle = nvshmemt_ibrc_ep_get_handle_wrapper;
    ibrc_state->ib_transport_ftable->ep_connect = nvshmemt_ibrc_ep_connect_wrapper;
    ibrc_state->ib_transport_ftable->progress = nvshmemt_ibrc_progress;
    // progress_recv is registered when send-based atomics are available
    // (GDRCopy or SYSMEM). For SYSMEM, this is set later in get_mem_handle.
    ibrc_state->ib_transport_ftable->progress_recv = NULL;
#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        ibrc_state->ib_transport_ftable->progress_recv = progress_recv_wrapper;
    }
#endif

    if (ibrc_state->options->DISABLE_IB_NATIVE_ATOMICS) {
        use_ib_native_atomics = 0;
    }
    ibrc_state->qp_depth = ibrc_state->options->QP_DEPTH;
    ibrc_state->srq_depth = ibrc_state->options->SRQ_DEPTH;
    // qp_depth and srq_depth are now accessed directly from ibrc_state

    status = nvshmemt_ib_common_parse_hca_filter(hca_filter, *ibrc_state);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                          "HCA filter parsing failed.\n");
    transport->device_assignment_mode = nvshmemt_ib_common_device_assignment_mode(hca_filter);

    status = nvshmemt_ib_common_enumerate_devices(&ftable, *ibrc_state, sizeof(struct ibrc_device),
                                                  hca_filter, dev_list, num_devices);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Device enumeration failed.\n");

    for (int i = 0; i < ibrc_state->n_dev_ids; i++) {
        struct ibrc_device *dev =
            &((struct ibrc_device *)ibrc_state->devices)[ibrc_state->dev_ids[i]];
        nvshmemt_ibrc_max_rd_atomic =
            std::min(nvshmemt_ibrc_max_rd_atomic, dev->common_device.device_attr.max_qp_rd_atom);
    }

    nvshmemt_ib_common_log_device_assignment(*ibrc_state);

    status = nvshmemt_ib_common_discover_pci_paths(
        transport, *ibrc_state, sizeof(struct ibrc_device), &ftable, &mlx5dv_ftable);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PCI path discovery failed.\n");

    for (int i = 0; i < transport->n_devices; i++) {
        if (((struct ibrc_device *)ibrc_state->devices)[ibrc_state->dev_ids[i]]
                .common_device.data_direct &&
            !ibrc_state->options->IB_NUM_RC_PER_DEVICE_provided) {
            ibrc_state->options->IB_NUM_RC_PER_DEVICE = 8;
            INFO(ibrc_state->log_level,
                 "Setting IB_NUM_RC_PER_DEVICE = 8 as data direct device is detected");
        }
    }

    nvshmemt_ib_common_warn_missing_hcas(hca_filter);

    // allocate buffer pool
    bpool_size = ibrc_state->srq_depth;
    nvshmemi_ib_malloc((void **)&bpool, bpool_size * sizeof(ibrc_buf_t), ibrc_state->log_level);
    NVSHMEMI_NULL_ERROR_JMP(bpool, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "buf poll allocation failed \n");
    for (int i = 0; i < bpool_size; i++) {
        bpool_free.push_back((void *)(bpool + i));
    }

    transport->host_ops.can_reach_peer = nvshmemt_ibrc_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ib_common_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibrc_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibrc_release_mem_handle;
    transport->host_ops.rma = nvshmemt_ibrc_rma;
    transport->host_ops.amo = nvshmemt_ibrc_amo;
    transport->host_ops.fence = nvshmemt_ib_common_fence;
    transport->host_ops.quiet = nvshmemt_ib_common_quiet;
    transport->host_ops.finalize = nvshmemt_ibrc_finalize;
    transport->host_ops.show_info = nvshmemt_ibrc_show_info;
    transport->host_ops.progress = nvshmemt_ibrc_progress;
    transport->host_ops.put_signal = nvshmemt_put_signal;

    transport->host_ops.enforce_cst = nvshmemt_ibrc_enforce_cst_at_target;
#if !defined(NVSHMEM_PPC64LE) && !defined(NVSHMEM_AARCH64)
    if (!use_gdrcopy)
#endif
        transport->host_ops.enforce_cst_at_target = nvshmemt_ibrc_enforce_cst_at_target;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

    status = nvshmemt_ib_common_check_dmabuf_support(ibrc_state->dmabuf_support, table,
                                                     ibrc_state->options->IB_DISABLE_DMABUF);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "DMA-BUF support check failed.\n");

out:

    if (status) {
        if (ibrc_state) {
            if (ibrc_state->devices) {
                free(ibrc_state->devices);
            }
            if (ibrc_state->dev_ids) {
                free(ibrc_state->dev_ids);
            }
            if (ibrc_state->port_ids) {
                free(ibrc_state->port_ids);
            }
            if (ibrc_state->options) {
                free(ibrc_state->options);
            }
            free(ibrc_state);
        }
        if (transport) {
            free(transport);
        }
    }
    return status;
}
