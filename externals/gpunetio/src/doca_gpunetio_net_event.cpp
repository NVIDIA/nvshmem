/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <atomic>
#include <new>
#include <set>
#include <endian.h>
#include <string.h>
#include <fcntl.h>
#include <cuda_runtime.h>

#include "doca_gpunetio_log.hpp"
#include "doca_gpunetio_net_event.hpp"
#include "doca_internal.hpp"
#include "doca_verbs_dev.hpp"
#include "doca_verbs_cq.hpp"

enum { DOCA_MLX5_EVENT_TYPE_COMPLETION_EVENTS = 0x0 };

static struct doca_gpu_net_event *event_from_handle(doca_gpu_net_event_t event) {
    return reinterpret_cast<struct doca_gpu_net_event *>(event);
}

static doca_error_t update_cq_arm_dbr(uint32_t *arm_dbr, bool is_gpu_memory, cudaStream_t stream) {
    __be32 dbr_val = htobe32(MLX5_CQ_DB_REQ_NOT_SOL);
    if (is_gpu_memory) {
        cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaMemcpyAsync(arm_dbr, &dbr_val, sizeof(dbr_val), cudaMemcpyHostToDevice, stream));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to copy CQ arm DBR to GPU memory");
            return DOCA_ERROR_DRIVER;
        }
    } else {
        *arm_dbr = dbr_val;
    }
    return DOCA_SUCCESS;
}

static doca_error_t arm_cq(uint32_t cqn, uint64_t *uar_db_reg) {
    __be64 db_val = htobe64((static_cast<uint64_t>(MLX5_CQ_DB_REQ_NOT_SOL) << 32) | cqn);
    *reinterpret_cast<volatile __be64 *>(uar_db_reg) = db_val;
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_net_event_create(doca_dev_t *net_dev, doca_gpu_net_event_t *out_event) {
    doca_error_t status = DOCA_SUCCESS;
    struct doca_gpu_net_event *ev = nullptr;
    int fd_flags = 0;
    int fd_fcntl_status = 0;

    if (net_dev == nullptr || out_event == nullptr) {
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    if (net_dev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if (net_dev->open == nullptr || net_dev->open->get_ctx() == nullptr) {
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    try {
        ev = new struct doca_gpu_net_event();
    } catch (...) {
        status = DOCA_ERROR_NO_MEMORY;
        goto out;
    }

    status = doca_verbs_wrapper_mlx5dv_devx_create_event_channel(
        net_dev->open->get_ctx(), MLX5DV_DEVX_CREATE_EVENT_CHANNEL_FLAGS_OMIT_EV_DATA,
        &ev->channel);
    if (status) goto out;

    fd_flags = fcntl(ev->channel->fd, F_GETFL);
    if (fd_flags < 0) {
        status = DOCA_ERROR_OPERATING_SYSTEM;
        goto out;
    }

    fd_fcntl_status = fcntl(ev->channel->fd, F_SETFL, fd_flags | O_NONBLOCK);
    if (fd_fcntl_status < 0) {
        status = DOCA_ERROR_OPERATING_SYSTEM;
        goto out;
    }

    *out_event = reinterpret_cast<doca_gpu_net_event_t>(ev);

out:
    if (status) {
        if (ev) {
            if (ev->channel) {
                doca_verbs_wrapper_mlx5dv_devx_destroy_event_channel(ev->channel);
                ev->channel = nullptr;
            }
            delete ev;
        }
    }
    return status;
}

doca_error_t doca_gpu_net_event_subscribe(doca_gpu_net_event_t event,
                                          struct doca_gpu_verbs_qp *qps[], unsigned int num_qps) {
    doca_error_t status = DOCA_SUCCESS;
    cudaError_t cuda_status = cudaSuccess;

    struct doca_gpu_net_event *ev = nullptr;

    if (event == nullptr || qps == nullptr || num_qps == 0) return DOCA_ERROR_INVALID_VALUE;

    ev = event_from_handle(event);
    if (ev->degraded) return DOCA_ERROR_BAD_STATE;

    cudaStream_t stream = nullptr;
    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
        status = DOCA_ERROR_DRIVER;
        goto out;
    }

    for (unsigned int i = 0; i < num_qps; ++i) {
        struct doca_gpu_verbs_qp *qp = qps[i];
        doca_verbs_cq_t *cq_sq = nullptr;

        cudaPointerAttributes attributes;
        bool is_device_ptr = false;

        uint32_t *ci_dbr = nullptr;
        uint32_t *arm_dbr = nullptr;
        uint64_t *uar_db_reg = nullptr;

        if (qp == nullptr) {
            status = DOCA_ERROR_INVALID_VALUE;
            goto out;
        }

        if (ev->qps.find(qp) != ev->qps.end()) continue;

        cq_sq = qp->cq_sq;
        if (cq_sq == nullptr) {
            status = DOCA_ERROR_INVALID_VALUE;
            goto out;
        }

        status = doca_verbs_cq_get_dbr_addr(cq_sq, &uar_db_reg, &ci_dbr, &arm_dbr);
        if (status) goto out;

        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
            cudaPointerGetAttributes(&attributes, static_cast<const void *>(arm_dbr)));
        if (cuda_status != cudaSuccess && cuda_status != cudaErrorInvalidValue) {
            // cudaErrorInvalidValue is expected if the pointer was not allocated in, mapped by or
            // registered with context supporting unified addressing. In this case, the pointer is
            // not a device pointer.
            DOCA_LOG(LOG_ERR, "Failed to get CUDA pointer attributes");
            status = DOCA_ERROR_DRIVER;
            goto out;
        } else if (cuda_status == cudaSuccess && attributes.type == cudaMemoryTypeDevice)
            is_device_ptr = true;

        status = update_cq_arm_dbr(arm_dbr, is_device_ptr, stream);
        if (status) goto out;
    }

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamSynchronize(stream));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to synchronize GPU CQ arm DBR write");
        status = DOCA_ERROR_DRIVER;
        goto out;
    }
    std::atomic_thread_fence(std::memory_order_release);

    for (unsigned int i = 0; i < num_qps; ++i) {
        struct doca_gpu_verbs_qp *qp = qps[i];
        doca_verbs_cq_t *cq_sq = qp->cq_sq;

        struct mlx5dv_devx_obj *cq_obj = nullptr;
        uint32_t cqn = 0;

        uint32_t *ci_dbr = nullptr;
        uint32_t *arm_dbr = nullptr;
        uint64_t *uar_db_reg = nullptr;

        uint16_t event_num = DOCA_MLX5_EVENT_TYPE_COMPLETION_EVENTS;

        if (ev->qps.find(qp) != ev->qps.end()) continue;

        if (cq_sq->type != DOCA_VERBS_SDK_LIB_TYPE_OPEN) {
            status = DOCA_ERROR_NOT_SUPPORTED;
            goto out;
        }
        if (cq_sq->open == nullptr) {
            status = DOCA_ERROR_INVALID_VALUE;
            goto out;
        }
        cq_obj = cq_sq->open->get_cq_obj();

        status = doca_verbs_cq_get_cqn(cq_sq, &cqn);
        if (status) goto out;

        status = doca_verbs_cq_get_dbr_addr(cq_sq, &uar_db_reg, &ci_dbr, &arm_dbr);
        if (status) goto out;

        try {
            ev->qps.insert(qp);
        } catch (...) {
            status = DOCA_ERROR_NO_MEMORY;
            goto out;
        }
        qp->refcount++;

        status = doca_verbs_wrapper_mlx5dv_devx_subscribe_devx_event(
            ev->channel, cq_obj, sizeof(event_num), &event_num, reinterpret_cast<uint64_t>(qp));
        if (status) goto out;

        status = arm_cq(cqn, uar_db_reg);
        if (status) goto out;
    }

out:
    if (status) {
        // There is no MLX5 API for unsubscribing events. The caller should destroy the event object
        // to cleanup.
        ev->degraded = true;
    }
    if (stream) {
        cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamDestroy(stream));
        if (cuda_status != cudaSuccess) {
            DOCA_LOG(LOG_ERR, "Failed to destroy CUDA stream");
            status = DOCA_ERROR_DRIVER;
        }
    }
    return status;
}

doca_error_t doca_gpu_net_event_get(doca_gpu_net_event_t event, struct doca_gpu_verbs_qp **out_qp) {
    doca_error_t status = DOCA_SUCCESS;
    ssize_t bytes_read = 0;

    if (event == nullptr || out_qp == nullptr) return DOCA_ERROR_INVALID_VALUE;

    *out_qp = nullptr;

    auto *ev = event_from_handle(event);
    struct doca_gpu_verbs_qp *qp = nullptr;
    if (ev->degraded) return DOCA_ERROR_BAD_STATE;

    struct mlx5dv_devx_async_event_hdr event_hdr {
        0
    };

    status = doca_verbs_wrapper_mlx5dv_devx_get_event(ev->channel, &event_hdr, sizeof(event_hdr),
                                                      &bytes_read);
    if (status) goto out;

    if (bytes_read < 0) {
        if (bytes_read == -1)
            status = DOCA_SUCCESS;
        else
            status = DOCA_ERROR_DRIVER;
        goto out;
    } else if ((size_t)bytes_read < sizeof(event_hdr)) {
        status = DOCA_ERROR_UNEXPECTED;
        goto out;
    }

    qp = reinterpret_cast<struct doca_gpu_verbs_qp *>(event_hdr.cookie);
    if (ev->qps.find(qp) == ev->qps.end()) {
        status = DOCA_ERROR_UNEXPECTED;
        goto out;
    }

    *out_qp = qp;

out:
    return status;
}

doca_error_t doca_gpu_net_event_destroy(doca_gpu_net_event_t event) {
    doca_error_t status = DOCA_SUCCESS;

    if (event == nullptr) return DOCA_ERROR_INVALID_VALUE;

    auto *ev = event_from_handle(event);
    if (ev->channel != nullptr) {
        status = doca_verbs_wrapper_mlx5dv_devx_destroy_event_channel(ev->channel);
    }

    if (status == DOCA_SUCCESS) {
        ev->channel = nullptr;
        for (auto qp : ev->qps) {
            qp->refcount--;
        }
        ev->qps.clear();
        delete ev;
    } else {
        ev->degraded = true;
    }
    return status;
}
