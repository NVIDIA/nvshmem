/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/nvshmem_api.h"  // IWYU pragma: keep
#include <cuda_runtime.h>
#include <driver_types.h>

#include "internal/host/nvshmem_internal.h"
#include "internal/host/nvshmemi_region.h"
#include "internal/host/nvshmemi_types.h"
#include "internal/host/nvshmem_nvtx.hpp"
#include "non_abi/c/nvshmemi_error_macros.h"
#include "internal/host_transport/transport.h"
#include "internal/host/util.h"

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
/*
 * nvshmem_fence() defers completion of outstanding CFT work by recording an event and marking a
 * per-PE dependency. This helper orders each later operation after that deferred work:
 *  - the same internal stream is already ordered;
 *  - another CUDA stream waits on the event recorded by fence;
 *  - a host operation synchronizes this PE's recorded fence event.
 *
 * CUDA operations keep the dependency pending for later operations. A host synchronization or
 * nvshmem_quiet() completes the CFT work and clears it.
 */
int nvshmemi_order_after_cft_fence(int pe, bool next_on_internal_stream, bool has_next_stream,
                                   cudaStream_t next_stream) {
    if (!nvshmemi_state->cft_fence_pending[pe]) {
        return NVSHMEMX_SUCCESS;
    }

    int status = NVSHMEMX_SUCCESS;
    const int distance = (pe - nvshmemi_state->mype + nvshmemi_state->npes) % nvshmemi_state->npes;
    const int stream_idx = distance % nvshmemi_options.MAX_PEER_STREAMS;
    cudaStream_t internal_stream = nvshmemi_state->custreams[stream_idx];

    if (next_on_internal_stream) {
        next_stream = internal_stream;
        has_next_stream = true;
    }

    if (has_next_stream) {
        if (next_stream != internal_stream) {
            cudaEvent_t fence_event = nvshmemi_state->cft_fence_events[pe];
            CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(next_stream, fence_event, 0), status, out);
        }
    } else {
        CUDA_RUNTIME_CHECK_GOTO(cudaEventSynchronize(nvshmemi_state->cft_fence_events[pe]), status,
                                out);
        nvshmemi_state->cft_fence_pending[pe] = false;
    }

out:
    return status;
}
#endif

void nvshmem_fence(void) {
    NVTX_FUNC_RANGE_IN_GROUP(MEMORDER);
    NVSHMEMI_CHECK_INIT_STATUS();

    int status = 0;
    int tbitmap = nvshmemi_state->transport_bitmap;

    status = nvshmemi_region_host_flush_active();
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "region flush failed in nvshmem_fence()\n");
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
    /*
     * Do not wait for outstanding CFT work here. Record an event for each affected PE so later
     * operations can preserve fence ordering without blocking the host. Each PE has a distinct
     * event so a later fence for another PE on the same hashed stream cannot advance this PE's
     * dependency.
     */
    for (int pe = 0; pe < nvshmemi_state->npes; pe++) {
        if (nvshmemi_state->cft_active_pes[pe]) {
            const int distance =
                (pe - nvshmemi_state->mype + nvshmemi_state->npes) % nvshmemi_state->npes;
            const int stream_idx = distance % nvshmemi_options.MAX_PEER_STREAMS;
            CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(nvshmemi_state->cft_fence_events[pe],
                                                    nvshmemi_state->custreams[stream_idx]),
                                    status, out);
            nvshmemi_state->cft_fence_pending[pe] = true;
            nvshmemi_state->cft_active_pes[pe] = false;
        }
    }
#endif
    for (int j = 0; j < nvshmemi_state->num_initialized_transports; j++) {
        if (tbitmap & 1) {
            struct nvshmem_transport *tcurr =
                ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
            if ((tcurr->attr & NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS)) {
                for (int s = 0; s < nvshmemi_options.MAX_PEER_STREAMS; s++) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
                    /* Synchronize only mapped-P2P NBI streams; CFT-only streams use the deferred
                     * event dependency recorded above. */
                    if (!nvshmemi_state->p2p_active_internal_streams[s]) {
                        continue;
                    }
#endif
                    cudaStream_t custrm = nvshmemi_state->custreams[s];
                    CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
                    nvshmemi_state->p2p_active_internal_streams[s] = false;
#endif
                }
            } else if (tcurr->host_ops.fence) {
                for (int k = 0; k < nvshmemi_state->npes; k++) {
                    status = tcurr->host_ops.fence(tcurr, k, 0, NVSHMEMX_QP_HOST);
                    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                          "nvshmem_fence() failed \n");
                }
            }
        }
        tbitmap >>= 1;
    }
out:
    return;
}
