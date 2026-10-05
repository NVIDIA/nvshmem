/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>  // for cudaMemcpyAsync
#include <driver_types.h>  // for cudaStream_t
#include <stddef.h>        // for NULL, size_t
#include <stdint.h>        // for uint64_t
#include <stdlib.h>        // for exit
#include <cstring>
#include <vector>
#include "non_abi/device/common/nvshmemi_runtime_types.h"  // for nvshmemi_device...
#include "non_abi/device/common/nvshmem_common.cuh"        // for NVSHMEMI_REPT_F...
#include "non_abi/device/common/logical_endpoint_types.h"
#include "c_api/nvshmem_api.h"                                  // for nvshmem_char_put
#include "c_api/nvshmemx_api.h"                                 // for nvshmemx_char_g...
#include "non_abi/c/nvshmemi_error_macros.h"                    // for NVSHMEMI_ERROR_...
#include "non_abi/device/transport/nvshmem_common_transport.h"  // for NVSHMEMI_OP_PUT
#include "internal/host/debug.h"                                // for INFO, NVSHMEM_P2P
#include "internal/host/nvshmemi_cft.h"
#include "internal/host/nvshmem_internal.h"  // for NVSHMEMI_CHECK_...
#include "internal/host/nvshmemi_rma_translation.hpp"
#include "internal/host/nvshmem_nvtx.hpp"             // for nvtx_cond_range
#include "internal/host/nvshmemi_symmetric_heap.hpp"  // for nvshmemi_symmet...
#include "internal/host/nvshmemi_types.h"             // for nvshmemi_state
#include "internal/host/util.h"                       // for CUDA_RUNTIME_CH...
#include "internal/host/cuda_interface_sync.h"
#include "internal/host_transport/transport.h"  // for rma_bytesdesc_t
#include "internal/host/nvshmemi_region.h"

#define NOT_A_CUDA_STREAM ((cudaStream_t)0)

enum {
    NO_NBI = 0,
    NBI,
};

enum {
    NO_ASYNC = 0,
    ASYNC,
};

enum {
    SRC_STRIDE_CONTIG = 1,
};

enum {
    DEST_STRIDE_CONTIG = 1,
};

int nvshmemi_proxy_rma_launcher(void *args[], cudaStream_t cstrm, bool is_nbi, bool is_signal);

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
static int nvshmemi_cft_copy(cudaStream_t stream, CUlogicalEndpointId le_id, void *local,
                             size_t remote_base_offset, rma_bytesdesc_t bytesdesc, bool is_put) {
    constexpr size_t max_batch_ops = 1024;
    int status = NVSHMEMX_SUCCESS;
    if (bytesdesc.nelems == 0) {
        return status;
    }

    const bool is_contig = bytesdesc.srcstride == 1 && bytesdesc.deststride == 1;
    const size_t num_ops = is_contig ? 1 : bytesdesc.nelems;
    const size_t op_list_size = num_ops < max_batch_ops ? num_ops : max_batch_ops;
    std::vector<CUDA_FABRIC_COPY_BATCH_OP> op_list(op_list_size);
    size_t submitted_ops = 0;

    do {
        const size_t remaining_ops = num_ops - submitted_ops;
        const size_t batch_size = remaining_ops < max_batch_ops ? remaining_ops : max_batch_ops;

        for (size_t i = 0; i < batch_size; i++) {
            const size_t op_idx = submitted_ops + i;
            const size_t local_offset =
                is_contig ? 0 : op_idx * bytesdesc.srcstride * bytesdesc.elembytes;
            const size_t remote_offset =
                is_contig ? 0 : op_idx * bytesdesc.deststride * bytesdesc.elembytes;
            CUfabricCpyOperand pointer_operand = {};
            CUfabricCpyOperand endpoint_operand = {};

            pointer_operand.type = CU_FABRIC_COPY_OPERAND_TYPE_POINTER;
            pointer_operand.op.ptr.ptr = static_cast<CUdeviceptr>(
                reinterpret_cast<uintptr_t>(static_cast<char *>(local) + local_offset));
            endpoint_operand.type = CU_FABRIC_COPY_OPERAND_TYPE_LOGICAL_ENDPOINT;
            endpoint_operand.op.le.id = le_id;
            endpoint_operand.op.le.offset = remote_base_offset + remote_offset;

            op_list[i].src = is_put ? pointer_operand : endpoint_operand;
            op_list[i].dst = is_put ? endpoint_operand : pointer_operand;
            op_list[i].size =
                is_contig ? bytesdesc.nelems * bytesdesc.elembytes : bytesdesc.elembytes;
            op_list[i].flags = 0;
        }

        status = CUPFN(nvshmemi_cuda_syms, cuFabricCpyBatchAsync)(
            batch_size, op_list.data(), 0, reinterpret_cast<CUstream>(stream));
        if (status == CUDA_ERROR_NOT_SUPPORTED && submitted_ops == 0) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        NVSHMEMI_CU_NZ_ERROR_JMP(nvshmemi_cuda_syms, status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "cuFabricCpyBatchAsync() failed\n");
        submitted_ops += batch_size;
    } while (submitted_ops < num_ops);

out:
    return status;
}

bool nvshmemid_is_cft_p_on_stream_supported(int pe) {
    if (!nvshmemi_state || pe < 0 || pe >= nvshmemi_state->npes || pe == nvshmemi_state->mype ||
        !nvshmemi_state->heap_obj) {
        return false;
    }

    const uint64_t le_id_with_flag = nvshmemi_state->heap_obj->get_unicast_le_id(pe);
    return IS_VALID_LE_ID(le_id_with_flag) && nvshmemi_state->cft_scalar_staging &&
           CUPFN(nvshmemi_cuda_syms, cuFabricCpyBatchAsync) &&
           CUPFN(nvshmemi_cuda_syms, cuStreamWriteValue64);
}

static int nvshmemi_cft_staged_copy(cudaStream_t stream, CUlogicalEndpointId le_id, void *staging,
                                    uint64_t value, size_t remote_base_offset,
                                    rma_bytesdesc_t bytesdesc) {
    int status = CUPFN(nvshmemi_cuda_syms, cuStreamWriteValue64)(
        reinterpret_cast<CUstream>(stream),
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(staging)), value,
        CU_STREAM_WRITE_VALUE_DEFAULT);
    NVSHMEMI_CU_NZ_ERROR_JMP(nvshmemi_cuda_syms, status, NVSHMEMX_ERROR_INTERNAL, out,
                             "staging CFT value failed\n");

    status = nvshmemi_cft_copy(stream, le_id, staging, remote_base_offset, bytesdesc, true);

out:
    return status;
}

static int nvshmemi_cft_rma(cudaStream_t custrm /* internal stream */, cudaEvent_t cuev,
                            rma_verb_t verb, void *local, void *remote, rma_bytesdesc_t bytesdesc,
                            uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
    int status = NVSHMEMX_SUCCESS;
    const uint64_t le_id_with_flag = nvshmemi_state->heap_obj->get_unicast_le_id(pe);
    const bool is_stream_p = verb.is_stream && verb.desc == NVSHMEMI_OP_P;
    const bool is_stream_g = verb.is_stream && verb.desc == NVSHMEMI_OP_G;
    const bool is_stream_scalar = is_stream_p || is_stream_g;
    const bool is_put_signal = verb.desc == NVSHMEMI_OP_PUT_SIGNAL;
    const bool is_signal_set = is_put_signal && sig_op == NVSHMEMI_AMO_SIGNAL_SET;
    const bool is_signal_add = is_put_signal && sig_op == NVSHMEMI_AMO_SIGNAL_ADD;
    const bool is_stream_signal = is_put_signal && verb.is_stream;
    const bool is_supported_scalar_size =
        bytesdesc.elembytes == sizeof(uint8_t) || bytesdesc.elembytes == sizeof(uint16_t) ||
        bytesdesc.elembytes == sizeof(uint32_t) || bytesdesc.elembytes == sizeof(uint64_t);
    const int distance = (pe - nvshmemi_state->mype + nvshmemi_state->npes) % nvshmemi_state->npes;
    const int stream_idx = distance % nvshmemi_options.MAX_PEER_STREAMS;
    void *application_local = local;

    if (!CUPFN(nvshmemi_cuda_syms, cuFabricCpyBatchAsync) || !IS_VALID_LE_ID(le_id_with_flag)) {
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }
    if (is_stream_scalar && (!nvshmemi_state->cft_scalar_staging || !is_supported_scalar_size)) {
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }
    if (is_stream_p && !CUPFN(nvshmemi_cuda_syms, cuStreamWriteValue64)) {
        return NVSHMEMX_ERROR_NOT_SUPPORTED;
    }
    size_t signal_remote_offset = 0;
    if (is_put_signal) {
        if (sig_addr == nullptr || (!is_signal_set && !is_signal_add)) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        signal_remote_offset = reinterpret_cast<uintptr_t>(sig_addr) -
                               reinterpret_cast<uintptr_t>(nvshmemi_device_state.heap_base);
        if (is_signal_set && (!nvshmemi_state->cft_scalar_staging ||
                              !CUPFN(nvshmemi_cuda_syms, cuStreamWriteValue64))) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
        /* Reject an ineligible CFT atomic before submitting the payload so another route can safely
         * handle the complete operation. */
        if (is_signal_add &&
            !nvshmemi_is_le_atomic_implemented_on_host(
                le_id_with_flag, nvshmemi_device_state.tma_policy != NVSHMEMX_TMA_DISABLE,
                nvshmemi_device_state.heap_base, sig_addr)) {
            return NVSHMEMX_ERROR_NOT_SUPPORTED;
        }
    }

    cudaStream_t stream_for_op =
        ((!verb.is_stream) || verb.is_nbi || is_stream_scalar || is_stream_signal) ? custrm
                                                                                   : verb.cstrm;
    const bool is_put = verb.desc == NVSHMEMI_OP_P || verb.desc == NVSHMEMI_OP_PUT ||
                        verb.desc == NVSHMEMI_OP_PUT_SIGNAL;
    const size_t remote_base_offset =
        static_cast<char *>(remote) - static_cast<char *>(nvshmemi_device_state.heap_base);
    const CUlogicalEndpointId le_id =
        static_cast<CUlogicalEndpointId>(PARSE_LE_ID(le_id_with_flag));

    status = nvshmemi_order_after_cft_fence(pe, stream_for_op == custrm, true, stream_for_op);
    if (status != NVSHMEMX_SUCCESS) {
        goto out;
    }

    /*
     * On-stream NBI operations and scalar p/g_on_stream execute on NVSHMEM's internal stream rather
     * than directly on the user-provided stream. Scalar operations have one persistent staging slot
     * per internal stream; using that stream serializes staging and CFT copies so different user
     * streams cannot overwrite the slot concurrently. Record the user's stream and make the
     * internal stream wait so the CFT operation cannot start before work previously enqueued by the
     * application. The event establishes user stream -> internal stream ordering without blocking
     * the host; scalar g_on_stream synchronizes separately below before returning its value.
     */
    if ((verb.is_nbi || is_stream_scalar || is_stream_signal) && verb.is_stream) {
        CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, verb.cstrm), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(custrm, cuev, 0), status, out);
    }
    if (is_stream_scalar) {
        local = nvshmemi_state->cft_scalar_staging + stream_idx;
    }
    if (is_stream_p) {
        uint64_t value = 0;
        std::memcpy(&value, application_local, bytesdesc.elembytes);
        status = nvshmemi_cft_staged_copy(stream_for_op, le_id, local, value, remote_base_offset,
                                          bytesdesc);
    } else {
        status =
            nvshmemi_cft_copy(stream_for_op, le_id, local, remote_base_offset, bytesdesc, is_put);
    }
    if (status != NVSHMEMX_SUCCESS) {
        goto out;
    }

    if (is_put_signal) {
        /* Complete the CFT payload put before issuing the signal update. This synchronization is
         * required so the remote PE cannot observe the signal before the corresponding payload has
         * been delivered. */
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(stream_for_op), status, out);
        if (is_signal_set) {
            const rma_bytesdesc_t signal_bytesdesc = {1, static_cast<int>(sizeof(signal)), 1, 1};
            uint64_t *signal_staging = nvshmemi_state->cft_scalar_staging + stream_idx;
            status = nvshmemi_cft_staged_copy(stream_for_op, le_id, signal_staging, signal,
                                              signal_remote_offset, signal_bytesdesc);
            if (status != NVSHMEMX_SUCCESS) {
                goto out;
            }
        } else {
            call_nvshmemi_cft_signal_add_kernel(sig_addr, signal, pe, stream_for_op);
        }
    }

    if (is_stream_g) {
        /* g_on_stream returns a scalar by value, so copy out of persistent staging and wait while
         * its stack destination is still alive. This can also complete earlier work on custrm. */
        CUDA_RUNTIME_CHECK_GOTO(cudaMemcpyAsync(application_local, local, bytesdesc.elembytes,
                                                cudaMemcpyDeviceToHost, custrm),
                                status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
    }

    /*
     * Scalar p_on_stream must also order subsequent application work after the internally staged
     * CFT copy. Record completion on the internal stream and make the user stream wait, forming the
     * reverse internal stream -> user stream dependency. NBI operations do not add this reverse
     * dependency; their completion is handled by quiet/fence.
     */
    if (is_stream_p || is_stream_signal) {
        CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, custrm), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(verb.cstrm, cuev, 0), status, out);
    }

    if (verb.is_nbi || is_stream_p || is_stream_signal) {
        nvshmemi_state->active_internal_streams[stream_idx] = 1;
        nvshmemi_state->used_internal_streams = 1;
        /*
         * The PE flag lets fence create a destination-specific dependency. The stream flag remains
         * live until host quiet, even if quiet-on-stream clears the generic activity bookkeeping.
         */
        nvshmemi_state->cft_active_pes[pe] = true;
        nvshmemi_state->cft_active_internal_streams[stream_idx] = true;
    } else if (!verb.is_stream) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
    }

out:
    return status;
}
#endif

static int nvshmemi_p2p_rma_optimized(cudaStream_t custrm /* internal stream */, cudaEvent_t cuev,
                                      rma_verb_t verb, rma_memdesc_t *dest, rma_memdesc_t *src,
                                      rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                      uint64_t signal, int sig_op, int pe) {
    int status = 0;
    bool is_contig = ((bytesdesc.srcstride == 1) && (bytesdesc.deststride == 1)) ? true : false;
    bool is_single_word =
        ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_G)) ? true : false;
    if (verb.is_stream) {
        if (verb.is_nbi) {
            CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, verb.cstrm), status, out);
            CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(custrm, cuev, 0), status, out);
            if (is_contig) { /*can include iput,iget in future*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                    cudaMemcpyDeviceToDevice, custrm),
                    status, out);
                if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
                    nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, custrm);
                }
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyHostToDevice, verb.cstrm),
                            status, out);
                    } else { /*!is P*/
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyDeviceToHost, verb.cstrm),
                            status, out);
                    }
                } else { /*!is_single_word*/
                    CUDA_RUNTIME_CHECK_GOTO(
                        cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                        cudaMemcpyDeviceToDevice, verb.cstrm),
                        status, out);
                    if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
                        nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, verb.cstrm);
                    }
                }
            } else { /*!is_contig*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes,
                                      src->ptr, bytesdesc.srcstride * bytesdesc.elembytes,
                                      bytesdesc.elembytes, bytesdesc.nelems,
                                      cudaMemcpyDeviceToDevice, verb.cstrm),
                    status, out);
            } /*is_contig*/
        } /*is_nbi*/
    } else { /*!is_stream*/
        if (verb.is_nbi) {
            if (is_contig) { /*can include iput,iget in future*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                    cudaMemcpyDeviceToDevice, custrm),
                    status, out);
            }
        } else { /*!is_nbi*/
            if (is_contig) {
                if (is_single_word) {
                    if (verb.desc == NVSHMEMI_OP_P) {
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyHostToDevice, custrm),
                            status, out);
                    } else { /*!is P*/
                        CUDA_RUNTIME_CHECK_GOTO(
                            cudaMemcpyAsync(dest->ptr, src->ptr,
                                            bytesdesc.nelems * bytesdesc.elembytes,
                                            cudaMemcpyDeviceToHost, custrm),
                            status, out);
                    }
                } else { /*!is_single_word*/
                    CUDA_RUNTIME_CHECK_GOTO(
                        cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                        cudaMemcpyDeviceToDevice, custrm),
                        status, out);
                }
            } else { /*!is_contig*/
                CUDA_RUNTIME_CHECK_GOTO(
                    cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes,
                                      src->ptr, bytesdesc.srcstride * bytesdesc.elembytes,
                                      bytesdesc.elembytes, bytesdesc.nelems,
                                      cudaMemcpyDeviceToDevice, custrm),
                    status, out);
            } /*is_contig*/
            CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
        } /*is_nbi*/
    } /*is_stream*/

out:
    return status;
}

static int nvshmemi_p2p_rma_registered(cudaStream_t custrm /* internal stream */, cudaEvent_t cuev,
                                       rma_verb_t verb, rma_memdesc_t *dest, rma_memdesc_t *src,
                                       rma_bytesdesc_t bytesdesc, uint64_t *sig_addr,
                                       uint64_t signal, int sig_op, int pe) {
    cudaStream_t stream_for_op;
    int status = 0;

    if ((!verb.is_stream) || verb.is_nbi) {
        stream_for_op = (cudaStream_t)custrm;
    } else {
        stream_for_op = verb.cstrm;
    }

    if (verb.is_nbi && verb.is_stream) {
        CUDA_RUNTIME_CHECK_GOTO(cudaEventRecord(cuev, verb.cstrm), status, out);
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamWaitEvent(custrm, cuev, 0), status, out);
    }

    if ((bytesdesc.srcstride == 1) && (bytesdesc.deststride == 1)) {
        status = cudaMemcpyAsync(dest->ptr, src->ptr, bytesdesc.nelems * bytesdesc.elembytes,
                                 cudaMemcpyDefault, stream_for_op);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuMemcpyAsync() failed\n");
        if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
            nvshmemi_signal_op_on_stream(sig_addr, signal, sig_op, pe, stream_for_op);
        }
    } else {
        status = cudaMemcpy2DAsync(dest->ptr, bytesdesc.deststride * bytesdesc.elembytes, src->ptr,
                                   bytesdesc.srcstride * bytesdesc.elembytes, bytesdesc.elembytes,
                                   bytesdesc.nelems, cudaMemcpyDefault, stream_for_op);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cudaMemcpy2DAsync() failed\n");
    }

    if (!verb.is_stream && !verb.is_nbi) {
        CUDA_RUNTIME_CHECK_GOTO(cudaStreamSynchronize(custrm), status, out);
    }

out:
    return status;
}

static inline int nvshmemi_prepare_and_post_mapped_rma(rma_verb_t verb, size_t nelems,
                                                       size_t elembytes, uint64_t *sig_addr,
                                                       uint64_t signal, void *local, void *remote,
                                                       ptrdiff_t lstride, ptrdiff_t rstride,
                                                       int sig_op, int pe) {
    const int distance = (pe - nvshmemi_state->mype + nvshmemi_state->npes) % nvshmemi_state->npes;
    const int stream_idx = distance % nvshmemi_options.MAX_PEER_STREAMS;
    cudaStream_t custrm = nvshmemi_state->custreams[stream_idx];
    cudaEvent_t cuev = nvshmemi_state->cuevents[stream_idx];
    rma_memdesc_t dest, src;
    rma_bytesdesc_t bytesdesc = {(size_t)nelems, (int)elembytes, lstride, rstride};

    if (verb.is_nbi) {
        nvshmemi_state->active_internal_streams[stream_idx] = 1;
        nvshmemi_state->used_internal_streams = 1;
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
        /*
         * Keep mapped-P2P activity distinct from CFT activity: P2P fence still synchronizes these
         * streams, while a CFT-only stream uses the deferred per-PE event dependency.
         */
        nvshmemi_state->p2p_active_internal_streams[stream_idx] = true;
#endif
    }

    void *const mapped_ptr =
        nvshmemi_translate_mapped_ptr(remote, pe, nvshmemi_state->heap_obj->get_local_pe_bases());

    if ((verb.desc == NVSHMEMI_OP_P) || (verb.desc == NVSHMEMI_OP_PUT) ||
        (verb.desc == NVSHMEMI_OP_PUT_SIGNAL)) {
        dest.ptr = mapped_ptr;
        dest.offset = (char *)remote - (char *)(nvshmemi_device_state.heap_base);
        src.ptr = local;
    } else {
        src.ptr = mapped_ptr;
        src.offset = (char *)remote - (char *)(nvshmemi_device_state.heap_base);
        dest.ptr = local;
        bytesdesc.srcstride = rstride;
        bytesdesc.deststride = lstride;
    }

    /* when memory is in the heap, we can make some assumptions about memory locations that reduce
     * op latency by ~100ns. */
    if (local >= nvshmemi_device_state.heap_base &&
        (local <
         (void *)((char *)nvshmemi_device_state.heap_base + nvshmemi_device_state.heap_size))) {
        return nvshmemi_p2p_rma_optimized(custrm, cuev, verb, &dest, &src, bytesdesc, sig_addr,
                                          signal, sig_op, pe);
    }
    return nvshmemi_p2p_rma_registered(custrm, cuev, verb, &dest, &src, bytesdesc, sig_addr, signal,
                                       sig_op, pe);
}

static void nvshmemi_prepare_and_post_rma(const char *apiname, nvshmemi_op_t desc, int is_nbi,
                                          int is_stream, void *lptr, void *rptr, ptrdiff_t lstride,
                                          ptrdiff_t rstride, size_t nelems, size_t elembytes,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          cudaStream_t cstrm) {
    rma_verb_t verb = {desc, is_nbi, is_stream, cstrm};
    rma_bytesdesc_t bytesdesc = {(size_t)nelems, (int)elembytes, 1, 1};
    int status = 0;
    const bool has_mapped_peer = nvshmemi_state->heap_obj->get_local_pe_bases()[pe] != nullptr;

#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
#if defined(NVSHMEM_PRIORITIZE_LOGICAL_ENDPOINT)
    constexpr bool prioritize_cft = true;
#else
    constexpr bool prioritize_cft = false;
#endif
    /* Logical endpoints do not support self access. Always use the local mapped pointer route. */
    if (pe == nvshmemi_state->mype) {
        status = nvshmemi_prepare_and_post_mapped_rma(verb, nelems, elembytes, sig_addr, signal,
                                                      lptr, rptr, lstride, rstride, sig_op, pe);
        goto out;
    }
    if (!has_mapped_peer || prioritize_cft) {
        const int distance =
            (pe - nvshmemi_state->mype + nvshmemi_state->npes) % nvshmemi_state->npes;
        const int stream_idx = distance % nvshmemi_options.MAX_PEER_STREAMS;
        cudaStream_t custrm = nvshmemi_state->custreams[stream_idx];
        cudaEvent_t cuev = nvshmemi_state->cuevents[stream_idx];
        rma_bytesdesc_t cft_bytesdesc = {nelems, static_cast<int>(elembytes), lstride, rstride};

        status = nvshmemi_cft_rma(custrm, cuev, verb, lptr, rptr, cft_bytesdesc, sig_addr, signal,
                                  sig_op, pe);
        if (status != NVSHMEMX_ERROR_NOT_SUPPORTED) {
            goto out;
        }
        status = NVSHMEMX_SUCCESS;
    }
#endif

    /* Use the mapped pointer immediately unless CFT was prioritized; otherwise it is the
     * fallback. */
    if (has_mapped_peer) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
        status = nvshmemi_order_after_cft_fence(pe, !verb.is_stream || verb.is_nbi,
                                                verb.is_stream && !verb.is_nbi, verb.cstrm);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "failed to order mapped RMA after CFT fence\n");
#endif
        status = nvshmemi_prepare_and_post_mapped_rma(verb, nelems, elembytes, sig_addr, signal,
                                                      lptr, rptr, lstride, rstride, sig_op, pe);
        goto out;
    }

    /* Remote Peer strided ops, G operations, and put_signal are not supported for remote
     * transports. */
    if ((lstride > 1) || (rstride > 1) || desc == NVSHMEMI_OP_G ||
        (verb.desc == NVSHMEMI_OP_PUT_SIGNAL && !is_stream)) {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_ERROR_PRINT("NOT IMPLEMENTED %s \n", apiname);
        goto out;
    }

    /* off stream */
    if (!verb.is_stream) {
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
        status = nvshmemi_order_after_cft_fence(pe, false, false, NOT_A_CUDA_STREAM);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "failed to order transport RMA after CFT fence\n");
#endif
        const auto remote_target = nvshmemi_translate_unmapped_ptr(
            rptr, pe, nvshmemi_state->heap_obj->get_remote_pe_bases());
        const auto transport_pe = remote_target.transport_pe;
        const int transport_id = nvshmemi_state->selected_transport_for_rma[transport_pe.value()];
        /* IBGDA will not set the RMA transport because it doesn't work on host APIs. */
        if (transport_id < 0) {
            NVSHMEMI_ERROR_EXIT("[%d] rma not supported on transport to pe: %d \n",
                                nvshmemi_state->mype, transport_pe.value());
        }
        struct nvshmem_transport *tcurr = nvshmemi_state->transports[transport_id];

        if (verb.desc == NVSHMEMI_OP_P) {
            rma_memdesc_t localdesc, remotedesc;
            localdesc.ptr = lptr;
            localdesc.handle = NULL;
            remotedesc.ptr = remote_target.remote_ptr;
            nvshmemi_get_remote_mem_handle(&remotedesc, NULL, rptr, transport_pe.value(),
                                           transport_id);
            status = tcurr->host_ops.rma(tcurr, transport_pe.value(), verb, &remotedesc, &localdesc,
                                         bytesdesc, NVSHMEMX_QP_HOST);
            if (unlikely(status)) {
                NVSHMEMI_ERROR_PRINT("aborting due to error in process_channel_dma\n");
                exit(-1);
            }
        } else {
            nvshmem_transport_op_attrs_t attrs;
            if (unlikely((nvshmemi_region_host_active_hints & NVSHMEMI_REGION_HINT_BATCH_RMA) !=
                         0) &&
                verb.is_nbi && (verb.desc == NVSHMEMI_OP_PUT || verb.desc == NVSHMEMI_OP_GET) &&
                lstride == 1 && rstride == 1 && nvshmemi_region_host_prepare_rma_attrs(&attrs)) {
                nvshmemi_process_multisend_rma_with_hints(tcurr, transport_id, remote_target, verb,
                                                          rptr, lptr, nelems * elembytes,
                                                          NVSHMEMX_QP_HOST, &attrs);
            } else {
                nvshmemi_process_multisend_rma(tcurr, transport_id, remote_target, verb, rptr, lptr,
                                               nelems * elembytes, NVSHMEMX_QP_HOST);
            }
        }
        goto out;
    }

    /* on stream */
#if defined(NVSHMEM_CFT_HANDLES_SUPPORT) && CUDART_VERSION >= 13050
    status = nvshmemi_order_after_cft_fence(pe, false, true, verb.cstrm);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "failed to order proxy RMA after CFT fence\n");
#endif
    if (verb.desc == NVSHMEMI_OP_PUT_SIGNAL) {
        verb.desc = NVSHMEMI_OP_PUT;
        void *args[] = {&rptr, &lptr, &bytesdesc, &sig_addr, &signal, &sig_op, &pe, &verb.desc};
        status = nvshmemi_proxy_rma_launcher(args, cstrm, is_nbi, true);
    } else {
        void *args[] = {&rptr, &lptr, &bytesdesc, &pe, &verb.desc};
        status = nvshmemi_proxy_rma_launcher(args, cstrm, is_nbi, false);
    }

out:
    if (status) {
        NVSHMEMI_ERROR_PRINT("aborting due to error in %s \n", apiname);
        exit(-1);
    }
}

/***** Put APIs ******/

#define NVSHMEM_TYPE_PUT(Name, TYPE)                                                              \
    void nvshmem_##Name##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) {            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        /*INFO(NVSHMEM_P2P, "[%d] bulk put : (remote)dest %p, (local)source %p, %d elements,      \
         * remote PE %d", nvshmemi_state->mype, dest, source, nelems, pe);*/                      \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,            \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT)
#undef NVSHMEM_TYPE_PUT

#define NVSHMEMX_TYPE_PUT_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_put_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe,   \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,     \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,  \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_ON_STREAM

#define NVSHMEM_PUTSIZE(Name, Type)                                                              \
    void nvshmem_put##Name(void *dest, const void *source, size_t nelems, int pe) {              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                  \
        NVSHMEMI_CHECK_INIT_STATUS();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,           \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe, \
                                      NOT_A_CUDA_STREAM);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE)
#undef NVSHMEM_PUTSIZE

#define NVSHMEMX_PUTSIZE_ON_STREAM(Name, Type)                                                    \
    void nvshmemx_put##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe,    \
                                        cudaStream_t cstrm) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmemx_put" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,     \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_ON_STREAM

/*XXX:Should other comm/put APIs call into nvshmem_putmem (suggested by SP)*/
void nvshmem_putmem(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_putmem_on_stream", NVSHMEMI_OP_PUT, NO_NBI, ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

#define NVSHMEM_TYPE_P(Name, TYPE)                                                            \
    void nvshmem_##Name##_p(TYPE *dest, const TYPE value, int pe) {                           \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                               \
        NVSHMEMI_CHECK_INIT_STATUS();                                                         \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p", NVSHMEMI_OP_P, NO_NBI, NO_ASYNC, \
                                      (void *)&value, (void *)dest, SRC_STRIDE_CONTIG,        \
                                      DEST_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_P)
#undef NVSHMEM_TYPE_P

#define NVSHMEMX_TYPE_P_ON_STREAM(Name, TYPE)                                                      \
    void nvshmemx_##Name##_p_on_stream(TYPE *dest, const TYPE value, int pe, cudaStream_t cstrm) { \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_p_on_stream", NVSHMEMI_OP_P, NO_NBI,      \
                                      ASYNC, (void *)&value, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,        \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_P_ON_STREAM)
#undef NVSHMEMX_TYPE_P_ON_STREAM

#define NVSHMEM_TYPE_IPUT(Name, TYPE)                                                              \
    void nvshmem_##Name##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,       \
                               size_t nelems, int pe) {                                            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, sst, dst, nelems,              \
                                      sizeof(TYPE), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IPUT)
#undef NVSHMEM_TYPE_IPUT

#define NVSHMEMX_TYPE_IPUT_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_iput_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,           \
                                          ptrdiff_t sst, size_t nelems, int pe,                    \
                                          cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iput_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, sst, dst, nelems,       \
                                      sizeof(TYPE), NULL, 0, -1, pe, cstrm);                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IPUT_ON_STREAM)
#undef NVSHMEMX_TYPE_IPUT_ON_STREAM

#define NVSHMEM_IPUTSIZE(Name, Type)                                                              \
    void nvshmem_iput##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,         \
                            size_t nelems, int pe) {                                              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "", NVSHMEMI_OP_PUT, NO_NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, sst, dst, nelems,             \
                                      sizeof(Type), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IPUTSIZE)
#undef NVSHMEM_IPUTSIZE

#define NVSHMEMX_IPUTSIZE_ON_STREAM(Name, Type)                                                   \
    void nvshmemx_iput##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,           \
                                         ptrdiff_t sst, size_t nelems, int pe,                    \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iput" #Name "_on_stream", NVSHMEMI_OP_PUT, NO_NBI, \
                                      ASYNC, (void *)source, (void *)dest, sst, dst, nelems,      \
                                      sizeof(Type), NULL, 0, -1, pe, cstrm);                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IPUTSIZE_ON_STREAM)
#undef NVSHMEMX_IPUTSIZE_ON_STREAM

#define NVSHMEM_TYPE_PUT_NBI(type, TYPE)                                                           \
    void nvshmem_##type##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,             \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_PUT_NBI)
#undef NVSHMEM_TYPE_PUT_NBI

/* PUT_SIGNAL API */
#define NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM(Name, TYPE)                                             \
    void nvshmemx_##Name##_put_signal_on_stream(TYPE *dest, const TYPE *source, size_t nelems,     \
                                                uint64_t *sig_addr, uint64_t signal, int sig_op,   \
                                                int pe, cudaStream_t cstrm) {                      \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_put_signal_on_stream",                    \
                                      NVSHMEMI_OP_PUT_SIGNAL, NO_NBI, ASYNC, (void *)source,       \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(TYPE), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_SIGNAL_ON_STREAM

#define NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM(Name, Type)                                              \
    void nvshmemx_put##Name##_signal_on_stream(void *dest, const void *source, size_t nelems,      \
                                               uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                               int pe, cudaStream_t cstrm) {                       \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmemx_put" #Name "_signal_on_stream",                    \
                                      NVSHMEMI_OP_PUT_SIGNAL, NO_NBI, ASYNC, (void *)source,       \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(Type), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_SIGNAL_ON_STREAM

void nvshmemx_putmem_signal_on_stream(void *dest, const void *source, size_t bytes,
                                      uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                      cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    TRACE(NVSHMEM_P2P,
          "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
          nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_putmem_signal_on_stream", NVSHMEMI_OP_PUT_SIGNAL,
                                  NO_NBI, ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, sig_addr, signal, sig_op, pe,
                                  cstrm);
}

#define NVSHMEMX_TYPE_PUT_NBI_ON_STREAM(type, TYPE)                                                \
    void nvshmemx_##type##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,        \
                                             int pe, cudaStream_t cstrm) {                         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_NBI_ON_STREAM

#define NVSHMEM_PUTSIZE_NBI(Name, Type)                                                           \
    void nvshmem_put##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC, \
                                      (void *)source, (void *)dest, SRC_STRIDE_CONTIG,            \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_PUTSIZE_NBI)
#undef NVSHMEM_PUTSIZE_NBI

#define NVSHMEMX_PUTSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_put##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_nbi_on_stream", NVSHMEMI_OP_PUT, NBI,  \
                                      ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,      \
                                      DEST_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_NBI_ON_STREAM

void nvshmem_putmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi", NVSHMEMI_OP_PUT, NBI, NO_ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_nbi_on_stream", NVSHMEMI_OP_PUT, NBI, ASYNC,
                                  (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

/* PUT_SIGNAL_NBI */
#define NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM(type, TYPE)                                         \
    void nvshmemx_##type##_put_signal_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    uint64_t *sig_addr, uint64_t signal,           \
                                                    int sig_op, int pe, cudaStream_t cstrm) {      \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_put_signal_nbi_on_stream",                \
                                      NVSHMEMI_OP_PUT_SIGNAL, NBI, ASYNC, (void *)source,          \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(TYPE), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_PUT_SIGNAL_NBI_ON_STREAM

#define NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM(Name, Type)                                          \
    void nvshmemx_put##Name##_signal_nbi_on_stream(void *dest, const void *source, size_t nelems,  \
                                                   uint64_t *sig_addr, uint64_t signal,            \
                                                   int sig_op, int pe, cudaStream_t cstrm) {       \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_put" #Name "_signal_nbi_on_stream",                 \
                                      NVSHMEMI_OP_PUT_SIGNAL, NBI, ASYNC, (void *)source,          \
                                      (void *)dest, SRC_STRIDE_CONTIG, DEST_STRIDE_CONTIG, nelems, \
                                      sizeof(Type), sig_addr, signal, sig_op, pe, cstrm);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM)
#undef NVSHMEMX_PUTSIZE_SIGNAL_NBI_ON_STREAM

void nvshmemx_putmem_signal_nbi_on_stream(void *dest, const void *source, size_t bytes,
                                          uint64_t *sig_addr, uint64_t signal, int sig_op, int pe,
                                          cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped put : (remote)dest %p, (local)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_putmem_signal_nbi_on_stream", NVSHMEMI_OP_PUT_SIGNAL,
                                  NBI, ASYNC, (void *)source, (void *)dest, SRC_STRIDE_CONTIG,
                                  DEST_STRIDE_CONTIG, bytes, 1, sig_addr, signal, sig_op, pe,
                                  cstrm);
}

/***** Get APIs ******/

#define NVSHMEM_TYPE_GET(Name, TYPE)                                                              \
    void nvshmem_##Name##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) {            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_get", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,           \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET)
#undef NVSHMEM_TYPE_GET

#define NVSHMEMX_TYPE_GET_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe,   \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_get_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,    \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,   \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_ON_STREAM

#define NVSHMEM_GETSIZE(Name, Type)                                                              \
    void nvshmem_get##Name(void *dest, const void *source, size_t nelems, int pe) {              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                  \
        NVSHMEMI_CHECK_INIT_STATUS();                                                            \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,          \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,  \
                                      NOT_A_CUDA_STREAM);                                        \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE)
#undef NVSHMEM_GETSIZE

#define NVSHMEMX_GETSIZE_ON_STREAM(Name, Type)                                                    \
    void nvshmemx_get##Name##_on_stream(void *dest, const void *source, size_t nelems, int pe,    \
                                        cudaStream_t cstrm) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmemx_get" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,    \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      cstrm);                                                     \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_ON_STREAM)
#undef NVSHMEMX_GETSIZE_ON_STREAM

void nvshmem_getmem(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, (void *)dest,
                                  (void *)source, DEST_STRIDE_CONTIG, SRC_STRIDE_CONTIG, bytes, 1,
                                  NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmemx_getmem_on_stream", NVSHMEMI_OP_GET, NO_NBI, ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}

#define NVSHMEM_TYPE_G(Name, TYPE)                                                            \
    TYPE nvshmem_##Name##_g(const TYPE *source, int pe) {                                     \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                               \
        NVSHMEMI_CHECK_INIT_STATUS();                                                         \
        TYPE value;                                                                           \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",                \
             nvshmemi_state->mype, source, pe);                                               \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g", NVSHMEMI_OP_G, NO_NBI, NO_ASYNC, \
                                      (void *)&value, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      NOT_A_CUDA_STREAM);                                     \
        return value;                                                                         \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_G)
#undef NVSHMEM_TYPE_G

#define NVSHMEMX_TYPE_G_ON_STREAM(Name, TYPE)                                                      \
    TYPE nvshmemx_##Name##_g_on_stream(const TYPE *source, int pe, cudaStream_t cstrm) {           \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        TYPE value;                                                                                \
        INFO(NVSHMEM_P2P, "[%d] single get : (remote)source %p, remote PE %d",                     \
             nvshmemi_state->mype, source, pe);                                                    \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_g_on_stream", NVSHMEMI_OP_G, NO_NBI,      \
                                      ASYNC, (void *)&value, (void *)source, DEST_STRIDE_CONTIG,   \
                                      SRC_STRIDE_CONTIG, 1, sizeof(TYPE), NULL, 0, -1, pe, cstrm); \
        return value;                                                                              \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_G_ON_STREAM)
#undef NVSHMEMX_TYPE_G_ON_STREAM

#define NVSHMEM_TYPE_IGET(Name, TYPE)                                                              \
    void nvshmem_##Name##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst, ptrdiff_t sst,       \
                               size_t nelems, int pe) {                                            \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,              \
                                      sizeof(TYPE), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);           \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IGET)
#undef NVSHMEM_TYPE_IGET

#define NVSHMEMX_TYPE_IGET_ON_STREAM(Name, TYPE)                                                   \
    void nvshmemx_##Name##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst,           \
                                          ptrdiff_t sst, size_t nelems, int pe,                    \
                                          cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                    \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #Name "_iget_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, dst, sst, nelems,       \
                                      sizeof(TYPE), NULL, 0, -1, pe, cstrm);                       \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_IGET_ON_STREAM)
#undef NVSHMEMX_TYPE_IGET_ON_STREAM

#define NVSHMEM_IGETSIZE(Name, Type)                                                              \
    void nvshmem_iget##Name(void *dest, const void *source, ptrdiff_t dst, ptrdiff_t sst,         \
                            size_t nelems, int pe) {                                              \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "", NVSHMEMI_OP_GET, NO_NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, dst, sst, nelems,             \
                                      sizeof(Type), NULL, 0, -1, pe, NOT_A_CUDA_STREAM);          \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IGETSIZE)
#undef NVSHMEM_IGETSIZE

#define NVSHMEMX_IGETSIZE_ON_STREAM(Name, Type)                                                   \
    void nvshmemx_iget##Name##_on_stream(void *dest, const void *source, ptrdiff_t dst,           \
                                         ptrdiff_t sst, size_t nelems, int pe,                    \
                                         cudaStream_t cstrm) {                                    \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_BLOCKING);                                                   \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_iget" #Name "_on_stream", NVSHMEMI_OP_GET, NO_NBI, \
                                      ASYNC, (void *)dest, (void *)source, dst, sst, nelems,      \
                                      sizeof(Type), NULL, 0, -1, pe, cstrm);                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_IGETSIZE_ON_STREAM)
#undef NVSHMEMX_IGETSIZE_ON_STREAM

#define NVSHMEM_TYPE_GET_NBI(type, TYPE)                                                           \
    void nvshmem_##type##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_get_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,            \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      NOT_A_CUDA_STREAM);                                          \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_GET_NBI)
#undef NVSHMEM_TYPE_GET_NBI

#define NVSHMEMX_TYPE_GET_NBI_ON_STREAM(type, TYPE)                                                \
    void nvshmemx_##type##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems,        \
                                             int pe, cudaStream_t cstrm) {                         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_" #type "_get_nbi_on_stream", NVSHMEMI_OP_GET, NBI, \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(TYPE), NULL, 0, -1, pe,    \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_TYPE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_TYPE_GET_NBI_ON_STREAM

#define NVSHMEM_GETSIZE_NBI(Name, Type)                                                           \
    void nvshmem_get##Name##_nbi(void *dest, const void *source, size_t nelems, int pe) {         \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                \
        NVSHMEMI_CHECK_INIT_STATUS();                                                             \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC, \
                                      (void *)dest, (void *)source, DEST_STRIDE_CONTIG,           \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,   \
                                      NOT_A_CUDA_STREAM);                                         \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_GETSIZE_NBI)
#undef NVSHMEM_GETSIZE_NBI

#define NVSHMEMX_GETSIZE_NBI_ON_STREAM(Name, Type)                                                 \
    void nvshmemx_get##Name##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm) {                                  \
        NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);                                                 \
        NVSHMEMI_CHECK_INIT_STATUS();                                                              \
        nvshmemi_prepare_and_post_rma("nvshmem_get" #Name "_nbi_on_stream", NVSHMEMI_OP_GET, NBI,  \
                                      ASYNC, (void *)dest, (void *)source, DEST_STRIDE_CONTIG,     \
                                      SRC_STRIDE_CONTIG, nelems, sizeof(Type), NULL, 0, -1, pe,    \
                                      cstrm);                                                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEMX_GETSIZE_NBI_ON_STREAM)
#undef NVSHMEMX_GETSIZE_NBI_ON_STREAM

void nvshmem_getmem_nbi(void *dest, const void *source, size_t bytes, int pe) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi", NVSHMEMI_OP_GET, NBI, NO_ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, NOT_A_CUDA_STREAM);
}

void nvshmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm) {
    NVTX_FUNC_RANGE_IN_GROUP(RMA_NONBLOCKING);
    NVSHMEMI_CHECK_INIT_STATUS();
    INFO(NVSHMEM_P2P,
         "[%d] untyped get : (local)dest %p, (remote)source %p, %zu bytes, remote PE %d",
         nvshmemi_state->mype, dest, source, bytes, pe);
    nvshmemi_prepare_and_post_rma("nvshmem_getmem_nbi_on_stream", NVSHMEMI_OP_GET, NBI, ASYNC,
                                  (void *)dest, (void *)source, DEST_STRIDE_CONTIG,
                                  SRC_STRIDE_CONTIG, bytes, 1, NULL, 0, -1, pe, cstrm);
}
