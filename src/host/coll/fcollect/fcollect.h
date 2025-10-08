/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See License.txt for license information
 */

#ifndef NVSHMEMI_FCOLLECT_CPU_H
#define NVSHMEMI_FCOLLECT_CPU_H
#include <driver_types.h>                    // for cudaStream_t, CUstr...
#include <stddef.h>                          // for size_t
#include "device_host/nvshmem_types.h"       // for nvshmemi_team_t
#include "cpu_coll.h"                        // for nvshmemi_get_nccl_dt
#include "device_host/nvshmem_common.cuh"    // for nvshmemi_team_pool
#include "internal/host/nvshmem_internal.h"  // for nvshmemi_use_nccl
#include "internal/host/util.h"              // for NCCL_CHECK
#include "non_abi/nvshmem_build_options.h"   // for NVSHMEM_USE_NCCL
#include "internal/host/nvshmemi_team.h"
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"  // for ncclComm, ncclComm_t
#endif

template <typename TYPE>
void nvshmemi_call_fcollect_on_stream_kernel(nvshmem_team_t team, TYPE *dest, const TYPE *source,
                                             size_t nelems, cudaStream_t stream);

template <typename TYPE>
int nvshmemi_fcollect_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems,
                                cudaStream_t stream) {
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
#ifdef NVSHMEM_USE_NCCL
    if (teami->nvls_rsc_base_ptr == NULL && nvshmemi_use_nccl &&
        nvshmemi_get_nccl_dt<TYPE>() != ncclNumTypes) {
        NCCL_CHECK(nccl_ftable.AllGather(source, dest, nelems, nvshmemi_get_nccl_dt<TYPE>(),
                                         (ncclComm_t)teami->nccl_comm, stream));
    } else
#endif
    {
        if (teami->are_gpus_p2p_connected && !nvshmemi_disable_ce_collectives &&
            teami->nvls_rsc_base_ptr != NULL && nvshmemi_can_use_cuda_64_bit_stream_memops) {
            for (int i = 1; i <= teami->size; i++) {
                int dst_pe = (teami->my_pe + i) % teami->size;
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    nvshmemi_ptr(dest + teami->my_pe * nelems,
                                 nvshmemi_team_translate_pe_to_team_world_wrap(teami, dst_pe)),
                    source, nelems * sizeof(TYPE), cudaMemcpyDefault, stream));
            }
            nvshmemi_coll_p2p_sync(teami, stream);
        } else {
            nvshmemi_call_fcollect_on_stream_kernel<TYPE>(team, dest, source, nelems, stream);
        }
    }
    return 0;
}
#endif /* NVSHMEMI_FCOLLECT_CPU_H */
