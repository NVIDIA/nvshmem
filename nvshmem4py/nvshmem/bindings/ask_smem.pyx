# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ``nvshmemx_ask_smem`` is a header-inline ``__host__ __device__`` API, so it
# cannot be loaded through CyBind's host-library symbol table. Keep this in a
# standalone extension: CyBind's generated C declarations intentionally do
# not include NVSHMEM headers.
cdef extern from "device/nvshmemx_defines.h":
    ctypedef enum nvshmemx_smem_amount_t:
        pass

    int nvshmemx_ask_smem(nvshmemx_smem_amount_t amount)


cpdef int ask_smem(int amount):
    return nvshmemx_ask_smem(<nvshmemx_smem_amount_t>amount)
