# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated from NVSHMEM with version 3.3.0. 
# Modify it directly at your own risk.


from libc.stdint cimport intptr_t

from .cynvshmem cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvshmemx_region_attrs_t region_attrs
ctypedef nvshmemx_uniqueid_args_v1 uniqueid_args
ctypedef nvshmem_team_config_v2 team_config
ctypedef nvshmemx_init_args_v2 init_args
ctypedef nvshmemx_init_attr_v2 init_attr

ctypedef cudaStream_t Stream
ctypedef CUmodule Module
ctypedef CUlibrary Library



###############################################################################
# Enum
###############################################################################

ctypedef nvshmemx_signal_op_t _Signal_op
ctypedef nvshmemx_tma_policy_t _Tma_policy
ctypedef nvshmemx_smem_amount_t _Smem_amount
ctypedef nvshmemx_region_hint_t _Region_hint
ctypedef nvshmemx_cmp_type_t _Cmp_type
ctypedef nvshmemx_thread_support_t _Thread_support
ctypedef nvshmemx_proxy_status_t _Proxy_status
ctypedef nvshmemx_init_status_t _Init_status
ctypedef nvshmemx_qp_handle_index_t _Qp_handle_index
ctypedef nvshmem_pe_index_t _Pe_index
ctypedef nvshmem_team_id_t _Team_id
ctypedef nvshmemx_status _Status
ctypedef flags _Flags


###############################################################################
# Functions
###############################################################################

cpdef barrier(int32_t team)
cpdef void barrier_all() except*
cpdef int init_status() except? 0
cpdef void query_thread(intptr_t provided) except*
cpdef int my_pe() except? -1
cpdef int n_pes() except? -1
cpdef void info_get_version(intptr_t major, intptr_t minor) except*
cpdef void info_get_name(intptr_t name) except*
cpdef void vendor_get_version_info(intptr_t major, intptr_t minor, intptr_t patch) except*
cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t calloc(size_t count, size_t size) except? 0
cpdef intptr_t align(size_t alignment, size_t size) except? 0
cpdef void free(intptr_t ptr) except*
cpdef intptr_t ptr(intptr_t dest, int pe) except? 0
cpdef intptr_t mc_ptr(int32_t team, intptr_t ptr) except? 0
cpdef void uint_atomic_inc(intptr_t dest, int pe) except*
cpdef void ulong_atomic_inc(intptr_t dest, int pe) except*
cpdef void ulonglong_atomic_inc(intptr_t dest, int pe) except*
cpdef void int32_atomic_inc(intptr_t dest, int pe) except*
cpdef void uint32_atomic_inc(intptr_t dest, int pe) except*
cpdef void int64_atomic_inc(intptr_t dest, int pe) except*
cpdef void uint64_atomic_inc(intptr_t dest, int pe) except*
cpdef void int_atomic_inc(intptr_t dest, int pe) except*
cpdef void long_atomic_inc(intptr_t dest, int pe) except*
cpdef void longlong_atomic_inc(intptr_t dest, int pe) except*
cpdef void size_atomic_inc(intptr_t dest, int pe) except*
cpdef void ptrdiff_atomic_inc(intptr_t dest, int pe) except*
cpdef unsigned int uint_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef int32_t int32_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef int64_t int64_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef int int_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef long long_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef long long longlong_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef size_t size_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef ptrdiff_t ptrdiff_atomic_fetch_inc(intptr_t dest, int pe) except? -1
cpdef unsigned int uint_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef int32_t int32_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef int64_t int64_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef int int_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef long long_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef long long longlong_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef size_t size_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef ptrdiff_t ptrdiff_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef float float_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef double double_atomic_fetch(intptr_t dest, int pe) except? -1
cpdef void uint_atomic_add(intptr_t dest, unsigned int value, int pe) except*
cpdef void ulong_atomic_add(intptr_t dest, unsigned long value, int pe) except*
cpdef void ulonglong_atomic_add(intptr_t dest, unsigned long long value, int pe) except*
cpdef void int32_atomic_add(intptr_t dest, int32_t value, int pe) except*
cpdef void uint32_atomic_add(intptr_t dest, uint32_t value, int pe) except*
cpdef void int64_atomic_add(intptr_t dest, int64_t value, int pe) except*
cpdef void uint64_atomic_add(intptr_t dest, uint64_t value, int pe) except*
cpdef void int_atomic_add(intptr_t dest, int value, int pe) except*
cpdef void long_atomic_add(intptr_t dest, long value, int pe) except*
cpdef void longlong_atomic_add(intptr_t dest, long long value, int pe) except*
cpdef void size_atomic_add(intptr_t dest, size_t value, int pe) except*
cpdef void ptrdiff_atomic_add(intptr_t dest, ptrdiff_t value, int pe) except*
cpdef void float_atomic_add(intptr_t dest, float value, int pe) except*
cpdef void double_atomic_add(intptr_t dest, double value, int pe) except*
cpdef float float_atomic_fetch_add(intptr_t dest, float value, int pe) except? -1
cpdef double double_atomic_fetch_add(intptr_t dest, double value, int pe) except? -1
cpdef void uint_atomic_set(intptr_t dest, unsigned int value, int pe) except*
cpdef void ulong_atomic_set(intptr_t dest, unsigned long value, int pe) except*
cpdef void ulonglong_atomic_set(intptr_t dest, unsigned long long value, int pe) except*
cpdef void int32_atomic_set(intptr_t dest, int32_t value, int pe) except*
cpdef void uint32_atomic_set(intptr_t dest, uint32_t value, int pe) except*
cpdef void int64_atomic_set(intptr_t dest, int64_t value, int pe) except*
cpdef void uint64_atomic_set(intptr_t dest, uint64_t value, int pe) except*
cpdef void int_atomic_set(intptr_t dest, int value, int pe) except*
cpdef void long_atomic_set(intptr_t dest, long value, int pe) except*
cpdef void longlong_atomic_set(intptr_t dest, long long value, int pe) except*
cpdef void size_atomic_set(intptr_t dest, size_t value, int pe) except*
cpdef void ptrdiff_atomic_set(intptr_t dest, ptrdiff_t value, int pe) except*
cpdef void float_atomic_set(intptr_t dest, float value, int pe) except*
cpdef void double_atomic_set(intptr_t dest, double value, int pe) except*
cpdef unsigned int uint_atomic_fetch_add(intptr_t dest, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch_add(intptr_t dest, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch_add(intptr_t dest, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_fetch_add(intptr_t dest, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch_add(intptr_t dest, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_fetch_add(intptr_t dest, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch_add(intptr_t dest, uint64_t value, int pe) except? -1
cpdef int int_atomic_fetch_add(intptr_t dest, int value, int pe) except? -1
cpdef long long_atomic_fetch_add(intptr_t dest, long value, int pe) except? -1
cpdef long long longlong_atomic_fetch_add(intptr_t dest, long long value, int pe) except? -1
cpdef size_t size_atomic_fetch_add(intptr_t dest, size_t value, int pe) except? -1
cpdef ptrdiff_t ptrdiff_atomic_fetch_add(intptr_t dest, ptrdiff_t value, int pe) except? -1
cpdef unsigned int uint_atomic_swap(intptr_t dest, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_swap(intptr_t dest, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_swap(intptr_t dest, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_swap(intptr_t dest, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_swap(intptr_t dest, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_swap(intptr_t dest, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_swap(intptr_t dest, uint64_t value, int pe) except? -1
cpdef int int_atomic_swap(intptr_t dest, int value, int pe) except? -1
cpdef long long_atomic_swap(intptr_t dest, long value, int pe) except? -1
cpdef long long longlong_atomic_swap(intptr_t dest, long long value, int pe) except? -1
cpdef size_t size_atomic_swap(intptr_t dest, size_t value, int pe) except? -1
cpdef ptrdiff_t ptrdiff_atomic_swap(intptr_t dest, ptrdiff_t value, int pe) except? -1
cpdef float float_atomic_swap(intptr_t dest, float value, int pe) except? -1
cpdef double double_atomic_swap(intptr_t dest, double value, int pe) except? -1
cpdef unsigned int uint_atomic_compare_swap(intptr_t dest, unsigned int cond, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_compare_swap(intptr_t dest, unsigned long cond, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_compare_swap(intptr_t dest, unsigned long long cond, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_compare_swap(intptr_t dest, int32_t cond, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_compare_swap(intptr_t dest, uint32_t cond, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_compare_swap(intptr_t dest, int64_t cond, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_compare_swap(intptr_t dest, uint64_t cond, uint64_t value, int pe) except? -1
cpdef int int_atomic_compare_swap(intptr_t dest, int cond, int value, int pe) except? -1
cpdef long long_atomic_compare_swap(intptr_t dest, long cond, long value, int pe) except? -1
cpdef long long longlong_atomic_compare_swap(intptr_t dest, long long cond, long long value, int pe) except? -1
cpdef size_t size_atomic_compare_swap(intptr_t dest, size_t cond, size_t value, int pe) except? -1
cpdef ptrdiff_t ptrdiff_atomic_compare_swap(intptr_t dest, ptrdiff_t cond, ptrdiff_t value, int pe) except? -1
cpdef void uint_atomic_and(intptr_t dest, unsigned int value, int pe) except*
cpdef void ulong_atomic_and(intptr_t dest, unsigned long value, int pe) except*
cpdef void ulonglong_atomic_and(intptr_t dest, unsigned long long value, int pe) except*
cpdef void int32_atomic_and(intptr_t dest, int32_t value, int pe) except*
cpdef void uint32_atomic_and(intptr_t dest, uint32_t value, int pe) except*
cpdef void int64_atomic_and(intptr_t dest, int64_t value, int pe) except*
cpdef void uint64_atomic_and(intptr_t dest, uint64_t value, int pe) except*
cpdef void uint_atomic_or(intptr_t dest, unsigned int value, int pe) except*
cpdef void ulong_atomic_or(intptr_t dest, unsigned long value, int pe) except*
cpdef void ulonglong_atomic_or(intptr_t dest, unsigned long long value, int pe) except*
cpdef void int32_atomic_or(intptr_t dest, int32_t value, int pe) except*
cpdef void uint32_atomic_or(intptr_t dest, uint32_t value, int pe) except*
cpdef void int64_atomic_or(intptr_t dest, int64_t value, int pe) except*
cpdef void uint64_atomic_or(intptr_t dest, uint64_t value, int pe) except*
cpdef void uint_atomic_xor(intptr_t dest, unsigned int value, int pe) except*
cpdef void ulong_atomic_xor(intptr_t dest, unsigned long value, int pe) except*
cpdef void ulonglong_atomic_xor(intptr_t dest, unsigned long long value, int pe) except*
cpdef void int32_atomic_xor(intptr_t dest, int32_t value, int pe) except*
cpdef void uint32_atomic_xor(intptr_t dest, uint32_t value, int pe) except*
cpdef void int64_atomic_xor(intptr_t dest, int64_t value, int pe) except*
cpdef void uint64_atomic_xor(intptr_t dest, uint64_t value, int pe) except*
cpdef unsigned int uint_atomic_fetch_and(intptr_t dest, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch_and(intptr_t dest, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch_and(intptr_t dest, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_fetch_and(intptr_t dest, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch_and(intptr_t dest, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_fetch_and(intptr_t dest, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch_and(intptr_t dest, uint64_t value, int pe) except? -1
cpdef unsigned int uint_atomic_fetch_or(intptr_t dest, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch_or(intptr_t dest, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch_or(intptr_t dest, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_fetch_or(intptr_t dest, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch_or(intptr_t dest, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_fetch_or(intptr_t dest, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch_or(intptr_t dest, uint64_t value, int pe) except? -1
cpdef unsigned int uint_atomic_fetch_xor(intptr_t dest, unsigned int value, int pe) except? -1
cpdef unsigned long ulong_atomic_fetch_xor(intptr_t dest, unsigned long value, int pe) except? -1
cpdef unsigned long long ulonglong_atomic_fetch_xor(intptr_t dest, unsigned long long value, int pe) except? -1
cpdef int32_t int32_atomic_fetch_xor(intptr_t dest, int32_t value, int pe) except? -1
cpdef uint32_t uint32_atomic_fetch_xor(intptr_t dest, uint32_t value, int pe) except? -1
cpdef int64_t int64_atomic_fetch_xor(intptr_t dest, int64_t value, int pe) except? -1
cpdef uint64_t uint64_atomic_fetch_xor(intptr_t dest, uint64_t value, int pe) except? -1
cpdef uint64_t signal_fetch(intptr_t sig_addr) except? -1
cpdef int team_my_pe(int32_t team) except? -1
cpdef int team_n_pes(int32_t team) except? -1
cpdef void team_get_config(int32_t team, intptr_t config) except*
cpdef int team_translate_pe(int32_t src_team, int src_pe, int32_t dest_team) except? -1
cpdef team_split_strided(int32_t parent_team, int pe_start, int pe_stride, int pe_size, intptr_t config, long config_mask, intptr_t new_team)
cpdef team_get_uniqueid(intptr_t uniqueid)
cpdef team_init(intptr_t team, intptr_t config, long config_mask, int npes, int pe_idx_in_team)
cpdef team_split_2d(int32_t parent_team, int xrange, intptr_t xaxis_config, long xaxis_mask, intptr_t xaxis_team, intptr_t yaxis_config, long yaxis_mask, intptr_t yaxis_team)
cpdef void team_destroy(int32_t team) except*
cpdef bfloat16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef half_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef float_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef double_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef char_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef short_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef schar_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef long_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef longlong_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int8_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int32_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int64_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint8_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint32_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint64_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef size_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef barrier_on_stream(int32_t team, intptr_t stream)
cpdef void barrier_all_on_stream(intptr_t stream) except*
cpdef int team_sync_on_stream(int32_t team, intptr_t stream) except? 0
cpdef void sync_all_on_stream(intptr_t stream) except*
cpdef bfloat16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef half_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef float_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef double_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef char_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef short_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef schar_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef int_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef long_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef longlong_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef int8_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef int16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef int32_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef int64_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef uint8_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef uint16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef uint32_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef uint64_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef size_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream)
cpdef bfloat16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef half_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef float_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef double_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef char_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef short_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef schar_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef long_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef longlong_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int8_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int32_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int64_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint8_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint32_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef uint64_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef size_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream)
cpdef int8_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int8_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int8_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int8_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int8_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int8_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int32_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int64_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint8_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint32_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef uint64_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef size_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef char_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef schar_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef short_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef int_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef long_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef longlong_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef bfloat16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef half_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef float_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef double_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream)
cpdef hostlib_init_attr(unsigned int flags, intptr_t attr)
cpdef void hostlib_finalize() except*
cpdef set_attr_uniqueid_args(int myrank, int nranks, intptr_t uniqueid, intptr_t attr)
cpdef set_attr_mpi_comm_args(intptr_t mpi_comm, intptr_t nvshmem_attr)
cpdef get_uniqueid(intptr_t uniqueid)
cpdef int cumodule_init(intptr_t module) except? -1
cpdef int cumodule_finalize(intptr_t module) except? -1
cpdef intptr_t buffer_register_symmetric(intptr_t buf_ptr, size_t size, int flags) except? 0
cpdef intptr_t buffer_register_symmetric_at_preferred_address(intptr_t buf_ptr, size_t size, intptr_t preferred_addr, int flags) except? 0
cpdef int buffer_unregister_symmetric(intptr_t mmap_ptr, size_t size) except? 0
cpdef int culibrary_init(intptr_t library) except? -1
cpdef int culibrary_finalize(intptr_t library) except? -1
cpdef void putmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void putmem_signal_on_stream(intptr_t dest, intptr_t source, size_t bytes, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*
cpdef void putmem_signal_nbi_on_stream(intptr_t dest, intptr_t source, size_t bytes, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*
cpdef void putmem_nbi_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void getmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void getmem_nbi_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void quiet_on_stream(intptr_t cstrm) except*
cpdef void flush_on_stream(intptr_t cstrm) except*
cpdef void signal_op_on_stream(intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*
cpdef void signal_wait_until_on_stream(intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t cstream) except*
