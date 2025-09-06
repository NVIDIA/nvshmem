# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information
#
# This code was automatically generated from NVSHMEM with version 3.3.0. 
# Modify it directly at your own risk.


from libc.stdint cimport intptr_t

from .cynvshmem cimport *


###############################################################################
# Types
###############################################################################

ctypedef nvshmemx_uniqueid_args_v1 uniqueid_args
ctypedef nvshmem_team_config_v2 team_config
ctypedef nvshmemx_init_args_v1 init_args
ctypedef nvshmemx_init_attr_v1 init_attr

ctypedef cudaStream_t Stream
ctypedef CUmodule Module



###############################################################################
# Enum
###############################################################################

ctypedef nvshmemx_signal_op_t _Signal_op
ctypedef nvshmemx_cmp_type_t _Cmp_type
ctypedef nvshmemx_init_status_t _Init_status
ctypedef nvshmem_team_id_t _Team_id
ctypedef nvshmemx_status _Status
ctypedef flags _Flags


###############################################################################
# Functions
###############################################################################

cpdef int init_status() except? 0
cpdef int my_pe() except? -1
cpdef int n_pes() except? -1
cpdef void info_get_version(intptr_t major, intptr_t minor) except*
cpdef void vendor_get_version_info(intptr_t major, intptr_t minor, intptr_t patch) except*
cpdef intptr_t malloc(size_t size) except? 0
cpdef intptr_t calloc(size_t count, size_t size) except? 0
cpdef intptr_t align(size_t count, size_t size) except? 0
cpdef void free(intptr_t ptr) except*
cpdef intptr_t ptr(intptr_t dest, int pe) except? 0
cpdef intptr_t mc_ptr(int32_t team, intptr_t ptr) except? 0
cpdef int team_my_pe(int32_t team) except? -1
cpdef int team_n_pes(int32_t team) except? -1
cpdef barrier(int32_t team)
cpdef void barrier_all() except*
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
cpdef int team_sync_on_stream(int32_t team, intptr_t stream) except? 0
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
cpdef int cumodule_init(intptr_t module) except? 0
cpdef int cumodule_finalize(intptr_t module) except? 0
cpdef void putmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void putmem_signal_on_stream(intptr_t dest, intptr_t source, size_t bytes, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*
cpdef void getmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*
cpdef void quiet_on_stream(intptr_t cstrm) except*
cpdef void signal_wait_until_on_stream(intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t cstream) except*