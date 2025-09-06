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



from ._internal cimport nvshmem as _nvshmem


###############################################################################
# Wrapper functions
###############################################################################

cdef int nvshmemx_init_status() except* nogil:
    return _nvshmem._nvshmemx_init_status()


cdef int nvshmem_my_pe() except* nogil:
    return _nvshmem._nvshmem_my_pe()


cdef int nvshmem_n_pes() except* nogil:
    return _nvshmem._nvshmem_n_pes()


cdef void nvshmem_info_get_version(int* major, int* minor) except* nogil:
    _nvshmem._nvshmem_info_get_version(major, minor)


cdef void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) except* nogil:
    _nvshmem._nvshmemx_vendor_get_version_info(major, minor, patch)


cdef void* nvshmem_malloc(size_t size) except* nogil:
    return _nvshmem._nvshmem_malloc(size)


cdef void* nvshmem_calloc(size_t count, size_t size) except* nogil:
    return _nvshmem._nvshmem_calloc(count, size)


cdef void* nvshmem_align(size_t count, size_t size) except* nogil:
    return _nvshmem._nvshmem_align(count, size)


cdef void nvshmem_free(void* ptr) except* nogil:
    _nvshmem._nvshmem_free(ptr)


cdef void* nvshmem_ptr(const void* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ptr(dest, pe)


cdef void* nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr) except* nogil:
    return _nvshmem._nvshmemx_mc_ptr(team, ptr)


cdef int nvshmem_team_my_pe(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_team_my_pe(team)


cdef int nvshmem_team_n_pes(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_team_n_pes(team)


cdef int nvshmem_barrier(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_barrier(team)


cdef void nvshmem_barrier_all() except* nogil:
    _nvshmem._nvshmem_barrier_all()


cdef int nvshmemx_bfloat16_alltoall_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_half_alltoall_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_float_alltoall_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_double_alltoall_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_char_alltoall_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_short_alltoall_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_schar_alltoall_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int_alltoall_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_long_alltoall_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_longlong_alltoall_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int8_alltoall_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int16_alltoall_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int32_alltoall_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int64_alltoall_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint8_alltoall_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint16_alltoall_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint32_alltoall_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint64_alltoall_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_size_alltoall_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_alltoall_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_barrier_on_stream(team, stream)


cdef int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_team_sync_on_stream(team, stream)


cdef int nvshmemx_bfloat16_broadcast_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_half_broadcast_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_float_broadcast_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_double_broadcast_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_char_broadcast_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_short_broadcast_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_schar_broadcast_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_int_broadcast_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_long_broadcast_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_longlong_broadcast_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_int8_broadcast_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_int16_broadcast_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_int32_broadcast_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_int64_broadcast_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_uint8_broadcast_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_uint16_broadcast_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_uint32_broadcast_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_uint64_broadcast_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_size_broadcast_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_broadcast_on_stream(team, dest, src, nelem, PE_root, stream)


cdef int nvshmemx_bfloat16_fcollect_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_half_fcollect_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_float_fcollect_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_double_fcollect_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_char_fcollect_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_short_fcollect_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_schar_fcollect_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int_fcollect_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_long_fcollect_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_longlong_fcollect_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int8_fcollect_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int16_fcollect_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int32_fcollect_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int64_fcollect_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint8_fcollect_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint16_fcollect_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint32_fcollect_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_uint64_fcollect_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_size_fcollect_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_fcollect_on_stream(team, dest, src, nelem, stream)


cdef int nvshmemx_int8_max_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_max_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_max_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_max_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_max_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_max_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_max_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_max_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_max_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_max_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_max_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_max_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_max_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_max_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_max_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_max_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_max_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_max_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_max_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_max_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int8_min_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_min_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_min_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_min_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_min_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_min_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_min_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_min_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_min_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_min_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_min_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_min_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_min_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_min_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_min_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_min_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_min_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_min_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_min_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_min_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int8_sum_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_sum_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_sum_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_sum_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_sum_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_sum_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_sum_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_sum_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_sum_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_sum_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_sum_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_sum_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_sum_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_sum_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_sum_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_sum_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_sum_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_sum_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_sum_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_sum_reduce_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int8_max_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_max_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_max_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_max_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_max_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_max_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_max_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_max_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_max_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_max_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_max_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_max_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_max_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_max_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_max_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_max_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_max_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_max_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_max_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_max_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int8_min_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_min_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_min_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_min_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_min_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_min_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_min_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_min_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_min_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_min_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_min_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_min_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_min_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_min_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_min_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_min_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_min_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_min_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_min_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_min_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int8_sum_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int8_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int16_sum_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int16_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int32_sum_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int32_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int64_sum_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int64_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint8_sum_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint8_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint16_sum_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint16_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint32_sum_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint32_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_uint64_sum_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_uint64_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_size_sum_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_size_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_char_sum_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_char_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_schar_sum_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_schar_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_short_sum_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_short_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_int_sum_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_int_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_long_sum_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_long_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_longlong_sum_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_longlong_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_bfloat16_sum_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_bfloat16_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_half_sum_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_half_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_float_sum_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_float_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_double_sum_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_double_sum_reducescatter_on_stream(team, dest, src, nreduce, stream)


cdef int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except* nogil:
    return _nvshmem._nvshmemx_hostlib_init_attr(flags, attr)


cdef void nvshmemx_hostlib_finalize() except* nogil:
    _nvshmem._nvshmemx_hostlib_finalize()


cdef int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except* nogil:
    return _nvshmem._nvshmemx_set_attr_uniqueid_args(myrank, nranks, uniqueid, attr)


cdef int nvshmemx_set_attr_mpi_comm_args(void* mpi_comm, nvshmemx_init_attr_t* nvshmem_attr) except* nogil:
    return _nvshmem._nvshmemx_set_attr_mpi_comm_args(mpi_comm, nvshmem_attr)


cdef int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except* nogil:
    return _nvshmem._nvshmemx_get_uniqueid(uniqueid)


cdef int nvshmemx_cumodule_init(CUmodule module) except* nogil:
    return _nvshmem._nvshmemx_cumodule_init(module)


cdef int nvshmemx_cumodule_finalize(CUmodule module) except* nogil:
    return _nvshmem._nvshmemx_cumodule_finalize(module)


cdef void nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_signal_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm)


cdef void nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_getmem_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_quiet_on_stream(cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_quiet_on_stream(cstrm)


cdef void nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value, cudaStream_t cstream) except* nogil:
    _nvshmem._nvshmemx_signal_wait_until_on_stream(sig_addr, cmp, cmp_value, cstream)
