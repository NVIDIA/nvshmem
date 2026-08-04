# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated from NVSHMEM with version 3.3.0. 
# Modify it directly at your own risk.



from ._internal cimport nvshmem as _nvshmem


###############################################################################
# Wrapper functions
###############################################################################

cdef int nvshmem_barrier(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_barrier(team)


cdef void nvshmem_barrier_all() except* nogil:
    _nvshmem._nvshmem_barrier_all()


cdef int nvshmemx_init_status() except* nogil:
    return _nvshmem._nvshmemx_init_status()


cdef void nvshmem_query_thread(int* provided) except* nogil:
    _nvshmem._nvshmem_query_thread(provided)


cdef int nvshmem_my_pe() except* nogil:
    return _nvshmem._nvshmem_my_pe()


cdef int nvshmem_n_pes() except* nogil:
    return _nvshmem._nvshmem_n_pes()


cdef void nvshmem_info_get_version(int* major, int* minor) except* nogil:
    _nvshmem._nvshmem_info_get_version(major, minor)


cdef void nvshmem_info_get_name(char* name) except* nogil:
    _nvshmem._nvshmem_info_get_name(name)


cdef void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) except* nogil:
    _nvshmem._nvshmemx_vendor_get_version_info(major, minor, patch)


cdef void* nvshmem_malloc(size_t size) except* nogil:
    return _nvshmem._nvshmem_malloc(size)


cdef void* nvshmem_calloc(size_t count, size_t size) except* nogil:
    return _nvshmem._nvshmem_calloc(count, size)


cdef void* nvshmem_align(size_t alignment, size_t size) except* nogil:
    return _nvshmem._nvshmem_align(alignment, size)


cdef void nvshmem_free(void* ptr) except* nogil:
    _nvshmem._nvshmem_free(ptr)


cdef void* nvshmem_ptr(const void* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ptr(dest, pe)


cdef void* nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr) except* nogil:
    return _nvshmem._nvshmemx_mc_ptr(team, ptr)


cdef void nvshmem_uint_atomic_inc(unsigned int* dest, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_inc(dest, pe)


cdef void nvshmem_ulong_atomic_inc(unsigned long* dest, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_inc(dest, pe)


cdef void nvshmem_ulonglong_atomic_inc(unsigned long long* dest, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_inc(dest, pe)


cdef void nvshmem_int32_atomic_inc(int32_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_inc(dest, pe)


cdef void nvshmem_uint32_atomic_inc(uint32_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_inc(dest, pe)


cdef void nvshmem_int64_atomic_inc(int64_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_inc(dest, pe)


cdef void nvshmem_uint64_atomic_inc(uint64_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_inc(dest, pe)


cdef void nvshmem_int_atomic_inc(int* dest, int pe) except* nogil:
    _nvshmem._nvshmem_int_atomic_inc(dest, pe)


cdef void nvshmem_long_atomic_inc(long* dest, int pe) except* nogil:
    _nvshmem._nvshmem_long_atomic_inc(dest, pe)


cdef void nvshmem_longlong_atomic_inc(long long* dest, int pe) except* nogil:
    _nvshmem._nvshmem_longlong_atomic_inc(dest, pe)


cdef void nvshmem_size_atomic_inc(size_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_size_atomic_inc(dest, pe)


cdef void nvshmem_ptrdiff_atomic_inc(ptrdiff_t* dest, int pe) except* nogil:
    _nvshmem._nvshmem_ptrdiff_atomic_inc(dest, pe)


cdef unsigned int nvshmem_uint_atomic_fetch_inc(unsigned int* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch_inc(dest, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch_inc(unsigned long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch_inc(dest, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch_inc(unsigned long long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch_inc(dest, pe)


cdef int32_t nvshmem_int32_atomic_fetch_inc(int32_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch_inc(dest, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch_inc(uint32_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch_inc(dest, pe)


cdef int64_t nvshmem_int64_atomic_fetch_inc(int64_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch_inc(dest, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch_inc(uint64_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch_inc(dest, pe)


cdef int nvshmem_int_atomic_fetch_inc(int* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int_atomic_fetch_inc(dest, pe)


cdef long nvshmem_long_atomic_fetch_inc(long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_long_atomic_fetch_inc(dest, pe)


cdef long long nvshmem_longlong_atomic_fetch_inc(long long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_longlong_atomic_fetch_inc(dest, pe)


cdef size_t nvshmem_size_atomic_fetch_inc(size_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_size_atomic_fetch_inc(dest, pe)


cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch_inc(ptrdiff_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ptrdiff_atomic_fetch_inc(dest, pe)


cdef unsigned int nvshmem_uint_atomic_fetch(const unsigned int* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch(dest, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch(const unsigned long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch(dest, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch(const unsigned long long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch(dest, pe)


cdef int32_t nvshmem_int32_atomic_fetch(const int32_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch(dest, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch(const uint32_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch(dest, pe)


cdef int64_t nvshmem_int64_atomic_fetch(const int64_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch(dest, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch(const uint64_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch(dest, pe)


cdef int nvshmem_int_atomic_fetch(const int* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_int_atomic_fetch(dest, pe)


cdef long nvshmem_long_atomic_fetch(const long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_long_atomic_fetch(dest, pe)


cdef long long nvshmem_longlong_atomic_fetch(const long long* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_longlong_atomic_fetch(dest, pe)


cdef size_t nvshmem_size_atomic_fetch(const size_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_size_atomic_fetch(dest, pe)


cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch(const ptrdiff_t* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_ptrdiff_atomic_fetch(dest, pe)


cdef float nvshmem_float_atomic_fetch(const float* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_float_atomic_fetch(dest, pe)


cdef double nvshmem_double_atomic_fetch(const double* dest, int pe) except* nogil:
    return _nvshmem._nvshmem_double_atomic_fetch(dest, pe)


cdef void nvshmem_uint_atomic_add(unsigned int* dest, unsigned int value, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_add(dest, value, pe)


cdef void nvshmem_ulong_atomic_add(unsigned long* dest, unsigned long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_add(dest, value, pe)


cdef void nvshmem_ulonglong_atomic_add(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_add(dest, value, pe)


cdef void nvshmem_int32_atomic_add(int32_t* dest, int32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_add(dest, value, pe)


cdef void nvshmem_uint32_atomic_add(uint32_t* dest, uint32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_add(dest, value, pe)


cdef void nvshmem_int64_atomic_add(int64_t* dest, int64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_add(dest, value, pe)


cdef void nvshmem_uint64_atomic_add(uint64_t* dest, uint64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_add(dest, value, pe)


cdef void nvshmem_int_atomic_add(int* dest, int value, int pe) except* nogil:
    _nvshmem._nvshmem_int_atomic_add(dest, value, pe)


cdef void nvshmem_long_atomic_add(long* dest, long value, int pe) except* nogil:
    _nvshmem._nvshmem_long_atomic_add(dest, value, pe)


cdef void nvshmem_longlong_atomic_add(long long* dest, long long value, int pe) except* nogil:
    _nvshmem._nvshmem_longlong_atomic_add(dest, value, pe)


cdef void nvshmem_size_atomic_add(size_t* dest, size_t value, int pe) except* nogil:
    _nvshmem._nvshmem_size_atomic_add(dest, value, pe)


cdef void nvshmem_ptrdiff_atomic_add(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil:
    _nvshmem._nvshmem_ptrdiff_atomic_add(dest, value, pe)


cdef void nvshmemx_float_atomic_add(float* dest, float value, int pe) except* nogil:
    _nvshmem._nvshmemx_float_atomic_add(dest, value, pe)


cdef void nvshmemx_double_atomic_add(double* dest, double value, int pe) except* nogil:
    _nvshmem._nvshmemx_double_atomic_add(dest, value, pe)


cdef float nvshmemx_float_atomic_fetch_add(float* dest, float value, int pe) except* nogil:
    return _nvshmem._nvshmemx_float_atomic_fetch_add(dest, value, pe)


cdef double nvshmemx_double_atomic_fetch_add(double* dest, double value, int pe) except* nogil:
    return _nvshmem._nvshmemx_double_atomic_fetch_add(dest, value, pe)


cdef void nvshmem_uint_atomic_set(unsigned int* dest, unsigned int value, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_set(dest, value, pe)


cdef void nvshmem_ulong_atomic_set(unsigned long* dest, unsigned long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_set(dest, value, pe)


cdef void nvshmem_ulonglong_atomic_set(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_set(dest, value, pe)


cdef void nvshmem_int32_atomic_set(int32_t* dest, int32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_set(dest, value, pe)


cdef void nvshmem_uint32_atomic_set(uint32_t* dest, uint32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_set(dest, value, pe)


cdef void nvshmem_int64_atomic_set(int64_t* dest, int64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_set(dest, value, pe)


cdef void nvshmem_uint64_atomic_set(uint64_t* dest, uint64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_set(dest, value, pe)


cdef void nvshmem_int_atomic_set(int* dest, int value, int pe) except* nogil:
    _nvshmem._nvshmem_int_atomic_set(dest, value, pe)


cdef void nvshmem_long_atomic_set(long* dest, long value, int pe) except* nogil:
    _nvshmem._nvshmem_long_atomic_set(dest, value, pe)


cdef void nvshmem_longlong_atomic_set(long long* dest, long long value, int pe) except* nogil:
    _nvshmem._nvshmem_longlong_atomic_set(dest, value, pe)


cdef void nvshmem_size_atomic_set(size_t* dest, size_t value, int pe) except* nogil:
    _nvshmem._nvshmem_size_atomic_set(dest, value, pe)


cdef void nvshmem_ptrdiff_atomic_set(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil:
    _nvshmem._nvshmem_ptrdiff_atomic_set(dest, value, pe)


cdef void nvshmem_float_atomic_set(float* dest, float value, int pe) except* nogil:
    _nvshmem._nvshmem_float_atomic_set(dest, value, pe)


cdef void nvshmem_double_atomic_set(double* dest, double value, int pe) except* nogil:
    _nvshmem._nvshmem_double_atomic_set(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_fetch_add(unsigned int* dest, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch_add(dest, value, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch_add(unsigned long* dest, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch_add(dest, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch_add(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch_add(dest, value, pe)


cdef int32_t nvshmem_int32_atomic_fetch_add(int32_t* dest, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch_add(dest, value, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch_add(uint32_t* dest, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch_add(dest, value, pe)


cdef int64_t nvshmem_int64_atomic_fetch_add(int64_t* dest, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch_add(dest, value, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch_add(uint64_t* dest, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch_add(dest, value, pe)


cdef int nvshmem_int_atomic_fetch_add(int* dest, int value, int pe) except* nogil:
    return _nvshmem._nvshmem_int_atomic_fetch_add(dest, value, pe)


cdef long nvshmem_long_atomic_fetch_add(long* dest, long value, int pe) except* nogil:
    return _nvshmem._nvshmem_long_atomic_fetch_add(dest, value, pe)


cdef long long nvshmem_longlong_atomic_fetch_add(long long* dest, long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_longlong_atomic_fetch_add(dest, value, pe)


cdef size_t nvshmem_size_atomic_fetch_add(size_t* dest, size_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_size_atomic_fetch_add(dest, value, pe)


cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch_add(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_ptrdiff_atomic_fetch_add(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_swap(unsigned int* dest, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_swap(dest, value, pe)


cdef unsigned long nvshmem_ulong_atomic_swap(unsigned long* dest, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_swap(dest, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_swap(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_swap(dest, value, pe)


cdef int32_t nvshmem_int32_atomic_swap(int32_t* dest, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_swap(dest, value, pe)


cdef uint32_t nvshmem_uint32_atomic_swap(uint32_t* dest, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_swap(dest, value, pe)


cdef int64_t nvshmem_int64_atomic_swap(int64_t* dest, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_swap(dest, value, pe)


cdef uint64_t nvshmem_uint64_atomic_swap(uint64_t* dest, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_swap(dest, value, pe)


cdef int nvshmem_int_atomic_swap(int* dest, int value, int pe) except* nogil:
    return _nvshmem._nvshmem_int_atomic_swap(dest, value, pe)


cdef long nvshmem_long_atomic_swap(long* dest, long value, int pe) except* nogil:
    return _nvshmem._nvshmem_long_atomic_swap(dest, value, pe)


cdef long long nvshmem_longlong_atomic_swap(long long* dest, long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_longlong_atomic_swap(dest, value, pe)


cdef size_t nvshmem_size_atomic_swap(size_t* dest, size_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_size_atomic_swap(dest, value, pe)


cdef ptrdiff_t nvshmem_ptrdiff_atomic_swap(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_ptrdiff_atomic_swap(dest, value, pe)


cdef float nvshmem_float_atomic_swap(float* dest, float value, int pe) except* nogil:
    return _nvshmem._nvshmem_float_atomic_swap(dest, value, pe)


cdef double nvshmem_double_atomic_swap(double* dest, double value, int pe) except* nogil:
    return _nvshmem._nvshmem_double_atomic_swap(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_compare_swap(unsigned int* dest, unsigned int cond, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_compare_swap(dest, cond, value, pe)


cdef unsigned long nvshmem_ulong_atomic_compare_swap(unsigned long* dest, unsigned long cond, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_compare_swap(dest, cond, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_compare_swap(unsigned long long* dest, unsigned long long cond, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_compare_swap(dest, cond, value, pe)


cdef int32_t nvshmem_int32_atomic_compare_swap(int32_t* dest, int32_t cond, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_compare_swap(dest, cond, value, pe)


cdef uint32_t nvshmem_uint32_atomic_compare_swap(uint32_t* dest, uint32_t cond, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_compare_swap(dest, cond, value, pe)


cdef int64_t nvshmem_int64_atomic_compare_swap(int64_t* dest, int64_t cond, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_compare_swap(dest, cond, value, pe)


cdef uint64_t nvshmem_uint64_atomic_compare_swap(uint64_t* dest, uint64_t cond, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_compare_swap(dest, cond, value, pe)


cdef int nvshmem_int_atomic_compare_swap(int* dest, int cond, int value, int pe) except* nogil:
    return _nvshmem._nvshmem_int_atomic_compare_swap(dest, cond, value, pe)


cdef long nvshmem_long_atomic_compare_swap(long* dest, long cond, long value, int pe) except* nogil:
    return _nvshmem._nvshmem_long_atomic_compare_swap(dest, cond, value, pe)


cdef long long nvshmem_longlong_atomic_compare_swap(long long* dest, long long cond, long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_longlong_atomic_compare_swap(dest, cond, value, pe)


cdef size_t nvshmem_size_atomic_compare_swap(size_t* dest, size_t cond, size_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_size_atomic_compare_swap(dest, cond, value, pe)


cdef ptrdiff_t nvshmem_ptrdiff_atomic_compare_swap(ptrdiff_t* dest, ptrdiff_t cond, ptrdiff_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_ptrdiff_atomic_compare_swap(dest, cond, value, pe)


cdef void nvshmem_uint_atomic_and(unsigned int* dest, unsigned int value, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_and(dest, value, pe)


cdef void nvshmem_ulong_atomic_and(unsigned long* dest, unsigned long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_and(dest, value, pe)


cdef void nvshmem_ulonglong_atomic_and(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_and(dest, value, pe)


cdef void nvshmem_int32_atomic_and(int32_t* dest, int32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_and(dest, value, pe)


cdef void nvshmem_uint32_atomic_and(uint32_t* dest, uint32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_and(dest, value, pe)


cdef void nvshmem_int64_atomic_and(int64_t* dest, int64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_and(dest, value, pe)


cdef void nvshmem_uint64_atomic_and(uint64_t* dest, uint64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_and(dest, value, pe)


cdef void nvshmem_uint_atomic_or(unsigned int* dest, unsigned int value, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_or(dest, value, pe)


cdef void nvshmem_ulong_atomic_or(unsigned long* dest, unsigned long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_or(dest, value, pe)


cdef void nvshmem_ulonglong_atomic_or(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_or(dest, value, pe)


cdef void nvshmem_int32_atomic_or(int32_t* dest, int32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_or(dest, value, pe)


cdef void nvshmem_uint32_atomic_or(uint32_t* dest, uint32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_or(dest, value, pe)


cdef void nvshmem_int64_atomic_or(int64_t* dest, int64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_or(dest, value, pe)


cdef void nvshmem_uint64_atomic_or(uint64_t* dest, uint64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_or(dest, value, pe)


cdef void nvshmem_uint_atomic_xor(unsigned int* dest, unsigned int value, int pe) except* nogil:
    _nvshmem._nvshmem_uint_atomic_xor(dest, value, pe)


cdef void nvshmem_ulong_atomic_xor(unsigned long* dest, unsigned long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulong_atomic_xor(dest, value, pe)


cdef void nvshmem_ulonglong_atomic_xor(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    _nvshmem._nvshmem_ulonglong_atomic_xor(dest, value, pe)


cdef void nvshmem_int32_atomic_xor(int32_t* dest, int32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int32_atomic_xor(dest, value, pe)


cdef void nvshmem_uint32_atomic_xor(uint32_t* dest, uint32_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint32_atomic_xor(dest, value, pe)


cdef void nvshmem_int64_atomic_xor(int64_t* dest, int64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_int64_atomic_xor(dest, value, pe)


cdef void nvshmem_uint64_atomic_xor(uint64_t* dest, uint64_t value, int pe) except* nogil:
    _nvshmem._nvshmem_uint64_atomic_xor(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_fetch_and(unsigned int* dest, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch_and(dest, value, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch_and(unsigned long* dest, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch_and(dest, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch_and(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch_and(dest, value, pe)


cdef int32_t nvshmem_int32_atomic_fetch_and(int32_t* dest, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch_and(dest, value, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch_and(uint32_t* dest, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch_and(dest, value, pe)


cdef int64_t nvshmem_int64_atomic_fetch_and(int64_t* dest, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch_and(dest, value, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch_and(uint64_t* dest, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch_and(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_fetch_or(unsigned int* dest, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch_or(dest, value, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch_or(unsigned long* dest, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch_or(dest, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch_or(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch_or(dest, value, pe)


cdef int32_t nvshmem_int32_atomic_fetch_or(int32_t* dest, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch_or(dest, value, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch_or(uint32_t* dest, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch_or(dest, value, pe)


cdef int64_t nvshmem_int64_atomic_fetch_or(int64_t* dest, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch_or(dest, value, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch_or(uint64_t* dest, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch_or(dest, value, pe)


cdef unsigned int nvshmem_uint_atomic_fetch_xor(unsigned int* dest, unsigned int value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint_atomic_fetch_xor(dest, value, pe)


cdef unsigned long nvshmem_ulong_atomic_fetch_xor(unsigned long* dest, unsigned long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulong_atomic_fetch_xor(dest, value, pe)


cdef unsigned long long nvshmem_ulonglong_atomic_fetch_xor(unsigned long long* dest, unsigned long long value, int pe) except* nogil:
    return _nvshmem._nvshmem_ulonglong_atomic_fetch_xor(dest, value, pe)


cdef int32_t nvshmem_int32_atomic_fetch_xor(int32_t* dest, int32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int32_atomic_fetch_xor(dest, value, pe)


cdef uint32_t nvshmem_uint32_atomic_fetch_xor(uint32_t* dest, uint32_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint32_atomic_fetch_xor(dest, value, pe)


cdef int64_t nvshmem_int64_atomic_fetch_xor(int64_t* dest, int64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_int64_atomic_fetch_xor(dest, value, pe)


cdef uint64_t nvshmem_uint64_atomic_fetch_xor(uint64_t* dest, uint64_t value, int pe) except* nogil:
    return _nvshmem._nvshmem_uint64_atomic_fetch_xor(dest, value, pe)


cdef uint64_t nvshmem_signal_fetch(uint64_t* sig_addr) except* nogil:
    return _nvshmem._nvshmem_signal_fetch(sig_addr)


cdef int nvshmem_team_my_pe(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_team_my_pe(team)


cdef int nvshmem_team_n_pes(nvshmem_team_t team) except* nogil:
    return _nvshmem._nvshmem_team_n_pes(team)


cdef void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t* config) except* nogil:
    _nvshmem._nvshmem_team_get_config(team, config)


cdef int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) except* nogil:
    return _nvshmem._nvshmem_team_translate_pe(src_team, src_pe, dest_team)


cdef int nvshmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride, int PE_size, const nvshmem_team_config_t* config, long config_mask, nvshmem_team_t* new_team) except* nogil:
    return _nvshmem._nvshmem_team_split_strided(parent_team, PE_start, PE_stride, PE_size, config, config_mask, new_team)


cdef int nvshmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t* uniqueid) except* nogil:
    return _nvshmem._nvshmemx_team_get_uniqueid(uniqueid)


cdef int nvshmemx_team_init(nvshmem_team_t* team, nvshmem_team_config_t* config, long config_mask, int npes, int pe_idx_in_team) except* nogil:
    return _nvshmem._nvshmemx_team_init(team, config, config_mask, npes, pe_idx_in_team)


cdef int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange, const nvshmem_team_config_t* xaxis_config, long xaxis_mask, nvshmem_team_t* xaxis_team, const nvshmem_team_config_t* yaxis_config, long yaxis_mask, nvshmem_team_t* yaxis_team) except* nogil:
    return _nvshmem._nvshmem_team_split_2d(parent_team, xrange, xaxis_config, xaxis_mask, xaxis_team, yaxis_config, yaxis_mask, yaxis_team)


cdef void nvshmem_team_destroy(nvshmem_team_t team) except* nogil:
    _nvshmem._nvshmem_team_destroy(team)


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


cdef void nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil:
    _nvshmem._nvshmemx_barrier_all_on_stream(stream)


cdef int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil:
    return _nvshmem._nvshmemx_team_sync_on_stream(team, stream)


cdef void nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil:
    _nvshmem._nvshmemx_sync_all_on_stream(stream)


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


cdef void* nvshmemx_buffer_register_symmetric(void* buf_ptr, size_t size, int flags) except* nogil:
    return _nvshmem._nvshmemx_buffer_register_symmetric(buf_ptr, size, flags)


cdef void* nvshmemx_buffer_register_symmetric_at_preferred_address(void* buf_ptr, size_t size, void* preferred_addr, int flags) except* nogil:
    return _nvshmem._nvshmemx_buffer_register_symmetric_at_preferred_address(buf_ptr, size, preferred_addr, flags)


cdef int nvshmemx_buffer_unregister_symmetric(void* mmap_ptr, size_t size) except* nogil:
    return _nvshmem._nvshmemx_buffer_unregister_symmetric(mmap_ptr, size)


cdef int nvshmemx_culibrary_init(CUlibrary library) except* nogil:
    return _nvshmem._nvshmemx_culibrary_init(library)


cdef int nvshmemx_culibrary_finalize(CUlibrary library) except* nogil:
    return _nvshmem._nvshmemx_culibrary_finalize(library)


cdef void nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_signal_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm)


cdef void nvshmemx_putmem_signal_nbi_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_signal_nbi_on_stream(dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm)


cdef void nvshmemx_putmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_putmem_nbi_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_getmem_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_getmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_getmem_nbi_on_stream(dest, source, bytes, pe, cstrm)


cdef void nvshmemx_quiet_on_stream(cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_quiet_on_stream(cstrm)


cdef void nvshmemx_flush_on_stream(cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_flush_on_stream(cstrm)


cdef void nvshmemx_signal_op_on_stream(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil:
    _nvshmem._nvshmemx_signal_op_on_stream(sig_addr, signal, sig_op, pe, cstrm)


cdef void nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value, cudaStream_t cstream) except* nogil:
    _nvshmem._nvshmemx_signal_wait_until_on_stream(sig_addr, cmp, cmp_value, cstream)
