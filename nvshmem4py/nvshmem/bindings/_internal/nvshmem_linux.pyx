# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated with version 3.3.0. Do not modify it directly.

from libc.stdint cimport intptr_t

class FunctionNotFoundError(RuntimeError): pass

class NotSupportedError(RuntimeError): pass



###############################################################################
# Extern
###############################################################################

cdef extern from "<dlfcn.h>" nogil:
    void* dlopen(const char*, int)
    char* dlerror()
    void* dlsym(void*, const char*)
    int dlclose(void*)

    enum:
        RTLD_LAZY
        RTLD_NOW
        RTLD_GLOBAL
        RTLD_LOCAL

    const void* RTLD_DEFAULT 'RTLD_DEFAULT'


###############################################################################
# Wrapper init
###############################################################################

cdef bint __py_nvshmem_init = False

cdef void* __nvshmemx_init_status = NULL
cdef void* __nvshmem_my_pe = NULL
cdef void* __nvshmem_n_pes = NULL
cdef void* __nvshmem_info_get_version = NULL
cdef void* __nvshmemx_vendor_get_version_info = NULL
cdef void* __nvshmem_malloc = NULL
cdef void* __nvshmem_calloc = NULL
cdef void* __nvshmem_align = NULL
cdef void* __nvshmem_free = NULL
cdef void* __nvshmem_ptr = NULL
cdef void* __nvshmemx_mc_ptr = NULL
cdef void* __nvshmem_team_my_pe = NULL
cdef void* __nvshmem_team_n_pes = NULL
cdef void* __nvshmem_barrier = NULL
cdef void* __nvshmem_barrier_all = NULL
cdef void* __nvshmemx_bfloat16_alltoall_on_stream = NULL
cdef void* __nvshmemx_half_alltoall_on_stream = NULL
cdef void* __nvshmemx_float_alltoall_on_stream = NULL
cdef void* __nvshmemx_double_alltoall_on_stream = NULL
cdef void* __nvshmemx_char_alltoall_on_stream = NULL
cdef void* __nvshmemx_short_alltoall_on_stream = NULL
cdef void* __nvshmemx_schar_alltoall_on_stream = NULL
cdef void* __nvshmemx_int_alltoall_on_stream = NULL
cdef void* __nvshmemx_long_alltoall_on_stream = NULL
cdef void* __nvshmemx_longlong_alltoall_on_stream = NULL
cdef void* __nvshmemx_int8_alltoall_on_stream = NULL
cdef void* __nvshmemx_int16_alltoall_on_stream = NULL
cdef void* __nvshmemx_int32_alltoall_on_stream = NULL
cdef void* __nvshmemx_int64_alltoall_on_stream = NULL
cdef void* __nvshmemx_uint8_alltoall_on_stream = NULL
cdef void* __nvshmemx_uint16_alltoall_on_stream = NULL
cdef void* __nvshmemx_uint32_alltoall_on_stream = NULL
cdef void* __nvshmemx_uint64_alltoall_on_stream = NULL
cdef void* __nvshmemx_size_alltoall_on_stream = NULL
cdef void* __nvshmemx_barrier_on_stream = NULL
cdef void* __nvshmemx_team_sync_on_stream = NULL
cdef void* __nvshmemx_bfloat16_broadcast_on_stream = NULL
cdef void* __nvshmemx_half_broadcast_on_stream = NULL
cdef void* __nvshmemx_float_broadcast_on_stream = NULL
cdef void* __nvshmemx_double_broadcast_on_stream = NULL
cdef void* __nvshmemx_char_broadcast_on_stream = NULL
cdef void* __nvshmemx_short_broadcast_on_stream = NULL
cdef void* __nvshmemx_schar_broadcast_on_stream = NULL
cdef void* __nvshmemx_int_broadcast_on_stream = NULL
cdef void* __nvshmemx_long_broadcast_on_stream = NULL
cdef void* __nvshmemx_longlong_broadcast_on_stream = NULL
cdef void* __nvshmemx_int8_broadcast_on_stream = NULL
cdef void* __nvshmemx_int16_broadcast_on_stream = NULL
cdef void* __nvshmemx_int32_broadcast_on_stream = NULL
cdef void* __nvshmemx_int64_broadcast_on_stream = NULL
cdef void* __nvshmemx_uint8_broadcast_on_stream = NULL
cdef void* __nvshmemx_uint16_broadcast_on_stream = NULL
cdef void* __nvshmemx_uint32_broadcast_on_stream = NULL
cdef void* __nvshmemx_uint64_broadcast_on_stream = NULL
cdef void* __nvshmemx_size_broadcast_on_stream = NULL
cdef void* __nvshmemx_bfloat16_fcollect_on_stream = NULL
cdef void* __nvshmemx_half_fcollect_on_stream = NULL
cdef void* __nvshmemx_float_fcollect_on_stream = NULL
cdef void* __nvshmemx_double_fcollect_on_stream = NULL
cdef void* __nvshmemx_char_fcollect_on_stream = NULL
cdef void* __nvshmemx_short_fcollect_on_stream = NULL
cdef void* __nvshmemx_schar_fcollect_on_stream = NULL
cdef void* __nvshmemx_int_fcollect_on_stream = NULL
cdef void* __nvshmemx_long_fcollect_on_stream = NULL
cdef void* __nvshmemx_longlong_fcollect_on_stream = NULL
cdef void* __nvshmemx_int8_fcollect_on_stream = NULL
cdef void* __nvshmemx_int16_fcollect_on_stream = NULL
cdef void* __nvshmemx_int32_fcollect_on_stream = NULL
cdef void* __nvshmemx_int64_fcollect_on_stream = NULL
cdef void* __nvshmemx_uint8_fcollect_on_stream = NULL
cdef void* __nvshmemx_uint16_fcollect_on_stream = NULL
cdef void* __nvshmemx_uint32_fcollect_on_stream = NULL
cdef void* __nvshmemx_uint64_fcollect_on_stream = NULL
cdef void* __nvshmemx_size_fcollect_on_stream = NULL
cdef void* __nvshmemx_int8_max_reduce_on_stream = NULL
cdef void* __nvshmemx_int16_max_reduce_on_stream = NULL
cdef void* __nvshmemx_int32_max_reduce_on_stream = NULL
cdef void* __nvshmemx_int64_max_reduce_on_stream = NULL
cdef void* __nvshmemx_uint8_max_reduce_on_stream = NULL
cdef void* __nvshmemx_uint16_max_reduce_on_stream = NULL
cdef void* __nvshmemx_uint32_max_reduce_on_stream = NULL
cdef void* __nvshmemx_uint64_max_reduce_on_stream = NULL
cdef void* __nvshmemx_size_max_reduce_on_stream = NULL
cdef void* __nvshmemx_char_max_reduce_on_stream = NULL
cdef void* __nvshmemx_schar_max_reduce_on_stream = NULL
cdef void* __nvshmemx_short_max_reduce_on_stream = NULL
cdef void* __nvshmemx_int_max_reduce_on_stream = NULL
cdef void* __nvshmemx_long_max_reduce_on_stream = NULL
cdef void* __nvshmemx_longlong_max_reduce_on_stream = NULL
cdef void* __nvshmemx_bfloat16_max_reduce_on_stream = NULL
cdef void* __nvshmemx_half_max_reduce_on_stream = NULL
cdef void* __nvshmemx_float_max_reduce_on_stream = NULL
cdef void* __nvshmemx_double_max_reduce_on_stream = NULL
cdef void* __nvshmemx_int8_min_reduce_on_stream = NULL
cdef void* __nvshmemx_int16_min_reduce_on_stream = NULL
cdef void* __nvshmemx_int32_min_reduce_on_stream = NULL
cdef void* __nvshmemx_int64_min_reduce_on_stream = NULL
cdef void* __nvshmemx_uint8_min_reduce_on_stream = NULL
cdef void* __nvshmemx_uint16_min_reduce_on_stream = NULL
cdef void* __nvshmemx_uint32_min_reduce_on_stream = NULL
cdef void* __nvshmemx_uint64_min_reduce_on_stream = NULL
cdef void* __nvshmemx_size_min_reduce_on_stream = NULL
cdef void* __nvshmemx_char_min_reduce_on_stream = NULL
cdef void* __nvshmemx_schar_min_reduce_on_stream = NULL
cdef void* __nvshmemx_short_min_reduce_on_stream = NULL
cdef void* __nvshmemx_int_min_reduce_on_stream = NULL
cdef void* __nvshmemx_long_min_reduce_on_stream = NULL
cdef void* __nvshmemx_longlong_min_reduce_on_stream = NULL
cdef void* __nvshmemx_bfloat16_min_reduce_on_stream = NULL
cdef void* __nvshmemx_half_min_reduce_on_stream = NULL
cdef void* __nvshmemx_float_min_reduce_on_stream = NULL
cdef void* __nvshmemx_double_min_reduce_on_stream = NULL
cdef void* __nvshmemx_int8_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_int16_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_int32_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_int64_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_uint8_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_uint16_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_uint32_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_uint64_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_size_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_char_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_schar_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_short_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_int_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_long_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_longlong_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_bfloat16_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_half_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_float_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_double_sum_reduce_on_stream = NULL
cdef void* __nvshmemx_int8_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int16_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int32_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int64_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint8_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint16_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint32_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint64_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_size_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_char_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_schar_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_short_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_long_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_longlong_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_bfloat16_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_half_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_float_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_double_max_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int8_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int16_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int32_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int64_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint8_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint16_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint32_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint64_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_size_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_char_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_schar_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_short_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_long_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_longlong_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_bfloat16_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_half_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_float_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_double_min_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int8_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int16_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int32_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int64_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint8_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint16_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint32_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_uint64_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_size_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_char_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_schar_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_short_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_int_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_long_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_longlong_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_bfloat16_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_half_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_float_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_double_sum_reducescatter_on_stream = NULL
cdef void* __nvshmemx_hostlib_init_attr = NULL
cdef void* __nvshmemx_hostlib_finalize = NULL
cdef void* __nvshmemx_set_attr_uniqueid_args = NULL
cdef void* __nvshmemx_set_attr_mpi_comm_args = NULL
cdef void* __nvshmemx_get_uniqueid = NULL
cdef void* __nvshmemx_cumodule_init = NULL
cdef void* __nvshmemx_cumodule_finalize = NULL
cdef void* __nvshmemx_putmem_on_stream = NULL
cdef void* __nvshmemx_putmem_signal_on_stream = NULL
cdef void* __nvshmemx_getmem_on_stream = NULL
cdef void* __nvshmemx_quiet_on_stream = NULL
cdef void* __nvshmemx_signal_wait_until_on_stream = NULL


cdef void* load_library() except* nogil:
    cdef void* handle
    handle = dlopen("libnvshmem_host.so.3", RTLD_NOW | RTLD_GLOBAL)
    if handle == NULL:
        with gil:
            err_msg = dlerror()
            raise RuntimeError(f'Failed to dlopen libnvshmem ({err_msg.decode()})')
    return handle


cdef int _check_or_init_nvshmem() except -1 nogil:
    global __py_nvshmem_init
    if __py_nvshmem_init:
        return 0

    # Load function
    cdef void* handle = NULL
    global __nvshmemx_init_status
    __nvshmemx_init_status = dlsym(RTLD_DEFAULT, 'nvshmemx_init_status')
    if __nvshmemx_init_status == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_init_status = dlsym(handle, 'nvshmemx_init_status')

    global __nvshmem_my_pe
    __nvshmem_my_pe = dlsym(RTLD_DEFAULT, 'nvshmem_my_pe')
    if __nvshmem_my_pe == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_my_pe = dlsym(handle, 'nvshmem_my_pe')

    global __nvshmem_n_pes
    __nvshmem_n_pes = dlsym(RTLD_DEFAULT, 'nvshmem_n_pes')
    if __nvshmem_n_pes == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_n_pes = dlsym(handle, 'nvshmem_n_pes')

    global __nvshmem_info_get_version
    __nvshmem_info_get_version = dlsym(RTLD_DEFAULT, 'nvshmem_info_get_version')
    if __nvshmem_info_get_version == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_info_get_version = dlsym(handle, 'nvshmem_info_get_version')

    global __nvshmemx_vendor_get_version_info
    __nvshmemx_vendor_get_version_info = dlsym(RTLD_DEFAULT, 'nvshmemx_vendor_get_version_info')
    if __nvshmemx_vendor_get_version_info == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_vendor_get_version_info = dlsym(handle, 'nvshmemx_vendor_get_version_info')

    global __nvshmem_malloc
    __nvshmem_malloc = dlsym(RTLD_DEFAULT, 'nvshmem_malloc')
    if __nvshmem_malloc == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_malloc = dlsym(handle, 'nvshmem_malloc')

    global __nvshmem_calloc
    __nvshmem_calloc = dlsym(RTLD_DEFAULT, 'nvshmem_calloc')
    if __nvshmem_calloc == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_calloc = dlsym(handle, 'nvshmem_calloc')

    global __nvshmem_align
    __nvshmem_align = dlsym(RTLD_DEFAULT, 'nvshmem_align')
    if __nvshmem_align == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_align = dlsym(handle, 'nvshmem_align')

    global __nvshmem_free
    __nvshmem_free = dlsym(RTLD_DEFAULT, 'nvshmem_free')
    if __nvshmem_free == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_free = dlsym(handle, 'nvshmem_free')

    global __nvshmem_ptr
    __nvshmem_ptr = dlsym(RTLD_DEFAULT, 'nvshmem_ptr')
    if __nvshmem_ptr == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_ptr = dlsym(handle, 'nvshmem_ptr')

    global __nvshmemx_mc_ptr
    __nvshmemx_mc_ptr = dlsym(RTLD_DEFAULT, 'nvshmemx_mc_ptr')
    if __nvshmemx_mc_ptr == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_mc_ptr = dlsym(handle, 'nvshmemx_mc_ptr')

    global __nvshmem_team_my_pe
    __nvshmem_team_my_pe = dlsym(RTLD_DEFAULT, 'nvshmem_team_my_pe')
    if __nvshmem_team_my_pe == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_team_my_pe = dlsym(handle, 'nvshmem_team_my_pe')

    global __nvshmem_team_n_pes
    __nvshmem_team_n_pes = dlsym(RTLD_DEFAULT, 'nvshmem_team_n_pes')
    if __nvshmem_team_n_pes == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_team_n_pes = dlsym(handle, 'nvshmem_team_n_pes')

    global __nvshmem_barrier
    __nvshmem_barrier = dlsym(RTLD_DEFAULT, 'nvshmem_barrier')
    if __nvshmem_barrier == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_barrier = dlsym(handle, 'nvshmem_barrier')

    global __nvshmem_barrier_all
    __nvshmem_barrier_all = dlsym(RTLD_DEFAULT, 'nvshmem_barrier_all')
    if __nvshmem_barrier_all == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmem_barrier_all = dlsym(handle, 'nvshmem_barrier_all')

    global __nvshmemx_bfloat16_alltoall_on_stream
    __nvshmemx_bfloat16_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_alltoall_on_stream')
    if __nvshmemx_bfloat16_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_alltoall_on_stream = dlsym(handle, 'nvshmemx_bfloat16_alltoall_on_stream')

    global __nvshmemx_half_alltoall_on_stream
    __nvshmemx_half_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_alltoall_on_stream')
    if __nvshmemx_half_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_alltoall_on_stream = dlsym(handle, 'nvshmemx_half_alltoall_on_stream')

    global __nvshmemx_float_alltoall_on_stream
    __nvshmemx_float_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_alltoall_on_stream')
    if __nvshmemx_float_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_alltoall_on_stream = dlsym(handle, 'nvshmemx_float_alltoall_on_stream')

    global __nvshmemx_double_alltoall_on_stream
    __nvshmemx_double_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_alltoall_on_stream')
    if __nvshmemx_double_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_alltoall_on_stream = dlsym(handle, 'nvshmemx_double_alltoall_on_stream')

    global __nvshmemx_char_alltoall_on_stream
    __nvshmemx_char_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_alltoall_on_stream')
    if __nvshmemx_char_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_alltoall_on_stream = dlsym(handle, 'nvshmemx_char_alltoall_on_stream')

    global __nvshmemx_short_alltoall_on_stream
    __nvshmemx_short_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_alltoall_on_stream')
    if __nvshmemx_short_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_alltoall_on_stream = dlsym(handle, 'nvshmemx_short_alltoall_on_stream')

    global __nvshmemx_schar_alltoall_on_stream
    __nvshmemx_schar_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_alltoall_on_stream')
    if __nvshmemx_schar_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_alltoall_on_stream = dlsym(handle, 'nvshmemx_schar_alltoall_on_stream')

    global __nvshmemx_int_alltoall_on_stream
    __nvshmemx_int_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_alltoall_on_stream')
    if __nvshmemx_int_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_alltoall_on_stream = dlsym(handle, 'nvshmemx_int_alltoall_on_stream')

    global __nvshmemx_long_alltoall_on_stream
    __nvshmemx_long_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_alltoall_on_stream')
    if __nvshmemx_long_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_alltoall_on_stream = dlsym(handle, 'nvshmemx_long_alltoall_on_stream')

    global __nvshmemx_longlong_alltoall_on_stream
    __nvshmemx_longlong_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_alltoall_on_stream')
    if __nvshmemx_longlong_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_alltoall_on_stream = dlsym(handle, 'nvshmemx_longlong_alltoall_on_stream')

    global __nvshmemx_int8_alltoall_on_stream
    __nvshmemx_int8_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_alltoall_on_stream')
    if __nvshmemx_int8_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_alltoall_on_stream = dlsym(handle, 'nvshmemx_int8_alltoall_on_stream')

    global __nvshmemx_int16_alltoall_on_stream
    __nvshmemx_int16_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_alltoall_on_stream')
    if __nvshmemx_int16_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_alltoall_on_stream = dlsym(handle, 'nvshmemx_int16_alltoall_on_stream')

    global __nvshmemx_int32_alltoall_on_stream
    __nvshmemx_int32_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_alltoall_on_stream')
    if __nvshmemx_int32_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_alltoall_on_stream = dlsym(handle, 'nvshmemx_int32_alltoall_on_stream')

    global __nvshmemx_int64_alltoall_on_stream
    __nvshmemx_int64_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_alltoall_on_stream')
    if __nvshmemx_int64_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_alltoall_on_stream = dlsym(handle, 'nvshmemx_int64_alltoall_on_stream')

    global __nvshmemx_uint8_alltoall_on_stream
    __nvshmemx_uint8_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_alltoall_on_stream')
    if __nvshmemx_uint8_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_alltoall_on_stream = dlsym(handle, 'nvshmemx_uint8_alltoall_on_stream')

    global __nvshmemx_uint16_alltoall_on_stream
    __nvshmemx_uint16_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_alltoall_on_stream')
    if __nvshmemx_uint16_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_alltoall_on_stream = dlsym(handle, 'nvshmemx_uint16_alltoall_on_stream')

    global __nvshmemx_uint32_alltoall_on_stream
    __nvshmemx_uint32_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_alltoall_on_stream')
    if __nvshmemx_uint32_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_alltoall_on_stream = dlsym(handle, 'nvshmemx_uint32_alltoall_on_stream')

    global __nvshmemx_uint64_alltoall_on_stream
    __nvshmemx_uint64_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_alltoall_on_stream')
    if __nvshmemx_uint64_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_alltoall_on_stream = dlsym(handle, 'nvshmemx_uint64_alltoall_on_stream')

    global __nvshmemx_size_alltoall_on_stream
    __nvshmemx_size_alltoall_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_alltoall_on_stream')
    if __nvshmemx_size_alltoall_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_alltoall_on_stream = dlsym(handle, 'nvshmemx_size_alltoall_on_stream')

    global __nvshmemx_barrier_on_stream
    __nvshmemx_barrier_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_barrier_on_stream')
    if __nvshmemx_barrier_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_barrier_on_stream = dlsym(handle, 'nvshmemx_barrier_on_stream')

    global __nvshmemx_team_sync_on_stream
    __nvshmemx_team_sync_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_team_sync_on_stream')
    if __nvshmemx_team_sync_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_team_sync_on_stream = dlsym(handle, 'nvshmemx_team_sync_on_stream')

    global __nvshmemx_bfloat16_broadcast_on_stream
    __nvshmemx_bfloat16_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_broadcast_on_stream')
    if __nvshmemx_bfloat16_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_broadcast_on_stream = dlsym(handle, 'nvshmemx_bfloat16_broadcast_on_stream')

    global __nvshmemx_half_broadcast_on_stream
    __nvshmemx_half_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_broadcast_on_stream')
    if __nvshmemx_half_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_broadcast_on_stream = dlsym(handle, 'nvshmemx_half_broadcast_on_stream')

    global __nvshmemx_float_broadcast_on_stream
    __nvshmemx_float_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_broadcast_on_stream')
    if __nvshmemx_float_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_broadcast_on_stream = dlsym(handle, 'nvshmemx_float_broadcast_on_stream')

    global __nvshmemx_double_broadcast_on_stream
    __nvshmemx_double_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_broadcast_on_stream')
    if __nvshmemx_double_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_broadcast_on_stream = dlsym(handle, 'nvshmemx_double_broadcast_on_stream')

    global __nvshmemx_char_broadcast_on_stream
    __nvshmemx_char_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_broadcast_on_stream')
    if __nvshmemx_char_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_broadcast_on_stream = dlsym(handle, 'nvshmemx_char_broadcast_on_stream')

    global __nvshmemx_short_broadcast_on_stream
    __nvshmemx_short_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_broadcast_on_stream')
    if __nvshmemx_short_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_broadcast_on_stream = dlsym(handle, 'nvshmemx_short_broadcast_on_stream')

    global __nvshmemx_schar_broadcast_on_stream
    __nvshmemx_schar_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_broadcast_on_stream')
    if __nvshmemx_schar_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_broadcast_on_stream = dlsym(handle, 'nvshmemx_schar_broadcast_on_stream')

    global __nvshmemx_int_broadcast_on_stream
    __nvshmemx_int_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_broadcast_on_stream')
    if __nvshmemx_int_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_broadcast_on_stream = dlsym(handle, 'nvshmemx_int_broadcast_on_stream')

    global __nvshmemx_long_broadcast_on_stream
    __nvshmemx_long_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_broadcast_on_stream')
    if __nvshmemx_long_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_broadcast_on_stream = dlsym(handle, 'nvshmemx_long_broadcast_on_stream')

    global __nvshmemx_longlong_broadcast_on_stream
    __nvshmemx_longlong_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_broadcast_on_stream')
    if __nvshmemx_longlong_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_broadcast_on_stream = dlsym(handle, 'nvshmemx_longlong_broadcast_on_stream')

    global __nvshmemx_int8_broadcast_on_stream
    __nvshmemx_int8_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_broadcast_on_stream')
    if __nvshmemx_int8_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_broadcast_on_stream = dlsym(handle, 'nvshmemx_int8_broadcast_on_stream')

    global __nvshmemx_int16_broadcast_on_stream
    __nvshmemx_int16_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_broadcast_on_stream')
    if __nvshmemx_int16_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_broadcast_on_stream = dlsym(handle, 'nvshmemx_int16_broadcast_on_stream')

    global __nvshmemx_int32_broadcast_on_stream
    __nvshmemx_int32_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_broadcast_on_stream')
    if __nvshmemx_int32_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_broadcast_on_stream = dlsym(handle, 'nvshmemx_int32_broadcast_on_stream')

    global __nvshmemx_int64_broadcast_on_stream
    __nvshmemx_int64_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_broadcast_on_stream')
    if __nvshmemx_int64_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_broadcast_on_stream = dlsym(handle, 'nvshmemx_int64_broadcast_on_stream')

    global __nvshmemx_uint8_broadcast_on_stream
    __nvshmemx_uint8_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_broadcast_on_stream')
    if __nvshmemx_uint8_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_broadcast_on_stream = dlsym(handle, 'nvshmemx_uint8_broadcast_on_stream')

    global __nvshmemx_uint16_broadcast_on_stream
    __nvshmemx_uint16_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_broadcast_on_stream')
    if __nvshmemx_uint16_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_broadcast_on_stream = dlsym(handle, 'nvshmemx_uint16_broadcast_on_stream')

    global __nvshmemx_uint32_broadcast_on_stream
    __nvshmemx_uint32_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_broadcast_on_stream')
    if __nvshmemx_uint32_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_broadcast_on_stream = dlsym(handle, 'nvshmemx_uint32_broadcast_on_stream')

    global __nvshmemx_uint64_broadcast_on_stream
    __nvshmemx_uint64_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_broadcast_on_stream')
    if __nvshmemx_uint64_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_broadcast_on_stream = dlsym(handle, 'nvshmemx_uint64_broadcast_on_stream')

    global __nvshmemx_size_broadcast_on_stream
    __nvshmemx_size_broadcast_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_broadcast_on_stream')
    if __nvshmemx_size_broadcast_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_broadcast_on_stream = dlsym(handle, 'nvshmemx_size_broadcast_on_stream')

    global __nvshmemx_bfloat16_fcollect_on_stream
    __nvshmemx_bfloat16_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_fcollect_on_stream')
    if __nvshmemx_bfloat16_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_fcollect_on_stream = dlsym(handle, 'nvshmemx_bfloat16_fcollect_on_stream')

    global __nvshmemx_half_fcollect_on_stream
    __nvshmemx_half_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_fcollect_on_stream')
    if __nvshmemx_half_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_fcollect_on_stream = dlsym(handle, 'nvshmemx_half_fcollect_on_stream')

    global __nvshmemx_float_fcollect_on_stream
    __nvshmemx_float_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_fcollect_on_stream')
    if __nvshmemx_float_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_fcollect_on_stream = dlsym(handle, 'nvshmemx_float_fcollect_on_stream')

    global __nvshmemx_double_fcollect_on_stream
    __nvshmemx_double_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_fcollect_on_stream')
    if __nvshmemx_double_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_fcollect_on_stream = dlsym(handle, 'nvshmemx_double_fcollect_on_stream')

    global __nvshmemx_char_fcollect_on_stream
    __nvshmemx_char_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_fcollect_on_stream')
    if __nvshmemx_char_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_fcollect_on_stream = dlsym(handle, 'nvshmemx_char_fcollect_on_stream')

    global __nvshmemx_short_fcollect_on_stream
    __nvshmemx_short_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_fcollect_on_stream')
    if __nvshmemx_short_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_fcollect_on_stream = dlsym(handle, 'nvshmemx_short_fcollect_on_stream')

    global __nvshmemx_schar_fcollect_on_stream
    __nvshmemx_schar_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_fcollect_on_stream')
    if __nvshmemx_schar_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_fcollect_on_stream = dlsym(handle, 'nvshmemx_schar_fcollect_on_stream')

    global __nvshmemx_int_fcollect_on_stream
    __nvshmemx_int_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_fcollect_on_stream')
    if __nvshmemx_int_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_fcollect_on_stream = dlsym(handle, 'nvshmemx_int_fcollect_on_stream')

    global __nvshmemx_long_fcollect_on_stream
    __nvshmemx_long_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_fcollect_on_stream')
    if __nvshmemx_long_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_fcollect_on_stream = dlsym(handle, 'nvshmemx_long_fcollect_on_stream')

    global __nvshmemx_longlong_fcollect_on_stream
    __nvshmemx_longlong_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_fcollect_on_stream')
    if __nvshmemx_longlong_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_fcollect_on_stream = dlsym(handle, 'nvshmemx_longlong_fcollect_on_stream')

    global __nvshmemx_int8_fcollect_on_stream
    __nvshmemx_int8_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_fcollect_on_stream')
    if __nvshmemx_int8_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_fcollect_on_stream = dlsym(handle, 'nvshmemx_int8_fcollect_on_stream')

    global __nvshmemx_int16_fcollect_on_stream
    __nvshmemx_int16_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_fcollect_on_stream')
    if __nvshmemx_int16_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_fcollect_on_stream = dlsym(handle, 'nvshmemx_int16_fcollect_on_stream')

    global __nvshmemx_int32_fcollect_on_stream
    __nvshmemx_int32_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_fcollect_on_stream')
    if __nvshmemx_int32_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_fcollect_on_stream = dlsym(handle, 'nvshmemx_int32_fcollect_on_stream')

    global __nvshmemx_int64_fcollect_on_stream
    __nvshmemx_int64_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_fcollect_on_stream')
    if __nvshmemx_int64_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_fcollect_on_stream = dlsym(handle, 'nvshmemx_int64_fcollect_on_stream')

    global __nvshmemx_uint8_fcollect_on_stream
    __nvshmemx_uint8_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_fcollect_on_stream')
    if __nvshmemx_uint8_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_fcollect_on_stream = dlsym(handle, 'nvshmemx_uint8_fcollect_on_stream')

    global __nvshmemx_uint16_fcollect_on_stream
    __nvshmemx_uint16_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_fcollect_on_stream')
    if __nvshmemx_uint16_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_fcollect_on_stream = dlsym(handle, 'nvshmemx_uint16_fcollect_on_stream')

    global __nvshmemx_uint32_fcollect_on_stream
    __nvshmemx_uint32_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_fcollect_on_stream')
    if __nvshmemx_uint32_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_fcollect_on_stream = dlsym(handle, 'nvshmemx_uint32_fcollect_on_stream')

    global __nvshmemx_uint64_fcollect_on_stream
    __nvshmemx_uint64_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_fcollect_on_stream')
    if __nvshmemx_uint64_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_fcollect_on_stream = dlsym(handle, 'nvshmemx_uint64_fcollect_on_stream')

    global __nvshmemx_size_fcollect_on_stream
    __nvshmemx_size_fcollect_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_fcollect_on_stream')
    if __nvshmemx_size_fcollect_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_fcollect_on_stream = dlsym(handle, 'nvshmemx_size_fcollect_on_stream')

    global __nvshmemx_int8_max_reduce_on_stream
    __nvshmemx_int8_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_max_reduce_on_stream')
    if __nvshmemx_int8_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_max_reduce_on_stream = dlsym(handle, 'nvshmemx_int8_max_reduce_on_stream')

    global __nvshmemx_int16_max_reduce_on_stream
    __nvshmemx_int16_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_max_reduce_on_stream')
    if __nvshmemx_int16_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_max_reduce_on_stream = dlsym(handle, 'nvshmemx_int16_max_reduce_on_stream')

    global __nvshmemx_int32_max_reduce_on_stream
    __nvshmemx_int32_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_max_reduce_on_stream')
    if __nvshmemx_int32_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_max_reduce_on_stream = dlsym(handle, 'nvshmemx_int32_max_reduce_on_stream')

    global __nvshmemx_int64_max_reduce_on_stream
    __nvshmemx_int64_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_max_reduce_on_stream')
    if __nvshmemx_int64_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_max_reduce_on_stream = dlsym(handle, 'nvshmemx_int64_max_reduce_on_stream')

    global __nvshmemx_uint8_max_reduce_on_stream
    __nvshmemx_uint8_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_max_reduce_on_stream')
    if __nvshmemx_uint8_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_max_reduce_on_stream = dlsym(handle, 'nvshmemx_uint8_max_reduce_on_stream')

    global __nvshmemx_uint16_max_reduce_on_stream
    __nvshmemx_uint16_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_max_reduce_on_stream')
    if __nvshmemx_uint16_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_max_reduce_on_stream = dlsym(handle, 'nvshmemx_uint16_max_reduce_on_stream')

    global __nvshmemx_uint32_max_reduce_on_stream
    __nvshmemx_uint32_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_max_reduce_on_stream')
    if __nvshmemx_uint32_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_max_reduce_on_stream = dlsym(handle, 'nvshmemx_uint32_max_reduce_on_stream')

    global __nvshmemx_uint64_max_reduce_on_stream
    __nvshmemx_uint64_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_max_reduce_on_stream')
    if __nvshmemx_uint64_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_max_reduce_on_stream = dlsym(handle, 'nvshmemx_uint64_max_reduce_on_stream')

    global __nvshmemx_size_max_reduce_on_stream
    __nvshmemx_size_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_max_reduce_on_stream')
    if __nvshmemx_size_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_max_reduce_on_stream = dlsym(handle, 'nvshmemx_size_max_reduce_on_stream')

    global __nvshmemx_char_max_reduce_on_stream
    __nvshmemx_char_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_max_reduce_on_stream')
    if __nvshmemx_char_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_max_reduce_on_stream = dlsym(handle, 'nvshmemx_char_max_reduce_on_stream')

    global __nvshmemx_schar_max_reduce_on_stream
    __nvshmemx_schar_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_max_reduce_on_stream')
    if __nvshmemx_schar_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_max_reduce_on_stream = dlsym(handle, 'nvshmemx_schar_max_reduce_on_stream')

    global __nvshmemx_short_max_reduce_on_stream
    __nvshmemx_short_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_max_reduce_on_stream')
    if __nvshmemx_short_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_max_reduce_on_stream = dlsym(handle, 'nvshmemx_short_max_reduce_on_stream')

    global __nvshmemx_int_max_reduce_on_stream
    __nvshmemx_int_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_max_reduce_on_stream')
    if __nvshmemx_int_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_max_reduce_on_stream = dlsym(handle, 'nvshmemx_int_max_reduce_on_stream')

    global __nvshmemx_long_max_reduce_on_stream
    __nvshmemx_long_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_max_reduce_on_stream')
    if __nvshmemx_long_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_max_reduce_on_stream = dlsym(handle, 'nvshmemx_long_max_reduce_on_stream')

    global __nvshmemx_longlong_max_reduce_on_stream
    __nvshmemx_longlong_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_max_reduce_on_stream')
    if __nvshmemx_longlong_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_max_reduce_on_stream = dlsym(handle, 'nvshmemx_longlong_max_reduce_on_stream')

    global __nvshmemx_bfloat16_max_reduce_on_stream
    __nvshmemx_bfloat16_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_max_reduce_on_stream')
    if __nvshmemx_bfloat16_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_max_reduce_on_stream = dlsym(handle, 'nvshmemx_bfloat16_max_reduce_on_stream')

    global __nvshmemx_half_max_reduce_on_stream
    __nvshmemx_half_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_max_reduce_on_stream')
    if __nvshmemx_half_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_max_reduce_on_stream = dlsym(handle, 'nvshmemx_half_max_reduce_on_stream')

    global __nvshmemx_float_max_reduce_on_stream
    __nvshmemx_float_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_max_reduce_on_stream')
    if __nvshmemx_float_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_max_reduce_on_stream = dlsym(handle, 'nvshmemx_float_max_reduce_on_stream')

    global __nvshmemx_double_max_reduce_on_stream
    __nvshmemx_double_max_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_max_reduce_on_stream')
    if __nvshmemx_double_max_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_max_reduce_on_stream = dlsym(handle, 'nvshmemx_double_max_reduce_on_stream')

    global __nvshmemx_int8_min_reduce_on_stream
    __nvshmemx_int8_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_min_reduce_on_stream')
    if __nvshmemx_int8_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_min_reduce_on_stream = dlsym(handle, 'nvshmemx_int8_min_reduce_on_stream')

    global __nvshmemx_int16_min_reduce_on_stream
    __nvshmemx_int16_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_min_reduce_on_stream')
    if __nvshmemx_int16_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_min_reduce_on_stream = dlsym(handle, 'nvshmemx_int16_min_reduce_on_stream')

    global __nvshmemx_int32_min_reduce_on_stream
    __nvshmemx_int32_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_min_reduce_on_stream')
    if __nvshmemx_int32_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_min_reduce_on_stream = dlsym(handle, 'nvshmemx_int32_min_reduce_on_stream')

    global __nvshmemx_int64_min_reduce_on_stream
    __nvshmemx_int64_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_min_reduce_on_stream')
    if __nvshmemx_int64_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_min_reduce_on_stream = dlsym(handle, 'nvshmemx_int64_min_reduce_on_stream')

    global __nvshmemx_uint8_min_reduce_on_stream
    __nvshmemx_uint8_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_min_reduce_on_stream')
    if __nvshmemx_uint8_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_min_reduce_on_stream = dlsym(handle, 'nvshmemx_uint8_min_reduce_on_stream')

    global __nvshmemx_uint16_min_reduce_on_stream
    __nvshmemx_uint16_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_min_reduce_on_stream')
    if __nvshmemx_uint16_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_min_reduce_on_stream = dlsym(handle, 'nvshmemx_uint16_min_reduce_on_stream')

    global __nvshmemx_uint32_min_reduce_on_stream
    __nvshmemx_uint32_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_min_reduce_on_stream')
    if __nvshmemx_uint32_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_min_reduce_on_stream = dlsym(handle, 'nvshmemx_uint32_min_reduce_on_stream')

    global __nvshmemx_uint64_min_reduce_on_stream
    __nvshmemx_uint64_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_min_reduce_on_stream')
    if __nvshmemx_uint64_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_min_reduce_on_stream = dlsym(handle, 'nvshmemx_uint64_min_reduce_on_stream')

    global __nvshmemx_size_min_reduce_on_stream
    __nvshmemx_size_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_min_reduce_on_stream')
    if __nvshmemx_size_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_min_reduce_on_stream = dlsym(handle, 'nvshmemx_size_min_reduce_on_stream')

    global __nvshmemx_char_min_reduce_on_stream
    __nvshmemx_char_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_min_reduce_on_stream')
    if __nvshmemx_char_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_min_reduce_on_stream = dlsym(handle, 'nvshmemx_char_min_reduce_on_stream')

    global __nvshmemx_schar_min_reduce_on_stream
    __nvshmemx_schar_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_min_reduce_on_stream')
    if __nvshmemx_schar_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_min_reduce_on_stream = dlsym(handle, 'nvshmemx_schar_min_reduce_on_stream')

    global __nvshmemx_short_min_reduce_on_stream
    __nvshmemx_short_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_min_reduce_on_stream')
    if __nvshmemx_short_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_min_reduce_on_stream = dlsym(handle, 'nvshmemx_short_min_reduce_on_stream')

    global __nvshmemx_int_min_reduce_on_stream
    __nvshmemx_int_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_min_reduce_on_stream')
    if __nvshmemx_int_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_min_reduce_on_stream = dlsym(handle, 'nvshmemx_int_min_reduce_on_stream')

    global __nvshmemx_long_min_reduce_on_stream
    __nvshmemx_long_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_min_reduce_on_stream')
    if __nvshmemx_long_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_min_reduce_on_stream = dlsym(handle, 'nvshmemx_long_min_reduce_on_stream')

    global __nvshmemx_longlong_min_reduce_on_stream
    __nvshmemx_longlong_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_min_reduce_on_stream')
    if __nvshmemx_longlong_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_min_reduce_on_stream = dlsym(handle, 'nvshmemx_longlong_min_reduce_on_stream')

    global __nvshmemx_bfloat16_min_reduce_on_stream
    __nvshmemx_bfloat16_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_min_reduce_on_stream')
    if __nvshmemx_bfloat16_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_min_reduce_on_stream = dlsym(handle, 'nvshmemx_bfloat16_min_reduce_on_stream')

    global __nvshmemx_half_min_reduce_on_stream
    __nvshmemx_half_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_min_reduce_on_stream')
    if __nvshmemx_half_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_min_reduce_on_stream = dlsym(handle, 'nvshmemx_half_min_reduce_on_stream')

    global __nvshmemx_float_min_reduce_on_stream
    __nvshmemx_float_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_min_reduce_on_stream')
    if __nvshmemx_float_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_min_reduce_on_stream = dlsym(handle, 'nvshmemx_float_min_reduce_on_stream')

    global __nvshmemx_double_min_reduce_on_stream
    __nvshmemx_double_min_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_min_reduce_on_stream')
    if __nvshmemx_double_min_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_min_reduce_on_stream = dlsym(handle, 'nvshmemx_double_min_reduce_on_stream')

    global __nvshmemx_int8_sum_reduce_on_stream
    __nvshmemx_int8_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_sum_reduce_on_stream')
    if __nvshmemx_int8_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_int8_sum_reduce_on_stream')

    global __nvshmemx_int16_sum_reduce_on_stream
    __nvshmemx_int16_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_sum_reduce_on_stream')
    if __nvshmemx_int16_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_int16_sum_reduce_on_stream')

    global __nvshmemx_int32_sum_reduce_on_stream
    __nvshmemx_int32_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_sum_reduce_on_stream')
    if __nvshmemx_int32_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_int32_sum_reduce_on_stream')

    global __nvshmemx_int64_sum_reduce_on_stream
    __nvshmemx_int64_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_sum_reduce_on_stream')
    if __nvshmemx_int64_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_int64_sum_reduce_on_stream')

    global __nvshmemx_uint8_sum_reduce_on_stream
    __nvshmemx_uint8_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_sum_reduce_on_stream')
    if __nvshmemx_uint8_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_uint8_sum_reduce_on_stream')

    global __nvshmemx_uint16_sum_reduce_on_stream
    __nvshmemx_uint16_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_sum_reduce_on_stream')
    if __nvshmemx_uint16_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_uint16_sum_reduce_on_stream')

    global __nvshmemx_uint32_sum_reduce_on_stream
    __nvshmemx_uint32_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_sum_reduce_on_stream')
    if __nvshmemx_uint32_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_uint32_sum_reduce_on_stream')

    global __nvshmemx_uint64_sum_reduce_on_stream
    __nvshmemx_uint64_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_sum_reduce_on_stream')
    if __nvshmemx_uint64_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_uint64_sum_reduce_on_stream')

    global __nvshmemx_size_sum_reduce_on_stream
    __nvshmemx_size_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_sum_reduce_on_stream')
    if __nvshmemx_size_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_size_sum_reduce_on_stream')

    global __nvshmemx_char_sum_reduce_on_stream
    __nvshmemx_char_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_sum_reduce_on_stream')
    if __nvshmemx_char_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_char_sum_reduce_on_stream')

    global __nvshmemx_schar_sum_reduce_on_stream
    __nvshmemx_schar_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_sum_reduce_on_stream')
    if __nvshmemx_schar_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_schar_sum_reduce_on_stream')

    global __nvshmemx_short_sum_reduce_on_stream
    __nvshmemx_short_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_sum_reduce_on_stream')
    if __nvshmemx_short_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_short_sum_reduce_on_stream')

    global __nvshmemx_int_sum_reduce_on_stream
    __nvshmemx_int_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_sum_reduce_on_stream')
    if __nvshmemx_int_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_int_sum_reduce_on_stream')

    global __nvshmemx_long_sum_reduce_on_stream
    __nvshmemx_long_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_sum_reduce_on_stream')
    if __nvshmemx_long_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_long_sum_reduce_on_stream')

    global __nvshmemx_longlong_sum_reduce_on_stream
    __nvshmemx_longlong_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_sum_reduce_on_stream')
    if __nvshmemx_longlong_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_longlong_sum_reduce_on_stream')

    global __nvshmemx_bfloat16_sum_reduce_on_stream
    __nvshmemx_bfloat16_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_sum_reduce_on_stream')
    if __nvshmemx_bfloat16_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_bfloat16_sum_reduce_on_stream')

    global __nvshmemx_half_sum_reduce_on_stream
    __nvshmemx_half_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_sum_reduce_on_stream')
    if __nvshmemx_half_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_half_sum_reduce_on_stream')

    global __nvshmemx_float_sum_reduce_on_stream
    __nvshmemx_float_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_sum_reduce_on_stream')
    if __nvshmemx_float_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_float_sum_reduce_on_stream')

    global __nvshmemx_double_sum_reduce_on_stream
    __nvshmemx_double_sum_reduce_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_sum_reduce_on_stream')
    if __nvshmemx_double_sum_reduce_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_sum_reduce_on_stream = dlsym(handle, 'nvshmemx_double_sum_reduce_on_stream')

    global __nvshmemx_int8_max_reducescatter_on_stream
    __nvshmemx_int8_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_max_reducescatter_on_stream')
    if __nvshmemx_int8_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int8_max_reducescatter_on_stream')

    global __nvshmemx_int16_max_reducescatter_on_stream
    __nvshmemx_int16_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_max_reducescatter_on_stream')
    if __nvshmemx_int16_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int16_max_reducescatter_on_stream')

    global __nvshmemx_int32_max_reducescatter_on_stream
    __nvshmemx_int32_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_max_reducescatter_on_stream')
    if __nvshmemx_int32_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int32_max_reducescatter_on_stream')

    global __nvshmemx_int64_max_reducescatter_on_stream
    __nvshmemx_int64_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_max_reducescatter_on_stream')
    if __nvshmemx_int64_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int64_max_reducescatter_on_stream')

    global __nvshmemx_uint8_max_reducescatter_on_stream
    __nvshmemx_uint8_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_max_reducescatter_on_stream')
    if __nvshmemx_uint8_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint8_max_reducescatter_on_stream')

    global __nvshmemx_uint16_max_reducescatter_on_stream
    __nvshmemx_uint16_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_max_reducescatter_on_stream')
    if __nvshmemx_uint16_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint16_max_reducescatter_on_stream')

    global __nvshmemx_uint32_max_reducescatter_on_stream
    __nvshmemx_uint32_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_max_reducescatter_on_stream')
    if __nvshmemx_uint32_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint32_max_reducescatter_on_stream')

    global __nvshmemx_uint64_max_reducescatter_on_stream
    __nvshmemx_uint64_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_max_reducescatter_on_stream')
    if __nvshmemx_uint64_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint64_max_reducescatter_on_stream')

    global __nvshmemx_size_max_reducescatter_on_stream
    __nvshmemx_size_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_max_reducescatter_on_stream')
    if __nvshmemx_size_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_size_max_reducescatter_on_stream')

    global __nvshmemx_char_max_reducescatter_on_stream
    __nvshmemx_char_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_max_reducescatter_on_stream')
    if __nvshmemx_char_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_char_max_reducescatter_on_stream')

    global __nvshmemx_schar_max_reducescatter_on_stream
    __nvshmemx_schar_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_max_reducescatter_on_stream')
    if __nvshmemx_schar_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_schar_max_reducescatter_on_stream')

    global __nvshmemx_short_max_reducescatter_on_stream
    __nvshmemx_short_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_max_reducescatter_on_stream')
    if __nvshmemx_short_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_short_max_reducescatter_on_stream')

    global __nvshmemx_int_max_reducescatter_on_stream
    __nvshmemx_int_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_max_reducescatter_on_stream')
    if __nvshmemx_int_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int_max_reducescatter_on_stream')

    global __nvshmemx_long_max_reducescatter_on_stream
    __nvshmemx_long_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_max_reducescatter_on_stream')
    if __nvshmemx_long_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_long_max_reducescatter_on_stream')

    global __nvshmemx_longlong_max_reducescatter_on_stream
    __nvshmemx_longlong_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_max_reducescatter_on_stream')
    if __nvshmemx_longlong_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_longlong_max_reducescatter_on_stream')

    global __nvshmemx_bfloat16_max_reducescatter_on_stream
    __nvshmemx_bfloat16_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_max_reducescatter_on_stream')
    if __nvshmemx_bfloat16_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_bfloat16_max_reducescatter_on_stream')

    global __nvshmemx_half_max_reducescatter_on_stream
    __nvshmemx_half_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_max_reducescatter_on_stream')
    if __nvshmemx_half_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_half_max_reducescatter_on_stream')

    global __nvshmemx_float_max_reducescatter_on_stream
    __nvshmemx_float_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_max_reducescatter_on_stream')
    if __nvshmemx_float_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_float_max_reducescatter_on_stream')

    global __nvshmemx_double_max_reducescatter_on_stream
    __nvshmemx_double_max_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_max_reducescatter_on_stream')
    if __nvshmemx_double_max_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_max_reducescatter_on_stream = dlsym(handle, 'nvshmemx_double_max_reducescatter_on_stream')

    global __nvshmemx_int8_min_reducescatter_on_stream
    __nvshmemx_int8_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_min_reducescatter_on_stream')
    if __nvshmemx_int8_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int8_min_reducescatter_on_stream')

    global __nvshmemx_int16_min_reducescatter_on_stream
    __nvshmemx_int16_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_min_reducescatter_on_stream')
    if __nvshmemx_int16_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int16_min_reducescatter_on_stream')

    global __nvshmemx_int32_min_reducescatter_on_stream
    __nvshmemx_int32_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_min_reducescatter_on_stream')
    if __nvshmemx_int32_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int32_min_reducescatter_on_stream')

    global __nvshmemx_int64_min_reducescatter_on_stream
    __nvshmemx_int64_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_min_reducescatter_on_stream')
    if __nvshmemx_int64_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int64_min_reducescatter_on_stream')

    global __nvshmemx_uint8_min_reducescatter_on_stream
    __nvshmemx_uint8_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_min_reducescatter_on_stream')
    if __nvshmemx_uint8_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint8_min_reducescatter_on_stream')

    global __nvshmemx_uint16_min_reducescatter_on_stream
    __nvshmemx_uint16_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_min_reducescatter_on_stream')
    if __nvshmemx_uint16_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint16_min_reducescatter_on_stream')

    global __nvshmemx_uint32_min_reducescatter_on_stream
    __nvshmemx_uint32_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_min_reducescatter_on_stream')
    if __nvshmemx_uint32_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint32_min_reducescatter_on_stream')

    global __nvshmemx_uint64_min_reducescatter_on_stream
    __nvshmemx_uint64_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_min_reducescatter_on_stream')
    if __nvshmemx_uint64_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint64_min_reducescatter_on_stream')

    global __nvshmemx_size_min_reducescatter_on_stream
    __nvshmemx_size_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_min_reducescatter_on_stream')
    if __nvshmemx_size_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_size_min_reducescatter_on_stream')

    global __nvshmemx_char_min_reducescatter_on_stream
    __nvshmemx_char_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_min_reducescatter_on_stream')
    if __nvshmemx_char_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_char_min_reducescatter_on_stream')

    global __nvshmemx_schar_min_reducescatter_on_stream
    __nvshmemx_schar_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_min_reducescatter_on_stream')
    if __nvshmemx_schar_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_schar_min_reducescatter_on_stream')

    global __nvshmemx_short_min_reducescatter_on_stream
    __nvshmemx_short_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_min_reducescatter_on_stream')
    if __nvshmemx_short_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_short_min_reducescatter_on_stream')

    global __nvshmemx_int_min_reducescatter_on_stream
    __nvshmemx_int_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_min_reducescatter_on_stream')
    if __nvshmemx_int_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int_min_reducescatter_on_stream')

    global __nvshmemx_long_min_reducescatter_on_stream
    __nvshmemx_long_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_min_reducescatter_on_stream')
    if __nvshmemx_long_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_long_min_reducescatter_on_stream')

    global __nvshmemx_longlong_min_reducescatter_on_stream
    __nvshmemx_longlong_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_min_reducescatter_on_stream')
    if __nvshmemx_longlong_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_longlong_min_reducescatter_on_stream')

    global __nvshmemx_bfloat16_min_reducescatter_on_stream
    __nvshmemx_bfloat16_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_min_reducescatter_on_stream')
    if __nvshmemx_bfloat16_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_bfloat16_min_reducescatter_on_stream')

    global __nvshmemx_half_min_reducescatter_on_stream
    __nvshmemx_half_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_min_reducescatter_on_stream')
    if __nvshmemx_half_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_half_min_reducescatter_on_stream')

    global __nvshmemx_float_min_reducescatter_on_stream
    __nvshmemx_float_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_min_reducescatter_on_stream')
    if __nvshmemx_float_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_float_min_reducescatter_on_stream')

    global __nvshmemx_double_min_reducescatter_on_stream
    __nvshmemx_double_min_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_min_reducescatter_on_stream')
    if __nvshmemx_double_min_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_min_reducescatter_on_stream = dlsym(handle, 'nvshmemx_double_min_reducescatter_on_stream')

    global __nvshmemx_int8_sum_reducescatter_on_stream
    __nvshmemx_int8_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int8_sum_reducescatter_on_stream')
    if __nvshmemx_int8_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int8_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int8_sum_reducescatter_on_stream')

    global __nvshmemx_int16_sum_reducescatter_on_stream
    __nvshmemx_int16_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int16_sum_reducescatter_on_stream')
    if __nvshmemx_int16_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int16_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int16_sum_reducescatter_on_stream')

    global __nvshmemx_int32_sum_reducescatter_on_stream
    __nvshmemx_int32_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int32_sum_reducescatter_on_stream')
    if __nvshmemx_int32_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int32_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int32_sum_reducescatter_on_stream')

    global __nvshmemx_int64_sum_reducescatter_on_stream
    __nvshmemx_int64_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int64_sum_reducescatter_on_stream')
    if __nvshmemx_int64_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int64_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int64_sum_reducescatter_on_stream')

    global __nvshmemx_uint8_sum_reducescatter_on_stream
    __nvshmemx_uint8_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint8_sum_reducescatter_on_stream')
    if __nvshmemx_uint8_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint8_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint8_sum_reducescatter_on_stream')

    global __nvshmemx_uint16_sum_reducescatter_on_stream
    __nvshmemx_uint16_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint16_sum_reducescatter_on_stream')
    if __nvshmemx_uint16_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint16_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint16_sum_reducescatter_on_stream')

    global __nvshmemx_uint32_sum_reducescatter_on_stream
    __nvshmemx_uint32_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint32_sum_reducescatter_on_stream')
    if __nvshmemx_uint32_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint32_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint32_sum_reducescatter_on_stream')

    global __nvshmemx_uint64_sum_reducescatter_on_stream
    __nvshmemx_uint64_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_uint64_sum_reducescatter_on_stream')
    if __nvshmemx_uint64_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_uint64_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_uint64_sum_reducescatter_on_stream')

    global __nvshmemx_size_sum_reducescatter_on_stream
    __nvshmemx_size_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_size_sum_reducescatter_on_stream')
    if __nvshmemx_size_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_size_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_size_sum_reducescatter_on_stream')

    global __nvshmemx_char_sum_reducescatter_on_stream
    __nvshmemx_char_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_char_sum_reducescatter_on_stream')
    if __nvshmemx_char_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_char_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_char_sum_reducescatter_on_stream')

    global __nvshmemx_schar_sum_reducescatter_on_stream
    __nvshmemx_schar_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_schar_sum_reducescatter_on_stream')
    if __nvshmemx_schar_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_schar_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_schar_sum_reducescatter_on_stream')

    global __nvshmemx_short_sum_reducescatter_on_stream
    __nvshmemx_short_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_short_sum_reducescatter_on_stream')
    if __nvshmemx_short_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_short_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_short_sum_reducescatter_on_stream')

    global __nvshmemx_int_sum_reducescatter_on_stream
    __nvshmemx_int_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_int_sum_reducescatter_on_stream')
    if __nvshmemx_int_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_int_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_int_sum_reducescatter_on_stream')

    global __nvshmemx_long_sum_reducescatter_on_stream
    __nvshmemx_long_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_long_sum_reducescatter_on_stream')
    if __nvshmemx_long_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_long_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_long_sum_reducescatter_on_stream')

    global __nvshmemx_longlong_sum_reducescatter_on_stream
    __nvshmemx_longlong_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_longlong_sum_reducescatter_on_stream')
    if __nvshmemx_longlong_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_longlong_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_longlong_sum_reducescatter_on_stream')

    global __nvshmemx_bfloat16_sum_reducescatter_on_stream
    __nvshmemx_bfloat16_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_bfloat16_sum_reducescatter_on_stream')
    if __nvshmemx_bfloat16_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_bfloat16_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_bfloat16_sum_reducescatter_on_stream')

    global __nvshmemx_half_sum_reducescatter_on_stream
    __nvshmemx_half_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_half_sum_reducescatter_on_stream')
    if __nvshmemx_half_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_half_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_half_sum_reducescatter_on_stream')

    global __nvshmemx_float_sum_reducescatter_on_stream
    __nvshmemx_float_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_float_sum_reducescatter_on_stream')
    if __nvshmemx_float_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_float_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_float_sum_reducescatter_on_stream')

    global __nvshmemx_double_sum_reducescatter_on_stream
    __nvshmemx_double_sum_reducescatter_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_double_sum_reducescatter_on_stream')
    if __nvshmemx_double_sum_reducescatter_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_double_sum_reducescatter_on_stream = dlsym(handle, 'nvshmemx_double_sum_reducescatter_on_stream')

    global __nvshmemx_hostlib_init_attr
    __nvshmemx_hostlib_init_attr = dlsym(RTLD_DEFAULT, 'nvshmemx_hostlib_init_attr')
    if __nvshmemx_hostlib_init_attr == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_hostlib_init_attr = dlsym(handle, 'nvshmemx_hostlib_init_attr')

    global __nvshmemx_hostlib_finalize
    __nvshmemx_hostlib_finalize = dlsym(RTLD_DEFAULT, 'nvshmemx_hostlib_finalize')
    if __nvshmemx_hostlib_finalize == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_hostlib_finalize = dlsym(handle, 'nvshmemx_hostlib_finalize')

    global __nvshmemx_set_attr_uniqueid_args
    __nvshmemx_set_attr_uniqueid_args = dlsym(RTLD_DEFAULT, 'nvshmemx_set_attr_uniqueid_args')
    if __nvshmemx_set_attr_uniqueid_args == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_set_attr_uniqueid_args = dlsym(handle, 'nvshmemx_set_attr_uniqueid_args')

    global __nvshmemx_set_attr_mpi_comm_args
    __nvshmemx_set_attr_mpi_comm_args = dlsym(RTLD_DEFAULT, 'nvshmemx_set_attr_mpi_comm_args')
    if __nvshmemx_set_attr_mpi_comm_args == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_set_attr_mpi_comm_args = dlsym(handle, 'nvshmemx_set_attr_mpi_comm_args')

    global __nvshmemx_get_uniqueid
    __nvshmemx_get_uniqueid = dlsym(RTLD_DEFAULT, 'nvshmemx_get_uniqueid')
    if __nvshmemx_get_uniqueid == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_get_uniqueid = dlsym(handle, 'nvshmemx_get_uniqueid')

    global __nvshmemx_cumodule_init
    __nvshmemx_cumodule_init = dlsym(RTLD_DEFAULT, 'nvshmemx_cumodule_init')
    if __nvshmemx_cumodule_init == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_cumodule_init = dlsym(handle, 'nvshmemx_cumodule_init')

    global __nvshmemx_cumodule_finalize
    __nvshmemx_cumodule_finalize = dlsym(RTLD_DEFAULT, 'nvshmemx_cumodule_finalize')
    if __nvshmemx_cumodule_finalize == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_cumodule_finalize = dlsym(handle, 'nvshmemx_cumodule_finalize')

    global __nvshmemx_putmem_on_stream
    __nvshmemx_putmem_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_putmem_on_stream')
    if __nvshmemx_putmem_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_putmem_on_stream = dlsym(handle, 'nvshmemx_putmem_on_stream')

    global __nvshmemx_putmem_signal_on_stream
    __nvshmemx_putmem_signal_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_putmem_signal_on_stream')
    if __nvshmemx_putmem_signal_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_putmem_signal_on_stream = dlsym(handle, 'nvshmemx_putmem_signal_on_stream')

    global __nvshmemx_getmem_on_stream
    __nvshmemx_getmem_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_getmem_on_stream')
    if __nvshmemx_getmem_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_getmem_on_stream = dlsym(handle, 'nvshmemx_getmem_on_stream')

    global __nvshmemx_quiet_on_stream
    __nvshmemx_quiet_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_quiet_on_stream')
    if __nvshmemx_quiet_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_quiet_on_stream = dlsym(handle, 'nvshmemx_quiet_on_stream')

    global __nvshmemx_signal_wait_until_on_stream
    __nvshmemx_signal_wait_until_on_stream = dlsym(RTLD_DEFAULT, 'nvshmemx_signal_wait_until_on_stream')
    if __nvshmemx_signal_wait_until_on_stream == NULL:
        if handle == NULL:
            handle = load_library()
        __nvshmemx_signal_wait_until_on_stream = dlsym(handle, 'nvshmemx_signal_wait_until_on_stream')

    __py_nvshmem_init = True
    return 0


cdef dict func_ptrs = None


cpdef dict _inspect_function_pointers():
    global func_ptrs
    if func_ptrs is not None:
        return func_ptrs

    _check_or_init_nvshmem()
    cdef dict data = {}

    global __nvshmemx_init_status
    data["__nvshmemx_init_status"] = <intptr_t>__nvshmemx_init_status

    global __nvshmem_my_pe
    data["__nvshmem_my_pe"] = <intptr_t>__nvshmem_my_pe

    global __nvshmem_n_pes
    data["__nvshmem_n_pes"] = <intptr_t>__nvshmem_n_pes

    global __nvshmem_info_get_version
    data["__nvshmem_info_get_version"] = <intptr_t>__nvshmem_info_get_version

    global __nvshmemx_vendor_get_version_info
    data["__nvshmemx_vendor_get_version_info"] = <intptr_t>__nvshmemx_vendor_get_version_info

    global __nvshmem_malloc
    data["__nvshmem_malloc"] = <intptr_t>__nvshmem_malloc

    global __nvshmem_calloc
    data["__nvshmem_calloc"] = <intptr_t>__nvshmem_calloc

    global __nvshmem_align
    data["__nvshmem_align"] = <intptr_t>__nvshmem_align

    global __nvshmem_free
    data["__nvshmem_free"] = <intptr_t>__nvshmem_free

    global __nvshmem_ptr
    data["__nvshmem_ptr"] = <intptr_t>__nvshmem_ptr

    global __nvshmemx_mc_ptr
    data["__nvshmemx_mc_ptr"] = <intptr_t>__nvshmemx_mc_ptr

    global __nvshmem_team_my_pe
    data["__nvshmem_team_my_pe"] = <intptr_t>__nvshmem_team_my_pe

    global __nvshmem_team_n_pes
    data["__nvshmem_team_n_pes"] = <intptr_t>__nvshmem_team_n_pes

    global __nvshmem_barrier
    data["__nvshmem_barrier"] = <intptr_t>__nvshmem_barrier

    global __nvshmem_barrier_all
    data["__nvshmem_barrier_all"] = <intptr_t>__nvshmem_barrier_all

    global __nvshmemx_bfloat16_alltoall_on_stream
    data["__nvshmemx_bfloat16_alltoall_on_stream"] = <intptr_t>__nvshmemx_bfloat16_alltoall_on_stream

    global __nvshmemx_half_alltoall_on_stream
    data["__nvshmemx_half_alltoall_on_stream"] = <intptr_t>__nvshmemx_half_alltoall_on_stream

    global __nvshmemx_float_alltoall_on_stream
    data["__nvshmemx_float_alltoall_on_stream"] = <intptr_t>__nvshmemx_float_alltoall_on_stream

    global __nvshmemx_double_alltoall_on_stream
    data["__nvshmemx_double_alltoall_on_stream"] = <intptr_t>__nvshmemx_double_alltoall_on_stream

    global __nvshmemx_char_alltoall_on_stream
    data["__nvshmemx_char_alltoall_on_stream"] = <intptr_t>__nvshmemx_char_alltoall_on_stream

    global __nvshmemx_short_alltoall_on_stream
    data["__nvshmemx_short_alltoall_on_stream"] = <intptr_t>__nvshmemx_short_alltoall_on_stream

    global __nvshmemx_schar_alltoall_on_stream
    data["__nvshmemx_schar_alltoall_on_stream"] = <intptr_t>__nvshmemx_schar_alltoall_on_stream

    global __nvshmemx_int_alltoall_on_stream
    data["__nvshmemx_int_alltoall_on_stream"] = <intptr_t>__nvshmemx_int_alltoall_on_stream

    global __nvshmemx_long_alltoall_on_stream
    data["__nvshmemx_long_alltoall_on_stream"] = <intptr_t>__nvshmemx_long_alltoall_on_stream

    global __nvshmemx_longlong_alltoall_on_stream
    data["__nvshmemx_longlong_alltoall_on_stream"] = <intptr_t>__nvshmemx_longlong_alltoall_on_stream

    global __nvshmemx_int8_alltoall_on_stream
    data["__nvshmemx_int8_alltoall_on_stream"] = <intptr_t>__nvshmemx_int8_alltoall_on_stream

    global __nvshmemx_int16_alltoall_on_stream
    data["__nvshmemx_int16_alltoall_on_stream"] = <intptr_t>__nvshmemx_int16_alltoall_on_stream

    global __nvshmemx_int32_alltoall_on_stream
    data["__nvshmemx_int32_alltoall_on_stream"] = <intptr_t>__nvshmemx_int32_alltoall_on_stream

    global __nvshmemx_int64_alltoall_on_stream
    data["__nvshmemx_int64_alltoall_on_stream"] = <intptr_t>__nvshmemx_int64_alltoall_on_stream

    global __nvshmemx_uint8_alltoall_on_stream
    data["__nvshmemx_uint8_alltoall_on_stream"] = <intptr_t>__nvshmemx_uint8_alltoall_on_stream

    global __nvshmemx_uint16_alltoall_on_stream
    data["__nvshmemx_uint16_alltoall_on_stream"] = <intptr_t>__nvshmemx_uint16_alltoall_on_stream

    global __nvshmemx_uint32_alltoall_on_stream
    data["__nvshmemx_uint32_alltoall_on_stream"] = <intptr_t>__nvshmemx_uint32_alltoall_on_stream

    global __nvshmemx_uint64_alltoall_on_stream
    data["__nvshmemx_uint64_alltoall_on_stream"] = <intptr_t>__nvshmemx_uint64_alltoall_on_stream

    global __nvshmemx_size_alltoall_on_stream
    data["__nvshmemx_size_alltoall_on_stream"] = <intptr_t>__nvshmemx_size_alltoall_on_stream

    global __nvshmemx_barrier_on_stream
    data["__nvshmemx_barrier_on_stream"] = <intptr_t>__nvshmemx_barrier_on_stream

    global __nvshmemx_team_sync_on_stream
    data["__nvshmemx_team_sync_on_stream"] = <intptr_t>__nvshmemx_team_sync_on_stream

    global __nvshmemx_bfloat16_broadcast_on_stream
    data["__nvshmemx_bfloat16_broadcast_on_stream"] = <intptr_t>__nvshmemx_bfloat16_broadcast_on_stream

    global __nvshmemx_half_broadcast_on_stream
    data["__nvshmemx_half_broadcast_on_stream"] = <intptr_t>__nvshmemx_half_broadcast_on_stream

    global __nvshmemx_float_broadcast_on_stream
    data["__nvshmemx_float_broadcast_on_stream"] = <intptr_t>__nvshmemx_float_broadcast_on_stream

    global __nvshmemx_double_broadcast_on_stream
    data["__nvshmemx_double_broadcast_on_stream"] = <intptr_t>__nvshmemx_double_broadcast_on_stream

    global __nvshmemx_char_broadcast_on_stream
    data["__nvshmemx_char_broadcast_on_stream"] = <intptr_t>__nvshmemx_char_broadcast_on_stream

    global __nvshmemx_short_broadcast_on_stream
    data["__nvshmemx_short_broadcast_on_stream"] = <intptr_t>__nvshmemx_short_broadcast_on_stream

    global __nvshmemx_schar_broadcast_on_stream
    data["__nvshmemx_schar_broadcast_on_stream"] = <intptr_t>__nvshmemx_schar_broadcast_on_stream

    global __nvshmemx_int_broadcast_on_stream
    data["__nvshmemx_int_broadcast_on_stream"] = <intptr_t>__nvshmemx_int_broadcast_on_stream

    global __nvshmemx_long_broadcast_on_stream
    data["__nvshmemx_long_broadcast_on_stream"] = <intptr_t>__nvshmemx_long_broadcast_on_stream

    global __nvshmemx_longlong_broadcast_on_stream
    data["__nvshmemx_longlong_broadcast_on_stream"] = <intptr_t>__nvshmemx_longlong_broadcast_on_stream

    global __nvshmemx_int8_broadcast_on_stream
    data["__nvshmemx_int8_broadcast_on_stream"] = <intptr_t>__nvshmemx_int8_broadcast_on_stream

    global __nvshmemx_int16_broadcast_on_stream
    data["__nvshmemx_int16_broadcast_on_stream"] = <intptr_t>__nvshmemx_int16_broadcast_on_stream

    global __nvshmemx_int32_broadcast_on_stream
    data["__nvshmemx_int32_broadcast_on_stream"] = <intptr_t>__nvshmemx_int32_broadcast_on_stream

    global __nvshmemx_int64_broadcast_on_stream
    data["__nvshmemx_int64_broadcast_on_stream"] = <intptr_t>__nvshmemx_int64_broadcast_on_stream

    global __nvshmemx_uint8_broadcast_on_stream
    data["__nvshmemx_uint8_broadcast_on_stream"] = <intptr_t>__nvshmemx_uint8_broadcast_on_stream

    global __nvshmemx_uint16_broadcast_on_stream
    data["__nvshmemx_uint16_broadcast_on_stream"] = <intptr_t>__nvshmemx_uint16_broadcast_on_stream

    global __nvshmemx_uint32_broadcast_on_stream
    data["__nvshmemx_uint32_broadcast_on_stream"] = <intptr_t>__nvshmemx_uint32_broadcast_on_stream

    global __nvshmemx_uint64_broadcast_on_stream
    data["__nvshmemx_uint64_broadcast_on_stream"] = <intptr_t>__nvshmemx_uint64_broadcast_on_stream

    global __nvshmemx_size_broadcast_on_stream
    data["__nvshmemx_size_broadcast_on_stream"] = <intptr_t>__nvshmemx_size_broadcast_on_stream

    global __nvshmemx_bfloat16_fcollect_on_stream
    data["__nvshmemx_bfloat16_fcollect_on_stream"] = <intptr_t>__nvshmemx_bfloat16_fcollect_on_stream

    global __nvshmemx_half_fcollect_on_stream
    data["__nvshmemx_half_fcollect_on_stream"] = <intptr_t>__nvshmemx_half_fcollect_on_stream

    global __nvshmemx_float_fcollect_on_stream
    data["__nvshmemx_float_fcollect_on_stream"] = <intptr_t>__nvshmemx_float_fcollect_on_stream

    global __nvshmemx_double_fcollect_on_stream
    data["__nvshmemx_double_fcollect_on_stream"] = <intptr_t>__nvshmemx_double_fcollect_on_stream

    global __nvshmemx_char_fcollect_on_stream
    data["__nvshmemx_char_fcollect_on_stream"] = <intptr_t>__nvshmemx_char_fcollect_on_stream

    global __nvshmemx_short_fcollect_on_stream
    data["__nvshmemx_short_fcollect_on_stream"] = <intptr_t>__nvshmemx_short_fcollect_on_stream

    global __nvshmemx_schar_fcollect_on_stream
    data["__nvshmemx_schar_fcollect_on_stream"] = <intptr_t>__nvshmemx_schar_fcollect_on_stream

    global __nvshmemx_int_fcollect_on_stream
    data["__nvshmemx_int_fcollect_on_stream"] = <intptr_t>__nvshmemx_int_fcollect_on_stream

    global __nvshmemx_long_fcollect_on_stream
    data["__nvshmemx_long_fcollect_on_stream"] = <intptr_t>__nvshmemx_long_fcollect_on_stream

    global __nvshmemx_longlong_fcollect_on_stream
    data["__nvshmemx_longlong_fcollect_on_stream"] = <intptr_t>__nvshmemx_longlong_fcollect_on_stream

    global __nvshmemx_int8_fcollect_on_stream
    data["__nvshmemx_int8_fcollect_on_stream"] = <intptr_t>__nvshmemx_int8_fcollect_on_stream

    global __nvshmemx_int16_fcollect_on_stream
    data["__nvshmemx_int16_fcollect_on_stream"] = <intptr_t>__nvshmemx_int16_fcollect_on_stream

    global __nvshmemx_int32_fcollect_on_stream
    data["__nvshmemx_int32_fcollect_on_stream"] = <intptr_t>__nvshmemx_int32_fcollect_on_stream

    global __nvshmemx_int64_fcollect_on_stream
    data["__nvshmemx_int64_fcollect_on_stream"] = <intptr_t>__nvshmemx_int64_fcollect_on_stream

    global __nvshmemx_uint8_fcollect_on_stream
    data["__nvshmemx_uint8_fcollect_on_stream"] = <intptr_t>__nvshmemx_uint8_fcollect_on_stream

    global __nvshmemx_uint16_fcollect_on_stream
    data["__nvshmemx_uint16_fcollect_on_stream"] = <intptr_t>__nvshmemx_uint16_fcollect_on_stream

    global __nvshmemx_uint32_fcollect_on_stream
    data["__nvshmemx_uint32_fcollect_on_stream"] = <intptr_t>__nvshmemx_uint32_fcollect_on_stream

    global __nvshmemx_uint64_fcollect_on_stream
    data["__nvshmemx_uint64_fcollect_on_stream"] = <intptr_t>__nvshmemx_uint64_fcollect_on_stream

    global __nvshmemx_size_fcollect_on_stream
    data["__nvshmemx_size_fcollect_on_stream"] = <intptr_t>__nvshmemx_size_fcollect_on_stream

    global __nvshmemx_int8_max_reduce_on_stream
    data["__nvshmemx_int8_max_reduce_on_stream"] = <intptr_t>__nvshmemx_int8_max_reduce_on_stream

    global __nvshmemx_int16_max_reduce_on_stream
    data["__nvshmemx_int16_max_reduce_on_stream"] = <intptr_t>__nvshmemx_int16_max_reduce_on_stream

    global __nvshmemx_int32_max_reduce_on_stream
    data["__nvshmemx_int32_max_reduce_on_stream"] = <intptr_t>__nvshmemx_int32_max_reduce_on_stream

    global __nvshmemx_int64_max_reduce_on_stream
    data["__nvshmemx_int64_max_reduce_on_stream"] = <intptr_t>__nvshmemx_int64_max_reduce_on_stream

    global __nvshmemx_uint8_max_reduce_on_stream
    data["__nvshmemx_uint8_max_reduce_on_stream"] = <intptr_t>__nvshmemx_uint8_max_reduce_on_stream

    global __nvshmemx_uint16_max_reduce_on_stream
    data["__nvshmemx_uint16_max_reduce_on_stream"] = <intptr_t>__nvshmemx_uint16_max_reduce_on_stream

    global __nvshmemx_uint32_max_reduce_on_stream
    data["__nvshmemx_uint32_max_reduce_on_stream"] = <intptr_t>__nvshmemx_uint32_max_reduce_on_stream

    global __nvshmemx_uint64_max_reduce_on_stream
    data["__nvshmemx_uint64_max_reduce_on_stream"] = <intptr_t>__nvshmemx_uint64_max_reduce_on_stream

    global __nvshmemx_size_max_reduce_on_stream
    data["__nvshmemx_size_max_reduce_on_stream"] = <intptr_t>__nvshmemx_size_max_reduce_on_stream

    global __nvshmemx_char_max_reduce_on_stream
    data["__nvshmemx_char_max_reduce_on_stream"] = <intptr_t>__nvshmemx_char_max_reduce_on_stream

    global __nvshmemx_schar_max_reduce_on_stream
    data["__nvshmemx_schar_max_reduce_on_stream"] = <intptr_t>__nvshmemx_schar_max_reduce_on_stream

    global __nvshmemx_short_max_reduce_on_stream
    data["__nvshmemx_short_max_reduce_on_stream"] = <intptr_t>__nvshmemx_short_max_reduce_on_stream

    global __nvshmemx_int_max_reduce_on_stream
    data["__nvshmemx_int_max_reduce_on_stream"] = <intptr_t>__nvshmemx_int_max_reduce_on_stream

    global __nvshmemx_long_max_reduce_on_stream
    data["__nvshmemx_long_max_reduce_on_stream"] = <intptr_t>__nvshmemx_long_max_reduce_on_stream

    global __nvshmemx_longlong_max_reduce_on_stream
    data["__nvshmemx_longlong_max_reduce_on_stream"] = <intptr_t>__nvshmemx_longlong_max_reduce_on_stream

    global __nvshmemx_bfloat16_max_reduce_on_stream
    data["__nvshmemx_bfloat16_max_reduce_on_stream"] = <intptr_t>__nvshmemx_bfloat16_max_reduce_on_stream

    global __nvshmemx_half_max_reduce_on_stream
    data["__nvshmemx_half_max_reduce_on_stream"] = <intptr_t>__nvshmemx_half_max_reduce_on_stream

    global __nvshmemx_float_max_reduce_on_stream
    data["__nvshmemx_float_max_reduce_on_stream"] = <intptr_t>__nvshmemx_float_max_reduce_on_stream

    global __nvshmemx_double_max_reduce_on_stream
    data["__nvshmemx_double_max_reduce_on_stream"] = <intptr_t>__nvshmemx_double_max_reduce_on_stream

    global __nvshmemx_int8_min_reduce_on_stream
    data["__nvshmemx_int8_min_reduce_on_stream"] = <intptr_t>__nvshmemx_int8_min_reduce_on_stream

    global __nvshmemx_int16_min_reduce_on_stream
    data["__nvshmemx_int16_min_reduce_on_stream"] = <intptr_t>__nvshmemx_int16_min_reduce_on_stream

    global __nvshmemx_int32_min_reduce_on_stream
    data["__nvshmemx_int32_min_reduce_on_stream"] = <intptr_t>__nvshmemx_int32_min_reduce_on_stream

    global __nvshmemx_int64_min_reduce_on_stream
    data["__nvshmemx_int64_min_reduce_on_stream"] = <intptr_t>__nvshmemx_int64_min_reduce_on_stream

    global __nvshmemx_uint8_min_reduce_on_stream
    data["__nvshmemx_uint8_min_reduce_on_stream"] = <intptr_t>__nvshmemx_uint8_min_reduce_on_stream

    global __nvshmemx_uint16_min_reduce_on_stream
    data["__nvshmemx_uint16_min_reduce_on_stream"] = <intptr_t>__nvshmemx_uint16_min_reduce_on_stream

    global __nvshmemx_uint32_min_reduce_on_stream
    data["__nvshmemx_uint32_min_reduce_on_stream"] = <intptr_t>__nvshmemx_uint32_min_reduce_on_stream

    global __nvshmemx_uint64_min_reduce_on_stream
    data["__nvshmemx_uint64_min_reduce_on_stream"] = <intptr_t>__nvshmemx_uint64_min_reduce_on_stream

    global __nvshmemx_size_min_reduce_on_stream
    data["__nvshmemx_size_min_reduce_on_stream"] = <intptr_t>__nvshmemx_size_min_reduce_on_stream

    global __nvshmemx_char_min_reduce_on_stream
    data["__nvshmemx_char_min_reduce_on_stream"] = <intptr_t>__nvshmemx_char_min_reduce_on_stream

    global __nvshmemx_schar_min_reduce_on_stream
    data["__nvshmemx_schar_min_reduce_on_stream"] = <intptr_t>__nvshmemx_schar_min_reduce_on_stream

    global __nvshmemx_short_min_reduce_on_stream
    data["__nvshmemx_short_min_reduce_on_stream"] = <intptr_t>__nvshmemx_short_min_reduce_on_stream

    global __nvshmemx_int_min_reduce_on_stream
    data["__nvshmemx_int_min_reduce_on_stream"] = <intptr_t>__nvshmemx_int_min_reduce_on_stream

    global __nvshmemx_long_min_reduce_on_stream
    data["__nvshmemx_long_min_reduce_on_stream"] = <intptr_t>__nvshmemx_long_min_reduce_on_stream

    global __nvshmemx_longlong_min_reduce_on_stream
    data["__nvshmemx_longlong_min_reduce_on_stream"] = <intptr_t>__nvshmemx_longlong_min_reduce_on_stream

    global __nvshmemx_bfloat16_min_reduce_on_stream
    data["__nvshmemx_bfloat16_min_reduce_on_stream"] = <intptr_t>__nvshmemx_bfloat16_min_reduce_on_stream

    global __nvshmemx_half_min_reduce_on_stream
    data["__nvshmemx_half_min_reduce_on_stream"] = <intptr_t>__nvshmemx_half_min_reduce_on_stream

    global __nvshmemx_float_min_reduce_on_stream
    data["__nvshmemx_float_min_reduce_on_stream"] = <intptr_t>__nvshmemx_float_min_reduce_on_stream

    global __nvshmemx_double_min_reduce_on_stream
    data["__nvshmemx_double_min_reduce_on_stream"] = <intptr_t>__nvshmemx_double_min_reduce_on_stream

    global __nvshmemx_int8_sum_reduce_on_stream
    data["__nvshmemx_int8_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_int8_sum_reduce_on_stream

    global __nvshmemx_int16_sum_reduce_on_stream
    data["__nvshmemx_int16_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_int16_sum_reduce_on_stream

    global __nvshmemx_int32_sum_reduce_on_stream
    data["__nvshmemx_int32_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_int32_sum_reduce_on_stream

    global __nvshmemx_int64_sum_reduce_on_stream
    data["__nvshmemx_int64_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_int64_sum_reduce_on_stream

    global __nvshmemx_uint8_sum_reduce_on_stream
    data["__nvshmemx_uint8_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_uint8_sum_reduce_on_stream

    global __nvshmemx_uint16_sum_reduce_on_stream
    data["__nvshmemx_uint16_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_uint16_sum_reduce_on_stream

    global __nvshmemx_uint32_sum_reduce_on_stream
    data["__nvshmemx_uint32_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_uint32_sum_reduce_on_stream

    global __nvshmemx_uint64_sum_reduce_on_stream
    data["__nvshmemx_uint64_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_uint64_sum_reduce_on_stream

    global __nvshmemx_size_sum_reduce_on_stream
    data["__nvshmemx_size_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_size_sum_reduce_on_stream

    global __nvshmemx_char_sum_reduce_on_stream
    data["__nvshmemx_char_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_char_sum_reduce_on_stream

    global __nvshmemx_schar_sum_reduce_on_stream
    data["__nvshmemx_schar_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_schar_sum_reduce_on_stream

    global __nvshmemx_short_sum_reduce_on_stream
    data["__nvshmemx_short_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_short_sum_reduce_on_stream

    global __nvshmemx_int_sum_reduce_on_stream
    data["__nvshmemx_int_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_int_sum_reduce_on_stream

    global __nvshmemx_long_sum_reduce_on_stream
    data["__nvshmemx_long_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_long_sum_reduce_on_stream

    global __nvshmemx_longlong_sum_reduce_on_stream
    data["__nvshmemx_longlong_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_longlong_sum_reduce_on_stream

    global __nvshmemx_bfloat16_sum_reduce_on_stream
    data["__nvshmemx_bfloat16_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_bfloat16_sum_reduce_on_stream

    global __nvshmemx_half_sum_reduce_on_stream
    data["__nvshmemx_half_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_half_sum_reduce_on_stream

    global __nvshmemx_float_sum_reduce_on_stream
    data["__nvshmemx_float_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_float_sum_reduce_on_stream

    global __nvshmemx_double_sum_reduce_on_stream
    data["__nvshmemx_double_sum_reduce_on_stream"] = <intptr_t>__nvshmemx_double_sum_reduce_on_stream

    global __nvshmemx_int8_max_reducescatter_on_stream
    data["__nvshmemx_int8_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int8_max_reducescatter_on_stream

    global __nvshmemx_int16_max_reducescatter_on_stream
    data["__nvshmemx_int16_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int16_max_reducescatter_on_stream

    global __nvshmemx_int32_max_reducescatter_on_stream
    data["__nvshmemx_int32_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int32_max_reducescatter_on_stream

    global __nvshmemx_int64_max_reducescatter_on_stream
    data["__nvshmemx_int64_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int64_max_reducescatter_on_stream

    global __nvshmemx_uint8_max_reducescatter_on_stream
    data["__nvshmemx_uint8_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint8_max_reducescatter_on_stream

    global __nvshmemx_uint16_max_reducescatter_on_stream
    data["__nvshmemx_uint16_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint16_max_reducescatter_on_stream

    global __nvshmemx_uint32_max_reducescatter_on_stream
    data["__nvshmemx_uint32_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint32_max_reducescatter_on_stream

    global __nvshmemx_uint64_max_reducescatter_on_stream
    data["__nvshmemx_uint64_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint64_max_reducescatter_on_stream

    global __nvshmemx_size_max_reducescatter_on_stream
    data["__nvshmemx_size_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_size_max_reducescatter_on_stream

    global __nvshmemx_char_max_reducescatter_on_stream
    data["__nvshmemx_char_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_char_max_reducescatter_on_stream

    global __nvshmemx_schar_max_reducescatter_on_stream
    data["__nvshmemx_schar_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_schar_max_reducescatter_on_stream

    global __nvshmemx_short_max_reducescatter_on_stream
    data["__nvshmemx_short_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_short_max_reducescatter_on_stream

    global __nvshmemx_int_max_reducescatter_on_stream
    data["__nvshmemx_int_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int_max_reducescatter_on_stream

    global __nvshmemx_long_max_reducescatter_on_stream
    data["__nvshmemx_long_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_long_max_reducescatter_on_stream

    global __nvshmemx_longlong_max_reducescatter_on_stream
    data["__nvshmemx_longlong_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_longlong_max_reducescatter_on_stream

    global __nvshmemx_bfloat16_max_reducescatter_on_stream
    data["__nvshmemx_bfloat16_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_bfloat16_max_reducescatter_on_stream

    global __nvshmemx_half_max_reducescatter_on_stream
    data["__nvshmemx_half_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_half_max_reducescatter_on_stream

    global __nvshmemx_float_max_reducescatter_on_stream
    data["__nvshmemx_float_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_float_max_reducescatter_on_stream

    global __nvshmemx_double_max_reducescatter_on_stream
    data["__nvshmemx_double_max_reducescatter_on_stream"] = <intptr_t>__nvshmemx_double_max_reducescatter_on_stream

    global __nvshmemx_int8_min_reducescatter_on_stream
    data["__nvshmemx_int8_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int8_min_reducescatter_on_stream

    global __nvshmemx_int16_min_reducescatter_on_stream
    data["__nvshmemx_int16_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int16_min_reducescatter_on_stream

    global __nvshmemx_int32_min_reducescatter_on_stream
    data["__nvshmemx_int32_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int32_min_reducescatter_on_stream

    global __nvshmemx_int64_min_reducescatter_on_stream
    data["__nvshmemx_int64_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int64_min_reducescatter_on_stream

    global __nvshmemx_uint8_min_reducescatter_on_stream
    data["__nvshmemx_uint8_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint8_min_reducescatter_on_stream

    global __nvshmemx_uint16_min_reducescatter_on_stream
    data["__nvshmemx_uint16_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint16_min_reducescatter_on_stream

    global __nvshmemx_uint32_min_reducescatter_on_stream
    data["__nvshmemx_uint32_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint32_min_reducescatter_on_stream

    global __nvshmemx_uint64_min_reducescatter_on_stream
    data["__nvshmemx_uint64_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint64_min_reducescatter_on_stream

    global __nvshmemx_size_min_reducescatter_on_stream
    data["__nvshmemx_size_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_size_min_reducescatter_on_stream

    global __nvshmemx_char_min_reducescatter_on_stream
    data["__nvshmemx_char_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_char_min_reducescatter_on_stream

    global __nvshmemx_schar_min_reducescatter_on_stream
    data["__nvshmemx_schar_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_schar_min_reducescatter_on_stream

    global __nvshmemx_short_min_reducescatter_on_stream
    data["__nvshmemx_short_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_short_min_reducescatter_on_stream

    global __nvshmemx_int_min_reducescatter_on_stream
    data["__nvshmemx_int_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int_min_reducescatter_on_stream

    global __nvshmemx_long_min_reducescatter_on_stream
    data["__nvshmemx_long_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_long_min_reducescatter_on_stream

    global __nvshmemx_longlong_min_reducescatter_on_stream
    data["__nvshmemx_longlong_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_longlong_min_reducescatter_on_stream

    global __nvshmemx_bfloat16_min_reducescatter_on_stream
    data["__nvshmemx_bfloat16_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_bfloat16_min_reducescatter_on_stream

    global __nvshmemx_half_min_reducescatter_on_stream
    data["__nvshmemx_half_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_half_min_reducescatter_on_stream

    global __nvshmemx_float_min_reducescatter_on_stream
    data["__nvshmemx_float_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_float_min_reducescatter_on_stream

    global __nvshmemx_double_min_reducescatter_on_stream
    data["__nvshmemx_double_min_reducescatter_on_stream"] = <intptr_t>__nvshmemx_double_min_reducescatter_on_stream

    global __nvshmemx_int8_sum_reducescatter_on_stream
    data["__nvshmemx_int8_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int8_sum_reducescatter_on_stream

    global __nvshmemx_int16_sum_reducescatter_on_stream
    data["__nvshmemx_int16_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int16_sum_reducescatter_on_stream

    global __nvshmemx_int32_sum_reducescatter_on_stream
    data["__nvshmemx_int32_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int32_sum_reducescatter_on_stream

    global __nvshmemx_int64_sum_reducescatter_on_stream
    data["__nvshmemx_int64_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int64_sum_reducescatter_on_stream

    global __nvshmemx_uint8_sum_reducescatter_on_stream
    data["__nvshmemx_uint8_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint8_sum_reducescatter_on_stream

    global __nvshmemx_uint16_sum_reducescatter_on_stream
    data["__nvshmemx_uint16_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint16_sum_reducescatter_on_stream

    global __nvshmemx_uint32_sum_reducescatter_on_stream
    data["__nvshmemx_uint32_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint32_sum_reducescatter_on_stream

    global __nvshmemx_uint64_sum_reducescatter_on_stream
    data["__nvshmemx_uint64_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_uint64_sum_reducescatter_on_stream

    global __nvshmemx_size_sum_reducescatter_on_stream
    data["__nvshmemx_size_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_size_sum_reducescatter_on_stream

    global __nvshmemx_char_sum_reducescatter_on_stream
    data["__nvshmemx_char_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_char_sum_reducescatter_on_stream

    global __nvshmemx_schar_sum_reducescatter_on_stream
    data["__nvshmemx_schar_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_schar_sum_reducescatter_on_stream

    global __nvshmemx_short_sum_reducescatter_on_stream
    data["__nvshmemx_short_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_short_sum_reducescatter_on_stream

    global __nvshmemx_int_sum_reducescatter_on_stream
    data["__nvshmemx_int_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_int_sum_reducescatter_on_stream

    global __nvshmemx_long_sum_reducescatter_on_stream
    data["__nvshmemx_long_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_long_sum_reducescatter_on_stream

    global __nvshmemx_longlong_sum_reducescatter_on_stream
    data["__nvshmemx_longlong_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_longlong_sum_reducescatter_on_stream

    global __nvshmemx_bfloat16_sum_reducescatter_on_stream
    data["__nvshmemx_bfloat16_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_bfloat16_sum_reducescatter_on_stream

    global __nvshmemx_half_sum_reducescatter_on_stream
    data["__nvshmemx_half_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_half_sum_reducescatter_on_stream

    global __nvshmemx_float_sum_reducescatter_on_stream
    data["__nvshmemx_float_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_float_sum_reducescatter_on_stream

    global __nvshmemx_double_sum_reducescatter_on_stream
    data["__nvshmemx_double_sum_reducescatter_on_stream"] = <intptr_t>__nvshmemx_double_sum_reducescatter_on_stream

    global __nvshmemx_hostlib_init_attr
    data["__nvshmemx_hostlib_init_attr"] = <intptr_t>__nvshmemx_hostlib_init_attr

    global __nvshmemx_hostlib_finalize
    data["__nvshmemx_hostlib_finalize"] = <intptr_t>__nvshmemx_hostlib_finalize

    global __nvshmemx_set_attr_uniqueid_args
    data["__nvshmemx_set_attr_uniqueid_args"] = <intptr_t>__nvshmemx_set_attr_uniqueid_args

    global __nvshmemx_set_attr_mpi_comm_args
    data["__nvshmemx_set_attr_mpi_comm_args"] = <intptr_t>__nvshmemx_set_attr_mpi_comm_args

    global __nvshmemx_get_uniqueid
    data["__nvshmemx_get_uniqueid"] = <intptr_t>__nvshmemx_get_uniqueid

    global __nvshmemx_cumodule_init
    data["__nvshmemx_cumodule_init"] = <intptr_t>__nvshmemx_cumodule_init

    global __nvshmemx_cumodule_finalize
    data["__nvshmemx_cumodule_finalize"] = <intptr_t>__nvshmemx_cumodule_finalize

    global __nvshmemx_putmem_on_stream
    data["__nvshmemx_putmem_on_stream"] = <intptr_t>__nvshmemx_putmem_on_stream

    global __nvshmemx_putmem_signal_on_stream
    data["__nvshmemx_putmem_signal_on_stream"] = <intptr_t>__nvshmemx_putmem_signal_on_stream

    global __nvshmemx_getmem_on_stream
    data["__nvshmemx_getmem_on_stream"] = <intptr_t>__nvshmemx_getmem_on_stream

    global __nvshmemx_quiet_on_stream
    data["__nvshmemx_quiet_on_stream"] = <intptr_t>__nvshmemx_quiet_on_stream

    global __nvshmemx_signal_wait_until_on_stream
    data["__nvshmemx_signal_wait_until_on_stream"] = <intptr_t>__nvshmemx_signal_wait_until_on_stream

    func_ptrs = data
    return data


cpdef _inspect_function_pointer(str name):
    global func_ptrs
    if func_ptrs is None:
        func_ptrs = _inspect_function_pointers()
    return func_ptrs[name]


###############################################################################
# Wrapper functions
###############################################################################

cdef int _nvshmemx_init_status() except* nogil:
    global __nvshmemx_init_status
    _check_or_init_nvshmem()
    if __nvshmemx_init_status == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_init_status is not found")
    return (<int (*)() nogil>__nvshmemx_init_status)(
        )


cdef int _nvshmem_my_pe() except* nogil:
    global __nvshmem_my_pe
    _check_or_init_nvshmem()
    if __nvshmem_my_pe == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_my_pe is not found")
    return (<int (*)() nogil>__nvshmem_my_pe)(
        )


cdef int _nvshmem_n_pes() except* nogil:
    global __nvshmem_n_pes
    _check_or_init_nvshmem()
    if __nvshmem_n_pes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_n_pes is not found")
    return (<int (*)() nogil>__nvshmem_n_pes)(
        )


cdef void _nvshmem_info_get_version(int* major, int* minor) except* nogil:
    global __nvshmem_info_get_version
    _check_or_init_nvshmem()
    if __nvshmem_info_get_version == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_info_get_version is not found")
    (<void (*)(int*, int*) nogil>__nvshmem_info_get_version)(
        major, minor)


cdef void _nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) except* nogil:
    global __nvshmemx_vendor_get_version_info
    _check_or_init_nvshmem()
    if __nvshmemx_vendor_get_version_info == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_vendor_get_version_info is not found")
    (<void (*)(int*, int*, int*) nogil>__nvshmemx_vendor_get_version_info)(
        major, minor, patch)


cdef void* _nvshmem_malloc(size_t size) except* nogil:
    global __nvshmem_malloc
    _check_or_init_nvshmem()
    if __nvshmem_malloc == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_malloc is not found")
    return (<void* (*)(size_t) nogil>__nvshmem_malloc)(
        size)


cdef void* _nvshmem_calloc(size_t count, size_t size) except* nogil:
    global __nvshmem_calloc
    _check_or_init_nvshmem()
    if __nvshmem_calloc == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_calloc is not found")
    return (<void* (*)(size_t, size_t) nogil>__nvshmem_calloc)(
        count, size)


cdef void* _nvshmem_align(size_t count, size_t size) except* nogil:
    global __nvshmem_align
    _check_or_init_nvshmem()
    if __nvshmem_align == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_align is not found")
    return (<void* (*)(size_t, size_t) nogil>__nvshmem_align)(
        count, size)


cdef void _nvshmem_free(void* ptr) except* nogil:
    global __nvshmem_free
    _check_or_init_nvshmem()
    if __nvshmem_free == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_free is not found")
    (<void (*)(void*) nogil>__nvshmem_free)(
        ptr)


cdef void* _nvshmem_ptr(const void* dest, int pe) except* nogil:
    global __nvshmem_ptr
    _check_or_init_nvshmem()
    if __nvshmem_ptr == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_ptr is not found")
    return (<void* (*)(const void*, int) nogil>__nvshmem_ptr)(
        dest, pe)


cdef void* _nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr) except* nogil:
    global __nvshmemx_mc_ptr
    _check_or_init_nvshmem()
    if __nvshmemx_mc_ptr == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_mc_ptr is not found")
    return (<void* (*)(nvshmem_team_t, const void*) nogil>__nvshmemx_mc_ptr)(
        team, ptr)


cdef int _nvshmem_team_my_pe(nvshmem_team_t team) except* nogil:
    global __nvshmem_team_my_pe
    _check_or_init_nvshmem()
    if __nvshmem_team_my_pe == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_team_my_pe is not found")
    return (<int (*)(nvshmem_team_t) nogil>__nvshmem_team_my_pe)(
        team)


cdef int _nvshmem_team_n_pes(nvshmem_team_t team) except* nogil:
    global __nvshmem_team_n_pes
    _check_or_init_nvshmem()
    if __nvshmem_team_n_pes == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_team_n_pes is not found")
    return (<int (*)(nvshmem_team_t) nogil>__nvshmem_team_n_pes)(
        team)


cdef int _nvshmem_barrier(nvshmem_team_t team) except* nogil:
    global __nvshmem_barrier
    _check_or_init_nvshmem()
    if __nvshmem_barrier == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_barrier is not found")
    return (<int (*)(nvshmem_team_t) nogil>__nvshmem_barrier)(
        team)


cdef void _nvshmem_barrier_all() except* nogil:
    global __nvshmem_barrier_all
    _check_or_init_nvshmem()
    if __nvshmem_barrier_all == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmem_barrier_all is not found")
    (<void (*)() nogil>__nvshmem_barrier_all)(
        )


cdef int _nvshmemx_bfloat16_alltoall_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_half_alltoall_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_float_alltoall_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_double_alltoall_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_char_alltoall_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_short_alltoall_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_schar_alltoall_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int_alltoall_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_long_alltoall_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_longlong_alltoall_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int8_alltoall_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int16_alltoall_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int32_alltoall_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int64_alltoall_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint8_alltoall_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint16_alltoall_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint32_alltoall_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint64_alltoall_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_size_alltoall_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_alltoall_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_alltoall_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_alltoall_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_alltoall_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil:
    global __nvshmemx_barrier_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_barrier_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_barrier_on_stream is not found")
    return (<int (*)(nvshmem_team_t, cudaStream_t) nogil>__nvshmemx_barrier_on_stream)(
        team, stream)


cdef int _nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil:
    global __nvshmemx_team_sync_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_team_sync_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_team_sync_on_stream is not found")
    return (<int (*)(nvshmem_team_t, cudaStream_t) nogil>__nvshmemx_team_sync_on_stream)(
        team, stream)


cdef int _nvshmemx_bfloat16_broadcast_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, int, cudaStream_t) nogil>__nvshmemx_bfloat16_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_half_broadcast_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, int, cudaStream_t) nogil>__nvshmemx_half_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_float_broadcast_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, int, cudaStream_t) nogil>__nvshmemx_float_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_double_broadcast_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, int, cudaStream_t) nogil>__nvshmemx_double_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_char_broadcast_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, int, cudaStream_t) nogil>__nvshmemx_char_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_short_broadcast_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, int, cudaStream_t) nogil>__nvshmemx_short_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_schar_broadcast_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, int, cudaStream_t) nogil>__nvshmemx_schar_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_int_broadcast_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, int, cudaStream_t) nogil>__nvshmemx_int_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_long_broadcast_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, int, cudaStream_t) nogil>__nvshmemx_long_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_longlong_broadcast_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, int, cudaStream_t) nogil>__nvshmemx_longlong_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_int8_broadcast_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_int8_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_int16_broadcast_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_int16_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_int32_broadcast_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_int32_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_int64_broadcast_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_int64_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_uint8_broadcast_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_uint8_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_uint16_broadcast_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_uint16_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_uint32_broadcast_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_uint32_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_uint64_broadcast_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_uint64_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_size_broadcast_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_broadcast_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_broadcast_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_broadcast_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, int, cudaStream_t) nogil>__nvshmemx_size_broadcast_on_stream)(
        team, dest, src, nelem, PE_root, stream)


cdef int _nvshmemx_bfloat16_fcollect_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_half_fcollect_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_float_fcollect_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_double_fcollect_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_char_fcollect_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_short_fcollect_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_schar_fcollect_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int_fcollect_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_long_fcollect_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_longlong_fcollect_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int8_fcollect_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int16_fcollect_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int32_fcollect_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int64_fcollect_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint8_fcollect_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint16_fcollect_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint32_fcollect_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_uint64_fcollect_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_size_fcollect_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_fcollect_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_fcollect_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_fcollect_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_fcollect_on_stream)(
        team, dest, src, nelem, stream)


cdef int _nvshmemx_int8_max_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_max_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_max_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_max_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_max_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_max_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_max_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_max_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_max_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_max_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_max_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_max_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_max_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_max_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_max_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_max_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_max_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_max_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_max_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_max_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_max_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_max_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_max_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int8_min_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_min_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_min_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_min_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_min_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_min_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_min_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_min_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_min_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_min_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_min_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_min_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_min_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_min_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_min_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_min_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_min_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_min_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_min_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_min_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_min_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_min_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_min_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int8_sum_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_sum_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_sum_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_sum_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_sum_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_sum_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_sum_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_sum_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_sum_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_sum_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_sum_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_sum_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_sum_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_sum_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_sum_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_sum_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_sum_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_sum_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_sum_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_sum_reduce_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_sum_reduce_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_sum_reduce_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_sum_reduce_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int8_max_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_max_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_max_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_max_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_max_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_max_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_max_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_max_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_max_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_max_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_max_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_max_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_max_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_max_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_max_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_max_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_max_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_max_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_max_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_max_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_max_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_max_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_max_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int8_min_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_min_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_min_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_min_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_min_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_min_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_min_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_min_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_min_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_min_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_min_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_min_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_min_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_min_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_min_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_min_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_min_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_min_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_min_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_min_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_min_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_min_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_min_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int8_sum_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int8_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int8_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int8_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int8_t*, const int8_t*, size_t, cudaStream_t) nogil>__nvshmemx_int8_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int16_sum_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int16_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int16_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int16_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int16_t*, const int16_t*, size_t, cudaStream_t) nogil>__nvshmemx_int16_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int32_sum_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int32_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int32_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int32_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int32_t*, const int32_t*, size_t, cudaStream_t) nogil>__nvshmemx_int32_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int64_sum_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int64_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int64_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int64_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int64_t*, const int64_t*, size_t, cudaStream_t) nogil>__nvshmemx_int64_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint8_sum_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint8_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint8_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint8_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint8_t*, const uint8_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint8_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint16_sum_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint16_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint16_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint16_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint16_t*, const uint16_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint16_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint32_sum_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint32_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint32_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint32_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint32_t*, const uint32_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint32_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_uint64_sum_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_uint64_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_uint64_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_uint64_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, uint64_t*, const uint64_t*, size_t, cudaStream_t) nogil>__nvshmemx_uint64_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_size_sum_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_size_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_size_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_size_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, size_t*, const size_t*, size_t, cudaStream_t) nogil>__nvshmemx_size_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_char_sum_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_char_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_char_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_char_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, char*, const char*, size_t, cudaStream_t) nogil>__nvshmemx_char_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_schar_sum_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_schar_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_schar_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_schar_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, signed char*, const signed char*, size_t, cudaStream_t) nogil>__nvshmemx_schar_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_short_sum_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_short_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_short_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_short_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, short*, const short*, size_t, cudaStream_t) nogil>__nvshmemx_short_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_int_sum_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_int_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_int_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_int_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, int*, const int*, size_t, cudaStream_t) nogil>__nvshmemx_int_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_long_sum_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_long_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_long_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_long_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long*, const long*, size_t, cudaStream_t) nogil>__nvshmemx_long_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_longlong_sum_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_longlong_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_longlong_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_longlong_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, long long*, const long long*, size_t, cudaStream_t) nogil>__nvshmemx_longlong_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_bfloat16_sum_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_bfloat16_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_bfloat16_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_bfloat16_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, __nv_bfloat16*, const __nv_bfloat16*, size_t, cudaStream_t) nogil>__nvshmemx_bfloat16_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_half_sum_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_half_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_half_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_half_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, half*, const half*, size_t, cudaStream_t) nogil>__nvshmemx_half_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_float_sum_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_float_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_float_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_float_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, float*, const float*, size_t, cudaStream_t) nogil>__nvshmemx_float_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_double_sum_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil:
    global __nvshmemx_double_sum_reducescatter_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_double_sum_reducescatter_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_double_sum_reducescatter_on_stream is not found")
    return (<int (*)(nvshmem_team_t, double*, const double*, size_t, cudaStream_t) nogil>__nvshmemx_double_sum_reducescatter_on_stream)(
        team, dest, src, nreduce, stream)


cdef int _nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except* nogil:
    global __nvshmemx_hostlib_init_attr
    _check_or_init_nvshmem()
    if __nvshmemx_hostlib_init_attr == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_hostlib_init_attr is not found")
    return (<int (*)(unsigned int, nvshmemx_init_attr_t*) nogil>__nvshmemx_hostlib_init_attr)(
        flags, attr)


cdef void _nvshmemx_hostlib_finalize() except* nogil:
    global __nvshmemx_hostlib_finalize
    _check_or_init_nvshmem()
    if __nvshmemx_hostlib_finalize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_hostlib_finalize is not found")
    (<void (*)() nogil>__nvshmemx_hostlib_finalize)(
        )


cdef int _nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except* nogil:
    global __nvshmemx_set_attr_uniqueid_args
    _check_or_init_nvshmem()
    if __nvshmemx_set_attr_uniqueid_args == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_set_attr_uniqueid_args is not found")
    return (<int (*)(const int, const int, const nvshmemx_uniqueid_t*, nvshmemx_init_attr_t*) nogil>__nvshmemx_set_attr_uniqueid_args)(
        myrank, nranks, uniqueid, attr)


cdef int _nvshmemx_set_attr_mpi_comm_args(void* mpi_comm, nvshmemx_init_attr_t* nvshmem_attr) except* nogil:
    global __nvshmemx_set_attr_mpi_comm_args
    _check_or_init_nvshmem()
    if __nvshmemx_set_attr_mpi_comm_args == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_set_attr_mpi_comm_args is not found")
    return (<int (*)(void*, nvshmemx_init_attr_t*) nogil>__nvshmemx_set_attr_mpi_comm_args)(
        mpi_comm, nvshmem_attr)


cdef int _nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except* nogil:
    global __nvshmemx_get_uniqueid
    _check_or_init_nvshmem()
    if __nvshmemx_get_uniqueid == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_get_uniqueid is not found")
    return (<int (*)(nvshmemx_uniqueid_t*) nogil>__nvshmemx_get_uniqueid)(
        uniqueid)


cdef int _nvshmemx_cumodule_init(CUmodule module) except* nogil:
    global __nvshmemx_cumodule_init
    _check_or_init_nvshmem()
    if __nvshmemx_cumodule_init == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_cumodule_init is not found")
    return (<int (*)(CUmodule) nogil>__nvshmemx_cumodule_init)(
        module)


cdef int _nvshmemx_cumodule_finalize(CUmodule module) except* nogil:
    global __nvshmemx_cumodule_finalize
    _check_or_init_nvshmem()
    if __nvshmemx_cumodule_finalize == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_cumodule_finalize is not found")
    return (<int (*)(CUmodule) nogil>__nvshmemx_cumodule_finalize)(
        module)


cdef void _nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    global __nvshmemx_putmem_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_putmem_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_putmem_on_stream is not found")
    (<void (*)(void*, const void*, size_t, int, cudaStream_t) nogil>__nvshmemx_putmem_on_stream)(
        dest, source, bytes, pe, cstrm)


cdef void _nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil:
    global __nvshmemx_putmem_signal_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_putmem_signal_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_putmem_signal_on_stream is not found")
    (<void (*)(void*, const void*, size_t, uint64_t*, uint64_t, int, int, cudaStream_t) nogil>__nvshmemx_putmem_signal_on_stream)(
        dest, source, bytes, sig_addr, signal, sig_op, pe, cstrm)


cdef void _nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil:
    global __nvshmemx_getmem_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_getmem_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_getmem_on_stream is not found")
    (<void (*)(void*, const void*, size_t, int, cudaStream_t) nogil>__nvshmemx_getmem_on_stream)(
        dest, source, bytes, pe, cstrm)


cdef void _nvshmemx_quiet_on_stream(cudaStream_t cstrm) except* nogil:
    global __nvshmemx_quiet_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_quiet_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_quiet_on_stream is not found")
    (<void (*)(cudaStream_t) nogil>__nvshmemx_quiet_on_stream)(
        cstrm)


cdef void _nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value, cudaStream_t cstream) except* nogil:
    global __nvshmemx_signal_wait_until_on_stream
    _check_or_init_nvshmem()
    if __nvshmemx_signal_wait_until_on_stream == NULL:
        with gil:
            raise FunctionNotFoundError("function nvshmemx_signal_wait_until_on_stream is not found")
    (<void (*)(uint64_t*, int, uint64_t, cudaStream_t) nogil>__nvshmemx_signal_wait_until_on_stream)(
        sig_addr, cmp, cmp_value, cstream)