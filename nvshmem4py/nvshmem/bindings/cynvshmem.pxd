# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This code was automatically generated from NVSHMEM with version 3.3.0. 
# Modify it directly at your own risk.


from libc.stdint cimport (
    int8_t,  uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t,
    intptr_t, uintptr_t
)

###############################################################################
# Types (structs, enums, ...)
###############################################################################

# enums
ctypedef enum nvshmemx_signal_op_t "nvshmemx_signal_op_t":
    NVSHMEM_SIGNAL_SET "NVSHMEM_SIGNAL_SET" = 9
    NVSHMEM_SIGNAL_ADD "NVSHMEM_SIGNAL_ADD" = 10

ctypedef enum nvshmemx_tma_policy_t "nvshmemx_tma_policy_t":
    NVSHMEMX_TMA_DISABLE "NVSHMEMX_TMA_DISABLE" = 0
    NVSHMEMX_TMA_ENABLE "NVSHMEMX_TMA_ENABLE" = 1
    NVSHMEMX_TMA_FORCE "NVSHMEMX_TMA_FORCE" = 2
    NVSHMEMX_TMA_POLICY_MAX "NVSHMEMX_TMA_POLICY_MAX" = 32767

ctypedef enum nvshmemx_smem_amount_t "nvshmemx_smem_amount_t":
    NVSHMEMX_SMEM_RECOMMENDED "NVSHMEMX_SMEM_RECOMMENDED" = 0
    NVSHMEMX_SMEM_MINIMUM "NVSHMEMX_SMEM_MINIMUM" = 1
    NVSHMEMX_SMEM_BARRIERS_ONLY "NVSHMEMX_SMEM_BARRIERS_ONLY" = 2
    NVSHMEMX_SMEM_AMOUNT_MAX "NVSHMEMX_SMEM_AMOUNT_MAX" = 32767

ctypedef enum nvshmemx_region_hint_t "nvshmemx_region_hint_t":
    NVSHMEMX_REGION_HINT_NONE "NVSHMEMX_REGION_HINT_NONE" = 0
    NVSHMEMX_REGION_HINT_BATCH_RMA "NVSHMEMX_REGION_HINT_BATCH_RMA" = (1u << 0)

ctypedef enum nvshmemx_cmp_type_t "nvshmemx_cmp_type_t":
    NVSHMEM_CMP_EQ "NVSHMEM_CMP_EQ" = 0
    NVSHMEM_CMP_NE "NVSHMEM_CMP_NE"
    NVSHMEM_CMP_GT "NVSHMEM_CMP_GT"
    NVSHMEM_CMP_LE "NVSHMEM_CMP_LE"
    NVSHMEM_CMP_LT "NVSHMEM_CMP_LT"
    NVSHMEM_CMP_GE "NVSHMEM_CMP_GE"
    NVSHMEM_CMP_SENTINEL "NVSHMEM_CMP_SENTINEL" = 32767

ctypedef enum nvshmemx_thread_support_t "nvshmemx_thread_support_t":
    NVSHMEM_THREAD_SINGLE "NVSHMEM_THREAD_SINGLE" = 0
    NVSHMEM_THREAD_FUNNELED "NVSHMEM_THREAD_FUNNELED"
    NVSHMEM_THREAD_SERIALIZED "NVSHMEM_THREAD_SERIALIZED"
    NVSHMEM_THREAD_MULTIPLE "NVSHMEM_THREAD_MULTIPLE"
    NVSHMEM_THREAD_TYPE_SENTINEL "NVSHMEM_THREAD_TYPE_SENTINEL" = 32767

ctypedef enum nvshmemx_proxy_status_t "nvshmemx_proxy_status_t":
    PROXY_GLOBAL_EXIT_NOT_REQUESTED "PROXY_GLOBAL_EXIT_NOT_REQUESTED" = 0
    PROXY_GLOBAL_EXIT_INIT "PROXY_GLOBAL_EXIT_INIT"
    PROXY_GLOBAL_EXIT_REQUESTED "PROXY_GLOBAL_EXIT_REQUESTED"
    PROXY_GLOBAL_EXIT_FINISHED "PROXY_GLOBAL_EXIT_FINISHED"
    PROXY_GLOBAL_EXIT_MAX_STATE "PROXY_GLOBAL_EXIT_MAX_STATE" = 32767

ctypedef enum nvshmemx_init_status_t "nvshmemx_init_status_t":
    NVSHMEM_STATUS_NOT_INITIALIZED "NVSHMEM_STATUS_NOT_INITIALIZED" = 0
    NVSHMEM_STATUS_IS_BOOTSTRAPPED "NVSHMEM_STATUS_IS_BOOTSTRAPPED"
    NVSHMEM_STATUS_IS_INITIALIZED "NVSHMEM_STATUS_IS_INITIALIZED"
    NVSHMEM_STATUS_LIMITED_MPG "NVSHMEM_STATUS_LIMITED_MPG"
    NVSHMEM_STATUS_FULL_MPG "NVSHMEM_STATUS_FULL_MPG"
    NVSHMEM_STATUS_INVALID "NVSHMEM_STATUS_INVALID" = 32767

ctypedef enum nvshmemx_qp_handle_index_t "nvshmemx_qp_handle_index_t":
    NVSHMEMX_QP_HOST "NVSHMEMX_QP_HOST" = 0
    NVSHMEMX_QP_DEFAULT "NVSHMEMX_QP_DEFAULT" = 1
    NVSHMEMX_QP_ANY "NVSHMEMX_QP_ANY" = 32767
    NVSHMEMX_QP_ALL "NVSHMEMX_QP_ALL" = 32767

ctypedef enum nvshmem_pe_index_t "nvshmem_pe_index_t":
    NVSHMEM_PE_INVALID "NVSHMEM_PE_INVALID" = -(1)
    NVSHMEMX_PE_ANY "NVSHMEMX_PE_ANY" = (1 << 15)
    NVSHMEMX_PE_ALL "NVSHMEMX_PE_ALL" = (1 << 15)

ctypedef enum nvshmem_team_id_t "nvshmem_team_id_t":
    NVSHMEM_TEAM_INVALID "NVSHMEM_TEAM_INVALID" = -(1)
    NVSHMEM_TEAM_WORLD "NVSHMEM_TEAM_WORLD" = 0
    NVSHMEM_TEAM_WORLD_INDEX "NVSHMEM_TEAM_WORLD_INDEX" = 0
    NVSHMEM_TEAM_SHARED "NVSHMEM_TEAM_SHARED" = 1
    NVSHMEM_TEAM_SHARED_INDEX "NVSHMEM_TEAM_SHARED_INDEX" = 1
    NVSHMEMX_TEAM_NODE "NVSHMEMX_TEAM_NODE" = 2
    NVSHMEM_TEAM_NODE_INDEX "NVSHMEM_TEAM_NODE_INDEX" = 2
    NVSHMEMX_TEAM_SAME_MYPE_NODE "NVSHMEMX_TEAM_SAME_MYPE_NODE" = 3
    NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX "NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX" = 3
    NVSHMEMI_TEAM_SAME_GPU "NVSHMEMI_TEAM_SAME_GPU" = 4
    NVSHMEM_TEAM_SAME_GPU_INDEX "NVSHMEM_TEAM_SAME_GPU_INDEX" = 4
    NVSHMEMI_TEAM_GPU_LEADERS "NVSHMEMI_TEAM_GPU_LEADERS" = 5
    NVSHMEM_TEAM_GPU_LEADERS_INDEX "NVSHMEM_TEAM_GPU_LEADERS_INDEX" = 5
    NVSHMEM_TEAM_MC_SHARED "NVSHMEM_TEAM_MC_SHARED" = 6
    NVSHMEM_TEAM_MC_SHARED_INDEX "NVSHMEM_TEAM_MC_SHARED_INDEX" = 6
    NVSHMEM_TEAMS_MIN "NVSHMEM_TEAMS_MIN" = 7
    NVSHMEM_TEAM_INDEX_MAX "NVSHMEM_TEAM_INDEX_MAX" = 32767

ctypedef enum nvshmemx_status "nvshmemx_status":
    NVSHMEMX_SUCCESS "NVSHMEMX_SUCCESS"
    NVSHMEMX_ERROR_INVALID_VALUE "NVSHMEMX_ERROR_INVALID_VALUE"
    NVSHMEMX_ERROR_OUT_OF_MEMORY "NVSHMEMX_ERROR_OUT_OF_MEMORY"
    NVSHMEMX_ERROR_NOT_SUPPORTED "NVSHMEMX_ERROR_NOT_SUPPORTED"
    NVSHMEMX_ERROR_SYMMETRY "NVSHMEMX_ERROR_SYMMETRY"
    NVSHMEMX_ERROR_GPU_NOT_SELECTED "NVSHMEMX_ERROR_GPU_NOT_SELECTED"
    NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED "NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED"
    NVSHMEMX_ERROR_INTERNAL "NVSHMEMX_ERROR_INTERNAL"
    NVSHMEMX_ERROR_SENTINEL "NVSHMEMX_ERROR_SENTINEL" = 32767

ctypedef enum flags "flags":
    NVSHMEMX_INIT_THREAD_PES "NVSHMEMX_INIT_THREAD_PES" = 1
    NVSHMEMX_INIT_WITH_MPI_COMM "NVSHMEMX_INIT_WITH_MPI_COMM" = (1 << 1)
    NVSHMEMX_INIT_WITH_SHMEM "NVSHMEMX_INIT_WITH_SHMEM" = (1 << 2)
    NVSHMEMX_INIT_WITH_UNIQUEID "NVSHMEMX_INIT_WITH_UNIQUEID" = (1 << 3)
    NVSHMEMX_INIT_MAX "NVSHMEMX_INIT_MAX" = (1 << 31)


# types
cdef extern from "cuda_runtime_api.h":
    cdef struct CUstream_st
    ctypedef CUstream_st* cudaStream_t

ctypedef void* CUmodule 'CUmodule'
ctypedef void* CUlibrary 'CUlibrary'

# Types for NVSHMEM Collectives
# Floats
cdef extern from "cuda_fp16.h":
    ctypedef struct __half:
        pass
    ctypedef __half half

cdef extern from "cuda_bf16.h":
    ctypedef struct __nv_bfloat16:
        pass
    ctypedef __nv_bfloat16 bfloat16

cdef extern from "limits.h":
    cdef int16_t INT16_MAX;



# Longs
ctypedef long longlong
ctypedef signed char schar



ctypedef struct nvshmemx_uniqueid_v1 'nvshmemx_uniqueid_v1':
    int version
    char internal[124]
ctypedef int nvshmemx_qp_handle_t 'nvshmemx_qp_handle_t'
ctypedef uint64_t nvshmemx_team_uniqueid_t 'nvshmemx_team_uniqueid_t'
ctypedef int32_t nvshmem_team_t 'nvshmem_team_t'
ctypedef uint64_t nvshmemx_region_handle_t 'nvshmemx_region_handle_t'
ctypedef struct nvshmemx_region_attrs_t 'nvshmemx_region_attrs_t':
    uint32_t hints
    char reserved[60]
ctypedef nvshmemx_uniqueid_v1 nvshmemx_uniqueid_t 'nvshmemx_uniqueid_t'
ctypedef struct nvshmemx_uniqueid_args_v1 'nvshmemx_uniqueid_args_v1':
    int version
    nvshmemx_uniqueid_v1* id
    int myrank
    int nranks
ctypedef struct nvshmem_team_config_v2 'nvshmem_team_config_v2':
    int version
    int num_contexts
    nvshmemx_team_uniqueid_t uniqueid
    char padding[48]
ctypedef nvshmem_team_t nvshmemx_team_t 'nvshmemx_team_t'
ctypedef nvshmemx_uniqueid_args_v1 nvshmemx_uniqueid_args_t 'nvshmemx_uniqueid_args_t'
ctypedef nvshmem_team_config_v2 nvshmem_team_config_t 'nvshmem_team_config_t'
ctypedef struct nvshmemx_init_args_v2 'nvshmemx_init_args_v2':
    int version
    nvshmemx_uniqueid_args_t uid_args
    int cuda_device_id
    char content[92]
ctypedef nvshmemx_init_args_v2 nvshmemx_init_args_t 'nvshmemx_init_args_t'
ctypedef struct nvshmemx_init_attr_v2 'nvshmemx_init_attr_v2':
    int version
    void* mpi_comm
    nvshmemx_init_args_t args
ctypedef nvshmemx_init_attr_v2 nvshmemx_init_attr_t 'nvshmemx_init_attr_t'


###############################################################################
# Functions
###############################################################################

cdef int nvshmem_barrier(nvshmem_team_t team) except* nogil
cdef void nvshmem_barrier_all() except* nogil
cdef int nvshmemx_init_status() except* nogil
cdef void nvshmem_query_thread(int* provided) except* nogil
cdef int nvshmem_my_pe() except* nogil
cdef int nvshmem_n_pes() except* nogil
cdef void nvshmem_info_get_version(int* major, int* minor) except* nogil
cdef void nvshmem_info_get_name(char* name) except* nogil
cdef void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) except* nogil
cdef void* nvshmem_malloc(size_t size) except* nogil
cdef void* nvshmem_calloc(size_t count, size_t size) except* nogil
cdef void* nvshmem_align(size_t alignment, size_t size) except* nogil
cdef void nvshmem_free(void* ptr) except* nogil
cdef void* nvshmem_ptr(const void* dest, int pe) except* nogil
cdef void* nvshmemx_mc_ptr(nvshmem_team_t team, const void* ptr) except* nogil
cdef void nvshmem_uint_atomic_inc(unsigned int* dest, int pe) except* nogil
cdef void nvshmem_ulong_atomic_inc(unsigned long* dest, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_inc(unsigned long long* dest, int pe) except* nogil
cdef void nvshmem_int32_atomic_inc(int32_t* dest, int pe) except* nogil
cdef void nvshmem_uint32_atomic_inc(uint32_t* dest, int pe) except* nogil
cdef void nvshmem_int64_atomic_inc(int64_t* dest, int pe) except* nogil
cdef void nvshmem_uint64_atomic_inc(uint64_t* dest, int pe) except* nogil
cdef void nvshmem_int_atomic_inc(int* dest, int pe) except* nogil
cdef void nvshmem_long_atomic_inc(long* dest, int pe) except* nogil
cdef void nvshmem_longlong_atomic_inc(long long* dest, int pe) except* nogil
cdef void nvshmem_size_atomic_inc(size_t* dest, int pe) except* nogil
cdef void nvshmem_ptrdiff_atomic_inc(ptrdiff_t* dest, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch_inc(unsigned int* dest, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch_inc(unsigned long* dest, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch_inc(unsigned long long* dest, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch_inc(int32_t* dest, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch_inc(uint32_t* dest, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch_inc(int64_t* dest, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch_inc(uint64_t* dest, int pe) except* nogil
cdef int nvshmem_int_atomic_fetch_inc(int* dest, int pe) except* nogil
cdef long nvshmem_long_atomic_fetch_inc(long* dest, int pe) except* nogil
cdef long long nvshmem_longlong_atomic_fetch_inc(long long* dest, int pe) except* nogil
cdef size_t nvshmem_size_atomic_fetch_inc(size_t* dest, int pe) except* nogil
cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch_inc(ptrdiff_t* dest, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch(const unsigned int* dest, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch(const unsigned long* dest, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch(const unsigned long long* dest, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch(const int32_t* dest, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch(const uint32_t* dest, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch(const int64_t* dest, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch(const uint64_t* dest, int pe) except* nogil
cdef int nvshmem_int_atomic_fetch(const int* dest, int pe) except* nogil
cdef long nvshmem_long_atomic_fetch(const long* dest, int pe) except* nogil
cdef long long nvshmem_longlong_atomic_fetch(const long long* dest, int pe) except* nogil
cdef size_t nvshmem_size_atomic_fetch(const size_t* dest, int pe) except* nogil
cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch(const ptrdiff_t* dest, int pe) except* nogil
cdef float nvshmem_float_atomic_fetch(const float* dest, int pe) except* nogil
cdef double nvshmem_double_atomic_fetch(const double* dest, int pe) except* nogil
cdef void nvshmem_uint_atomic_add(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef void nvshmem_ulong_atomic_add(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_add(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef void nvshmem_int32_atomic_add(int32_t* dest, int32_t value, int pe) except* nogil
cdef void nvshmem_uint32_atomic_add(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef void nvshmem_int64_atomic_add(int64_t* dest, int64_t value, int pe) except* nogil
cdef void nvshmem_uint64_atomic_add(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef void nvshmem_int_atomic_add(int* dest, int value, int pe) except* nogil
cdef void nvshmem_long_atomic_add(long* dest, long value, int pe) except* nogil
cdef void nvshmem_longlong_atomic_add(long long* dest, long long value, int pe) except* nogil
cdef void nvshmem_size_atomic_add(size_t* dest, size_t value, int pe) except* nogil
cdef void nvshmem_ptrdiff_atomic_add(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil
cdef void nvshmemx_float_atomic_add(float* dest, float value, int pe) except* nogil
cdef void nvshmemx_double_atomic_add(double* dest, double value, int pe) except* nogil
cdef float nvshmemx_float_atomic_fetch_add(float* dest, float value, int pe) except* nogil
cdef double nvshmemx_double_atomic_fetch_add(double* dest, double value, int pe) except* nogil
cdef void nvshmem_uint_atomic_set(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef void nvshmem_ulong_atomic_set(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_set(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef void nvshmem_int32_atomic_set(int32_t* dest, int32_t value, int pe) except* nogil
cdef void nvshmem_uint32_atomic_set(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef void nvshmem_int64_atomic_set(int64_t* dest, int64_t value, int pe) except* nogil
cdef void nvshmem_uint64_atomic_set(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef void nvshmem_int_atomic_set(int* dest, int value, int pe) except* nogil
cdef void nvshmem_long_atomic_set(long* dest, long value, int pe) except* nogil
cdef void nvshmem_longlong_atomic_set(long long* dest, long long value, int pe) except* nogil
cdef void nvshmem_size_atomic_set(size_t* dest, size_t value, int pe) except* nogil
cdef void nvshmem_ptrdiff_atomic_set(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil
cdef void nvshmem_float_atomic_set(float* dest, float value, int pe) except* nogil
cdef void nvshmem_double_atomic_set(double* dest, double value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch_add(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch_add(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch_add(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch_add(int32_t* dest, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch_add(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch_add(int64_t* dest, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch_add(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef int nvshmem_int_atomic_fetch_add(int* dest, int value, int pe) except* nogil
cdef long nvshmem_long_atomic_fetch_add(long* dest, long value, int pe) except* nogil
cdef long long nvshmem_longlong_atomic_fetch_add(long long* dest, long long value, int pe) except* nogil
cdef size_t nvshmem_size_atomic_fetch_add(size_t* dest, size_t value, int pe) except* nogil
cdef ptrdiff_t nvshmem_ptrdiff_atomic_fetch_add(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_swap(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_swap(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_swap(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_swap(int32_t* dest, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_swap(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_swap(int64_t* dest, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_swap(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef int nvshmem_int_atomic_swap(int* dest, int value, int pe) except* nogil
cdef long nvshmem_long_atomic_swap(long* dest, long value, int pe) except* nogil
cdef long long nvshmem_longlong_atomic_swap(long long* dest, long long value, int pe) except* nogil
cdef size_t nvshmem_size_atomic_swap(size_t* dest, size_t value, int pe) except* nogil
cdef ptrdiff_t nvshmem_ptrdiff_atomic_swap(ptrdiff_t* dest, ptrdiff_t value, int pe) except* nogil
cdef float nvshmem_float_atomic_swap(float* dest, float value, int pe) except* nogil
cdef double nvshmem_double_atomic_swap(double* dest, double value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_compare_swap(unsigned int* dest, unsigned int cond, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_compare_swap(unsigned long* dest, unsigned long cond, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_compare_swap(unsigned long long* dest, unsigned long long cond, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_compare_swap(int32_t* dest, int32_t cond, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_compare_swap(uint32_t* dest, uint32_t cond, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_compare_swap(int64_t* dest, int64_t cond, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_compare_swap(uint64_t* dest, uint64_t cond, uint64_t value, int pe) except* nogil
cdef int nvshmem_int_atomic_compare_swap(int* dest, int cond, int value, int pe) except* nogil
cdef long nvshmem_long_atomic_compare_swap(long* dest, long cond, long value, int pe) except* nogil
cdef long long nvshmem_longlong_atomic_compare_swap(long long* dest, long long cond, long long value, int pe) except* nogil
cdef size_t nvshmem_size_atomic_compare_swap(size_t* dest, size_t cond, size_t value, int pe) except* nogil
cdef ptrdiff_t nvshmem_ptrdiff_atomic_compare_swap(ptrdiff_t* dest, ptrdiff_t cond, ptrdiff_t value, int pe) except* nogil
cdef void nvshmem_uint_atomic_and(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef void nvshmem_ulong_atomic_and(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_and(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef void nvshmem_int32_atomic_and(int32_t* dest, int32_t value, int pe) except* nogil
cdef void nvshmem_uint32_atomic_and(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef void nvshmem_int64_atomic_and(int64_t* dest, int64_t value, int pe) except* nogil
cdef void nvshmem_uint64_atomic_and(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef void nvshmem_uint_atomic_or(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef void nvshmem_ulong_atomic_or(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_or(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef void nvshmem_int32_atomic_or(int32_t* dest, int32_t value, int pe) except* nogil
cdef void nvshmem_uint32_atomic_or(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef void nvshmem_int64_atomic_or(int64_t* dest, int64_t value, int pe) except* nogil
cdef void nvshmem_uint64_atomic_or(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef void nvshmem_uint_atomic_xor(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef void nvshmem_ulong_atomic_xor(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef void nvshmem_ulonglong_atomic_xor(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef void nvshmem_int32_atomic_xor(int32_t* dest, int32_t value, int pe) except* nogil
cdef void nvshmem_uint32_atomic_xor(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef void nvshmem_int64_atomic_xor(int64_t* dest, int64_t value, int pe) except* nogil
cdef void nvshmem_uint64_atomic_xor(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch_and(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch_and(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch_and(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch_and(int32_t* dest, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch_and(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch_and(int64_t* dest, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch_and(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch_or(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch_or(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch_or(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch_or(int32_t* dest, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch_or(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch_or(int64_t* dest, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch_or(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef unsigned int nvshmem_uint_atomic_fetch_xor(unsigned int* dest, unsigned int value, int pe) except* nogil
cdef unsigned long nvshmem_ulong_atomic_fetch_xor(unsigned long* dest, unsigned long value, int pe) except* nogil
cdef unsigned long long nvshmem_ulonglong_atomic_fetch_xor(unsigned long long* dest, unsigned long long value, int pe) except* nogil
cdef int32_t nvshmem_int32_atomic_fetch_xor(int32_t* dest, int32_t value, int pe) except* nogil
cdef uint32_t nvshmem_uint32_atomic_fetch_xor(uint32_t* dest, uint32_t value, int pe) except* nogil
cdef int64_t nvshmem_int64_atomic_fetch_xor(int64_t* dest, int64_t value, int pe) except* nogil
cdef uint64_t nvshmem_uint64_atomic_fetch_xor(uint64_t* dest, uint64_t value, int pe) except* nogil
cdef uint64_t nvshmem_signal_fetch(uint64_t* sig_addr) except* nogil
cdef int nvshmem_team_my_pe(nvshmem_team_t team) except* nogil
cdef int nvshmem_team_n_pes(nvshmem_team_t team) except* nogil
cdef void nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t* config) except* nogil
cdef int nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) except* nogil
cdef int nvshmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride, int PE_size, const nvshmem_team_config_t* config, long config_mask, nvshmem_team_t* new_team) except* nogil
cdef int nvshmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t* uniqueid) except* nogil
cdef int nvshmemx_team_init(nvshmem_team_t* team, nvshmem_team_config_t* config, long config_mask, int npes, int pe_idx_in_team) except* nogil
cdef int nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange, const nvshmem_team_config_t* xaxis_config, long xaxis_mask, nvshmem_team_t* xaxis_team, const nvshmem_team_config_t* yaxis_config, long yaxis_mask, nvshmem_team_t* yaxis_team) except* nogil
cdef void nvshmem_team_destroy(nvshmem_team_t team) except* nogil
cdef int nvshmemx_bfloat16_alltoall_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_alltoall_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_alltoall_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_alltoall_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_alltoall_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_alltoall_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_alltoall_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_alltoall_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_alltoall_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_alltoall_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_alltoall_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_alltoall_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_alltoall_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_alltoall_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_alltoall_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_alltoall_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_alltoall_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_alltoall_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_alltoall_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil
cdef void nvshmemx_barrier_all_on_stream(cudaStream_t stream) except* nogil
cdef int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) except* nogil
cdef void nvshmemx_sync_all_on_stream(cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_broadcast_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_broadcast_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_broadcast_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_broadcast_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_broadcast_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_broadcast_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_broadcast_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_broadcast_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_broadcast_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_broadcast_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_broadcast_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_broadcast_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_broadcast_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_broadcast_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_broadcast_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_broadcast_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_broadcast_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_broadcast_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_broadcast_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, int PE_root, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_fcollect_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_fcollect_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_fcollect_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_fcollect_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_fcollect_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_fcollect_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_fcollect_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_fcollect_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_fcollect_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_fcollect_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_fcollect_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_fcollect_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_fcollect_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_fcollect_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_fcollect_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_fcollect_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_fcollect_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_fcollect_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_fcollect_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nelem, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_max_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_max_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_max_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_max_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_max_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_max_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_max_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_max_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_max_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_max_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_max_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_max_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_max_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_max_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_max_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_max_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_max_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_max_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_max_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_min_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_min_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_min_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_min_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_min_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_min_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_min_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_min_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_min_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_min_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_min_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_min_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_min_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_min_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_min_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_min_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_min_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_min_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_min_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_sum_reduce_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_sum_reduce_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_sum_reduce_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_sum_reduce_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_sum_reduce_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_sum_reduce_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_sum_reduce_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_sum_reduce_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_sum_reduce_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_sum_reduce_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_sum_reduce_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_sum_reduce_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_sum_reduce_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_sum_reduce_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_sum_reduce_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_sum_reduce_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_sum_reduce_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_sum_reduce_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_sum_reduce_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_max_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_max_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_max_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_max_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_max_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_max_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_max_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_max_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_max_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_max_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_max_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_max_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_max_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_max_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_max_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_max_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_max_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_max_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_max_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_min_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_min_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_min_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_min_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_min_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_min_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_min_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_min_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_min_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_min_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_min_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_min_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_min_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_min_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_min_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_min_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_min_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_min_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_min_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int8_sum_reducescatter_on_stream(nvshmem_team_t team, int8_t* dest, const int8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int16_sum_reducescatter_on_stream(nvshmem_team_t team, int16_t* dest, const int16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int32_sum_reducescatter_on_stream(nvshmem_team_t team, int32_t* dest, const int32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int64_sum_reducescatter_on_stream(nvshmem_team_t team, int64_t* dest, const int64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint8_sum_reducescatter_on_stream(nvshmem_team_t team, uint8_t* dest, const uint8_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint16_sum_reducescatter_on_stream(nvshmem_team_t team, uint16_t* dest, const uint16_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint32_sum_reducescatter_on_stream(nvshmem_team_t team, uint32_t* dest, const uint32_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_uint64_sum_reducescatter_on_stream(nvshmem_team_t team, uint64_t* dest, const uint64_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_size_sum_reducescatter_on_stream(nvshmem_team_t team, size_t* dest, const size_t* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_char_sum_reducescatter_on_stream(nvshmem_team_t team, char* dest, const char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_schar_sum_reducescatter_on_stream(nvshmem_team_t team, signed char* dest, const signed char* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_short_sum_reducescatter_on_stream(nvshmem_team_t team, short* dest, const short* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_int_sum_reducescatter_on_stream(nvshmem_team_t team, int* dest, const int* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_long_sum_reducescatter_on_stream(nvshmem_team_t team, long* dest, const long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_longlong_sum_reducescatter_on_stream(nvshmem_team_t team, long long* dest, const long long* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_bfloat16_sum_reducescatter_on_stream(nvshmem_team_t team, __nv_bfloat16* dest, const __nv_bfloat16* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_half_sum_reducescatter_on_stream(nvshmem_team_t team, half* dest, const half* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_float_sum_reducescatter_on_stream(nvshmem_team_t team, float* dest, const float* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_double_sum_reducescatter_on_stream(nvshmem_team_t team, double* dest, const double* src, size_t nreduce, cudaStream_t stream) except* nogil
cdef int nvshmemx_hostlib_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) except* nogil
cdef void nvshmemx_hostlib_finalize() except* nogil
cdef int nvshmemx_set_attr_uniqueid_args(const int myrank, const int nranks, const nvshmemx_uniqueid_t* uniqueid, nvshmemx_init_attr_t* attr) except* nogil
cdef int nvshmemx_set_attr_mpi_comm_args(void* mpi_comm, nvshmemx_init_attr_t* nvshmem_attr) except* nogil
cdef int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* uniqueid) except* nogil
cdef int nvshmemx_cumodule_init(CUmodule module) except* nogil
cdef int nvshmemx_cumodule_finalize(CUmodule module) except* nogil
cdef void* nvshmemx_buffer_register_symmetric(void* buf_ptr, size_t size, int flags) except* nogil
cdef void* nvshmemx_buffer_register_symmetric_at_preferred_address(void* buf_ptr, size_t size, void* preferred_addr, int flags) except* nogil
cdef int nvshmemx_buffer_unregister_symmetric(void* mmap_ptr, size_t size) except* nogil
cdef int nvshmemx_culibrary_init(CUlibrary library) except* nogil
cdef int nvshmemx_culibrary_finalize(CUlibrary library) except* nogil
cdef void nvshmemx_putmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_putmem_signal_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_putmem_signal_nbi_on_stream(void* dest, const void* source, size_t bytes, uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_putmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_getmem_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_getmem_nbi_on_stream(void* dest, const void* source, size_t bytes, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_quiet_on_stream(cudaStream_t cstrm) except* nogil
cdef void nvshmemx_flush_on_stream(cudaStream_t cstrm) except* nogil
cdef void nvshmemx_signal_op_on_stream(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe, cudaStream_t cstrm) except* nogil
cdef void nvshmemx_signal_wait_until_on_stream(uint64_t* sig_addr, int cmp, uint64_t cmp_value, cudaStream_t cstream) except* nogil