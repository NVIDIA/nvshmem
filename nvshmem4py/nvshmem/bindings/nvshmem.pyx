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

cimport cython  # NOQA
from cpython cimport buffer as _buffer
from cpython.memoryview cimport PyMemoryView_FromMemory
from libc.stdint cimport (
    int8_t,  uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t,
    intptr_t, uintptr_t
)


from enum import IntEnum as _IntEnum

import numpy as _numpy


###############################################################################
# POD
###############################################################################

uniqueid_dtype = _numpy.dtype([
    ("version", _numpy.int32, ),
    ("internal", _numpy.int8, (124,)),
    ], align=True)


cdef class uniqueid:
    """Empty-initialize an array of `nvshmemx_uniqueid_v1`.

    The resulting object is of length `size` and of dtype `uniqueid_dtype`.
    If default-constructed, the instance represents a single struct.

    Args:
        size (int): number of structs, default=1.


    .. seealso:: `nvshmemx_uniqueid_v1`
    """
    cdef:
        readonly object _data

    def __init__(self, size=1):
        arr = _numpy.empty(size, dtype=uniqueid_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(nvshmemx_uniqueid_v1), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(nvshmemx_uniqueid_v1)}"

    def __repr__(self):
        if self._data.size > 1:
            return f"<{__name__}.uniqueid_Array_{self._data.size} object at {hex(id(self))}>"
        else:
            return f"<{__name__}.uniqueid object at {hex(id(self))}>"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :py:`int`."""
        return self._data.ctypes.data

    def __int__(self):
        if self._data.size > 1:
            raise TypeError("int() argument must be a bytes-like object of size 1. "
                            "To get the pointer address of an array, use .ptr")
        return self._data.ctypes.data

    def __len__(self):
        return self._data.size

    def __eq__(self, other):
        if not isinstance(other, uniqueid):
            return False
        if self._data.size != other._data.size:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        return bool((self._data == other._data).all())

    @property
    def version(self):
        """version (~_numpy.int32): """
        if self._data.size == 1:
            return int(self._data.version[0])
        return self._data.version

    @version.setter
    def version(self, val):
        self._data.version = val

    def __getitem__(self, key):
        if isinstance(key, int):
            size = self._data.size
            if key >= size or key <= -(size+1):
                raise IndexError("index is out of bounds")
            if key < 0:
                key += size
            return uniqueid.from_data(self._data[key:key+1])
        out = self._data[key]
        if isinstance(out, _numpy.recarray) and out.dtype == uniqueid_dtype:
            return uniqueid.from_data(out)
        return out

    def __setitem__(self, key, val):
        self._data[key] = val

    @staticmethod
    def from_data(data):
        """Create an uniqueid instance wrapping the given NumPy array.

        Args:
            data (_numpy.ndarray): a 1D array of dtype `uniqueid_dtype` holding the data.
        """
        cdef uniqueid obj = uniqueid.__new__(uniqueid)
        if not isinstance(data, (_numpy.ndarray, _numpy.recarray)):
            raise TypeError("data argument must be a NumPy ndarray")
        if data.ndim != 1:
            raise ValueError("data array must be 1D")
        if data.dtype != uniqueid_dtype:
            raise ValueError("data array must be of dtype uniqueid_dtype")
        obj._data = data.view(_numpy.recarray)

        return obj

    @staticmethod
    def from_ptr(intptr_t ptr, size_t size=1, bint readonly=False):
        """Create an uniqueid instance wrapping the given pointer.

        Args:
            ptr (intptr_t): pointer address as Python :py:`int` to the data.
            size (int): number of structs, default=1.
            readonly (bool): whether the data is read-only (to the user). default is `False`.
        """
        if ptr == 0:
            raise ValueError("ptr must not be null (0)")
        cdef uniqueid obj = uniqueid.__new__(uniqueid)
        cdef flag = _buffer.PyBUF_READ if readonly else _buffer.PyBUF_WRITE
        cdef object buf = PyMemoryView_FromMemory(
            <char*>ptr, sizeof(nvshmemx_uniqueid_v1) * size, flag)
        data = _numpy.ndarray((size,), buffer=buf,
                              dtype=uniqueid_dtype)
        obj._data = data.view(_numpy.recarray)

        return obj


cdef class UniqueId(uniqueid): pass

# POD wrapper for nvshmemx_init_attr_t. cybind can't generate this automatically
# because it doesn't fully support nested structs (https://gitlab-master.nvidia.com/leof/cybind/-/issues/67).
# The nested structure is made opaque.
# TODO: remove this once cybind supports nested structs.

init_attr_dtype = _numpy.dtype([
    ("version", _numpy.int32, ),
    ("mpi_comm", _numpy.intp, ),
    ("args", _numpy.int8, (sizeof(nvshmemx_init_args_t),)),  # opaque
    ], align=True)


cdef class InitAttr:

    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=init_attr_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(nvshmemx_init_attr_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(nvshmemx_init_attr_t)}"

    @property
    def ptr(self):
        """Get the pointer address to the data as Python :py:`int`."""
        return self._data.ctypes.data

    @property
    def version(self):
        """version (~_numpy.int32): """
        return int(self._data.version[0])

    @version.setter
    def version(self, val):
        self._data.version = val

    @property
    def mpi_comm(self):
        """mpi_comm (~_numpy.intp): """
        return int(self._data.mpi_comm[0])

    @mpi_comm.setter
    def mpi_comm(self, val):
        self._data.mpi_comm = val


###############################################################################
# Enum
###############################################################################

class Signal_op(_IntEnum):
    """See `nvshmemx_signal_op_t`."""
    SIGNAL_SET = NVSHMEM_SIGNAL_SET
    SIGNAL_ADD = NVSHMEM_SIGNAL_ADD

class Cmp_type(_IntEnum):
    """See `nvshmemx_cmp_type_t`."""
    CMP_EQ = NVSHMEM_CMP_EQ
    CMP_NE = NVSHMEM_CMP_NE
    CMP_GT = NVSHMEM_CMP_GT
    CMP_LE = NVSHMEM_CMP_LE
    CMP_LT = NVSHMEM_CMP_LT
    CMP_GE = NVSHMEM_CMP_GE
    CMP_SENTINEL = NVSHMEM_CMP_SENTINEL

class Init_status(_IntEnum):
    """See `nvshmemx_init_status_t`."""
    STATUS_NOT_INITIALIZED = NVSHMEM_STATUS_NOT_INITIALIZED
    STATUS_IS_BOOTSTRAPPED = NVSHMEM_STATUS_IS_BOOTSTRAPPED
    STATUS_IS_INITIALIZED = NVSHMEM_STATUS_IS_INITIALIZED
    STATUS_LIMITED_MPG = NVSHMEM_STATUS_LIMITED_MPG
    STATUS_FULL_MPG = NVSHMEM_STATUS_FULL_MPG
    STATUS_INVALID = NVSHMEM_STATUS_INVALID

class Team_id(_IntEnum):
    """See `nvshmem_team_id_t`."""
    TEAM_INVALID = NVSHMEM_TEAM_INVALID
    TEAM_WORLD = NVSHMEM_TEAM_WORLD
    TEAM_WORLD_INDEX = NVSHMEM_TEAM_WORLD_INDEX
    TEAM_SHARED = NVSHMEM_TEAM_SHARED
    TEAM_SHARED_INDEX = NVSHMEM_TEAM_SHARED_INDEX
    TEAM_NODE = NVSHMEMX_TEAM_NODE
    TEAM_NODE_INDEX = NVSHMEM_TEAM_NODE_INDEX
    TEAM_SAME_MYPE_NODE = NVSHMEMX_TEAM_SAME_MYPE_NODE
    TEAM_SAME_MYPE_NODE_INDEX = NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX
    TEAM_SAME_GPU = NVSHMEMI_TEAM_SAME_GPU
    TEAM_SAME_GPU_INDEX = NVSHMEM_TEAM_SAME_GPU_INDEX
    TEAM_GPU_LEADERS = NVSHMEMI_TEAM_GPU_LEADERS
    TEAM_GPU_LEADERS_INDEX = NVSHMEM_TEAM_GPU_LEADERS_INDEX
    TEAMS_MIN = NVSHMEM_TEAMS_MIN
    TEAM_INDEX_MAX = NVSHMEM_TEAM_INDEX_MAX

class Status(_IntEnum):
    """See `nvshmemx_status`."""
    SUCCESS = NVSHMEMX_SUCCESS
    ERROR_INVALID_VALUE = NVSHMEMX_ERROR_INVALID_VALUE
    ERROR_OUT_OF_MEMORY = NVSHMEMX_ERROR_OUT_OF_MEMORY
    ERROR_NOT_SUPPORTED = NVSHMEMX_ERROR_NOT_SUPPORTED
    ERROR_SYMMETRY = NVSHMEMX_ERROR_SYMMETRY
    ERROR_GPU_NOT_SELECTED = NVSHMEMX_ERROR_GPU_NOT_SELECTED
    ERROR_COLLECTIVE_LAUNCH_FAILED = NVSHMEMX_ERROR_COLLECTIVE_LAUNCH_FAILED
    ERROR_INTERNAL = NVSHMEMX_ERROR_INTERNAL
    ERROR_SENTINEL = NVSHMEMX_ERROR_SENTINEL

class Flags(_IntEnum):
    """See `flags`."""
    INIT_THREAD_PES = NVSHMEMX_INIT_THREAD_PES
    INIT_WITH_MPI_COMM = NVSHMEMX_INIT_WITH_MPI_COMM
    INIT_WITH_SHMEM = NVSHMEMX_INIT_WITH_SHMEM
    INIT_WITH_UNIQUEID = NVSHMEMX_INIT_WITH_UNIQUEID
    INIT_MAX = NVSHMEMX_INIT_MAX


###############################################################################
# Error handling
###############################################################################

class NVSHMEMError(Exception):

    def __init__(self, status):
        self.status = status
        cdef str err = f"Status code {status}"
        super(NVSHMEMError, self).__init__(err)

    def __reduce__(self):
        return (type(self), (self.status,))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise NVSHMEMError(status)


###############################################################################
# Wrapper functions
###############################################################################

cpdef int init_status() except? 0:
    return nvshmemx_init_status()


cpdef int my_pe() except? -1:
    return nvshmem_my_pe()


cpdef int n_pes() except? -1:
    return nvshmem_n_pes()


cpdef void info_get_version(intptr_t major, intptr_t minor) except*:
    nvshmem_info_get_version(<int*>major, <int*>minor)


cpdef void vendor_get_version_info(intptr_t major, intptr_t minor, intptr_t patch) except*:
    nvshmemx_vendor_get_version_info(<int*>major, <int*>minor, <int*>patch)


cpdef intptr_t malloc(size_t size) except? 0:
    return <intptr_t>nvshmem_malloc(size)


cpdef intptr_t calloc(size_t count, size_t size) except? 0:
    return <intptr_t>nvshmem_calloc(count, size)


cpdef intptr_t align(size_t count, size_t size) except? 0:
    return <intptr_t>nvshmem_align(count, size)


cpdef void free(intptr_t ptr) except*:
    nvshmem_free(<void*>ptr)


cpdef intptr_t ptr(intptr_t dest, int pe) except? 0:
    return <intptr_t>nvshmem_ptr(<const void*>dest, pe)


cpdef intptr_t mc_ptr(int32_t team, intptr_t ptr) except? 0:
    return <intptr_t>nvshmemx_mc_ptr(<nvshmem_team_t>team, <const void*>ptr)


cpdef int team_my_pe(int32_t team) except? -1:
    return nvshmem_team_my_pe(<nvshmem_team_t>team)


cpdef int team_n_pes(int32_t team) except? -1:
    return nvshmem_team_n_pes(<nvshmem_team_t>team)


cpdef barrier(int32_t team):
    with nogil:
        status = nvshmem_barrier(<nvshmem_team_t>team)
    check_status(status)


cpdef void barrier_all() except*:
    nvshmem_barrier_all()


cpdef bfloat16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_alltoall_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nelem, <Stream>stream)
    check_status(status)


cpdef half_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_half_alltoall_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nelem, <Stream>stream)
    check_status(status)


cpdef float_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_float_alltoall_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nelem, <Stream>stream)
    check_status(status)


cpdef double_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_double_alltoall_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nelem, <Stream>stream)
    check_status(status)


cpdef char_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_char_alltoall_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nelem, <Stream>stream)
    check_status(status)


cpdef short_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_short_alltoall_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nelem, <Stream>stream)
    check_status(status)


cpdef schar_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_alltoall_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int_alltoall_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nelem, <Stream>stream)
    check_status(status)


cpdef long_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_long_alltoall_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nelem, <Stream>stream)
    check_status(status)


cpdef longlong_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_alltoall_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int8_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_alltoall_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_alltoall_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int32_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_alltoall_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int64_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_alltoall_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint8_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_alltoall_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint16_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_alltoall_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint32_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_alltoall_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint64_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_alltoall_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef size_alltoall_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_size_alltoall_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef barrier_on_stream(int32_t team, intptr_t stream):
    with nogil:
        status = nvshmemx_barrier_on_stream(<nvshmem_team_t>team, <Stream>stream)
    check_status(status)


cpdef int team_sync_on_stream(int32_t team, intptr_t stream) except? 0:
    return nvshmemx_team_sync_on_stream(<nvshmem_team_t>team, <Stream>stream)


cpdef bfloat16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_broadcast_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef half_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_half_broadcast_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef float_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_float_broadcast_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef double_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_double_broadcast_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef char_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_char_broadcast_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef short_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_short_broadcast_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef schar_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_broadcast_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef int_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_int_broadcast_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef long_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_long_broadcast_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef longlong_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_broadcast_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef int8_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_broadcast_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef int16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_broadcast_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef int32_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_broadcast_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef int64_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_broadcast_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef uint8_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_broadcast_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef uint16_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_broadcast_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef uint32_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_broadcast_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef uint64_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_broadcast_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef size_broadcast_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, int pe_root, intptr_t stream):
    with nogil:
        status = nvshmemx_size_broadcast_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nelem, pe_root, <Stream>stream)
    check_status(status)


cpdef bfloat16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_fcollect_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nelem, <Stream>stream)
    check_status(status)


cpdef half_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_half_fcollect_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nelem, <Stream>stream)
    check_status(status)


cpdef float_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_float_fcollect_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nelem, <Stream>stream)
    check_status(status)


cpdef double_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_double_fcollect_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nelem, <Stream>stream)
    check_status(status)


cpdef char_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_char_fcollect_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nelem, <Stream>stream)
    check_status(status)


cpdef short_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_short_fcollect_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nelem, <Stream>stream)
    check_status(status)


cpdef schar_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_fcollect_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int_fcollect_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nelem, <Stream>stream)
    check_status(status)


cpdef long_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_long_fcollect_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nelem, <Stream>stream)
    check_status(status)


cpdef longlong_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_fcollect_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int8_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_fcollect_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_fcollect_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int32_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_fcollect_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int64_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_fcollect_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint8_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_fcollect_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint16_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_fcollect_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint32_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_fcollect_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef uint64_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_fcollect_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef size_fcollect_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nelem, intptr_t stream):
    with nogil:
        status = nvshmemx_size_fcollect_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nelem, <Stream>stream)
    check_status(status)


cpdef int8_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_max_reduce_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_max_reduce_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_max_reduce_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_max_reduce_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_max_reduce_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_max_reduce_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_max_reduce_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_max_reduce_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_max_reduce_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_max_reduce_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_max_reduce_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_max_reduce_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_max_reduce_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_max_reduce_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_max_reduce_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_max_reduce_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_max_reduce_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_max_reduce_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_max_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_max_reduce_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int8_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_min_reduce_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_min_reduce_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_min_reduce_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_min_reduce_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_min_reduce_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_min_reduce_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_min_reduce_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_min_reduce_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_min_reduce_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_min_reduce_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_min_reduce_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_min_reduce_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_min_reduce_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_min_reduce_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_min_reduce_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_min_reduce_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_min_reduce_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_min_reduce_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_min_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_min_reduce_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int8_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_sum_reduce_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_sum_reduce_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_sum_reduce_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_sum_reduce_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_sum_reduce_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_sum_reduce_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_sum_reduce_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_sum_reduce_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_sum_reduce_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_sum_reduce_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_sum_reduce_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_sum_reduce_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_sum_reduce_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_sum_reduce_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_sum_reduce_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_sum_reduce_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_sum_reduce_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_sum_reduce_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_sum_reduce_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_sum_reduce_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int8_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_max_reducescatter_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_max_reducescatter_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_max_reducescatter_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_max_reducescatter_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_max_reducescatter_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_max_reducescatter_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_max_reducescatter_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_max_reducescatter_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_max_reducescatter_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_max_reducescatter_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_max_reducescatter_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_max_reducescatter_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_max_reducescatter_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_max_reducescatter_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_max_reducescatter_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_max_reducescatter_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_max_reducescatter_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_max_reducescatter_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_max_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_max_reducescatter_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int8_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_min_reducescatter_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_min_reducescatter_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_min_reducescatter_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_min_reducescatter_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_min_reducescatter_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_min_reducescatter_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_min_reducescatter_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_min_reducescatter_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_min_reducescatter_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_min_reducescatter_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_min_reducescatter_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_min_reducescatter_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_min_reducescatter_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_min_reducescatter_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_min_reducescatter_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_min_reducescatter_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_min_reducescatter_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_min_reducescatter_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_min_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_min_reducescatter_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int8_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int8_sum_reducescatter_on_stream(<nvshmem_team_t>team, <int8_t*>dest, <const int8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int16_sum_reducescatter_on_stream(<nvshmem_team_t>team, <int16_t*>dest, <const int16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int32_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int32_sum_reducescatter_on_stream(<nvshmem_team_t>team, <int32_t*>dest, <const int32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int64_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int64_sum_reducescatter_on_stream(<nvshmem_team_t>team, <int64_t*>dest, <const int64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint8_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint8_sum_reducescatter_on_stream(<nvshmem_team_t>team, <uint8_t*>dest, <const uint8_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint16_sum_reducescatter_on_stream(<nvshmem_team_t>team, <uint16_t*>dest, <const uint16_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint32_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint32_sum_reducescatter_on_stream(<nvshmem_team_t>team, <uint32_t*>dest, <const uint32_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef uint64_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_uint64_sum_reducescatter_on_stream(<nvshmem_team_t>team, <uint64_t*>dest, <const uint64_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef size_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_size_sum_reducescatter_on_stream(<nvshmem_team_t>team, <size_t*>dest, <const size_t*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef char_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_char_sum_reducescatter_on_stream(<nvshmem_team_t>team, <char*>dest, <const char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef schar_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_schar_sum_reducescatter_on_stream(<nvshmem_team_t>team, <signed char*>dest, <const signed char*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef short_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_short_sum_reducescatter_on_stream(<nvshmem_team_t>team, <short*>dest, <const short*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef int_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_int_sum_reducescatter_on_stream(<nvshmem_team_t>team, <int*>dest, <const int*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef long_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_long_sum_reducescatter_on_stream(<nvshmem_team_t>team, <long*>dest, <const long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef longlong_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_longlong_sum_reducescatter_on_stream(<nvshmem_team_t>team, <long long*>dest, <const long long*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef bfloat16_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_bfloat16_sum_reducescatter_on_stream(<nvshmem_team_t>team, <__nv_bfloat16*>dest, <const __nv_bfloat16*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef half_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_half_sum_reducescatter_on_stream(<nvshmem_team_t>team, <half*>dest, <const half*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef float_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_float_sum_reducescatter_on_stream(<nvshmem_team_t>team, <float*>dest, <const float*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef double_sum_reducescatter_on_stream(int32_t team, intptr_t dest, intptr_t src, size_t nreduce, intptr_t stream):
    with nogil:
        status = nvshmemx_double_sum_reducescatter_on_stream(<nvshmem_team_t>team, <double*>dest, <const double*>src, nreduce, <Stream>stream)
    check_status(status)


cpdef hostlib_init_attr(unsigned int flags, intptr_t attr):
    with nogil:
        status = nvshmemx_hostlib_init_attr(flags, <nvshmemx_init_attr_t*>attr)
    check_status(status)


cpdef void hostlib_finalize() except*:
    nvshmemx_hostlib_finalize()


cpdef set_attr_uniqueid_args(int myrank, int nranks, intptr_t uniqueid, intptr_t attr):
    with nogil:
        status = nvshmemx_set_attr_uniqueid_args(<const int>myrank, <const int>nranks, <const nvshmemx_uniqueid_t*>uniqueid, <nvshmemx_init_attr_t*>attr)
    check_status(status)


cpdef set_attr_mpi_comm_args(intptr_t mpi_comm, intptr_t nvshmem_attr):
    with nogil:
        status = nvshmemx_set_attr_mpi_comm_args(<void*>mpi_comm, <nvshmemx_init_attr_t*>nvshmem_attr)
    check_status(status)


cpdef get_uniqueid(intptr_t uniqueid):
    with nogil:
        status = nvshmemx_get_uniqueid(<nvshmemx_uniqueid_t*>uniqueid)
    check_status(status)


cpdef int cumodule_init(intptr_t module) except? 0:
    return nvshmemx_cumodule_init(<void*>module)


cpdef int cumodule_finalize(intptr_t module) except? 0:
    return nvshmemx_cumodule_finalize(<void*>module)


cpdef void putmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*:
    nvshmemx_putmem_on_stream(<void*>dest, <const void*>source, bytes, pe, <Stream>cstrm)


cpdef void putmem_signal_on_stream(intptr_t dest, intptr_t source, size_t bytes, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*:
    nvshmemx_putmem_signal_on_stream(<void*>dest, <const void*>source, bytes, <uint64_t*>sig_addr, signal, sig_op, pe, <Stream>cstrm)


cpdef void getmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*:
    nvshmemx_getmem_on_stream(<void*>dest, <const void*>source, bytes, pe, <Stream>cstrm)


cpdef void quiet_on_stream(intptr_t cstrm) except*:
    nvshmemx_quiet_on_stream(<Stream>cstrm)


cpdef void signal_wait_until_on_stream(intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t cstream) except*:
    nvshmemx_signal_wait_until_on_stream(<uint64_t*>sig_addr, cmp, cmp_value, <Stream>cstream)
