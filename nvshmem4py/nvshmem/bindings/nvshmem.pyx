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

# Simple wrapper for nvshmemx_team_uniqueid_t typedef
# This is required because Cybind doesn't generate wrappers for classes that are raw typedefs.
cdef class team_uniqueid:
    """Simple wrapper for nvshmemx_team_uniqueid_t (typedef uint64_t)."""
    
    cdef:
        readonly object _data
    
    def __init__(self, value: int = 0):
        """Initialize with an optional value."""
        arr = _numpy.empty(1, dtype=_numpy.uint64)
        self._data = arr.view(_numpy.recarray)
        self._data[0] = value
    
    @property
    def ptr(self):
        """Get the pointer address to the data as Python :py:`int`."""
        return self._data.ctypes.data
    
    @property
    def value(self) -> int:
        """Get the value as Python int."""
        return int(self._data[0])
    
    @value.setter
    def value(self, val: int):
        """Set the value."""
        self._data[0] = val
    
    def __str__(self) -> str:
        return f"team_uniqueid({self.value})"
    
    def __repr__(self) -> str:
        return f"team_uniqueid(value={self.value})"

cdef class TeamUniqueId(team_uniqueid): pass

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


# POD wrapper for nvshmem_team_config_t. cybind can't generate this automatically
# because it doesn't fully support nested structs (https://gitlab-master.nvidia.com/leof/cybind/-/issues/67).
# The nested structure is made opaque.
# TODO: remove this once cybind supports nested structs.

team_config_dtype = _numpy.dtype([
    ("version", _numpy.int32, ),
    ("num_contexts", _numpy.int32, ),
    ("uniqueid", _numpy.uint64, ),
    ("padding", _numpy.int8, (48,)),  # TEAM_CONFIG_V2_PADDING = 48
    ], align=True)


cdef class TeamConfig:

    cdef:
        readonly object _data

    def __init__(self):
        arr = _numpy.empty(1, dtype=team_config_dtype)
        self._data = arr.view(_numpy.recarray)
        assert self._data.itemsize == sizeof(nvshmem_team_config_t), \
            f"itemsize {self._data.itemsize} mismatches struct size {sizeof(nvshmem_team_config_t)}"

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
    def num_contexts(self):
        """num_contexts (~_numpy.int32): """
        return int(self._data.num_contexts[0])

    @num_contexts.setter
    def num_contexts(self, val):
        self._data.num_contexts = val

    @property
    def uniqueid(self):
        """uniqueid (~_numpy.uint64): """
        return int(self._data.uniqueid[0])

    @uniqueid.setter
    def uniqueid(self, val):
        self._data.uniqueid = val


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

class Thread_support(_IntEnum):
    """See `nvshmemx_thread_support_t`."""
    THREAD_SINGLE = NVSHMEM_THREAD_SINGLE
    THREAD_FUNNELED = NVSHMEM_THREAD_FUNNELED
    THREAD_SERIALIZED = NVSHMEM_THREAD_SERIALIZED
    THREAD_MULTIPLE = NVSHMEM_THREAD_MULTIPLE
    THREAD_TYPE_SENTINEL = NVSHMEM_THREAD_TYPE_SENTINEL

class Proxy_status(_IntEnum):
    """See `nvshmemx_proxy_status_t`."""
    _PROXY_GLOBAL_EXIT_NOT_REQUESTED = PROXY_GLOBAL_EXIT_NOT_REQUESTED
    _PROXY_GLOBAL_EXIT_INIT = PROXY_GLOBAL_EXIT_INIT
    _PROXY_GLOBAL_EXIT_REQUESTED = PROXY_GLOBAL_EXIT_REQUESTED
    _PROXY_GLOBAL_EXIT_FINISHED = PROXY_GLOBAL_EXIT_FINISHED
    _PROXY_GLOBAL_EXIT_MAX_STATE = PROXY_GLOBAL_EXIT_MAX_STATE

class Init_status(_IntEnum):
    """See `nvshmemx_init_status_t`."""
    STATUS_NOT_INITIALIZED = NVSHMEM_STATUS_NOT_INITIALIZED
    STATUS_IS_BOOTSTRAPPED = NVSHMEM_STATUS_IS_BOOTSTRAPPED
    STATUS_IS_INITIALIZED = NVSHMEM_STATUS_IS_INITIALIZED
    STATUS_LIMITED_MPG = NVSHMEM_STATUS_LIMITED_MPG
    STATUS_FULL_MPG = NVSHMEM_STATUS_FULL_MPG
    STATUS_INVALID = NVSHMEM_STATUS_INVALID

class Qp_handle_index(_IntEnum):
    """See `nvshmemx_qp_handle_index_t`."""
    QP_HOST = NVSHMEMX_QP_HOST
    QP_DEFAULT = NVSHMEMX_QP_DEFAULT
    QP_ANY = NVSHMEMX_QP_ANY
    QP_ALL = NVSHMEMX_QP_ALL

class Pe_index(_IntEnum):
    """See `nvshmem_pe_index_t`."""
    PE_INVALID = NVSHMEM_PE_INVALID
    PE_ANY = NVSHMEMX_PE_ANY
    PE_ALL = NVSHMEMX_PE_ALL

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

cpdef barrier(int32_t team):
    with nogil:
        status = nvshmem_barrier(<nvshmem_team_t>team)
    check_status(status)


cpdef void barrier_all() except*:
    nvshmem_barrier_all()


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


cpdef void team_get_config(int32_t team, intptr_t config) except*:
    nvshmem_team_get_config(<nvshmem_team_t>team, <nvshmem_team_config_t*>config)


cpdef team_translate_pe(int32_t src_team, int src_pe, int32_t dest_team):
    with nogil:
        status = nvshmem_team_translate_pe(<nvshmem_team_t>src_team, src_pe, <nvshmem_team_t>dest_team)
    check_status(status)


cpdef team_split_strided(int32_t parent_team, int pe_start, int pe_stride, int pe_size, intptr_t config, long config_mask, intptr_t new_team):
    with nogil:
        status = nvshmem_team_split_strided(<nvshmem_team_t>parent_team, pe_start, pe_stride, pe_size, <const nvshmem_team_config_t*>config, config_mask, <nvshmem_team_t*>new_team)
    check_status(status)


cpdef team_get_uniqueid(intptr_t uniqueid):
    with nogil:
        status = nvshmemx_team_get_uniqueid(<nvshmemx_team_uniqueid_t*>uniqueid)
    check_status(status)


cpdef team_init(intptr_t team, intptr_t config, long config_mask, int npes, int pe_idx_in_team):
    with nogil:
        status = nvshmemx_team_init(<nvshmem_team_t*>team, <nvshmem_team_config_t*>config, config_mask, npes, pe_idx_in_team)
    check_status(status)


cpdef team_split_2d(int32_t parent_team, int xrange, intptr_t xaxis_config, long xaxis_mask, intptr_t xaxis_team, intptr_t yaxis_config, long yaxis_mask, intptr_t yaxis_team):
    with nogil:
        status = nvshmem_team_split_2d(<nvshmem_team_t>parent_team, xrange, <const nvshmem_team_config_t*>xaxis_config, xaxis_mask, <nvshmem_team_t*>xaxis_team, <const nvshmem_team_config_t*>yaxis_config, yaxis_mask, <nvshmem_team_t*>yaxis_team)
    check_status(status)


cpdef void team_destroy(int32_t team) except*:
    nvshmem_team_destroy(<nvshmem_team_t>team)


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


cpdef void barrier_all_on_stream(intptr_t stream) except*:
    nvshmemx_barrier_all_on_stream(<Stream>stream)


cpdef int team_sync_on_stream(int32_t team, intptr_t stream) except? 0:
    return nvshmemx_team_sync_on_stream(<nvshmem_team_t>team, <Stream>stream)


cpdef void sync_all_on_stream(intptr_t stream) except*:
    nvshmemx_sync_all_on_stream(<Stream>stream)


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


cpdef int cumodule_init(intptr_t module) except? -1:
    return nvshmemx_cumodule_init(<void*>module)


cpdef int cumodule_finalize(intptr_t module) except? -1:
    return nvshmemx_cumodule_finalize(<void*>module)


cpdef intptr_t buffer_register_symmetric(intptr_t buf_ptr, size_t size, int flags) except? 0:
    return <intptr_t>nvshmemx_buffer_register_symmetric(<void*>buf_ptr, size, flags)


cpdef int buffer_unregister_symmetric(intptr_t mmap_ptr, size_t size) except? 0:
    return nvshmemx_buffer_unregister_symmetric(<void*>mmap_ptr, size)


cpdef int culibrary_init(intptr_t library) except? -1:
    return nvshmemx_culibrary_init(<void*>library)


cpdef int culibrary_finalize(intptr_t library) except? -1:
    return nvshmemx_culibrary_finalize(<void*>library)


cpdef void putmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*:
    nvshmemx_putmem_on_stream(<void*>dest, <const void*>source, bytes, pe, <Stream>cstrm)


cpdef void putmem_signal_on_stream(intptr_t dest, intptr_t source, size_t bytes, intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*:
    nvshmemx_putmem_signal_on_stream(<void*>dest, <const void*>source, bytes, <uint64_t*>sig_addr, signal, sig_op, pe, <Stream>cstrm)


cpdef void getmem_on_stream(intptr_t dest, intptr_t source, size_t bytes, int pe, intptr_t cstrm) except*:
    nvshmemx_getmem_on_stream(<void*>dest, <const void*>source, bytes, pe, <Stream>cstrm)


cpdef void quiet_on_stream(intptr_t cstrm) except*:
    nvshmemx_quiet_on_stream(<Stream>cstrm)


cpdef void signal_op_on_stream(intptr_t sig_addr, uint64_t signal, int sig_op, int pe, intptr_t cstrm) except*:
    nvshmemx_signal_op_on_stream(<uint64_t*>sig_addr, signal, sig_op, pe, <Stream>cstrm)


cpdef void signal_wait_until_on_stream(intptr_t sig_addr, int cmp, uint64_t cmp_value, intptr_t cstream) except*:
    nvshmemx_signal_wait_until_on_stream(<uint64_t*>sig_addr, cmp, cmp_value, <Stream>cstream)
