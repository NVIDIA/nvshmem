# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See License.txt for license information

import logging
import os
import ctypes

import nvshmem.core
import nvshmem.bindings as bindings
from nvshmem.core.nvshmem_types import *
import nvshmem.core.utils as utils
import nvshmem.core.memory as memory
from nvshmem import __version__
from nvshmem.core._internal_tracking import _mr_references, _cached_device, _debug_mode, InternalInitStatus

from cuda.core.experimental._memory import Buffer, MemoryResource
from cuda.core.experimental import Device, system

import numpy as np

try:
    import mpi4py.MPI as mpi
    from mpi4py.MPI import Comm
    _mpi4py_enabled = True
except ImportError:
    Comm = None
    _mpi4py_enabled = False

__all__ = ['get_unique_id', 'init', 'finalize', 'get_version']

logger = logging.getLogger("nvshmem")

def get_version() -> Version:
    """
    Get the NVSHMEM4Py version
   
    Returns an object of type nvshmem.core.Version which is a Python class
    This class contains several strings which represent versions related to NVSHMEM

    ``Version.openshmem_spec_version`` is the OpenSHMEM Spec that this NVSHMEM was built against

    ``Version.nvshmem4py_version`` is the version of the NVSHMEM4Py python library

    ``Version.libnvshmem_version`` is the version of NVSHMEM library that this package has opened
    """
    # Spec version
    spec_major = ctypes.c_int()
    spec_minor = ctypes.c_int()
    bindings.info_get_version(ctypes.addressof(spec_major), ctypes.addressof(spec_minor))

    # libnvshmem version
    lib_major = ctypes.c_int()
    lib_minor = ctypes.c_int()
    lib_patch = ctypes.c_int()
    bindings.vendor_get_version_info(ctypes.addressof(lib_major), ctypes.addressof(lib_minor), ctypes.addressof(lib_patch))


    return Version(openshmem_spec_version=f"{int(spec_major.value)}.{int(spec_minor.value)}",
                   nvshmem4py_version=__version__,
                   libnvshmem_version=f"{int(lib_major.value)}.{int(lib_minor.value)}.{lib_patch.value}") 

def get_unique_id(empty=False) -> bindings.uniqueid:
    """
    Retrieve or create a unique ID used for UID-based NVSHMEM initialization.

    This function wraps the underlying NVSHMEM binding for obtaining a unique ID
    required when using the ``uid`` initializer method. Only a single rank (typically rank 0)
    should call this with `empty=False` to generate the ID. Other ranks should call it
    with ``empty=True`` and receive the ID through a user-defined communication mechanism
    (e.g., MPI broadcast or socket transfer).

    Args:
        empty (bool): If True, returns an empty (uninitialized) unique ID structure.
            If False, calls the underlying NVSHMEM function to generate a valid unique ID.

    Returns:
        ``UniqueID``: A `UniqueID` object containing the generated or empty NVSHMEM unique ID.

    Raises:
       ``NvshmemError``: If retrieving the unique ID from NVSHMEM fails.

    Example:
        >>> if rank == 0:
        ...     uid = nvshmem.core.get_unique_id()
        ... else:
        ...     uid = nvshmem.core.get_unique_id(empty=True)
        ...
        >>> nvshmem.core.init(uid=uid, rank=rank, nranks=size, initializer_method="uid")
    """
    unique_id = bindings.uniqueid()
    if empty:
        return unique_id
    # This function is in TRANSPARENT mode
    # The binding will check status
    bindings.get_uniqueid(unique_id.ptr)
    return unique_id

def init(device: Device=None, uid: bindings.uniqueid=None, rank: int=None, nranks: int=None, mpi_comm: Comm=None, initializer_method: str="") -> None:
    """
    Initialize the NVSHMEM runtime with either MPI or UID-based bootstrapping.

    Args:
        - device (``cuda.core.Device``, required): A Device() that will be bound to this process. All NVSHMEM operations on this process will use this Device
        - uid (``nvshmem.UniqueID``, optional): A unique identifier used for UID-based initialization.
            Must be provided if `initializer_method` is "uid".
        - rank (int, optional): Rank of the calling process in the NVSHMEM job. Required for UID-based init.
        - nranks (int, optional): Total number of NVSHMEM ranks in the job. Required for UID-based init.
        - mpi_comm (``mpi4py.MPI.Comm``, optional): MPI communicator to use for MPI-based initialization.
            Defaults to ``MPI.COMM_WORLD`` if ``None`` and ``initializer_method`` is "mpi".
        - initializer_method (str): Specifies the initialization method. Must be either "mpi" or "uid".

    Raises:
        - NvshmemInvalid: If an invalid initialization method is provided, or required arguments
            for the selected method are missing or incorrect.

        - NvshmemError: If NVSHMEM fails to initialize using the specified method.

    Notes:
        - If using MPI-based init, ensure ``mpi4py`` is compiled against the same MPI
          distribution you're running with. A mismatch can result in undefined behavior.
        - For UID-based init, the user is responsible for distributing the `uid` to all
          processes and passing in the correct ``rank`` and ``nranks``.
        - UID-based init is useful for bootstrapping over non-MPI runtimes or custom transports.
        - Internally, this sets up a ``bindings.InitAttr()`` structure which is passed to
          the NVSHMEM host library.

    Example:
        >>> from mpi4py import MPI
        >>> import nvshmem.core as nvshmem
        >>> nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
        # OR for UID mode
        >>> uid = nvshmem.get_unique_id()
        >>> nvshmem.init(uid=uid, rank=0, nranks=1, initializer_method="uid")
    """
    # If Device is None, that's ok.
    _cached_device["device"] = device

    attr = bindings.InitAttr()

    if initializer_method not in ["mpi", "uid", "emulated_mpi"]:
        raise NvshmemInvalid("Invalid init method requested")

    if initializer_method == "mpi":
        """
        The user requested MPI bootstrap. This means we internally use MPI to perform our bootstrap.
        It hasn't been done yet 
        """
        # Step 1: Detect MPI_Comm size

        if not _mpi4py_enabled or (not isinstance(mpi_comm, mpi.Comm) and mpi_comm is not None):
            raise NvshmemInvalid("Invalid MPI communicator passed in")

        # This has the effect of detecting the MPI distro
        # OMPI uses void *, MPICH family uses int
        # Note that as of today, MPI4Py defines this statically by compiling
        # User must make sure the MPI4Py they have is compiled for the MPI distro they are using.
        # Otherwise, weird errors will occur
        if mpi._sizeof(mpi.Comm) == ctypes.sizeof(ctypes.c_int):
            MPI_Comm_t = ctypes.c_int
        else:
            MPI_Comm_t = ctypes.c_void_p

        if mpi_comm is None:
            # If None, assume user wants COMM_WORLD
            mpi_comm = mpi.COMM_WORLD

        # Cast to the correct type for the MPI distro we found
        mpi_comm_val = MPI_Comm_t(mpi_comm.handle)
        # Get pointer
        mpi_comm_ptr = ctypes.pointer(mpi_comm_val)
        # Cast to void* for portability
        mpi_comm_addr = ctypes.cast(mpi_comm_ptr, ctypes.c_void_p).value

        attr.mpi_comm = mpi_comm_addr
        init_status = bindings.hostlib_init_attr(bindings.Flags.INIT_WITH_MPI_COMM, attr.ptr)

        # Cybind converts the success status code (0) to NoneType
        if init_status != None:
            raise NvshmemError("Failed to perform MPI based init")

    if initializer_method == "uid":
        """
        The user requested a UID bootstrap. That means they took it on themself to distribute the UID
        We assume they know what their doing and already broadcasted the UID to anyone who needs it
        
        The user should have gotten the UID from `nvshmem.core.get_unique_id()`. Only one rank calls this function.

        In future, if UID is None, MPI_comm is not None, and initializer method is "uid",
        we may want to provide a convenience functionality for using MPI internally to do the UID bootstrap.
        """
        if uid is None or rank is None or nranks is None or mpi_comm:
            raise NvshmemInvalid("Invalid parameters for UID based init {uid} {rank} {nranks}")

        # It's the caller's responsibility to pass in correct rank and nranks if they use UID bootstrap
        uid_status = bindings.set_attr_uniqueid_args(rank, nranks, uid.ptr, attr.ptr)
        status = bindings.hostlib_init_attr(bindings.Flags.INIT_WITH_UNIQUEID, attr.ptr)

        # Convert status to Exception
        # Cybind converts the success status code to NoneType
        if uid_status != None or status != None:
            raise NvshmemError("Failed to perform UID-based Init. status = {status}")

    if initializer_method == "emulated_mpi":
        """
        This method is an "emulated MPI init" where the user passes an MPI comm, and instead of passing it 
        into nvshmem_init directly, we perform a broadcast across it.

        This is useful for times when you don't want to recompile NVSHMEM for your MPI distro du-jour.
        """
        if not _mpi4py_enabled:
            raise NvshmemInvalid("MPI4Py Required for managed_uid init")
        if mpi_comm is None:
            # If None, assume user wants COMM_WORLD
            mpi_comm = mpi.COMM_WORLD
        rank = mpi_comm.Get_rank()
        nranks = mpi_comm.Get_size()
        local_rank_per_node = rank % system.num_devices
 
        # Create an empty uniqueid for all ranks
        uniqueid = get_unique_id(empty=True)
        if rank == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = get_unique_id()

        # Broadcast UID to all ranks
        mpi_comm.Bcast(uniqueid._data.view(np.int8), root=0)

        # It's the caller's responsibility to pass in correct rank and nranks if they use UID bootstrap
        uid_status = bindings.set_attr_uniqueid_args(rank, nranks, uniqueid.ptr, attr.ptr)
        status = bindings.hostlib_init_attr(bindings.Flags.INIT_WITH_UNIQUEID, attr.ptr)

        # Convert status to Exception
        # Cybind converts the success status code to NoneType
        if uid_status != None or status != None:
            raise NvshmemError("Failed to perform emulated_mpi-based Init. status = {status}")


    log_level = os.environ.get("NVSHMEM_DEBUG")
    if log_level in ("INFO", "DEBUG"):
        _debug_mode = True

    if not log_level or log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", None]:
        # Default to ERROR level
        log_level = "ERROR"

    # Now that we have PE set, re-initialize logging with mype None so that the logger picks up the correct one.
    utils._configure_logging(level=log_level)

    nvshmem.core._internal_tracking._is_initialized["status"] = InternalInitStatus.INITIALIZED

def finalize() -> None:
    """
    Finalize the NVSHMEM runtime.

    This function wraps the NVSHMEM finalization routine. It should be called after all 
    NVSHMEM operations are complete and before the application exits.

    Typically, this is called once per process to clean up NVSHMEM resources.

    Raises:
        ``NvshmemError``: If the NVSHMEM finalization fails.

    Example:
        >>> nvshmem.core.finalize()
    """
    logger.debug("nvshmem_finalize() called")
    non_peer_bufs = []
    for mr in _mr_references.values():
        for ptr, buf in mr._mem_references.items():
            if not buf["is_peer_buffer"] and buf["ref_count"] > 0:
                logger.error(f"Found un-freed memory object with address {ptr} at fini time")
                non_peer_bufs.append(buf)
    if len(non_peer_bufs) > 0:
        logger.error(f"Found {len(non_peer_bufs)} un-freed memory objects at fini time")

    memory._free_all_buffers()

    fini_status = bindings.hostlib_finalize()
    # Cybind converts the success status code to NoneType
    if fini_status != None:
        raise NvshmemError("Failed to finalize Hostlib")

    nvshmem.core._internal_tracking._is_initialized["status"] = InternalInitStatus.DE_INITIALIZED
