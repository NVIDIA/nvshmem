"""
This file is for init/fini tests for NVSHMEM4Py
The reason it's separate from the basic sanity test is it needs to be launched in multiples.
Test both the UID and MPI based init/fini
"""

# Usage: mpirun -np <N> python init_test.py -i <init type>
# Must set `-x NVSHMEM_BOOTSTRAP=MPI` if you want to use the MPI init

import nvshmem
import nvshmem.core

import argparse
import sys
import os
import platform

import numpy as np
from mpi4py import MPI
from cuda.core import Device, system, Program, ProgramOptions, LinkerOptions, ObjectCode, Linker

# User should not import this - it's here so we can print stuff
import nvshmem.bindings


def test_mpi_comm_init():
    # Test device init and bootstrap
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    nvshmem.core.finalize()
    print("Init/Fini with MPI passed with cuda.core init/fini as well")


def test_multi_init():
    # Test multiple calls to init with overlapping sessions
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    print("called init1")
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    print("called init2")
    print(f"Hello from PE {nvshmem.bindings.my_pe()} npes {nvshmem.bindings.n_pes()}")
    nvshmem.core.finalize()
    print("called fini1")
    nvshmem.core.finalize()
    print("called fini2")
    print("Init/Fini multi-init with MPI passed")


def test_uid_init():
    # This will use mpi4py to perform a UID based init with bcast.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    local_rank_per_node = rank % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank == 0:
        # Rank 0 gets a real uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # Broadcast UID to all ranks
    comm.Bcast(uniqueid._data.view(np.int8), root=0)

    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks, mpi_comm=None, initializer_method="uid")
    nvshmem.core.finalize()
    print("Init/Fini with UID passed")


def test_emulated_mpi_init():
    # Test device init and bootstrap
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev,
                      uid=None,
                      rank=None,
                      nranks=None,
                      mpi_comm=MPI.COMM_WORLD,
                      initializer_method="emulated_mpi")
    nvshmem.core.finalize()
    print("Init/Fini with emulated MPI passed with cuda.core init/fini as well")


def test_none_device_init():
    # Test init with None device and allocate with device set
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()

    nvshmem.core.init(device=None, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    dev = Device(local_rank_per_node)
    dev.set_current()
    buf = nvshmem.core.buffer(1024)
    print(buf)
    nvshmem.core.free(buf)
    nvshmem.core.finalize()
    print("Init/Fini with MPI passed with cuda.core init/fini as well")
    print("2 stage init with None device passed")


# simple put/quiet kernel

code = """
#define __NVSHMEM_NUMBA_SUPPORT__ 1
extern "C" {

__device__ int nvshmem_my_pe(void);
__device__ int nvshmem_n_pes(void); 
}

__device__ void nvshmem_my_pe_kernel(int *src, int *dest, size_t num_elems)
{
    int pe = (nvshmem_my_pe() + 1) % nvshmem_n_pes();
}

"""

lib_code = """
#define __NVSHMEM_NUMBA_SUPPORT__ 1
#include<nvshmem.h>                 
#include<nvshmemx.h>
"""


def test_module_init():
    # Test host lib + device state init with None device and allocate with device set
    print("Starting module init test")
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    # compile a simple A+B kernel using NVRTC and emit cubin to be loaded as ObjectCode
    # with a valid cubin, kernel ptr and module object
    arch = "".join(f"{i}" for i in dev.compute_capability)
    try:
        nvshmem_include_path = os.environ["NVSHMEM_HOME"] + "/include"
        rdma_core_include_path = os.environ["RDMA_CORE_HOME"] + "/include"
    except KeyError:
        print(
            "NVSHMEM_HOME or RDMA_CORE_HOME not set. This test requires NVSHMEM_HOME and RDMA_CORE_HOME env vars to be set"
        )
        print("Skipping the module init test")
        return
    cpu_arch = platform.machine()
    program_options = ProgramOptions(
        std="c++11",
        arch=f"sm_{arch}",
        include_path=["/usr/local/cuda/include/", nvshmem_include_path, rdma_core_include_path],
        relocatable_device_code=True,
        link_time_optimization=True)
    prog = Program(code, code_type="c++", options=program_options)
    mod = prog.compile("ltoir")

    # Get a library object using LTOIR with NVRTC
    prog_lib = Program(lib_code, code_type="c++", options=program_options)
    lib = prog_lib.compile("ltoir")

    link_options = LinkerOptions(arch=f"sm_{arch}", link_time_optimization=True)
    linker = Linker(lib, mod, options=link_options)
    linked = linker.link("cubin")
    kernel_obj = nvshmem.core.NvshmemKernelObject.from_handle(linked.handle)
    nvshmem.core.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    # Use the module to initialize NVSHMEM device state
    nvshmem.core.library_init(kernel_obj)
    nvshmem.core.library_finalize(kernel_obj)
    print("Host library + Device state init/fini passed with cuda.core")
    nvshmem.core.finalize()
    print("Module init test passed")


def test_find_device_bitcode_library():
    """
    Test for find_device_bitcode_library utility function.
    It should find the library containing device bitcode,
    typically needed for LTO or device linking.
    """
    print("Starting test for find_device_bitcode_library")
    # Attempt to import the function under test
    try:
        from nvshmem.core import find_device_bitcode_library
    except ImportError:
        print("Could not import find_device_bitcode_library; skipping test.")
        return
    try:
        lib_path = find_device_bitcode_library()
    except Exception as e:
        print(f"Exception raised calling find_device_bitcode_library: {e}")
        assert False, "Exception in find_device_bitcode_library"

    # Check the returned path
    assert lib_path is not None, "Library path should not be None"
    assert lib_path.endswith("libnvshmem_device.bc"), f"Unexpected library found: {lib_path}"
    print(f"find_device_bitcode_library returned: {lib_path}")
    print("find_device_bitcode_library test passed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type",
                        "-i",
                        type=str,
                        help="Init type to use",
                        choices=["mpi", "uid", "emulated_mpi"],
                        default="uid")
    args = parser.parse_args()

    if args.init_type == "mpi":
        test_multi_init()
        test_mpi_comm_init()
        test_none_device_init()
        test_module_init()
        test_find_device_bitcode_library()
    elif args.init_type == "uid":
        test_uid_init()
    elif args.init_type == "emulated_mpi":
        test_emulated_mpi_init()
    else:
        print(f"Unexpected init type: {args.init_type}")
        sys.exit(1)
