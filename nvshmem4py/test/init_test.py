# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import numpy as np
from mpi4py import MPI
from cuda.core import Device, system, Program, ProgramOptions, LinkerOptions, ObjectCode, Linker

# User should not import this - it's here so we can print stuff
import nvshmem.bindings


def test_host_information_apis():
    """Exercise host-library information and TMA shared-memory queries."""
    name = nvshmem.core.get_name()
    assert isinstance(name, str) and name, "nvshmem_info_get_name returned an empty name"

    thread_support = nvshmem.core.query_thread()
    assert int(thread_support) >= 0, "nvshmem_query_thread returned an invalid support level"

    recommended = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_RECOMMENDED)
    minimum = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_MINIMUM)
    barriers_only = nvshmem.core.ask_smem(nvshmem.core.SmemAmount.SMEM_BARRIERS_ONLY)
    assert recommended >= minimum >= barriers_only > 0, "invalid host TMA shared-memory requirements"
    assert nvshmem.core.ask_smem(nvshmem.core.SmemAmount.MAX) == recommended
    print(
        f"Host info APIs passed: name={name}, thread_support={thread_support}, smem={recommended}/{minimum}/{barriers_only}"
    )


def test_mpi_comm_init():
    # Test device init and bootstrap
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.get_num_devices()
    dev = Device(local_rank_per_node)
    dev.set_current()
    nvshmem.core.init(device=dev, uid=None, rank=None, nranks=None, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    try:
        test_host_information_apis()
    finally:
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
    try:
        test_host_information_apis()
    finally:
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
    try:
        test_host_information_apis()
    finally:
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
    cuda_include_path = os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/include"
    # CCCL headers (cuda/std/cstdint etc.) may come from the pip package
    from cuda.pathfinder import find_nvidia_header_directory
    cccl_include_path = find_nvidia_header_directory("cccl")
    if cccl_include_path is None:
        print("CCCL headers not found. Skipping the module init test")
        return
    program_options = ProgramOptions(
        std="c++17",
        arch=f"sm_{arch}",
        include_path=[cuda_include_path, nvshmem_include_path, rdma_core_include_path, cccl_include_path],
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
    Tests default arch ("90"), per-arch lookup, LTOIR, and static library formats.
    """
    print("Starting test for find_device_bitcode_library")
    try:
        from nvshmem.core import find_device_bitcode_library, DeviceLibLanguage
    except ImportError:
        print("Could not import find_device_bitcode_library; skipping test.")
        return

    import os

    # Test 1: default prefers the current device's per-arch bitcode entry point,
    # then falls back to the backward-compatible bitcode entry point.
    try:
        lib_path = find_device_bitcode_library()
    except Exception as e:
        print(f"Exception raised calling find_device_bitcode_library(): {e}")
        raise AssertionError(f"Exception in find_device_bitcode_library(): {e}")
    assert lib_path is not None, "Library path should not be None"
    lib_name = os.path.basename(lib_path)
    assert lib_name == "libnvshmem_device.bc" or (
        lib_name.startswith("libnvshmem_device_sm_")
        and lib_name.endswith(".bc")), f"Expected NVSHMEM device bitcode, got: {lib_path}"
    assert os.path.exists(lib_path), f"File does not exist: {lib_path}"
    print(f"  default:   {lib_path}")

    # Test 2: per-arch bitcode lookup (sm_90), when the package ships it.
    lib_path_90_expected = os.path.join(os.path.dirname(lib_path), "libnvshmem_device_sm_90.bc")
    if os.path.exists(lib_path_90_expected):
        try:
            lib_path_90 = find_device_bitcode_library(arch="90")
        except Exception as e:
            print(f"Exception raised calling find_device_bitcode_library(arch='90'): {e}")
            raise AssertionError(f"Exception in find_device_bitcode_library(arch='90'): {e}")
        assert lib_path_90 == lib_path_90_expected, f"Expected {lib_path_90_expected}, got: {lib_path_90}"
        print(f"  arch=90:   {lib_path_90}")
    else:
        print(f"  arch=90:   not available (skipped): {lib_path_90_expected}")

    # Test 3: LTOIR library (arch-independent fatbin)
    try:
        ltoir_path = find_device_bitcode_library(language=DeviceLibLanguage.LTOIR)
    except Exception as e:
        print(f"LTOIR lookup raised exception (may not be installed): {e}")
        ltoir_path = None
    if ltoir_path:
        assert ltoir_path.endswith("libnvshmem_device.ltoir.fatbin"), f"Expected .ltoir, got: {ltoir_path}"
        assert os.path.exists(ltoir_path), f"File does not exist: {ltoir_path}"
        print(f"  LTOIR:     {ltoir_path}")
    else:
        print("  LTOIR:     not available (skipped)")

    # Test 4: static library
    try:
        static_path = find_device_bitcode_library(language=DeviceLibLanguage.STATIC)
    except Exception as e:
        print(f"Static library lookup raised exception (may not be installed): {e}")
        static_path = None
    if static_path:
        assert static_path.endswith("libnvshmem_device.a"), f"Expected .a, got: {static_path}"
        assert os.path.exists(static_path), f"File does not exist: {static_path}"
        print(f"  static:    {static_path}")
    else:
        print("  static:    not available (skipped)")

    raised = False
    try:
        find_device_bitcode_library(arch="999")
    except Exception:
        raised = True
    assert raised, "Expected exception for invalid arch=999"
    print("  arch=999:  correctly raised exception")

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
