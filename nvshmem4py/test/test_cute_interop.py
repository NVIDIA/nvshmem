"""
Unit tests for CuTe DSL interoperability functionality in nvshmem.core
"""
try:
    import cutlass.cute as cute
    _cute_enabled = True
except:
    _cute_enabled = False

from utils import uid_init, mpi_init
import argparse
import os
import gc
import struct

from cuda.core import VirtualMemoryResource, VirtualMemoryResourceOptions

import nvshmem.core
import nvshmem.core.interop.cute as cute_interop
import cuda.bindings.driver as cudrv


from cuda.core import Device, system

from mpi4py import MPI


def _fill_tensor(tensor, value, device_id=None):
    # Avoid JIT compilation by using a driver memset for float32 tensors.
    buf, size, dtype = cute_interop.tensor_get_buffer(tensor)
    if dtype != cute.Float32:
        raise ValueError(f"Unsupported dtype for fill: {dtype}")
    # Convert float to uint32 bit pattern for cuMemsetD32
    value_u32 = struct.unpack('I', struct.pack('f', float(value)))[0]
    count = size // 4
    cudrv.cuMemsetD32(buf.handle, value_u32, count)

def test_mc_tensor():
    print("Testing Multicast Cute DSL Tensor")
    dev = Device()
    local_rank_per_node = dev.device_id
    if not _cute_enabled:
        print("WARNING: Cute DSL not found. Not running Cute DSL Interop test")
        return
    if not dev.properties.multicast_supported or nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping MC memory test because Multicast memory is not supported on this platform")
        return
    tensor = cute_interop.tensor(1024)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        mc_tensor = cute_interop.get_multicast_tensor(nvshmem.core.Teams.TEAM_SHARED, tensor)
        print(tensor, tensor.data_ptr())
        if mc_tensor is not None:
            print(mc_tensor.data_ptr())

    nvshmem.core.free_tensor(tensor)
    print("End MC tensor test")


def test_interop_cute():
    print("Testing Cute DSL interop buffer")
    if not _cute_enabled:
        print("WARNING: Cute DSL not found. Not running Cute DSL Interop test")
        return
    # Get a buffer
    local_rank_per_node = Device().device_id
    print("Rank:", local_rank_per_node)
    # For the Cute DSL test, we 
    tensor = cute_interop.tensor((1,2,3,4,5), dtype=cute.Float32)
    print(tensor)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    _fill_tensor(tensor, float(nvshmem.core.my_pe() + 1), device_id=local_rank_per_node)
    print(tensor)
    # Free Buffer
    cute_interop.free_tensor(tensor)
    print("Ending test Cute DSL interop buffer")

def test_peer_tensor():
    print("Testing peer tensor")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    tensor = cute_interop.tensor((256, 1), dtype=cute.Float32)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        try:
            peer_tensor = cute_interop.get_peer_tensor(tensor, ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()))
        except Exception as e:
            print(f"Error getting peer tensor: {e}")
            peer_tensor = None
            # If >1 node (NVL domain) exists, this is an error condition. TODO enhance this to differentiate
            print("peer_buf tensor failed to create. Check to make sure it failed because of >1 node")
        print(peer_tensor, tensor)
        # Don't need to call free on a peer buf.
        # However, it is safe to do so. nvshmem.core.free knows to skip nvshmem_free()
        #nvshmem.core.free(peer_buf)
    cute_interop.free_tensor(tensor)
    print("End peer tensor test")


def test_fortran_morder_alloc_cute():
    print("Testing allocating Fortran-ordered memory Cute DSL")
    if not _cute_enabled:
        print("WARNING: Cute DSL not found. Not running Cute DSL Interop test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    print("Rank:", local_rank_per_node)
    dev = Device(local_rank_per_node)
    # For the Cute DSL test, we 
    tensor = cute_interop.tensor((1,2,3,4,5), dtype=cute.Float32, morder="F")
    print(tensor)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    _fill_tensor(tensor, float(nvshmem.core.my_pe() + 1), device_id=local_rank_per_node)
    print(tensor)
    # Free Buffer
    cute_interop.free_tensor(tensor)
    print("Done tsting allocating Fortran-ordered memory Cute DSL")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_interop_cute()
    test_mc_tensor()
    test_peer_tensor()
    test_fortran_morder_alloc_cute()
    print("All tests passed")
    cute_interop.cleanup_cute()
    nvshmem.core.finalize()
