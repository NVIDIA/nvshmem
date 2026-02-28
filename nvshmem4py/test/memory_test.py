"""
Unit tests for memory management functionality in nvshmem.core
"""
try:
    import torch
    from torch import float32
    _torch_enabled = True
except:
    float32 = None
    _torch_enabled = False

try:
    import cupy
    _cupy_enabled = True
except:
    _cupy_enabled = False

from utils import uid_init, mpi_init
import argparse
import os
import gc

from cuda.core import VirtualMemoryResource, VirtualMemoryResourceOptions

import nvshmem.core

# HACK! Do not do this!
from nvshmem.core._internal_tracking import _mr_references

from cuda.core import Device, system

from mpi4py import MPI


def test_buffer():
    print("Testing basic buffer")

    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf = nvshmem.core.buffer(1024)
    nvshmem.core.free(buf)
    print("End basic buffer test")


def test_peer_buffer():
    print("Testing peer buffer")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    buf = nvshmem.core.buffer(1024)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        try:
            peer_buf = nvshmem.core.get_peer_buffer(buf, ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()))
        except Exception as e:
            peer_buf = None
            # If >1 node (NVL domain) exists, this is an error condition. TODO enhance this to differentiate
            print("peer_buf failed to create. Check to make sure it failed because of >1 node")
            peer_buf = None
        else:
            print(peer_buf, peer_buf.handle, buf, buf.handle)
        if peer_buf:
            print(peer_buf, peer_buf.handle)
            # Don't need to call free on a peer buf.
            # However, it is safe to do so. nvshmem.core.free knows to skip nvshmem_free()
            nvshmem.core.free(peer_buf)

    nvshmem.core.free(buf)
    print("End peer buffer test")


def test_mc_buffer():
    print("Testing Multicast buffer")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    if not dev.properties.multicast_supported or nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping MC memory test because Multicast memory is not supported on this platform")
        return

    buf = nvshmem.core.buffer(1024)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        mc_buf = nvshmem.core.get_multicast_buffer(nvshmem.core.Teams.TEAM_SHARED, buf)
        print(buf, buf.handle)
        if mc_buf:
            print(mc_buf, mc_buf.handle)

    nvshmem.core.free(buf)
    print("End MC buffer test")


def test_mc_tensor():
    print("Testing Multicast Torch Tensor")
    dev = Device()
    local_rank_per_node = dev.device_id
    if not _torch_enabled:
        print("WARNING: Torch not found. Not running Torch Interop test")
        return
    dev = Device(local_rank_per_node)
    if not dev.properties.multicast_supported or nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping MC memory test because Multicast memory is not supported on this platform")
        return
    tensor = nvshmem.core.tensor(1024)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        mc_tensor = nvshmem.core.get_multicast_tensor(nvshmem.core.Teams.TEAM_SHARED, tensor)
        print(tensor, tensor.data_ptr())
        if mc_tensor is not None:
            print(mc_tensor.data_ptr())

    nvshmem.core.free_tensor(tensor)
    print("End MC tensor test")


def test_mc_array():
    print("Testing Multicast Torch array")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    if not dev.properties.multicast_supported or nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping MC memory test because Multicast memory is not supported on this platform")
        return

    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy Interop test")
        return
    array = nvshmem.core.array(1024)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        mc_array = nvshmem.core.get_multicast_array(nvshmem.core.Teams.TEAM_SHARED, array)
        print(array, array.data.ptr)
        if mc_array is not None:
            print(mc_array.data.ptr)

    nvshmem.core.free_array(array)
    print("End MC array test")


def test_interop_torch():
    print("Testing Torch interop buffer")
    if not _torch_enabled:
        print("WARNING: Torch not found. Not running Torch Interop test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    print("Rank:", local_rank_per_node)
    dev = Device(local_rank_per_node)
    # For the Torch test, we
    tensor = nvshmem.core.tensor((1, 2, 3, 4, 5), dtype=float32)
    print(tensor)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    tensor[:] = nvshmem.core.my_pe() + 1
    print(tensor)
    # Free Buffer
    nvshmem.core.free_tensor(tensor)
    print("Ending test Torch interop buffer")


def test_interop_cupy():
    print("Testing CuPy interop mem")
    if not _cupy_enabled:
        print("WARNING: CuPy not found. Not running CuPy Interop test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device(local_rank_per_node)
    array = nvshmem.core.array((1, 2, 3, 4), dtype="float32")
    print(array)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    array[:] = nvshmem.core.my_pe() + 1
    print(array)
    # Free Buffer
    nvshmem.core.free_array(array)
    print("Done testing CuPy Interop mem")


def test_peer_array():
    print("Testing peer array")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    arr = nvshmem.core.array((256, 1), dtype="float32")

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        try:
            peer_arr = nvshmem.core.get_peer_array(arr, ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()))
        except Exception as e:
            peer_arr = None
            # If >1 node (NVL domain) exists, this is an error condition. TODO enhance this to differentiate
            print("peer_buf array failed to create. Check to make sure it failed because of >1 node")
        else:
            print(peer_arr, arr)
        # Don't need to call free on a peer buf.
        # However, it is safe to do so. nvshmem.core.free knows to skip nvshmem_free()
        #nvshmem.core.free(peer_buf)

    nvshmem.core.free_array(arr)
    print("End peer array test")


def test_peer_tensor():
    print("Testing peer tensor")
    # TODO: TEAM_NODE not showing up because of anonymous enum
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    tensor = nvshmem.core.tensor((256, 1), dtype=float32)

    # PE0 calls get_peer_buffer on the Buffer
    if local_rank_per_node == 0:
        try:
            peer_tensor = nvshmem.core.get_peer_tensor(tensor, ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()))
        except Exception as e:
            peer_tensor = None
            # If >1 node (NVL domain) exists, this is an error condition. TODO enhance this to differentiate
            print("peer_buf tensor failed to create. Check to make sure it failed because of >1 node")
        print(peer_tensor, tensor)
        # Don't need to call free on a peer buf.
        # However, it is safe to do so. nvshmem.core.free knows to skip nvshmem_free()
        #nvshmem.core.free(peer_buf)
    nvshmem.core.free_tensor(tensor)
    print("End peer tensor test")


def test_del_buffer():
    print("Testing buffer scope")
    # Test that when we call del on a buffer, it doesn't actually go away
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf = nvshmem.core.buffer(1024)
    print("Created buffer")
    ptr = buf.handle
    print(buf, buf.handle, ptr)
    print(_mr_references)
    try:
        del buf
    except:
        print("Caught early-warning about explicit free")
    print("Deleted buffer. Printing pointer")
    print(ptr)
    print("Printing reference count")
    print(_mr_references)
    print(_mr_references[local_rank_per_node]._mem_references[ptr])
    assert _mr_references[local_rank_per_node]._mem_references[ptr]["ref_count"] > 0
    # This is technically a user error. They can't free the buf if they del it.
    # NVSHMEM Fini-time checking will save them from themself.
    # In the test, I am hacking it to follow the rules
    _mr_references[local_rank_per_node]._mem_references[ptr]["freed"] = True
    _mr_references[local_rank_per_node]._mem_references[ptr]["buffer"].close()
    print("done testing buffer scope")


def test_release_del_buffer():
    print("Testing buffer scope")
    # Test that when we call del on a buffer, it doesn't actually go away
    # And we get the error raised immediately as an Exception
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf = nvshmem.core.buffer(1024, release=True)
    ptr = buf.handle
    try:
        del buf
    except:
        print("Caught early-warning about explicit free")

    print(_mr_references[local_rank_per_node]._mem_references[ptr])
    assert _mr_references[local_rank_per_node]._mem_references[ptr]["ref_count"] >= 0
    # This keeps an error from being raised at fini time
    _mr_references[local_rank_per_node]._mem_references[ptr]["freed"] = True
    _mr_references[local_rank_per_node]._mem_references[ptr]["ref_count"] = 0
    print("done testing buffer scope")


def test_buffer_scope_release_gc():
    print("Testing buffer scope var 2")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf = nvshmem.core.buffer(8, release=True)
    ptr = buf.handle
    array = cupy.from_dlpack(buf)

    del buf, array
    try:
        gc.collect()
    except:
        print("Caught early-warning about explicit free")
    # This keeps an error from being raised at fini time
    _mr_references[local_rank_per_node]._mem_references[ptr]["freed"] = True
    _mr_references[local_rank_per_node]._mem_references[ptr]["ref_count"] = 0
    print("done testing buffer scope var 2")


def test_buffer_scope_release_gc_free():
    print("Testing buffer scope var 3")
    # All of this should complete without errors of any kind
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    buf = nvshmem.core.buffer(8, release=True)
    ptr = buf.handle
    array = cupy.from_dlpack(buf)

    nvshmem.core.free_array(array)

    del buf, array
    gc.collect()
    # This keeps an error from being raised at fini time
    _mr_references[local_rank_per_node]._mem_references[ptr]["freed"] = True
    print("Done testing buffer scope var 3")


def test_buffer_scope_release_gc_free_reuse():
    print("Testing buffer scope var 4")
    buf1 = nvshmem.core.buffer(8, release=True)
    buf2 = nvshmem.core.buffer(8, release=True)

    nvshmem.core.free(buf1)
    del buf1

    # Here it reuses memory block used by buf1. It looks like after creating
    # a new entry in self._mem_references[ptr], a Buffer.close() is being called with the same
    # pointer, and nvshmem.core thinks that there is an attempt to close the buffer without
    # having called free, which is not true.
    buf3 = nvshmem.core.buffer(8, release=True)

    gc.collect()

    nvshmem.core.free(buf2)
    nvshmem.core.free(buf3)
    print("Done testing buffer scope var 4")


def test_get_peer_memory_scope():
    print("Testing child peer buffer scope")

    if nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_NODE) == 1:
        print("Skipping test because team_npes is 1")
        return
    import random
    for _ in range(10):
        shape = (16, 1024)
        t = nvshmem.core.tensor(shape, dtype=torch.float16)
        rank = int(nvshmem.core.my_pe())
        if rank == 0:
            t_peer = nvshmem.core.get_peer_tensor(t, ((nvshmem.core.my_pe() + 1) % nvshmem.core.n_pes()))
        if rank == 0:
            nvshmem.core.free_tensor(t_peer)
        nvshmem.core.free_tensor(t)
    print("Done testing child peer buffer scope")


def test_fortran_morder_alloc_torch():
    print("Testing allocating Fortran-ordered memory Torch")
    if not _torch_enabled:
        print("WARNING: Torch not found. Not running Torch Interop test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    print("Rank:", local_rank_per_node)
    dev = Device(local_rank_per_node)
    # For the Torch test, we
    tensor = nvshmem.core.tensor((1, 2, 3, 4, 5), dtype=float32, morder="F")
    print(tensor)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    tensor[:] = nvshmem.core.my_pe() + 1
    print(tensor)
    # Free Buffer
    nvshmem.core.free_tensor(tensor)
    print("Done tsting allocating Fortran-ordered memory Torch")


def test_fortran_morder_alloc_cupy():
    print("Testing allocating Fortran-ordered memory Cupy")
    if not _cupy_enabled:
        print("WARNING: Cupy not found. Not running Cupy Interop test")
        return
    # Get a buffer
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    print("Rank:", local_rank_per_node)
    # For the Torch test, we
    array = nvshmem.core.array((1, 2, 3, 4, 5), dtype="float32", morder="F")
    print(array)
    # Set to my_pe + 1 so that it doesn't ever show 0.
    array[:] = nvshmem.core.my_pe() + 1
    if not cupy.isfortran(array):
        raise Exception("Array is not Fortran-ordered")
    print(array)
    # Free Buffer
    nvshmem.core.free_array(array)
    print("Done tsting allocating Fortran-ordered memory Cupy")


import cuda.bindings.driver as driver
import warnings

CUDA_COH_FULL = "CUDA_COH_FULL"
CUDA_COH_CDMM = "CUDA_COH_CDMM"
CUDA_COH_MIGRATION = "CUDA_COH_MIGRATION"
CUDA_COH_NONE = "CUDA_COH_NONE"


def detect_cuda_coherence_model(device_ordinal: int = 0) -> str:
    err, = driver.cuInit(0)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed: {err}")

    err, dev = driver.cuDeviceGet(device_ordinal)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGet({device_ordinal}) failed: {err}")

    attr1 = 0
    if hasattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES"):
        err, val = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, dev)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetAttribute(HOST_PAGE_TABLES) failed: {err}")
        attr1 = int(val != 0)
    else:
        warnings.warn("CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES is not defined, "
                      "assuming non-coherent platform")

    attr2 = 0
    if hasattr(driver.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST"):
        err, val = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, dev)
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGetAttribute(DIRECT_MANAGED_MEM_ACCESS_FROM_HOST) failed: {err}")
        attr2 = int(val != 0)
    else:
        warnings.warn("CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST is not defined, "
                      "assuming no migration support")

    if attr1 and attr2:
        return CUDA_COH_FULL  # CDMM off
    elif attr1:
        return CUDA_COH_CDMM  # CDMM on
    elif attr2:
        return CUDA_COH_MIGRATION  # Pascal+ migration
    else:
        return CUDA_COH_NONE  # Maxwell and older


def test_external_buffer():
    print("Testing external buffer")
    local_rank_per_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    dev = Device()
    local_rank_per_node = dev.device_id
    print("Rank:", local_rank_per_node)

    # Use cuda-bindings CuMemCreate binding to allocate a VMM buffer and wrap it for external registration.
    # We implement a new MemoryResource for VMM buffers.
    print("Creating VMMResource")

    # For coherent platforms, always use fabric handles to support MNNVL.
    # For non-coherent platforms, use the default handle type (posix_fd).
    coherence_model = detect_cuda_coherence_model(dev.device_id)
    print("Coherence model:", coherence_model)
    if coherence_model == CUDA_COH_FULL:
        options = VirtualMemoryResourceOptions(handle_type="fabric", gpu_direct_rdma=True)
    else:
        options = VirtualMemoryResourceOptions(gpu_direct_rdma=True)

    if coherence_model != CUDA_COH_FULL and coherence_model != CUDA_COH_MIGRATION:
        print("NOTICE: Non-coherent platform detected or CDMM mode detected. Using posix_fd handle type.")

    resource = VirtualMemoryResource(dev, config=options)
    print("Allocating Buffer using VMM APIs via VMMResource")
    buffer1 = resource.allocate(536870912)
    buffer2 = resource.allocate(536870912)

    print("Creating Torch Tensors from the buffers")
    tensor_src = torch.from_dlpack(buffer1)
    tensor_dst = torch.from_dlpack(buffer2)

    print("Registering Tensors with NVSHMEM")
    tensor_src_reg = nvshmem.core.register_external_tensor(tensor_src)
    tensor_dst_reg = nvshmem.core.register_external_tensor(tensor_dst)

    tensor_src[:] = nvshmem.core.my_pe() + 1
    tensor_dst[:] = 0

    print("tensor_src before reduce:", tensor_src)
    print("tensor_dst before reduce:", tensor_dst)

    nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD,
                        tensor_dst_reg,
                        tensor_src_reg,
                        op="sum",
                        stream=dev.create_stream())
    dev.sync()
    print("tensor_dst after reduce:", tensor_dst)

    print("Unregistering Buffers from NVSHMEM")
    nvshmem.core.unregister_external_tensor(tensor_dst_reg)
    nvshmem.core.unregister_external_tensor(tensor_src_reg)

    print("Deleting Buffers")
    del buffer1, buffer2
    gc.collect()

    print("Done testing external buffer")


def test_peer_buffer_reuse_updates_size():
    print("Testing peer buffer cleanup on free")
    if nvshmem.core.team_n_pes(nvshmem.core.Teams.TEAM_SHARED) < 3:
        print("Skipping test because it requires a team with at least 3 PEs")
        return

    rank = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_SHARED)
    # get_peer_buffer expects global PE IDs; translate TEAM_SHARED ranks 1,2 to TEAM_WORLD
    # so each node's rank-0 PE uses same-node peers (avoids cross-node access on multi-node)
    if rank == 0:
        peers = [
            nvshmem.core.team_translate_pe(nvshmem.core.Teams.TEAM_SHARED, i, nvshmem.core.Teams.TEAM_WORLD)
            for i in range(1, 3)
        ]
    else:
        peers = []

    buf = nvshmem.core.buffer(1024)
    peer_bufs = []
    mr = buf.memory_resource
    try:
        if rank == 0:
            for pe in peers:
                peer_bufs.append(nvshmem.core.get_peer_buffer(buf, pe))

        if rank == 0:
            parent_ptr = int(buf.handle)
            child_ptrs_before = [
                ptr for ptr, entry in mr._mem_references.items()
                if entry.get("parent") == parent_ptr and entry.get("type") is not None and entry.get("type").name in (
                    "PEER", "MULTIMEM")
            ]
            if len(child_ptrs_before) < len(peers):
                raise Exception(
                    f"Expected at least {len(peers)} child entries before free, found {len(child_ptrs_before)}")

        nvshmem.core.free(buf)

        if rank == 0:
            for child_ptr in child_ptrs_before:
                entry = mr._mem_references.get(child_ptr)
                if entry is None:
                    # Entry was removed, which is acceptable as long as it isn't stale.
                    continue
                if not entry.get("freed"):
                    raise Exception("Child entry not marked freed after parent free")
                if "buffer" in entry:
                    raise Exception("Child entry buffer not cleared after parent free")
    finally:
        if rank == 0:
            for peer_buf in peer_bufs:
                nvshmem.core.free(peer_buf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_buffer()
    test_peer_buffer()
    test_peer_array()
    test_peer_tensor()
    test_interop_cupy()
    test_interop_torch()
    test_del_buffer()
    test_mc_buffer()
    test_mc_tensor()
    test_mc_array()
    test_release_del_buffer()
    test_buffer_scope_release_gc()
    test_buffer_scope_release_gc_free()
    test_buffer_scope_release_gc_free_reuse()
    test_fortran_morder_alloc_torch()
    test_fortran_morder_alloc_cupy()
    test_external_buffer()
    test_get_peer_memory_scope()
    test_peer_buffer_reuse_updates_size()

    nvshmem.core.finalize()
