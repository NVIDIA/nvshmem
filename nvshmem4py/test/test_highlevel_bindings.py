import cffi
import argparse

from cuda.core import Device

from numba import cuda, int32
from numba.types import float32, Array
from numba.core.extending import overload

import nvshmem
from nvshmem.bindings import barrier_all
from nvshmem.bindings.device.numba import my_pe, n_pes, int_p, float_p
from utils import uid_init, mpi_init


def test_highlevel_bindings(dev: Device):

    ffi = cffi.FFI()

    def p():
        pass

    @overload(p)
    def p_ol(arr, mype, peer):
        if arr == Array(dtype=int32, ndim=arr.ndim, layout=arr.layout):

            def impl(arr, mype, peer):
                ptr = ffi.from_buffer(arr)
                int_p(ptr, mype, peer)

            return impl
        elif arr == Array(dtype=float32, ndim=arr.ndim, layout=arr.layout):

            def impl(arr, mype, peer):
                ptr = ffi.from_buffer(arr)
                float_p(ptr, mype, peer)

            return impl

    @cuda.jit(lto=True)
    def app_kernel(dest):
        mype = my_pe()
        npes = n_pes()
        peer = int32((mype + 1) % npes)

        p(dest, mype, peer)

    dest = nvshmem.core.array((1, ), dtype="float32")

    app_kernel[1, 1, 0](dest)

    barrier_all()
    dev.sync()

    nvshmem.core.free_array(dest)
    nvshmem.core.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        dev = uid_init()
    elif args.init_type == "mpi":
        dev = mpi_init()

    test_highlevel_bindings(dev)
