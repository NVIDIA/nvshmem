import cffi
import argparse

from cuda.core import Device

from numba import cuda, int32

import nvshmem
from nvshmem.bindings.device.numba import my_pe, n_pes, int_p
from nvshmem.bindings import barrier_all, my_pe as h_my_pe

from utils import uid_init, mpi_init


def test_ring(dev: Device):
    ffi = cffi.FFI()

    @cuda.jit(lto=True)
    def app_kernel(dest):
        ptr = ffi.from_buffer(dest)
        mype = my_pe()
        npes = n_pes()
        peer = int32((mype + 1) % npes)

        int_p(ptr, mype, peer)

    dest = nvshmem.core.array((1, ), dtype="int32")

    app_kernel[1, 1, 0](dest)

    barrier_all()
    dev.sync()

    print(f"{h_my_pe()}: received message {dest[0]}")

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

    test_ring(dev)
