import cffi
import argparse

import numpy as np
from numba import cuda

from utils import uid_init, mpi_init

from nvshmem.bindings.device.numba import vendor_get_version_info, info_get_name
from nvshmem.core import finalize


def test_get_version():
    ffi = cffi.FFI()

    @cuda.jit
    def kernel(arr, name):
        ptr = ffi.from_buffer(arr)
        ptr2 = ffi.from_buffer(arr[1:])
        ptr3 = ffi.from_buffer(arr[2:])
        vendor_get_version_info(ptr, ptr2, ptr3)

        nameptr = ffi.from_buffer(name)
        info_get_name(nameptr)

    arr = np.zeros(3, dtype=np.int32)
    name = np.zeros(100, dtype=np.int8)

    kernel[1, 1](arr, name)
    cuda.synchronize()
    print(f"ver: {arr[0]}.{arr[1]}.{arr[2]}")
    print("".join(chr(i) for i in name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    try:
        test_get_version()
    finally:
        finalize()
