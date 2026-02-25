from numba import cuda
import cupy as cp
import argparse

from utils import uid_init, mpi_init

from nvshmem.bindings.device.numba import n_pes, sync_all


def test_npe():

    @cuda.jit()
    def kernel_nvshmem(destination):
        npes = n_pes()
        sync_all()
        destination[0] = npes

    npes = cp.zeros(1, dtype="int32")
    kernel_nvshmem[1, 1](npes)

    assert npes[0] > 0
    print(f"{npes[0]=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_npe()
