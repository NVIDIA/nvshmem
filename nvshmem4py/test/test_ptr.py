from numba import cuda, int32
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.cuda.extending import intrinsic

# from numba.cuda.np.numpy_support import carray
import cupy as cp
import argparse

from utils import uid_init, mpi_init

import nvshmem
from nvshmem.bindings.device.numba import ptr, n_pes, my_pe, sync_all, int_g
from numba.cuda.np.numpy_support import carray

import cffi

ffi = cffi.FFI()


@intrinsic
def _noneptr_as_typedptr(typingctx, optr, arrty):
    """
    Bitcast an opaque none* (void*) back to T* using arr.dtype.
    """
    if not (isinstance(optr, types.CPointer) and optr.dtype is types.none):
        return None
    if not isinstance(arrty, types.Array):
        return None

    ret = types.CPointer(arrty.dtype)
    sig = ret(optr, arrty)

    def codegen(cgctx, builder, sig, args):
        optr_val, _ = args
        ret_llty = cgctx.get_value_type(sig.return_type)  # T* type
        return builder.bitcast(optr_val, ret_llty)  # T*

    return sig, codegen


@intrinsic
def _array_data_noneptr(typingctx, arrty):
    """
    Return the data pointer of a Numba Array as an opaque none* (CPointer(none)).
    CUDA 0.21+: arr.data is T**; one load yields T*. We then bitcast to none*.
    """
    if not isinstance(arrty, types.Array):
        return None
    sig = types.CPointer(types.uint8)(arrty)

    def codegen(cgctx, builder, sig, args):
        arr_val = args[0]
        ary = arrayobj.make_array(arrty)(cgctx, builder, arr_val)
        optr = builder.bitcast(ary.data, cgutils.voidptr_t)  # i8*
        return optr

    return sig, codegen


def test_ptr():

    @cuda.jit()
    def kernel_nvshmem(arr):
        mype = my_pe()
        # npes = n_pes()

        arr_ptr = _array_data_noneptr(arr)
        # arr_ptr = ffi.from_buffer(arr)
        # void_ptr = _any_pointer_to_void_pointer(arr_ptr)
        # peer = int32((mype + 1) % npes)

        other_ptr = ptr(arr_ptr, 1)

        other_ptr_typed = _noneptr_as_typedptr(other_ptr, arr)

        peer_array = carray(other_ptr_typed, arr.shape, arr.dtype)

        # other_ptr = ptr(void_ptr, 1)
        # sync_all()

    dest = nvshmem.core.array((1, ), dtype="int16")

    kernel_nvshmem[1, 1](dest)

    nvshmem.core.free_array(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-type", "-i", type=str, help="Init type to use", choices=["mpi", "uid"], default="uid")
    args = parser.parse_args()
    if args.init_type == "uid":
        uid_init()
    elif args.init_type == "mpi":
        mpi_init()

    test_ptr()
