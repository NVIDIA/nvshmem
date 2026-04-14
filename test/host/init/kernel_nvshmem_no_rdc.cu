/*
 * Non-RDC variant of kernel_nvshmem.cu.
 *
 * Compiled WITHOUT -rdc=true and WITH -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING.
 * All device functions are inlined from headers; no device linking required.
 * Device state must be populated at runtime via nvshmemx_cumodule_init.
 */
#include <nvshmem.h>
#include <cstdio>

extern "C" __global__ void kernel_nvshmem(int *destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    assert(npes > 0);
    int peer = (mype + 1) % npes;
    nvshmem_int_p(destination, 3 * peer + 14, peer);
    nvshmem_barrier_all();
}
