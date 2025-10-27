#!/bin/bash
# NVSHMEM CI test-suite configs
source docker/ci-testsuite-configs/testsuite-utils.sh


MATRIX_TESTS_TO_SKIP=(
    "apps/cufft/alltoall_bw"
    "apps/interop/nccl_nvshmem_interop"
    "device/coll/alltoall"
    "device/coll/broadcast"
    "device/coll/fcollect"
    "device/coll/reduce_block_and"
    "device/coll/reduce_block_max"
    "device/coll/reduce_block_min"
    "device/coll/reduce_block_or"
    "device/coll/reduce_block_prod"
    "device/coll/reduce_block_sum"
    "device/coll/reduce_block_xor"
    "device/coll/reduce_thread_and"
    "device/coll/reduce_thread_max"
    "device/coll/reduce_thread_min"
    "device/coll/reduce_thread_or"
    "device/coll/reduce_thread_prod"
    "device/coll/reduce_thread_sum"
    "device/coll/reduce_thread_xor"
    "device/coll/reduce_warp_and"
    "device/coll/reduce_warp_max"
    "device/coll/reduce_warp_min"
    "device/coll/reduce_warp_or"
    "device/coll/reduce_warp_prod"
    "device/coll/reduce_warp_sum"
    "device/coll/reduce_warp_xor"
    "device/coll/reducescatter_block_and"
    "device/coll/reducescatter_block_max"
    "device/coll/reducescatter_block_min"
    "device/coll/reducescatter_block_or"
    "device/coll/reducescatter_block_prod"
    "device/coll/reducescatter_block_sum"
    "device/coll/reducescatter_block_xor"
    "device/coll/reducescatter_thread_and"
    "device/coll/reducescatter_thread_max"
    "device/coll/reducescatter_thread_min"
    "device/coll/reducescatter_thread_or"
    "device/coll/reducescatter_thread_prod"
    "device/coll/reducescatter_thread_sum"
    "device/coll/reducescatter_thread_xor"
    "device/coll/reducescatter_warp_and"
    "device/coll/reducescatter_warp_max"
    "device/coll/reducescatter_warp_min"
    "device/coll/reducescatter_warp_or"
    "device/coll/reducescatter_warp_prod"
    "device/coll/reducescatter_warp_sum"
    "device/coll/reducescatter_warp_xor"
    "device/pt-to-pt/atomic_add"
    "device/pt-to-pt/atomic_and"
    "device/pt-to-pt/atomic_compare_swap"
    "device/pt-to-pt/atomic_fetch"
    "device/pt-to-pt/atomic_inc"
    "device/pt-to-pt/atomic_or"
    "device/pt-to-pt/atomic_set"
    "device/pt-to-pt/atomic_swap"
    "device/pt-to-pt/atomic_xor"
    "device/pt-to-pt/iget"
    "device/pt-to-pt/iput"
    "device/pt-to-pt/put_signal"
    "device/pt-to-pt/put_signal_nbi"
    "device/pt-to-pt/set"
    "device/pt-to-pt/swap"
    "device/sync/test_some"
    "device/sync/wait_until_any"
    "device/tile/tile_*"
    "host/coll/alltoall"
    "host/coll/broadcast"
    "host/coll/fcollect"
    "host/coll/reduce"
    "host/coll/reducescatter"
    "host/init/shmem_init"
    "host/pt-to-pt/add"
    "host/pt-to-pt/and"
    "host/pt-to-pt/cswap"
    "host/pt-to-pt/fadd"
    "host/pt-to-pt/fand"
    "host/pt-to-pt/fetch"
    "host/pt-to-pt/finc"
    "host/pt-to-pt/for"
    "host/pt-to-pt/fxor"
    "host/pt-to-pt/g"
    "host/pt-to-pt/iget"
    "host/pt-to-pt/inc"
    "host/pt-to-pt/iput"
    "host/pt-to-pt/or"
    "host/pt-to-pt/put"
    "host/pt-to-pt/set"
    "host/pt-to-pt/swap"
    "host/pt-to-pt/xor"
)

MATRIX_TESTS_TO_FAIL=(
    "device/init/global_exit"
    "host/init/global_exit"
)


function configure_matrix_env() {
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:build/src/lib:$LD_LIBRARY_PATH"

    export NVSHMEM_DEBUG="INFO"
    export NVSHMEM_DEBUG_SUBSYS="ALL"

    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_LIST="mlx5_1:1"

    export NVSHMEM_REMOTE_TRANSPORT="IBRC"
    export NVSHMEM_SYMMETRIC_SIZE=7516192768
    export NVSHMEM_HEAP_KIND="SYSMEM"
    export NVSHMEM_DISABLE_GDRCOPY=1
    export NVSHMEM_DISABLE_NCCL=1
    export NVSHMEM_ENABLE_RAIL_OPT=1
    export NVSHMEMTEST_INIT_NUM_ITERS=20

    export NVSHMEM_BOOTSTRAP_PMI="PMIX"
    export NVSHMEM_TEST_SRUN_FLAGS="--mpi=pmix -n2 --ntasks-per-node=1"

    export MATRIX_SLURM_PARTITION="l40s-cicd"
    export MATRIX_SLURM_NODE_COUNT=2
    export MATRIX_SLURM_GPUS_PER_NODE_COUNT=1

    export MATRIX_SKIPPED_VARIANTS="host_pt-to-pt_get:-a"
}

