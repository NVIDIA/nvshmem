#!/bin/bash
# NVSHMEM CI test-suite configs
source docker/ci-testsuite-configs/testsuite-utils.sh


MATRIX_TESTS_TO_SKIP=(
    "device/coll/reduce_warp_max" # timeout
    "device/tile/tile_*"
)

MATRIX_TESTS_TO_FAIL=(
    "apps/cufft/alltoall_bw"
    "apps/interop/nccl_nvshmem_interop"
    "device/init/global_exit"
    "device/pt-to-pt/iget"
    "device/pt-to-pt/iput"
    "host/init/global_exit"
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
    "host/pt-to-pt/set"
    "host/pt-to-pt/swap"
    "host/pt-to-pt/xor"
)


function configure_matrix_env() {
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:build/src/lib:$LD_LIBRARY_PATH"


    export NVSHMEM_DEBUG="INFO"
    export NVSHMEM_DEBUG_SUBSYS="ALL"

    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_LIST="mlx5_1:1"

    export NVSHMEM_REMOTE_TRANSPORT="IBDEVX"
    export NVSHMEM_DISABLE_NCCL=1
    export NVSHMEM_SYMMETRIC_SIZE=7516192768
    export NVSHMEMTEST_INIT_NUM_ITERS=75

    export NVSHMEM_BOOTSTRAP_PMI="PMIX"
    export NVSHMEM_TEST_SRUN_FLAGS="--mpi=pmix -n2 --ntasks-per-node=1"

    export MATRIX_SLURM_NODE_COUNT=2
    export MATRIX_SLURM_GPUS_PER_NODE_COUNT=1

    export MATRIX_SKIPPED_VARIANTS="host_pt-to-pt_put:-a -r"

}

