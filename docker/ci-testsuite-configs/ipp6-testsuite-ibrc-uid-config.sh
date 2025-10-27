#!/bin/bash
# NVSHMEM CI test-suite configs
source docker/ci-testsuite-configs/testsuite-utils.sh


MATRIX_TESTS_TO_SKIP=(
    "apps/cufft/cufft_smoke_test"
    "device/pt-to-pt/atomic_fetch"
    "device/pt-to-pt/get"
    "device/pt-to-pt/get_nbi"
    "device/pt-to-pt/getmem"
    "device/pt-to-pt/getmem_nbi"
    "device/pt-to-pt/getsize"
    "device/pt-to-pt/getsize_nbi"
    "device/pt-to-pt/iget"
    "device/pt-to-pt/iput"
    "device/pt-to-pt/p"
    "device/pt-to-pt/put"
    "device/pt-to-pt/put_nbi"
    "device/pt-to-pt/putmem"
    "device/pt-to-pt/putsize"
    "device/pt-to-pt/putsize_nbi"
    "device/pt-to-pt/set"
    "device/pt-to-pt/signal"
    "device/pt-to-pt/signal_add"
    "device/pt-to-pt/swap"
    "device/tile/tile_reduce"
    "device/tile/tile_allreduce"
    "device/tile/tile_allreduce_1D"
    "device/tile/tile_allreduce_pred"
    "device/tile/tile_allgather"
    "device/tile/tile_allgather_1D"
    "device/tile/tile_allgather_pred"
    "host/coll/alltoall"
    "host/coll/broadcast"
    "host/coll/collective_launch_choose_grid"
    "host/coll/collective_launch_user_specified_grid"
    "host/coll/fcollect"
    "host/coll/reduce"
    "host/init/cuobject_init"
    "host/init/global_exit"
    "host/init/init_loop"
    "host/init/mpi_init"
    "host/init/nvshmemx_hostlib_init_attr"
    "host/init/nvshmemx_init_status"
    "host/init/shmem_init"
    "host/init/static_init"
    "host/interop/app"
    "host/interop/simplelib1"
    "host/interop/simplelib2"
    "host/mem/register_buffer"
)

# unused from testrunner config
#NVSHMEM_EXPECTED_FAILURES=(
#    "apps/cufft/alltoall_bw"
#    "device/pt-to-pt/iget"
#    "device/pt-to-pt/iput"
#)

MATRIX_TESTS_TO_FAIL=(
##    "device/init/global_exit"
##    "host/init/global_exit"
##    "host/init/shmem_init"
#    "apps/cufft/alltoall_bw"
#    "host/pt-to-pt/add"
#    "host/pt-to-pt/and"
#    "host/pt-to-pt/cswap"
#    "host/pt-to-pt/fadd"
#    "host/pt-to-pt/fand"
#    "host/pt-to-pt/fetch"
#    "host/pt-to-pt/finc"
#    "host/pt-to-pt/for"
#    "host/pt-to-pt/fxor"
#    "host/pt-to-pt/g"
#    "host/pt-to-pt/iget"
#    "host/pt-to-pt/inc"
#    "host/pt-to-pt/iput"
#    "host/pt-to-pt/or"
#    "host/pt-to-pt/set"
#    "host/pt-to-pt/swap"
#    "host/pt-to-pt/xor"
)


function configure_matrix_env() {
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:build/src/lib:$LD_LIBRARY_PATH"

    export NVSHMEM_DEBUG="INFO"
    export NVSHMEM_DEBUG_SUBSYS="ALL"

    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_LIST="mlx5_1:1"

    export NVSHMEM_REMOTE_TRANSPORT="IBRC"
    export NVSHMEM_BOOTSTRAP="UID"

    export NVSHMEMTEST_USE_UID_BOOTSTRAP=1
    export NVSHMEMTEST_INIT_NUM_ITERS=10

    export NVSHMEM_SYMMETRIC_SIZE=7516192768

    export NVSHMEM_DISABLE_NCCL=1

    export NVSHMEM_BOOTSTRAP_PMI="PMIX"
    export NVSHMEM_TEST_SRUN_FLAGS="--mpi=pmix -n2 --ntasks-per-node=1"

    export MATRIX_SLURM_PARTITION="l40s-cicd"
    export MATRIX_SLURM_NODE_COUNT=2
    export MATRIX_SLURM_GPUS_PER_NODE_COUNT=1

#    export MATRIX_SKIPPED_VARIANTS="host_pt-to-pt_put:-a"
}


