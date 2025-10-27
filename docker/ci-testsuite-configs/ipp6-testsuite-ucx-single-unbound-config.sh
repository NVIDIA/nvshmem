#!/bin/bash
# NVSHMEM CI test-suite configs
source docker/ci-testsuite-configs/testsuite-utils.sh


# skip failing tests for now MATRIX_TESTS_TO_FAIL=(
MATRIX_TESTS_TO_SKIP=(
#    "apps/cufft/alltoall_bw"
#    "device/pt-to-pt/atomic_fetch"
#    "device/pt-to-pt/iget"
#    "device/pt-to-pt/iput"
#
#    "host/coll/alltoall"
#    "host/coll/broadcast"
#    "host/coll/fcollect"
#    "host/coll/reduce"
#    "host/coll/reducescatter"
)


function configure_matrix_env() {
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:build/src/lib:$LD_LIBRARY_PATH"

    export NVSHMEM_DEBUG="INFO"
    export NVSHMEM_DEBUG_SUBSYS="ALL"

    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_LIST="mlx5_1:1"

    export NVSHMEM_REMOTE_TRANSPORT="UCX"
    export NVSHMEM_SYMMETRIC_SIZE=7516192768

    export NVSHMEM_BOOTSTRAP_PMI="PMI"
#    export NVSHMEM_TEST_SRUN_FLAGS="--mpi=pmi -n2 --ntasks-per-node=2"
    export NVSHMEM_TEST_SRUN_FLAGS="-n2 --ntasks-per-node=2"

    export NVSHMEM_DISABLE_NCCL=1

    export MATRIX_SLURM_PARTITION="a100x8-cicd"
    export MATRIX_SLURM_NODE_COUNT=1
    export MATRIX_SLURM_GPUS_PER_NODE_COUNT=2

#    export MATRIX_SKIPPED_VARIANTS="host_pt-to-pt_put:-a"
}


