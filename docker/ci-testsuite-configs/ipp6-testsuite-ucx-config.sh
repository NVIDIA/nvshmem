#!/bin/bash
# NVSHMEM CI test-suite configs
source docker/ci-testsuite-configs/testsuite-utils.sh


# skip failing tests for now MATRIX_TESTS_TO_FAIL=(
MATRIX_TESTS_TO_SKIP=(
    "apps/cufft/alltoall_bw"
    "device/pt-to-pt/atomic_fetch"
    "device/pt-to-pt/iget"
    "device/pt-to-pt/iput"
    "device/pt-to-pt/putsize_nbi"
)


function configure_matrix_env() {
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:build/src/lib:$LD_LIBRARY_PATH"

    export NCCL_DEBUG="INFO"
    export NCCL_DEBUG_SUBSYS="ALL"

    export NCCL_NET="IB"
    export NCCL_OOB_NET_ENABLE=1
    export NCCL_OOB_NET_IFNAME="mlx5_1:1"

    export NVSHMEM_DEBUG="INFO"
    export NVSHMEM_DEBUG_SUBSYS="ALL"

    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    export NVSHMEM_HCA_LIST="mlx5_1:1"

    export NVSHMEMTEST_LOCAL_CUDA_BUFFER=1
    export NVSHMEM_SYMMETRIC_SIZE=7516192768
    export NVSHMEM_REMOTE_TRANSPORT="UCX"

    export NVSHMEM_BOOTSTRAP_PMI="PMI2"
    export NVSHMEM_TEST_SRUN_FLAGS="--mpi=pmi2 -n2 --ntasks-per-node=1"

    export MATRIX_SLURM_NODE_COUNT=2
    export MATRIX_SLURM_GPUS_PER_NODE_COUNT=1
#    export MATRIX_SKIPPED_VARIANTS="host_pt-to-pt_put:-a"
}


