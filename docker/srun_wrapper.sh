#!/bin/bash -x
# use srun to run a script
script_type="$1"

if [ ! -d "docker" ]; then
  echo "Please launch from the top of the source tree"
  exit 1
fi

build_script="build_nvshmem_${script_type}.sh"

if [ ! -x "docker/$build_script" ]; then
  echo "Check script docker/$build_script is not found"
  exit 2
fi

source docker/cluster_utils.sh

build_cluster_tag="$(identify_build_cluster)"
source_cluster_config $build_cluster_tag

build_image_version="$(get_build_image_version)"

#export DOCKER_JOB_COMMAND="$(get_docker_job_command $(hostname))"

# recreate as needed
mkdir -p build

# assuming local drives or NFS mounted home dir
current_dir=$(realpath .)

# extract file ownership data
#export DOCKER_USER_ID=$(stat --format %u $0)
#export DOCKER_GROUP_ID=$(stat --format %g $0)

# Disable GPU detection to allow enroot to launch this on CPU-only nodes
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration
export NVIDIA_VISIBLE_DEVICES=void

eval "$(get_srun_command $current_dir $build_image_version /nvshmem/docker/${build_script})"
