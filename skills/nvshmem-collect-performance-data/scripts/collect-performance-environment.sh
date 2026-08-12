#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only, unprivileged evidence collector for NVSHMEM performance environments and reports.
# Emit all evidence to stdout; do not write files or change the target system.

set -uo pipefail
exec 2>&1

partial=0
diagnostics=()
last_output=""
last_status=0

section() {
    printf '\n[%s]\n' "$1"
}

add_diagnostic() {
    diagnostics+=("$1")
}

print_command() {
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
}

capture_command() {
    print_command "$@"
    last_output="$("$@" 2>&1)"
    last_status=$?
    printf 'exit_status=%d\n' "$last_status"
    if [[ -n "$last_output" ]]; then
        printf '%s\n' "$last_output"
    else
        printf 'output=(empty)\n'
    fi
}

print_path() {
    local command_name=$1
    local resolved
    resolved="$(command -v "$command_name" 2>/dev/null || true)"
    printf '%s=%s\n' "$command_name" "${resolved:-unavailable}"
}

os_release_value() {
    local key=$1
    if [[ -r /etc/os-release ]]; then
        sed -n "s/^${key}=//p" /etc/os-release | sed -n '1p' | tr -d '"'
    fi
}

section metadata
printf 'collector_version=1\n'
printf 'utc_time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || true)"
printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
printf 'kernel=%s\n' "$(uname -srmo 2>/dev/null || true)"
printf 'os_id=%s\n' "$(os_release_value ID)"
printf 'os_version=%s\n' "$(os_release_value VERSION_ID)"
printf 'cpu_arch=%s\n' "$(uname -m 2>/dev/null || true)"
if command -v lscpu >/dev/null 2>&1; then
    printf 'cpu_model=%s\n' "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | sed -n '1p')"
    printf 'cpu_numa_nodes=%s\n' "$(lscpu 2>/dev/null | sed -n 's/^NUMA node(s):[[:space:]]*//p' | sed -n '1p')"
else
    printf 'cpu_model=unknown\n'
    printf 'cpu_numa_nodes=unknown\n'
fi

container_type=none
if [[ -f /.dockerenv ]]; then
    container_type=docker-compatible
elif [[ -n "${APPTAINER_NAME:-}" || -n "${SINGULARITY_NAME:-}" ]]; then
    container_type=apptainer-or-singularity
elif [[ -r /proc/1/cgroup ]] && grep -Eq 'docker|containerd|kubepods|libpod' /proc/1/cgroup 2>/dev/null; then
    container_type=container-cgroup
fi
printf 'container=%s\n' "$container_type"

section nvshmem
print_path nvshmem-info
printf 'NVSHMEM_PREFIX=%s\n' "${NVSHMEM_PREFIX:-unset}"
printf 'NVSHMEM_HOME=%s\n' "${NVSHMEM_HOME:-unset}"
if command -v nvshmem-info >/dev/null 2>&1; then
    capture_command nvshmem-info -n -b
    if ((last_status != 0)); then
        add_diagnostic 'nvshmem-info -n -b failed; attempted version-only fallback.'
        capture_command nvshmem-info -n
        if ((last_status != 0)); then
            partial=1
            add_diagnostic 'nvshmem-info could not report the installed version.'
        fi
    fi
else
    partial=1
    add_diagnostic 'nvshmem-info is unavailable; load the target NVSHMEM environment.'
fi

section cuda-and-gpus
print_path nvidia-smi
printf 'CUDA_HOME=%s\n' "${CUDA_HOME:-unset}"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    capture_command nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv,noheader
    if ((last_status != 0)); then
        partial=1
        add_diagnostic 'GPU inventory failed; run on an allocated GPU compute node.'
    else
        gpu_count="$(printf '%s\n' "$last_output" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
        printf 'gpu_count=%s\n' "$gpu_count"
    fi
    capture_command nvidia-smi topo -m
    if ((last_status != 0)); then
        partial=1
        add_diagnostic 'nvidia-smi topo -m failed; GPU/NIC topology is incomplete.'
    fi
else
    partial=1
    add_diagnostic 'nvidia-smi is unavailable; run on an allocated GPU compute node.'
fi

print_path nvcc
if command -v nvcc >/dev/null 2>&1; then
    capture_command nvcc --version
fi
if [[ -r /proc/driver/nvidia/version ]]; then
    printf 'driver_module=%s\n' "$(sed -n '1p' /proc/driver/nvidia/version)"
else
    printf 'driver_module=unavailable\n'
fi
if [[ -d /sys/module/nvidia_peermem ]]; then
    printf 'nvidia_peermem=loaded\n'
else
    printf 'nvidia_peermem=not-detected\n'
fi

section network-interfaces
if command -v ip >/dev/null 2>&1; then
    capture_command ip -brief link show
else
    printf 'ip=unavailable\n'
    add_diagnostic 'ip is unavailable; general network-interface evidence is incomplete.'
fi

section rdma-ports
shopt -s nullglob
port_paths=(/sys/class/infiniband/*/ports/*)
ports_found=0
for port_path in "${port_paths[@]}"; do
    [[ -d "$port_path" ]] || continue
    ports_found=$((ports_found + 1))

    hca_dir=${port_path%/ports/*}
    hca_name=${hca_dir##*/}
    port_number=${port_path##*/}
    state=unknown
    physical_state=unknown
    link_layer=unknown
    rate=unknown
    [[ -r "$port_path/state" ]] && state="$(<"$port_path/state")"
    [[ -r "$port_path/phys_state" ]] && physical_state="$(<"$port_path/phys_state")"
    [[ -r "$port_path/link_layer" ]] && link_layer="$(<"$port_path/link_layer")"
    [[ -r "$port_path/rate" ]] && rate="$(<"$port_path/rate")"

    pci_device=unknown
    device_path="$(readlink -f "$hca_dir/device" 2>/dev/null || true)"
    [[ -n "$device_path" ]] && pci_device=${device_path##*/}

    net_names=()
    net_paths=("$hca_dir"/device/net/*)
    for net_path in "${net_paths[@]}"; do
        net_names+=("${net_path##*/}")
    done
    netdev=none
    if ((${#net_names[@]} > 0)); then
        netdev="$(IFS=,; printf '%s' "${net_names[*]}")"
    fi

    printf 'hca=%q port=%q pci=%q state=%q physical_state=%q link_layer=%q rate=%q netdev=%q\n' \
        "$hca_name" "$port_number" "$pci_device" "$state" "$physical_state" \
        "$link_layer" "$rate" "$netdev"
done
printf 'ports_found=%d\n' "$ports_found"
if ((ports_found == 0)); then
    add_diagnostic 'No RDMA ports were visible under /sys/class/infiniband; this may be valid for a non-RDMA system.'
fi

section rdma-tools
print_path rdma
print_path ibv_devinfo
print_path ibv_devices
if command -v rdma >/dev/null 2>&1; then
    capture_command rdma link show
fi
if command -v ibv_devices >/dev/null 2>&1; then
    capture_command ibv_devices
fi
if command -v ibv_devinfo >/dev/null 2>&1; then
    capture_command ibv_devinfo -v
fi

section launcher-and-allocation
for launcher in srun nvshmrun mpirun mpiexec oshrun; do
    print_path "$launcher"
done
if command -v srun >/dev/null 2>&1; then
    capture_command srun --version
fi
if command -v mpirun >/dev/null 2>&1; then
    capture_command mpirun --version
fi
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unset}"
printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST:-unset}"
printf 'SLURM_NNODES=%s\n' "${SLURM_NNODES:-unset}"
printf 'SLURM_NTASKS=%s\n' "${SLURM_NTASKS:-unset}"
printf 'SLURM_TASKS_PER_NODE=%s\n' "${SLURM_TASKS_PER_NODE:-unset}"
printf 'SLURM_GPUS=%s\n' "${SLURM_GPUS:-unset}"
printf 'SLURM_GPUS_ON_NODE=%s\n' "${SLURM_GPUS_ON_NODE:-unset}"

section relevant-environment
env | LC_ALL=C sort | awk -F= '
    $1 ~ /^NVSHMEM_/ ||
    $1 == "CUDA_HOME" ||
    $1 == "CUDA_VISIBLE_DEVICES" ||
    $1 == "LD_LIBRARY_PATH" ||
    $1 ~ /^NCCL_/ ||
    $1 ~ /^UCX_/ ||
    $1 == "FI_PROVIDER" ||
    $1 ~ /^FI_CXI_/ ||
    $1 ~ /^OMPI_/ ||
    $1 ~ /^PMI_/ ||
    $1 ~ /^PMIX_/ ||
    $1 ~ /^SLURM_(JOB_ID|JOB_NODELIST|NNODES|NTASKS|TASKS_PER_NODE|GPUS|GPUS_ON_NODE|GPUS_PER_TASK|LOCALID|PROCID)$/
'

section diagnostics
if ((${#diagnostics[@]} == 0)); then
    printf 'message=none\n'
else
    for diagnostic in "${diagnostics[@]}"; do
        printf 'message=%q\n' "$diagnostic"
    done
fi

if ((partial == 0)); then
    printf 'status=complete\n'
    exit 0
fi

printf 'status=partial\n'
printf 'next_action=%q\n' 'Run the collector in the target NVSHMEM environment on every participating compute node and retain the complete output.'
exit 2
