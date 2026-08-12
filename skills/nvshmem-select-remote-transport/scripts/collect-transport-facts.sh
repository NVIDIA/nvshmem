#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect read-only evidence used to select an NVSHMEM transport.
# Exit 0 means the core evidence categories were found; exit 2 means the
# output is still useful but one or more categories are incomplete.

set -u
export LC_ALL=C

usage() {
    printf '%s\n' \
        'Usage: collect-transport-facts.sh [--prefix PATH]' \
        '' \
        'Collect labeled, read-only NVSHMEM, GPU, RDMA, and provider facts.' \
        'Run this on the target compute node. No workload or benchmark is run.'
}

section() {
    printf '\n[%s]\n' "$1"
}

note_missing() {
    diagnostics+=("$1")
}

print_file() {
    local label=$1
    local path=$2
    if [[ -r "$path" ]]; then
        printf '%s: ' "$label"
        tr '\n' ' ' < "$path"
        printf '\n'
        return 0
    fi
    return 1
}

run_probe() {
    local label=$1
    shift
    printf '%s:\n' "$label"
    "$@" 2>&1
    local rc=$?
    printf '%s_exit_status: %d\n' "$label" "$rc"
    return "$rc"
}

prefix_arg=''
while (($#)); do
    case "$1" in
        --prefix)
            if (($# < 2)); then
                printf 'error: --prefix requires PATH\n' >&2
                usage >&2
                exit 64
            fi
            prefix_arg=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

diagnostics=()
version_ok=0
plugins_ok=0
gpu_ok=0
network_ok=0

nvshmem_prefix=''
prefix_source='unresolved'
if [[ -n "$prefix_arg" ]]; then
    nvshmem_prefix=$prefix_arg
    prefix_source='--prefix'
elif [[ -n "${NVSHMEM_PREFIX:-}" ]]; then
    nvshmem_prefix=$NVSHMEM_PREFIX
    prefix_source='NVSHMEM_PREFIX'
elif command -v nvshmem-info >/dev/null 2>&1; then
    info_on_path=$(command -v nvshmem-info)
    nvshmem_prefix=$(dirname "$(dirname "$info_on_path")")
    prefix_source='nvshmem-info on PATH'
fi

section collector
if ! hostname_value=$(hostname 2>/dev/null); then
    hostname_value='unknown'
fi
if ! effective_user=$(id -un 2>/dev/null); then
    effective_user="uid:$(id -u 2>/dev/null || printf unknown)"
fi
printf 'schema: nvshmem-transport-facts-v1\n'
printf 'timestamp_utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname: %s\n' "$hostname_value"
printf 'effective_user: %s\n' "$effective_user"
printf 'prefix_source: %s\n' "$prefix_source"
if [[ -n "$nvshmem_prefix" ]]; then
    printf 'nvshmem_prefix: %s\n' "$nvshmem_prefix"
    if [[ ! -d "$nvshmem_prefix" ]]; then
        note_missing "resolved NVSHMEM prefix is not a readable directory: $nvshmem_prefix"
    fi
else
    printf 'nvshmem_prefix: unresolved\n'
    note_missing 'exact NVSHMEM installation prefix is unresolved'
fi

section nvshmem-version
info_bin=''
if [[ -n "$nvshmem_prefix" && -x "$nvshmem_prefix/bin/nvshmem-info" ]]; then
    info_bin="$nvshmem_prefix/bin/nvshmem-info"
elif command -v nvshmem-info >/dev/null 2>&1; then
    info_bin=$(command -v nvshmem-info)
fi

if [[ -n "$info_bin" ]]; then
    printf 'nvshmem_info: %s\n' "$info_bin"
    version_output=$("$info_bin" -n 2>&1)
    version_rc=$?
    printf '%s\n' "$version_output"
    printf 'nvshmem_info_exit_status: %d\n' "$version_rc"
    if ((version_rc == 0)) && grep -Eq '[0-9]+\.[0-9]+([.][0-9]+)?' <<< "$version_output"; then
        version_ok=1
    fi
else
    printf 'nvshmem_info: not found\n'
fi

if ((version_ok == 0)) && [[ -n "$nvshmem_prefix" ]]; then
    header_evidence=''
    for header in \
        "$nvshmem_prefix/include/nvshmem.h" \
        "$nvshmem_prefix/include/nvshmem_common.h" \
        "$nvshmem_prefix/include/nvshmemx.h"; do
        if [[ -r "$header" ]]; then
            matches=$(grep -E '^#[[:space:]]*define[[:space:]]+NVSHMEM(_VERSION|_MAJOR_VERSION|_MINOR_VERSION|_PATCH_VERSION|_VERSION_MAJOR|_VERSION_MINOR|_VERSION_PATCH)' "$header" 2>/dev/null || true)
            if [[ -n "$matches" ]]; then
                printf 'header: %s\n%s\n' "$header" "$matches"
                header_evidence+="$matches "
            fi
        fi
    done
    if grep -Eq '[0-9]+' <<< "$header_evidence"; then
        version_ok=1
    fi
fi
if ((version_ok == 0)); then
    note_missing 'exact NVSHMEM version was not found'
fi

section nvshmem-plugins
plugin_count=0
if [[ -n "$nvshmem_prefix" && -d "$nvshmem_prefix" ]]; then
    shopt -s nullglob
    for libdir in "$nvshmem_prefix/lib" "$nvshmem_prefix/lib64"; do
        [[ -d "$libdir" ]] || continue
        printf 'searched_directory: %s\n' "$libdir"
        plugin_files=(
            "$libdir"/nvshmem_transport_*.so*
            "$libdir"/libnvshmem_transport_*.so*
        )
        for plugin in "${plugin_files[@]}"; do
            [[ -e "$plugin" || -L "$plugin" ]] || continue
            printf 'plugin: %s\n' "$plugin"
            ((plugin_count += 1))
        done
    done
    shopt -u nullglob
    plugins_ok=1
    printf 'plugin_count_in_selected_prefix: %d\n' "$plugin_count"
else
    printf 'selected_prefix_inventory: unavailable\n'
    if command -v ldconfig >/dev/null 2>&1; then
        printf 'loader_cache_hints (not an exact-prefix inventory):\n'
        ldconfig -p 2>/dev/null | grep -E 'nvshmem.*transport|transport.*nvshmem' || printf 'none\n'
    fi
fi
if ((plugins_ok == 0)); then
    note_missing 'selected-prefix transport plugin inventory is unavailable'
elif ((plugin_count == 0)); then
    printf 'plugin_inventory_result: empty\n'
fi

section nvshmem-configuration
printf 'transport_related_environment:\n'
selection_environment=$(env | grep -E '^(NVSHMEM_(REMOTE_TRANSPORT|IB_ENABLE_IBGDA|GPUNETIO_ENABLE_GDAKI|LIBFABRIC_PROVIDER|BOOTSTRAP|SYMMETRIC_SIZE|DISABLE_CUDA_VMM|ENABLE_NIC_PE_MAPPING|HCA_LIST|HCA_PE_MAPPING)|UCX_TLS|FI_PROVIDER|FI_EFA_ENABLE_SHM_TRANSFER)=' | sort || true)
if [[ -n "$selection_environment" ]]; then
    printf '%s\n' "$selection_environment"
else
    printf 'none\n'
fi
for config_path in \
    "${NVSHMEM_CONFIG_FILE:-}" \
    "${nvshmem_prefix:+$nvshmem_prefix/etc/nvshmem.conf}"; do
    [[ -n "$config_path" ]] || continue
    if [[ -r "$config_path" ]]; then
        printf 'configuration_file: %s\n' "$config_path"
        grep -E '^[[:space:]]*(NVSHMEM_)?(REMOTE_TRANSPORT|IB_ENABLE_IBGDA|GPUNETIO_ENABLE_GDAKI|LIBFABRIC_PROVIDER|DISABLE_CUDA_VMM|HCA_LIST|HCA_PE_MAPPING)[[:space:]]*=' "$config_path" 2>/dev/null || printf 'no transport-selection entries\n'
    fi
done

section gpu
if command -v nvidia-smi >/dev/null 2>&1; then
    if run_probe gpu_inventory nvidia-smi --query-gpu=index,name,pci.bus_id,compute_cap,driver_version --format=csv,noheader; then
        gpu_ok=1
    fi
    if run_probe gpu_topology nvidia-smi topo -m; then
        network_ok=1
    fi
    run_probe gpu_p2p_read nvidia-smi topo -p2p r || true
else
    printf 'nvidia_smi: not found\n'
    note_missing 'GPU and driver evidence is unavailable'
fi
if ((gpu_ok == 0)) && command -v nvidia-smi >/dev/null 2>&1; then
    note_missing 'nvidia-smi did not return a GPU inventory'
fi

section kernel-driver
run_probe uname uname -a || true
print_file nvidia_driver_version /proc/driver/nvidia/version || printf 'nvidia_driver_version: unavailable\n'
if [[ -r /proc/driver/nvidia/params ]]; then
    printf 'PeerMappingOverride: '
    grep -E '^PeerMappingOverride:' /proc/driver/nvidia/params 2>/dev/null || printf 'not present\n'
else
    printf 'PeerMappingOverride: unavailable\n'
fi
for module in nvidia nvidia_peermem nv_peer_mem; do
    if [[ -d "/sys/module/$module" ]]; then
        printf 'module: %s loaded\n' "$module"
        print_file "${module}_version" "/sys/module/$module/version" || true
        print_file "${module}_initstate" "/sys/module/$module/initstate" || true
    else
        printf 'module: %s not_loaded\n' "$module"
    fi
done
if [[ -d /sys/module/dma_buf ]]; then
    printf 'dma_buf_kernel_module: present\n'
else
    printf 'dma_buf_kernel_module: not_visible\n'
fi
if [[ -d /sys/kernel/dmabuf ]]; then
    printf 'dma_buf_sysfs: present\n'
else
    printf 'dma_buf_sysfs: not_visible\n'
fi
kernel_release=$(uname -r 2>/dev/null || true)
for kernel_config in /proc/config.gz "/boot/config-$kernel_release"; do
    [[ -r "$kernel_config" ]] || continue
    printf 'dma_buf_kernel_config: %s\n' "$kernel_config"
    if [[ "$kernel_config" == *.gz ]] && command -v zgrep >/dev/null 2>&1; then
        zgrep -E '^CONFIG_(DMA_SHARED_BUFFER|DMABUF_MOVE_NOTIFY)=' "$kernel_config" 2>/dev/null || printf 'not reported\n'
    else
        grep -E '^CONFIG_(DMA_SHARED_BUFFER|DMABUF_MOVE_NOTIFY)=' "$kernel_config" 2>/dev/null || printf 'not reported\n'
    fi
done
printf 'dma_buf_note: kernel infrastructure alone does not prove a working GPU RDMA registration path\n'
if command -v modinfo >/dev/null 2>&1; then
    for module in nvidia_peermem nv_peer_mem; do
        printf '%s_modinfo:\n' "$module"
        modinfo -F version "$module" 2>&1 || true
    done
fi

section rdma-ports
shopt -s nullglob
ib_devices=(/sys/class/infiniband/*)
if ((${#ib_devices[@]})); then
    network_ok=1
    for ibdev_path in "${ib_devices[@]}"; do
        ibdev=${ibdev_path##*/}
        printf 'rdma_device: %s\n' "$ibdev"
        canonical_device=$(readlink -f "$ibdev_path/device" 2>/dev/null || true)
        [[ -n "$canonical_device" ]] && printf '  pci_device: %s\n' "${canonical_device##*/}"
        print_file '  vendor' "$ibdev_path/device/vendor" || true
        print_file '  device' "$ibdev_path/device/device" || true
        driver_link=$(readlink -f "$ibdev_path/device/driver" 2>/dev/null || true)
        [[ -n "$driver_link" ]] && printf '  driver: %s\n' "${driver_link##*/}"
        net_paths=("$ibdev_path"/device/net/*)
        for net_path in "${net_paths[@]}"; do
            [[ -e "$net_path" ]] && printf '  netdev: %s\n' "${net_path##*/}"
        done
        port_paths=("$ibdev_path"/ports/*)
        for port_path in "${port_paths[@]}"; do
            [[ -d "$port_path" ]] || continue
            port=${port_path##*/}
            printf '  port: %s\n' "$port"
            print_file '    state' "$port_path/state" || true
            print_file '    phys_state' "$port_path/phys_state" || true
            print_file '    link_layer' "$port_path/link_layer" || true
            print_file '    rate' "$port_path/rate" || true
        done
    done
else
    printf 'rdma_device: none visible in /sys/class/infiniband\n'
fi
shopt -u nullglob

section rdma-stack
if command -v ofed_info >/dev/null 2>&1; then
    run_probe ofed_version ofed_info -s || true
else
    printf 'ofed_info: not found\n'
fi
if command -v rdma >/dev/null 2>&1; then
    run_probe rdma_version rdma -V || true
    run_probe rdma_links rdma link show || true
else
    printf 'rdma: not found\n'
fi
if command -v ibv_devinfo >/dev/null 2>&1; then
    run_probe verbs_devices ibv_devinfo -l || true
else
    printf 'ibv_devinfo: not found\n'
fi

section provider-stacks
if command -v ucx_info >/dev/null 2>&1; then
    run_probe ucx_version ucx_info -v || true
    printf 'ucx_transports:\n'
    ucx_info -d 2>&1 | grep -E '^#.*(Transport|Device):' || printf 'not reported\n'
else
    printf 'ucx_info: not found\n'
fi
if command -v fi_info >/dev/null 2>&1; then
    run_probe libfabric_version fi_info --version || true
    if run_probe libfabric_providers fi_info -l; then
        network_ok=1
    fi
else
    printf 'fi_info: not found\n'
fi
if command -v pkg-config >/dev/null 2>&1; then
    for package in libfabric ucx libmlx5 gdrapi gpunetio doca-gpunetio doca; do
        if pkg-config --exists "$package" 2>/dev/null; then
            printf 'pkg_config: %s %s\n' "$package" "$(pkg-config --modversion "$package" 2>/dev/null || printf unknown)"
        else
            printf 'pkg_config: %s not_found\n' "$package"
        fi
    done
else
    printf 'pkg-config: not found\n'
fi
if command -v ldconfig >/dev/null 2>&1; then
    printf 'provider_library_hints:\n'
    ldconfig -p 2>/dev/null | grep -E 'lib(ucp|ucs|uct|fabric|mlx5|efa|cxi|gdrapi|gpunetio|doca)' || printf 'none\n'
fi

section diagnostics
if ((network_ok == 0)); then
    note_missing 'neither GPU topology nor an RDMA/libfabric provider inventory was available'
fi
if ((${#diagnostics[@]})); then
    for diagnostic in "${diagnostics[@]}"; do
        printf 'partial: %s\n' "$diagnostic"
    done
else
    printf 'none\n'
fi

if ((version_ok && plugins_ok && gpu_ok && network_ok)); then
    printf 'collector_status: complete\n'
    exit 0
fi

printf 'collector_status: partial\n'
exit 2
