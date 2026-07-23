#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source_file="${NVSHMEM_FORTRAN_BITCODE_SOURCE:-${script_dir}/bitcode_fortran_smoke.CUF}"
bitcode="${NVSHMEM_FORTRAN_BITCODE_FILE:?NVSHMEM_FORTRAN_BITCODE_FILE is required}"
arch="${NVSHMEM_FORTRAN_BITCODE_ARCH:-80}"
symbols="${NVSHMEM_FORTRAN_BITCODE_SYMBOLS:-nvshmem_int32_sum_reduce nvshmemx_int32_sum_reduce_warp nvshmemx_int32_sum_reduce_block nvshmem_int32_put nvshmemx_int32_put_warp nvshmemx_int32_put_block}"

nvfortran="${NVFORTRAN:-nvfortran}"
llvm_link="${LLVM_LINK:-llvm-link}"
llvm_nm="${LLVM_NM:-llvm-nm}"
llvm_opt="${LLVM_OPT:-opt}"
llc="${LLC:-llc}"
if [ -n "${PTXAS:-}" ]; then
    ptxas=$PTXAS
elif [ -n "${CUDA_HOME:-}" ]; then
    ptxas="${CUDA_HOME}/bin/ptxas"
else
    ptxas=ptxas
fi

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_file() {
    [ -f "$1" ] || die "$2 not found: $1"
}

require_tool() {
    "$1" --version >/dev/null 2>&1 || die "required tool is not runnable: $1"
}

require_file "$source_file" "CUDA Fortran source"
require_file "$bitcode" "bitcode artifact"
[[ "$arch" =~ ^[0-9]+$ ]] || die "NVSHMEM_FORTRAN_BITCODE_ARCH must be numeric: $arch"

for tool in "$nvfortran" "$llvm_link" "$llvm_nm" "$llvm_opt" "$llc" "$ptxas"; do
    require_tool "$tool"
done

if [ -n "${NVSHMEM_PTX_FEATURE:-}" ]; then
    ptx_feature=$NVSHMEM_PTX_FEATURE
else
    cuda_major=$("$ptxas" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    case "$cuda_major" in
        13) ptx_feature=ptx86 ;;
        12) ptx_feature=ptx82 ;;
        11) ptx_feature=ptx78 ;;
        *) die "cannot infer PTX feature from $ptxas --version; set NVSHMEM_PTX_FEATURE" ;;
    esac
fi

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/nvshmem-fortran-bitcode.XXXXXX")
trap 'rm -rf -- "$work_dir"' EXIT

cp "$source_file" "$work_dir/"
cd "$work_dir"

source_name=$(basename "$source_file")
module_arg=()
if [ -n "${NVSHMEM_FORTRAN_MODULE_DIR:-}" ]; then
    module_arg=(-I "$NVSHMEM_FORTRAN_MODULE_DIR")
fi

"$nvfortran" -cuda -gpu="cc${arch},keep" -c "${module_arg[@]}" "$source_name"

shopt -s nullglob
gpu_ir_files=( *.gpu )
shopt -u nullglob
[ "${#gpu_ir_files[@]}" -eq 1 ] || die "expected one nvfortran .gpu LLVM IR file, found ${#gpu_ir_files[@]}"

gpu_ir=${gpu_ir_files[0]}
for symbol in $symbols; do
    grep -q -F "@${symbol}(" "$gpu_ir" || die "nvfortran did not emit expected device call: $symbol"
done

"$llvm_link" --only-needed "$gpu_ir" "$bitcode" -o linked.bc

for symbol in $symbols; do
    if "$llvm_nm" --undefined-only linked.bc | awk -v symbol="$symbol" '$NF == symbol { found = 1 } END { exit found ? 0 : 1 }'; then
        die "$bitcode left CUDA Fortran device call unresolved: $symbol"
    fi
done

"$llvm_opt" --passes='internalize,inline,mem2reg' linked.bc -o optimized.bc
"$llc" -mcpu="sm_${arch}" -mattr="$ptx_feature" optimized.bc -o linked.ptx
"$ptxas" -arch="sm_${arch}" linked.ptx -o linked.cubin

echo "PASS: CUDA Fortran device IR links and lowers with $bitcode (sm_${arch})"
