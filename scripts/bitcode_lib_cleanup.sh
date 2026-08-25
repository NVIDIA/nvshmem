#!/usr/bin/env bash
#
# Post-processes a disassembled LLVM IR file (.ll) before final llvm-as assembly.
#
# Usage: bitcode_lib_cleanup.sh <input.ll> <output.ll>

INPUT=$1
OUTPUT=$2

# Step 1: Strip nvvm-reflect-ftz metadata.
#
# The nvvm-reflect-ftz flag controls flush-to-zero behavior for floating point
# operations. It must not be baked into the bitcode library as it would override
# the application's own FTZ setting at link time.
FTZ_NODE="$(grep -E '!([0-9]+) = !\{[^"]+"nvvm-reflect-ftz"' "$INPUT" | cut -d ' ' -f 1)"
awk '!/nvvm-reflect-ftz/' "$INPUT" \
    | { [ -n "$FTZ_NODE" ] && sed "/^\!llvm\.module\.flags = /s/$FTZ_NODE, //" || cat; } \
    > "$OUTPUT.tmp"

# Compatibility was validated between NVSHMEM bitcode produced by LLVM 22.1.8
# and CUTLASS 4.6.1.
#
# Step 2: Replace an NVPTX intrinsic that fails in CUTLASS 4.6.1's CUDA 12
# libNVVM backend when linked from the LLVM 22.1.8 bitcode.
#
# Replacements applied:
#   - @llvm.nvvm.activemask()
#       -> inline PTX asm "activemask.b32"
#       The inline asm bypasses CUTLASS's incompatible intrinsic translation.
#
#   - Remove the declaration for the replaced intrinsic.
sed \
    -e 's/\(tail \)\{0,1\}call noundef i32 @llvm\.nvvm\.activemask()/call i32 asm sideeffect "activemask.b32 $0;", "=r"()/g' \
    -e '/^declare i32 @llvm\.nvvm\.activemask/d' \
    "$OUTPUT.tmp" > "$OUTPUT"

rm -f "$OUTPUT.tmp"
