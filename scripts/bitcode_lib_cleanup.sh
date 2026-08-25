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

# TODO: Re-evaluate Steps 2 and 3 once CUTLASS can consume NVSHMEM bitcode
# produced by LLVM 22.1.8 without these workarounds. CUTLASS 4.7.0's
# LLVM 23 RC (23a60f15f2fcafcf67b95b0a035053579958b732) compilation path is
# currently partly incompatible with the affected NVPTX intrinsics in that
# bitcode.
#
# Steps 2 and 3: Replace NVPTX intrinsics that fail in CUTLASS 4.7.0's CUDA 12
# libNVVM backend when linked from the LLVM 22.1.8 bitcode.
#
# Replacements applied:
#   - @llvm.nvvm.activemask()
#       -> inline PTX asm "activemask.b32"
#       The inline asm bypasses CUTLASS's incompatible intrinsic translation.
#
#   - @llvm.nvvm.isspacep.local(ptr %pointer)
#       -> inline PTX asm "isspacep.local"
#       The inline asm preserves the operation while bypassing CUTLASS's
#       incompatible intrinsic translation.
#
#   - Remove declarations for both replaced intrinsics.
sed \
    -e 's/\(tail \)\{0,1\}call noundef i32 @llvm\.nvvm\.activemask()/call i32 asm sideeffect "activemask.b32 $0;", "=r"()/g' \
    -e '/^declare i32 @llvm\.nvvm\.activemask/d' \
    -e 's/\(tail \)\{0,1\}call \(noundef \)\{0,1\}i1 @llvm\.nvvm\.isspacep\.local(ptr \([^)]*\))/call i1 asm sideeffect "isspacep.local $0, $1;", "=b,l"(ptr \3)/g' \
    -e '/^declare i1 @llvm\.nvvm\.isspacep\.local/d' \
    "$OUTPUT.tmp" > "$OUTPUT"

rm -f "$OUTPUT.tmp"
