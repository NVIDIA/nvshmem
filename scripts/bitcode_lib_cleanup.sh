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

# TODO: Re-evaluate Steps 2 and 3 once CUTLASS ships with LLVM >= 21.1.0. If
# CUTLASS's bundled LLVM has been updated past that version, these replacements
# can likely be removed. See: https://nvbugspro.nvidia.com/bug/5983831
#
# Steps 2 and 3: Replace NVPTX intrinsics introduced in LLVM 21.1.0 that are
# absent from CUTLASS's bundled LLVM build. When CUTLASS's JIT encounters an
# unknown intrinsic it emits a .nvvm fallback annotation in the PTX output that
# ptxas cannot parse, producing "Parsing error near '.nvvm': syntax error".
#
# Replacements applied:
#   - @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
#       -> @llvm.nvvm.barrier0()
#       LLVM 21.1.0 emits this for __syncthreads() (introduced May 2025).
#       @llvm.nvvm.barrier0 (PTX bar.sync 0) is the compatible equivalent.
#
#   - @llvm.nvvm.activemask()
#       -> inline PTX asm "activemask.b32"
#       LLVM 21.1.0 emits this for __activemask() whereas LLVM 18 emits inline
#       PTX asm directly. The inline asm bypasses LLVM's intrinsic lowering.
#
#   - Remove declarations for both replaced intrinsics.
#   - Remove the @llvm.nvvm.barrier.cta.sync.aligned.all declaration since
#     LLVM 21.1.0 no longer emits a @llvm.nvvm.barrier0 declaration.
sed \
    -e 's/call void @llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all(i32 0)/call void @llvm.nvvm.barrier0()/g' \
    -e '/^declare void @llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all/d' \
    -e 's/\(tail \)\{0,1\}call noundef i32 @llvm\.nvvm\.activemask()/call i32 asm sideeffect "activemask.b32 $0;", "=r"()/g' \
    -e '/^declare i32 @llvm\.nvvm\.activemask/d' \
    "$OUTPUT.tmp" > "$OUTPUT"

# LLVM 21.1.0 no longer emits a declaration for @llvm.nvvm.barrier0 since it
# uses the newer intrinsic instead — add it back explicitly.
if ! grep -q 'declare void @llvm.nvvm.barrier0()' "$OUTPUT"; then
    echo 'declare void @llvm.nvvm.barrier0()' >> "$OUTPUT"
fi

rm -f "$OUTPUT.tmp"
