#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated-code comparison for the transport-policy routes.

Compiles one small kernel per route, operation, and local memory space at the
selected scope, then reports the SASS instruction, branch, and call counts of
each kernel. Specialized routes are expected to drop the portable dispatch and
MemorySpace classification that Auto / Auto and the existing portable API keep.
The capability rows show the cost of nvshmemx_rma_capability, which runs the
put preflight without issuing work.

Only the kernel body is counted; out-of-line NVSHMEM device-library calls
(for example the network transfer functions) appear as CALL instructions.

Example:
  perftest/transport_specific_sass_report.py \\
      --include build/src/include --arch sm_90 --scope block
"""

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROUTES = [
    ("pointer_sm", "Pointer", "SM", "global"),
    ("pointer_sm", "Pointer", "SM", "generic"),
    ("pointer_tma", "Pointer", "TMA", "global"),
    ("pointer_tma", "Pointer", "TMA", "shared"),
    ("pointer_tma", "Pointer", "TMA", "generic"),
    ("cft_tma", "CFT", "TMA", "global"),
    ("cft_tma", "CFT", "TMA", "shared"),
    ("auto_network", "Auto", "Network", "global"),
    ("auto_auto", "Auto", "Auto", "global"),
    ("auto_auto", "Auto", "Auto", "generic"),
]

OPERATIONS = ["put", "put_nbi", "get", "put_signal", "quiet", "capability"]

LOCAL_OPERATIONS = {"put", "put_nbi", "get", "put_signal", "capability"}

BLOCK_ONLY_ROUTES = {"cft_tma"}

PORTABLE_SUFFIX = {"thread": "", "warp": "_warp", "block": "_block"}
PORTABLE_PREFIX = {"thread": "nvshmem_", "warp": "nvshmemx_", "block": "nvshmemx_"}

PROLOGUE = """
#define NVSHMEMX_ENABLE_TRANSPORT_POLICY
#include "nvshmemx.h"

namespace ms = nvshmemx::memory_space;
constexpr auto kScope = nvshmemx::threadgroup::{scope};

/* Hide the kernel-parameter origin so generic classification is not folded. */
__device__ inline uint8_t *opaque(uint8_t *ptr) {{
    uint8_t *out;
    asm("mov.b64 %0, %1;" : "=l"(out) : "l"(ptr));
    return out;
}}
"""

PREVIEW_KERNELS = {
    "put": "*status = nvshmemx_put<uint8_t, {a}, {p}, kScope, ms::source::{src}>("
    "remote, opaque(local), bytes, pe);",
    "put_nbi": "*status = nvshmemx_put_nbi<uint8_t, {a}, {p}, kScope, ms::source::{src}>("
    "remote, opaque(local), bytes, pe);",
    "get": "*status = nvshmemx_get<uint8_t, {a}, {p}, kScope, ms::source::global, "
    "ms::dest::{src}>(opaque(local), remote, bytes, pe);",
    "put_signal": "*status = nvshmemx_put_signal<uint8_t, {a}, {p}, kScope, ms::source::{src}>("
    "remote, opaque(local), bytes, signal, 1, NVSHMEM_SIGNAL_SET, pe);",
    "quiet": "*status = nvshmemx_quiet<{a}, {p}, kScope>();",
    "capability": "*status = nvshmemx_rma_capability<{a}, {p}, kScope, ms::source::{src}>("
    "remote, opaque(local), bytes, pe);",
}

PORTABLE_KERNELS = {
    "put": "{prefix}putmem{suffix}(remote, opaque(local), bytes, pe);",
    "put_nbi": "{prefix}putmem_nbi{suffix}(remote, opaque(local), bytes, pe);",
    "get": "{prefix}getmem{suffix}(opaque(local), remote, bytes, pe);",
    "put_signal": "{prefix}putmem_signal{suffix}(remote, opaque(local), bytes, signal, 1, "
    "NVSHMEM_SIGNAL_SET, pe);",
    "quiet": "{prefix}quiet{suffix}();",
}

SIGNATURE = ("uint8_t *remote, uint8_t *local, size_t bytes, uint64_t *signal, int pe, "
             "int *status")

INSTRUCTION = re.compile(r"^\s+/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;")
FUNCTION = re.compile(r"^\s*Function\s*:\s*(\S+)")
PREDICATE = re.compile(r"^@!?U?P[T0-9]+\s+")


def legal(label, op, space, scope):
    if scope == "block":
        return True
    if label in BLOCK_ONLY_ROUTES:
        return False
    if label == "pointer_tma":
        return op == "quiet" or (op != "get" and space != "global")
    return True


def kernel_source(scope):
    lines = [PROLOGUE.format(scope=scope)]
    rows = []
    for label, addressing, policy, space in ROUTES:
        for op in OPERATIONS:
            if not legal(label, op, space, scope):
                continue
            local = op in LOCAL_OPERATIONS
            name = f"{label}_{op}_{space}" if local else f"{label}_{op}"
            if any(row[0] == name for row in rows):
                continue
            body = PREVIEW_KERNELS[op].format(a=f"nvshmemx::addressing::{addressing}",
                                              p=f"nvshmemx::transport_policy::{policy}",
                                              src=space)
            lines.append(f'extern "C" __global__ void {name}({SIGNATURE}) {{ {body} }}')
            rows.append((name, label, op, space if local else "-"))
    for op in PORTABLE_KERNELS:
        name = f"portable_api_{op}"
        body = PORTABLE_KERNELS[op].format(prefix=PORTABLE_PREFIX[scope],
                                           suffix=PORTABLE_SUFFIX[scope])
        lines.append(f'extern "C" __global__ void {name}({SIGNATURE}) {{ {body} (void)status; }}')
        rows.append((name, "portable_api", op, "-"))
    return "\n".join(lines) + "\n", rows


def count_sass(sass):
    counts = collections.defaultdict(lambda: collections.Counter())
    function = None
    for line in sass.splitlines():
        match = FUNCTION.match(line)
        if match:
            function = match.group(1)
            continue
        match = INSTRUCTION.match(line)
        if function is None or not match:
            continue
        opcode = PREDICATE.sub("", match.group(1)).split()[0].split(".")[0]
        if opcode == "NOP":
            continue
        counts[function]["instructions"] += 1
        if opcode in ("BRA", "BRX", "JMP", "JMX"):
            counts[function]["branches"] += 1
        elif opcode in ("CALL", "JMPL"):
            counts[function]["calls"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", required=True, action="append",
                        help="NVSHMEM package include directory (repeatable)")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    parser.add_argument("--cuobjdump", default=None,
                        help="defaults to cuobjdump next to --nvcc")
    parser.add_argument("--arch", default="sm_90", help="SASS architecture, e.g. sm_90 or sm_100")
    parser.add_argument("--scope", default="block", choices=["thread", "warp", "block"])
    parser.add_argument("--keep", default=None, help="directory in which to keep the sources")
    args = parser.parse_args()

    cuobjdump = args.cuobjdump or os.path.join(os.path.dirname(args.nvcc), "cuobjdump")
    if not os.path.exists(cuobjdump):
        cuobjdump = shutil.which("cuobjdump") or "cuobjdump"

    source, rows = kernel_source(args.scope)
    workdir = args.keep or tempfile.mkdtemp(prefix="ts_sass_")
    os.makedirs(workdir, exist_ok=True)
    try:
        return report(args, cuobjdump, source, rows, workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


def report(args, cuobjdump, source, rows, workdir):
    cu = os.path.join(workdir, "transport_specific_sass.cu")
    cubin = os.path.join(workdir, "transport_specific_sass.cubin")
    with open(cu, "w") as handle:
        handle.write(source)

    command = [args.nvcc, "-std=c++17", "-rdc=true", f"-arch={args.arch}", "-cubin", cu, "-o",
               cubin]
    for include in args.include:
        command.append(f"-I{include}")
    compiled = subprocess.run(command, capture_output=True, text=True)
    if compiled.returncode != 0:
        sys.stderr.write(" ".join(command) + "\n" + compiled.stdout + compiled.stderr)
        return 1
    dumped = subprocess.run([cuobjdump, "-sass", cubin], capture_output=True, text=True)
    if dumped.returncode != 0:
        sys.stderr.write(dumped.stderr)
        return 1
    counts = count_sass(dumped.stdout)

    print(f"# transport-policy route SASS ({args.arch}, {args.scope} scope)")
    print(f"{'route':<14} {'operation':<11} {'local':<8} {'instructions':>12} {'branches':>9} "
          f"{'calls':>6}")
    for name, label, op, space in rows:
        c = counts.get(name)
        if c is None:
            print(f"{label:<14} {op:<11} {space:<8} {'missing':>12}")
            continue
        print(f"{label:<14} {op:<11} {space:<8} {c['instructions']:>12} {c['branches']:>9} "
              f"{c['calls']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
