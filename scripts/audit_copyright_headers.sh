#!/bin/bash
# Audit copyright headers for consistency issues flagged in MR review.
# Run from repo root: bash scripts/audit_copyright_headers.sh
set -euo pipefail

ERRORS=0
WARNINGS=0

error() { echo "ERROR: $*"; ((ERRORS++)) || true; }
warn()  { echo "WARNING: $*"; ((WARNINGS++)) || true; }
info()  { echo "INFO: $*"; }

echo "=== Copyright Header Audit ==="
echo

# 1. Check LICENSE file is Apache 2.0
echo "--- Checking LICENSE file ---"
if head -4 LICENSE | grep -q "Apache License" && head -4 LICENSE | grep -q "Version 2.0"; then
    info "LICENSE file is Apache 2.0"
else
    error "LICENSE file does not appear to be Apache 2.0"
fi
echo

# 2. Check for inconsistent copyright formats
echo "--- Checking copyright format consistency ---"
echo "Target format: Copyright (c) <YEAR>, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
echo

# Files missing "& AFFILIATES" (target format requires NVIDIA CORPORATION & AFFILIATES)
AFFILIATES=$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.cmake' '*.yml' '*.yaml' '*.txt.in' '*.in' '*.j2' '*.pyx' '*.pxd' 2>/dev/null | \
    grep -v "^include_gdrcopy/\|^include_nccl/\|^scripts/" | \
    while read f; do
        if grep -q "NVIDIA CORPORATION" "$f" 2>/dev/null && ! grep -q "AFFILIATES" "$f" 2>/dev/null; then
            echo "$f"
        fi
    done)
if [ -n "$AFFILIATES" ]; then
    COUNT=$(echo "$AFFILIATES" | wc -l)
    warn "$COUNT files use 'NVIDIA CORPORATION' instead of 'NVIDIA CORPORATION & AFFILIATES'"
    echo "$AFFILIATES" | head -10
    [ "$COUNT" -gt 10 ] && echo "  ... and $((COUNT - 10)) more"
fi
echo

# Files with wrong-case "ALL RIGHTS RESERVED" or missing "All rights reserved."
ALL_RIGHTS=$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.cmake' '*.yml' '*.yaml' '*.txt.in' '*.in' '*.j2' '*.pyx' '*.pxd' 2>/dev/null | \
    grep -v "^include_gdrcopy/\|^include_nccl/\|^scripts/" | \
    while read f; do
        if grep -q "Copyright.*NVIDIA CORPORATION" "$f" 2>/dev/null; then
            if grep -q "ALL RIGHTS RESERVED" "$f" 2>/dev/null || ! grep -q "All rights reserved\." "$f" 2>/dev/null; then
                echo "$f"
            fi
        fi
    done)
if [ -n "$ALL_RIGHTS" ]; then
    COUNT=$(echo "$ALL_RIGHTS" | wc -l)
    warn "$COUNT files uses 'ALL RIGHTS RESERVED' or nothing instead of 'All rights reserved'"
    echo "$ALL_RIGHTS" | head -10
    [ "$COUNT" -gt 10 ] && echo "  ... and $((COUNT - 10)) more"
fi
echo

# 3. Check for // comment style in .h files (should use /* */ block comments)
echo "--- Checking comment style in headers ---"
SLASH_HEADERS=$(git ls-files '*.h' '*.hpp' '*.cuh' '*.c' '*.cu' '*.cpp' 2>/dev/null | \
    xargs grep -l "^// Copyright" 2>/dev/null || true)
if [ -n "$SLASH_HEADERS" ]; then
    COUNT=$(echo "$SLASH_HEADERS" | wc -l)
    warn "$COUNT header files use // style for copyright (should use /* */ block comments)"
    echo "$SLASH_HEADERS"
fi
echo

# 4. Check for SPDX license identifier
echo "--- Checking SPDX identifiers ---"
MISSING_SPDX=$(git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.cmake' '*.yml' '*.yaml' '*.txt.in' '*.in' '*.j2' '*.pyx' '*.pxd' 2>/dev/null | \
    grep -v "^include_gdrcopy/\|^include_nccl/\|^scripts/" | \
    while read f; do
        if grep -q "Copyright.*NVIDIA" "$f" 2>/dev/null && ! grep -q "SPDX-License-Identifier" "$f" 2>/dev/null; then
            echo "$f"
        fi
    done)
if [ -n "$MISSING_SPDX" ]; then
    COUNT=$(echo "$MISSING_SPDX" | wc -l)
    warn "$COUNT files have NVIDIA copyright but missing SPDX-License-Identifier"
    echo "$MISSING_SPDX" | head -10
    [ "$COUNT" -gt 10 ] && echo "  ... and $((COUNT - 10)) more"
fi
echo

# 5. Check for year modifications vs devel (years should not be changed in this MR)
echo "--- Checking for year modifications vs devel ---"
YEAR_CHANGES=$(git diff origin/devel...HEAD -- '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' | \
    grep "^[-+].*Copyright (c)" | \
    grep -v "^---\|^+++" | \
    awk '
    /^-.*Copyright \(c\)/ {
        match($0, /Copyright \(c\) ([0-9][-0-9]+)/, m);
        if (m[1]) old[NR] = m[1]
    }
    /^\+.*Copyright \(c\)/ {
        match($0, /Copyright \(c\) ([0-9][-0-9]+)/, m);
        if (m[1]) new[NR] = m[1]
    }
    END {
        for (k in old) {
            if ((k+1) in new && old[k] != new[k+1]) {
                print "Year changed: " old[k] " -> " new[k+1]
            }
        }
    }
    ' | sort | uniq -c | sort -rn)
if [ -n "$YEAR_CHANGES" ]; then
    warn "Year modifications detected in copyright headers:"
    echo "$YEAR_CHANGES"
fi
echo

# 6. Summary
echo "=== Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
exit $ERRORS
