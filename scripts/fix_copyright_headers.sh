#!/bin/bash
# Fix copyright header consistency issues.
# Run from repo root: bash scripts/fix_copyright_headers.sh
# Review changes with: git diff
set -euo pipefail

echo "=== Fixing Copyright Headers ==="

# 1. Ensure "All rights reserved" is in NVIDIA copyright lines
# (don't touch third-party headers)
echo "--- Replacing 'ALL RIGHTS RESERVED' with 'All rights reserved' from NVIDIA copyright lines ---"
git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.pyx' | while read f; do
    # Skip third-party headers
    case "$f" in include_gdrcopy/*|include_nccl/*) continue;; esac
    if grep -q "NVIDIA CORPORATION" "$f" 2>/dev/null && ! grep -q "All rights reserved\." "$f" 2>/dev/null || \
           grep -q "ALL RIGHTS RESERVED" "$f" 2>/dev/null; then
        sed -i 's/\(NVIDIA CORPORATION[^.]*\)\.[[:space:]]*$/\1. All rights reserved./g' "$f"
        sed -i 's/\(NVIDIA CORPORATION[^.]*\)\.[[:space:]]*ALL RIGHTS RESERVED\./\1. All rights reserved./g' "$f"
    fi
done
echo "Done."

# 2. Replace "NVIDIA CORPORATION" with "NVIDIA CORPORATION & AFFILIATES"
# (skip third-party)
echo "--- Normalizing 'NVIDIA CORPORATION' -> 'NVIDIA CORPORATION & AFFILIATES' ---"
git ls-files '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.pyx' | while read f; do
    case "$f" in include_gdrcopy/*|include_nccl/*) continue;; esac
    if grep -q "NVIDIA CORPORATION" "$f" 2>/dev/null && ! grep -q "NVIDIA CORPORATION & AFFILIATES" "$f" 2>/dev/null; then
        sed -i 's/NVIDIA CORPORATION/NVIDIA CORPORATION \& AFFILIATES/g' "$f"
    fi
done
echo "Done."

# 3. Fix // comment style in .h files -> /* */ block comments
echo "--- Fixing // comment style in header files ---"
for f in $(git ls-files '*.h' '*.hpp' '*.cuh' | xargs grep -l "^// Copyright.*NVIDIA" 2>/dev/null); do
    case "$f" in include_gdrcopy/*|include_nccl/*) continue;; esac
    # Convert leading // block to /* */ block
    # Match consecutive // lines at the start of the file
    python3 -c "
import re, sys
with open('$f', 'r') as fh:
    content = fh.read()
# Match a block of // lines at the start (copyright + SPDX)
m = re.match(r'((?://[^\n]*\n)+)', content)
if m:
    block = m.group(1)
    # Convert // lines to * lines
    lines = block.strip().split('\n')
    new_lines = ['/*'] + [' * ' + line.lstrip('/ ') if line.strip('/ ') else ' *' for line in lines] + [' */']
    new_block = '\n'.join(new_lines) + '\n'
    content = content[len(block):]
    content = new_block + content
    with open('$f', 'w') as fh:
        fh.write(content)
    print(f'Fixed: $f')
"
done
echo "Done."

# 4. Revert year changes where years were modified (restore from devel)
echo "--- Reverting year modifications in copyright lines ---"
# Get list of files where copyright year was changed
git diff origin/devel...HEAD -- '*.c' '*.cpp' '*.h' '*.hpp' '*.cu' '*.cuh' '*.py' '*.sh' '*.pyx' | \
    grep "^diff --git" | sed 's|diff --git a/\(.*\) b/.*|\1|' | sort -u | while read f; do
    [ -f "$f" ] || continue
    # For each file, check if the copyright year was changed
    OLD_YEAR=$(git show origin/devel:"$f" 2>/dev/null | grep -m1 "Copyright (c).*NVIDIA" | sed 's/.*Copyright (c) \([0-9][0-9-]*\),.*/\1/' || true)
    NEW_YEAR=$(grep -m1 "Copyright (c).*NVIDIA" "$f" 2>/dev/null | sed 's/.*Copyright (c) \([0-9][0-9-]*\),.*/\1/' || true)
    if [ -n "$OLD_YEAR" ] && [ -n "$NEW_YEAR" ] && [ "$OLD_YEAR" != "$NEW_YEAR" ]; then
        # Restore the original year in the copyright line
        sed -i "s/Copyright (c) $NEW_YEAR,/Copyright (c) $OLD_YEAR,/" "$f"
    fi
done
echo "Done."

echo
echo "=== Fixes applied. Review with: git diff ==="
echo "Then re-run: bash scripts/audit_copyright_headers.sh"
