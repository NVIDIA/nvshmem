# NVSHMEM Tests

The NVSHMEM test suite covers host and device APIs, initialization, transports, and
interoperability with GPU libraries.

## Test Layout

- `host/`: Host and on-stream API tests.
- `device/`: CUDA kernel API tests.
- `apps/`: Application and interoperability tests.
- `common/`: Shared infrastructure and command-line handling.
- `unit/`: Unit tests for internal components.

Most tests exercise installed NVSHMEM interfaces; bootstrap and unit tests may use internal
headers.

## Building

The test suite is built by default with NVSHMEM (`NVSHMEM_BUILD_TESTS=ON`). To build it separately
against an existing NVSHMEM installation, configure it from the repository root with
`NVSHMEM_PREFIX` set to that installation:

```bash
export NVSHMEM_TEST_INSTALL="$PWD/test/test_install"

cmake -S test -B test/build -DNVSHMEM_PREFIX="$NVSHMEM_PREFIX"
cmake --build test/build --parallel
cmake --install test/build
```

## Running Tests

Installed executables retain the source hierarchy under `NVSHMEM_TEST_INSTALL`. Use the launcher
pattern from the main README's [verification example](../README.md#verify-the-installation),
replacing the perftest command and options with the selected test binary.

## Adding a Test

Add new tests to the matching interface directory and its `CMakeLists.txt`. Reuse `common/` for
initialization, options, and result reporting. Tests must run non-interactively and return a
nonzero status on failure.
