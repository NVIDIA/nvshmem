# NVSHMEM Performance Tests

The NVSHMEM performance tests measure latency, bandwidth, and message rate for host- and
device-initiated operations.

## Building

The performance tests are built by default with NVSHMEM (`NVSHMEM_BUILD_TESTS=ON`). To build them
separately against an existing NVSHMEM installation, configure them from the repository root with
`NVSHMEM_PREFIX` set to that installation:

```bash
export NVSHMEM_PERFTEST_INSTALL="$PWD/perftest/perftest_install"

cmake -S perftest -B perftest/build -DNVSHMEM_PREFIX="$NVSHMEM_PREFIX"
cmake --build perftest/build --parallel
cmake --install perftest/build
```

## Running Performance Tests

Installed executables retain the source hierarchy under `NVSHMEM_PERFTEST_INSTALL`. Reuse the main
README's [verification command](../README.md#verify-the-installation), replacing the binary with
`$NVSHMEM_PERFTEST_INSTALL/device/pt-to-pt/shmem_put_bw`.

## Command-Line Options

Run an executable with `--help` for its options. The common parser and help text are defined in
[`common/utils.cu`](common/utils.cu); each test uses only the applicable options.
