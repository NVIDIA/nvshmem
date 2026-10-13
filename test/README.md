# NVSHMEM Tests

The NVSHMEM test suite covers host and device APIs, initialization, transports, and
interoperability with GPU libraries.

## Test Layout

- `host/`: Host and on-stream API tests.
- `device/`: CUDA kernel API tests.
- `apps/`: Application and interoperability tests.
- `common/`: Shared infrastructure and command-line handling.

Most tests exercise installed NVSHMEM interfaces; bootstrap tests may use internal headers.

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

## Acknowledged Signal Slot Reuse

`device/pt-to-pt/putmem_signal_slot_reuse` checks full byte payloads and red zones
on every directed edge between PEs. A single consumer CTA waits for each signal,
checks all bytes, and returns an ACK before that destination slot can be reused.
Signals and ACKs use monotonically increasing 64-bit generations. Source buffers
are reused only after device-side completion.

The default is 257 bytes, four slots, and 32 epochs. Optional positional arguments
are `bytes slots epochs [source_offset [target_offset [delay_cycles]]]`. Offsets
range from 0 to 15 and select the exact source and destination residue modulo 16.
The delay alternates between PEs and epochs before submission and consumption;
it is measured in GPU clock cycles. For example, use the normal test launcher with:

```bash
device/pt-to-pt/putmem_signal_slot_reuse 257 4 32 3 11 8192
```

Append `--all-offsets` to test all 256 source/destination residue pairs in one
process, with zero positional offsets. Each pair reports its own result. The PEs
finish all device-side operations and synchronize between configurations before
initializing fresh signals, ACKs, payloads, and guards. For example:

```bash
device/pt-to-pt/putmem_signal_slot_reuse 257 4 32 0 0 8192 --all-offsets
```

Run with at least two PEs and use the existing bootstrap/transport configuration.
For cross-node coverage, verify the PE placement and selected transport in the logs.
Apply an external process or job timeout independently of GPU stream progress.
The test reports payload/guard failures and the first mismatch; it does not measure
per-QP WQE counter wraparound or transport performance.
