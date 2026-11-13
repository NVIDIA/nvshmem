#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in multi-rank MathDX/NVSHMEM TileLib 3D FFT integration runner.

Adapted from MathDX MR !207's ``mgmn_fft.py`` at
``4be03a81179f0d90c8f7d59422f66f5af18653e5``.  It validates the complete
linkage-provider path: MathDX's cuFFTDx TileLib supplies the three local FFT
stages and NVSHMEM's packaged TileLib performs the distributed Z transpose.

Run through the accompanying pytest entry point with at least two MPI ranks,
one CUDA device per local rank, the MathDX Tile wheel, a CuTile-enabled
NVSHMEM4Py wheel, and a matching TileIR/CuTile toolchain.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import site
from typing import Annotated

import cuda.tile as ct
import cupy as cp
import nvmath.distributed.fft
import nvmath.tile.fft as mfft
import nvshmem.core
from cuda.core import Device, Stream, system
from mpi4py import MPI
from nvmath.distributed.distribution import Slab
from nvshmem.core.device.tile import my_pe as nvshmem_my_pe
from nvshmem.core.device.tile import put_block as nvshmem_put_block


def _static_array(rank: int, index_dtype=ct.int32):
    """Return a CuTile Array annotation with every dimension static."""
    static_dims = tuple(range(rank))
    return Annotated[
        ct.Array,
        ct.ArrayAnnotation(
            index_dtype=index_dtype,
            static_shape_dims=static_dims,
            static_stride_dims=static_dims,
        ),
    ]


static_array_4d = _static_array(4, index_dtype=ct.int64)


def _assert_installed_public_api() -> None:
    """Reject an accidental checkout import when a wheel location is supplied."""
    configured_site_packages = os.environ.get("NVSHMEM_TEST_SITE_PACKAGES", "")
    if not configured_site_packages:
        return

    site_packages = Path(configured_site_packages).resolve()
    assert site_packages in (Path(path).resolve() for path in site.getsitepackages())
    assert Path(nvshmem.core.__file__).resolve().is_relative_to(site_packages)
    for api in (nvshmem_my_pe, nvshmem_put_block):
        assert Path(inspect.getfile(api)).resolve().is_relative_to(site_packages)
        # ``tile`` is a public re-export; the generated implementation remains
        # package-relative under ``cutile`` so its raw TileLib imports agree.
        assert api.__module__.startswith("nvshmem.core.device.tile.")


def _compute_fft_gflops(fft_size: int, time_ms: float) -> float:
    """Compute the conventional 3D C2C FFT throughput estimate."""
    elements = fft_size**3
    flops = 5 * elements * math.log2(elements)
    return (flops / 1e9) / (time_ms * 1e-3)


def _relative_l2_error(reference: cp.ndarray, result: cp.ndarray) -> float:
    """Compute the relative error without a full-size temporary array."""
    reference = reference.reshape(-1)
    result = result.reshape(-1)
    chunk_elements = (1 << 30) // reference.itemsize
    error_squared = 0.0
    magnitude_squared = 0.0
    for begin in range(0, reference.size, chunk_elements):
        end = min(begin + chunk_elements, reference.size)
        reference_chunk = reference[begin:end]
        difference = reference_chunk - result[begin:end]
        error_squared += float(cp.vdot(difference, difference))
        magnitude_squared += float(cp.vdot(reference_chunk, reference_chunk))
    return (math.sqrt(error_squared) if magnitude_squared == 0 else math.sqrt(error_squared / magnitude_squared))


FFT_CONFIGS = {
    # B200 row-major winners from MathDX's fft_best_configs.py.  The Z
    # FFTs/block must divide N / ranks, so one is safe for every rank count.
    "float": {
        256: {
            "x": (16, 128),
            "y": (8, 128),
            "z": (4, 32)
        },
        512: {
            "x": (32, 512),
            "y": (8, 128),
            "z": (2, 32)
        },
        1024: {
            "x": (16, 512),
            "y": (16, 512),
            "z": (1, 32)
        },
        2048: {
            "x": (16, 1024),
            "y": (4, 256),
            "z": (1, 64)
        },
    },
    "double": {
        256: {
            "x": (16, 128),
            "y": (8, 128),
            "z": (1, 32)
        },
        512: {
            "x": (8, 128),
            "y": (8, 256),
            "z": (1, 32)
        },
        1024: {
            "x": (16, 512),
            "y": (4, 128),
            "z": (1, 64)
        },
        2048: {
            "x": (4, 512),
            "y": (4, 512),
            "z": (1, 128)
        },
    },
}

REAL_DTYPES = {"float": cp.float32, "double": cp.float64}
COMPLEX_DTYPES = {"float": cp.complex64, "double": cp.complex128}

# Slabs of dist_x alive per rank. cuFFTMp transforms in place, so its
# reference keeps the input and reference copy in CuPy plus an operand and
# workspace in NVSHMEM's heap. The Tile run keeps the input, the Y-stage
# output, and the reference copy in CuPy plus the result in the heap.
REFERENCE_SLABS = 2 + 2
TILE_SLABS = 3 + 1
PEAK_SLABS = max(REFERENCE_SLABS, TILE_SLABS) + 1


def _get_fft_config(precision: str, fft_size: int) -> dict[str, tuple[int, int]]:
    """Return the per-stage FFTs-per-block and CTA-size configuration."""
    try:
        precision_configs = FFT_CONFIGS[precision]
    except KeyError as error:
        supported_precisions = ", ".join(FFT_CONFIGS)
        raise ValueError(f"--precision must be one of {supported_precisions}; got {precision}") from error
    try:
        stage_configs = precision_configs[fft_size]
    except KeyError as error:
        supported_sizes = ", ".join(str(size) for size in precision_configs)
        raise ValueError(f"--fft-size must be one of {supported_sizes} for {precision}; got {fft_size}") from error
    if any(cta_size % 32 for _, cta_size in stage_configs.values()):
        raise ValueError("every CTA size must be a multiple of 32")
    return stage_configs


def make_fft_kernel_y(fft_size: int, stage_config: tuple[int, int]):
    ffts_per_block, cta_size = stage_config

    @ct.kernel(num_worker_warps=cta_size // 32)
    def fft_kernel_y(
        input_data: static_array_4d,
        output_data: static_array_4d,
    ):
        z, x = ct.bid(0), ct.bid(1)
        data = ct.load(input_data, (x, 0, z, 0), shape=(1, fft_size, ffts_per_block, 2))
        result = mfft.fft(data, axis=1)
        ct.store(output_data, (x, 0, z, 0), result)

    return fft_kernel_y


def make_fft_kernel_z(fft_size: int, stage_config: tuple[int, int]):
    ffts_per_block, cta_size = stage_config

    @ct.kernel(num_worker_warps=cta_size // 32)
    def fft_kernel_z(
        input_data: static_array_4d,
        output_data: static_array_4d,
        num_pes: ct.Constant[int],
    ):
        y, x = ct.bid(0), ct.bid(1)
        data = ct.load(input_data, (x, y, 0, 0), shape=(1, ffts_per_block, fft_size, 2))
        result = mfft.fft(data, axis=2)
        communication_tile = ct.reshape(result, (ffts_per_block * fft_size * 2, ))

        dist_fft_size = fft_size // num_pes
        local_y = y * ffts_per_block
        dst_pe = local_y // dist_fft_size
        dst_y = local_y % dist_fft_size
        dst_x = nvshmem_my_pe() * dist_fft_size + x

        destination = output_data.slice(axis=0, start=dst_x, stop=dst_x + 1)
        destination = destination.slice(axis=1, start=dst_y, stop=dst_y + ffts_per_block)
        # ``communication_tile`` is a local Tile while ``destination`` is a
        # symmetric Array. This is the Array-or-Tile public API path. The
        # native NVSHMEM operation is void, so it is intentionally used as a
        # statement; the raw binding retains its DCE result internally.
        nvshmem_put_block(destination, communication_tile, ct.astype(dst_pe, ct.int32))

    return fft_kernel_z


def make_fft_kernel_x(fft_size: int, stage_config: tuple[int, int]):
    ffts_per_block, cta_size = stage_config

    @ct.kernel(num_worker_warps=cta_size // 32)
    def fft_kernel_x(
        input_data: static_array_4d,
        output_data: static_array_4d,
    ):
        z, y = ct.bid(0), ct.bid(1)
        data = ct.load(input_data, (0, y, z, 0), shape=(fft_size, 1, ffts_per_block, 2))
        result = mfft.fft(data, axis=0)
        ct.store(output_data, (0, y, z, 0), result)

    return fft_kernel_x


def _run_cufftmp_reference(
    data_dist_x: cp.ndarray,
    dist_x: tuple[int, int, int, int],
    dist_y: tuple[int, int, int, int],
    precision: str,
    stream: cp.cuda.Stream,
    warmup: int,
    cycles: int,
) -> tuple[cp.ndarray, float]:
    """Return the Slab.Y cuFFTMp result and average execution time."""
    nvshmem_stream = Stream.from_handle(stream.ptr)
    ref_input = nvmath.distributed.fft.allocate_operand(
        dist_x[:3],
        cp,
        input_dtype=COMPLEX_DTYPES[precision],
        distribution=Slab.X,
        memory_space="cuda",
    )
    try:
        ref_input[:] = data_dist_x[..., 0] + 1j * data_dist_x[..., 1]

        # cuFFTMp supplies a distributed reference with the same Slab.X ->
        # Slab.Y redistribution performed by the NVSHMEM Z stage below.
        ref_output = nvmath.distributed.fft.fft(
            ref_input,
            stream=stream,
            distribution=Slab.X,
            options={"redistribute": False},
        )
        ref_output_dist_y = cp.empty(dist_y, dtype=REAL_DTYPES[precision])
        ref_output_dist_y[..., 0] = ref_output.real
        ref_output_dist_y[..., 1] = ref_output.imag

        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        with nvmath.distributed.fft.FFT(
                ref_input,
                stream=stream,
                distribution=Slab.X,
                options={"redistribute": False},
        ) as plan:
            plan.plan()
            for _ in range(warmup):
                plan.execute()
            nvshmem.core.sync_all(nvshmem_stream)
            stream.synchronize()
            start.record(stream)
            for _ in range(cycles):
                plan.execute()
            stop.record(stream)
            stop.synchronize()
            cufftmp_ms = cp.cuda.get_elapsed_time(start, stop) / cycles
        return ref_output_dist_y, cufftmp_ms
    finally:
        nvmath.distributed.free_symmetric_memory(ref_input)


def _run_tile_fft(
    data_dist_x: cp.ndarray,
    dist_y: tuple[int, int, int, int],
    precision: str,
    fft_size: int,
    stage_configs: dict[str, tuple[int, int]],
    kernels: dict[str, object],
    stream: cp.cuda.Stream,
    num_pes: int,
    warmup: int,
    cycles: int,
) -> tuple[cp.ndarray, float]:
    """Run the MathDX Tile FFT and return its Slab.Y output and average time."""
    nvshmem_stream = Stream.from_handle(stream.ptr)
    data_dist_x_tmp = cp.zeros_like(data_dist_x)
    data_dist_y = nvmath.distributed.allocate_symmetric_memory(dist_y, cp, dtype=REAL_DTYPES[precision])
    data_dist_y.fill(0)

    def fft_launch() -> None:
        ct.launch(
            stream,
            (fft_size // stage_configs["y"][0], fft_size // num_pes),
            kernels["y"],
            (data_dist_x, data_dist_x_tmp),
        )
        ct.launch(
            stream,
            (fft_size // stage_configs["z"][0], fft_size // num_pes),
            kernels["z"],
            (data_dist_x_tmp, data_dist_y, num_pes),
        )
        nvshmem.core.barrier_all(nvshmem_stream)
        ct.launch(
            stream,
            (fft_size // stage_configs["x"][0], fft_size // num_pes),
            kernels["x"],
            (data_dist_y, data_dist_y),
        )
        nvshmem.core.sync_all(nvshmem_stream)

    # Absorb the CuTile JIT, NVSHMEM library registration, and first-touch
    # allocations before measuring the iterations below.
    for _ in range(warmup):
        fft_launch()
    stream.synchronize()

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record(stream)
    for _ in range(cycles):
        fft_launch()
    stop.record(stream)
    stop.synchronize()
    tile_ms = cp.cuda.get_elapsed_time(start, stop) / cycles
    return data_dist_y, tile_ms


def _report(**result: object) -> None:
    """Print the machine-readable record that an integration job can collect."""
    print("MGMN_FFT_RESULT", json.dumps(result))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--precision", choices=sorted(FFT_CONFIGS), default="float")
    parser.add_argument("--fft-size", type=int, default=256, help="side length of the cubic FFT")
    parser.add_argument("--warmup", type=int, default=5, help="untimed launches")
    parser.add_argument("--cycles", type=int, default=10, help="timed launches")
    args = parser.parse_args(argv)
    if args.warmup <= 0 or args.cycles <= 0:
        parser.error("--warmup and --cycles must be positive")
    return args


def main(argv: list[str] | None = None) -> str:
    """Run and verify the distributed 3D FFT on every MPI rank.

    Return "PASS" or "FAIL", or "SKIP" when some rank lacks the GPU memory.
    """
    _assert_installed_public_api()
    args = _parse_args(argv)
    precision, fft_size = args.precision, args.fft_size
    warmup, cycles = args.warmup, args.cycles
    stage_configs = _get_fft_config(precision, fft_size)
    comm = MPI.COMM_WORLD
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    my_pe = comm.Get_rank()
    num_pes = comm.Get_size()
    if num_pes < 2:
        raise RuntimeError("the MGMN FFT integration test requires at least two MPI ranks")
    if os.environ.get("NVSHMEM4PY_CUTILE_MGMN_REQUIRE_MULTINODE") == "1":
        hosts = set(comm.allgather(MPI.Get_processor_name()))
        if len(hosts) < 2:
            raise RuntimeError("NVSHMEM4PY_CUTILE_MGMN_REQUIRE_MULTINODE=1 requires at least two hosts")
    device_count = system.get_num_devices()
    if device_count == 0:
        raise RuntimeError("no CUDA devices are visible")
    if fft_size % num_pes:
        raise ValueError("--fft-size must be divisible by the MPI rank count")
    if any(fft_size % stage_configs[stage][0] for stage in ("x", "y")):
        raise ValueError("the X and Y FFTS_PER_BLOCK values must divide --fft-size")
    if (fft_size // num_pes) % stage_configs["z"][0]:
        raise ValueError("the Z FFTS_PER_BLOCK value must divide --fft-size / number of ranks")

    local_rank = local_comm.Get_rank()
    if local_rank >= device_count:
        raise RuntimeError(f"rank {my_pe}: local rank {local_rank} has no matching CUDA device "
                           f"(only {device_count} visible)")
    device = Device(local_rank)
    device.set_current()
    dist_x = (fft_size // num_pes, fft_size, fft_size, 2)
    dist_y = (fft_size, fft_size // num_pes, fft_size, 2)

    data_dist_y = None
    stream = None
    kernels = None
    nvshmem_initialized = False
    nvmath_initialized = False
    try:
        # Every rank skips together rather than fail with an out-of-memory error.
        slab_bytes = math.prod(dist_x) * cp.dtype(REAL_DTYPES[precision]).itemsize
        needed_bytes = PEAK_SLABS * slab_bytes + (1 << 30)
        free_bytes = cp.cuda.runtime.memGetInfo()[0]
        if not comm.allreduce(needed_bytes <= free_bytes, op=MPI.LAND):
            print(f"{my_pe}: SKIP: needs about {needed_bytes >> 30} GiB per rank, "
                  f"{free_bytes >> 30} GiB free")
            _report(
                gpu=device.name,
                precision=precision,
                fft_size=fft_size,
                ranks=num_pes,
                rank=my_pe,
                verdict="SKIP",
            )
            return "SKIP"

        # Kernel objects own their CUDA libraries and unload them when
        # collected; retain them until NVSHMEM unregisters their libraries.
        kernels = {
            "x": make_fft_kernel_x(fft_size, stage_configs["x"]),
            "y": make_fft_kernel_y(fft_size, stage_configs["y"]),
            "z": make_fft_kernel_z(fft_size, stage_configs["z"]),
        }

        if my_pe == 0:
            print(f"3D MGMN FFT: {fft_size}^3 complex {precision}; {num_pes} MPI ranks")
            print("Configs: " + ", ".join(f"{stage}: FFTS_PER_BLOCK={ffts_per_block}, CTA_SIZE={cta_size}"
                                          for stage, (ffts_per_block, cta_size) in stage_configs.items()))

        nvshmem.core.init(device=device, mpi_comm=comm, initializer_method="mpi")
        nvshmem_initialized = True
        assert num_pes == nvshmem.core.n_pes()
        assert my_pe == nvshmem.core.my_pe()

        data_dist_x = cp.random.randn(*dist_x, dtype=REAL_DTYPES[precision])
        stream = cp.cuda.get_current_stream()

        # The first argument is a CUDA device ID, not a global rank.
        nvmath.distributed.initialize(local_rank, comm, backends=["nvshmem"])
        nvmath_initialized = True
        ref_output_dist_y, cufftmp_ms = _run_cufftmp_reference(data_dist_x, dist_x, dist_y, precision, stream, warmup,
                                                               cycles)
        print(f"{my_pe}: cuFFTMp {cufftmp_ms:.3f} ms, "
              f"{_compute_fft_gflops(fft_size, cufftmp_ms):.2f} GFlop/s")

        data_dist_y, tile_ms = _run_tile_fft(
            data_dist_x,
            dist_y,
            precision,
            fft_size,
            stage_configs,
            kernels,
            stream,
            num_pes,
            warmup,
            cycles,
        )
        print(f"{my_pe}: MathDX Tile provider {tile_ms:.3f} ms, "
              f"{_compute_fft_gflops(fft_size, tile_ms):.2f} GFlop/s")
        relative_error = _relative_l2_error(ref_output_dist_y, data_dist_y)
        threshold = 1e-4
        verdict = "PASS" if relative_error < threshold else "FAIL"
        print(f"{my_pe}: {verdict}: relative error {relative_error:.3e}")
        _report(
            gpu=device.name,
            precision=precision,
            fft_size=fft_size,
            ranks=num_pes,
            rank=my_pe,
            cufftmp_ms=round(float(cufftmp_ms), 3),
            cufftmp_gflops=round(_compute_fft_gflops(fft_size, cufftmp_ms), 2),
            tile_ms=round(float(tile_ms), 3),
            tile_gflops=round(_compute_fft_gflops(fft_size, tile_ms), 2),
            relative_error=float(f"{relative_error:.3e}"),
            verdict=verdict,
        )
        return verdict
    finally:
        try:
            if nvshmem_initialized:
                # No linked CUDA library or symmetric allocation may outlive
                # work that can still access it.
                if stream is not None:
                    stream.synchronize()
                else:
                    device.sync()
                if data_dist_y is not None:
                    nvmath.distributed.free_symmetric_memory(data_dist_y)
            if nvmath_initialized:
                nvmath.distributed.finalize()
        finally:
            # CuTile owns library registration, so unload every dispatcher
            # before shutting down NVSHMEM even when the workload failed.
            if kernels is not None:
                kernels.clear()
                kernels = None
                gc.collect()
            try:
                if nvshmem_initialized:
                    nvshmem.core.finalize()
            finally:
                local_comm.Free()


if __name__ == "__main__":
    raise SystemExit(1 if main() == "FAIL" else 0)
