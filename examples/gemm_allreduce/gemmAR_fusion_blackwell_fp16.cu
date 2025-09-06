/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A fused implementation of fp16 dense GEMM with AllReduce

    The GEMM code is modified from cutlass/examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu
    GEMM is fused with AllReduce using the NVSHMEM Tile-granular AllReduce API

    Compilation:
        Please set NVSHMEM_BUILD_CUTLASS_EXAMPLES=ON in cmake and
        set the environment variable CUTLASS_HOME=<path to CUTLASS directory>

    Usage:
      $ <path to hydra>/nvshmrun -n 2 -ppn 2 ./gemmAR_fusion_blackwell_fp16 --m=512 --n=512 --k=512
*/

#include <iostream>

#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include "helper.h"
#include "nvshmemAlloc.hpp"
#include "sm100_gemm_tma_warpspecialized_allreduce.hpp"
#include "allreduce_nvls_warpspecialized.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA = half_t;                    // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A matrix
                                                  // in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = half_t;                       // Element type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B matrix
                                                  // in units of elements (up to 16 bytes)

// C/D matrix configuration
using ElementC = float;                        // Element type for C and D matrix operands
using LayoutC = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C matrix
                                                  // in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator = float;  // Element type for internal accumulation
using ArchTag =
    cutlass::arch::Sm100;  // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0
using MmaTileShape_MNK = Shape<_256, _128, _64>;
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_2, _2, _1>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC, ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
    ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using CollectiveAllReduce = cutlass::comm::collective::CollectiveAllReduceMulticastWarpSpecialized<
    ElementC, MmaTileShape_MNK, typename CollectiveEpilogue::StrideD>;
// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::Sm100GemmARUniversal<
    Shape<int, int, int, int>,  // Indicates ProblemShape
    CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler,
    CollectiveAllReduce>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference =
    cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                                     ElementAccumulator, ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed = 1;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D_red;
nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D_red;

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

__global__ void ref_reduce_kernel(ElementC *out, ElementC **ref_D_ptr, ElementC *arrD_red,
                                  ElementC *arrD, size_t npes, size_t nelem) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    volatile ElementC *output = out;
    volatile ElementC *val_ptr;
    for (int i = tid; i < nelem; i += gridDim.x * blockDim.x) {
        val_ptr = ref_D_ptr[0] + i;
        output[i] = *(val_ptr);
        for (int n = 1; n < npes; ++n) {
            val_ptr = ref_D_ptr[n] + i;
            output[i] += *(val_ptr);
        }
    }
}

__global__ void compare_kernel(ElementC *expected_out, ElementC *actual_out, ElementC **ref_D_ptr,
                               ElementC *arrD_red, ElementC *arrD, int mype, size_t npes,
                               size_t nelem) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < nelem; i += gridDim.x * blockDim.x) {
        if (actual_out[i] != expected_out[i]) {
            printf("%d elem: %d, mismatch expected_out: %f, actual: %f computed: %f : %f \n", mype,
                   i, expected_out[i], actual_out[i], *(ref_D_ptr[0] + i), *(ref_D_ptr[1] + i));
        }
    }
}

//////  nvshmem variables //////
nvshmem_team_t *teams_dev, *teams;
int num_teams;
int mype, npes;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {
    bool help;

    float alpha, beta;
    int iterations;
    int m, n, k;

    Options() : help(false), m(8192), n(8192), k(8192), alpha(1.f), beta(0.f), iterations(10) {}

    // Parses the command line
    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
            return;
        }

        cmd.get_cmd_line_argument("m", m);
        cmd.get_cmd_line_argument("n", n);
        cmd.get_cmd_line_argument("k", k);
        cmd.get_cmd_line_argument("alpha", alpha, 1.f);
        cmd.get_cmd_line_argument("beta", beta, 0.f);
        cmd.get_cmd_line_argument("iterations", iterations);
    }

    /// Prints the usage statement.
    std::ostream &print_usage(std::ostream &out) const {
        out << "70_blackwell_fp16_gemm\n\n"
            << "  Blackwell FP16 GEMM using a Warp Specialized kernel.\n\n"
            << "Options:\n\n"
            << "  --help                      If specified, displays this usage statement\n\n"
            << "  --m=<int>                   Sets the M extent of the GEMM\n"
            << "  --n=<int>                   Sets the N extent of the GEMM\n"
            << "  --k=<int>                   Sets the K extent of the GEMM\n"
            << "  --alpha=<f32>               Epilogue scalar alpha\n"
            << "  --beta=<f32>                Epilogue scalar beta\n\n"
            << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

        out << "\n\nExamples:\n\n"
            << "$ "
            << "70_blackwell_fp16_gemm"
            << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

        return out;
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const {
        // Two flops per multiply-add
        uint64_t flop = uint64_t(2) * m * n * k;
        double gflop = double(flop) / double(1.0e9);
        return gflop / runtime_s;
    }
};

/// Result structure
struct Result {
    double avg_runtime_ms;
    double gflops;
    cutlass::Status status;
    cudaError_t error;
    bool passed;

    Result(double avg_runtime_ms = 0, double gflops = 0,
           cutlass::Status status = cutlass::Status::kSuccess, cudaError_t error = cudaSuccess)
        : avg_runtime_ms(avg_runtime_ms),
          gflops(gflops),
          status(status),
          error(error),
          passed(false) {}
};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(cutlass::DeviceAllocation<Element> &block, uint64_t seed = 2023) {
    Element scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
        scope_max = Element(2);
        scope_min = Element(0);
    } else if (bits_input <= 8) {
        scope_max = Element(2);
        scope_min = Element(-2);
    } else {
        scope_max = Element(8);
        scope_min = Element(-8);
    }

    cutlass::reference::device::BlockFillRandomUniform(block.get(), block.size(), seed, scope_max,
                                                       scope_min, 0);

    return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

    block_A.reset(options.m * options.k);
    block_B.reset(options.k * options.n);
    block_C.reset(options.m * options.n);
    block_D.reset(options.m * options.n);
    block_D_red.reset(options.m * options.n);
    block_ref_D.reset(options.m * options.n);
    block_ref_D_red.reset(options.m * options.n);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options) {
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {block_A.get(), stride_A, block_B.get(), stride_B},
        {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    };

    return arguments;
}

bool verify(const Options &options, int mype, int npes) {
    cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
    cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
    cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
    cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));
    cutlass::TensorRef ref_D_red(block_ref_D_red.get(),
                                 Gemm::LayoutD::packed({options.m, options.n}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference({options.m, options.n, options.k}, ElementAccumulator(options.alpha), ref_A,
                   ref_B, ElementAccumulator(options.beta), ref_C, ref_D);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // get reference from the other PE
    ElementC **ref_D_ptr_dev;
    ElementC **ref_D_ptr = (ElementC **)malloc(npes * sizeof(ElementC *));
    for (int i = 0; i < npes; ++i) {
        ref_D_ptr[i] = (ElementC *)nvshmem_ptr(block_ref_D.get(), i);
    }
    CUDA_CHECK(cudaMalloc(&ref_D_ptr_dev, npes * sizeof(ElementC *)));
    CUDA_CHECK(
        cudaMemcpy(ref_D_ptr_dev, ref_D_ptr, npes * sizeof(ElementC *), cudaMemcpyDeviceToHost));

    // using block_ref_D_red for storing reduced output
    ref_reduce_kernel<<<1, 256>>>(block_ref_D_red.get(), ref_D_ptr_dev, block_D_red.get(),
                                  block_D.get(), npes, block_D.size());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Comparing GEMM output first
    bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(),
                                                                block_D.size());
    if (passed) {
        fprintf(stderr, "PE: %d GEMM passed !!!\n", mype);
    } else {
        fprintf(stderr, "PE: %d GEMM failed !!!\n", mype);
    }

    compare_kernel<<<1, 256>>>(block_ref_D_red.get(), block_D_red.get(), ref_D_ptr_dev,
                               block_D_red.get(), block_D.get(), mype, npes, block_D.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    passed = passed && cutlass::reference::device::BlockCompareEqual(
                           block_ref_D_red.get(), block_D_red.get(), block_D.size());

    free(ref_D_ptr);
    CUDA_CHECK(cudaFree(ref_D_ptr_dev));
    return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options) {
    initialize(options);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    auto arguments = args_from_options(options);

    auto grid = gemm.get_grid_shape(arguments);
    dim3 blockShape = GemmKernel::get_block_shape();

    int sm_count;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int max_active_blocks = gemm.maximum_active_blocks();
    printf("%d Grid dimension: (%d, %d, %d), block: (%d, %d, %d), occupancy: %d\n", mype, grid.x,
           grid.y, grid.z, blockShape.x, blockShape.y, blockShape.z, sm_count);
    int max_concurrent_blocks = sm_count * max_active_blocks;
    if (max_concurrent_blocks < (grid.x * grid.y * grid.z)) {
        fprintf(stderr,
                "Grid size exceeds maximum concurrent blocks. Using Tile-granular "
                "APIs requires all thread blocks to be concurrent across PEs\n");
        exit(1);
    }
    // create teams
    // each block has 1 warpgroup acting as epilogue, so num_teams = #blocks
    num_teams = grid.x * grid.y * grid.z;
    teams = (nvshmem_team_t *)malloc(num_teams * sizeof(nvshmem_team_t));
    for (int i = 0; i < num_teams; ++i) {
        nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &teams[i]);
    }
    CUDA_CHECK(cudaMalloc((void **)&teams_dev, num_teams * sizeof(nvshmem_team_t)));
    CUDA_CHECK(
        cudaMemcpy(teams_dev, teams, num_teams * sizeof(nvshmem_team_t), cudaMemcpyHostToDevice));

    // populate AR arguments
    arguments.allReduceArgs = {block_D.get(),   block_D_red.get(), stride_D,
                               nvshmem_my_pe(), nvshmem_n_pes(),   teams_dev};

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    Result result;
    result.passed = verify(options, mype, npes);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
        exit(-1);
    }

    // Run profiling loop
    if (options.iterations > 0) {
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < options.iterations; ++iter) {
            CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
            CUTLASS_CHECK(gemm.run());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();

        // Compute average runtime and GFLOPs.
        float elapsed_ms = timer.elapsed_millis();
        result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
        result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

        std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k
                  << std::endl;
        std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
        std::cout << "  GFLOPS: " << result.gflops << std::endl;
    }

    return 0;
}

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {
    // initialize nvshmem
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(mype));
    printf(" Executing PE: %d out of %d\n", mype, npes);

    // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
    // and must have compute capability at least 100a.

    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
        std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        return 0;
    }

    cudaDeviceProp props;
    int current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (props.major != 10 || props.minor != 0) {
        std::cerr << "This example requires a GPU with compute capability 100a)." << std::endl;
        return 0;
    }

    //
    // Parse options
    //

    Options options;

    options.parse(argc, args);

    if (options.help) {
        options.print_usage(std::cout) << std::endl;
        return 0;
    }

    //
    // Evaluate CUTLASS kernels
    //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    run<Gemm>(options);
#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    nvshmem_barrier_all();

    for (int i = 0; i < num_teams; ++i) {
        nvshmem_team_destroy(teams[i]);
    }
    nvshmem_barrier_all();

    block_D.free();
    block_D_red.free();
    block_ref_D.free();
    free(teams);
    CUDA_CHECK(cudaFree(teams_dev));
    nvshmem_finalize();
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
