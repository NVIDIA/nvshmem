#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cassert>
#include <vector>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstdarg>
#include <array>
#include <memory>

#include "helpers.h"
#include "utils.h"

/**
 * This tests is used to exercice many ways of doing an all-to-all with NVSHMEM.
 * Specifically, given a CPU input and CPU output, it tries all combinations of
 * - {sync, async} cudaMemcpy HtoD to NVSHMEM memory
 * - RMA using {host, device, stream} x {nbi, not nbi, p} APIs
 * - {sync, async} cudaMemcpy DtoH from NVSHMEM memory
 * It also uses different streams for the various operations, with appropriate stream
 * synchronizations Finally, it also can use CUDA graphs
 *
 * The idea is to exercice as many NVSHMEM paths as possible and ensure that
 * - The A2A is correct
 * - There are no deadlocks
 * - CUDA graphs are supported
 */

/**
 * The all-to-all is equivalent to
 * MPI_alltoall(src, count_per_pe, MPI_INT, dst, count_per_pe, MPI_INT, MPI_COMM_WORLD)
 *
 * With 3 PEs:
 *
 *              count_per_pe                    count_per_pe * num_pes
 *              <--->                           <----------->
 * Source:      [ A | B | C ]   [ D | E | F ]   [ G | H | I ]
 *              |---->          src_pe
 *              src_idx
 *
 * After...
 *
 * Destination: [ A | D | G ]   [ B | E | H ]   [ C | F | I ]
 *                              |------->       dst_pe
 *                              dst_idx
 */

#define MPI_CHECK(stmt)                                                  \
    do {                                                                 \
        int result = (stmt);                                             \
        if (0 != result) {                                               \
            fprintf(stderr, "[%s:%d] MPI failed\n", __FILE__, __LINE__); \
            exit(1);                                                     \
        }                                                                \
    } while (0)

#define NVSHMEM_CHECK(stmt)                                                  \
    do {                                                                     \
        int result = (stmt);                                                 \
        if (0 != result) {                                                   \
            fprintf(stderr, "[%s:%d] NVSHMEM failed\n", __FILE__, __LINE__); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

using int64 = long long int;
using payload_t = int;

inline payload_t src_payload(int64 src_pe, int64 num_pes, int64 src_idx, int64 count_per_pe) {
    // Ideally this function should return different values for different (src_pe, src_idx) or
    // at least there should be no trivial pattern to it
    return src_pe + (int)src_idx;
}

inline payload_t dst_payload(int64 dst_pe, int64 num_pes, int64 dst_idx, int64 count_per_pe) {
    int64 src_pe = (dst_idx / count_per_pe);
    int64 src_idx = dst_pe * count_per_pe + dst_idx % count_per_pe;
    return src_payload(src_pe, num_pes, src_idx, count_per_pe);
}

__forceinline__ __device__ int64 tid() {
    return ((int64)threadIdx.x) + ((int64)blockDim.x * (int64)blockIdx.x);
}

// One thread sends one message of length count_per_pe
__global__ void all_to_all_put(payload_t* dst, payload_t* src, int64 count_per_pe, int64 my_pe,
                               int64 num_pes) {
    int64 dst_pe = tid();
    if (dst_pe < num_pes) {
        nvshmem_int_put(dst + my_pe * count_per_pe, src + dst_pe * count_per_pe, count_per_pe,
                        dst_pe);
    }
}

// One thread sends one message of length count_per_pe
__global__ void all_to_all_put_nbi(payload_t* dst, payload_t* src, int64 count_per_pe, int64 my_pe,
                                   int64 num_pes) {
    int64 dst_pe = tid();
    if (dst_pe < num_pes) {
        nvshmem_int_put_nbi(dst + my_pe * count_per_pe, src + dst_pe * count_per_pe, count_per_pe,
                            dst_pe);
    }
}

// One thread sends one single element
__global__ void all_to_all_p(payload_t* dst, payload_t* src, int64 count_per_pe, int64 my_pe,
                             int64 num_pes) {
    int64 idx = tid();
    if (idx < num_pes * count_per_pe) {
        int64 dst_pe = idx / count_per_pe;
        int64 ii = idx % count_per_pe;
        payload_t src_value = src[dst_pe * count_per_pe + ii];
        payload_t* dst_ptr = dst + my_pe * count_per_pe + ii;
        nvshmem_int_p(dst_ptr, src_value, dst_pe);
    }
}

__global__ void copy(payload_t* dst, payload_t* src, int64 count) {
    int64 idx = tid();
    if (idx < count) {
        dst[idx] = src[idx];
    }
}

int allreduce_int_max(int val) {
    int out = -1;
    int* tmp = (int*)nvshmem_malloc(sizeof(int) * 2);
    CUDA_CHECK(cudaMemcpy(tmp + 0, &val, sizeof(int), cudaMemcpyDefault));
    NVSHMEM_CHECK(nvshmem_int_max_reduce(NVSHMEM_TEAM_WORLD, tmp + 1, tmp + 0, 1));
    CUDA_CHECK(cudaMemcpy(&out, tmp + 1, sizeof(int), cudaMemcpyDefault));
    nvshmem_free(tmp);
    return out;
}

struct nvshmem_info_t {
    const int rank;
    const int size;
    const bool print;
    nvshmem_info_t() : rank(nvshmem_my_pe()), size(nvshmem_n_pes()), print(nvshmem_my_pe() == 0){};
};

/**
 * Parameters of a test
 */
struct params_t {
    // Is the cudaMemcpy HtoD sync or async ?
    enum class HtoD_t { sync = 0, async = 1 };
    // Is the cudaMemcpy HtoD sync or async ?
    enum class DtoH_t { sync = 0, async = 1 };
    // What implementation to use for the all-to-all ?
    enum class A2A_t {
        host = 0,
        host_nbi = 1,
        stream = 2,
        stream_nbi = 3,
        device_put = 4,
        device_put_nbi = 5,
        device_p = 6
    };

    static const char* to_string(HtoD_t i) {
        switch (i) {
            case HtoD_t::sync:
                return "sync";
            case HtoD_t::async:
                return "async";
        }
        return "Unknown";
    }

    static const char* to_string(DtoH_t i) {
        switch (i) {
            case DtoH_t::sync:
                return "sync";
            case DtoH_t::async:
                return "async";
        }
        return "Unknown";
    }

    static const char* to_string(A2A_t i) {
        switch (i) {
            case A2A_t::host:
                return "host";
            case A2A_t::host_nbi:
                return "host_nbi";
            case A2A_t::stream:
                return "stream";
            case A2A_t::stream_nbi:
                return "stream_nbi";
            case A2A_t::device_put:
                return "device_put";
            case A2A_t::device_put_nbi:
                return "device_put_nbi";
            case A2A_t::device_p:
                return "device_p";
        }
        return "Unknown";
    }

    int verbose;
    int64 repeat;
    int64 cycle;
    HtoD_t HtoD;
    DtoH_t DtoH;
    A2A_t A2A;
    int64 count_per_pe;
    bool use_CUDA_graphs;

    std::string name() const {
        std::stringstream sstr;
        sstr << repeat << "_" << cycle << "_" << to_string(HtoD) << "_" << to_string(DtoH) << "_"
             << to_string(A2A) << "_" << count_per_pe << "_" << use_CUDA_graphs;
        return sstr.str();
    }

    static params_t create(int verbose, int64 repeat, int64 cycle, HtoD_t HtoD, DtoH_t DtoH,
                           A2A_t A2A, int64 count_per_pe, bool graphs) {
        return {verbose,      repeat,
                cycle,        HtoD,
                DtoH,         A2A,
                count_per_pe, (A2A == A2A_t::host || A2A == A2A_t::host_nbi) ? false : graphs};
    }
};

static const std::array<params_t::HtoD_t, 2> allHtoD = {params_t::HtoD_t::sync,
                                                        params_t::HtoD_t::async};
static const std::array<params_t::DtoH_t, 2> allDtoH = {params_t::DtoH_t::sync,
                                                        params_t::DtoH_t::async};
static const std::array<params_t::A2A_t, 7> allA2A = {
    params_t::A2A_t::host,       params_t::A2A_t::host_nbi,   params_t::A2A_t::stream,
    params_t::A2A_t::stream_nbi, params_t::A2A_t::device_put, params_t::A2A_t::device_put_nbi,
    params_t::A2A_t::device_p};
static const std::array<bool, 2> allGraphs = {false, true};

struct test {
    enum class stream_t { HtoD = 0, DtoH = 1, comm = 2, sync = 3 };
    enum class event_t { comm = 0, sync = 1 };

    /**
     * Allocates resources
     */
    test(nvshmem_info_t info, params_t params)
        : info(info),
          count(params.count_per_pe * info.size),
          params(params),
          graph(nullptr),
          instance(nullptr) {
        src_h = std::vector<int>(count, 0);
        for (int64 i = 0; i < count; i++) {
            src_h[i] = src_payload(info.rank, info.size, i, params.count_per_pe);
        }
        dst_h = std::vector<int>(count, std::numeric_limits<int>::min());
        inout_d = (payload_t*)nvshmem_malloc(count * sizeof(payload_t));
        remote_d = (payload_t*)nvshmem_malloc(count * sizeof(payload_t));

        for (auto& stream : streams) {
            CUDA_CHECK(cudaStreamCreate(&stream));
        }
        for (auto& event : events) {
            CUDA_CHECK(cudaEventCreate(&event));
        }
    }

    /**
     * Runs the cudaMemcpy HtoD (sync or async)
     * This is synchronous w.r.t. to the host and returns ones the memcpy is done.
     */
    void run_HtoD() {
        switch (params.HtoD) {
            case params_t::HtoD_t::sync:
                CUDA_CHECK(cudaMemcpy(inout_d, src_h.data(), count * sizeof(payload_t),
                                      cudaMemcpyDefault));
                CUDA_CHECK(cudaDeviceSynchronize());
                break;
            case params_t::HtoD_t::async:
                CUDA_CHECK(cudaMemcpyAsync(inout_d, src_h.data(), count * sizeof(payload_t),
                                           cudaMemcpyDefault, streams[(int)stream_t::HtoD]));
                CUDA_CHECK(cudaStreamSynchronize(streams[(int)stream_t::HtoD]));
                break;
            default:
                ffprintf("Wrong HtoD\n");
                exit(1);
        }
    }

    /**
     * Runs the all-to-all
     * This is synchronous w.r.t. to the host and returns ones the all-to-all is done.
     * Input and output are in inout_d
     * remote_d is the destination buffer for the RMA's
     */
    void run_A2A() {
        /** For stream based,
         *  stream_t::comm: <start capture> <comm kernel>  <comm event> / <copy kernel> <end
         * capture> \                             / stream_t::sync \ <barrier kernel> <sync event>
         *  For host based
         *  nvshmem_int_put_{nbi} -> nvshmem_barrier_all -> <copy kernel>
         */

        cudaStream_t comm_stream = streams.at((int)stream_t::comm);
        cudaStream_t sync_stream = streams.at((int)stream_t::sync);
        cudaEvent_t comm_event = events.at((int)event_t::comm);
        cudaEvent_t sync_event = events.at((int)event_t::sync);

        // If stream capturing: start capturing in comm
        if (params.use_CUDA_graphs) {
            CUDA_CHECK(cudaStreamBeginCapture(comm_stream, cudaStreamCaptureModeGlobal));
        }

        // Communication, inout_d --> remote_d
        switch (params.A2A) {
            // Host based
            case params_t::A2A_t::host: {
                for (int64 id = 0; id < info.size; id++) {
                    const int64 dst_pe = (info.rank + id) % info.size;
                    nvshmem_int_put(remote_d + info.rank * params.count_per_pe,
                                    inout_d + dst_pe * params.count_per_pe, params.count_per_pe,
                                    dst_pe);
                }
                break;
            }
            case params_t::A2A_t::host_nbi: {
                for (int64 id = 0; id < info.size; id++) {
                    const int64 dst_pe = (info.rank + id) % info.size;
                    nvshmem_int_put_nbi(remote_d + info.rank * params.count_per_pe,
                                        inout_d + dst_pe * params.count_per_pe, params.count_per_pe,
                                        dst_pe);
                }
                break;
            }
            // Stream based, in comm_stream
            case params_t::A2A_t::stream: {
                for (int64 id = 0; id < info.size; id++) {
                    const int64 dst_pe = (info.rank + id) % info.size;
                    nvshmemx_int_put_on_stream(remote_d + info.rank * params.count_per_pe,
                                               inout_d + dst_pe * params.count_per_pe,
                                               params.count_per_pe, dst_pe, comm_stream);
                }
                break;
            }
            case params_t::A2A_t::stream_nbi: {
                for (int64 id = 0; id < info.size; id++) {
                    const int64 dst_pe = (info.rank + id) % info.size;
                    nvshmemx_int_put_nbi_on_stream(remote_d + info.rank * params.count_per_pe,
                                                   inout_d + dst_pe * params.count_per_pe,
                                                   params.count_per_pe, dst_pe, comm_stream);
                }
                break;
            }
            case params_t::A2A_t::device_put: {
                // 1 thread per destination PE
                const int64 num_threads = 64;
                const int64 num_blocks = (info.size + num_threads - 1) / num_threads;
                all_to_all_put<<<num_blocks, num_threads, 0, comm_stream>>>(
                    remote_d, inout_d, params.count_per_pe, info.rank, info.size);
                break;
            }
            case params_t::A2A_t::device_put_nbi: {
                // 1 thread per destination PE
                const int64 num_threads = 64;
                const int64 num_blocks = (info.size + num_threads - 1) / num_threads;
                all_to_all_put_nbi<<<num_blocks, num_threads, 0, comm_stream>>>(
                    remote_d, inout_d, params.count_per_pe, info.rank, info.size);
                break;
            }
            case params_t::A2A_t::device_p: {
                // 1 thread per element
                const int64 num_threads = 256;
                const int64 num_blocks = (count + num_threads - 1) / num_threads;
                all_to_all_p<<<num_blocks, num_threads, 0, comm_stream>>>(
                    remote_d, inout_d, params.count_per_pe, info.rank, info.size);
                break;
            }
            default:
                ffprintf("Wrong A2A\n");
                exit(1);
        }
        CUDA_CHECK(cudaGetLastError());

        // NVSHMEM barrier
        switch (params.A2A) {
            // Host based
            case params_t::A2A_t::host:
            case params_t::A2A_t::host_nbi:
                nvshmem_barrier_all();
                CUDA_CHECK(cudaGetLastError());
                break;
            // Stream based, in sync_stream, with appropriate event sync w.r.t. comm_stream
            case params_t::A2A_t::stream:
            case params_t::A2A_t::stream_nbi:
            case params_t::A2A_t::device_put:
            case params_t::A2A_t::device_put_nbi:
            case params_t::A2A_t::device_p:
                // Order comm_stream -> sync_stream
                CUDA_CHECK(cudaEventRecord(comm_event, comm_stream));
                CUDA_CHECK(cudaStreamWaitEvent(sync_stream, comm_event, 0));
                // Barrier in sync_stream
                nvshmemx_barrier_all_on_stream(sync_stream);
                CUDA_CHECK(cudaGetLastError());
                // Order sync_stream -> comm_stream
                CUDA_CHECK(cudaEventRecord(sync_event, sync_stream));
                CUDA_CHECK(cudaStreamWaitEvent(comm_stream, sync_event, 0));
                break;
        }

        // Copy remote_d --> inout_d
        const int64 num_threads = 256;
        const int64 num_blocks = (count + num_threads - 1) / num_threads;
        switch (params.A2A) {
            // Host based
            case params_t::A2A_t::host:
            case params_t::A2A_t::host_nbi:
                copy<<<num_blocks, num_threads>>>(inout_d, remote_d, count);
                CUDA_CHECK(cudaGetLastError());
                break;
            // Stream based, in comm_stream
            case params_t::A2A_t::stream:
            case params_t::A2A_t::stream_nbi:
            case params_t::A2A_t::device_put:
            case params_t::A2A_t::device_put_nbi:
            case params_t::A2A_t::device_p:
                copy<<<num_blocks, num_threads, 0, comm_stream>>>(inout_d, remote_d, count);
                CUDA_CHECK(cudaGetLastError());
        }

        // If we do stream capture, finish capturing comm_stream, replay the graph and sync
        // comm_stream
        if (params.use_CUDA_graphs) {
            switch (params.A2A) {
                case params_t::A2A_t::host:
                case params_t::A2A_t::host_nbi:
                    ffprintf("Capturing but using host API!\n");
                    exit(1);
                case params_t::A2A_t::stream:
                case params_t::A2A_t::stream_nbi:
                case params_t::A2A_t::device_put:
                case params_t::A2A_t::device_put_nbi:
                case params_t::A2A_t::device_p:
                    CUDA_CHECK(cudaStreamEndCapture(comm_stream, &graph));
                    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr,
                                                    (unsigned long long)0));
                    CUDA_CHECK(cudaGraphLaunch(instance, comm_stream));
                    break;
            }
        }

        // Sync comm_stream
        switch (params.A2A) {
            case params_t::A2A_t::host:
            case params_t::A2A_t::host_nbi:
                CUDA_CHECK(cudaDeviceSynchronize());
                break;
            case params_t::A2A_t::stream:
            case params_t::A2A_t::stream_nbi:
            case params_t::A2A_t::device_put:
            case params_t::A2A_t::device_put_nbi:
            case params_t::A2A_t::device_p:
                CUDA_CHECK(cudaStreamSynchronize(comm_stream));
                break;
        }
    }

    /**
     * Runs the cudaMemcpy DtoH (sync or async)
     * This is synchronous w.r.t. to the host and returns ones the memcpy is done.
     */
    void run_DtoH() {
        switch (params.DtoH) {
            case params_t::DtoH_t::sync:
                CUDA_CHECK(cudaMemcpy(dst_h.data(), inout_d, count * sizeof(payload_t),
                                      cudaMemcpyDefault));
                CUDA_CHECK(cudaDeviceSynchronize());
                break;
            case params_t::DtoH_t::async:
                CUDA_CHECK(cudaMemcpyAsync(dst_h.data(), inout_d, count * sizeof(payload_t),
                                           cudaMemcpyDefault, streams[(int)stream_t::DtoH]));
                CUDA_CHECK(cudaStreamSynchronize(streams[(int)stream_t::DtoH]));
                break;
            default:
                ffprintf("Wrong DtoH\n");
                exit(1);
        }
    }

    /**
     * Check for correctness
     * Returns 0 if no errors are found, 1 otherwise
     */
    int64 check() {
        int64 checked = 0;
        int64 errors = 0;
        for (int64 i = 0; i < count; i++) {
            payload_t expected = dst_payload(info.rank, info.size, i, params.count_per_pe);
            if (dst_h[i] != expected) {
                if (errors < 10) {
                    ffprintf("Error, got %d, expected %d at index %lld\n", dst_h[i], expected, i);
                }
                errors++;
            }
            checked++;
        }

        // Sanity check
        assert(checked == info.size * params.count_per_pe);
        assert(count == info.size * params.count_per_pe);
        if (errors != 0) {
            ffprintf("Rank %d/%d, %zu errors\n", info.rank, info.size, errors);
        }
        if (info.print && params.verbose > 0) {
            ffprintf("%s done, checked %zu, total errors = %zu\n", params.name().c_str(), checked,
                     errors);
        }

        return (errors == 0 ? 0 : 1);
    }

    /**
     * Frees all resources
     */
    ~test() {
        for (const auto& stream : streams) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        for (const auto& event : events) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
        nvshmem_free(inout_d);
        nvshmem_free(remote_d);
        if (params.use_CUDA_graphs) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }

    const nvshmem_info_t info;
    const int64 count;
    const params_t params;
    payload_t *inout_d, *remote_d;
    std::vector<int> src_h, dst_h;
    std::array<cudaStream_t, 4> streams;
    std::array<cudaEvent_t, 2> events;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
};

/**
 * Repro for https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3955366&cmtNo=
 * mpirun -n 8 ./nvsh_smoke_usage_pattern 0 8192 8 128
 * on Selene
 *
 * Usage
 *
 * mpirun -n <num procs> nvsh_smoke_usage_pattern <verbose> <repeat> <cycles> <max_count_per_pe>
 *                                                <HtoD> <DtoH> <A2A> <graph>
 * - verbose:           (default 0)   Verbosity. 0 = quiet, 1 = lots of output
 * - repeat:            (default 4)   Number of overall steps for the testsuite
 * - cycle:             (default 8)   Within each step, how many times to repeat everything
 * - max_count_per_pe:  (default 128) Maximum number of elements exchanged between pairs of PEs
 * - HtoD:              (optional)    if set, only uses this HtoD algorithm to HtoD exchanges.
 *                                    Otherwise, use all algorithms.
 *                                    See params_t::HtoD_t
 * - DtoH:              (optional)    if set, only uses this DtoH algorithm to DtoH exchanges.
 *                                    Otherwise, use all algorithms.
 *                                    See params_t::DtoH_t
 * - A2A:               (optional)    if set, only uses this A2A algorithm to the all-to-all.
 *                                    Otherwise, use all algorithms.
 *                                    See params_t::A2A_t
 * - graph:             (optional)    if set, force enabling(1)/disabling(0) graphs.
 *                                    Otherwise, use both.
 */

int main(int argc, char* argv[]) {
    init_wrapper(&argc, &argv);

    nvshmem_info_t info;
    if (info.print) {
        printf_date("NVSHMEM initialized on all PEs at ");
        fflush(stdout);
    }

    const int verbose = (argc >= 2) ? std::atoi(argv[1]) : 0;
    const int64 repeats = (argc >= 3) ? (int64)std::atoll(argv[2]) : (int64)4;
    const int64 cycles = (argc >= 4) ? (int64)std::atoll(argv[3]) : (int64)8;
    const int64 max_count_per_pe = (argc >= 5) ? (int64)std::atoll(argv[4]) : (int64)128;

    auto HtoD_opt = (argc >= 6) ? make_optional((params_t::HtoD_t)std::atoi(argv[5])) : nullopt;
    auto DtoH_opt = (argc >= 7) ? make_optional((params_t::DtoH_t)std::atoi(argv[6])) : nullopt;
    auto A2A_opt = (argc >= 8) ? make_optional((params_t::A2A_t)std::atoi(argv[7])) : nullopt;
    auto graph_opt = (argc >= 9) ? make_optional((bool)std::atoi(argv[8])) : nullopt;

    if (info.print) {
        printf("Rank %d/%d. Verbose %d, repeats %lld, cycles %lld, max_count_per_pe %lld\n",
               info.rank, info.size, verbose, repeats, cycles, max_count_per_pe);
        printf("HtoD_opt %s, DtoH_opt %s, A2A_opt %s, graph %s\n",
               HtoD_opt.has_value() ? params_t::to_string(HtoD_opt.value()) : "not_specified",
               DtoH_opt.has_value() ? params_t::to_string(DtoH_opt.value()) : "not_specified",
               A2A_opt.has_value() ? params_t::to_string(A2A_opt.value()) : "not_specified",
               graph_opt.has_value() ? (graph_opt.value() ? "yes" : "no") : "not_specified");
    }

    nvshmem_sync_all();
    if (info.print) {
        printf_date("CUDA initialized on all PEs at ");
        fflush(stdout);
    }

    std::mt19937 g(0 /** 100% random  - _should_ be the same on every rank **/);
    int64 errors = 0;

    for (int64 repeat = 0; repeat < repeats; repeat++) {
        if (info.print) {
            printf_date("Step %zu starting at ", repeat);
            fflush(stdout);
        }

        // All the combinations we test, possibly multiple times
        std::vector<params_t> params;
        std::uniform_int_distribution<int64> dist_count_per_pe(1, max_count_per_pe);
        for (int64 cycle = 0; cycle < cycles; cycle++) {
            for (const auto& HtoD : allHtoD) {
                for (const auto& DtoH : allDtoH) {
                    for (const auto& A2A : allA2A) {
                        for (const auto& graph : allGraphs) {
                            params.push_back(
                                params_t::create(verbose, repeat, cycle, HtoD_opt.value_or(HtoD),
                                                 DtoH_opt.value_or(DtoH), A2A_opt.value_or(A2A),
                                                 dist_count_per_pe(g), graph_opt.value_or(graph)));
                        }
                    }
                }
            }
        }

        /**
         * This creates and run many `test`, all at the same time, in random orders
         * This effectively interleave tests to increase the likelihood of errors
         * and hangs
         */

        // Alloc in random order
        std::shuffle(params.begin(), params.end(), g);
        std::vector<std::unique_ptr<test>> tests;
        for (auto& param : params) {
            tests.emplace_back(std::unique_ptr<test>(new test(info, param)));
        }

        // HtoD {sync, async} in random order
        std::shuffle(tests.begin(), tests.end(), g);
        for (auto& test : tests) {
            test->run_HtoD();
        }

        // AlltoAll {host, stream, device} x {nbi, not nbi, p} x {graph, no graph} in random order
        std::shuffle(tests.begin(), tests.end(), g);
        for (auto& test : tests) {
            test->run_A2A();
        }

        // DtoH {sync, async} in random order
        std::shuffle(tests.begin(), tests.end(), g);
        for (auto& test : tests) {
            test->run_DtoH();
        }

        // Check errors
        for (auto& test : tests) {
            errors += test->check();
        }

        // Free, in random order
        std::shuffle(tests.begin(), tests.end(), g);
        for (auto& test : tests) {
            test.reset();
        }

        int global_error = allreduce_int_max(errors);
        if (info.print) {
            ffprintf("Step %zu done with %zu tests. Errors so far %zu\n", repeat, tests.size(),
                     global_error);
        }

        fflush(stdout);
    }

    int error = (errors != 0 ? 1 : 0);
    error = allreduce_int_max(error);

    finalize_wrapper();

    if (error != 0) {
        printf("FAILED\n");
        return 1;
    } else {
        if (info.print) {
            printf("PASSED\n");
        }
        return 0;
    }
}
