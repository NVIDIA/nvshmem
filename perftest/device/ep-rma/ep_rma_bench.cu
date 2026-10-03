/*
 * b1_ep_bench.cu - EP-shaped small-message RMA benchmark for NVSHMEM on PCIe.
 *
 * Workload model (execution plan section 7.3):
 *   P = PEs, T = tokens per source PE, E = P*experts_per_pe, k = top-k,
 *   H = hidden size, b = bytes per element,
 *   M[s][d] = token-expert copies routed from source PE s to destination PE d
 *
 * Round (staged, fixed capacity, no cross-round overlap):
 *   pack -> RMA dispatch -> quiet/barrier -> validate dispatch -> expert+combine
 *        -> RMA combine back -> quiet/barrier -> validate combine
 *
 * Variants
 *   v0  one RMA per logical fragment: per (token,slot) copy one payload put
 *       (H*b bytes) + one metadata put (8 bytes)
 *   v2  application-level packing per destination PE: one packed payload put +
 *       one packed metadata put per destination
 *   v3  packing with metadata appended to the same contiguous staging block:
 *       one put per destination
 *
 * v1 (transport-native IBRC batching) is deliberately absent: this host has no
 * RDMA device, so that path is BLOCKED and must never be reported as tested.
 *
 * Completion contract: stream-ordered putmem + nvshmemx_quiet_on_stream, then
 * nvshmem_barrier_all(). Deliberately conservative; the plan requires the sync
 * boundary to be declared, not removed.
 *
 * Routing is a pure function of (mode, s, t, slot, seed), so every rank can
 * regenerate every source's routing table locally. No routing data crosses MPI,
 * which keeps the correctness oracle independent of the transport under test.
 */
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <nvshmem_host.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _e = (expr);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::fprintf(stderr, "rank %d: CUDA %s failed: %s\n", rank, #expr,    \
                         cudaGetErrorString(_e));                                 \
            MPI_Abort(MPI_COMM_WORLD, 2);                                         \
        }                                                                         \
    } while (0)

enum class DType { F32, F16 };

struct Copy {
    int token;      // token index on the source PE
    int slot;       // top-k slot
    int expert;     // global expert id
    int dest;       // destination PE
    int local_exp;  // expert index local to the destination PE
};

struct Args {
    int tokens = 64, topk = 2, epp = 4, hidden = 512, elem_bytes = 4;
    int iters = 20, warmup = 3;
    std::string variant = "v0", routing = "uniform", json;
    unsigned seed = 12345;
};

// Deterministic, order-independent router. Identical on every rank.
static std::vector<Copy> make_route(const std::string &mode, int s, int T, int k, int E, int epp,
                                    unsigned seed) {
    std::vector<Copy> out;
    out.reserve((size_t)T * k);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < k; ++j) {
            unsigned h = (unsigned)(s * 73856093u) ^ (unsigned)(t * 19349663u) ^
                         (unsigned)(j * 83492791u) ^ (seed * 2654435761u);
            h ^= h >> 13; h *= 1274126177u; h ^= h >> 16;
            int expert = (mode == "hotspot" && (h % 100u) < 50u) ? 0 : (int)(h % (unsigned)E);
            Copy c;
            c.token = t; c.slot = j; c.expert = expert;
            c.dest = expert / epp; c.local_exp = expert % epp;
            out.push_back(c);
        }
    }
    return out;
}

// Metadata word: token | slot<<16 | local_exp<<32
static __host__ __device__ inline unsigned long long pack_meta(const Copy &c) {
    return (unsigned long long)(unsigned)c.token |
           ((unsigned long long)(unsigned)(c.slot & 0xffff) << 16) |
           ((unsigned long long)(unsigned)c.local_exp << 32);
}
static __host__ __device__ inline int meta_token(unsigned long long m) { return (int)(m & 0xffffull); }
static __host__ __device__ inline int meta_slot(unsigned long long m) { return (int)((m >> 16) & 0xffffull); }
static __host__ __device__ inline int meta_lexp(unsigned long long m) { return (int)((m >> 32) & 0xffffffffull); }

__global__ void fill_send_f32(float *send, int T_, int H, int pe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T_ * H) return;
    send[i] = (float)((pe * 100 + (i / H)) % 64);
}
__global__ void fill_send_f16(__half *send, int T_, int H, int pe) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T_ * H) return;
    send[i] = __float2half((float)((pe * 100 + (i / H)) % 64));
}

__global__ void pack_f32(const float *send, float *stage, const int *tok, int H) {
    int c = blockIdx.x, t = tok[c];
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        stage[(size_t)c * H + h] = send[(size_t)t * H + h];
}
__global__ void pack_f16(const __half *send, __half *stage, const int *tok, int H) {
    int c = blockIdx.x, t = tok[c];
    for (int h = threadIdx.x; h < H; h += blockDim.x)
        stage[(size_t)c * H + h] = send[(size_t)t * H + h];
}

// Expert transform + accumulate, one CUDA block per SOURCE PE.
//
// Receive layout is recv[source_pe][copy_slot][hidden], so the copies this PE
// received from source s are the contiguous run at offset s*recv_cap; they are
// NOT a single contiguous run across sources. ret_stage is likewise per source,
// because token indices are only unique within their originating PE.
__global__ void expert_combine_f32(const float *recv, const unsigned long long *recv_meta,
                                   int recv_cap, const int *exp_counts, int T_, int H,
                                   float *ret_stage) {
    int s = blockIdx.x;
    int n = exp_counts[s];
    if (n <= 0) return;
    const float *pay = recv + (size_t)s * recv_cap * H;
    const unsigned long long *meta = recv_meta + (size_t)s * recv_cap;
    float *out = ret_stage + (size_t)s * T_ * H;
    for (int j = 0; j < n; ++j) {
        unsigned long long m = meta[j];
        int tok = meta_token(m);
        float scale = (float)(meta_lexp(m) + 1);
        for (int h = threadIdx.x; h < H; h += blockDim.x)
            atomicAdd(&out[(size_t)tok * H + h], scale * pay[(size_t)j * H + h]);
    }
}
__global__ void expert_combine_f16(const __half *recv, const unsigned long long *recv_meta,
                                   int recv_cap, const int *exp_counts, int T_, int H,
                                   float *ret_stage) {
    int s = blockIdx.x;
    int n = exp_counts[s];
    if (n <= 0) return;
    const __half *pay = recv + (size_t)s * recv_cap * H;
    const unsigned long long *meta = recv_meta + (size_t)s * recv_cap;
    float *out = ret_stage + (size_t)s * T_ * H;
    for (int j = 0; j < n; ++j) {
        unsigned long long m = meta[j];
        int tok = meta_token(m);
        float scale = (float)(meta_lexp(m) + 1);
        for (int h = threadIdx.x; h < H; h += blockDim.x)
            atomicAdd(&out[(size_t)tok * H + h], scale * __half2float(pay[(size_t)j * H + h]));
    }
}

__global__ void reduce_combine(const float *ret_in, int P, int T_, int H, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T_ * H) return;
    float acc = 0.f;
    for (int d = 0; d < P; ++d) acc += ret_in[(size_t)d * T_ * H + i];
    out[i] = acc;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, npes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nexti = [&](int &o) { if (i + 1 < argc) o = std::atoi(argv[++i]); };
        auto nexts = [&](std::string &o) { if (i + 1 < argc) o = argv[++i]; };
        if (s == "--tokens") nexti(a.tokens);
        else if (s == "--topk") nexti(a.topk);
        else if (s == "--experts-per-pe") nexti(a.epp);
        else if (s == "--hidden") nexti(a.hidden);
        else if (s == "--elem-bytes") nexti(a.elem_bytes);
        else if (s == "--iters") nexti(a.iters);
        else if (s == "--warmup") nexti(a.warmup);
        else if (s == "--variant") nexts(a.variant);
        else if (s == "--routing") nexts(a.routing);
        else if (s == "--json") nexts(a.json);
        else if (s == "--seed") { int v = 12345; nexti(v); a.seed = (unsigned)v; }
    }
    if (a.variant == "v3") {
        if (rank == 0)
            std::fprintf(stderr, "v3 (fused metadata, one put per destination) is NOT_IMPLEMENTED\n");
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    if (a.variant == "v1") {
        if (rank == 0)
            std::fprintf(stderr, "v1 (IBRC native batching) is BLOCKED: no RDMA device on this host\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    DType dt = (a.elem_bytes == 2) ? DType::F16 : DType::F32;
    const size_t EB = (dt == DType::F32) ? 4u : 2u;

    int visible = 0;
    CUDA_CHECK(cudaGetDeviceCount(&visible));
    CUDA_CHECK(cudaSetDevice(rank % visible));

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    const int mype = nvshmem_my_pe();
    const int P = nvshmem_n_pes();
    const int T_ = a.tokens, K = a.topk, EPP = a.epp, H = a.hidden;
    const int E = P * EPP;
    const size_t S = (size_t)T_ * K;

    // --- routing, regenerated locally for every source ---
    std::vector<std::vector<Copy>> all_routes(P);
    for (int s = 0; s < P; ++s) all_routes[s] = make_route(a.routing, s, T_, K, E, EPP, a.seed);
    const std::vector<Copy> &route = all_routes[mype];

    std::vector<int> dest_begin(P, 0), dest_count(P, 0);
    {
        std::vector<int> cursor(P, 0);
        int run = 0;
        for (int d = 0; d < P; ++d) {
            dest_begin[d] = run;
            for (const Copy &c : route) if (c.dest == d) ++run;
            dest_count[d] = run - dest_begin[d];
        }
    }
    const int ncopy = (int)route.size();

    std::vector<int> exp_counts(P, 0);
    for (int s = 0; s < P; ++s)
        for (const Copy &c : all_routes[s]) if (c.dest == mype) exp_counts[s]++;
    std::vector<std::array<int, 4>> exp_dispatch;
    for (int s = 0; s < P; ++s)
        for (const Copy &c : all_routes[s])
            if (c.dest == mype) exp_dispatch.push_back({s, c.token, c.slot, c.local_exp});
    std::sort(exp_dispatch.begin(), exp_dispatch.end());

    // --- symmetric buffers ---
    const size_t recv_cap = S;
    char *send_buf = (char *)nvshmem_malloc((size_t)T_ * H * EB);
    char *recv = (char *)nvshmem_malloc((size_t)P * recv_cap * H * EB);
    unsigned long long *recv_meta =
        (unsigned long long *)nvshmem_malloc((size_t)P * recv_cap * sizeof(unsigned long long));
    // Staging is never addressed by a peer: it is only ever read locally as the
    // source of a put. Plain device allocations are both correct and simpler.
    char *stage = nullptr;
    unsigned long long *stage_meta = nullptr;
    CUDA_CHECK(cudaMalloc(&stage, (size_t)std::max(1, ncopy) * H * EB));
    CUDA_CHECK(cudaMalloc(&stage_meta, (size_t)std::max(1, ncopy) * sizeof(unsigned long long)));
    float *ret_in = (float *)nvshmem_malloc((size_t)P * T_ * H * sizeof(float));
    float *ret_stage = nullptr, *combine_out = nullptr;
    CUDA_CHECK(cudaMalloc(&ret_stage, (size_t)P * T_ * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&combine_out, (size_t)T_ * H * sizeof(float)));
    if (!send_buf || !recv || !recv_meta || !ret_in) {
        std::fprintf(stderr, "rank %d: symmetric allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    // Two orderings are needed:
    //   * route order        - v0 walks `route` directly and indexes pinned_meta[ci]
    //   * destination-grouped - v2/v3 pack into stage[dest_begin[d] ...], so the
    //     device plan arrays must be emitted in exactly that order.
    std::vector<int> h_tok(ncopy);
    std::vector<unsigned long long> h_meta(ncopy);
    std::vector<unsigned long long> h_meta_route(ncopy);
    for (int c = 0; c < ncopy; ++c) h_meta_route[c] = pack_meta(route[c]);
    {
        int j = 0;
        for (int d = 0; d < P; ++d)
            for (int ci = 0; ci < ncopy; ++ci)
                if (route[ci].dest == d) {
                    h_tok[j] = route[ci].token;
                    h_meta[j] = pack_meta(route[ci]);
                    ++j;
                }
        if (j != ncopy) { std::fprintf(stderr, "rank %d: pack plan mismatch\n", rank); MPI_Abort(MPI_COMM_WORLD, 5); }
    }
    int *d_tok = nullptr;
    unsigned long long *d_meta = nullptr;
    unsigned long long *d_meta_route = nullptr;   // route order, used by v0
    CUDA_CHECK(cudaMalloc(&d_tok, sizeof(int) * std::max(1, ncopy)));
    CUDA_CHECK(cudaMalloc(&d_meta, sizeof(unsigned long long) * std::max(1, ncopy)));
    CUDA_CHECK(cudaMalloc(&d_meta_route, sizeof(unsigned long long) * std::max(1, ncopy)));
    if (ncopy) {
        CUDA_CHECK(cudaMemcpy(d_meta_route, h_meta_route.data(),
                              sizeof(unsigned long long) * ncopy, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tok, h_tok.data(), sizeof(int) * ncopy, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_meta, h_meta.data(), sizeof(unsigned long long) * ncopy,
                              cudaMemcpyHostToDevice));
    }
    int *d_exp_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_exp_counts, sizeof(int) * P));
    CUDA_CHECK(cudaMemcpy(d_exp_counts, exp_counts.data(), sizeof(int) * P,
                          cudaMemcpyHostToDevice));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreate(&st));

    auto fill_send = [&]() {
        int n = T_ * H, th = 256, bl = (n + th - 1) / th;
        if (dt == DType::F32) fill_send_f32<<<bl, th, 0, st>>>((float *)send_buf, T_, H, mype);
        else fill_send_f16<<<bl, th, 0, st>>>((__half *)send_buf, T_, H, mype);
    };
    fill_send();
    nvshmem_barrier_all();

    auto ref_send = [&](int s, int t) { return (float)((s * 100 + t) % 64); };
    std::vector<float> exp_combine((size_t)T_ * H, 0.f);
    for (int t = 0; t < T_; ++t) {
        float mult = 0.f;
        for (const Copy &c : route) if (c.token == t) mult += (float)(c.local_exp + 1);
        for (int h = 0; h < H; ++h) exp_combine[(size_t)t * H + h] = ref_send(mype, t) * mult;
    }

    struct Round {
        double pack_us = 0, dispatch_us = 0, quiet_us = 0, validate_dispatch_us = 0;
        double expert_combine_us = 0, combine_send_us = 0, validate_combine_us = 0, round_us = 0;
        long long puts = 0, payload_bytes = 0, meta_bytes = 0, local_copies = 0, remote_copies = 0;
        bool dispatch_ok = true, combine_ok = true;
        double combine_max_abs_err = 0.0;
        std::string dispatch_detail, combine_detail;
    };
    std::vector<Round> rounds;
    rounds.reserve(a.iters);

    for (int it = -a.warmup; it < a.iters; ++it) {
        Round R;
        // Clear the receive state BEFORE the common barrier. Clearing it after the
        // barrier is a race: a peer that passes the barrier first can have its
        // puts land in our receive buffers, which we would then wipe.
        CUDA_CHECK(cudaMemsetAsync(recv, 0, (size_t)P * recv_cap * H * EB, st));
        CUDA_CHECK(cudaMemsetAsync(recv_meta, 0,
                                   (size_t)P * recv_cap * sizeof(unsigned long long), st));
        CUDA_CHECK(cudaMemsetAsync(ret_stage, 0, (size_t)P * T_ * H * sizeof(float), st));
        CUDA_CHECK(cudaMemsetAsync(ret_in, 0, (size_t)P * T_ * H * sizeof(float), st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        nvshmem_barrier_all();
        MPI_Barrier(MPI_COMM_WORLD);   // buffers are now clean on every PE

        cudaEvent_t e0, e1, e2, e3, e4;
        for (cudaEvent_t *e : {&e0, &e1, &e2, &e3, &e4}) CUDA_CHECK(cudaEventCreate(e));
        double t_round0 = now_us();

        // ---- pack (v2/v3 only) ----
        CUDA_CHECK(cudaEventRecord(e0, st));
        if (a.variant != "v0" && ncopy > 0) {
            if (dt == DType::F32)
                pack_f32<<<ncopy, 128, 0, st>>>((const float *)send_buf, (float *)stage, d_tok, H);
            else
                pack_f16<<<ncopy, 128, 0, st>>>((const __half *)send_buf, (__half *)stage, d_tok, H);
            CUDA_CHECK(cudaMemcpyAsync(stage_meta, d_meta,
                                       sizeof(unsigned long long) * ncopy,
                                       cudaMemcpyDeviceToDevice, st));
        }
        CUDA_CHECK(cudaEventRecord(e1, st));

        // ---- RMA dispatch ----
        if (a.variant == "v0") {
            std::vector<int> cursor(P, 0);
            for (int ci = 0; ci < ncopy; ++ci) {
                const Copy &c = route[ci];
                int idx = cursor[c.dest]++;
                int d = c.dest;
                void *dst = recv + (((size_t)mype * recv_cap + idx) * H * EB);
                const void *src = send_buf + (size_t)c.token * H * EB;
                unsigned long long *mdst = &recv_meta[(size_t)mype * recv_cap + idx];
                const unsigned long long *msrc = &d_meta_route[ci];
                if (d == mype) {
                    CUDA_CHECK(cudaMemcpyAsync(dst, src, (size_t)H * EB,
                                               cudaMemcpyDeviceToDevice, st));
                    CUDA_CHECK(cudaMemcpyAsync(mdst, msrc, sizeof(unsigned long long),
                                               cudaMemcpyDeviceToDevice, st));
                    R.local_copies++;
                } else {
                    nvshmemx_putmem_on_stream(dst, src, (size_t)H * EB, d, st);
                    nvshmemx_putmem_on_stream(mdst, msrc,
                                              sizeof(unsigned long long), d, st);
                    R.puts += 2;
                    R.payload_bytes += (long long)H * EB;
                    R.meta_bytes += (long long)sizeof(unsigned long long);
                    R.remote_copies++;
                }
            }
        } else {
            for (int d = 0; d < P; ++d) {
                int cnt = dest_count[d];
                if (cnt == 0) continue;
                // The receiver indexes the copies it gets from source `mype` as
                // recv[mype][0 .. dest_count[d]-1]. That block belongs to a single
                // destination PE, so every destination's packed run starts at
                // offset 0 of ITS OWN view of recv[mype] -- not at the sender's
                // dest_begin[d] offset inside the shared staging buffer, which is
                // what the packed `stage`/`stage_meta` layout uses internally.
                void *dst = recv + (((size_t)mype * recv_cap) * H * EB);
                const void *src = stage + ((size_t)dest_begin[d] * H * EB);
                unsigned long long *mdst = &recv_meta[(size_t)mype * recv_cap];
                if (d == mype) {
                    CUDA_CHECK(cudaMemcpyAsync(dst, src, (size_t)cnt * H * EB,
                                               cudaMemcpyDeviceToDevice, st));
                    CUDA_CHECK(cudaMemcpyAsync(mdst, &stage_meta[dest_begin[d]],
                                               (size_t)cnt * sizeof(unsigned long long),
                                               cudaMemcpyDeviceToDevice, st));
                    R.local_copies += cnt;
                } else {
                    nvshmemx_putmem_on_stream(dst, src, (size_t)cnt * H * EB, d, st);
                    nvshmemx_putmem_on_stream(mdst, &stage_meta[dest_begin[d]],
                                              (size_t)cnt * sizeof(unsigned long long), d, st);
                    R.puts += 2;
                    R.payload_bytes += (long long)cnt * H * EB;
                    R.meta_bytes += (long long)cnt * sizeof(unsigned long long);
                    R.remote_copies += cnt;
                }
            }
        }
        CUDA_CHECK(cudaEventRecord(e2, st));
        // nvshmemx_quiet_on_stream() only ENQUEUES the quiet on this stream, so a
        // host-side barrier issued right after it can run before the puts have
        // completed. Block on the stream, then quiet globally, then barrier.
        nvshmemx_quiet_on_stream(st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        nvshmem_quiet();
        CUDA_CHECK(cudaEventRecord(e3, st));
        // Declared completion boundary: NVSHMEM quiet + barrier, then a host-level
        // MPI barrier so no PE can start validating while a peer is still in flight.
        nvshmem_barrier_all();
        MPI_Barrier(MPI_COMM_WORLD);
        double t_disp_done = now_us();
        CUDA_CHECK(cudaEventRecord(e4, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        // ---- validate dispatch (host oracle: multiset compare + payload check) ----
        if (it >= 0) {
            std::vector<unsigned long long> mh((size_t)P * recv_cap, 0);
            CUDA_CHECK(cudaMemcpy(mh.data(), recv_meta,
                                  (size_t)P * recv_cap * sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            std::vector<std::array<int, 4>> got;
            for (int s = 0; s < P; ++s)
                for (int j = 0; j < exp_counts[s]; ++j) {
                    unsigned long long m = mh[(size_t)s * recv_cap + j];
                    got.push_back({s, meta_token(m), meta_slot(m), meta_lexp(m)});
                }
            std::sort(got.begin(), got.end());
            R.dispatch_ok = (got == exp_dispatch);
            if (!R.dispatch_ok) {
                char b[256];
                std::snprintf(b, sizeof b, "got=%zu exp=%zu", got.size(), exp_dispatch.size());
                R.dispatch_detail = b;
                for (size_t q = 0; q < std::min(got.size(), exp_dispatch.size()); ++q) {
                    if (got[q] != exp_dispatch[q]) {
                        std::snprintf(b, sizeof b, " got=%zu exp=%zu first_diff@%zu got=(%d,%d,%d,%d) exp=(%d,%d,%d,%d)",
                                      got.size(), exp_dispatch.size(), q,
                                      got[q][0], got[q][1], got[q][2], got[q][3],
                                      exp_dispatch[q][0], exp_dispatch[q][1], exp_dispatch[q][2],
                                      exp_dispatch[q][3]);
                        R.dispatch_detail += b;
                        break;
                    }
                }
            }
            if (R.dispatch_ok) {
                std::vector<char> row((size_t)H * EB);
                for (int s = 0; s < P && R.dispatch_ok; ++s) {
                    if (exp_counts[s] == 0) continue;
                    int t = meta_token(mh[(size_t)s * recv_cap]);
                    float expect = ref_send(s, t);
                    CUDA_CHECK(cudaMemcpy(row.data(), recv + ((size_t)s * recv_cap * H * EB),
                                          (size_t)H * EB, cudaMemcpyDeviceToHost));
                    for (int h = 0; h < H; ++h) {
                        float v = (dt == DType::F32) ? ((const float *)row.data())[h]
                                                     : __half2float(((const __half *)row.data())[h]);
                        if (v != expect) { R.dispatch_ok = false; R.dispatch_detail = "payload mismatch"; break; }
                    }
                }
            }
        }
        double t_val1 = now_us();

        // ---- expert transform + combine accumulation ----
        {
            if (dt == DType::F32)
                expert_combine_f32<<<P, 128, 0, st>>>((const float *)recv, recv_meta,
                                                      (int)recv_cap, d_exp_counts, T_, H,
                                                      ret_stage);
            else
                expert_combine_f16<<<P, 128, 0, st>>>((const __half *)recv, recv_meta,
                                                      (int)recv_cap, d_exp_counts, T_, H,
                                                      ret_stage);
        }
        CUDA_CHECK(cudaStreamSynchronize(st));
        double t_exp1 = now_us();

        // ---- combine back: one put per source PE we RECEIVED from ----
        // The condition is exp_counts[s] > 0 (copies s sent to us), not "we sent
        // to s": routing is not symmetric per PE pair. A source that sent us
        // nothing must not be waited on.
        for (int s = 0; s < P; ++s) {
            if (exp_counts[s] <= 0) continue;
            const float *src_block = ret_stage + (size_t)s * T_ * H;
            if (s == mype) {
                CUDA_CHECK(cudaMemcpyAsync(&ret_in[(size_t)mype * T_ * H], src_block,
                                           (size_t)T_ * H * sizeof(float),
                                           cudaMemcpyDeviceToDevice, st));
            } else {
                nvshmemx_putmem_on_stream(&ret_in[(size_t)mype * T_ * H], src_block,
                                          (size_t)T_ * H * sizeof(float), s, st);
            }
        }
        nvshmemx_quiet_on_stream(st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        nvshmem_quiet();
        nvshmem_barrier_all();
        MPI_Barrier(MPI_COMM_WORLD);
        double t_comb_done = now_us();

        // ---- validate combine ----
        reduce_combine<<<(T_ * H + 255) / 256, 256, 0, st>>>(ret_in, P, T_, H, combine_out);
        std::vector<float> got_c((size_t)T_ * H);
        CUDA_CHECK(cudaMemcpy(got_c.data(), combine_out, (size_t)T_ * H * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double t_valc1 = now_us();
        if (it >= 0) {
            double maxerr = 0.0;
            for (size_t i = 0; i < got_c.size(); ++i)
                maxerr = std::max(maxerr, (double)std::fabs(got_c[i] - exp_combine[i]));
            R.combine_max_abs_err = maxerr;
            R.combine_ok = (maxerr == 0.0);
        }

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1)); R.pack_us = ms * 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e1, e2)); R.dispatch_us = ms * 1000.0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e2, e3)); R.quiet_us = ms * 1000.0;
        (void)e4;
        R.validate_dispatch_us = t_val1 - t_disp_done;
        R.expert_combine_us = t_exp1 - t_val1;
        R.combine_send_us = t_comb_done - t_exp1;
        R.validate_combine_us = t_valc1 - t_comb_done;
        R.round_us = t_valc1 - t_round0;
        for (cudaEvent_t e : {e0, e1, e2, e3, e4}) CUDA_CHECK(cudaEventDestroy(e));

        if (it >= 0) rounds.push_back(R);
    }

    if (!a.json.empty()) {
        char path[1200];
        std::snprintf(path, sizeof path, "%s.rank%d.json", a.json.c_str(), rank);
        FILE *f = std::fopen(path, "w");
        if (f) {
            std::fprintf(f,
                "{\n \"rank\": %d, \"npes\": %d, \"variant\": \"%s\", \"routing\": \"%s\",\n"
                " \"config\": {\"tokens\": %d, \"topk\": %d, \"experts_per_pe\": %d, \"hidden\": %d,"
                " \"elem_bytes\": %d, \"iters\": %d, \"seed\": %u},\n"
                " \"local_copies\": %lld, \"remote_copies\": %lld,\n \"rounds\": [\n",
                rank, P, a.variant.c_str(), a.routing.c_str(), T_, K, EPP, H, (int)EB, a.iters,
                a.seed, rounds.empty() ? 0LL : rounds.back().local_copies,
                rounds.empty() ? 0LL : rounds.back().remote_copies);
            for (size_t i = 0; i < rounds.size(); ++i) {
                const Round &R = rounds[i];
                std::fprintf(f,
                    "  {\"iter\": %zu, \"pack_us\": %.3f, \"dispatch_us\": %.3f, \"quiet_us\": %.3f,"
                    " \"validate_dispatch_us\": %.3f, \"expert_combine_us\": %.3f,"
                    " \"combine_send_us\": %.3f, \"validate_combine_us\": %.3f, \"round_us\": %.3f,"
                    " \"puts\": %lld, \"payload_bytes\": %lld, \"meta_bytes\": %lld,"
                    " \"dispatch_ok\": %s, \"combine_ok\": %s, \"combine_max_abs_err\": %.9g, \"dispatch_detail\": \"%s\"}%s\n",
                    i, R.pack_us, R.dispatch_us, R.quiet_us, R.validate_dispatch_us,
                    R.expert_combine_us, R.combine_send_us, R.validate_combine_us, R.round_us,
                    R.puts, R.payload_bytes, R.meta_bytes, R.dispatch_ok ? "true" : "false",
                    R.combine_ok ? "true" : "false", R.combine_max_abs_err,
                    R.dispatch_detail.c_str(),
                    (i + 1 == rounds.size()) ? "" : ",");
            }
            std::fprintf(f, " ]\n}\n");
            std::fclose(f);
        }
    }
    if (rank == 0) {
        std::printf("B1_DONE variant=%s routing=%s npes=%d tokens=%d topk=%d hidden=%d "
                    "elem_bytes=%d iters=%d\n",
                    a.variant.c_str(), a.routing.c_str(), P, T_, K, H, (int)EB, a.iters);
        std::fflush(stdout);
    }

    nvshmem_free(send_buf); nvshmem_free(recv); nvshmem_free(recv_meta);
    nvshmem_free(ret_in); cudaFree(stage); cudaFree(stage_meta);
    cudaFree(ret_stage); cudaFree(combine_out); cudaFree(d_exp_counts); cudaFree(d_tok); cudaFree(d_meta);
    cudaFree(d_meta_route);
    cudaStreamDestroy(st);
    nvshmemx_hostlib_finalize();
    MPI_Finalize();
    return 0;
}
