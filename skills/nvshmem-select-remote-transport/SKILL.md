---
name: nvshmem-select-remote-transport
description: Select an NVSHMEM remote transport from target system and kernel evidence. Use for inter-node selection, compatibility checks, or configuration.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - transport
    - tuning
---

# Select an NVSHMEM Remote Transport

## Purpose

Recommend one NVSHMEM transport configuration from static evidence. Separate hard eligibility from expected kernel fit, distinguish documented facts from inference.

Focus on read-only system probes and source inspection. Do not run NVSHMEM applications, correctness tests, or performance benchmarks.

## Prerequisites

- Keep [transport-matrix.md](references/transport-matrix.md) and [kernel-fit.md](references/kernel-fit.md) readable as part of the skill package. Read both before ranking transports.
- Require an exact NVSHMEM version and an inventory of installed transport plugins for an unconditional recommendation.
- Require the selected CUDA kernel/application source or an equivalent communication profile.
- Collect target-node evidence through an already available unprivileged shell or obtain it from the user. Never request root access or inspect a login node as if it were the target compute node.
- Use the sibling `$nvshmem-docs` workflow, or its readable `../nvshmem-docs/SKILL.md`, only when the user requests current/exact official verification or the bundled matrix cannot establish release applicability.
- Require no API key, privileged command, network access for inspection, or benchmark execution.

If direct target-node access is unavailable, provide the bundled collector and request its complete labeled output.

## Inputs

### Required for an Unconditional Recommendation

- Exact NVSHMEM version and installed transport-plugin inventory.
- Target fabric/provider: InfiniBand, RoCE, EFA, Slingshot/CXI, UCX-managed, or NVLink-only.
- Selected CUDA kernel/application source, or a profile covering:
  - host, on-stream, or device API use;
  - RMA, AMO, signaling, synchronization, or collective operations;
  - typical and range of message sizes;
  - issuing threads, warps, and CTAs;
  - peer fan-out and expected simultaneous message rate.
- Job scale: total PEs and local PEs per node.
- Intended outcome: general default, latency-sensitive, throughput-oriented, or compatibility-first.

### Optional

- `NVSHMEM_PREFIX`, CMake cache, module/container details, launcher command, environment, and NVSHMEM configuration files.
- GPU, NIC, driver, kernel, OFED/rdma-core, UCX, libfabric, GPUNetIO, or DOCA details not available through inspection.
- GPU-to-NIC topology and local PE binding.
- Host/device traffic mix, CPU availability, persistent-kernel behavior, batching opportunities, and completion frequency.
- Required atomics, registered user buffers, VMM, DCI/DCT, multi-NIC, Spectrum-X, or custom-QP features.
- Preferred or prohibited transports.

Inspect before asking. Ask only for missing facts that can change eligibility, ranking, or the emitted configuration. When required evidence remains unavailable, return conditional branches instead of inventing it.

## Instructions

### 1. Resolve the Target

- Identify the application, exactly selected kernel or host call path, deployment, NVSHMEM version, job scale, and optimization goal.
- Treat assessment and recommendation as read-only. Do not patch application code or change launcher/system configuration.
- Limit source inspection to the selected kernel, its device call graph, and directly corresponding host/on-stream call sites. Ignore unrelated NVSHMEM kernels.
- Determine whether the relevant communication can cross nodes. Local P2P/NVLink traffic does not use the selected remote transport.

### 2. Collect System Evidence

When already on the target compute node, announce the read-only probe and invoke the bundled collector:

```text
run_script("scripts/collect-transport-facts.sh")
```

When the installation prefix is known, pass it explicitly rather than relying on shell expansion:

```text
run_script("scripts/collect-transport-facts.sh", "--prefix", "/opt/nvshmem")
```

Replace the example prefix with the resolved absolute prefix. Resolve the script relative to this `SKILL.md`. Preserve partial output when it exits with status 2.

When not on the target node, ask the user to run `bash scripts/collect-transport-facts.sh [--prefix PATH]` there and return the complete output. Do not infer cluster-wide uniformity from one node without stating the assumption.

### 3. Classify the Communication

Read [kernel-fit.md](references/kernel-fit.md) and record:

- API surface and exact operations;
- remote versus local paths;
- message-size distribution and whether operations are packed;
- concurrent issuers, peers, and expected physical message rate;
- ordering/completion frequency and QP pressure;
- host/device mix, CPU availability, and persistent-kernel behavior;
- scale-sensitive resource needs such as RC versus DCI/DCT.

Do not infer a high message rate merely from many logical elements. Account for application aggregation, one-thread issuance, warp coalescing, and batching.

### 4. Apply Hard Eligibility Gates

Read [transport-matrix.md](references/transport-matrix.md) completely. Exclude a candidate when any required condition is disproven:

- plugin unavailable in the selected NVSHMEM installation;
- exact release or operation unsupported;
- fabric/provider mismatch;
- missing driver, peer-memory/DMA-BUF, GDRCopy, DevX, UCX, libfabric, GPUNetIO, or DOCA prerequisite;
- incompatible memory mode, registered-buffer use, scale, or required feature.

Treat a missing fact as `conditional`, not `eligible`. Keep build support, runtime loadability, and hardware suitability separate.

For IBGDA, evaluate both the GPU data path and a separate host remote transport. For GPUNetIO GDAKI, require `gpunetio` as the remote transport. Never enable IBGDA and GPUNetIO GDAKI together.

### 5. Rank Eligible Candidates

Apply these rules in order without assigning artificial scores:

1. **Fabric-native gate:** Prefer libfabric with `efa` on EFA and `cxi` on Slingshot/CXI when the exact release supports the required operations and memory mode.
2. **No remote network:** Select `none` only when all relevant PEs communicate within a peer-reachable NVLink domain and no remote path is required.
3. **Fine-grained parallel device traffic:** Favor an eligible GDAKI path when many GPU issuers submit small independent remote operations and a CPU proxy would serialize them.
4. **GDAKI choice:**
   - Favor IBGDA when DCI/DCT connection scaling is required, GPUNetIO is absent, or the exact release has stronger required operation coverage.
   - Favor GPUNetIO GDAKI when its RC connection scale is acceptable and a unified GPUNetIO CPU/GPU path or verified DOCA feature is useful.
   - Do not treat GPUNetIO design goals as proof that it outperforms IBGDA for this workload.
5. **Proxy-friendly traffic:** Favor a proxy path for host/on-stream-dominant work, packed larger transfers, low message concurrency, or latency-sensitive isolated operations.
6. **Mellanox proxy choice:** Use IBRC as the conservative verbs default. Recommend IBDevX only when mlx5 DevX is available, its exact operation coverage is sufficient, and a concrete requirement such as avoiding a GDRCopy-dependent path justifies it. Do not claim general IBDevX superiority.
7. **UCX choice:** Favor UCX for a UCX-controlled deployment or a verified release-specific capability unavailable from the native alternatives. Preserve its experimental/status caveats from the exact documentation.

If two candidates remain close, choose the compatibility-first option as the recommendation, report the other as runner-up, and state that static analysis cannot establish the performance winner.

### 6. Emit Minimal Selection Configuration

Emit only the variables required to select the resolved configuration:

| Configuration | Minimal exports |
| --- | --- |
| IBRC | `NVSHMEM_REMOTE_TRANSPORT=ibrc` |
| IBDevX | `NVSHMEM_REMOTE_TRANSPORT=ibdevx` |
| IBGDA | `NVSHMEM_REMOTE_TRANSPORT=<resolved-host-transport>` and `NVSHMEM_IB_ENABLE_IBGDA=1` |
| GPUNetIO CPU | `NVSHMEM_REMOTE_TRANSPORT=gpunetio` and `NVSHMEM_GPUNETIO_ENABLE_GDAKI=0` |
| GPUNetIO GDAKI | `NVSHMEM_REMOTE_TRANSPORT=gpunetio` and `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1` |
| UCX | `NVSHMEM_REMOTE_TRANSPORT=ucx` |
| libfabric | `NVSHMEM_REMOTE_TRANSPORT=libfabric` and `NVSHMEM_LIBFABRIC_PROVIDER=<cxi|efa|verbs>` |
| No remote | `NVSHMEM_REMOTE_TRANSPORT=none` |

Do not add HCA mapping, QP counts, batching, NIC handlers, memory-mode changes, or provider tuning to this block. Put mandatory compatibility preconditions outside the block and route mapping/tuning to the appropriate sibling skill.

Withhold an unconditional export block when version, plugin, provider, operation coverage, or kernel profile is unresolved.

### 7. Return the Recommendation

Use this exact section order:

```text
Recommendation: <transport or combined host/GPU configuration; conditional if necessary>

Selection configuration
<minimal export block, or why it is withheld>

System eligibility evidence
<version, plugins, fabric/provider, driver/module/dependency evidence>

Kernel-fit evidence
<API surface, operation shape, sizes, concurrency, fan-out, completion, scale>

Runner-up
<candidate and the fact or tradeoff that kept it second>

Excluded candidates
<candidate: hard gate or unresolved prerequisite>

Confidence and unknowns
<high|medium|low; documented facts versus inference; missing evidence>

Static follow-up checks
<read-only checks or exact facts still needed; never benchmarks>
```

State the documentation version consulted when live verification was used. Describe performance as expected fit, not an assured result.

## Routing

- Route exact HCA selection and PE-to-NIC mapping to `$nvshmem-configure-nic-pe-mapping` when available.
- Route missing packages, plugins, or builds to `$nvshmem-install` when available.
- Route runtime initialization failures to `$nvshmem-troubleshoot-and-report-bugs` when available.
- If a sibling skill is unavailable, provide the boundary and required next evidence without inventing its behavior.

## Available Script

| Script | Purpose | Arguments |
| --- | --- | --- |
| [`scripts/collect-transport-facts.sh`](scripts/collect-transport-facts.sh) | Collect read-only NVSHMEM, GPU, kernel-module, RDMA, and provider evidence from a target compute node. | Optional `--prefix PATH` |


## Examples

- `Use $nvshmem-select-remote-transport. Inspect this target compute node and src/exchange.cu, then recommend a throughput-oriented transport for 256 PEs with 8 local PEs per node.`
- `Use $nvshmem-select-remote-transport with the attached collector output and communication profile. Recommend a compatibility-first transport for 32 PEs with 8 local PEs per node, and withhold exports if required evidence is missing.`


## Guardrails

- Do not run NVSHMEM binaries other than the read-only `nvshmem-info -n` query.
- Do not launch distributed jobs, allocate nodes, benchmark transports, or mutate system/application state.
- Do not recommend a plugin merely because its source exists; require evidence that the selected installation contains it.
- Do not convert absent evidence into compatibility.
- Do not claim that Ethernet link-layer evidence alone proves working RoCE.
- Do not let kernel heuristics override an incompatible fabric/provider or missing prerequisite.
- Do not emit both IBGDA and GPUNetIO GDAKI enables.

## Troubleshooting

- If the collector exits with status 2, preserve its output and resolve only the missing facts listed under `diagnostics`.
- If a plugin is missing or fails to initialize, verify the selected prefix and dependencies, then route runtime failures to `$nvshmem-troubleshoot-and-report-bugs`.
- If the recommendation remains conditional, obtain the exact version, provider, operation coverage, or kernel profile before emitting exports.

## Limitations

- Static analysis estimates architectural fit; it does not validate runtime correctness, measure performance, or guarantee the fastest transport.
- The bundled compatibility matrix is release-sensitive; verify the exact target release against its official documentation when the matrix cannot establish applicability.
- The collector describes one node and one selected installation. It does not prove cluster-wide uniformity, working RoCE, or complete operation coverage.
- Incomplete source, topology, plugin, or provider evidence requires a conditional recommendation with no unconditional exports.
