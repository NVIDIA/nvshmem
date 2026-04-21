# NVSHMEM - GPUNetIO Transport Preview Branch

**Note: This is a preview branch of the NVSHMEM transport implementation using [DOCA GPUNetIO Open Source](https://github.com/NVIDIA-DOCA/gpunetio) on the GPU data path as an alternative to IBGDA for testing purposes. The transport will be released with version 3.7.**

## Build with GPUNetIO Open
Set the following build-time variables via CMake to build the transport itself and the GPUNetIO Open library (>= v3.0.0) along with NVSHMEM (recommended):
```
-DNVSHMEM_GPUNETIO_SUPPORT=ON -DNVSHMEM_BUILD_GPUNETIO_LIBRARY=ON
```
Alternatively, if an existing installation of the GPUNetIO Open library should be used, set the environment variable `GPUNETIO_HOME` before building.


## Run with GPUNetIO Open
Specify a remote transport (e.g., IBDEVX) and use the environment variable `NVSHMEM_GPUNETIO_ENABLE_GDAKI` to enable GPUNetIO GDAKI on the GPU data path:
```
NVSHMEM_REMOTE_TRANSPORT=IBDEVX NVSHMEM_GPUNETIO_ENABLE_GDAKI=1
```

### Transport Environment Variables
See [src/modules/transport/common/env_defs.h](src/modules/transport/common/env_defs.h) for all GPUNetIO environment variables.

- `NVSHMEM_GPUNETIO_ENABLE_GDAKI` (`bool`, disabled by default): Set to enable GPU-initiated communication transport via GPUNetIO.
- `NVSHMEM_GPUNETIO_NIC_HANDLER` (`string`, default `"auto"`): Specifies the processor used for ringing the NIC's DB. Choices are `auto`, `gpu`, `cpu`.
  - `auto`: use GPU SMs and fallback to CPU if it is not supported (default).
  - `gpu`: use GPU SMs, regular DB.
  - `cpu`: use CPU.
- `NVSHMEM_GPUNETIO_NUM_RC_PER_PE` (`int`, default `2`): Number of QPs per peer PE used.
- `NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH` (`int`, default `32`): Number of requests to be batched before submitting to the NIC. It will be rounded up to the nearest power of 2. Set to `1` for aggressive submission.
- `NVSHMEM_GPUNETIO_NUM_FETCH_SLOTS_PER_RC` (`int`, default `1024`): Number of internal buffer slots for fetch operations for each QP. It will be rounded up to the nearest power of 2.
- `NVSHMEM_GPUNETIO_ENABLE_ORDERING_SEMANTIC` (`bool`, default `false`): Set to enable ordering semantic for DDP (Direct Data Placement) mode for GPUNetIO, requires DOCA SDK >= 3.4.
- `DOCA_SDK_LIB_PATH` (`string`, default `""`): Path to DOCA SDK passed to GPUNetIO Open library.


****************



NVSHMEM Overview
****************

NVSHMEM™ is a parallel programming interface based on OpenSHMEM that provides efficient and
scalable communication for NVIDIA GPU clusters. NVSHMEM creates a global address space for
data that spans the memory of multiple GPUs and can be accessed with fine-grained 
GPU-initiated operations, CPU-initiated operations, and operations on CUDA® streams.

Quick Links
****************

Please see the following public links for information on building and working wih NVSHMEM:

[Project Homepage](https://developer.nvidia.com/nvshmem)

[Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html)

[Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/index.html)

[Best Practice Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/index.html)

[API Documentation](https://docs.nvidia.com/nvshmem/api/index.html)

[Devzone Topic Page](https://forums.developer.nvidia.com/tag/nvshmem)

The maintainers of the NVSHMEM project can also be contacted by e-mail at nvshmem@nvidia.com

Configuration file
******************

NVSHMEM options can be provided via a simple config file using `KEY=VALUE` syntax.

Config files are loaded in the following order (later files override earlier files):

- `/etc/nvshmem.conf`
- `~/.nvshmem.conf`
- The file pointed to by `NVSHMEM_CONF_FILE`

If a key is present in any loaded config file, its value **overrides the corresponding environment
variable**.

Example:

```
# Example /etc/nvshmem.conf file
NVSHMEM_DEBUG=WARN
# NVSHMEM_SOME_FLAG=1 # This line is a comment and would be ignored.
```
