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
