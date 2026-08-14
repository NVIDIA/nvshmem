# NVSHMEM Agent Skills
This folder provides skills for NVSHMEM to be used by AI agents.

## Skill Overview

### Getting Started and Support
- `nvshmem-install`: Plans and validates NVSHMEM installation using packages, containers, or source builds.
- `nvshmem-get-started`: Introduces NVSHMEM concepts and guides users through their first C/C++ or Python program, compilation, and launch.
- `nvshmem-troubleshoot-and-report-bugs`: Diagnoses launch, initialization, crash, hang, correctness, transport, and topology problems and prepares actionable bug reports.
- `nvshmem-docs`: Finds version-aware NVSHMEM documentation for APIs, releases, installation, transports, runtime settings, and troubleshooting, used by many other skills.

### Performance Tuning and System Configuration
- `nvshmem-tune-performance`: Routes performance-tuning requests to benchmarking, transport selection, NIC-to-PE mapping, or TMA enablement workflows.
- `nvshmem-enable-tma`: Prepares or reviews NVSHMEM CUDA kernels for TMA shared-memory registration and direct shared-memory transfers.
- `nvshmem-configure-nic-pe-mapping`: Recommends topology-aware NIC-to-PE mappings and the corresponding environment variable.
- `nvshmem-select-remote-transport`: Selects an appropriate remote transport such as IBRC, IBDevX, IBGDA, GPUNetIO, UCX, or libfabric—based on the system and kernel.
- `nvshmem-collect-performance-data`: Collects and packages NVSHMEM bandwidth and latency results together with system and topology evidence.

## Prerequisites
- The skills rely on the agent being able to fetch the NVSHMEM documentation from from `docs.nvidia.com` and `archive.docs.nvidia.com`. Ensure to grant this permission.
- Some skills depend on each other. For example, most skills require the `nvshmem-docs` skill to be available to search for appropriate documentation. Ensure to install all the NVSHMEM skills for the best experience.

## Installation and Usage
The skills can be installed and used in different ways:
- Point your agent to this skills directory and tell it to use the skills.
- Use symlinks to link the skills to your personal skills directory, e.g. `ln -s $NVSHMEM_SKILLS_DIRECTORY/nvshmem-install ~/.agents/skills/nvshmem-install`.
- When using the skills from a GitHub clone, the agent should automatically pick them up from the NVSHMEM repo's `.agents/skills`.
- Use the NVIDIA skills catalogue at https://github.com/NVIDIA/skills, see below for details.

### NVIDIA Skills Catalogue
The same skills are also provided in the NVIDIA skills catalogue at https://github.com/NVIDIA/skills.

Use the following commands to install the skills:

```
npx skills add nvidia/skills --skill nvshmem-install --yes
npx skills add nvidia/skills --skill nvshmem-get-started --yes
npx skills add nvidia/skills --skill nvshmem-docs --yes
npx skills add nvidia/skills --skill nvshmem-troubleshoot-and-report-bugs --yes
npx skills add nvidia/skills --skill nvshmem-tune-performance --yes
npx skills add nvidia/skills --skill nvshmem-enable-tma --yes
npx skills add nvidia/skills --skill nvshmem-configure-nic-pe-mapping --yes
npx skills add nvidia/skills --skill nvshmem-select-remote-transport --yes
npx skills add nvidia/skills --skill nvshmem-collect-performance-data --ye
```
