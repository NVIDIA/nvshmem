NVSHMEM4Py Overview
*******************

NVSHMEM4Py is a Python package that provides a Pythonic interface to NVSHMEM

NVSHMEM4Py follows the NVSHMEM SLA. The details of the NVSHMEM SLA [is available here](https://docs.nvidia.com/nvshmem/api/sla.html).

Quick Links
****************

NVSHMEM4Py is a component of NVSHMEM™. Please see the following public links for information on building and working wih NVSHMEM:

[Project Homepage](https://developer.nvidia.com/nvshmem)

[Release Notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/index.html)

[Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/index.html)

[Best Practice Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/best-practice-guide/index.html)

[API Documentation](https://docs.nvidia.com/nvshmem/api/index.html)

[Devzone Topic Page](https://forums.developer.nvidia.com/tag/nvshmem)

The maintainers of the NVSHMEM project can also be contacted by e-mail at nvshmem@nvidia.com

Wheel Build Configuration
*************************

By default, the build system discovers all Python versions >= 3.9 on the system and builds wheels for each one against CUDA 12 and 13. The following CMake options provide control over this behavior:

- ``NVSHMEM4PY_BUILD_ALL_WHEELS`` (default: ``ON``) — When ``OFF``, wheels are not built automatically during ``ninja``. Individual targets like ``build_nvshmem4py_wheel_cu12_3.12`` remain available.
- ``NVSHMEM4PY_PYTHON_VERSIONS`` — Semicolon-separated list of Python versions to build for (e.g., ``3.12`` or ``3.12;3.11``). When empty, all detected versions are used.
- ``NVSHMEM4PY_CUDA_VERSIONS`` — Semicolon-separated list of CUDA major versions (e.g., ``12`` or ``12;13``). When empty, defaults to ``12;13``.
- ``NVSHMEM4PY_PYTHON_EXECUTABLE_<major>_<minor>`` — Override the Python executable path for a specific version (e.g., ``-DNVSHMEM4PY_PYTHON_EXECUTABLE_3_12=/opt/venv/bin/python``).

Example: build only a Python 3.12 / CUDA 12 wheel::

    cmake -DNVSHMEM4PY_BUILD_ALL_WHEELS=OFF \
          -DNVSHMEM4PY_PYTHON_VERSIONS="3.12" \
          -DNVSHMEM4PY_CUDA_VERSIONS="12" ..
    ninja build_nvshmem4py_wheel_cu12_3.12

Example: build NVSHMEM core without any wheels::

    cmake -DNVSHMEM4PY_BUILD_ALL_WHEELS=OFF ..
    ninja
