# Contributing to NVSHMEM tests 

## What is an unit test ?
A unit-test under `test/unit` is limited to testing 1 top-level `nvshmem` internal API and mocking rest of the code/framework to bootstrap/teardown the aforementioned API to run either on bare-metal env or in a namespaced env (docker, VM, etc) with installed dependencies. Typically, these are rarely to never ran on GPU/NIC device. The test could include or depend directly on any nvshmem internal header files.

## What is a functional test ?
A functional-test under `test/functional` is limited to testing N top-level `nvshmem` external APIs of a given library. Typically, this should rarely to never demand mocking rest of the code/framework to bootstrap/teardown the aforementioned APIs and would run on a bare-metal env on one or multiple CPU/GPU/NIC devices (single or multi-node). The test must not include or depend directly on any nvshmem internal header file or sources.

## What is an integration test ?
A integration-test under `test/integration` is limited to testing N x M top-level `nvshmem` and other consumer libraries API/interfaces. Typically, this should rarely to never demand mocking rest of the code in its neighbourhood and would run on a bare-metal env on one or multiple CPU/GPU/NIC devices (single or multi-node). Similar to functional test, it must not include or depend directly on any nvshmem internal header file or sources.
