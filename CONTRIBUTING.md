# Contributing to NVSHMEM

Thank you for your interest in contributing to NVSHMEM! We appreciate the time and effort you're putting into helping improve the library. This document will help guide you through the contribution process.

## Before You Start

To help ensure your contribution can be accepted smoothly:

- **Open an issue first** for significant changes or new features. This allows us to discuss the approach before you invest significant time.
- **Check existing issues** to see if someone else is already working on something similar.
- **Ask questions** if you're unsure about anything. We're happy to help!

## Getting Started

1. **Fork the Repository**: Fork the NVSHMEM repository and clone your fork locally.

2. **Create a Branch**: Create a feature branch for your work.
   ```bash
   git checkout -b <my-feature-branch>
   ```

3. **Commit your Changes**: Commit and push into feature branch.
   ```bash
   git add <files-to-commit>
   git commit -s --message "<descriptive message, see below>"
   git push origin <my-feature-branch> --set-upstream
   ```

## What We're Looking For

We welcome contributions in the following areas:

- **Bug fixes**: Reproducible bugs with clear fixes are always appreciated
- **Performance improvements**: Targeted optimizations, guarded by appropriate checks
- **Platform support**: Extending NVSHMEM to new platforms, network fabrics, or transport backends
- **Documentation**: Improvements to README, comments, or usage examples
- **Test coverage**: New tests that exercise existing functionality

## Making Your Contribution Successful

To help us review and integrate your contribution efficiently:

### Scope and Focus
- Keep changes focused on a single issue or feature
- Break large changes into smaller, reviewable pieces when possible
- Avoid mixing unrelated changes (e.g., bug fix + formatting changes) in one PR

### Quality Standards
- Ensure your code compiles without warnings
- Test thoroughly on relevant hardware configurations
- Performance claims should be backed by measurements
- Public API changes require discussion (see below)

### Common Challenges

Some types of contributions require extra discussion before we can accept them:

- **Public API changes**: NVSHMEM's API stability is important to users. If your contribution changes public APIs, please open an issue to discuss the design first. We need to ensure backward compatibility and consistency.

- **Architecture-specific code**: NVSHMEM supports specific GPU architectures and network configurations. Contributions targeting unsupported architectures may not be accepted unless there's a clear plan for ongoing maintenance.

- **Transport-layer changes**: NVSHMEM supports multiple transport backends (IBRC, IBGDA, IBDevX, UCX, libfabric). Changes to transport modules should be tested across applicable configurations.

- **Workarounds for external issues**: If a change works around a problem in another component (drivers, libraries, frameworks), we may need to address it differently. Let's discuss the root cause first.

- **Incomplete implementations**: Partial features or fixes that don't fully address the problem are difficult to maintain. If you need help completing an implementation, let us know.

## New Features

If you're proposing a substantial new feature (e.g., new collective operations, transport mechanisms, algorithms, or significant architectural changes), we follow a more structured process:

1. **Open an issue** describing the feature and use case
2. **Discussion phase**: We'll discuss whether it fits NVSHMEM's direction
3. **Document Design**: Document the proposed architecture, a testing and validation plan, and any known performance impact
4. **Review**: The design will be reviewed by NVSHMEM maintainers
5. **Implementation**: We recommend waiting until the review has concluded. Once there's mutual agreement on the approach, proceed with implementation
6. **Pull request**: Submit your PR referencing the original issue and design doc

## Code Style

### General Guidelines
- Use clang-format for C/C++/CUDA code (configuration in `.clang-format`)
- Use yapf for Python code (nvshmem4py)
- Follow the existing style in files you're modifying

### Return Codes
- NVSHMEM functions should return an `int` status code
- All function calls should be guarded by `NVSHMEMI_NZ_ERROR_JMP` or similar macro
- All CUDA calls should be guarded with `CUDA_CHECK` or equivalent

### Comments and Documentation
- Write clear comments for complex algorithms
- Document assumptions and invariants
- Explain "why" not just "what" for non-obvious code

### Copyright Header
- This project will only accept contributions under the Apache2 license. For new files, use the following header:

```
/*
 * Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
```

## Pull Request Guidelines

### Before Submitting
1. Rebase your branch on the latest development branch
2. Ensure clean build with no warnings
3. Run applicable tests

### Commit Messages
- Use imperative mood: "Add feature" not "Added feature"
- First line: brief summary (50 chars or less)
- Second line blank
- Detailed explanation including the problem, solution, and any limitations
- Reference issue numbers where applicable

### Signed Commits
All commits must be signed off to certify you have the right to submit the code:
```bash
git commit -s -m "Your commit message"
```
This adds: `Signed-off-by: Your Name <your@email.com>`

This certifies your agreement with the Developer Certificate of Origin (DCO).

## Developer Certificate of Origin

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Code Review Process

After you submit a PR:

1. **Maintainer review**: NVSHMEM engineers will review your code
2. **Discussion**: We may ask questions or request changes
3. **Iteration**: Address feedback and update your PR
4. **Approval**: Once approved, we'll merge your contribution

Please be patient during review. We aim to provide initial feedback within a week.

## Getting Help

If you need help at any stage:
- Comment on your issue or PR
- Ask questions in GitHub Issues
- Refer to [NVSHMEM documentation](https://docs.nvidia.com/nvshmem/)

Thank you for contributing to NVSHMEM!
