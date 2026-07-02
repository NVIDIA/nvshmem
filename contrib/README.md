# Contribution Policy for `contrib/`

This directory hosts community and partner contributions that extend NVSHMEM
with new capabilities, language bindings, tools, or higher-level APIs. The
content here is developed and maintained by its respective contributors, not
by the NVSHMEM core team, and falls outside the NVSHMEM release quality
standards.

## Purpose

`contrib/` is the right home for additions that:

- Build on top of NVSHMEM's **public** APIs (`nvshmem.h`, `nvshmemx.h`, and
  `nvshmem_host.h`) without modifying NVSHMEM core (no change in `src/`)
- Provide language bindings, communication algorithms, higher-level APIs,
  tools, or reference implementations

Changes to NVSHMEM core (anything under `src/`) should follow the standard
contribution process described in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Current contributions

| Directory | Description |
|-----------|-------------|
| [`nvshmem4rust/`](nvshmem4rust/) | Rust host and CUDA-Oxide device bindings for NVSHMEM |

## Upstreaming to `contrib/`

Before submitting a new contribution, open a GitHub issue to discuss the
proposal. Include a brief description of what is being added and why it belongs
in this repository rather than a separate project. The NVSHMEM team will
provide feedback on whether the contribution is a good fit.

### Requirements for acceptance

A pull request adding a new `contrib/` entry must satisfy all of the following:

1. **Self-contained directory** — all code lives under a single subdirectory
   (`contrib/<name>/`). Apart from adding the project to the table above, the
   contribution must not modify files outside that directory.

2. **README** — a `README.md` explains the purpose, build instructions, usage,
   and hardware or software prerequisites.

3. **Named maintainer(s)** — a `## Maintainers` section in the subdirectory's
   `README.md` lists at least one GitHub username or email address responsible
   for ongoing maintenance. The maintainer does not have to be an NVSHMEM team
   member.

4. **License** — the contribution must be compatible with NVSHMEM's
   [Apache 2.0 license](../License.txt). Third-party dependencies must be
   documented in `ThirdPartyNotices.txt` or an equivalent file within the
   directory.

5. **Build and test** — the contribution builds cleanly without warnings
   against a current NVSHMEM release. A basic functional test or benchmark is
   strongly encouraged.

6. **No internal dependency** — code relies only on NVSHMEM's public headers
   and exported symbols. Depending on internal headers or unexported symbols is
   unsupported and may break without notice.

### Review process

Pull requests are reviewed by the NVSHMEM team. Unlike core changes, the
review focuses primarily on:

- Correctness and absence of obvious bugs
- Adherence to the requirements above
- Risk of confusion with NVSHMEM's official behavior, such as naming that
  could imply core NVSHMEM endorsement

The NVSHMEM team may request changes to scope, naming, or structure before
accepting a contribution.

## Maintenance responsibilities

Each contribution has **one or more designated maintainers** responsible for:

- Keeping the code building against current NVSHMEM releases
- Responding to bug reports and pull requests within a reasonable time
- Updating the contribution when NVSHMEM public APIs change in ways that
  affect it

The NVSHMEM core team does **not** maintain `contrib/` entries and is not
obligated to fix breakage caused by NVSHMEM changes. When a breaking NVSHMEM
change is anticipated, the team will make a best-effort attempt to notify
affected `contrib/` maintainers in advance.

### Removal policy

A contribution may be removed if:

- It fails to build against a current NVSHMEM release and the maintainer does
  not respond to a removal warning within **four weeks**
- The maintainer explicitly requests removal
- The contribution violates licensing requirements or contains security issues
  that cannot be resolved promptly

Removal will be preceded by a notice on the relevant issue or pull request.
