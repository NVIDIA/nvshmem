# Compatibility with NVSHMEM

NVSHMEM public releases follow semantic versioning with a numeric `MAJOR.MINOR.PATCH` base version.
Internal development versioning extends this with the suffixed values as defined in the subsection
that follows.

For the numeric base version, backward-incompatible API or ABI changes require a `MAJOR` version
update, backward-compatible API or ABI additions require a `MINOR` version update, and compatible
bug fixes use a `PATCH` version update.

## Artifact version stages

The CMake project version remains numeric as `MAJOR.MINOR.PATCH.0` so ABI and runtime vendor
version checks continue to use numeric fields. Publishable artifacts derive a separate
artifact version through `cmake_config/NVSHMEMVersioning.cmake`, using branch defaults from
`cmake_config/NVSHMEMVersionStage.cmake`:

- `dev`: snapshots from development and other non-release branches (including `devel`), formatted
  as `MAJOR.MINOR.PATCH.dev<N>`.
- `rc`: release candidates from `release/vMAJOR.MINOR.PATCH`, formatted as `MAJOR.MINOR.PATCHrc<N>`.
- `release`: final release artifacts, formatted as `MAJOR.MINOR.PATCH`.

The stage and stage number are selected by branch defaults or build configuration.
