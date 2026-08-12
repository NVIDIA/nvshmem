---
name: nvshmem-troubleshoot-and-report-bugs
description: Diagnose NVSHMEM runtime failures and prepare bug reports for launch, initialization, crashes, hangs, correctness, transport, or topology issues.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - troubleshooting
    - bug
---

# NVSHMEM Troubleshooting and Bug Reporting

## Purpose

Diagnose NVSHMEM runtime failures from high-signal evidence. Match documented problems conservatively, verify version-sensitive claims with live official documentation, separate NVSHMEM defects from application or environment failures, and prepare a portable report when escalation is useful.

## Prerequisites

- Require no credentials, API keys, or privileged access.
- Work from the user's exact launch command, complete output, and generated logs. Use direct read-only inspection only when the user supplies access and asks for it.
- Keep `$nvshmem-docs` available to verify version-specific behavior and `$nvshmem-install` available for installation or build handoffs. When a sibling skill is not registered, read its `SKILL.md` directly.
- Keep [troubleshooting-knowledge-base.md](references/troubleshooting-knowledge-base.md) available for diagnosis and [bug-report-template.md](references/bug-report-template.md) available for escalation.

## Inputs

- **Required:** Obtain the exact command or complete batch-script launch stanza and complete stdout and stderr from every PE, including lines before the first error and any per-PE logs.
- **Conditional:** Obtain output that confirms INFO-level NVSHMEM logging before diagnosing an in-process NVSHMEM failure. Apply the documented pre-start exception only when NVSHMEM demonstrably cannot emit logs.
- **Optional:** Obtain the full NVSHMEM version, CUDA and driver versions, launcher and bootstrap, transport settings, PE/node/GPU counts, rank-to-GPU mapping, topology, smallest failing scale, and last known-good configuration only when they distinguish plausible causes.
- **Source precedence:** Prefer raw artifacts from the current run, then explicit values in the current user prompt or invocation, then non-conflicting agent context, then focused follow-up answers. Report conflicts and prefer the current raw artifact unless the user confirms a correction.

## Limitations

- Do not confirm a root cause from paraphrased errors, incomplete PE output, a partial signature match, or timing changes alone.
- Treat the bundled knowledge base as a routing snapshot, not current authority. Require live official documentation for version-specific claims.
- Do not claim that a fixed-in release proves every earlier release is affected or that an upgrade is compatible without checking the collected environment.
- Do not perform privileged platform changes or directly submit reports. Provide read-only checks, administrator questions, and paste-ready Markdown instead.

## Instructions

### Start With the Command and Output

Ask first for both of the following when either is missing, then stop:

1. The exact command or complete batch-script launch stanza, including environment settings, launcher options, container invocation, and module loads.
2. The complete stdout and stderr from every PE, including lines before the first error and any per-PE log files.

Do not begin with a broad environment questionnaire. Do not diagnose from a paraphrased error when raw output can be obtained.

### Require INFO Logging

Determine whether the captured output contains `NVSHMEM_DEBUG=INFO` or clear INFO-level NVSHMEM records. The command line alone is not sufficient because a launcher may fail to propagate its environment. If the output does not confirm INFO logging, ask the user to repeat the same run with:

```text
NVSHMEM_DEBUG=INFO
NVSHMEM_DEBUG_SUBSYS=ALL
NVSHMEM_DEBUG_FILE=nvshmem-%h-%p.log
```

Preserve the executable, launcher, PE and node counts, binding, transport, container, and workload arguments. Put the variables where the selected launcher propagates them to every PE. For a simple direct launch, prefix them to the command. For `srun`, `mpirun`, `nvshmrun`, or another launcher, invoke `$nvshmem-docs` for the detected NVSHMEM version before giving launcher-specific syntax when propagation is uncertain.

Ask for the repeated command, stdout and stderr, and every generated `nvshmem-<host>-<pid>.log`. Wait for that evidence before analyzing NVSHMEM behavior.

Apply one exception: diagnose immediately when the failure demonstrably occurs before NVSHMEM starts and cannot emit NVSHMEM logs, such as shell parsing, executable lookup, loader startup, or launcher admission failure. State why the debug rerun cannot add NVSHMEM evidence. Do not use the exception merely because an existing error looks familiar.

### Build the Evidence Record

After the logging gate, extract before asking for more information:

- the first causal error or divergence on each affected PE, separate from cascading aborts;
- the NVSHMEM version and any host/device library mismatch reported by INFO logs;
- launcher and bootstrap, selected transports, relevant `NVSHMEM_*` settings, PE/node/GPU counts, and rank-to-GPU mapping;
- whether the failure changes with one PE, one node, different GPU pairs, or multiple nodes;
- the operation and phase: launch, initialization, allocation, communication, synchronization, collective, finalization, or process teardown.

Ask only for missing facts that distinguish plausible causes. Never fabricate a version, topology, environment setting, or reproduction result.

### Diagnose Conservatively

Read [troubleshooting-knowledge-base.md](references/troubleshooting-knowledge-base.md) for every diagnosis after the logging gate or pre-start exception. Use its topic index to select the relevant section, then search exact log signatures within that section before falling back to semantic matches.

Classify the result as exactly one of:

1. **Confirmed bundled knowledge-base match**: the signature and required conditions match an entry.
2. **Version-specific known issue or later-release fix**: official release notes establish applicability to the detected version.
3. **Likely NVSHMEM diagnosis not in the knowledge base**: evidence points into NVSHMEM, but no bundled or documented issue matches.
4. **Likely external issue**: evidence points to the application, CUDA, launcher, loader, network, driver, scheduler, container, or platform configuration.

Use class 1 whenever a bundled entry is confirmed, even when that entry assigns ownership to an external component; state ownership separately in `Diagnosis`. Use class 4 only when no bundled or versioned documented issue matches. For partial signature matches, use class 3 or 4 rather than calling the match confirmed. Treat a similar release-note symptom as a lead, not proof that the user has the same defect. A release note that says only that a bug was fixed establishes the fix release, not every affected earlier release. Describe an older installed version as predating a candidate fix unless official evidence explicitly establishes that it is affected or a controlled upgrade test confirms the match.

Invoke `$nvshmem-docs` with the complete detected version before asserting environment-variable behavior, requirements, limitations, known issues, fixes, or upgrade guidance. Ask for the complete version when logs do not provide it and a version claim matters. When the sibling skill is not registered, read and follow `../nvshmem-docs/SKILL.md` directly.

Before recommending a target upgrade, verify that release's compatibility and known issues against the collected CUDA, driver, GPU, CPU, NCCL, bootstrap, and transport context. If that context is incomplete, present the release as a candidate test and name the missing compatibility checks instead of calling it suitable.

Separate:

- **Observation**: what the command and logs directly show.
- **Documentation evidence**: the matched FAQ, environment-variable entry, limitation, known issue, or fixed issue.
- **Inference**: the most likely explanation and its confidence.
- **Verification**: the smallest safe test that can confirm or reject the inference.

If no entry matches, say exactly: `No matching entry was found in the bundled NVSHMEM troubleshooting knowledge base.` Then provide a likely diagnosis, confidence, competing explanations, and targeted next checks.

For a likely external issue, identify the likely owner and give concrete next checks. Do not stop at “not NVSHMEM.” Route installation and build failures to `$nvshmem-install`; route performance-only regressions to an available NVSHMEM performance skill, or to `$nvshmem-docs` when none is available.

Do not perform privileged platform changes. For IMEX, driver, fabric-manager, device-node, kernel-module, NIC, or scheduler configuration, provide read-only checks and tell the user what to ask the administrator to verify.

### Present the Diagnosis

Return these headings:

- `Evidence Collected`
- `Diagnosis`
- `Knowledge-Base Match`
- `Version Applicability`
- `Next Actions`

Include direct official source links for documentation-backed claims. Keep commands copyable and mark placeholders. Prefer one discriminating test at a time over a long generic checklist.

### Prepare a Bug Report or Question

Offer a report when the issue remains unresolved, is likely an NVSHMEM defect, or the user explicitly requests one. Do not routinely offer escalation for a resolved local configuration problem or a clearly external failure.

When INFO-level NVSHMEM logs are absent, explicitly ask the user to reproduce the failure before support submission with:

```text
NVSHMEM_DEBUG=INFO
NVSHMEM_DEBUG_SUBSYS=ALL
NVSHMEM_DEBUG_FILE=nvshmem-%h-%p.log
```

Preserve the executable, launcher, PE and node counts, binding, transport, container, and workload arguments. Ask for the exact repeated command, complete stdout/stderr from every PE, and all generated log files. A draft may still be generated when the user asks to preserve the investigation, but label it incomplete and include this rerun as a required pre-submission step; do not present it as support-ready.

When preparing one, read [bug-report-template.md](references/bug-report-template.md) and fill it from collected evidence. Mirror the `NVSHMEM issue or bug` GitHub form: start the title with `[Issue]:`, use its exact field headings, and select exactly one supported impact value. Preserve exact commands and the first causal error. Use `Not provided` for missing facts and list those gaps; never invent them. Remind the user to redact credentials, tokens, internal hostnames and IPs, and user-identifying data before sharing logs.

Return the Markdown in a fenced block so it can be pasted into GitHub, NVONLINE, or email. Do not require or directly submit through a connector.

## Troubleshooting

Use these patterns when the investigation stalls:

| Error or signal | Likely cause | Solution |
| --- | --- | --- |
| No INFO records appear after setting debug variables | The launcher did not propagate the environment, or the process failed before NVSHMEM started | Verify propagation with version-matched launcher syntax. Use the pre-start exception only when the emitting component proves NVSHMEM did not start. |
| Logs show only peer aborts or launcher cleanup | The visible lines are cascading consequences | Compare every PE log and use the earliest causal or divergent line, including preceding context. |
| A knowledge-base signature matches but its required conditions do not | The match is only a hypothesis | Classify the issue as likely NVSHMEM or likely external, state the mismatch, and run one discriminating test. |
| The NVSHMEM version is missing or live documentation is unavailable | Version applicability cannot be verified | Ask for the complete version and label version-sensitive guidance unverified; do not infer applicability from the snapshot. |
| INFO logging changes or removes the failure | The failure may be timing-sensitive | Preserve both commands and outcomes, repeat enough trials to characterize frequency, and do not call the logging change a fix. |

## Examples

- **Missing evidence:** Given only “NVSHMEM hangs during init,” ask for the exact launch stanza and complete output from every PE, then stop.
- **Missing INFO logs:** Given a complete command and non-INFO output from an in-process failure, request the same run with the three debug settings and all generated logs; preserve scale and arguments.
- **Pre-start failure:** Given `error while loading shared libraries` before any PE enters NVSHMEM, state why INFO logging cannot help, identify the loader as the emitting component, and route an NVSHMEM-library installation problem to `$nvshmem-install`.
- **Confirmed bundled match:** Given `No more teams available` plus matching team-usage conditions, classify it as a confirmed bundled match, verify `NVSHMEM_MAX_TEAMS` for the complete version, and propose one minimal rerun.
- **No match:** State the required no-match sentence, separate observation from inference, give one discriminating test, and offer a filled report only if the issue remains unresolved or looks like an NVSHMEM defect.
