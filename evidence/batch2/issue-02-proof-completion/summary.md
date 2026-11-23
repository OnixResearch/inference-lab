# Issue #2: Masked Proof Completion through Compiler Lab's Frozen Gate

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`real_prover_model_and_compiler_lab_lean_toolchain`)
- Cohort family: `lean-4.35.0-pinned`
- Generator-neutral (`#75`) vs temporary adapter equivalence: `true`
- Offline replay identical: `true`
- Decision: `INCONCLUSIVE_UNAVAILABLE_HARDWARE_RUNTIME`

## Candidate Admissions & Frozen-Context Guards
| Candidate | Arm | Verdict | Frozen OK | Reason |
| --- | --- | --- | --- | --- |
| `cand-01-valid-proof` | `diffusion-plus-repair` | `proved` | `true` | none |
| `cand-02-sorry-hole-high-conf` | `ar-plus-repair` | `rejected` | `true` | new_axiom_or_unproved_hole_forbidden |
| `cand-03-exec-body-modified` | `ar-completion` | `rejected` | `false` | frozen_executable_body_modified |
| `cand-04-weakened-contract` | `ar-completion` | `rejected` | `false` | frozen_statement_or_contract_weakened |
| `cand-05-disallowed-import` | `ar-plus-repair` | `rejected` | `false` | disallowed_import_outside_trust_closure |
| `cand-06-placeholder-leansearch` | `aesop-duper-baseline` | `rejected` | `true` | authoring_placeholder_not_replaced |
| `cand-07-mixed-verus-cohort` | `ar-completion` | `unsupported` | `false` | mixed_language_or_checker_family_forbidden |
| `cand-08-davinci-unmatched-python` | `davinci-tool-agent-matched` | `unsupported` | `true` | unmatched_tool_interface_optional_python_disabled |
| `cand-09-over-repair-budget` | `diffusion-plus-repair` | `rejected` | `true` | admission_budget_exceeded_including_repairs |
| `cand-10-checker-timeout` | `ar-plus-repair` | `timeout` | `true` | checker_budget_timeout |
| `cand-11-tested-not-proved` | `deterministic-reuse-tactics` | `tested` | `true` | bounded_tests_passed_without_complete_proof |
| `cand-12-checker-unknown` | `aesop-duper-baseline` | `unknown` | `true` | checker_returned_unknown_inconclusive |
| `cand-13-checker-infra-failure` | `ar-plus-repair` | `internal_failure` | `true` | checker_infrastructure_failure |
| `cand-14-stale-profile-identity` | `ar-completion` | `rejected` | `false` | stale_profile_or_toolchain_identity |
| `cand-15-frozen-span-modified` | `ar-completion` | `rejected` | `false` | frozen_context_span_modified |

> Bounded contract harness only. Compiler Lab retains proof intake/checking authority; no real model or Lean/Verus checker ran locally.
