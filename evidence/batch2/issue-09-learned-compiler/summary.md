# Issue #9: Learned-Compiler Asset Recovery & Preserved Baselines

- Execution class: `blocked_on_artifacts`
- Artifact/host gate: `blocked_on_artifacts` (`external_learned_compiler_checkout_and_darwin_apple_container_host`)
- Checked Compiler Lab tree: `50dcbdced71df3a4f4cfad2d99638ad6e7bd69ae` (`research/learned-compiler` absent; migration explicitly blocked on missing assets)

## Preserved Historical Baselines
- `brittonr/compiler-lab#30` (`qwen-causal-baseline (released vs adapted; reasoning-mode separated)`): historical claim preserved; current replay=`blocked_on_artifacts`; new measurement=`none`
- `brittonr/compiler-lab#31` (`diffusiongemma-baseline (BF16 / unmapped_float32 / mapped_float32)`): historical claim preserved; current replay=`blocked_on_artifacts`; new measurement=`none`

## Integration Boundary Controls
| Control | Outcome | Isolation Weakened | Reason |
| --- | --- | --- | --- |
| `synthetic-positive-qwen-apple-container` | `accepted` | `false` | none |
| `synthetic-negative-diffusion-failed-assembly` | `invalid` | `false` | failed_assembly_gate |
| `negative-linux-host-no-isolation-weakening` | `unavailable_hardware_runtime` | `false` | linux_host_lacks_darwin_or_apple_container_isolation |
| `negative-abi-profile-mismatch` | `invalid` | `false` | frozen_abi_profile_mismatch |
| `negative-frozen-checker-identity-mismatch` | `invalid` | `false` | frozen_checker_identity_mismatch |
| `negative-feedback-final-contamination` | `invalid` | `false` | feedback_final_partition_contamination |
| `negative-attempt-budget-exhausted` | `exhausted` | `false` | attempt_budget_exhausted |
| `negative-hidden-teacher-compiler-repair` | `invalid` | `false` | hidden_teacher_replacement_or_model_oracle_forbidden |
| `negative-model-supplies-trusted-expected-output` | `invalid` | `false` | hidden_teacher_replacement_or_model_oracle_forbidden |

> Explicitly `blocked_on_artifacts`. No private Compiler Lab asset or unlocated checkpoint was fabricated or copied.
