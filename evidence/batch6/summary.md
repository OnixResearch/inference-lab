# Batch 6 Contract & Reference-Math Harness Suite (`#47`–`#51`, `#53`–`#55`, `#57`, `#59`, `#61`)

- Total suites: 11 (`bounded_contract_harness`: 11, `blocked_on_artifacts`: 0)
- Physical GPU / NVMe / 2xB200 / RTX 3090 sm_86 / DGX Spark / live server execution: `false`

| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |
| ---: | --- | --- | --- | --- |
| `#47` | `adv-spec-reasoning` | `issue-47-adv-spec-reasoning` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#48` | `nvme-coalescing` | `issue-48-nvme-coalescing` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#49` | `harness-lora-compare` | `issue-49-harness-lora-compare` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#50` | `latent-feedback` | `issue-50-latent-feedback` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#51` | `simplehc-mhc` | `issue-51-simplehc-mhc` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#53` | `serving-hill-climb` | `issue-53-serving-hill-climb` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#54` | `tufazip-online-codec` | `issue-54-tufazip-online-codec` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#55` | `clairic-extraction` | `issue-55-clairic-extraction` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#57` | `inkling-mini-serve` | `issue-57-inkling-mini-serve` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#59` | `maya-ik2-serve` | `issue-59-maya-ik2-serve` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#61` | `decode-megakernels` | `issue-61-decode-megakernels` | `bounded_contract_harness` | `unavailable_hardware_runtime` |

> All outputs are deterministic pure-Rust CPU contract verifications and numerical references. Physical accelerators, NVMe drives, live model serves, and missing checkpoints/repos remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates.
