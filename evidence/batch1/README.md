# Batch 1 contract harness evidence

Retained SHA-256-verified outputs for Batch 1 issues (`#1`, `#40`, `#41`, `#43`, `#52`, `#56`, `#58`, `#60`), each executed twice into fresh directories and verified byte-identical across all four emitted files (`inputs.json`, `results.json`, `summary.md`, `COMPLETE`). Exact digests are recorded in [manifest.json](manifest.json).

## Suites

| Suite | Issue(s) | Execution class | Physical / artifact gate | Summary |
| --- | --- | --- | --- | --- |
| `issue-01-diffusion` | `#1` | `bounded_contract_harness` | `unavailable_hardware_runtime` (real DiffuCoder/AR models) | [summary.md](issue-01-diffusion/summary.md) |
| `issue-40-trace` | `#40` | `bounded_contract_harness` | `unavailable_hardware_runtime` (NVIDIA Nsight/CUPTI capture) | [summary.md](issue-40-trace/summary.md) |
| `issue-41-stage` | `#41` | `bounded_contract_harness` | `unavailable_hardware_runtime` (Apple M2 Ultra / Tenstorrent P150) | [summary.md](issue-41-stage/summary.md) |
| `issue-43-power` | `#43` | `bounded_contract_harness` | `unavailable_hardware_runtime` (physical accelerator / XProf) | [summary.md](issue-43-power/summary.md) |
| `issue-52-tokenizer` | `#52` | `bounded_contract_harness` | `unavailable_hardware_runtime` (live serving backends) | [summary.md](issue-52-tokenizer/summary.md) |
| `issue-56-58-60-related-work` | `#56`, `#58`, `#60` | `blocked_on_artifacts` | `blocked_on_artifacts` / `unavailable_hardware_runtime` | [summary.md](issue-56-58-60-related-work/summary.md) |

## Honesty boundary

All numerical parity, trace-correlation, cadence-control, tokenizer-parity, and proposal-contract checks above executed deterministically on CPU. No physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, AWS F2 FPGA, or 27B serve ran locally. Upstream external figures remain strictly unverified non-gating related work.
