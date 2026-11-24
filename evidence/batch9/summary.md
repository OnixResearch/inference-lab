# Batch 9 Contract & Reference-Math Harness Suite (`#85`–`#95`)

- Total suites: 11 (`bounded_contract_harness`: 11, `blocked_on_artifacts`: 0)
- Physical GPU / DGX Spark GB10 / 2-node RDMA / vLLM+SGLang OCR / live model execution: `false`

| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |
| ---: | --- | --- | --- | --- |
| `#85` | `link-sparse-attention` | `issue-85-link-sparse-attention` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#86` | `mit-qda-quant` | `issue-86-mit-qda-quant` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#87` | `knowledgator-sys1-extract` | `issue-87-knowledgator-sys1-extract` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#88` | `unlimited-ocr-longdoc` | `issue-88-unlimited-ocr-longdoc` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#89` | `loop-kv-compression` | `issue-89-loop-kv-compression` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#90` | `alora-prefix-cache` | `issue-90-alora-prefix-cache` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#91` | `perseus-mpk-rdma` | `issue-91-perseus-mpk-rdma` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#92` | `moe-spec-roofline` | `issue-92-moe-spec-roofline` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#93` | `anchor-label-drafter` | `issue-93-anchor-label-drafter` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#94` | `parasite-feature-control` | `issue-94-parasite-feature-control` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#95` | `ling-fp8kv-alloc-cap` | `issue-95-ling-fp8kv-alloc-cap` | `bounded_contract_harness` | `unavailable_hardware_runtime` |

> All outputs are deterministic pure-Rust CPU contract verifications and numerical references (`default_enabled: false`). Physical accelerators, DGX Spark GB10 262K serves, 2-node RDMA clusters, live model serves, and unlocated public upstream artifacts remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates.
