# Batch 3 Contract & Reference-Math Harness Suite (`#12`–`#21`)

- Total suites: 10 (`bounded_contract_harness`: 9, `blocked_on_artifacts`: 1)
- Physical GPU / CUDA / ggml binary / Tenstorrent P150x2 / JAX / Qwen3.8-27B live serve executed: `false`

| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |
| ---: | --- | --- | --- | --- |
| `#12` | `rsi-bench` | `issue-12-rsi-bench` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#13` | `djev-protocol` | `issue-13-djev-protocol` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#14` | `rwkv-leaf` | `issue-14-rwkv-leaf` | `blocked_on_artifacts` | `blocked_on_artifacts` |
| `#15` | `ggml-ops` | `issue-15-ggml-ops` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#16` | `rwkv-ggml` | `issue-16-rwkv-ggml` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#17` | `graphical-sampling` | `issue-17-graphical-sampling` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#18` | `qwen38-quant` | `issue-18-qwen38-quant` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#19` | `deepseek-ref` | `issue-19-deepseek-ref` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#20` | `mla-comparison` | `issue-20-mla-comparison` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#21` | `sparse-moe` | `issue-21-sparse-moe` | `bounded_contract_harness` | `unavailable_hardware_runtime` |

> All outputs are deterministic pure-Rust CPU contract verifications, reference math, or explicit `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates.
