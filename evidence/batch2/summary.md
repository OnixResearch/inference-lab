# Batch 2 Contract Harness Suite (`#2`–`#11`)

- Total suites: 10 (`bounded_contract_harness`: 8, `blocked_on_artifacts`: 2)
- Physical GPU / Spark / Tenstorrent / RDMA / Darwin / live model executed: `false`

| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |
| ---: | --- | --- | --- | --- |
| `#2` | `proof-completion` | `issue-02-proof-completion` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#3` | `drafting` | `issue-03-drafting` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#4` | `kernel-opt` | `issue-04-kernel-opt` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#5` | `kv-routing` | `issue-05-kv-routing` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#6` | `adaptive-spec` | `issue-06-adaptive-spec` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#7` | `serving-bench` | `issue-07-serving-bench` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#8` | `deepseek-spark` | `issue-08-deepseek-spark` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#9` | `learned-compiler` | `issue-09-learned-compiler` | `blocked_on_artifacts` | `blocked_on_artifacts` |
| `#10` | `structured-read` | `issue-10-structured-read` | `blocked_on_artifacts` | `blocked_on_artifacts` |
| `#11` | `lifecycle` | `issue-11-lifecycle` | `bounded_contract_harness` | `unavailable_hardware_runtime` |

> All outputs are deterministic CPU contract verifications or explicit `blocked_on_artifacts` inventories. No external claims are imported as acceptance gates.
