# Batch 8 Contract & Reference-Math Harness Suite (`#74`–`#84`)

- Total suites: 11 (`bounded_contract_harness`: 11, `blocked_on_artifacts`: 0)
- Physical Blackwell/DGX Spark / SmartNIC RDMA / Strix Halo + RX 7900 XTX / Dual R9700 / Mac Studio + RTX 5090 eGPU / custom silicon execution: `false`

| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |
| ---: | --- | --- | --- | --- |
| `#74` | `pdl-breakable-graphs` | `issue-74-pdl-breakable-graphs` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#75` | `tensorlean-kernel-verify` | `issue-75-tensorlean-kernel-verify` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#76` | `blink-cpu-free-serving` | `issue-76-blink-cpu-free-serving` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#77` | `qwen35-neuron-steering` | `issue-77-qwen35-neuron-steering` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#78` | `hnet-byte-modeling` | `issue-78-hnet-byte-modeling` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#79` | `tkcc-hillclimb-profile` | `issue-79-tkcc-hillclimb-profile` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#80` | `exl3-longctx-quant` | `issue-80-exl3-longctx-quant` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#81` | `saber-dflash2-power` | `issue-81-saber-dflash2-power` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#82` | `mimo-hysparse2-kv` | `issue-82-mimo-hysparse2-kv` | `bounded_contract_harness` | `blocked_on_artifacts` |
| `#83` | `apple-nvidia-sidecar` | `issue-83-apple-nvidia-sidecar` | `bounded_contract_harness` | `unavailable_hardware_runtime` |
| `#84` | `silicon-codesign-audit` | `issue-84-silicon-codesign-audit` | `bounded_contract_harness` | `blocked_on_artifacts` |

> All outputs are deterministic pure-Rust CPU contract verifications and numerical references (`default_enabled: false`). Physical accelerators, SmartNIC GPUDirect RDMA clusters, eGPU links, live model serves, fabricated silicon, and unlocated/unverified upstream artifacts remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates.
