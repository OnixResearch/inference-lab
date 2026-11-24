# Issue #83: `apple-nvidia-sidecar` Heterogeneous Apple-Host + Discrete-NVIDIA Split

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Apple Mac Studio + discrete NVIDIA RTX 5090 eGPU link (pinned apple-nvidia-sidecar))
- Upstream pin: `joesinvestments/apple-nvidia-sidecar@c59a276a` (`MIT`, prefill err `0.000e0`, decode err `0.000e0`, reset err `0.000e0`)

## Link Bandwidth & Concurrency Scaling Model
| Link Profile | Eff Gbps | Batch | Payload (B) | Handoff (us) | Est Decode tok/s | Link Overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `thunderbolt4_32gbps_egpu` | `26.0` | 1 | 10240 | `62.3` | `213.6` | `0.0133` |
| `thunderbolt4_32gbps_egpu` | `26.0` | 2 | 20480 | `68.6` | `391.5` | `0.0134` |
| `thunderbolt4_32gbps_egpu` | `26.0` | 4 | 40960 | `81.2` | `671.0` | `0.0136` |
| `thunderbolt4_32gbps_egpu` | `26.0` | 8 | 81920 | `106.4` | `1043.5` | `0.0139` |
| `thunderbolt5_pcie4x4_64gbps_egpu` | `52.0` | 1 | 10240 | `39.2` | `214.6` | `0.0084` |
| `thunderbolt5_pcie4x4_64gbps_egpu` | `52.0` | 2 | 20480 | `42.3` | `393.5` | `0.0083` |
| `thunderbolt5_pcie4x4_64gbps_egpu` | `52.0` | 4 | 40960 | `48.6` | `674.7` | `0.0082` |
| `thunderbolt5_pcie4x4_64gbps_egpu` | `52.0` | 8 | 81920 | `61.2` | `1049.7` | `0.0080` |

## Negative Controls (8 / 8 rejected)
- `nc1_silent_cpu_fallback_rejected` -> `invalid` (silent CPU fallback when discrete NVIDIA eGPU is absent is refused)
- `nc2_missing_apple_or_nvidia_hw_gated` -> `unavailable_hardware_runtime` (absent Mac Studio + eGPU RTX 5090 link returns unavailable_hardware_runtime)
- `nc3_corrupted_handoff_payload_detected` -> `invalid` (bit-corrupted inter-stage activation tensor rejected by SHA-256 envelope)
- `nc4_handoff_schema_mismatch_rejected` -> `invalid` (unknown inter-stage envelope schema rejected)
- `nc5_nan_activation_handoff_rejected` -> `invalid` (NaN hidden state refused before wire serialization)
- `nc6_zero_link_bandwidth_rejected` -> `invalid` (zero Thunderbolt/PCIe bandwidth refused instead of Inf latency)
- `nc7_zero_batch_size_rejected` -> `invalid` (zero concurrency batch size refused)
- `nc8_unverified_sidecar_x_post_tps_non_gating` -> `invalid` (210 tok/s decode & 1,900 tok/s 8K prefill claims stay unverified related work)
