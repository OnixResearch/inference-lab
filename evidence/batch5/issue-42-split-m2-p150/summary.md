# Issue #42: M2 Ultra + Tenstorrent P150 Split-Model Qualification

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Mac Pro Apple M2 Ultra + internal PCIe Tenstorrent P150 / P150x2)
- Dense split parity (reusing `#41` over 4 checks): `max_err = 0.000e0`; Disjoint MoE split: `max_err = 2.776e-17`
- Derived logical payload (`hidden=8192`, `2B/elem`, NOT measured throughput): `16384 B/tok` (`16 KiB`), `67108864 B / 4096 rows` (`64 MiB`)

## Independent Profile Verdicts
- `m2-ultra-p150x1-dense-linux-asahi` (`mac_pro_internal_pcie_linux_asahi`, `P150x1`, `moe=false`): `UNAVAILABLE_HARDWARE_RUNTIME` — blocked on TT-Onix #48 (Asahi/ARM Linux PCIe host gate) and #50 (contiguous TT partition)
- `m2-ultra-p150x1-dense-macos-metal` (`mac_pro_internal_pcie_macos_metal`, `P150x1`, `moe=false`): `UNAVAILABLE_HARDWARE_RUNTIME` — blocked on TT-Onix #49 (Darwin DriverKit/native-kernel gate)
- `m2-ultra-p150x2-dense` (`mac_pro_internal_pcie_linux_asahi`, `P150x2`, `moe=false`): `UNAVAILABLE_HARDWARE_RUNTIME` — independent second-card qualification gated on P150x1 correctness + observed TT links
- `m2-ultra-p150x1-moe-static-placement` (`mac_pro_internal_pcie_linux_asahi`, `P150x1`, `moe=true`): `UNAVAILABLE_HARDWARE_RUNTIME` — independent MoE checkpoint qualification gated on TT-Onix #51

## Negative Controls (8 / 8 rejected)
- `nc1_network_prototype_not_internal_pcie` -> `invalid` (reference/simulator/network/conventional-Linux prototype cannot substitute for internal-PCIe Mac Pro)
- `nc2_unified_plus_gddr_not_coherent_pool` -> `invalid` (Apple unified memory and P150 GDDR are distributed capacity, not one coherent pool)
- `nc3_performance_before_numerical_acceptance` -> `invalid` (numerical and continuation acceptance must precede any performance claim)
- `nc4_quant_translation_not_exact_parity` -> `invalid` (changed quantization across stage boundary is numerical translation, not exact parity)
- `nc5_dense_pass_not_borrowed_for_moe` -> `invalid` (MoE split requires independent checkpoint and routing qualification; cannot borrow dense pass)
- `nc6_p150x1_not_inherited_by_p150x2` -> `invalid` (P150x2 requires independent link/parallelism qualification; cannot inherit P150x1 pass)
- `nc7_overlapping_moe_expert_assignment` -> `invalid` (duplicate expert 2 across Apple and P150 rejected)
- `nc8_incomplete_moe_expert_partition` -> `invalid` (missing expert 5 in split partition rejected)
