# Design: GB10 Dual Parity Gate, TxVerify Assumption Audit, & Lab Receipts

## 1. Dual Parity Gate & Adversarial Fault Traps (`src/decode_megakernels.rs`)
- Add `DualParityGate` enforcing `cosine >= 0.9999`, `max_abs_diff <= 1e-3`, and `forbid_non_finite = true` across every megakernel comparison.
- Add `SkipScaleNorm` (passes cosine-only `== 1.0`, fails `max_abs_diff`) and `MultiContextTlsDrift` (interleaved dual-context state-digest corruption) fault-injection traps to prove gate sensitivity.

## 2. Explicit Kernel Assumption Manifest (TxVerify Pattern)
- Attach `KernelAssumptionManifest { sync_barriers, warp_layout, tls_context_guard, numeric_precision }` to `AgentKernelBuildEntry` so undocumented barrier/context assumptions fail audit.

## 3. Pinned `mneme.lab-run.v1` Receipts (`src/deepseek_spark.rs`, `src/main.rs`)
- Emit deterministic `inputs.json`/`results.json`/`summary.md` + `COMPLETE` receipts pinned to `OnixResearch/inference-lab` and exact commit SHA.

## Verification Plan
- `cargo test` confirming `SkipScaleNorm` passes cosine-only yet is caught by `DualParityGate`, `MultiContextTlsDrift` triggers A/A state-digest failure, and emitted receipts validate cleanly against `mneme.lab-run.v1`.
