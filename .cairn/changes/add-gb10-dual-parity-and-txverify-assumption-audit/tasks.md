# Tasks

## Phase 1 — Dual Parity Gate, Traps, & Assumption Manifest
- [ ] Add `DualParityGate`, `SkipScaleNorm` & `MultiContextTlsDrift` traps, and `KernelAssumptionManifest` in `src/decode_megakernels.rs`.

## Phase 2 — Pinned Receipts & Verification
- [ ] Emit pinned `mneme.lab-run.v1` receipts from `src/deepseek_spark.rs` and `src/main.rs`.
- [ ] Add regression tests proving dual-gate detection, multi-context A/A sensitivity, and receipt schema validity; pass `cargo test` and `cargo clippy --all-targets -- -D warnings`.
