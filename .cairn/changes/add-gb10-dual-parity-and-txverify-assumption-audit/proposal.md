# Why

Cross-vault evidence from `rsi-lab-mneme/10-wave-m-swarm-cuda-optimization-and-eat.adoc` (Wave M GB10 `sm_121` BF16 Tensor Core optimization) and `x-bookmarks-mneme` (`TxVerify` Blackwell PTX formal verification; `papers/arxiv-2611.26450.adoc` *A CPU Inside an LLM*) uncovered two real Blackwell/GB10 kernel hazards missing from `inference-lab`:

1. **Cosine-only parity blind spots:** Cosine similarity missed both scale-invariant `skip_l2_normalize_scale` (`cos = 1.0`) and truncation drift (`cos = 0.99994, max_abs_diff = 0.0404`), requiring a dual gate (`cos >= 0.9999 && max_abs_diff <= 1e-3`, forbidding NaN/Inf).
2. **Multi-context CUDA TLS drift & implicit sync assumptions:** Dual GPU handles on one thread caused `state_digest` mismatches and drop-order SIGSEGV without a `cuCtxSetCurrent` TLS guard, and Blackwell kernels need explicit core/assumption inventories (TxVerify pattern).
3. **Zero receipts in `onix-labs-mneme`:** `inference-lab` does not yet emit pinned `mneme.lab-run.v1` receipts.

# What Changes

- Extend `src/decode_megakernels.rs` (`MegakernelFaults`, `evaluate_batch_equivalence_and_timing`, `AgentKernelBuildEntry`) with `DualParityGate { min_cosine: 0.9999, max_abs_diff: 1e-3, forbid_non_finite: true }`, `SkipScaleNorm` & `MultiContextTlsDrift` fault-injection traps, and `KernelAssumptionManifest`.
- Update `src/deepseek_spark.rs` and `src/main.rs` to emit pinned `mneme.lab-run.v1` receipts consumable by `mneme lab ingest`.
