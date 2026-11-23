# Issue #31: Qwen3.8-27B QSA vs Matched Dense Control Contract

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_nvidia_q8_0_qwen38_27b_qsa_vs_dense_llamacpp`)
- Decision: `INCONCLUSIVE_PHYSICAL_27B_Q8_0_PAIRED_SERVE_REQUIRED`
- Population (planned=5): dense_accepted=4, qsa_accepted=3, sparse_induced_losses=1, both_failed_not_retained_capability=1

## Negative controls
- `llama_qsa_dense_flag_ignored_detected` -> `invalid` (selected_tokens=16 != context_tokens=64)
- `plausible_decoy_and_broken_multifile_edit_rejected` -> `invalid` (decoy=invalid, bad_edit=invalid)
- `incomplete_population_rejected` -> `invalid` (incomplete or empty population: observed 4 vs expected 5)
- `x86_sm70_binary_does_not_qualify_spark_arm64_gb10` -> `unsupported` (prebuilt `x86_64_sm70` binary does not qualify `aarch64_sm121_gb10` target)
- `issue_23_flashtopk_not_dropin_for_block_qsa` -> `unsupported` (released #23 FlashTopK@090902ab contract does not support 4-token block-level QSA)
- `qsa_retains_full_target_kv_plus_indexer_buffers` -> `invalid` (qsa_kv_plus_indexer=2368 bytes > dense_kv=2048 bytes (QSA does not reduce target KV capacity))
- `bf16_ppl_cannot_substitute_for_native_task_accuracy` -> `invalid` (BF16 reference/PPL or attention-mass capture cannot substitute for native quantized task accuracy)
- `retrained_indexer_with_changed_backbone_mislabeled` -> `invalid` (retrained-indexer arm with changed backbone/quant/runtime must be labeled whole-configuration comparison)
