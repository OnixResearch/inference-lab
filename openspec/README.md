# OpenSpec

Behavior changes and new experiment harnesses are specified first as a proposal, design, tasks, and observable requirement specs under `changes/`.

Active changes:
- [bootstrap-and-batch1-contracts](changes/bootstrap-and-batch1-contracts/proposal.md): repository bootstrap and Batch 1 (`#1`, `#40`, `#41`, `#43`, `#52`, `#56`, `#58`, `#60`).
- [batch2-drafting-serving-lifecycle](changes/batch2-drafting-serving-lifecycle/proposal.md): Batch 2 (`#2`–`#11`) drafting, serving, kernel-opt, KV-routing, recovery, structured-read, and lifecycle contracts.
- [batch3-rwkv-ggml-mla-moe-quant](changes/batch3-rwkv-ggml-mla-moe-quant/proposal.md): Batch 3 (`#12`–`#21`) RWKV, `djev`, `ggml`, graphical sampling, Qwen3.8 quant, DeepSeek ref, MLA, and sparse MoE contracts.
- [batch4-mtp-sparse-quant-placement-qsa](changes/batch4-mtp-sparse-quant-placement-qsa/proposal.md): Batch 4 (`#22`–`#31`) MTP decoder, FlashTopK, MACaQ-R, heterogeneous rollouts, NVFP4/MXFP8, Spark engines, KV compression, expert placement, vision in-fusion, and Qwen3.8 QSA contracts.
- [batch5-qsa-recurrent-tiered-split-interventions](changes/batch5-qsa-recurrent-tiered-split-interventions/proposal.md): Batch 5 (`#32`–`#39`, `#42`, `#44`–`#46`) QSA state coherence, Inkling residuals, Phoenix nano, Nemotron Spark, compact hybrids, native recurrent bench, tiered eval, CPU contention, M2+P150 split, Quark eval, intervention replay, and BioLLM protocol contracts.

Task checkboxes are updated only after the corresponding implementation and validation have been observed.
