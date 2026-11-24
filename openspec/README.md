# OpenSpec

Behavior changes and new experiment harnesses are specified first as a proposal, design, tasks, and observable requirement specs under `changes/`.

Active changes:
- [bootstrap-and-batch1-contracts](changes/bootstrap-and-batch1-contracts/proposal.md): repository bootstrap and Batch 1 (`#1`, `#40`, `#41`, `#43`, `#52`, `#56`, `#58`, `#60`).
- [batch2-drafting-serving-lifecycle](changes/batch2-drafting-serving-lifecycle/proposal.md): Batch 2 (`#2`–`#11`) drafting, serving, kernel-opt, KV-routing, recovery, structured-read, and lifecycle contracts.
- [batch3-rwkv-ggml-mla-moe-quant](changes/batch3-rwkv-ggml-mla-moe-quant/proposal.md): Batch 3 (`#12`–`#21`) RWKV, `djev`, `ggml`, graphical sampling, Qwen3.8 quant, DeepSeek ref, MLA, and sparse MoE contracts.
- [batch4-mtp-sparse-quant-placement-qsa](changes/batch4-mtp-sparse-quant-placement-qsa/proposal.md): Batch 4 (`#22`–`#31`) MTP decoder, FlashTopK, MACaQ-R, heterogeneous rollouts, NVFP4/MXFP8, Spark engines, KV compression, expert placement, vision in-fusion, and Qwen3.8 QSA contracts.
- [batch5-qsa-recurrent-tiered-split-interventions](changes/batch5-qsa-recurrent-tiered-split-interventions/proposal.md): Batch 5 (`#32`–`#39`, `#42`, `#44`–`#46`) QSA state coherence, Inkling residuals, Phoenix nano, Nemotron Spark, compact hybrids, native recurrent bench, tiered eval, CPU contention, M2+P150 split, Quark eval, intervention replay, and BioLLM protocol contracts.
- [batch6-speculative-nvme-hillclimb-tufazip-clairic-mini-megakernels](changes/batch6-speculative-nvme-hillclimb-tufazip-clairic-mini-megakernels/proposal.md): Batch 6 (`#47`–`#51`, `#53`–`#55`, `#57`, `#59`, `#61`) adversarial speculative reasoning, NVMe coalescing, harness LoRA comparison, latent feedback, SimpleHC vs mHC, serving hill-climb, tufazip online codec, Clairic extraction, Inkling 2 Mini & maya-ik2 serve qualification, and persistent decode megakernels.
- [batch7-disagg-kv-linquant-varloop-ternary-uno-lfm3](changes/batch7-disagg-kv-linquant-varloop-ternary-uno-lfm3/proposal.md): Batch 7 (`#62`–`#73`) disaggregated 10GbE KV prefix transfer, 2-Spark RDMA TP & 4-bit attention, 2-Spark KV checkpointing, CopyDrafter, Hypertune soft-prefill KV adapters, Qwen3.8 Longbook all-GDN, LinQuant algebraic recurrent prefill, TML varloop + 2DSD equal-cost protocol, ttf-home ternary Qwen3.8, Draft-OPD/D-PACE replay, K2 Uno/MoVA, and LFM3/d1 typed decisions.
- [import-tt-onix-qwen38-benchmark](changes/import-tt-onix-qwen38-benchmark/proposal.md): Import `OnixResearch/tt-onix`'s portable `qwen38-benchmark` runner, frozen workloads, Nix compatibility package, and documentation (`#7`).
- [qwen38-benchmark-stream-deadlines](changes/qwen38-benchmark-stream-deadlines/proposal.md): Add bounded SSE event-gap tracking, explicit token-to-event cumulative deadline checks, failure-retaining population accounting, and frozen prefill-interference mode to `qwen38-benchmark` (`tt-onix#17`).
- [import-tt-onix-retention-eval](changes/import-tt-onix-retention-eval/proposal.md): Import `OnixResearch/tt-onix`'s portable six-passage full-vocabulary retention evaluator (`retention.py`, frozen fixtures, standalone CLI, and Nix package).
- [import-tt-onix-http-batch-group](changes/import-tt-onix-http-batch-group/proposal.md): Import `OnixResearch/tt-onix`'s portable HTTP barrier batch-group driver and frozen three-worker subagent review workload.

Task checkboxes are updated only after the corresponding implementation and validation have been observed.
