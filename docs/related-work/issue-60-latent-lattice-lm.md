# Issue #60 — Duggal et al. Latent Lattice Language Models (LLLMs)

- **Tracked issue**: [OnixResearch/inference-lab#60](https://github.com/OnixResearch/inference-lab/issues/60)
- **Status**: `blocked_on_artifacts` (no public training/inference code, no checkpoints, no license)
- **Activation**: `default_off = true` (explicit opt-in only; does not alter default lab serving, training, or routing)
- **Verification status**: **Unverified author claims only.** No model build, forward pass, or evaluation was executed here.

## Exact pins observed (18 Nov 2026)

| Artifact | Location | Exact revision / pin | Observed content | Runnable pin? |
| --- | --- | --- | --- | --- |
| Paper | `https://arxiv.org/abs/2611.21608v1` | `arXiv:2611.21608v1` | Shivam Duggal, Zongze Wu, William T. Freeman, Antonio Torralba, Richard Zhang, Eli Shechtman | Paper only |
| Thread | `https://x.com/ShivamDuggal4/status/2123110506450092217` (links `...3138288773189891`) | 18 Nov 2026 | Author summary thread | Social post only |
| Project page | `https://shivamduggal4.github.io/latent-lattice-language-models/` | Static site | HTML/media | Website only |
| Project-page repository | `ShivamDuggal4/latent-lattice-language-models` | `23876839517155db4333fb08d26b8b118a3db2e1` (18 Nov 2026, 2:57 PM ET) | HTML, fonts, images, and videos only; **no training/inference code, no checkpoints, no license file** | **No — must not be treated as an implementation** |

## Unverified author claims (non-gating)

All figures below are author-reported for their tested setting (400M–600M parameters, 10B–30B FineWeb-Edu tokens, 2048 context, LLaMA-style decoder, GPT-NeoX tokenizer) and **must not** be imported as acceptance gates or targets:

- Adds per-layer horizontal recurrent latent streams ("Lattice of Latents") conditioning attention, KV-cache writes, MLP computation, and decoding.
- On 20B FineWeb-Edu tokens (406M params): claimed validation perplexity `14.40` vs `15.41` for parameter-matched Transformer; with hyper-connections + input residuals: claimed `13.61` vs `14.22` (enhanced Transformer) and `15.41` (base Transformer).
- On Gated DeltaNet, Mamba-2, Triadic GDN, and GDN-Mamba-2 hybrids: claimed improvements across native state sizes, with Lattice-GDN / Lattice-Triadic-GDN allegedly beating pure baselines with `4x` larger native states.
- Project page claims 22 layers vs 40-layer Transformer with same attention width (1024) yields a `45%` smaller KV cache, and optional 3-pass training-time Jacobi refinement costs `~2.6x` FLOPs while inference uses sequential rollout at base FLOPs.

## Gate conditions and strict disambiguation

1. **Artifact gate**: confirm public training/inference code (and checkpoints if an eval is requested) under an explicit license, pinning exact SHAs before running anything. Retaining this blocked-on-artifacts record and stopping is a complete result.
2. **Scope bound if unblocked**: never reproduce the 10B–30B FineWeb-Edu pretrain; run only a small frozen eval or short-sequence forward-pass smoke test if it fits declared hardware (recording inability to run as `unsupported`, never as refuting the authors' claims).
3. **Strict disambiguation**:
   - **Not `OnixResearch/lattice`**: completely unrelated to the Onix candidate-worker repository referenced in `#1`.
   - **Distinct from `#57`**: looped Inkling 2 Mini local serving remains separate.
   - **Distinct from `#51`**: SimpleHC vs mHC training-overhead comparison remains separate.
   - **Distinct from `#37`**: official RWKV-7, Mamba-3, and xLSTM recurrent-state benchmarks remain separate.
   - **Distinct from `#33`**: Inkling attention-residual ablations remain separate.
