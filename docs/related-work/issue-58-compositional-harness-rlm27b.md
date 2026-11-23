# Issue #58 — Zhang, Chakraborty, and Khattab's compositional-harness results and pinned RLM-27B

- **Tracked issue**: [OnixResearch/inference-lab#58](https://github.com/OnixResearch/inference-lab/issues/58)
- **Status**: `blocked_on_license_and_hardware_gate` (undeclared Hugging Face checkpoint/dataset licenses AND no local 27B serving host)
- **Activation**: `default_off = true` (explicit opt-in only; does not alter default lab serving, training, or routing)
- **Verification status**: **Unverified author claims only.** No 27B serve, harness rollout, or benchmark score was executed here.

## Exact pins observed (20 Nov 2026, 4:30 PM America/New_York)

| Artifact | Location | Exact revision / pin | License status |
| --- | --- | --- | --- |
| Paper | `https://www.alphaxiv.org/abs/2611.harness-cg` | `2611.harness-cg` (20 Nov 2026) | Paper |
| Earlier blog | `https://alexzhang13.github.io/blog/2026/harness/` | 20 Jul 2026 | Blog post |
| Thread | `https://x.com/a1zhang/status/2123774098241147225` (length note `...4107833499680`) | 20 Nov 2026 | Social post |
| Reproduction code | `alexzhang13/harness-cg` (`src/rlm_train/env.py`, `training/README.md`, `training/prime_rl/install.sh`) | `5272afdcb25a156c1b9b89cb479ae0604af249d7` | MIT |
| Training backend pin | `PrimeIntellect-ai/prime-rl` | `eaf95997f7cdbf85681075a61dd5bbfd2ba0231c` | Upstream pin (training reproduction is a non-goal) |
| Base model | `Qwen/Qwen3.8-27B` | Upstream base | Apache-2.0 |
| Fine-tune checkpoint | `a1zhang/RLM-27B` | `f0b3b27db57c211234cb55ec512d87f45dabdc4c` | **UNDECLARED (active gate — do not assume Apache-2.0 inheritance)** |
| Tail-recurse checkpoint | `a1zhang/RLM-27B-TailRecurse` | `bbab801cbe871aefe3f898093e461c807381dcc2` | **UNDECLARED (active gate)** |
| Tail-recurse base variant | `a1zhang/Qwen3.8-27B-TailRecurse` | `c57b846de3aba4287df68c1cf42ae10bc83ce8a0` | **UNDECLARED (active gate)** |
| Synthetic SFT traces | `a1zhang/synthetic-rlm-sft-data` | `9a31220cf40c1301eebde9a4282d1360b2d233f5` | **UNDECLARED (active gate; "no real benchmark data" claim unchecked)** |
| Tail-recursion SFT data | `a1zhang/rlm-tail-recursion-sft-data` | `a46f1f6aa02b88a2580a15e086af4a96c3a0c5af` | MIT |
| Experiment logs | `a1zhang/harness_compgen_exp_logs` | `1204c5cc7f05ddf6d1b5afe8536d2e30a56782a4` | **UNDECLARED (active gate)** |

## Unverified author claims (non-gating)

All figures below are author-reported and **must not** be used as lab thresholds or gates:

- RL on short tasks in programmatic neurosymbolic harnesses (`RLM`, `LM + REPL` with `enable_sub_llm=False`, Pi wrapper) allegedly generalizes to 2–64× longer held-out tasks and transfers across domains sharing decomposition structure.
- Surrogate harness training (single-call synthetic stand-ins) followed by full-parameter SFT of `Qwen3.8-27B` (one epoch, 8×H100) into `RLM-27B` allegedly lifts equal-weight mean across eight held-out long-context benches (including CTC-Bench and ExpertLongBench) from `46.1` to `58.5` (`26.9%` relative gain), with `~2×` fewer tokens and fewer root turns (`18.4` vs `6.5`).

## Gate conditions and non-goals

1. **License gate**: confirm serving and redistribution terms for `RLM-27B` (and any optional TailRecurse/trace artifacts) before downloading beyond card inspection.
2. **Hardware gate**: requires a declared 27B serving host (coordinated with `#18` and `#7`). Without that host, retaining this gate record and stopping is a complete result.
3. **Evaluation hygiene if unblocked**: freeze task identities, independent checkers (keeping gold labels and hidden tests out of model inputs), turn/step/token budgets, sampling/thinking settings, and report root turns, sub-call turns, tokens, wall time, failures, timeouts, and truncations. Exclude or separate tasks overlapping synthetic training families once checkable.
4. **Non-goals**: reproducing `prime-rl` training, 8×H100 SFT, or synthetic datagen; importing CTC-Bench/ExpertLongBench corpora or weights into this repo; taking over `#49` (Pi/OpenCode LoRA cross), `#18`, `#28`, or `#31`.

## Adjacent related-work note (distinct paper — does NOT add a task here)

- George Xenos and Sotiris Giannopoulos, *Programming in the Machine Code Space* (`https://sgiannopoulos.github.io/blog/machine-code-programming/`, pinned site repo `sgiannopoulos/sgiannopoulos.github.io@22dc86b4b31529fd082b1cd8c5c4b591edd8c0f8`, undeclared license; source post `https://x.com/AcousimHss/status/2124149496338862090`).
- Reports unverified one-shot hex-dump token costs (`~2.4 tokens/byte`, `~163,000` tokens for 67,392-byte `ministat`) and cites the earlier Recursive Language Models paper (`arXiv:2512.24601`, Zhang, Kraska, Khattab, 2025) as a possible direction — distinct from the Nov 2026 `RLM-27B` compositional-harness paper.
- The default-off benign-ELF check (explicitly excluding malware experiments) lives in `brittonr/compiler-lab#275`.
