# Inference Lab

Reproducible experiments for model inference contracts, backend qualification, numerical parity, observability correlation, power/cadence measurement, and tokenizer/template conformance.

The core library is pure, deterministic, dependency-free Rust (`unsafe_code = "forbid"`). The pinned Nix flake locks Rust 1.85.1 and runs formatting, Clippy, unit/integration tests, and doctests offline.

> [!IMPORTANT]
> Every harness currently in this repository runs as a deterministic `bounded_contract_harness` on CPU. Physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, AWS F2 FPGA, and 27B serving runs are explicitly recorded as `unavailable_hardware_runtime` or `blocked_on_artifacts`. No physical accelerator throughput, latency, or model-quality result is claimed here.

## Quick start

```sh
nix develop --no-update-lock-file --option builders ''
cargo test --locked --all-targets
mkdir -p runs
cargo run --locked -- benchmark --out runs/seed-baseline
```

All CLI commands require a new `--out` directory whose parent already exists. Each run writes `inputs.json`, `results.json`, `summary.md`, and finally an empty `COMPLETE` marker. An absent `COMPLETE` marker means publication is incomplete; its presence indicates publication completed, not that physical hardware ran or every task succeeded.

## Implemented harnesses (Bootstrap + Batch 1)

| Command | Tracked issue(s) | Execution class | Current bounded scope |
| --- | --- | --- | --- |
| `benchmark` | Bootstrap seed | `bounded_contract_harness` | Deterministic serving/KV-budget contract comparing three admission profiles with non-collapsing `accepted`, `invalid`, `oom`, `timeout`, `unsupported` outcomes |
| `diffusion` | [#1](https://github.com/OnixResearch/inference-lab/issues/1) | `bounded_contract_harness` | Versioned Noble candidate/repair contract, sampler capability probes, EBM proposal/scoring/selection/admission fixture, offline pool replay, matched budgets, and default-off `DependentWeave`/JSONL/socketteer interchange check |
| `trace` | [#40](https://github.com/OnixResearch/inference-lab/issues/40) | `bounded_contract_harness` | Host/GPU trace correlation schema, clock-domain mapping enforcement, batched-kernel non-attribution, completeness/event-loss tracking, and 7 adversarial negative controls |
| `stage` | [#41](https://github.com/OnixResearch/inference-lab/issues/41) | `bounded_contract_harness` | Versioned transformer stage & tensor envelope contract, real pure-Rust 4-layer dense causal transformer (unsplit vs 2-stage vs 3-stage split, full & chunked prefill, forced-token decode, reset/reuse), poisoned-buffer/stale-KV guards, and zero silent CPU fallback for Apple GPU / Tenstorrent P150 |
| `power` | [#43](https://github.com/OnixResearch/inference-lab/issues/43) | `bounded_contract_harness` | Power & throttle cadence qualification profile, device-vs-host axis separation, sensor-gated energy reporting, observer perturbation, and known-equivalent / wrong-but-fast / order-drift controls |
| `tokenizer` | [#52](https://github.com/OnixResearch/inference-lab/issues/52) | `bounded_contract_harness` | Deterministic BPE + special-token reference tokenizer (inspired by Lean Verified Tokenizers), pinned chat-template renderer (`chatml-v1`, `qwen-thinking-v1`), role-injection defense, and disqualifying backend parity checker |
| `related-work` | [#56](https://github.com/OnixResearch/inference-lab/issues/56), [#58](https://github.com/OnixResearch/inference-lab/issues/58), [#60](https://github.com/OnixResearch/inference-lab/issues/60) | `blocked_on_artifacts` / `unavailable_hardware_runtime` | Default-off pinned gate manifests for Maple FPGA (`#56`), compositional-harness RLM-27B (`#58`), and Latent Lattice LMs (`#60`); all upstream claims retained strictly as unverified non-gating notes |

## Implemented harnesses (Batch 2: `#2`–`#11`)

| Command | Tracked issue(s) | Execution class | Current bounded scope |
| --- | --- | --- | --- |
| `proof-completion` | [#2](https://github.com/OnixResearch/inference-lab/issues/2) | `bounded_contract_harness` | Masked proof completion through Compiler Lab's frozen gate (`#75`/`#160`/`#218`/`#219`), frozen-context guards, task-family partition discipline, intake equivalence, matched budgets, and offline replay |
| `drafting` | [#3](https://github.com/OnixResearch/inference-lab/issues/3) | `bounded_contract_harness` | Greedy token-by-token conformance, exact analytical stochastic rejection-sampling verification (residual correction + bonus token), silent-fallback disqualification, and non-gating DFlash2/DNeo claim pins |
| `kernel-opt` | [#4](https://github.com/OnixResearch/inference-lab/issues/4) | `bounded_contract_harness` | Candidate-independent gate ordering (integrity -> FP64-vs-2x-BF16 + hidden prompts + batch-reorder bit identity + Cachet schedule budget -> soak -> local repeat noise floor), separate factors, and retained `runs/` log |
| `kv-routing` | [#5](https://github.com/OnixResearch/inference-lab/issues/5) | `bounded_contract_harness` | 13-dimension snapshot compatibility + generation fencing, RSI `#86` same-config/different-weights refusal, TT `#46` exact-vs-translated guard, soft-affinity load escape, RDMA vs recompute comparison, and ChaosControl transfer faults |
| `adaptive-spec` | [#6](https://github.com/OnixResearch/inference-lab/issues/6) | `bounded_contract_harness` | Reuses `#3` block verifier & `#7` workload identity for fixed vs validation-frozen adaptive lengths, CPU-overlap vs continuous-batching guard, `#22` rollback controls, and `sm_100a` extrapolation guard |
| `serving-bench` | [#7](https://github.com/OnixResearch/inference-lab/issues/7) | `bounded_contract_harness` | `tt-onix@5ed5e90c` (`LGPL-2.1-only`, `onix.qwen38-benchmark.v1`) compatibility contract, offline reducer & tamper verifier, branch-aware agent trace replay (6 controls), Eunomia transport controls, and OASYS RLM trace profile (5 controls) |
| `deepseek-spark` | [#8](https://github.com/OnixResearch/inference-lab/issues/8) | `bounded_contract_harness` | `dsv41-1017@7f466a12` migration manifest, repaired Prometheus counter accounting, burst vs sustained separation, observable cache states, 9 device-free fixtures, and legacy substring vs independent semantic quality checks |
| `learned-compiler` | [#9](https://github.com/OnixResearch/inference-lab/issues/9) | `blocked_on_artifacts` | Explicit asset-recovery inventory for unlocated `research/learned-compiler` & `acceptance.json` at `compiler-lab@50dcbdce`, preserved `#30`/`#31` baselines, Darwin/AppleContainer isolation gate, and model-free controls |
| `structured-read` | [#10](https://github.com/OnixResearch/inference-lab/issues/10) | `blocked_on_artifacts` | Conditional dependency gate (rejecting same-adapter URLs, mocks-only, chat-only models), AR/diffusion/Clef capability separation, explicit normalization scope, calibration invalidation, 3 operation-separated positive fixtures, and 10 adversarial negative fixtures |
| `lifecycle` | [#11](https://github.com/OnixResearch/inference-lab/issues/11) | `bounded_contract_harness` | Disposable Spark NixOS adapter contract (`models.nix@1abf9f9e`), `A -> B -> A` & RSI `#86` `parent -> variant -> parent`, Systems `#10` raw vs exclusive elapsed-time partition, unified-memory provenance/overlap guards, 13 lifecycle/Animus `#17` controls, and residency-policy comparison |
| `batch2-suite` | `#2`–`#11` | `bounded_contract_harness` + `blocked_on_artifacts` | Runs all 10 Batch 2 suites into a fresh directory with per-suite subdirs, aggregate reports, SHA-256 `manifest.json`, and empty `COMPLETE` written last |

## Implemented harnesses (Batch 3: `#12`–`#21`)

| Command | Tracked issue(s) | Execution class | Current bounded scope |
| --- | --- | --- | --- |
| `rsi-bench` | [#12](https://github.com/OnixResearch/inference-lab/issues/12) | `bounded_contract_harness` | Standalone RSI benchmark consolidation manifest (`rsi-lab@01e47e26`), RSI `#82` kernel-sweep repair gate, pure-Rust RWKV-7 recurrence invariance fixture, encoder phase separation, fake-quant non-speedup guard, and immutable closed `#18` receipts |
| `djev-protocol` | [#13](https://github.com/OnixResearch/inference-lab/issues/13) | `bounded_contract_harness` | Dual-consumer `djev.systemone.v1` protocol split, canonical UTF-8/non-ASCII wire-byte serializer & span-offset guard, closed RSI `#42` effective cache-identity regressions, scoring-vs-generative/proof separation, and whole-decision deadline controls |
| `rwkv-leaf` | [#14](https://github.com/OnixResearch/inference-lab/issues/14) | `blocked_on_artifacts` | Deferred conditional extraction gate (requires two concrete non-mock native-RWKV consumers), pure-Rust byte-prefix trie tokenizer, structural safetensors admission validator, zero-reallocation reusable-workspace recurrent step, and FP32-CPU vs FLA-BF16-CUDA guard |
| `ggml-ops` | [#15](https://github.com/OnixResearch/inference-lab/issues/15) | `bounded_contract_harness` | Pinned `ggml@417a024b` `test-backend-ops` 4-axis qualification (`advertised_support`, `executed_success`, `numerical_pass`, `performance_measured`), pure-Rust `F32`/`Q8_0`/`Q4_0` dot reference, explicit `MXFP4`/`NVFP4` unsupported status, silent-fallback disqualification, and `grants_placement_authority=false` |
| `rwkv-ggml` | [#16](https://github.com/OnixResearch/inference-lab/issues/16) | `bounded_contract_harness` | Independent FP32 ggml-style `wkv7` vs FP64 native RSI recurrence across token/chunked parity, 64-step drift checkpoints, binary state export/import round-trips with SHA-256 receipts, negative controls, and `asserts_agent_persistence_correctness=false` |
| `graphical-sampling` | [#17](https://github.com/OnixResearch/inference-lab/issues/17) | `bounded_contract_harness` | Compiler Lab `#222` (`fd0bb432`) exact joint enumeration oracle (`<= 256` states), sequential vs serial-color vs color-phase CPU Gibbs, deterministic per-(sweep, node) RNG trajectory parity, honest IPS ESS (`None` on stuck chains), 8 controls, and non-gating AMARA/default-off CRAM separation |
| `qwen38-quant` | [#18](https://github.com/OnixResearch/inference-lab/issues/18) | `bounded_contract_harness` | Reuses `#7` & `#3` with exact `64 KiB/token` calculated KV planning, offline rescoring (`zero accepted -> cost=None`), 12 conformance fixtures, MACaQ (`3810efc`), QSA (`#31`/`#32`), P150x2 concurrency (`tt-stack@e55c6527`), default-off Monoglot pruning, `blocked_on_artifacts` ternary `ttf`, QDA Spearman KL rank correlation, and out-of-scope `vegaml` (`rsi-lab#151`) |
| `deepseek-ref` | [#19](https://github.com/OnixResearch/inference-lab/issues/19) | `bounded_contract_harness` | Corrected pinned DeepSeek-style reference (`6901be60`, MIT) with pure-Rust TINY MLA + MoE + MTP, exact Eq. 18 (`0.017` vs buggy `0.003`), `n_mtp=0`, all-dense, decoupled eval/log intervals, exact-fit `get_batch` & short-data guard, gradient/mask/NaN/checkpoint regressions |
| `mla-comparison` | [#20](https://github.com/OnixResearch/inference-lab/issues/20) | `bounded_contract_harness` | Reuses `#19` across uncached, explicit-expansion, and absorbed MLA (`0` materialized per-head K/V intermediates), `batch=2` irregular chunks, `1e6` sentinel poisoning, 144-vs-480 element accounting (non-speedup), prefix/recompilation sweeps, gradient parity, and 7 controls |
| `sparse-moe` | [#21](https://github.com/OnixResearch/inference-lab/issues/21) | `bounded_contract_harness` | Reuses `#19` across dense, independent tiny scalar, and grouped sparse MoE dispatch with full analytical backward pass + FD & 3-step update parity, BF16-underflow-immune load counts, non-dropping capacity accounting, pair sweep, and 7 negative controls |
| `batch3-suite` | `#12`–`#21` | `bounded_contract_harness` + `blocked_on_artifacts` | Runs all 10 Batch 3 suites into a fresh directory with per-suite subdirs, aggregate reports, SHA-256 `manifest.json`, and empty `COMPLETE` written last |

## Ownership boundaries

| Owner | Responsibility retained there |
| --- | --- |
| [Noble](https://github.com/OnixResearch/noble-lang) | Structured editor AST, deterministic CEGIS baseline (`cegis.mjs`), proofs, and kernel acceptance |
| [Compiler Lab](https://github.com/brittonr/compiler-lab) | Native candidate checks, source-proof intake, compiler-input admission, and RTL/Verilog admission |
| [RSI Lab](https://github.com/OnixResearch/rsi-lab) | Recurrent-state learning, retention, promotion, and exact-controller protocols |
| [TT-Onix](https://github.com/OnixResearch/tt-onix) | Physical Tenstorrent worker execution, device telemetry, and Linux/macOS host gates |
| [Systems Lab](https://github.com/OnixResearch/systems-lab) | Cross-system composition, observer-overhead comparison, and shared-serving queue experiments |
| [Kiln](https://github.com/OnixResearch/kiln) / [Lattice](https://github.com/OnixResearch/lattice) / [Animus](https://github.com/OnixResearch/animus) | Global work admission, workflow bindings, campaign exploration, and production integration |
| Inference Lab | Model/backend adapters, sampler/stage/trace/power/tokenizer qualification harnesses, and comparative evidence |

## Read next

- [Architecture and boundaries](docs/architecture.md)
- [Experiment protocol and non-collapsing outcomes](docs/experiment-protocol.md)
- [Pinned source review](docs/source-review.md)
- [Research roadmap](docs/roadmap.md)
- [Pinned related-work notes (#56, #58, #60)](docs/related-work/README.md)
- [Bootstrap validation evidence](evidence/bootstrap/README.md)
- [Batch 1 contract harness evidence](evidence/batch1/README.md)
- [Batch 2 contract harness evidence (`#2`–`#11`)](evidence/batch2/README.md)
- [Batch 3 contract & reference-math evidence (`#12`–`#21`)](evidence/batch3/README.md)
- [OpenSpec change record (Batch 1)](openspec/changes/bootstrap-and-batch1-contracts/proposal.md)
- [OpenSpec change record (Batch 2)](openspec/changes/batch2-drafting-serving-lifecycle/proposal.md)
- [OpenSpec change record (Batch 3)](openspec/changes/batch3-rwkv-ggml-mla-moe-quant/proposal.md)
- [Contribution rules](AGENTS.md)
