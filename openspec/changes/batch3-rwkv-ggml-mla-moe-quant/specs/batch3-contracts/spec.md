# Specification: Batch 3 RWKV, djev, ggml, graphical sampling, Qwen3.8 quant, DeepSeek ref, MLA, and sparse MoE contracts

## Scenario 1: Batch 3 CLI publication, SHA-256 manifest, and backward preservation
- All 18 prior CLI subcommands, `evidence/{bootstrap,batch1,batch2}/`, `Cargo.toml`, `Cargo.lock`, `flake.nix`, `flake.lock`, and all 34 prior tests remain intact and passing.
- Each of the 10 new per-issue subcommands (`rsi-bench`, `djev-protocol`, `rwkv-leaf`, `ggml-ops`, `rwkv-ggml`, `graphical-sampling`, `qwen38-quant`, `deepseek-ref`, `mla-comparison`, `sparse-moe`) writes `inputs.json`, `results.json`, `summary.md`, and empty `COMPLETE` last into a fresh `--out NEW_DIRECTORY` and refuses overwrites.
- `batch3-suite --out NEW_DIRECTORY` publishes all 10 per-issue subdirectories (each ending with empty `COMPLETE`), top-level `inputs.json`, `results.json`, `summary.md`, `manifest.json` (with exact SHA-256 digests of every emitted file), and empty `COMPLETE` last.
- Two consecutive runs of `batch3-suite` produce byte-identical outputs across every emitted file matching retained `evidence/batch3/` with zero `/home/` paths.

## Scenario 2: Issue #12 — Standalone RSI inference benchmark consolidation
- Manifest enumerates all 5 selected donor surfaces at `OnixResearch/rsi-lab@01e47e268d71bc1c53cc36e11c179f2b8f22f5c5` and preserves RSI ownership of `packing_flip_replay.py`, stateful wrappers, trainer/simulator, and immutable closed `#18` receipts.
- Pure-Rust RWKV-7 recurrence invariance fixture passes token-by-token vs chunked output and final-state parity and catches altered transforms, perturbed final states, RSI `#82` unfixed sweep violations (RNG/allocation in timed region, missing state parity), encoder cold/warm/compile conflation, fake-quantization mislabeled as INT8 speedup, private log leaks, and overwriting `#18` receipts.

## Scenario 3: Issue #13 — Reusable djev protocol client split
- Dual-consumer gate requires RSI adapter plus named `#1` structured-read worker pinning the same `djev.systemone.v1` protocol revision.
- Canonical wire-byte serializer preserves deterministic key ordering, UTF-8 / non-ASCII input, ordered candidate labels, and span offset bounds, matching frozen golden-byte SHA-256 vectors.
- Closed RSI `#42` cache-identity regressions verify identical effective reads reuse labels while changing rendering, candidate order, template, samples, seed, tournament policy, canvas, served revision, or wire version invalidates reuse; absent immutable model digest yields `non_reusable_run_namespace`.
- Scoring capability remains distinct from generative masked completion and Noble proof acceptance; cancellation, late results, saturated queues, transport timeouts (never resetting whole-decision deadlines), malformed responses, and protocol mismatches are caught.

## Scenario 4: Issue #14 — Conditional native RWKV numerical leaf gate
- Reports `execution_class = blocked_on_artifacts` until two concrete non-mock consumers with named call sites demonstrate the leaf API (rejecting papers, planned benchmarks, and second mocks).
- Pure-Rust byte-prefix trie tokenizer matches pinned semantics and diverges from generic whitespace tokenization; structural safetensors admission validator enforces header bounds, overlap checks, dtype/shape bounds, digest match, and file-immutability contract.
- Reusable-workspace recurrent step verifies `workspace_allocations == 0` across multi-token stepping; FP32 CPU architectural behavior remains explicitly non-interchangeable with FLA BF16 CUDA policy parity; RSI wrapper tests preserve per-agent isolation, offset refusal, snapshot purity, and atomic failure rollback.

## Scenario 5: Issue #15 — Pinned ggml backend-ops qualification harness
- Pins `ggml-org/ggml@417a024b4fb6bc3116d41af7c067166a039bdf12` and keeps `advertised_support`, `executed_success`, `numerical_pass`, and `performance_measured` strictly independent.
- Pure-Rust bounded `F32`, `Q8_0`, and `Q4_0` reference kernels verify numerical parity; `MXFP4`/`NVFP4` remain explicitly `unsupported` when unavailable; incorrect outputs, non-finite values, and silent CPU fallback disqualify qualification; provider export sets `grants_placement_authority = false`.

## Scenario 6: Issue #16 — ggml RWKV-7 recurrence and state-continuation parity
- Compares two independent pure-Rust implementations (FP64 native reference vs FP32 ggml-style `wkv7`) across token-at-a-time vs chunked execution, 64-step long-horizon drift, and binary state export/import round-trips with SHA-256 receipts.
- Rejects altered exported state, sequence-boundary shifts, excessive precision drift, unsupported backward/training claims, and silent CPU fallback for `ggml-cuda`, while asserting `asserts_agent_persistence_correctness = false`.

## Scenario 7: Issue #17 — Discrete graphical sampling with exact oracle and honest ESS
- Exact enumeration oracle (`<= 256` joint states) computes exact normalization constant, marginals, conditionals, and expected energy on coupled and irregular positive-factor models.
- Sequential Gibbs, serial color-schedule Gibbs, and color-phase Gibbs execute on CPU; matched schedule + per-(sweep, node) RNG stream mapping yields bit-identical trajectories; IPS ESS (`ess-ips-v1`) returns `None` for stuck/degenerate chains.
- Catches stale neighbor cache, invalid 1-coloring on coupled nodes, omitted commits, swapped factor indexing, correlated/reused RNG streams, invalid normalization, incomplete chains, and silent hardware fallback; AMARA and default-off CRAM remain strictly non-gating.

## Scenario 8: Issue #18 — Qwen3.8-27B quantization and serving evaluation profile
- Reuses `#7` metric semantics and `#3` speculative/MTP rules, verifies exact `64 KiB/token` calculated KV planning (`2 GiB` @ 32k, `16 GiB` @ 256k, labeled non-observed), blocks overlapping unified-memory summation, separates matched-target parity from changed-quant task quality, and preserves `cost_per_accepted_task = None` when zero tasks are accepted.
- Runs positive and negative conformance fixtures (system/developer messages, nested tools, history replay, unsupported reasoning, missing usage, malformed/truncated outputs, stale quant/template digest, silent fallback, cross-request state contamination, misaligned Monoglot vocabulary pruning).
- Explicitly models pinned non-gating extensions: MACaQ (`3810efc`, creation vs serving cost, PR `#4`/`#5`, `max_num_seqs=1`), QSA (`#31`/`#32`), Tenstorrent P150x2 concurrency/board identity (`tt-stack@e55c6527`), default-off Monoglot pruning, `blocked_on_artifacts` ternary `ttf`, QDA checkpoints with Spearman KL-vs-accepted-task rank correlation (aligned + inverted control), and out-of-scope `vegaml` (`rsi-lab#151`).

## Scenario 9: Issue #19 — Corrected pinned DeepSeek-style reference
- Pins `Vaibhaav-Tiwari/deepseek-v3-jax@6901be60941b98dc654ed542465290db9e8bafa6` (MIT notice) and implements pure-Rust TINY MLA + routed/shared MoE + MTP with explicit FP64/BF16 precision policy.
- Reproduces and verifies fixes for all 5 upstream defects: Eq. 18 raw-affinity sequence loss vs bias-adjusted dispatch through the full `moe()` caller on the exact `logit([0.9, 0.8, 0.2, 0.1])` fixture, `n_mtp = 0` MTP-off, `n_dense == n_layers` all-dense, independent eval/log intervals, and inclusive `get_batch` final window + short-data error.
- Verifies full vs chunked vs cached decode parity, finite-difference gradients, non-finite rejection, negative controls (wrong mask, altered gate weights, shifted MTP target), and checkpoint round-trip.

## Scenario 10: Issue #20 — Explicit vs absorbed MLA comparison
- Reuses `#19`'s frozen parameters across full uncached baseline, compressed-cache explicit expansion, and absorbed MLA, verifying parity across prefill, single-token decode, multi-token chunks, `batch > 1`, and multiple heads.
- Proves absorbed MLA constructs 0 per-head K/V expansion intermediates, verifies unused-slot `1e6` sentinel poisoning causes zero drift, tracks full-allocated vs valid-prefix vs bucketed-prefix recompilations, distinguishes 144-vs-480 persistent elements from speedup, verifies gradient parity, and catches 7 negative controls with first divergent position.

## Scenario 11: Issue #21 — Sparse MoE dispatch vs dense reference
- Reuses `#19`'s parameters across dense reference, independent tiny scalar reference, and sparse MoE dispatch with deterministic tie-breaking, bias-selection-only / raw-affinity-weighting / single `route_scale` / single shared-expert semantics, non-dropping capacity accounting, and assignment-derived load counts immune to BF16 gate underflow.
- Verifies forward parity, full analytical backward pass (`dX`, `dW_router`, `dW_shared`, `dW_experts`) parity against dense and finite differences away from ties, declared tie behavior at discontinuities, 3-step matched update parity (including bias updates), actual-pair execution sweep, and all 7 adversarial negative controls.
