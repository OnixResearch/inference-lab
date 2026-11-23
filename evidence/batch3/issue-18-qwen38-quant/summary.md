# Issue #18: Qwen3.8-27B Quantization & Serving Evaluation Profile

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_128gb_spark_strix_or_p150x2_qwen38_27b_serving`)
- Calculated KV planning (non-observed): `65536 B/tok` (`2147483648 B` @ 32768, `17179869184 B` @ 262144)
- QDA KL-vs-accepted-task Spearman `rho`: aligned=`1.0`, inverted negative control=`-1.0`
- Scoped decision: `INCONCLUSIVE_PENDING_PHYSICAL_27B_SAME_NODE_RUN`

## Offline Arm Scoring (Zero Model Calls)
| Arm | Planned | Accepted | Cost/Accepted | Exact Denom |
| --- | ---: | ---: | ---: | --- |
| `qwen38-27b-q8-approximate-anchor` | 6 | 4 | `23.75` | `true` |
| `qwen38-27b-broken-quant-zero-accepted` | 3 | 0 | `null` | `true` |

## Conformance Fixtures
- `01-positive-leading-system-developer-nested-tool-history` (`template_and_tool_conformance`): `accepted` — Leading system/developer role, nested JSON tool arguments, and assistant/tool history replay bound to exact chat_template.jinja digest.
- `02-negative-unsupported-reasoning-effort-value` (`sampling_and_reasoning`): `unsupported` — Unsupported reasoning_effort value rejected explicitly rather than silently ignored.
- `03-negative-missing-usage-tokens-unknown-not-zero` (`telemetry_honesty`): `invalid` — Absent prompt/output/reasoning usage counters stay None/unknown, never coerced to 0.
- `04-negative-stale-quant-digest-ud-q3-k-xl-replaced` (`artifact_identity`): `invalid` — Quant name alone is insufficient: commit 4af51b3542f775b0a8e4b310bfb3f17e01c306c9 replaced UD-Q3_K_XL; exact file SHA-256 mismatch rejected.
- `05-negative-qwen-vs-unsloth-chat-template-mismatch` (`template_identity`): `invalid` — Qwen upstream and Unsloth chat_template.jinja differ; unverified template substitution fails conformance.
- `06-negative-silent-fallback-or-incomplete-population` (`population_and_backend`): `invalid` — Silent backend fallback and dropped timeout/truncated attempts fail qualification.
- `07-negative-preserve-thinking-leaked-into-animus-canonical` (`animus_boundary`): `invalid` — Provider preserve_thinking output stays isolated in raw adapter boundary; never injected into Animus canonical messages or verdicts.
- `08-negative-p150x2-cross-request-state-contamination` (`tenstorrent_concurrency`): `invalid` — Concurrent P150x2 slot reuse leaking prompt/history state fails qualification regardless of throughput; generic Blackhole/P300/4-chip cannot qualify P150x2.
- `09-positive-monoglot-bit-identical-non-vocab-qualification` (`monoglot_prune_default_off`): `accepted` — Default-off Monoglot prune passes pre-scoring gate when non-vocab tensors are bit-identical, English/code greedy parity passes, and multilingual inflation limit is declared.
- `10-negative-monoglot-misaligned-head-rejected-before-scoring` (`monoglot_prune_default_off`): `invalid` — Misaligned vocabulary-pruned head fails pre-scoring qualification before task scoring.
- `11-negative-macaq-r-dense-27b-unsupported-see-issue-24` (`macaq_extension`): `unsupported` — Native MACaQ-R refinement targets MoE projections (#24); dense Qwen3.8-27B is unsupported by that command.
- `12-control-tt-metal-078-smoke-admission-eligible-false-preserved` (`tenstorrent_scope`): `unsupported` — tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a TT-Metal 0.78 smoke (2 short greedy calls @ 2048 tokens) preserves admission_eligible=false.

> No physical 27B serving run occurred locally. Upstream Unsloth, MACaQ, Monoglot, `ttf`, QDA, `tt-stack`, and `vegaml` figures remain strictly unverified and non-gating.
