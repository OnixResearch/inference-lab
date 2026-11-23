# Issue #52 — Tokenizer and chat-template conformance report

- **Execution class**: `bounded_contract_harness`
- **Reference tokenizer**: `reference-bpe-verified-inspired-v1` (`sha256:8038b4253e7d750b`)
- **Live backend gate**: `unavailable_hardware_runtime`

| Backend | Case | Parity passed | Verdict | First mismatch index |
| --- | --- | --- | --- | --- |
| `reference-exact-backend` | `case-01-basic-chatml` | `true` | `parity_exact` | none |
| `reference-exact-backend` | `case-02-qwen-thinking-unicode` | `true` | `parity_exact` | none |
| `reference-exact-backend` | `case-03-role-boundary-injection` | `true` | `parity_exact` | none |
| `reference-exact-backend` | `case-04-empty-message-and-trailing-newline` | `true` | `parity_exact` | none |
| `faulty-backend-splits-special-tokens` | `case-01-basic-chatml` | `false` | `disqualifying_parity_mismatch` | 0 |
| `faulty-backend-drops-trailing-newline` | `case-02-qwen-thinking-unicode` | `false` | `disqualifying_parity_mismatch` | 33 |
| `faulty-backend-allows-role-injection` | `case-03-role-boundary-injection` | `false` | `disqualifying_parity_mismatch` | 18 |
| `faulty-backend-bpe-merge-drift` | `case-01-basic-chatml` | `false` | `disqualifying_parity_mismatch` | 29 |

Every tokenizer or chat-template divergence is emitted as `disqualifying_parity_mismatch` with exact token-offset diagnostics rather than noise.
