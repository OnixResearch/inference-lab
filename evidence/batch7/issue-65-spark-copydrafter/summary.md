# Issue #65: Pinned `spark-ds41f` & Isolated `CopyDrafter` Speculative Drafting

- Execution class: `bounded_contract_harness` (`default_enabled: false`, pinned `christopherowen/spark-ds41f@75fb3e72` `Apache-2.0`)
- Hardware gate: `unavailable_hardware_runtime` (4x DGX Spark cluster serving pinned christopherowen/spark-ds41f@75fb3e72 (DeepSeek-V4.1-Flash 1M))

## Lossless CopyDrafter Verification
- `baseline_off_repetitive` (`enabled=false`): `accepted=0/0`, `passes=12`, `exact=true`, `stochastic_l1=0.000e0`, `outcome=accepted`
- `copydrafter_on_repetitive` (`enabled=true`): `accepted=10/10`, `passes=3`, `exact=true`, `stochastic_l1=6.089e-16`, `outcome=accepted`
- `copydrafter_on_novel` (`enabled=true`): `accepted=7/7`, `passes=6`, `exact=true`, `stochastic_l1=5.573e-16`, `outcome=accepted`

## Synthetic 1-Stream / 16-Stream & 131K–1M Context Profiles
- `baseline_1stream_131k` (`streams=1`, `ctx=131072`, `copy=false`): `acc=0.000`, `tok_s=Some(48.0)`, `ttft_s=Some(3.4)`, `outcome=accepted`
- `copydrafter_1stream_131k` (`streams=1`, `ctx=131072`, `copy=true`): `acc=1.000`, `tok_s=Some(88.80000000000001)`, `ttft_s=Some(1.4)`, `outcome=accepted`
- `baseline_16stream_131k` (`streams=16`, `ctx=131072`, `copy=false`): `acc=0.000`, `tok_s=Some(180.0)`, `ttft_s=Some(3.8)`, `outcome=accepted`
- `copydrafter_16stream_131k` (`streams=16`, `ctx=131072`, `copy=true`): `acc=0.800`, `tok_s=Some(302.40000000000003)`, `ttft_s=Some(1.7)`, `outcome=accepted`
- `copydrafter_1stream_1m` (`streams=1`, `ctx=1048576`, `copy=true`): `acc=0.900`, `tok_s=Some(63.540000000000006)`, `ttft_s=Some(11.2)`, `outcome=accepted`

## Negative Controls (8 / 8 rejected)
- `nc1_unverified_copy_draft_breaks_losslessness` -> `invalid` (accepting copy draft without target verification failed greedy_exact_match)
- `nc2_unverified_drafter_blocks_throughput` -> `invalid` (lossless_verified=false suppressed decode_tok_s and ttft_s to None)
- `nc3_zero_streams_rejected` -> `invalid` (streams=0 refused before throughput calculation)
- `nc4_acceptance_rate_above_one_rejected` -> `invalid` (acceptance_rate=1.2 outside [0, 1] refused)
- `nc5_unnormalized_distribution_rejected` -> `invalid` (target distribution summing to 0.5 refused by rejection sampler)
- `nc6_nan_draft_prob_rejected` -> `invalid` (NaN draft probability rejected before marginal computation)
- `nc7_zero_proposals_returns_none_not_nan` -> `unsupported` (0 draft proposals returned acceptance_rate=None instead of 0/0 NaN)
- `nc8_default_on_copydrafter_refused` -> `unsupported` (CopyDrafter must stay isolated behind default_enabled=false toggle)
