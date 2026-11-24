# Issue #73: Pinned `LFM3-1.7B` (Tri-Hybrid 256k) & `d1-2B` Single-Pass Typed Decision Qualification

- Execution class: `bounded_contract_harness` (`default_enabled: false`, `pin_status: post_only_checkpoints_unpinned_offline`, `LFM3-Audio-2B` excluded to `audio-lab#2`)
- Hardware / artifact gate: `blocked_on_artifacts` (GPU serving host evaluating Liquid AI LFM3-1.7B (256k tri-hybrid) & d1-2B single-pass decision checkpoints)
- Miniature tri-hybrid cached-vs-full parity: `max_abs_err=0.000e0` (tol `<= 1e-12`)

## `d1-2B` Single-Pass vs Autoregressive Typed Decision Conformance & Calibration
- `d1_2b_single_pass_typed`: `schema=6/6`, `consumer_correct=5/6`, `ece=Some(0.215)`, `brier=Some(0.07265)`, `mean_lat_ms=Some(4.1499999999999995)`
- `autoregressive_structured_baseline`: `schema=5/6`, `consumer_correct=4/6`, `ece=Some(0.29166666666666674)`, `brier=Some(0.29211666666666664)`, `mean_lat_ms=Some(30.100000000000005)`

## `LFM3-1.7B` 8K–256K Context Scaling Contract
- `ctx=8192`: `state_kv=46.0 MB`, `tok_s=144.2`, `retention=0.93`
- `ctx=32768`: `state_kv=100.1 MB`, `tok_s=141.8`, `retention=0.92`
- `ctx=131072`: `state_kv=316.4 MB`, `tok_s=133.0`, `retention=0.89`
- `ctx=262144`: `state_kv=604.7 MB`, `tok_s=122.9`, `retention=0.84`

## Negative Controls (8 / 8 rejected)
- `nc1_high_confidence_wrong_decision_not_accepted` -> `invalid` (schema-valid wrong decision at p=0.99 scored 0/1 consumer accuracy & Brier=0.9801)
- `nc2_schema_invalid_output_fails_consumer_check` -> `invalid` (schema_valid=false rejected from consumer_correct_count despite matching label)
- `nc3_lfm3_audio_2b_out_of_scope_audio_lab_2` -> `unsupported` (LFM3-Audio-2B belongs strictly to audio-lab#2 and is refused here)
- `nc4_probability_above_one_rejected` -> `invalid` (reported_prob=1.25 outside [0, 1] refused before ECE/Brier)
- `nc5_nan_probability_rejected` -> `invalid` (NaN reported_prob refused before calibration binning)
- `nc6_empty_fixture_returns_none_not_nan` -> `invalid` (empty decision fixture returned ece=None, brier=None, invalid)
- `nc7_nan_tri_hybrid_input_rejected` -> `invalid` (NaN activation refused in LFM3 tri-hybrid cached decode)
- `nc8_unpinned_lfm3_d1_checkpoints_blocked` -> `blocked_on_artifacts` (LFM3-1.7B & d1-2B public checkpoint revisions unpinned offline)
