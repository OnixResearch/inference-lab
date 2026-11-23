# Issue #19: Corrected Pinned DeepSeek-Style Reference & Regressions

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`pinned_python_jax_optax_cpu_and_optional_gpu_suite`)
- Upstream pin: `Vaibhaav-Tiwari/deepseek-v3-jax@6901be60941b98dc654ed542465290db9e8bafa6` (`MIT`)
- Eq. 18 sequence loss: expected=`0.017`, corrected=`0.017`, reproduced upstream buggy=`0.003`
- Checkpoint roundtrip exact: `true` (`includes_optimizer_rng_state=false`)

## Defect Reproduction & Correction Regressions
- `01-eq18-raw-affinity-seq-loss-vs-biased-dispatch`: `accepted` (bug_reproduced=`true`, corrected_pass=`true`) — Full moe() caller matches hand-derived Eq. 18 (0.017) using raw affinity top-k while keeping bias-adjusted dispatch counts [0,0,1,1] and reproducing upstream buggy loss (0.003).
- `02-mtp-off-ablation-n-mtp-zero`: `accepted` (bug_reproduced=`true`, corrected_pass=`true`) — Reproduces upstream n_mtp=0 empty-reduction failure; fixed path returns finite main loss, 0.0 MTP aux loss, and mtp_executed=false.
- `03-all-dense-ablation-n-dense-eq-n-layers`: `accepted` (bug_reproduced=`true`, corrected_pass=`true`) — Reproduces upstream empty MoE-stack crash; fixed 2-layer all-dense stack executes 0 MoE layers with finite outputs and 0.0 aux loss.
- `04-eval-interval-decoupled-from-log-interval`: `accepted` (bug_reproduced=`true`, corrected_pass=`true`) — Reproduces upstream coupled-log skipped evals; fixed eval_interval=2 executes at steps [2,4,6] independently of log_interval=5.
- `05-get-batch-inclusive-final-window-and-short-data-error`: `accepted` (bug_reproduced=`true`, corrected_pass=`true`) — Reproduces upstream exclusive-bound omission (0 windows on len==seq_len+1); fixed path reaches exact final window and rejects len<=seq_len.
- `06-negative-wrong-causal-mask-altered-gates-and-shifted-mtp-detected`: `accepted` (bug_reproduced=`false`, corrected_pass=`true`) — Disabling causal mask, altering MoE gate weights, and shifting MTP target index are all detected by executable regressions.
- `07-positive-finite-diff-grad-nonfinite-rejection-and-binary-ckpt`: `accepted` (bug_reproduced=`false`, corrected_pass=`true`) — Finite-difference router gradient is finite, NaN inputs rejected, binary LE checkpoint export/import round-trips bit-exact while rejecting truncated bytes.

> Architectural TINY reference trained from scratch (not an official DeepSeek checkpoint importer). Operation fixtures are reused directly by `#20` and `#21`.
