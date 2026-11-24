# Portable six-passage retention evaluator (`retention-eval`)

> [!NOTE]
> Imported byte-for-byte (`retention.py`, `data/passages.json`, `data/qwen38-six-passages.json`) from [`OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`](https://github.com/OnixResearch/tt-onix/tree/5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a/pkgs/tenstorrent-eval-tools) under `LGPL-2.1-only` (see `pkgs/retention-eval/PROVENANCE.json` and `LICENSES/LGPL-2.1-only.txt`).
>
> Tenstorrent-specific analytical tile traffic modeling (`traffic.py`) and P150x2 hardware logit collection (`collector.py`) remain in `OnixResearch/tt-onix/pkgs/tenstorrent-eval-tools`.

`retention-eval` provides backend-independent, provenance-bound full-vocabulary negative log-likelihood (NLL) scoring and paired receipt comparison over the frozen six-passage corpus (`qwen38-six-passage-v1`). No model weights or tokenizer binaries are redistributed; the shipped fixture records texts, token IDs, and asset SHA-256 digests only.

## Build and verify

```sh
# Lightweight stdlib-only CLI (compare / manifest validation / offline checks)
nix run .#retention-eval -- --help
nix build .#checks.x86_64-linux.retention-eval

# Full ML environment (adds torch + transformers for freeze and score)
nix run .#retention-eval-full -- --help
```

## Commands

### 1. Freeze actual tokenizer identities (`freeze`)

```sh
nix run .#retention-eval-full -- freeze \
  --model "$model" \
  --checkpoint 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --out corpus.json
```

`pkgs/retention-eval/data/qwen38-six-passages.json` freezes the six passages (trailing newlines included), token IDs (`89, 135, 63, 59, 55, 62`), scored target counts (`76, 122, 50, 46, 42, 49`), tokenizer asset SHA-256 digests, and authoritative `config.json` vocabulary size (`248320`). No chat template is applied (`add_special_tokens=True`, `local_files_only=True`, `trust_remote_code=False`).

### 2. Score exported full-vocabulary logits (`score`)

```sh
nix run .#retention-eval-full -- score \
  --corpus corpus.json \
  --logits baseline.pt \
  --out baseline.json

nix run .#retention-eval-full -- score \
  --corpus corpus.json \
  --logits treatment.pt \
  --baseline baseline.json \
  --changes changes.json \
  --out treatment.json
```

Scoring rules:

- Each `rows` tensor must have exact shape `[len(input_ids) - 13, vocabulary_size]`, predicting token indices `13..end`. Extra padding columns or truncated vocabularies are rejected.
- Every logit tensor is cast to `float32` **before** `float64` `log_softmax`; negative target log-probabilities are averaged within each passage and then averaged equally across the six passages.
- Hardware exports (`execution` starting with `hardware:`) must carry an `indexed-safetensors-sha256-v1` `weight_manifest` whose canonical digest matches `provenance.model = "weights-sha256:<digest>"`.
- Baseline and treatment provenance must match on every field except explicitly declared `package`, `cache`, and/or complete `precision_flags` values in `changes.json`.

### 3. Compare baseline and treatment receipts (`compare`)

```sh
nix run .#retention-eval -- compare \
  --baseline baseline.json \
  --treatment treatment.json \
  --original-diagnostic \
  --out comparison.json
```

`compare` enforces exact `baseline_sha256` content binding and matching corpus/config/tokenizer/vocabulary identities. `--original-diagnostic` optionally reports whether `mean_nll_delta <= +0.010` nats on these exact six passages only; this is a diagnostic, never a global quality floor or serving admission gate.
