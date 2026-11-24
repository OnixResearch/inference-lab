# Import `tt-onix` portable six-passage retention evaluator (`retention-eval`)

## Why

`OnixResearch/tt-onix` currently bundles two distinct concerns inside `pkgs/tenstorrent-eval-tools`:

1. **Tenstorrent-specific hardware/kernel tools**: `traffic.py` (analytical Tensix 32×32 tile/L1/multicast modeling) and `collector.py` (explicit P150x2 hardware export importing `ttnn` and `qwen36_service`).
2. **Backend-independent retention evaluation**: `retention.py` (`freeze`, `score`, `compare`, and safetensors index+shard content binding) plus its frozen six-passage fixtures (`data/passages.json`, `data/qwen38-six-passages.json`).

Relocating the portable retention evaluator and its frozen fixtures into `OnixResearch/inference-lab` establishes a single source of truth for provenance-bound full-vocabulary NLL scoring across backends, while allowing `tt-onix` to pin `retention.py` and `data/` from `inference-lab` inside `pkgs/tenstorrent-eval-tools` and retain only Tenstorrent-specific `traffic.py` and `collector.py` locally.

## Change

1. Copy the exact donor source files from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a` without altering scoring math, schema constants, or fixture bytes:
   - `pkgs/retention-eval/retention.py`
   - `pkgs/retention-eval/data/passages.json`
   - `pkgs/retention-eval/data/qwen38-six-passages.json`
2. Add a standalone backend-agnostic CLI (`pkgs/retention-eval/cli.py`) exposing `retention-eval {freeze,score,compare}` without Tenstorrent `traffic` or `collect` commands.
3. Preserve `LGPL-2.1-only` license attribution (`LICENSES/LGPL-2.1-only.txt`) and commit `pkgs/retention-eval/PROVENANCE.json` recording donor repository, commit, license, schema (`qwen38-six-passage-v1`), rollback pin, and exact SHA-256 digests.
4. Add `pkgs/retention-eval/tests/test_retention.py` testing checkpoint content binding, corpus/provenance/comparison validation, CLI error handling, and (when PyTorch is present) full float32-to-float64 log-softmax scoring parity.
5. Add `pkgs/retention-eval/default.nix` and wire `packages.<system>.retention-eval`, `packages.<system>.retention-eval-full`, `apps.<system>.retention-eval`, and `checks.<system>.retention-eval` in `flake.nix`.
6. Document the portable retention CLI in `docs/retention-eval.md` and add Rust integration tests (`tests/retention_eval_import.rs`) verifying exact SHA-256 provenance, `LGPL-2.1-only` license preservation, and zero local home-directory path leaks.

## Scope and limitations

- The pure dependency-free Rust core (`src/lib.rs`, `unsafe_code = "forbid"`), `Cargo.toml`, `Cargo.lock`, and retained `evidence/` bundles are untouched.
- Tenstorrent tile traffic modeling (`traffic.py`) and P150x2 logit collection (`collector.py`) remain in `OnixResearch/tt-onix`.
- No live model weights, tokenizer files, or physical hardware runs are executed or claimed by this import.
