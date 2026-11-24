# Spec: `retention-eval` import from `tt-onix`

## Requirements

1. **Exact donor byte preservation**: `pkgs/retention-eval/retention.py`, `pkgs/retention-eval/data/passages.json`, and `pkgs/retention-eval/data/qwen38-six-passages.json` MUST match the exact bytes and SHA-256 digests from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
2. **Machine-readable provenance and license preservation**:
   - `pkgs/retention-eval/PROVENANCE.json` MUST record donor repository, commit, license (`LGPL-2.1-only`), schema (`qwen38-six-passage-v1`), rollback pin, and exact SHA-256 digests of every imported file.
   - `LICENSES/LGPL-2.1-only.txt` MUST remain intact and referenced by `PROVENANCE.json`.
3. **Clean backend boundary**:
   - `pkgs/retention-eval/` MUST NOT import `ttnn`, `qwen36_service`, or any Tenstorrent driver/device package.
   - `retention-eval` CLI MUST expose only `freeze`, `score`, and `compare`.
4. **Offline device-free verification**:
   - `nix flake check --no-update-lock-file --option builders ''` MUST build and test `checks.<system>.retention-eval` offline with no drivers, hardware, model weights, or credentials.
   - When `REQUIRE_TORCH=1` (as in `retention-eval-full`), full float32-to-float64 log-softmax scoring tests MUST run and pass rather than skip.
5. **No regressions or path leaks**:
   - Existing Rust harnesses and retained `evidence/` bundles MUST remain unchanged.
   - Zero local home-directory or user-specific filesystem paths MAY appear in any imported file, manifest, doc, test, or OpenSpec document.
