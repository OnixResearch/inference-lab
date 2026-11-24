# Spec: `qwen38-benchmark` import from `tt-onix`

## Requirements

1. **Exact donor byte preservation**: `pkgs/qwen38-benchmark/qwen38_benchmark.py`, `pkgs/qwen38-benchmark/corpus.json`, `pkgs/qwen38-benchmark/agentic_trace.json`, and `pkgs/qwen38-benchmark/default.nix` MUST match the exact bytes and SHA-256 digests from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
2. **Machine-readable provenance and license preservation**:
   - `pkgs/qwen38-benchmark/PROVENANCE.json` MUST record donor repository, commit, license (`LGPL-2.1-only`), schema (`onix.qwen38-benchmark.v1`), rollback pin, and exact SHA-256 digests of every imported file.
   - `LICENSES/LGPL-2.1-only.txt` MUST contain the verbatim GNU Lesser General Public License v2.1 text.
   - Every path listed in `serving_bench::run_default_harness().donor.imported_files` MUST exist in the repository.
3. **Offline device-free Nix packaging and check**:
   - `nix build .#qwen38-benchmark --no-update-lock-file --option builders ''` MUST succeed offline and run `qwen38-benchmark --self-test` in both `checkPhase` and `installCheckPhase` with no Tenstorrent driver, hardware, model weights, or credentials.
   - `nix flake check --no-update-lock-file --option builders ''` MUST include `checks.<system>.qwen38-benchmark` alongside existing Rust package and quality checks.
4. **Compatibility parity with donor command**:
   - The imported command and donor command MUST produce identical workload digests (`workload_sha256`), identical deterministic reducer summaries, identical parity pass/fail decisions, and identical tamper rejection on retained positive and negative fixtures.
5. **No regressions or path leaks**:
   - Existing `evidence/bootstrap/` and `evidence/batch1/`–`evidence/batch6/` two-run byte-identical checks MUST remain unchanged.
   - Zero local home-directory or user-specific filesystem paths MAY appear in any imported file, manifest, doc, test, or OpenSpec document.
