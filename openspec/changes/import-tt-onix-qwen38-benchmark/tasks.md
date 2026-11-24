# Tasks: Import `tt-onix` `qwen38-benchmark`

- [x] Register `import-tt-onix-qwen38-benchmark` OpenSpec change (`proposal.md`, `design.md`, `tasks.md`, `specs/qwen38-benchmark-import/spec.md`) in `openspec/README.md`.
- [x] Import byte-identical `pkgs/qwen38-benchmark/{qwen38_benchmark.py,corpus.json,agentic_trace.json,default.nix}` and `LICENSES/LGPL-2.1-only.txt` from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`, and write `pkgs/qwen38-benchmark/PROVENANCE.json`.
- [x] Import portable command/metric/agentic documentation into `docs/qwen38-benchmark.md` with donor pointer back to `OnixResearch/tt-onix`.
- [x] Wire `packages.<system>.qwen38-benchmark`, `apps.<system>.qwen38-benchmark`, and `checks.<system>.qwen38-benchmark` in `flake.nix`, and update `README.md`, `docs/architecture.md`, and `docs/source-review.md`.
- [x] Add `tests/qwen38_benchmark_import.rs` verifying `DonorProvenance` file presence, exact SHA-256 digests, `LGPL-2.1-only` license preservation, and zero local home-directory path leaks.
- [x] Verify `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, `nix flake check --no-update-lock-file --option builders ''`, and donor-vs-imported fixture parity (workload digest, reducer, parity, and tamper rejection).
