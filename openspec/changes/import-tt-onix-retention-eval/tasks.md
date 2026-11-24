# Tasks: Import `tt-onix` `retention-eval`

- [x] Register `import-tt-onix-retention-eval` OpenSpec change (`proposal.md`, `design.md`, `tasks.md`, `specs/retention-eval-import/spec.md`) in `openspec/README.md`.
- [x] Import byte-identical `pkgs/retention-eval/retention.py`, `pkgs/retention-eval/data/passages.json`, and `pkgs/retention-eval/data/qwen38-six-passages.json` from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`, and write `pkgs/retention-eval/PROVENANCE.json`.
- [x] Add standalone CLI (`pkgs/retention-eval/cli.py`), unit test suite (`pkgs/retention-eval/tests/test_retention.py`), and Nix package (`pkgs/retention-eval/default.nix`).
- [x] Write portable documentation (`docs/retention-eval.md`) and wire `packages.<system>.retention-eval`, `packages.<system>.retention-eval-full`, `apps.<system>.retention-eval`, and `checks.<system>.retention-eval` in `flake.nix`.
- [x] Add Rust integration test (`tests/retention_eval_import.rs`) verifying exact SHA-256 provenance, `LGPL-2.1-only` license preservation, and zero local home-directory path leaks.
- [x] Verify `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, Python unit tests, and `nix flake check --no-update-lock-file --option builders ''`.
