# Tasks: Import `tt-onix` `http-batch-group`

- [x] Register `import-tt-onix-http-batch-group` OpenSpec change (`proposal.md`, `design.md`, `tasks.md`, `specs/http-batch-group-import/spec.md`) in `openspec/README.md`.
- [x] Import byte-identical `pkgs/http-batch-group/qwen38-batch-group.py` and `pkgs/http-batch-group/workloads/qwen38-subagent-workload.json` from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`, and write `pkgs/http-batch-group/PROVENANCE.json`.
- [x] Add loopback test suite (`pkgs/http-batch-group/tests/test_batch_group.py`) and Nix package (`pkgs/http-batch-group/default.nix`).
- [x] Write portable documentation (`docs/http-batch-group.md`) and wire `packages.<system>.http-batch-group`, `apps.<system>.http-batch-group`, and `checks.<system>.http-batch-group` in `flake.nix`.
- [x] Add Rust integration test (`tests/http_batch_group_import.rs`) verifying exact SHA-256 provenance, `LGPL-2.1-only` license preservation, and zero local home-directory path leaks.
- [x] Verify `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, Python loopback tests, and `nix flake check --no-update-lock-file --option builders ''`.
