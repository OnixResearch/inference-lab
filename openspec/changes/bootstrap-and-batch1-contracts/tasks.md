# Implementation tasks

- [x] Scaffold Rust 1.85.1 crate (`unsafe_code = "forbid"`, zero external dependencies), offline-compatible Nix flake (`flake.nix`, `flake.lock`), `.gitignore`, `.gitattributes`, `.envrc`, CI workflow, `README.md`, `AGENTS.md`, `docs/`, and `openspec/`.
- [x] Write pinned default-off related-work notes (`docs/related-work/issue-56-fpga-maple.md`, `issue-58-compositional-harness-rlm27b.md`, `issue-60-latent-lattice-lm.md`) and `docs/source-review.md`.
- [x] Implement shared deterministic JSON/SHA-256 (`src/json.rs`), provenance (`src/provenance.rs`), and bootstrap seed (`src/seed.rs`).
- [x] Implement Issue `#1` diffusion proposal contract, EBM selection fixture, Noble admission controls, and default-off `DependentWeave`/JSONL/socketteer interchange check (`src/diffusion.rs`).
- [x] Implement Issue `#40` host/GPU trace correlation qualification and negative controls (`src/trace.rs`).
- [x] Implement Issue `#41` versioned transformer stage contract, 4-layer pure-Rust split/unsplit numerical reference, poisoned-buffer/stale-KV guards, and zero silent CPU fallback (`src/stage.rs`).
- [x] Implement Issue `#43` power/throttle cadence qualification, sensor-gated energy reporting, and known-equivalent / wrong-but-fast / order-drift controls (`src/power.rs`).
- [x] Implement Issue `#52` deterministic BPE + special-token reference tokenizer, chat-template renderer, and disqualifying parity checker (`src/tokenizer.rs`).
- [x] Implement Issues `#56`, `#58`, `#60` machine-readable default-off gate manifests and validator (`src/related_work.rs`).
- [x] Implement imperative CLI shell (`src/main.rs`) enforcing fresh output directories and `COMPLETE` last.
- [x] Add comprehensive adversarial unit and integration tests covering every contract invariant and negative control.
- [x] Validate `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings`, `cargo test --locked --all-targets`, `cargo test --locked --doc`, and `nix flake check --no-update-lock-file --option builders ''`.
- [x] Generate 2-run byte-identical retained `evidence/bootstrap/` and `evidence/batch1/` artifacts with SHA-256 manifests and zero `/home/` leaks.
- [x] Complete adversarial verifier audit before committing and pushing.
