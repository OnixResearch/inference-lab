# Import `tt-onix` portable serving benchmark (`qwen38-benchmark`)

## Why

Batch 2 (`#7`) established the pure-Rust compatibility contract, offline reducer, and provenance model (`src/serving_bench.rs`) for `OnixResearch/tt-onix`'s portable HTTP/SSE serving benchmark (`qwen38_benchmark.py`, `corpus.json`, `agentic_trace.json`, `default.nix`, `docs/qwen38-benchmark.md`), while noting that the executable Python worker and frozen fixtures had not yet been relocated from `tt-onix`.

Importing the actual standard-library Python runner, frozen workloads, Nix package, and portable documentation into `OnixResearch/inference-lab` makes Inference Lab the single source of truth for the client benchmark while allowing `tt-onix` to switch `pkgs/qwen38-benchmark` to a pinned compatibility export with an explicit rollback pin to `tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.

## Change

1. Copy the exact donor source files from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a` without altering workload bytes or measurement semantics:
   - `pkgs/qwen38-benchmark/qwen38_benchmark.py`
   - `pkgs/qwen38-benchmark/corpus.json`
   - `pkgs/qwen38-benchmark/agentic_trace.json`
   - `pkgs/qwen38-benchmark/default.nix`
2. Preserve `LGPL-2.1-only` license attribution (`LICENSES/LGPL-2.1-only.txt`) and commit a machine-readable `pkgs/qwen38-benchmark/PROVENANCE.json` recording donor repository, commit, license, schema identity (`onix.qwen38-benchmark.v1`), rollback pin, and exact SHA-256 digests.
3. Import the portable runner, accounting, comparison, and agentic-mode documentation into `docs/qwen38-benchmark.md`, pointing Tenstorrent-specific deployment, vLLM preflight, and P150x2 service notes back to `OnixResearch/tt-onix`.
4. Expose `packages.<system>.qwen38-benchmark`, `apps.<system>.qwen38-benchmark`, and `checks.<system>.qwen38-benchmark` from `flake.nix` so offline `nix flake check` builds and runs `qwen38-benchmark --self-test` with no Tenstorrent driver, device, model, or credentials.
5. Add Rust integration tests (`tests/qwen38_benchmark_import.rs`) verifying exact SHA-256 provenance against `src/serving_bench.rs`'s `DonorProvenance`, `PROVENANCE.json`, `LICENSES/LGPL-2.1-only.txt`, and zero local home-directory path leaks.

## Scope and limitations

- The imported runner remains a standard-library Python client worker; it does not modify the pure dependency-free Rust core (`unsafe_code = "forbid"`), `Cargo.toml`, `Cargo.lock`, or retained `evidence/batch1/`–`evidence/batch6/` bundles.
- No live model endpoint, Tenstorrent hardware run, or production service cutover is claimed by this import.
- Generic stream-stall/per-token-deadline extensions (`tt-onix#17`) remain follow-on work built on top of this imported baseline after compatibility parity is verified.
