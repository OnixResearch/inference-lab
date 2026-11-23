# Working in Inference Lab

## Purpose

Build bounded, reproducible inference, serving, numerical-parity, observability, and proposal-contract experiments. Read the architecture, protocol, and pinned source review before changing a harness or adding a backend adapter.

Keep contract logic, reference math, conformance checkers, and deterministic evaluation in a pure dependency-free Rust core (`unsafe_code = "forbid"`). Keep CLI parsing, filesystem publication, process invocation, clocks, network I/O, credentials, and device telemetry acquisition in explicit shells/adapters. Use pinned inputs and finite candidate, sequence, step, retry, duration, memory, and output bounds.

## Evidence rules

- Separate execution classes explicitly: `bounded_contract_harness` (deterministic pure-Rust verification of contracts, negative controls, and reference math), `unavailable_hardware_runtime` (physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, AWS F2 FPGA, NVMe, or 27B serve not executed locally), and `blocked_on_artifacts` (upstream code, weights, reference, or license unavailable). Never conflate a contract harness or CPU reference with a physical hardware run.
- Record exact provenance for every run: hardware/device identity, backend/runtime/driver/kernel revision, model/checkpoint/effective-weight digests (with explicit digest algorithm), tokenizer and chat-template revisions, precision/quantization, batch/context/warm-cold state, exact command, and input hashes.
- Preserve non-collapsing outcomes: never collapse `accepted`, `invalid`, `unsupported`, `exhausted`, `timeout`, `oom`, `cancelled`, `internal_failure`, `unavailable_hardware_runtime`, or `blocked_on_artifacts` into zero, missing, or silent CPU/fallback execution.
- Missing, dropped, stale, overflowed, or unmapped telemetry is `unknown` or `incomplete`, never zero and never evidence of absence (e.g., absent throttle counters never prove no throttling). Do not subtract timestamps across distinct clock domains without a calibrated mapping and explicit uncertainty bounds.
- Vendor posts, X threads, blog posts, and paper numbers are strictly **unverified related work**. Never import external claims as acceptance gates, performance targets, or fabricated GPU/FPGA/TPU measurements.
- Retain every planned item (positive and negative candidates, failed/rejected/timed-out/OOM tasks, mismatched tokens, unresolved trace records) with explicit population denominators. Informational model confidence is never a consumer checker verdict.
- Keep hard consumer admission, tokenizer/template parity checks, and numerical correctness gates outside soft score composition. A faster candidate that fails correctness or parity is disqualified.
- Write each run into a fresh output directory (its parent must exist; never overwrite an existing directory) and write an empty `COMPLETE` marker **last**. `COMPLETE` means artifact publication finished; it never proves workload success or hardware qualification.
- Keep zero local `/home/` or user-specific filesystem paths in committed code, docs, tests, manifests, or retained logs.

## Boundaries

Do not build a production inference router, fleet scheduler, policy trainer, or replacement acceptance authority here:

- **Noble** owns its structured source representation, symbolic/CEGIS synthesis baseline (`cegis.mjs`), proofs, and kernel acceptance.
- **Compiler Lab** owns native candidate checks, source-proof intake, compiler-input reduction, and RTL/Verilog admission.
- **RSI Lab** owns recurrent-state learning, retention, promotion, and stateful agent protocols.
- **TT-Onix** owns physical Tenstorrent worker execution, device telemetry, and host gates.
- **Spark NixOS / Aspen / Molten / Kiln / Lattice / Animus / Systems Lab / ChaosControl** retain their respective runtime, reservation, scheduling, orchestration, and fault-replay authorities.

Adapters execute only explicitly selected resources. Repository bootstrap and harness runs must never start, stop, or evict resident model services or reconfigure host devices.

## Changes and validation

For every new experiment or semantic change, add or update an OpenSpec proposal, design, tasks, and requirement specs under `openspec/changes/` first. Check off tasks only from observed evidence.

Validate changes with:

- `cargo fmt --all --check`
- `cargo clippy --all-targets --locked -- -D warnings`
- `cargo test --locked --all-targets` and `cargo test --locked --doc`
- `nix flake check --no-update-lock-file --option builders ''` (or without `--option builders ''` on machines with configured remote builders)
- Two-run byte-identical output checks against retained `evidence/`.

Keep ad-hoc runs under ignored `runs/`. Never commit secrets, model weights, external eval corpora, private endpoints, or unbounded logs. Commit compact reviewed evidence only with SHA-256 manifests and explicit scope limitations.
