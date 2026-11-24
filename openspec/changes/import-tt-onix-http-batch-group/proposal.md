# Import `tt-onix` portable HTTP batch-group driver and frozen subagent workload (`http-batch-group`)

## Why

An audit of `OnixResearch/tt-onix/scripts/` showed that most serving probes (`qwen38-prefix-cache-screen.py`, `qwen38-continuous-slot-screen.py`, `qwen38-subagent-benchmark.py`, and `flash_next_memory_gate.py`) directly assert P150x2 physical PCIe device IDs `[0, 1]`, `qwen36_service` decode evidence, or TTNN tile/DRAM layouts, and therefore properly remain in `tt-onix`.

Two assets in `tt-onix/scripts/` are completely backend-independent:

1. `scripts/qwen38-batch-group.py`: a standard-library Python HTTP client that opens `N` connections in advance and releases `POST /v1/completions` requests simultaneously from a `threading.Barrier` so they land within one server batch-collection window.
2. `scripts/qwen38-subagent-workload.json`: a frozen three-worker Python code-review workload (`qwen38-three-review-subagents-v1`) usable against any OpenAI-compatible serving stack or agent harness.

Importing both byte-for-byte into `OnixResearch/inference-lab` under `pkgs/http-batch-group/` gives any backend comparison access to the barrier-release client and frozen subagent workload with offline loopback verification.

## Change

1. Copy the exact donor files from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`:
   - `pkgs/http-batch-group/qwen38-batch-group.py`
   - `pkgs/http-batch-group/workloads/qwen38-subagent-workload.json`
2. Record `LGPL-2.1-only` provenance and exact SHA-256 digests in `pkgs/http-batch-group/PROVENANCE.json`.
3. Add stdlib loopback tests (`pkgs/http-batch-group/tests/test_batch_group.py`) exercising concurrent barrier release, response collection, HTTP error retry, non-HTTP scheme rejection, and frozen subagent workload schema integrity.
4. Package via `pkgs/http-batch-group/default.nix` and wire `packages.<system>.http-batch-group`, `apps.<system>.http-batch-group`, and `checks.<system>.http-batch-group` in `flake.nix`.
5. Document usage and ownership split in `docs/http-batch-group.md`, and add Rust integration tests (`tests/http_batch_group_import.rs`).

## Scope and limitations

- Tenstorrent-coupled drivers (`qwen38-prefix-cache-screen.py`, `qwen38-continuous-slot-screen.py`, `qwen38-subagent-benchmark.py`, `flash_next_memory_gate.py`) stay in `OnixResearch/tt-onix`.
- The pure dependency-free Rust core (`src/lib.rs`) and retained `evidence/` bundles are unchanged.
- Loopback tests verify client mechanics only; they make no server throughput or hardware claims.
