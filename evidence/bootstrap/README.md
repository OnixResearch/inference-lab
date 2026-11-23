# Bootstrap validation evidence

Executed with pinned Rust 1.85.1 under the offline Nix flake (`x86_64-linux`). This directory retains one authored `serving-contract-v1` seed run, captured validation logs, and exact source/build-input/artifact/log SHA-256 digests in [manifest.json](manifest.json).

## Checks observed

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed (`format.log`) |
| `cargo clippy --locked --all-targets -- -D warnings` | Passed (`clippy.log`) |
| `cargo test --locked --all-targets` | 22 passed (`tests.log`) |
| `cargo test --locked --doc` | Passed; 0 doctests (`doctests.log`) |
| Two `benchmark` CLI runs | Both completed; all four artifacts byte-identical (`benchmark.log`, `benchmark-repeat.log`) |
| `jq` JSON validation | `inputs.json` and `results.json` parsed cleanly |
| Local offline `nix flake check --no-update-lock-file --option builders ''` | Passed on `x86_64-linux` (`package-and-tests` and `rust-quality`) |

## Scope and limitations

- **Execution class**: `bounded_contract_harness`
- **Physical hardware gate**: `unavailable_hardware_runtime` (no physical GPU or model server executed)
- Logical ticks are deterministic contract units, never wall-clock latency or hardware throughput.
- The empty `COMPLETE` marker confirms publication completed after `inputs.json`, `results.json`, and `summary.md`; it does not assert physical serving qualification.
- Additional flake platforms (`aarch64-linux`, `x86_64-darwin`, `aarch64-darwin`) are configured and require separate execution to qualify.
