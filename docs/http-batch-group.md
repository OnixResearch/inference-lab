# Portable HTTP batch-group driver and frozen subagent workload (`http-batch-group`)

> [!NOTE]
> Imported byte-for-byte (`qwen38-batch-group.py`, `workloads/qwen38-subagent-workload.json`) from [`OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`](https://github.com/OnixResearch/tt-onix/tree/5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a/scripts) under `LGPL-2.1-only` (see `pkgs/http-batch-group/PROVENANCE.json` and `LICENSES/LGPL-2.1-only.txt`).
>
> Tenstorrent P150x2-coupled scripts (`qwen38-subagent-benchmark.py`, `qwen38-prefix-cache-screen.py`, `qwen38-continuous-slot-screen.py`, and `flash_next_memory_gate.py`) remain in `OnixResearch/tt-onix/scripts/`.

`http-batch-group` is a zero-dependency standard-library Python HTTP client that pre-opens `N` TCP connections to an OpenAI-compatible `/v1/completions` server and releases all requests simultaneously from a `threading.Barrier`, avoiding process/connection setup skew when probing short server batch-collection windows.

The package also ships the frozen three-worker Python code-review workload (`qwen38-three-review-subagents-v1`) at `$out/share/http-batch-group/qwen38-subagent-workload.json`.

## Build and verify

```sh
nix run .#http-batch-group -- --help
nix build .#checks.x86_64-linux.http-batch-group
```

## Example invocation

```sh
nix run .#http-batch-group -- \
  --url http://127.0.0.1:8000 \
  --request req-0.json \
  --request req-1.json \
  --output-directory runs/batch-group-out \
  --attempts 4
```

This is a client-side request-release tool only; it makes no throughput, latency, or hardware admission claims on its own.
