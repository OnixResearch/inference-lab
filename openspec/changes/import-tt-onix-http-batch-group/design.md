# Design: `http-batch-group` import and `tt-onix` scripts boundary

## Ownership split for `tt-onix/scripts/`

| Asset | Owner | Reason |
| --- | --- | --- |
| `qwen38-batch-group.py` | `OnixResearch/inference-lab` (`pkgs/http-batch-group/`) | Pure stdlib HTTP `/v1/completions` barrier-release driver; zero backend or device assumptions |
| `qwen38-subagent-workload.json` | `OnixResearch/inference-lab` (`pkgs/http-batch-group/workloads/`) | Frozen three-worker Python review workload (`qwen38-three-review-subagents-v1`) |
| `qwen38-subagent-benchmark.py`, `qwen38-subagent-benchmark.yml` | `OnixResearch/tt-onix` (`scripts/`) | Asserts P150x2 PCIe `[0, 1]` and `qwen36_service` continuous-slot gateway evidence |
| `qwen38-prefix-cache-screen.py`, `qwen38-continuous-slot-screen.py` | `OnixResearch/tt-onix` (`scripts/`) | Asserts `physical_pcie_device_ids == [0, 1]` and emits `qwen36_service` admission schemas |
| `flash_next_memory_gate.py` | `OnixResearch/tt-onix` (`scripts/`) | Models `TTNN_TILE_BYTES` and `P150_NOMINAL_BYTES` for P150x2 layer pipelining |

## Provenance and licensing

- Donor repository: `OnixResearch/tt-onix`
- Donor commit (rollback pin): `5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`
- License: `LGPL-2.1-only` (`LICENSES/LGPL-2.1-only.txt`)
- Imported byte-identical files (`pkgs/http-batch-group/PROVENANCE.json`):
  - `pkgs/http-batch-group/qwen38-batch-group.py` (`dcf425373c423384b5cdb5669723322a36ab1b04c0308cf17245f0e47c83cbf0`)
  - `pkgs/http-batch-group/workloads/qwen38-subagent-workload.json` (`b354dbc38cd2b51b0468eaaecc8c049404078dfa503ab1d1a024e732b9e5c486`)

## Nix packaging

`pkgs/http-batch-group/default.nix` wraps `qwen38-batch-group.py` as `bin/http-batch-group` using stdlib `python3`, installs `share/http-batch-group/qwen38-subagent-workload.json` and `share/http-batch-group/PROVENANCE.json`, and runs `tests/test_batch_group.py` in both `checkPhase` and `installCheckPhase` against a local loopback `http.server`.
