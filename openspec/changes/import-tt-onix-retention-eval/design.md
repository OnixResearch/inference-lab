# Design: `retention-eval` import and `tenstorrent-eval-tools` split

## Ownership split

| Component | Owner | Reason |
| --- | --- | --- |
| `retention.py` (`freeze`, `score`, `compare`, `weight_manifest`) | `OnixResearch/inference-lab` (`pkgs/retention-eval/`) | Pure CPU/offline full-vocabulary NLL scoring and content-bound provenance; zero Tenstorrent imports |
| `data/passages.json`, `data/qwen38-six-passages.json` | `OnixResearch/inference-lab` (`pkgs/retention-eval/data/`) | Frozen six-passage text and token/hash fixtures (no model weights or tokenizer binaries) |
| `traffic.py` | `OnixResearch/tt-onix` (`pkgs/tenstorrent-eval-tools/`) | Analytical Tensix 32×32 tile/L1/multicast modeling specific to Blackhole kernels |
| `collector.py` | `OnixResearch/tt-onix` (`pkgs/tenstorrent-eval-tools/`) | Hardware-only P150x2 collector importing `ttnn` and `qwen36_service` |

## Provenance and licensing

- Donor repository: `OnixResearch/tt-onix`
- Donor commit (rollback pin): `5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`
- License: `LGPL-2.1-only` (`LICENSES/LGPL-2.1-only.txt`)
- Imported byte-identical files and SHA-256 digests (`pkgs/retention-eval/PROVENANCE.json`):
  - `pkgs/retention-eval/retention.py` (`9d258e5773fc2cd1c0b37d8e8e849956969260ea9c1324918e9f83e90038e478`)
  - `pkgs/retention-eval/data/passages.json` (`fb09d9303eed5cf7cc3c9baf3c1a743f9721190665b2051c39ba257de5b42c7e`)
  - `pkgs/retention-eval/data/qwen38-six-passages.json` (`07b95ff60e0a5f596c3751502c4819eab79233ab181aee5afad660a7cd53e77b`)

## Nix packaging and offline check tiers

`retention.py` imports `transformers` lazily inside `freeze()` and `torch` lazily inside `score()`; all manifest hashing, schema validation, provenance matching, and receipt comparison functions use Python's standard library only.

`pkgs/retention-eval/default.nix` therefore supports two explicit profiles via `withMlDeps ? false`:

- `packages.<system>.retention-eval` (`withMlDeps = false`): standard-library-only wrapper (`python3`) checked on every flake system inside `checks.<system>.retention-eval`, verifying checkpoint manifest hashing, traversal rejection, corpus/provenance/comparison validation, and CLI smoke offline in seconds.
- `packages.<system>.retention-eval-full` (`withMlDeps = true`): full `python3.withPackages (p: [ p.torch p.transformers ])` environment with `REQUIRE_TORCH=1` so all PyTorch float32-to-float64 log-softmax scoring tests are strictly enforced.

## Downstream `tt-onix` compatibility

After this commit lands on `OnixResearch/inference-lab`, `OnixResearch/tt-onix` updates `pkgs/tenstorrent-eval-tools/default.nix` to fetch `retention.py` and `data/` from this pinned `inference-lab` commit SHA and SRI hash, removes the duplicate local copies, and keeps running the full 16-test `test_behavior.py` suite (covering both local `traffic.py` and pinned `retention.py` with PyTorch).
