# Design: `qwen38-benchmark` import and compatibility packaging

## Architecture boundary

`inference-lab` keeps its pure dependency-free Rust core (`src/lib.rs`, `unsafe_code = "forbid"`) untouched. The imported `qwen38-benchmark` worker lives under `pkgs/qwen38-benchmark/` as a self-contained standard-library Python package wrapped by Nix (`pkgs/qwen38-benchmark/default.nix`), matching `#7`'s requirement to keep Python as a pinned worker rather than rewriting tested HTTP/SSE client logic in Rust.

## Provenance and licensing

- Donor repository: `OnixResearch/tt-onix`
- Donor commit (rollback pin): `5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`
- License for imported files: `LGPL-2.1-only` (verbatim license text retained at `LICENSES/LGPL-2.1-only.txt`; `meta.license = lib.licenses.lgpl21Only` in `pkgs/qwen38-benchmark/default.nix`)
- Imported byte-identical files and SHA-256 digests recorded in `pkgs/qwen38-benchmark/PROVENANCE.json`:
  - `pkgs/qwen38-benchmark/qwen38_benchmark.py` (`42a054d03b2555d7d4f933759efcf9b4cc53c060cc2f84609f0c1b6aebe21fd3`)
  - `pkgs/qwen38-benchmark/corpus.json` (`08e682aad384802a0415bd56aa1fb2840b8a8f343acf6eb7b00f2afeb48e2015`)
  - `pkgs/qwen38-benchmark/agentic_trace.json` (`6c1127cbd528a3ca28b22f236133e6f180307f4cae2f59a30f749a46683e8de3`)
  - `pkgs/qwen38-benchmark/default.nix` (`d2c2e6bbf91d728c6bed388e27849556141ce4d128851257a56753229a2649db`)

## Nix flake integration

`flake.nix` adds:
- `packages.${system}.qwen38-benchmark = pkgs.callPackage ./pkgs/qwen38-benchmark { };`
- `apps.${system}.qwen38-benchmark` exposing `${self.packages.${system}.qwen38-benchmark}/bin/qwen38-benchmark`
- `checks.${system}.qwen38-benchmark = self.packages.${system}.qwen38-benchmark;`

Because `pkgs/qwen38-benchmark/default.nix` enables both `doCheck = true` and `doInstallCheck = true` running `qwen38_benchmark.py --self-test` against a loopback fake OpenAI server using only Python's standard library, `nix flake check --no-update-lock-file --option builders ''` verifies the imported worker offline across all supported flake systems with zero external Python packages, drivers, or hardware.

## Downstream `tt-onix` compatibility

After this commit lands on `OnixResearch/inference-lab`, `OnixResearch/tt-onix` replaces its local copy of `pkgs/qwen38-benchmark/{qwen38_benchmark.py,corpus.json,agentic_trace.json}` with a pinned `fetchFromGitHub` compatibility derivation referencing this exact `inference-lab` commit SHA and SRI hash, while documenting the explicit rollback commit `5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
