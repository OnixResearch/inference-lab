# Spec: `http-batch-group` import from `tt-onix`

## Requirements

1. **Exact donor byte preservation**: `pkgs/http-batch-group/qwen38-batch-group.py` and `pkgs/http-batch-group/workloads/qwen38-subagent-workload.json` MUST match the exact bytes and SHA-256 digests from `OnixResearch/tt-onix@5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a`.
2. **Machine-readable provenance and license preservation**:
   - `pkgs/http-batch-group/PROVENANCE.json` MUST record donor repository, commit, license (`LGPL-2.1-only`), rollback pin, and exact SHA-256 digests of every imported file.
   - `LICENSES/LGPL-2.1-only.txt` MUST remain intact and referenced by `PROVENANCE.json`.
3. **Offline device-free verification**:
   - `nix flake check --no-update-lock-file --option builders ''` MUST build and test `checks.<system>.http-batch-group` offline using only Python's standard library against a loopback HTTP server.
4. **No regressions or path leaks**:
   - Existing Rust harnesses and retained `evidence/` bundles MUST remain unchanged.
   - Zero local home-directory or user-specific filesystem paths MAY appear in any imported file, manifest, doc, test, or OpenSpec document.
