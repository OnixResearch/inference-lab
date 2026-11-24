use std::fs;
use std::path::Path;

use inference_lab::json::sha256_hex;

const EXPECTED_FILES: &[(&str, &str)] = &[
    (
        "pkgs/http-batch-group/qwen38-batch-group.py",
        "dcf425373c423384b5cdb5669723322a36ab1b04c0308cf17245f0e47c83cbf0",
    ),
    (
        "pkgs/http-batch-group/workloads/qwen38-subagent-workload.json",
        "b354dbc38cd2b51b0468eaaecc8c049404078dfa503ab1d1a024e732b9e5c486",
    ),
    (
        "LICENSES/LGPL-2.1-only.txt",
        "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331",
    ),
];

#[test]
fn imported_http_batch_group_matches_donor_provenance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let provenance_text = fs::read_to_string(root.join("pkgs/http-batch-group/PROVENANCE.json"))
        .expect("PROVENANCE.json must exist");

    assert!(provenance_text.contains("OnixResearch/tt-onix"));
    assert!(provenance_text.contains("5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a"));
    assert!(provenance_text.contains("LGPL-2.1-only"));

    for &(rel, expected_sha256) in EXPECTED_FILES {
        let bytes = fs::read(root.join(rel)).expect("imported donor file must be readable");
        let actual_sha256 = sha256_hex(&bytes);
        assert_eq!(
            actual_sha256, expected_sha256,
            "SHA-256 mismatch for imported donor file {rel}"
        );
        assert!(
            provenance_text.contains(expected_sha256),
            "PROVENANCE.json must record SHA-256 for {rel}"
        );
    }
}

#[test]
fn imported_http_batch_group_has_no_local_home_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "pkgs/http-batch-group/qwen38-batch-group.py",
        "pkgs/http-batch-group/workloads/qwen38-subagent-workload.json",
        "pkgs/http-batch-group/default.nix",
        "pkgs/http-batch-group/PROVENANCE.json",
        "pkgs/http-batch-group/tests/test_batch_group.py",
        "docs/http-batch-group.md",
        "openspec/changes/import-tt-onix-http-batch-group/proposal.md",
        "openspec/changes/import-tt-onix-http-batch-group/design.md",
        "openspec/changes/import-tt-onix-http-batch-group/tasks.md",
        "openspec/changes/import-tt-onix-http-batch-group/specs/http-batch-group-import/spec.md",
    ];
    for rel in files {
        let text = fs::read_to_string(root.join(rel)).expect("imported text file must be readable");
        assert!(
            !text.contains("/home/"),
            "imported file {rel} must not contain local /home/ paths"
        );
    }
}
