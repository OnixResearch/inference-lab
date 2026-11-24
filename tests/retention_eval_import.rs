use std::fs;
use std::path::Path;

use inference_lab::json::sha256_hex;

const EXPECTED_FILES: &[(&str, &str)] = &[
    (
        "pkgs/retention-eval/retention.py",
        "9d258e5773fc2cd1c0b37d8e8e849956969260ea9c1324918e9f83e90038e478",
    ),
    (
        "pkgs/retention-eval/data/passages.json",
        "fb09d9303eed5cf7cc3c9baf3c1a743f9721190665b2051c39ba257de5b42c7e",
    ),
    (
        "pkgs/retention-eval/data/qwen38-six-passages.json",
        "07b95ff60e0a5f596c3751502c4819eab79233ab181aee5afad660a7cd53e77b",
    ),
    (
        "LICENSES/LGPL-2.1-only.txt",
        "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331",
    ),
];

#[test]
fn imported_retention_eval_matches_donor_provenance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let provenance_text = fs::read_to_string(root.join("pkgs/retention-eval/PROVENANCE.json"))
        .expect("PROVENANCE.json must exist");

    assert!(provenance_text.contains("OnixResearch/tt-onix"));
    assert!(provenance_text.contains("5ed5e90cdd265443fa2f9b73e4ce46aeaa71837a"));
    assert!(provenance_text.contains("LGPL-2.1-only"));
    assert!(provenance_text.contains("qwen38-six-passage-v1"));

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

    let cli_text =
        fs::read_to_string(root.join("pkgs/retention-eval/cli.py")).expect("cli.py must exist");
    assert!(!cli_text.contains("ttnn"));
    assert!(!cli_text.contains("qwen36_service"));
}

#[test]
fn imported_retention_eval_has_no_local_home_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "pkgs/retention-eval/retention.py",
        "pkgs/retention-eval/cli.py",
        "pkgs/retention-eval/default.nix",
        "pkgs/retention-eval/PROVENANCE.json",
        "pkgs/retention-eval/data/passages.json",
        "pkgs/retention-eval/data/qwen38-six-passages.json",
        "pkgs/retention-eval/tests/test_retention.py",
        "docs/retention-eval.md",
        "openspec/changes/import-tt-onix-retention-eval/proposal.md",
        "openspec/changes/import-tt-onix-retention-eval/design.md",
        "openspec/changes/import-tt-onix-retention-eval/tasks.md",
        "openspec/changes/import-tt-onix-retention-eval/specs/retention-eval-import/spec.md",
    ];
    for rel in files {
        let text = fs::read_to_string(root.join(rel)).expect("imported text file must be readable");
        assert!(
            !text.contains("/home/"),
            "imported file {rel} must not contain local /home/ paths"
        );
    }
}
