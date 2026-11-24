use std::fs;
use std::path::Path;

use inference_lab::json::sha256_hex;
use inference_lab::serving_bench;

const EXPECTED_FILES: &[(&str, &str)] = &[
    (
        "pkgs/qwen38-benchmark/qwen38_benchmark.py",
        "42a054d03b2555d7d4f933759efcf9b4cc53c060cc2f84609f0c1b6aebe21fd3",
    ),
    (
        "pkgs/qwen38-benchmark/corpus.json",
        "08e682aad384802a0415bd56aa1fb2840b8a8f343acf6eb7b00f2afeb48e2015",
    ),
    (
        "pkgs/qwen38-benchmark/agentic_trace.json",
        "6c1127cbd528a3ca28b22f236133e6f180307f4cae2f59a30f749a46683e8de3",
    ),
    (
        "pkgs/qwen38-benchmark/default.nix",
        "d2c2e6bbf91d728c6bed388e27849556141ce4d128851257a56753229a2649db",
    ),
    (
        "LICENSES/LGPL-2.1-only.txt",
        "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331",
    ),
];

#[test]
fn imported_qwen38_benchmark_matches_donor_provenance() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let report = serving_bench::run_default_harness();
    let provenance_text = fs::read_to_string(root.join("pkgs/qwen38-benchmark/PROVENANCE.json"))
        .expect("PROVENANCE.json must exist");

    assert!(provenance_text.contains(&report.donor.donor_repo));
    assert!(provenance_text.contains(&report.donor.donor_commit));
    assert!(provenance_text.contains(&report.donor.license));
    assert!(provenance_text.contains(&report.donor.legacy_schema));

    for rel in &report.donor.imported_files {
        let path = root.join(rel);
        assert!(
            path.is_file(),
            "file declared in DonorProvenance must exist: {}",
            rel
        );
    }

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

    let license_text = fs::read_to_string(root.join("LICENSES/LGPL-2.1-only.txt"))
        .expect("LGPL-2.1-only license text must exist");
    assert!(license_text.contains("GNU LESSER GENERAL PUBLIC LICENSE"));
    assert!(license_text.contains("Version 2.1, February 1999"));
}

#[test]
fn imported_qwen38_benchmark_has_no_local_home_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = [
        "pkgs/qwen38-benchmark/qwen38_benchmark.py",
        "pkgs/qwen38-benchmark/corpus.json",
        "pkgs/qwen38-benchmark/agentic_trace.json",
        "pkgs/qwen38-benchmark/default.nix",
        "pkgs/qwen38-benchmark/PROVENANCE.json",
        "docs/qwen38-benchmark.md",
        "openspec/changes/import-tt-onix-qwen38-benchmark/proposal.md",
        "openspec/changes/import-tt-onix-qwen38-benchmark/design.md",
        "openspec/changes/import-tt-onix-qwen38-benchmark/tasks.md",
        "openspec/changes/import-tt-onix-qwen38-benchmark/specs/qwen38-benchmark-import/spec.md",
    ];
    for rel in files {
        let text = fs::read_to_string(root.join(rel)).expect("imported text file must be readable");
        assert!(
            !text.contains("/home/"),
            "imported file {rel} must not contain local /home/ paths"
        );
    }
}
