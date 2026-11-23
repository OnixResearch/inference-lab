use inference_lab::provenance::ExecutionClass;
use inference_lab::related_work::{
    default_records, run_default_harness, validate_related_work_gates,
};

#[test]
fn related_work_records_are_default_off_unverified_and_strictly_gated() {
    let report = run_default_harness();
    assert!(report.gate_validation_passed);
    assert!(report.validation_errors.is_empty());
    assert_eq!(report.records.len(), 3);

    for rec in &report.records {
        assert!(
            rec.default_off,
            "issue #{} must be default-off",
            rec.issue_number
        );
        assert!(
            !rec.claims_verified,
            "issue #{} claims must stay unverified",
            rec.issue_number
        );
        assert!(!rec.active_gates.is_empty());
        assert!(!rec.non_gating_unverified_author_claims.is_empty());
    }

    // Issue #56: empty repo has no SHA and is NOT runnable.
    let r56 = report
        .records
        .iter()
        .find(|r| r.issue_number == 56)
        .unwrap();
    assert_eq!(r56.gate_status, ExecutionClass::BlockedOnArtifacts);
    let empty_repo = r56
        .pinned_sources
        .iter()
        .find(|s| s.label == "empty_placeholder_repo")
        .unwrap();
    assert_eq!(empty_repo.commit_sha, None);
    assert!(!empty_repo.runnable_implementation);

    // Issue #58: RLM-27B commit SHA pinned, undeclared license kept as explicit gate.
    let r58 = report
        .records
        .iter()
        .find(|r| r.issue_number == 58)
        .unwrap();
    let rlm27b = r58
        .pinned_sources
        .iter()
        .find(|s| s.label == "hf_rlm_27b")
        .unwrap();
    assert_eq!(
        rlm27b.commit_sha.as_deref(),
        Some("f0b3b27db57c211234cb55ec512d87f45dabdc4c")
    );
    assert!(rlm27b.license_status.contains("undeclared"));
    assert!(!rlm27b.runnable_implementation);

    // Issue #60: project-page repo pinned but marked HTML-only, NOT runnable, distinct from OnixResearch/lattice.
    let r60 = report
        .records
        .iter()
        .find(|r| r.issue_number == 60)
        .unwrap();
    assert_eq!(r60.gate_status, ExecutionClass::BlockedOnArtifacts);
    let page = r60
        .pinned_sources
        .iter()
        .find(|s| s.label == "project_page_html_only_repo")
        .unwrap();
    assert_eq!(
        page.commit_sha.as_deref(),
        Some("23876839517155db4333fb08d26b8b118a3db2e1")
    );
    assert!(!page.runnable_implementation);
    assert!(r60
        .preserved_distinct_owners
        .contains(&"OnixResearch/lattice".to_string()));
}

#[test]
fn gate_validator_rejects_default_on_verified_claims_or_empty_repo_pins() {
    let mut bad = default_records();
    bad[0].default_off = false;
    bad[0].claims_verified = true;
    bad[0].pinned_sources[1].runnable_implementation = true; // empty repo without SHA!
    bad[1].pinned_sources[2].runnable_implementation = true; // undeclared license!

    let res = validate_related_work_gates(&bad);
    assert!(!res.gate_validation_passed);
    assert!(res
        .validation_errors
        .iter()
        .any(|e| e.contains("default_off")));
    assert!(res
        .validation_errors
        .iter()
        .any(|e| e.contains("unverified_author_claims")));
    assert!(res
        .validation_errors
        .iter()
        .any(|e| e.contains("without_sha")));
    assert!(res
        .validation_errors
        .iter()
        .any(|e| e.contains("undeclared_license")));
}
