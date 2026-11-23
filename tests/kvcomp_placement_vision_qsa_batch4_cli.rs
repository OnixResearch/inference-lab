use std::fs;
use std::process::Command;

use inference_lab::expert_placement::{
    run_default_harness as run_expert_placement, validate_host_memory_claim, HostMemoryClaim,
};
use inference_lab::json::sha256_hex;
use inference_lab::kv_compression::{
    check_grounded_end_task, run_default_harness as run_kv_compression, GroundedEndTask,
    StructuredAnswer,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::qwen38_qsa::{
    check_qsa_task, run_default_harness as run_qwen38_qsa, LongContextTaskKind,
};
use inference_lab::vision_infusion::{
    run_default_harness as run_vision_infusion, validate_matched_visual_budget,
};

#[test]
fn kv_compression_verifies_identity_parity_peak_memory_and_grounded_end_task_controls() {
    let report = run_kv_compression().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.identity_control_max_err <= 1e-12);
    assert_eq!(report.matrix_rows.len(), 12); // 3 context lengths * 4 arms

    // Direct grounded checker test rejects superficial substring decoy.
    let task = GroundedEndTask {
        task_id: "t".into(),
        decisive_fact_pos: 5,
        required_answer_value: "v=2".into(),
        required_citation_id: "c2".into(),
        distractor_substring: "v=1".into(),
    };
    let bad = StructuredAnswer {
        answer_value: "v=1".into(),
        cited_fact_id: "c1".into(),
        raw_text: "not v=2, actually v=1".into(),
    };
    assert_eq!(check_grounded_end_task(&task, &bad), OutcomeStatus::Invalid);

    assert_eq!(report.negative_controls.len(), 7);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "KV compression control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn expert_placement_verifies_warm_states_counterbalanced_paired_reducer_and_controls() {
    let report = run_expert_placement().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.simulations.len(), 8); // 4 policies * (favorable + unfavorable)
    assert!(report.paired_reduction.counterbalanced);
    assert_eq!(report.paired_reduction.planned_pairs, 4);
    assert_eq!(report.paired_reduction.both_accepted_pairs, 3);
    assert_eq!(report.paired_reduction.failed_or_excluded_pairs, 1);

    assert!(validate_host_memory_claim(&HostMemoryClaim {
        claims_madvise_guarantees_reclamation: false,
        unified_memory_host: true,
        claims_cpu_gpu_swap_frees_physical_capacity: false,
    })
    .is_ok());

    // Dynamic frequency sizing does not panic on expert >= 16 or gpu_slots >= 16!
    let large_sim = inference_lab::expert_placement::simulate_expert_placement(
        inference_lab::expert_placement::PlacementPolicy::StrataAdaptivePlacement,
        "large-id-test",
        &[0, 24, 31, 24],
        18,
        true,
    );
    assert_eq!(large_sim.expert_hits + large_sim.expert_misses, 4);
    assert!(inference_lab::expert_placement::results_json(&report)
        .contains("\"synthetic_contract_fixture\":true"));

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Expert placement control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn vision_infusion_matches_principal_visual_budgets_and_catches_all_routing_mask_controls() {
    let report = run_vision_infusion().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.principal_arms_vision_budget_matched);
    assert_eq!(report.quality_status, "quality_unestablished");
    assert_eq!(
        report.trained_variants_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert_eq!(
        validate_matched_visual_budget(&report.arch_rows[1..]).unwrap(),
        1024
    );

    for r in &report.arch_rows {
        assert!(
            r.packed_vs_single_max_err <= 1e-12,
            "packed vs single drift on {}",
            r.arch.as_str()
        );
        assert!(
            r.text_only_invariance_max_err <= 1e-12,
            "text-only invariance drift on {}",
            r.arch.as_str()
        );
        assert!(
            r.zero_init_vs_encoder_free_max_err <= 1e-12,
            "zero-init branch parity drift on {}",
            r.arch.as_str()
        );
    }
    // Empty requests slice rejected cleanly instead of panicking!
    let enc_free = inference_lab::vision_infusion::MultimodalTinyModel::build(
        inference_lab::vision_infusion::VisionArch::EncoderFree,
        false,
    );
    assert!(inference_lab::vision_infusion::forward_packed_batch(
        &enc_free,
        &[],
        0,
        inference_lab::vision_infusion::FaultInjection::default()
    )
    .is_err());
    // Matched visual params do NOT imply identical FLOPs!
    assert_ne!(
        report.arch_rows[1].flops_img16_txt4,
        report.arch_rows[2].flops_img16_txt4
    );

    assert_eq!(report.negative_controls.len(), 7);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Vision infusion control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn qwen38_qsa_verifies_2x2_matrix_full_kv_retention_population_accounting_and_controls() {
    let report = run_qwen38_qsa().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(report.matrix_2x2.len(), 4);
    assert!(report.zero_accepted_cost_undefined);
    assert_eq!(report.population.planned_tasks, 5);
    assert_eq!(report.population.dense_accepted, 4);
    assert_eq!(report.population.qsa_accepted, 3);
    assert_eq!(report.population.sparse_induced_losses, 1);
    assert_eq!(report.population.both_failed_not_retained_capability, 1);

    // Direct checker & validation guards (duplicate task_id, empty planned, NaN cost, d==0).
    assert_eq!(
        check_qsa_task(
            LongContextTaskKind::ExactLookupWithDecoys,
            "TOKEN-7F3A-EXACT"
        ),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        check_qsa_task(
            LongContextTaskKind::ExactLookupWithDecoys,
            "TOKEN-7F3A-FAKE"
        ),
        OutcomeStatus::Invalid
    );
    assert!(inference_lab::qwen38_qsa::rescore_qsa_population(0, &[]).is_err());
    assert!(inference_lab::qwen38_qsa::results_json(&report)
        .contains("\"synthetic_contract_fixture\":true"));

    assert_eq!(report.negative_controls.len(), 8);
    for c in &report.negative_controls {
        assert_ne!(
            c.outcome,
            OutcomeStatus::Accepted,
            "Qwen3.8 QSA control {} must reject",
            c.case_id
        );
    }
}

#[test]
fn batch4_cli_commands_and_suite_are_byte_identical_leak_free_and_refuse_overwrite() {
    let bin = env!("CARGO_BIN_EXE_inference-lab");
    let base = std::env::temp_dir().join(format!(
        "inference-lab-batch4-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let suite1 = base.join("suite-run1");
    let suite2 = base.join("suite-run2");

    for dir in [&suite1, &suite2] {
        let out = Command::new(bin)
            .args(["batch4-suite", "--out", dir.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "batch4-suite failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // Overwriting an existing directory must fail cleanly.
    let overwrite = Command::new(bin)
        .args(["batch4-suite", "--out", suite1.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!overwrite.status.success());

    // Single subcommand smoke check & overwrite refusal.
    let single = base.join("single-mtp");
    let s_out = Command::new(bin)
        .args(["mtp-decoder", "--out", single.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(s_out.status.success());
    assert_eq!(fs::read(single.join("COMPLETE")).unwrap(), b"");

    let subdirs = [
        "issue-22-mtp-decoder",
        "issue-23-flash-topk",
        "issue-24-macaq-r",
        "issue-25-hetero-rollout",
        "issue-26-nvfp4-mxfp8",
        "issue-27-spark-engines",
        "issue-28-kv-compression",
        "issue-29-expert-placement",
        "issue-30-vision-infusion",
        "issue-31-qwen38-qsa",
    ];

    let manifest = fs::read_to_string(suite1.join("manifest.json")).unwrap();
    assert_eq!(
        manifest,
        fs::read_to_string(suite2.join("manifest.json")).unwrap()
    );
    assert!(!manifest.contains("/home/"));

    for top in [
        "inputs.json",
        "results.json",
        "summary.md",
        "manifest.json",
        "COMPLETE",
    ] {
        let b1 = fs::read(suite1.join(top)).unwrap();
        let b2 = fs::read(suite2.join(top)).unwrap();
        assert_eq!(b1, b2, "top-level {top} not byte-identical");
        if top == "COMPLETE" {
            assert!(b1.is_empty());
        } else {
            let txt = String::from_utf8(b1.clone()).unwrap();
            assert!(!txt.contains("/home/"), "leak in top-level {top}");
            if top != "manifest.json" {
                assert!(manifest.contains(&sha256_hex(&b1)));
            }
        }
    }

    for sub in subdirs {
        for f in ["inputs.json", "results.json", "summary.md", "COMPLETE"] {
            let b1 = fs::read(suite1.join(sub).join(f)).unwrap();
            let b2 = fs::read(suite2.join(sub).join(f)).unwrap();
            assert_eq!(b1, b2, "{sub}/{f} not byte-identical");
            if f == "COMPLETE" {
                assert!(b1.is_empty());
            } else {
                let txt = String::from_utf8(b1.clone()).unwrap();
                assert!(!txt.contains("/home/"), "leak in {sub}/{f}");
                assert!(manifest.contains(&sha256_hex(&b1)));
            }
        }
    }

    let _ = fs::remove_dir_all(&base);
}
