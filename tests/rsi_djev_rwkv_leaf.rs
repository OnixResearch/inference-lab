use inference_lab::djev_protocol::{
    base_golden_request, compute_effective_cache_identity, encode_canonical_request_bytes,
    evaluate_djev_response, run_default_harness as run_djev, validate_dual_consumer_gate,
    BoundedResponseInput, CacheIdentityPolicy, DjevCapability, SpanOffset,
};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};
use inference_lab::rsi_bench::{
    evaluate_rsi82_sweep, fake_quant_int8_slice, run_default_harness as run_rsi_bench,
    rwkv7_forward_f64, synthetic_rwkv7_sequence, validate_encoder_phase_separation,
    validate_fake_quant_claim, validate_optional_runtime_availability,
    validate_public_publication_allowlist, validate_rsi18_receipt_immutability,
    Rsi82SweepQualificationInput, Rwkv7StateF64,
};
use inference_lab::rwkv_leaf::{
    evaluate_leaf_trigger, run_default_harness as run_rwkv_leaf,
    validate_checkpoint_version_policy, validate_precision_interchange_claim,
    validate_rsi_wrapper_contract, validate_safetensors_admission, BytePrefixVocab,
    LeafConsumerCandidate, RwkvReusableWorkspace, SafetensorsAdmissionInput, TensorHeaderEntry,
};

#[test]
fn rsi_bench_enforces_rsi82_gate_recurrence_invariances_and_fake_quant_honesty() {
    let report = run_rsi_bench().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );
    assert!(!report.fake_quant_implies_int8_speedup);
    assert!(report.historical_rsi18_preserved_immutable);
    assert_eq!(report.donor_surfaces.len(), 5);
    assert!(report.fake_quant_max_abs_err > 0.0 && report.fake_quant_max_abs_err.is_finite());

    // RSI #82 verdicts: positive passes, all 3 negative controls fail.
    assert!(report.rsi82_verdicts[0].qualified_baseline);
    for neg in &report.rsi82_verdicts[1..] {
        assert!(!neg.qualified_baseline);
        assert_eq!(neg.outcome, OutcomeStatus::Invalid);
    }

    // Direct negative checks: NaN and negative parity errors fail RSI #82.
    for bad_err in [f64::NAN, -0.5] {
        let bad_verdict = evaluate_rsi82_sweep(&Rsi82SweepQualificationInput {
            case_id: "bad-err".into(),
            precomputed_equivalent_inputs: true,
            output_max_abs_err: bad_err,
            final_state_max_abs_err: 0.0,
            tolerance: 1e-12,
            rng_or_alloc_inside_timed_region: false,
            warmups_completed: 2,
            synchronized_wall_and_device_timing: true,
            retains_prior_sweep_as_superseded: true,
        });
        assert!(!bad_verdict.qualified_baseline);
        assert_eq!(bad_verdict.outcome, OutcomeStatus::Invalid);
    }

    // Exercise deterministic consolidation validators directly (positive + negative).
    assert_eq!(
        validate_encoder_phase_separation(false, 4.2),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_encoder_phase_separation(true, 4.2),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_fake_quant_claim(true, false),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_fake_quant_claim(true, true),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_rsi18_receipt_immutability("r18", "r19", false),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_rsi18_receipt_immutability("r18", "r18", false),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_public_publication_allowlist(false),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_public_publication_allowlist(true),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_optional_runtime_availability(true, false),
        OutcomeStatus::UnavailableHardwareRuntime
    );

    // Non-finite input in RWKV-7 recurrence and fake_quant_int8_slice must error cleanly.
    let mut bad_seq = synthetic_rwkv7_sequence(1, 2, 1);
    bad_seq[0].decay_w[0][0] = f64::NAN;
    let st = Rwkv7StateF64::zeros(1, 2).unwrap();
    assert!(rwkv7_forward_f64(&st, &bad_seq).is_err());
    assert!(fake_quant_int8_slice(&[1.0, f64::INFINITY]).is_err());

    // Check key controls in report.
    let by_id = |id: &str| report.controls.iter().find(|c| c.control_id == id).unwrap();
    assert_eq!(
        by_id("01-positive-rwkv7-chunked-output-and-state-parity").outcome,
        OutcomeStatus::Accepted
    );
    assert_eq!(
        by_id("02-negative-altered-transform-fails-parity").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("05-negative-fake-quant-claimed-as-int8-speedup-rejected").outcome,
        OutcomeStatus::Invalid
    );
    assert_eq!(
        by_id("08-control-missing-gpu-or-fla-stays-unavailable").outcome,
        OutcomeStatus::UnavailableHardwareRuntime
    );
}

#[test]
fn djev_protocol_preserves_rsi42_cache_identity_wire_bytes_and_whole_decision_deadlines() {
    let report = run_djev().unwrap();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert!(report.dual_consumer_gate_satisfied);
    assert_ne!(report.golden_ascii_sha256, report.golden_non_ascii_sha256);

    // Identical effective read reuses cache; every mutation invalidates it.
    assert!(!report.cache_regressions[0].invalidated_prior_cache);
    for mut_check in &report.cache_regressions[1..] {
        assert!(
            mut_check.invalidated_prior_cache,
            "{} must invalidate prior cache",
            mut_check.mutation_name
        );
    }

    // Missing immutable model digest must produce NonReusableRunNamespace (never invented digest),
    // and empty run_namespace is rejected!
    let mut no_digest = base_golden_request();
    no_digest.immutable_model_digest = None;
    let pol = compute_effective_cache_identity(&no_digest, "ephemeral-42").unwrap();
    assert!(!pol.is_reusable());
    assert!(matches!(
        pol,
        CacheIdentityPolicy::NonReusableRunNamespace(_)
    ));
    assert_eq!(
        compute_effective_cache_identity(&no_digest, "   "),
        Err(OutcomeStatus::Invalid)
    );

    // Dual-consumer gate rejects single or duplicate consumer, accepts two distinct consumers.
    assert!(!validate_dual_consumer_gate(&["rsi-only"]));
    assert!(!validate_dual_consumer_gate(&["rsi-only", "rsi-only"]));
    assert!(validate_dual_consumer_gate(&[
        "rsi",
        "inference-lab-worker"
    ]));

    // expected_choices_len == 0 is rejected cleanly.
    let zero_choices = evaluate_djev_response(&BoundedResponseInput {
        case_id: "zero-choices".into(),
        requested_capability: DjevCapability::StructuredReadScoring,
        supported_capabilities: vec![DjevCapability::StructuredReadScoring],
        raw_response_bytes: 64,
        max_response_bytes: 1024,
        transport_ok: true,
        response_json_well_formed: true,
        schema_valid: true,
        choice_scores: vec![],
        expected_choices_len: 0,
        whole_decision_deadline_ms: 500,
        queue_wait_ms: 10,
        transport_elapsed_ms: 10,
        cancelled: false,
        stale_decision_epoch: false,
        queue_saturated: false,
    });
    assert_eq!(zero_choices.outcome, OutcomeStatus::Invalid);

    // Mid-UTF-8 span offset and unsupported wire version rejected cleanly.
    let mut bad_span = base_golden_request();
    bad_span.prompt_text = "ééé".into();
    bad_span.read_span = SpanOffset {
        start_byte: 1,
        end_byte: 4,
    };
    assert_eq!(
        encode_canonical_request_bytes(&bad_span),
        Err(OutcomeStatus::Invalid)
    );

    // Whole-decision deadline must sum queue_wait + transport_elapsed (never reset by transport).
    let deadline_verdict = evaluate_djev_response(&BoundedResponseInput {
        case_id: "queue-plus-transport-timeout".into(),
        requested_capability: DjevCapability::StructuredReadScoring,
        supported_capabilities: vec![DjevCapability::StructuredReadScoring],
        raw_response_bytes: 128,
        max_response_bytes: 1024,
        transport_ok: true,
        response_json_well_formed: true,
        schema_valid: true,
        choice_scores: vec![-0.1, -0.2],
        expected_choices_len: 2,
        whole_decision_deadline_ms: 200,
        queue_wait_ms: 150,
        transport_elapsed_ms: 80,
        cancelled: false,
        stale_decision_epoch: false,
        queue_saturated: false,
    });
    assert_eq!(deadline_verdict.outcome, OutcomeStatus::Timeout);
    assert!(!deadline_verdict.grants_generative_or_proof_authority);

    // Never grants generative or proof authority across all response controls.
    for r in &report.response_controls {
        assert!(!r.grants_generative_or_proof_authority);
    }
    assert_eq!(report.response_controls[0].outcome, OutcomeStatus::Accepted);
    assert_eq!(
        report.response_controls[1].outcome,
        OutcomeStatus::Unsupported
    );
    assert_eq!(
        report.response_controls[2].outcome,
        OutcomeStatus::Unsupported
    );
}

#[test]
fn rwkv_leaf_stays_blocked_on_artifacts_and_verifies_trie_safetensors_and_zero_realloc_workspace() {
    let report = run_rwkv_leaf().unwrap();
    assert_eq!(report.execution_class, ExecutionClass::BlockedOnArtifacts);
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::BlockedOnArtifacts
    );
    assert!(!report.fp32_cpu_interchangeable_with_fla_bf16_cuda);
    assert!(report.trie_roundtrip_exact);
    assert!(report.generic_whitespace_diverges);
    assert_eq!(report.workspace_reallocations, 0);
    assert_eq!(report.workspace_steps_executed, 6);

    // Trigger checks: current state & mock adapter rejected; two concrete non-mock consumers accepted.
    assert!(!report.trigger_verdicts[0].trigger_satisfied);
    assert!(!report.trigger_verdicts[1].trigger_satisfied);
    assert!(report.trigger_verdicts[2].trigger_satisfied);

    let single = evaluate_leaf_trigger(
        "single-consumer",
        &[LeafConsumerCandidate {
            consumer_id: "rsi".into(),
            concrete_call_site: Some("rwkv7_backend.rs".into()),
            is_live_non_mock_consumer: true,
            pins_leaf_revision: true,
        }],
    );
    assert!(!single.trigger_satisfied);

    // Byte-prefix trie roundtrip on arbitrary UTF-8 + binary bytes.
    let vocab = BytePrefixVocab::pinned_fixture();
    let raw = b"rwkv7\x00\xff state\n\nstep";
    let ids = vocab.encode(raw).unwrap();
    assert_eq!(vocab.decode(&ids).unwrap(), raw);

    // Workspace rejects NaN input without panicking.
    let mut ws = RwkvReusableWorkspace::with_dim(2);
    let mut st = [0.0_f32; 2];
    let mut out = [0.0_f32; 2];
    assert!(ws
        .step_in_place(&mut st, &[f32::NAN, 1.0], 0.9, &mut out)
        .is_err());

    // Safetensors admission catches unsupported dtype, empty names, and duplicate names cleanly.
    let d_x = inference_lab::provenance::DigestRef::sha256(b"x");
    let (status, _) = validate_safetensors_admission(&SafetensorsAdmissionInput {
        case_id: "unsupported-dtype".into(),
        header_bytes: 128,
        max_header_bytes: 1024,
        payload_bytes: 64,
        file_immutable_for_map_lifetime: true,
        expected_model_digest: d_x.clone(),
        observed_model_digest: d_x.clone(),
        tensors: vec![TensorHeaderEntry {
            name: "t".into(),
            dtype: "FP8_UNKNOWN".into(),
            shape: vec![4],
            offset_start: 0,
            offset_end: 4,
        }],
    });
    assert_eq!(status, OutcomeStatus::Unsupported);

    let (dup_status, _) = validate_safetensors_admission(&SafetensorsAdmissionInput {
        case_id: "duplicate-name".into(),
        header_bytes: 128,
        max_header_bytes: 1024,
        payload_bytes: 64,
        file_immutable_for_map_lifetime: true,
        expected_model_digest: d_x.clone(),
        observed_model_digest: d_x,
        tensors: vec![
            TensorHeaderEntry {
                name: "w".into(),
                dtype: "F32".into(),
                shape: vec![4],
                offset_start: 0,
                offset_end: 16,
            },
            TensorHeaderEntry {
                name: "w".into(),
                dtype: "F32".into(),
                shape: vec![4],
                offset_start: 16,
                offset_end: 32,
            },
        ],
    });
    assert_eq!(dup_status, OutcomeStatus::Invalid);

    // Wrapper/policy validators.
    assert_eq!(
        validate_rsi_wrapper_contract(true, true, true, false),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_checkpoint_version_policy(true, true),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_checkpoint_version_policy(true, false),
        OutcomeStatus::Invalid
    );
    assert_eq!(
        validate_precision_interchange_claim(false),
        OutcomeStatus::Accepted
    );
    assert_eq!(
        validate_precision_interchange_claim(true),
        OutcomeStatus::Invalid
    );
}
