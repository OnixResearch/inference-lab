use inference_lab::provenance::ExecutionClass;
use inference_lab::trace::{run_default_harness, TraceMode};

#[test]
fn trace_positive_control_joins_scoped_identities_and_leaves_batched_attribution_unavailable() {
    let report = run_default_harness();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );
    assert_eq!(
        report.hardware_gate.status,
        ExecutionClass::UnavailableHardwareRuntime
    );

    let pos = &report.controls[0];
    assert_eq!(pos.control_name, "positive-known-kernel-copy-sync");
    assert!(pos.causal_path_substantiated);
    assert!(pos.errors.is_empty());
    assert!(pos.unresolved_metadata_ids.is_empty());
    assert_eq!(pos.joined_rows.len(), 3);

    // Kernel 2 has CUDA Graph ID 9001 and resolved deferred metadata.
    let graph_row = &pos.joined_rows[1];
    assert_eq!(graph_row.cuda_graph_id, Some(9001));
    assert_eq!(
        graph_row.metadata_label.as_deref(),
        Some("nvtx:decode_step_0")
    );
    assert_eq!(graph_row.launch_to_device_delta_ns, Some(4_000));
    assert_eq!(graph_row.uncertainty_ns, Some(250));

    // Kernel 3 reuses correlation_id=102 in context 2 with batch=4: joined to ctx 2 host event,
    // and per-request attribution stays explicitly unavailable!
    let batched_row = &pos.joined_rows[2];
    assert_eq!(
        batched_row.host_event_id.as_deref(),
        Some("host-ctx2-batched-3")
    );
    assert_eq!(
        batched_row.request_attribution,
        "unavailable_batched_multi_request"
    );
    assert!(!pos.unsupported_conclusions.is_empty());

    // Timeline preserves all 5 distinct cross-layer event kinds with units and calibrated uncertainty.
    let kinds: Vec<&str> = pos.timeline.iter().map(|t| t.event_kind.as_str()).collect();
    for expected in [
        "host_api_enqueue",
        "device_execution_or_copy",
        "linux_sched",
        "request_phase",
        "client_sse_arrival",
    ] {
        assert!(
            kinds.contains(&expected),
            "timeline missing event_kind {expected}"
        );
    }
}

#[test]
fn trace_all_seven_negative_controls_block_causal_substantiation() {
    let report = run_default_harness();
    assert_eq!(report.controls.len(), 8);

    for neg in &report.controls[1..] {
        assert!(
            !neg.causal_path_substantiated,
            "negative control {} must NOT substantiate a causal path",
            neg.control_name
        );
        assert_eq!(neg.status, "incomplete_cannot_substantiate");
        assert!(
            !neg.errors.is_empty() || !neg.unresolved_metadata_ids.is_empty(),
            "negative control {} must record explicit error or unresolved metadata",
            neg.control_name
        );
    }

    // Verify specific failure detection on key controls:
    let clock_neg = report
        .controls
        .iter()
        .find(|c| c.control_name == "neg-unmapped-clock-domain")
        .unwrap();
    assert!(clock_neg
        .errors
        .iter()
        .any(|e| e.contains("missing_clock_mapping_from_device_global_timer_ns")));
    assert!(clock_neg
        .errors
        .iter()
        .any(|e| e.contains("missing_clock_mapping_from_client_monotonic_ns")));

    let dup_neg = report
        .controls
        .iter()
        .find(|c| c.control_name == "neg-duplicate-out-of-order-and-collision")
        .unwrap();
    assert!(dup_neg
        .warnings
        .iter()
        .any(|w| w.starts_with("duplicate_host_event_id:")));
    assert!(dup_neg
        .warnings
        .iter()
        .any(|w| w.starts_with("out_of_order_host_event:")));

    let missing_drop = report
        .controls
        .iter()
        .find(|c| c.control_name == "neg-missing-drop-counter-not-zero")
        .unwrap();
    assert!(missing_drop
        .errors
        .iter()
        .any(|e| e == "dropped_events_counter_missing_data"));

    let late_meta = report
        .controls
        .iter()
        .find(|c| c.control_name == "neg-late-metadata-unresolved")
        .unwrap();
    assert_eq!(late_meta.unresolved_metadata_ids, vec!["meta-late-99"]);
}

#[test]
fn perturbation_matrix_ranks_untraced_only() {
    let report = run_default_harness();
    assert_eq!(report.perturbation_matrix.len(), 4);
    for row in &report.perturbation_matrix {
        if row.mode == TraceMode::Untraced {
            assert!(row.ranking_eligible);
        } else {
            assert!(!row.ranking_eligible);
        }
    }
}
