use inference_lab::kv_routing::{run_default_harness, RoutingPolicy};
use inference_lab::provenance::{ExecutionClass, OutcomeStatus};

#[test]
fn kv_routing_rejects_same_config_different_weights_and_enforces_fencing() {
    let report = run_default_harness();
    assert_eq!(
        report.execution_class,
        ExecutionClass::BoundedContractHarness
    );

    let by_id = |id: &str| {
        report
            .compatibility_controls
            .iter()
            .find(|c| c.case_id == id)
            .unwrap()
    };

    assert!(by_id("positive-exact-same-variant").compatible);
    assert!(by_id("positive-exact-same-variant").credits_intended_variant);

    // Quantization collapse observation is preserved explicitly while still applying full contract.
    let q_obs = by_id("observation-quantization-collapse-same-effective-weights");
    assert!(q_obs.compatible);
    assert!(q_obs.quantization_collapse_observed);

    // RSI #86 same-config different-weights, adapter swap, stale generation, cross-arch, and TT #46 translated parity fail.
    for id in [
        "negative-rsi86-same-config-different-weights",
        "negative-adapter-replacement",
        "negative-stale-deployment-generation",
        "negative-heterogeneous-sm100a-to-sm121",
        "negative-tt46-translated-state-parity-fail-despite-task-pass",
    ] {
        let c = by_id(id);
        assert!(!c.compatible, "{id} must be rejected");
        assert!(!c.credits_intended_variant);
        assert!(!c.mismatch_reasons.is_empty());
    }

    // Soft affinity load escape and all 4 policy arms check compatibility before affinity.
    assert_eq!(report.policy_comparisons.len(), 4);
    for p in &report.policy_comparisons {
        assert!(p.compatibility_checked_before_affinity);
        assert!(p.post_dispatch_duplicate_prevented);
    }
    let soft = report
        .policy_comparisons
        .iter()
        .find(|p| p.policy == RoutingPolicy::BoundedSoftAffinityLoadEscape)
        .unwrap();
    assert!(soft.escaped_saturated_owner);
    assert!(soft.pre_dispatch_reconsideration);

    // ChaosControl separates safe cache-miss fallback from fatal consistency/transport/OOM failures.
    let fence = report
        .chaos_fault_scenarios
        .iter()
        .find(|f| f.fault_id == "chaos-redis-ownership-fence-violated")
        .unwrap();
    assert!(!fence.safe_cache_miss_fallback);
    assert_eq!(fence.outcome, OutcomeStatus::Invalid);
    assert!(fence.silent_fallback_prevented);
}
