# Issue #25: Heterogeneous Rollout Serving & Versioned HTTP LoRA Qualification

- Execution class: `bounded_contract_harness`
- Hardware gate: `unavailable_hardware_runtime` (`physical_two_backend_nvidia_amd_rollout_pool`)
- Decision: `INCONCLUSIVE_PHYSICAL_TWO_BACKEND_QUALIFICATION_REQUIRED`

## Fault & fleet-reconciliation controls
- `wrong_base_model_metadata` -> `invalid` (eligible_after=`false`, health=rejected_install_old_intact)
- `corrupt_or_truncated_payload` -> `invalid` (eligible_after=`false`, health=rejected_install_old_intact)
- `duplicate_delivery_idempotent_no_reactivation` -> `accepted` (eligible_after=`true`, deliveries=4, activations=3)
- `stale_deployment_generation` -> `invalid` (eligible_after=`false`, health=rejected_install_old_intact)
- `unavailable_engine_partial_fleet_update` -> `internal_failure` (eligible_after=`false`, health=unavailable)
- `interrupted_activation_uncertain_ineligible` -> `internal_failure` (eligible_after=`false`, health=uncertain_ineligible)
- `late_old_generation_response_rejected` -> `invalid` (eligible_after=`true`, response_gen=2 vs current_gen=3)
- `failed_parent_restoration_ineligible` -> `internal_failure` (eligible_after=`false`, health=uncertain_ineligible)
- `unsupported_multi_lora_and_disaggregation` -> `unsupported` (eligible_after=`true`, multi_lora=unsupported, disaggregated_prefill_decode=unsupported)
- `echoed_version_spoof_and_token_misalignment` -> `invalid` (eligible_after=`false`, probe_spoof_outcome=invalid, tok_mismatch_outcome=invalid)
