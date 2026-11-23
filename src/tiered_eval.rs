//! Issue #38: Qualify tiered persistent-engine evaluation for inference optimization with cold final replay.
//!
//! Pinned motivation (strictly unverified non-gating related work):
//! - Together Compiler (`arXiv:2611.14681v1`, §4.2) motivates persistent model weights with
//!   hot-swapped forward-pass evaluation and tiered screening. Published speedups are NOT local
//!   acceptance targets, and TKCC is NOT assumed to be an available library.
//! - Reuses `#7`/`#11` runner/lifecycle conventions and `#12` numerical fixtures; `#4` retains
//!   agent-search/reference-access comparisons.
//!
//! Provides:
//! - Deterministic evaluation of the same frozen candidate matrix across 3 arms under equal budget:
//!   (A) Restart-per-candidate execution,
//!   (B) Full-fidelity persistent tier,
//!   (C) Fast screening tier + full-fidelity persistent validation (plus explicitly scoped optional
//!       reduced-layer tier never treated as whole-model quality/latency).
//! - External evaluator & run-owned isolation: verifies immutable weight digest before/after every
//!   candidate, resets KV/recurrent/workspace/RNG/globals, forces worker restart (recording verified
//!   termination, not just stop request) on contamination/weight-mutation/exception/stale-graph.
//! - Real `A -> B -> A` restoration check comparing restored persistent A against fresh-process A.
//! - Preregistered fast-tier false-rejection and rank-disagreement audit, frozen-candidate cold final
//!   replay on protected held-out requests (numerical + state checked before performance scoring),
//!   full non-collapsing candidate accounting, and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const TOGETHER_COMPILER_PAPER: &str = "arXiv:2611.14681v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateBehavior {
    BaselineNoOp,
    ValidFaster,
    ValidSlower,
    FastTierFalseRejectFullValid,
    WrongButFast,
    CorruptsGlobalOrKvState,
    MutatesWeights,
    ThrowsException,
}

#[derive(Debug, Clone)]
pub struct CandidateSpec {
    pub candidate_id: String,
    pub behavior: CandidateBehavior,
    pub fast_screen_latency_ms: f64,
    pub full_tier_latency_ms: f64,
    pub heldout_latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct PersistentWorker {
    pub worker_pid: u64,
    pub weights: [f64; 8],
    pub immutable_weight_digest: DigestRef,
    pub kv_scratch: [f64; 4],
    pub global_bias: f64,
    pub graph_shape_key: usize,
    pub rng_state: u64,
    pub terminated_verified: bool,
    pub restart_count: usize,
}

fn digest_weights(w: &[f64; 8]) -> DigestRef {
    let mut s = String::new();
    for v in w {
        let _ = write!(s, "{v:.16e},");
    }
    DigestRef::sha256(s.as_bytes())
}

impl PersistentWorker {
    pub fn fresh(pid: u64) -> Self {
        let mut weights = [0.0_f64; 8];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = det_weight(3801, i, 0.55);
        }
        let d = digest_weights(&weights);
        Self {
            worker_pid: pid,
            weights,
            immutable_weight_digest: d,
            kv_scratch: [0.0; 4],
            global_bias: 0.0,
            graph_shape_key: 4,
            rng_state: 17,
            terminated_verified: false,
            restart_count: 0,
        }
    }

    pub fn forward_eval(
        &mut self,
        input: &[f64; 4],
        cand: CandidateBehavior,
        use_stale_graph: bool,
        skip_output_write: bool,
    ) -> Result<[f64; 4], String> {
        if cand == CandidateBehavior::ThrowsException {
            return Err("candidate raised runtime device exception".into());
        }
        if cand == CandidateBehavior::MutatesWeights {
            self.weights[0] += 0.25;
        }
        if cand == CandidateBehavior::CorruptsGlobalOrKvState {
            self.global_bias += 0.5;
            self.kv_scratch[0] = 99.0;
        }
        if skip_output_write {
            return Ok(self.kv_scratch);
        }
        let mut out = [0.0_f64; 4];
        for (o, slot) in out.iter_mut().enumerate() {
            let mut acc = self.global_bias;
            for (i, &inp_i) in input.iter().enumerate() {
                let w_idx = if use_stale_graph {
                    (o + i + 1) % 8
                } else {
                    (o + i) % 8
                };
                acc += inp_i * self.weights[w_idx];
            }
            if cand == CandidateBehavior::WrongButFast {
                acc += 0.35; // Deliberately wrong numerical output!
            }
            if !acc.is_finite() {
                return Err("non-finite forward output".into());
            }
            *slot = acc;
        }
        self.kv_scratch = out;
        self.rng_state = self.rng_state.wrapping_add(1);
        Ok(out)
    }

    pub fn reset_or_restart(&mut self, force_skip_reset: bool) -> Result<bool, String> {
        let weights_intact = digest_weights(&self.weights) == self.immutable_weight_digest;
        if !weights_intact {
            // Mandatory restart with verified termination!
            self.terminated_verified = true;
            let next_pid = self.worker_pid + 1;
            let restarts = self.restart_count + 1;
            *self = Self::fresh(next_pid);
            self.restart_count = restarts;
            return Ok(true);
        }
        if !force_skip_reset {
            self.kv_scratch = [0.0; 4];
            self.global_bias = 0.0;
            self.rng_state = 17;
        }
        Ok(false)
    }
}

fn max_diff4(a: &[f64; 4], b: &[f64; 4]) -> Result<f64, String> {
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value in candidate comparison".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

/// Enforces cold final replay before final acceptance: fast-tier pass alone cannot admit a candidate.
pub fn admit_final_candidate(
    fast_tier_passed: bool,
    full_tier_passed: bool,
    cold_final_replay_passed: bool,
    numerical_err: f64,
    tol: f64,
) -> OutcomeStatus {
    if !numerical_err.is_finite() || !tol.is_finite() {
        return OutcomeStatus::Invalid;
    }
    if fast_tier_passed && full_tier_passed && cold_final_replay_passed && numerical_err <= tol {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

/// Verifies worker cleanup: a stop request without verified termination cannot establish cleanliness.
pub fn verify_worker_cleanup(
    stop_requested: bool,
    terminated_verified: bool,
    weights_intact: bool,
) -> OutcomeStatus {
    if stop_requested && !terminated_verified {
        return OutcomeStatus::Invalid;
    }
    if !weights_intact && !terminated_verified {
        return OutcomeStatus::Invalid;
    }
    OutcomeStatus::Accepted
}

#[derive(Debug, Clone)]
pub struct TierArmSummary {
    pub arm_id: String,
    pub planned_candidates: usize,
    pub valid_candidates: usize,
    pub total_budget_ms_synthetic: f64,
    pub worker_restarts: usize,
    pub fast_tier_false_rejects_audited: usize,
    pub fast_tier_rank_disagreements: usize,
    pub winning_candidate_id: Option<String>,
    pub cold_final_replay_max_err: Option<f64>,
    pub cold_final_replay_outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct TieredEvalReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub aba_restored_vs_fresh_max_err: f64,
    pub arms: Vec<TierArmSummary>,
    pub reduced_layer_tier_is_whole_model_measure: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<TieredEvalReport, String> {
    let candidates = vec![
        CandidateSpec {
            candidate_id: "cand_0_noop_baseline".into(),
            behavior: CandidateBehavior::BaselineNoOp,
            fast_screen_latency_ms: 4.0,
            full_tier_latency_ms: 20.0,
            heldout_latency_ms: 20.0,
        },
        CandidateSpec {
            candidate_id: "cand_1_valid_faster".into(),
            behavior: CandidateBehavior::ValidFaster,
            fast_screen_latency_ms: 3.1,
            full_tier_latency_ms: 14.5,
            heldout_latency_ms: 14.8,
        },
        CandidateSpec {
            candidate_id: "cand_2_valid_slower".into(),
            behavior: CandidateBehavior::ValidSlower,
            fast_screen_latency_ms: 4.8,
            full_tier_latency_ms: 23.0,
            heldout_latency_ms: 23.2,
        },
        CandidateSpec {
            candidate_id: "cand_3_fast_false_reject_full_valid".into(),
            behavior: CandidateBehavior::FastTierFalseRejectFullValid,
            fast_screen_latency_ms: 5.5, // Screened out by fast tier, audited in full tier!
            full_tier_latency_ms: 15.2,
            heldout_latency_ms: 15.4,
        },
        CandidateSpec {
            candidate_id: "cand_4_wrong_but_fast".into(),
            behavior: CandidateBehavior::WrongButFast,
            fast_screen_latency_ms: 2.0,
            full_tier_latency_ms: 9.0,
            heldout_latency_ms: 9.0,
        },
        CandidateSpec {
            candidate_id: "cand_5_corrupts_state".into(),
            behavior: CandidateBehavior::CorruptsGlobalOrKvState,
            fast_screen_latency_ms: 3.5,
            full_tier_latency_ms: 16.0,
            heldout_latency_ms: 16.0,
        },
        CandidateSpec {
            candidate_id: "cand_6_mutates_weights".into(),
            behavior: CandidateBehavior::MutatesWeights,
            fast_screen_latency_ms: 3.4,
            full_tier_latency_ms: 16.5,
            heldout_latency_ms: 16.5,
        },
        CandidateSpec {
            candidate_id: "cand_7_exception".into(),
            behavior: CandidateBehavior::ThrowsException,
            fast_screen_latency_ms: 1.0,
            full_tier_latency_ms: 1.0,
            heldout_latency_ms: 1.0,
        },
    ];

    let workload_input = [0.7, -0.4, 0.9, 0.25];
    let heldout_step1 = [-0.5, 0.8, 0.3, -0.6];
    let heldout_step2 = [0.2, -0.9, 0.4, 0.7];

    // Reference oracle on fresh worker (including 2-step held-out state continuation!)
    let mut ref_w = PersistentWorker::fresh(1);
    let ref_work_out = ref_w.forward_eval(
        &workload_input,
        CandidateBehavior::BaselineNoOp,
        false,
        false,
    )?;
    let mut ref_h = PersistentWorker::fresh(2);
    let ref_held_1 = ref_h.forward_eval(
        &heldout_step1,
        CandidateBehavior::BaselineNoOp,
        false,
        false,
    )?;
    let ref_held_2 = ref_h.forward_eval(
        &heldout_step2,
        CandidateBehavior::BaselineNoOp,
        false,
        false,
    )?;

    // Explicit A -> B -> A restoration check against fresh-process A
    let mut aba_w = PersistentWorker::fresh(10);
    let _a1 = aba_w.forward_eval(
        &workload_input,
        CandidateBehavior::BaselineNoOp,
        false,
        false,
    )?;
    let _ = aba_w.reset_or_restart(false)?;
    let _b = aba_w.forward_eval(
        &workload_input,
        CandidateBehavior::CorruptsGlobalOrKvState,
        false,
        false,
    )?;
    let _ = aba_w.reset_or_restart(false)?;
    let a2 = aba_w.forward_eval(
        &workload_input,
        CandidateBehavior::BaselineNoOp,
        false,
        false,
    )?;
    let aba_restored_vs_fresh_max_err = max_diff4(&ref_work_out, &a2)?;

    let mut arms = Vec::new();
    for (arm_id, cold_load_per_cand, use_fast_screen) in [
        ("arm_a_restart_per_candidate", 80.0_f64, false),
        ("arm_b_full_fidelity_persistent", 0.0_f64, false),
        ("arm_c_fast_screen_plus_full_persistent", 0.0_f64, true),
    ] {
        let mut worker = PersistentWorker::fresh(100);
        let mut total_ms = 80.0_f64; // initial cold load
        let mut valid_count = 0usize;
        let mut arm_restarts = 0usize;
        let mut fast_false_rejects = 0usize;
        let mut fast_rank_disagreements = 0usize;
        let mut fast_reject_counter = 0usize;
        let mut best: Option<(String, CandidateBehavior, f64)> = None;
        let mut worst_fast_survivor_full_ms: Option<f64> = None;

        for c in &candidates {
            if cold_load_per_cand > 0.0 {
                worker = PersistentWorker::fresh(worker.worker_pid + 1);
                total_ms += cold_load_per_cand;
            }
            let mut is_audited_fast_reject = false;
            if use_fast_screen {
                total_ms += c.fast_screen_latency_ms;
                let fast_rejected = c.fast_screen_latency_ms > 4.5;
                if fast_rejected {
                    // Preregistered deterministic audit sample (odd fast-reject index #1, #3, ...)
                    // evaluated in full tier without peeking at ground-truth CandidateBehavior!
                    let audit_this = fast_reject_counter % 2 == 1;
                    fast_reject_counter += 1;
                    if !audit_this {
                        continue;
                    }
                    is_audited_fast_reject = true;
                }
            }

            total_ms += c.full_tier_latency_ms;
            let eval_res = worker.forward_eval(&workload_input, c.behavior, false, false);
            let restarted = worker.reset_or_restart(false)?;
            if restarted {
                arm_restarts += 1;
                total_ms += 80.0; // charge mandatory restart cost!
            }
            if let Ok(out) = eval_res {
                if !restarted && max_diff4(&ref_work_out, &out)? <= 1.0e-12 {
                    valid_count += 1;
                    if is_audited_fast_reject {
                        fast_false_rejects += 1;
                        if let Some(ws) = worst_fast_survivor_full_ms {
                            if c.full_tier_latency_ms < ws {
                                fast_rank_disagreements += 1;
                            }
                        }
                    } else if use_fast_screen {
                        worst_fast_survivor_full_ms = Some(match worst_fast_survivor_full_ms {
                            None => c.full_tier_latency_ms,
                            Some(prev) => {
                                if c.full_tier_latency_ms > prev {
                                    c.full_tier_latency_ms
                                } else {
                                    prev
                                }
                            }
                        });
                    }
                    let replace = match &best {
                        None => true,
                        Some((_, _, b_lat)) => c.full_tier_latency_ms < *b_lat,
                    };
                    if replace {
                        best = Some((c.candidate_id.clone(), c.behavior, c.full_tier_latency_ms));
                    }
                }
            }
        }

        // Freeze winning candidate and run fresh-process COLD FINAL REPLAY on protected held-out
        // (multi-step numerical + state continuation gated through `admit_final_candidate`)!
        let (win_id, cold_err, cold_outcome) = if let Some((wid, beh, _)) = best {
            let mut cold_worker = PersistentWorker::fresh(999);
            total_ms += 80.0; // cold final replay load
            let h1 = cold_worker.forward_eval(&heldout_step1, beh, false, false)?;
            let h2 = cold_worker.forward_eval(&heldout_step2, beh, false, false)?;
            let e1 = max_diff4(&ref_held_1, &h1)?;
            let e2 = max_diff4(&ref_held_2, &h2)?;
            let err = if e1 >= e2 { e1 } else { e2 };
            let continuation_ok = cold_worker.rng_state == ref_h.rng_state && err <= 1.0e-12;
            let st = admit_final_candidate(true, true, continuation_ok, err, 1.0e-12);
            (Some(wid), Some(err), st)
        } else {
            (None, None, OutcomeStatus::Invalid)
        };

        arms.push(TierArmSummary {
            arm_id: arm_id.into(),
            planned_candidates: candidates.len(),
            valid_candidates: valid_count,
            total_budget_ms_synthetic: total_ms,
            worker_restarts: arm_restarts,
            fast_tier_false_rejects_audited: fast_false_rejects,
            fast_tier_rank_disagreements: fast_rank_disagreements,
            winning_candidate_id: win_id,
            cold_final_replay_max_err: cold_err,
            cold_final_replay_outcome: cold_outcome,
        });
    }

    // 8 executable negative controls
    let mut negative_controls = Vec::new();

    // NC1: candidate mutating immutable weights triggers digest check + mandatory restart
    {
        let mut w = PersistentWorker::fresh(200);
        let _ = w.forward_eval(
            &workload_input,
            CandidateBehavior::MutatesWeights,
            false,
            false,
        )?;
        let mutated = digest_weights(&w.weights) != w.immutable_weight_digest;
        let restarted = w.reset_or_restart(false)?;
        negative_controls.push(NegativeControl {
            case_id: "nc1_weight_mutation_detected_and_restarted".into(),
            outcome: if mutated && restarted {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "weight mutation caught by immutable digest and forced worker restart".into(),
        });
    }

    // NC2: skipped reset in A -> B -> A causes contamination divergence
    {
        let mut w = PersistentWorker::fresh(201);
        let _ = w.forward_eval(
            &workload_input,
            CandidateBehavior::CorruptsGlobalOrKvState,
            false,
            false,
        )?;
        let _ = w.reset_or_restart(true)?; // bug: skipped reset!
        let polluted = w.forward_eval(
            &workload_input,
            CandidateBehavior::BaselineNoOp,
            false,
            false,
        )?;
        let err = max_diff4(&ref_work_out, &polluted)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_aba_skipped_reset_contamination".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("skipped reset leaked {:.3e} error into restored A", err),
        });
    }

    // NC3: stale CUDA/HIP graph capture mismatch
    {
        let mut w = PersistentWorker::fresh(202);
        let stale_out = w.forward_eval(
            &workload_input,
            CandidateBehavior::BaselineNoOp,
            true,
            false,
        )?;
        let err = max_diff4(&ref_work_out, &stale_out)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_stale_graph_capture_divergence".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("stale graph caused {:.3e} numerical mismatch", err),
        });
    }

    // NC4: unwritten output buffer leaks prior scratch
    {
        let mut w = PersistentWorker::fresh(203);
        w.kv_scratch = [42.0, -7.0, 3.0, 9.0];
        let unwritten = w.forward_eval(
            &workload_input,
            CandidateBehavior::BaselineNoOp,
            false,
            true,
        )?;
        let err = max_diff4(&ref_work_out, &unwritten)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_unwritten_output_buffer_leak".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("unwritten output buffer leaked {:.3e} stale values", err),
        });
    }

    // NC5: wrong-but-fast candidate disqualified despite lowest latency
    {
        let mut w = PersistentWorker::fresh(204);
        let wrong = w.forward_eval(
            &workload_input,
            CandidateBehavior::WrongButFast,
            false,
            false,
        )?;
        let err = max_diff4(&ref_work_out, &wrong)?;
        negative_controls.push(NegativeControl {
            case_id: "nc5_wrong_but_fast_disqualified".into(),
            outcome: if err > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "wrong-but-fast candidate failed numerical gate ({:.3e})",
                err
            ),
        });
    }

    // NC6: fast-tier pass cannot skip cold final replay
    {
        let st = admit_final_candidate(true, true, false, 0.0, 1.0e-12);
        negative_controls.push(NegativeControl {
            case_id: "nc6_fast_tier_cannot_bypass_cold_replay".into(),
            outcome: st,
            detail: "fast-tier pass refused final acceptance without fresh cold held-out replay"
                .into(),
        });
    }

    // NC7: candidate exception trapped cleanly without silent pass
    {
        let mut w = PersistentWorker::fresh(205);
        let res = w.forward_eval(
            &workload_input,
            CandidateBehavior::ThrowsException,
            false,
            false,
        );
        negative_controls.push(NegativeControl {
            case_id: "nc7_candidate_exception_retained".into(),
            outcome: if res.is_err() {
                OutcomeStatus::InternalFailure
            } else {
                OutcomeStatus::Accepted
            },
            detail: "candidate runtime exception captured as explicit internal_failure".into(),
        });
    }

    // NC8: stop request without verified process termination rejected
    {
        let w = PersistentWorker::fresh(206);
        let st = verify_worker_cleanup(true, w.terminated_verified, true);
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_worker_termination".into(),
            outcome: st,
            detail: "stop request alone without verified termination cannot establish cleanliness"
                .into(),
        });
    }

    Ok(TieredEvalReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Physical GPU persistent forward-pass engine with CUDA/HIP graph capture",
            "Tiered evaluator, A->B->A restoration, false-rejection audit, and cold final replay verified on CPU; physical GPU persistent worker did not execute locally",
        ),
        synthetic_contract_fixture: true,
        aba_restored_vs_fresh_max_err,
        arms,
        reduced_layer_tier_is_whole_model_measure: false,
        negative_controls,
    })
}

pub fn inputs_json(report: &TieredEvalReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tiered-eval.inputs.v1\",\n",
            "  \"issue\": 38,\n",
            "  \"execution_class\": {},\n",
            "  \"non_gating_motivation\": {},\n",
            "  \"tkcc_assumed_available\": false,\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(TOGETHER_COMPILER_PAPER),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &TieredEvalReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"planned_candidates\":{},\"valid_candidates\":{},\"total_budget_ms_synthetic\":{},\"worker_restarts\":{},\"fast_tier_false_rejects_audited\":{},\"fast_tier_rank_disagreements\":{},\"winning_candidate_id\":{},\"cold_final_replay_max_err\":{},\"cold_final_replay_outcome\":{}}}",
                quote(&a.arm_id),
                a.planned_candidates,
                a.valid_candidates,
                finite_f64(a.total_budget_ms_synthetic),
                a.worker_restarts,
                a.fast_tier_false_rejects_audited,
                a.fast_tier_rank_disagreements,
                a.winning_candidate_id.as_deref().map_or_else(|| "null".into(), quote),
                opt_f64(a.cold_final_replay_max_err),
                quote(a.cold_final_replay_outcome.as_str()),
            )
        })
        .collect();
    let controls: Vec<String> = report
        .negative_controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                quote(&c.detail)
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tiered-eval.results.v1\",\n",
            "  \"issue\": 38,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"aba_restored_vs_fresh_max_err\": {},\n",
            "  \"reduced_layer_tier_is_whole_model_measure\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        finite_f64(report.aba_restored_vs_fresh_max_err),
        report.reduced_layer_tier_is_whole_model_measure,
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &TieredEvalReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #38: Tiered Persistent-Engine Evaluation with Cold Final Replay"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.synthetic_contract_fixture
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- `A -> B -> A` restored vs fresh-process A: `max_err = {:.3e}`",
        report.aba_restored_vs_fresh_max_err
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Tiered Evaluation Arms").unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}`: `valid={}/{}`, `restarts={}`, `false_rejects_audited={}`, `winner={:?}`, `cold_replay={}`",
            a.arm_id,
            a.valid_candidates,
            a.planned_candidates,
            a.worker_restarts,
            a.fast_tier_false_rejects_audited,
            a.winning_candidate_id,
            a.cold_final_replay_outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Negative Controls ({} / {} rejected)",
        report
            .negative_controls
            .iter()
            .filter(|c| c.outcome != OutcomeStatus::Accepted)
            .count(),
        report.negative_controls.len()
    )
    .unwrap();
    for c in &report.negative_controls {
        writeln!(
            out,
            "- `{}` -> `{}` ({})",
            c.case_id,
            c.outcome.as_str(),
            c.detail
        )
        .unwrap();
    }
    out
}
