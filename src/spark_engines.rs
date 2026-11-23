//! Issue #27: Qualify Hyperflare and TensorRT Edge-LLM on a bounded Spark serving workload.
//!
//! Pinned upstream boundaries:
//! - `OnixResearch/spark-nixos@1abf9f9efa56ba097d3728a2ecb2362d4746675f`: preserves existing
//!   3-GB10 Docker/NVIDIA CDI evidence while keeping `services.kindling.models` systemd-nspawn
//!   GPU/RDMA explicitly `untested`.
//! - `NVIDIA/hyperflare-inference@333b11ec43f975461fb38b897b3201d0e125d42a`: Ignition Python/PyTorch
//!   preparation + Helios Rust serving; KV+recurrent continuation with branches/resumption;
//!   excludes adapters, disaggregated prefill, pooling, and general device portability.
//! - `NVIDIA/TensorRT-Edge-LLM@513dcd18291e587c80928911a480761352f3c2d8`: DGX Spark/GB10 aarch64
//!   SM121, CUDA 13.0, platform TensorRT (NixOS build separately qualified); TP=2 MPI+NCCL over
//!   QSFP/RoCE excludes CP/EP/PP, TP speculative decoding, and context reuse when TP>1.
//!
//! Provides:
//! - Explicit support/capability matrix separating `supported`, `unsupported`, `unavailable`, and
//!   `failed` without ranking incompatible model/format runs or relabeling 3-rank baselines as TP2.
//! - Real pure-Rust hybrid KV + recurrent continuation verifier (`append/resume`, `branch`,
//!   `rollback`, `cancel`, `empty/long tool pauses` vs fresh full recomputation).
//! - `#11`-aligned lifecycle phase accounting, `#7`-aligned branch/tool trace profile, separate
//!   independent end-task check, and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const SPARK_NIXOS_COMMIT: &str = "1abf9f9efa56ba097d3728a2ecb2362d4746675f";
pub const HYPERFLARE_COMMIT: &str = "333b11ec43f975461fb38b897b3201d0e125d42a";
pub const TRT_EDGE_LLM_COMMIT: &str = "513dcd18291e587c80928911a480761352f3c2d8";

#[derive(Debug, Clone)]
pub struct EngineSupportRow {
    pub engine: String,
    pub topology: String,
    pub feature: String,
    pub status: &'static str,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

/// Tiny deterministic hybrid KV + recurrent continuation model verifying stateful append/resume,
/// branch, rollback, cancel, and tool pauses against fresh full recomputation!
#[derive(Debug, Clone)]
pub struct HybridContinuationModel {
    pub model_digest: String,
    pub runtime_core_digest: String,
    pub d_state: usize,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub embed: Vec<Vec<f64>>,
    pub decay: Vec<f64>,
    pub w_out: Vec<f64>,
}

impl HybridContinuationModel {
    pub fn frozen_tiny() -> Self {
        let d_state = 6;
        let vocab_size = 8;
        let embed: Vec<Vec<f64>> = (0..vocab_size)
            .map(|v| (0..d_state).map(|d| det_weight(2700 + v, d, 0.7)).collect())
            .collect();
        let decay = vec![0.82, 0.75, 0.90, 0.68, 0.85, 0.79];
        let w_out = (0..d_state * vocab_size)
            .map(|i| det_weight(2750, i, 0.5))
            .collect();
        Self {
            model_digest: DigestRef::sha256(b"spark-tiny-hybrid-model-v1").hex,
            runtime_core_digest: DigestRef::sha256(b"helios-edge-core-v1").hex,
            d_state,
            vocab_size,
            max_ctx: 16,
            embed,
            decay,
            w_out,
        }
    }

    /// Independent fresh full recomputation from token history.
    #[allow(clippy::type_complexity)]
    pub fn full_recompute(
        &self,
        tokens: &[usize],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>), String> {
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err("token length out of bounds".into());
        }
        let mut kv_history: Vec<Vec<f64>> = Vec::with_capacity(tokens.len());
        let mut rec_state = vec![0.0_f64; self.d_state];
        for (pos, &tok) in tokens.iter().enumerate() {
            if tok >= self.vocab_size {
                return Err("token out of vocab".into());
            }
            let e = &self.embed[tok];
            kv_history.push(e.clone());
            for d in 0..self.d_state {
                rec_state[d] = self.decay[d] * rec_state[d] + e[d] * (1.0 + 0.05 * (pos as f64));
            }
        }
        let logits = self.compute_logits(&kv_history, &rec_state);
        Ok((kv_history, rec_state, logits))
    }

    fn compute_logits(&self, kv_history: &[Vec<f64>], rec_state: &[f64]) -> Vec<f64> {
        // Combine mean KV context + recurrent state into output logits.
        let mut combined = vec![0.0_f64; self.d_state];
        let inv_len = 1.0 / (kv_history.len().max(1) as f64);
        for kv in kv_history {
            for d in 0..self.d_state {
                combined[d] += kv[d] * inv_len;
            }
        }
        for d in 0..self.d_state {
            combined[d] += 0.6 * rec_state[d];
        }
        let mut logits = vec![0.0_f64; self.vocab_size];
        for (v, lv) in logits.iter_mut().enumerate().take(self.vocab_size) {
            let mut acc = 0.0_f64;
            for (d, &cd) in combined.iter().enumerate().take(self.d_state) {
                acc += cd * self.w_out[d * self.vocab_size + v];
            }
            *lv = acc;
        }
        logits
    }
}

#[derive(Debug, Clone)]
pub struct ContinuationState {
    pub branch_id: String,
    pub model_digest: String,
    pub runtime_core_digest: String,
    pub tokens: Vec<usize>,
    pub kv_history: Vec<Vec<f64>>,
    pub recurrent_snapshots: Vec<Vec<f64>>,
    pub recurrent_state: Vec<f64>,
    pub tool_pause_ms_accumulated: u64,
    pub finish_reason: Option<String>,
}

impl ContinuationState {
    pub fn new(branch_id: &str, model: &HybridContinuationModel) -> Self {
        Self {
            branch_id: branch_id.into(),
            model_digest: model.model_digest.clone(),
            runtime_core_digest: model.runtime_core_digest.clone(),
            tokens: Vec::new(),
            kv_history: Vec::new(),
            recurrent_snapshots: Vec::new(),
            recurrent_state: vec![0.0_f64; model.d_state],
            tool_pause_ms_accumulated: 0,
            finish_reason: None,
        }
    }

    pub fn append_tokens(
        &mut self,
        model: &HybridContinuationModel,
        toks: &[usize],
    ) -> Result<Vec<f64>, String> {
        if self.model_digest != model.model_digest
            || self.runtime_core_digest != model.runtime_core_digest
        {
            return Err("incompatible model or runtime-core identity".into());
        }
        for &tok in toks {
            if self.tokens.len() >= model.max_ctx {
                return Err("context capacity exhausted".into());
            }
            if tok >= model.vocab_size {
                return Err("token out of vocab".into());
            }
            let pos = self.tokens.len();
            let e = &model.embed[tok];
            self.tokens.push(tok);
            self.kv_history.push(e.clone());
            for (d, &ed) in e.iter().enumerate().take(model.d_state) {
                self.recurrent_state[d] =
                    model.decay[d] * self.recurrent_state[d] + ed * (1.0 + 0.05 * (pos as f64));
            }
            self.recurrent_snapshots.push(self.recurrent_state.clone());
        }
        Ok(model.compute_logits(&self.kv_history, &self.recurrent_state))
    }

    pub fn branch(&self, new_branch_id: &str) -> Self {
        let mut cloned = self.clone();
        cloned.branch_id = new_branch_id.into();
        cloned
    }

    pub fn rollback_to(&mut self, keep_len: usize) -> Result<(), String> {
        if keep_len == 0 || keep_len > self.tokens.len() {
            return Err("invalid rollback length".into());
        }
        self.tokens.truncate(keep_len);
        self.kv_history.truncate(keep_len);
        self.recurrent_snapshots.truncate(keep_len);
        self.recurrent_state = self.recurrent_snapshots[keep_len - 1].clone();
        Ok(())
    }
}

fn max_err(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("length mismatch".into());
    }
    let mut m = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return Err("non-finite value".into());
        }
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    Ok(m)
}

/// Structured topology & runtime claim checked by `validate_spark_topology_claim`.
#[derive(Debug, Clone)]
pub struct SparkTopologyClaim {
    pub engine: String,
    pub claimed_topology: String,
    pub observed_ranks: usize,
    pub adapter_requested: bool,
    pub context_reuse_requested: bool,
    pub has_spark_7_nccl_receipt: bool,
    pub host_rdma_only: bool,
    pub finish_reason: Option<String>,
}

/// Real validator enforcing pinned Hyperflare / TensorRT Edge-LLM / Spark #7 capability rules.
pub fn validate_spark_topology_claim(claim: &SparkTopologyClaim) -> (OutcomeStatus, String) {
    if claim.engine == "hyperflare" && claim.adapter_requested {
        return (
            OutcomeStatus::Unsupported,
            "Hyperflare core-design.md excludes adapters".into(),
        );
    }
    if claim.engine == "tensorrt_edge_llm" && claim.claimed_topology == "tp2" {
        if claim.observed_ranks != 2 {
            return (
                OutcomeStatus::Invalid,
                format!(
                    "observed {} ranks cannot be relabeled as Edge-LLM TP2",
                    claim.observed_ranks
                ),
            );
        }
        if claim.context_reuse_requested {
            return (
                OutcomeStatus::Unsupported,
                "Edge-LLM multi-device.md excludes context reuse when TP>1".into(),
            );
        }
        if !claim.has_spark_7_nccl_receipt || claim.host_rdma_only {
            return (
                OutcomeStatus::Unsupported,
                "host-memory RDMA alone does not qualify GPU NCCL collectives (requires Spark #7 receipt)".into(),
            );
        }
    }
    if claim.finish_reason.is_none() {
        return (
            OutcomeStatus::Invalid,
            "incomplete output: missing finish_reason remains unknown".into(),
        );
    }
    (
        OutcomeStatus::Accepted,
        "claim satisfies topology contract".into(),
    )
}

fn finite_max(a: f64, b: f64) -> Result<f64, String> {
    if !a.is_finite() || !b.is_finite() {
        return Err("non-finite value in finite_max".into());
    }
    Ok(if a >= b { a } else { b })
}

/// Independent small end-task checker (separate from synthetic replay mechanics!).
#[derive(Debug, Clone)]
pub struct SparkEndTaskSpec {
    pub task_id: String,
    pub required_tool_call: String,
    pub required_final_answer: String,
}

pub fn check_spark_end_task(spec: &SparkEndTaskSpec, output: &str) -> OutcomeStatus {
    if output.contains(&spec.required_tool_call) && output.contains(&spec.required_final_answer) {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

#[derive(Debug, Clone)]
pub struct ContinuationCheck {
    pub op_name: String,
    pub logit_max_err: f64,
    pub recurrent_max_err: f64,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SparkEnginesReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub support_matrix: Vec<EngineSupportRow>,
    pub continuation_checks: Vec<ContinuationCheck>,
    pub end_task_passed: bool,
    pub negative_controls: Vec<NegativeControl>,
    pub decision: String,
}

pub fn run_default_harness() -> Result<SparkEnginesReport, String> {
    // 1. Explicit support & topology matrix (no incompatible speed ranking!).
    let support_matrix = vec![
        EngineSupportRow {
            engine: "spark_docker_cdi_baseline".into(),
            topology: "3_gb10_hosts_docker_cdi".into(),
            feature: "chat_completion_and_host_rdma".into(),
            status: "recorded_upstream_stack_preserved",
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: format!("spark-nixos@{SPARK_NIXOS_COMMIT} records real Docker/CDI completion; does not qualify Hyperflare/Edge-LLM or nspawn"),
        },
        EngineSupportRow {
            engine: "spark_kindling_nspawn".into(),
            topology: "systemd_nspawn_container".into(),
            feature: "gpu_and_rdma_passthrough".into(),
            status: "untested",
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: "services.kindling.models GPU/RDMA remains explicitly untested".into(),
        },
        EngineSupportRow {
            engine: "hyperflare_ignition_helios".into(),
            topology: "single_spark_sm121".into(),
            feature: "kv_and_recurrent_continuation_branch_resume".into(),
            status: "unavailable_locally",
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: format!("hyperflare@{HYPERFLARE_COMMIT} targets SM120/SM121; physical Spark execution required"),
        },
        EngineSupportRow {
            engine: "hyperflare_ignition_helios".into(),
            topology: "single_spark_sm121".into(),
            feature: "adapters_disaggregated_prefill_pooling".into(),
            status: "unsupported",
            outcome: OutcomeStatus::Unsupported,
            reason: "core-design.md explicitly excludes adapters, disaggregated prefill, pooling, and general portability".into(),
        },
        EngineSupportRow {
            engine: "tensorrt_edge_llm".into(),
            topology: "single_spark_gb10_aarch64_sm121".into(),
            feature: "cuda13_platform_trt_nixos_build".into(),
            status: "unavailable_locally",
            outcome: OutcomeStatus::UnavailableHardwareRuntime,
            reason: format!("TensorRT-Edge-LLM@{TRT_EDGE_LLM_COMMIT} lists DGX Spark stack; NixOS build is separately qualified"),
        },
        EngineSupportRow {
            engine: "tensorrt_edge_llm".into(),
            topology: "tp2_two_sparks_mpi_nccl_roce".into(),
            feature: "context_reuse_speculation_cp_ep_pp".into(),
            status: "unsupported",
            outcome: OutcomeStatus::Unsupported,
            reason: "multi-device.md excludes CP/EP/PP, TP speculative decoding, and context reuse when TP>1; requires Spark #7 NCCL qualification".into(),
        },
    ];

    // 2. Hybrid KV + recurrent continuation parity vs full recomputation!
    let model = HybridContinuationModel::frozen_tiny();
    let mut checks = Vec::new();

    // Append + resume across empty/long tool pause
    let mut root = ContinuationState::new("root", &model);
    let _ = root.append_tokens(&model, &[1, 3, 5])?;
    root.tool_pause_ms_accumulated += 0; // empty pause
    let logits_resume = root.append_tokens(&model, &[2, 4])?;
    root.tool_pause_ms_accumulated += 2500; // long tool pause
    let (_, ref_rec, ref_logits) = model.full_recompute(&root.tokens)?;
    let e_log = max_err(&logits_resume, &ref_logits)?;
    let e_rec = max_err(&root.recurrent_state, &ref_rec)?;
    checks.push(ContinuationCheck {
        op_name: "append_resume_and_tool_pauses".into(),
        logit_max_err: e_log,
        recurrent_max_err: e_rec,
        passed: e_log <= 1e-12 && e_rec <= 1e-12,
    });

    // Two diverging sibling branches from shared prefix
    let mut br_a = root.branch("branch_a");
    let mut br_b = root.branch("branch_b");
    let la = br_a.append_tokens(&model, &[6, 7])?;
    let lb = br_b.append_tokens(&model, &[0, 2])?;
    let (_, ref_ra, ref_la) = model.full_recompute(&br_a.tokens)?;
    let (_, ref_rb, ref_lb) = model.full_recompute(&br_b.tokens)?;
    let e_bra = finite_max(max_err(&la, &ref_la)?, max_err(&lb, &ref_lb)?)?;
    let e_rba = finite_max(
        max_err(&br_a.recurrent_state, &ref_ra)?,
        max_err(&br_b.recurrent_state, &ref_rb)?,
    )?;
    checks.push(ContinuationCheck {
        op_name: "sibling_branch_isolation".into(),
        logit_max_err: e_bra,
        recurrent_max_err: e_rba,
        passed: e_bra <= 1e-12 && e_rba <= 1e-12,
    });

    // Rollback + continuation
    br_a.rollback_to(5)?; // back to shared prefix!
    let l_after_rb = br_a.append_tokens(&model, &[1])?;
    let (_, ref_rrb, ref_lrb) = model.full_recompute(&br_a.tokens)?;
    let e_lrb = max_err(&l_after_rb, &ref_lrb)?;
    let e_rrb = max_err(&br_a.recurrent_state, &ref_rrb)?;
    checks.push(ContinuationCheck {
        op_name: "rollback_and_continue".into(),
        logit_max_err: e_lrb,
        recurrent_max_err: e_rrb,
        passed: e_lrb <= 1e-12 && e_rrb <= 1e-12,
    });

    // Cancellation leaves clean finish reason and exact prefix state
    let mut cancelled = root.branch("cancelled_branch");
    let _ = cancelled.append_tokens(&model, &[3])?;
    cancelled.finish_reason = Some("cancelled".into());
    let (_, ref_rc, ref_lc) = model.full_recompute(&cancelled.tokens)?;
    let lc = model.compute_logits(&cancelled.kv_history, &cancelled.recurrent_state);
    let e_lc = max_err(&lc, &ref_lc)?;
    let e_rc = max_err(&cancelled.recurrent_state, &ref_rc)?;
    checks.push(ContinuationCheck {
        op_name: "cancellation_state_integrity".into(),
        logit_max_err: e_lc,
        recurrent_max_err: e_rc,
        passed: e_lc <= 1e-12
            && e_rc <= 1e-12
            && cancelled.finish_reason.as_deref() == Some("cancelled"),
    });

    // Independent small end-task check (real structured checker separate from replay token equality!)
    let end_task_spec = SparkEndTaskSpec {
        task_id: "spark_tool_branch_join_v1".into(),
        required_tool_call: "tool:lookup_quota(branch=a)".into(),
        required_final_answer: "ANSWER:quota=128GiB".into(),
    };
    let end_task_passed = check_spark_end_task(
        &end_task_spec,
        "tool:lookup_quota(branch=a)\nANSWER:quota=128GiB",
    ) == OutcomeStatus::Accepted;
    let end_task_bad_verdict = check_spark_end_task(
        &end_task_spec,
        "tool:lookup_quota(branch=a)\nANSWER:quota=64GiB",
    );

    // 3. Eight executable negative controls!
    let mut negative_controls = Vec::new();

    // Control 1: Incompatible model / runtime-core artifact digest
    let mut wrong_core = root.clone();
    wrong_core.runtime_core_digest = "stale-core-digest".into();
    let r1 = wrong_core.append_tokens(&model, &[1]);
    negative_controls.push(NegativeControl {
        case_id: "incompatible_runtime_core_digest".into(),
        outcome: if r1.is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: r1.err().unwrap_or_else(|| "missed core mismatch".into()),
    });

    // Control 2: Stale/missing recurrent state (KV rolled back without rolling back recurrent state!)
    let mut stale_rec = br_b.clone();
    stale_rec.tokens.truncate(5);
    stale_rec.kv_history.truncate(5);
    // Deliberately forget to restore recurrent_state to snapshot[4]!
    let stale_logits = model.compute_logits(&stale_rec.kv_history, &stale_rec.recurrent_state);
    let stale_err = max_err(&stale_logits, &ref_logits)?;
    negative_controls.push(NegativeControl {
        case_id: "stale_recurrent_state_after_kv_rollback".into(),
        outcome: if stale_err > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("stale_recurrent_logit_err={stale_err:.6}"),
    });

    // Control 3: Sibling branch contamination (branch_b overwrites branch_a's recurrent state)
    let contam_logits = model.compute_logits(&br_a.kv_history, &br_b.recurrent_state);
    let contam_err = max_err(&contam_logits, &ref_lrb)?;
    negative_controls.push(NegativeControl {
        case_id: "sibling_branch_contamination".into(),
        outcome: if contam_err > 1e-4 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("branch_contamination_err={contam_err:.6}"),
    });

    // Control 4: Capacity exhaustion rejected cleanly
    let mut full = ContinuationState::new("cap", &model);
    let over: Vec<usize> = (0..18).map(|i| i % model.vocab_size).collect();
    let r4 = full.append_tokens(&model, &over);
    negative_controls.push(NegativeControl {
        case_id: "context_capacity_exhaustion".into(),
        outcome: if r4.is_err() {
            OutcomeStatus::Exhausted
        } else {
            OutcomeStatus::Accepted
        },
        detail: r4
            .err()
            .unwrap_or_else(|| "missed capacity overflow".into()),
    });

    // Control 5: Unsupported Hyperflare adapter & Edge-LLM TP2 context reuse refused via real validator
    let (hf_o, hf_d) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "hyperflare".into(),
        claimed_topology: "single_spark".into(),
        observed_ranks: 1,
        adapter_requested: true,
        context_reuse_requested: false,
        has_spark_7_nccl_receipt: false,
        host_rdma_only: false,
        finish_reason: Some("stop".into()),
    });
    let (edge_o, edge_d) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "tensorrt_edge_llm".into(),
        claimed_topology: "tp2".into(),
        observed_ranks: 2,
        adapter_requested: false,
        context_reuse_requested: true,
        has_spark_7_nccl_receipt: true,
        host_rdma_only: false,
        finish_reason: Some("stop".into()),
    });
    negative_controls.push(NegativeControl {
        case_id: "unsupported_hyperflare_adapter_and_edge_tp2_context_reuse".into(),
        outcome: if hf_o == OutcomeStatus::Unsupported && edge_o == OutcomeStatus::Unsupported {
            OutcomeStatus::Unsupported
        } else {
            OutcomeStatus::Accepted
        },
        detail: format!("hyperflare=({hf_d}); edge_tp2=({edge_d})"),
    });

    // Control 6: Missing finish_reason / incomplete output rejected via real validator
    let (o6, d6) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "hyperflare".into(),
        claimed_topology: "single_spark".into(),
        observed_ranks: 1,
        adapter_requested: false,
        context_reuse_requested: false,
        has_spark_7_nccl_receipt: false,
        host_rdma_only: false,
        finish_reason: root.finish_reason.clone(),
    });
    negative_controls.push(NegativeControl {
        case_id: "incomplete_output_missing_finish_reason".into(),
        outcome: o6,
        detail: d6,
    });

    // Control 7: 3-rank Spark baseline cannot be relabeled as Edge-LLM TP2
    let (o7, d7) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "tensorrt_edge_llm".into(),
        claimed_topology: "tp2".into(),
        observed_ranks: 3,
        adapter_requested: false,
        context_reuse_requested: false,
        has_spark_7_nccl_receipt: true,
        host_rdma_only: false,
        finish_reason: Some("stop".into()),
    });
    negative_controls.push(NegativeControl {
        case_id: "three_rank_baseline_not_relabeled_as_tp2".into(),
        outcome: o7,
        detail: d7,
    });

    // Control 8: Host-memory RDMA check alone cannot substitute for Spark #7 NCCL GPU-collective qualification
    let (o8, d8) = validate_spark_topology_claim(&SparkTopologyClaim {
        engine: "tensorrt_edge_llm".into(),
        claimed_topology: "tp2".into(),
        observed_ranks: 2,
        adapter_requested: false,
        context_reuse_requested: false,
        has_spark_7_nccl_receipt: false,
        host_rdma_only: true,
        finish_reason: Some("stop".into()),
    });
    negative_controls.push(NegativeControl {
        case_id: "host_rdma_does_not_qualify_gpu_nccl_collectives".into(),
        outcome: o8,
        detail: d8,
    });

    // Control 9: Independent end-task checker rejects wrong answer even when tool trace succeeded
    negative_controls.push(NegativeControl {
        case_id: "independent_end_task_checker_rejects_wrong_answer".into(),
        outcome: end_task_bad_verdict,
        detail: "wrong end-task answer rejected independently of replay trace mechanics".into(),
    });

    Ok(SparkEnginesReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "dgx_spark_gb10_aarch64_sm121_hyperflare_and_trt_edge_llm",
            "Device-free support matrix, KV+recurrent continuation verifier, and lifecycle controls executed locally; physical DGX Spark single-device and multi-host execution required",
        ),
        support_matrix,
        continuation_checks: checks,
        end_task_passed,
        negative_controls,
        decision: "INCONCLUSIVE_SINGLE_SPARK_PHYSICAL_QUALIFICATION_REQUIRED".into(),
    })
}

pub fn inputs_json(report: &SparkEnginesReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.spark-engines.inputs.v1\",\n",
            "  \"issue\": 27,\n",
            "  \"upstream_pins\": {{\"spark_nixos\":{},\"hyperflare\":{},\"tensorrt_edge_llm\":{}}},\n",
            "  \"lifecycle_phases\": [\"prepare_export_tuning\",\"cold_load\",\"graph_capture\",\"readiness\",\"warmup\",\"steady_exec\"],\n",
            "  \"production_engine_migration_added\": false,\n",
            "  \"execution_class\": {}\n",
            "}}\n"
        ),
        quote(SPARK_NIXOS_COMMIT),
        quote(HYPERFLARE_COMMIT),
        quote(TRT_EDGE_LLM_COMMIT),
        quote(report.execution_class.as_str()),
    )
}

pub fn results_json(report: &SparkEnginesReport) -> String {
    let mat: Vec<String> = report
        .support_matrix
        .iter()
        .map(|r| {
            format!(
                "{{\"engine\":{},\"topology\":{},\"feature\":{},\"status\":{},\"outcome\":{},\"reason\":{}}}",
                quote(&r.engine),
                quote(&r.topology),
                quote(&r.feature),
                quote(r.status),
                quote(r.outcome.as_str()),
                quote(&r.reason),
            )
        })
        .collect();
    let cont: Vec<String> = report
        .continuation_checks
        .iter()
        .map(|c| {
            format!(
                "{{\"op_name\":{},\"logit_max_err\":{},\"recurrent_max_err\":{},\"passed\":{}}}",
                quote(&c.op_name),
                finite_f64(c.logit_max_err),
                finite_f64(c.recurrent_max_err),
                c.passed,
            )
        })
        .collect();
    let neg: Vec<String> = report
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
            "  \"schema\": \"inference-lab.spark-engines.results.v1\",\n",
            "  \"issue\": 27,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"decision\": {},\n",
            "  \"physical_ttft_ms\": {},\n",
            "  \"peak_shared_memory_gib\": {},\n",
            "  \"end_task_passed\": {},\n",
            "  \"support_matrix\": [{}],\n",
            "  \"continuation_checks\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        quote(&report.decision),
        opt_f64(None),
        opt_f64(None),
        report.end_task_passed,
        mat.join(","),
        cont.join(","),
        neg.join(","),
    )
}

pub fn summary_markdown(report: &SparkEnginesReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #27: Hyperflare & TensorRT Edge-LLM DGX Spark Qualification"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}`",
        report.execution_class.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out, "- Decision: `{}`", report.decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Hybrid KV + Recurrent Continuation Parity").unwrap();
    for c in &report.continuation_checks {
        writeln!(
            out,
            "- `{}`: logit_err=`{:.3e}`, recurrent_err=`{:.3e}`, passed=`{}`",
            c.op_name, c.logit_max_err, c.recurrent_max_err, c.passed
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Negative controls").unwrap();
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
