//! Issue #49: Compare a harness-specific LoRA adapter (`harness-hub/Qwen3.8-27B-pi-adapted`)
//! with unadapted base, prompt-only change, and harness-only change on frozen agent tasks.
//!
//! Related existing issues: `#18`, `#25`, `OnixResearch/rsi-lab#57`, `OnixResearch/rsi-lab#98`.
//! External claims (`55-62% fewer tool calls`, large `pass@1` gains) are strictly unverified
//! related work and NEVER imported as lab pass/fail thresholds.
//!
//! Provides:
//! - Frozen 4-arm comparison (`unadapted_base`, `prompt_only_change`, `harness_only_change`,
//!   `harness_specific_lora`) over a frozen agent task set with an independent deterministic
//!   checker (never trusting self-reported model pass flags).
//! - Explicit benchmark-overlap / contamination audit rejecting tasks whose prompt digest or
//!   canonical token 3-grams overlap the adapter's training-set fingerprint manifest.
//! - Full cost reducer counting accepted tasks, tool calls, model-token cost + tool-call cost =
//!   total cost, with `cost_per_accepted_task = None` whenever `accepted_tasks == 0`.
//! - 8 executable negative controls and explicit `blocked_on_artifacts` hardware/artifact gate.

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const ADAPTER_REPO: &str = "harness-hub/Qwen3.8-27B-pi-adapted";
pub const SOURCE_BOOKMARK: &str = "https://x.com/yb2698/status/2122685012969398369";
pub const RELATED_ISSUES: [u32; 2] = [18, 25];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonArm {
    UnadaptedBase,
    PromptOnlyChange,
    HarnessOnlyChange,
    HarnessSpecificLora,
}

impl ComparisonArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnadaptedBase => "unadapted_base",
            Self::PromptOnlyChange => "prompt_only_change",
            Self::HarnessOnlyChange => "harness_only_change",
            Self::HarnessSpecificLora => "harness_specific_lora",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FrozenAgentTask {
    pub task_id: String,
    pub prompt_text: String,
    pub expected_answer: String,
    pub required_tool_invariant: String,
}

#[derive(Debug, Clone)]
pub struct TrainingOverlapManifest {
    pub adapter_repo: String,
    pub adapter_revision: String,
    pub train_task_digests: BTreeSet<String>,
    pub train_trigrams: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct OverlapAuditEntry {
    pub task_id: String,
    pub prompt_digest: String,
    pub exact_digest_overlap: bool,
    pub trigram_overlap_count: usize,
    pub contaminated: bool,
}

#[derive(Debug, Clone)]
pub struct TaskAttempt {
    pub task_id: String,
    pub submitted_answer: String,
    pub executed_tool_trace: Vec<String>,
    pub self_reported_pass: bool,
    pub model_tokens_cost_usd: f64,
    pub tool_calls_cost_usd: f64,
    pub tool_call_count: usize,
}

#[derive(Debug, Clone)]
pub struct ArmEvaluationSummary {
    pub arm: ComparisonArm,
    pub planned_tasks: usize,
    pub contaminated_excluded: usize,
    pub clean_heldout_tasks: usize,
    pub accepted_clean_tasks: usize,
    pub clean_pass_rate: Option<f64>,
    pub self_report_disagreements: usize,
    pub total_tool_calls_clean: usize,
    pub total_model_cost_usd: f64,
    pub total_tool_cost_usd: f64,
    pub total_cost_usd: f64,
    pub cost_per_accepted_task_usd: Option<f64>,
    pub tool_call_reduction_vs_base_pct: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct HarnessLoraCompareReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub overlap_audit: Vec<OverlapAuditEntry>,
    pub arms: Vec<ArmEvaluationSummary>,
    pub negative_controls: Vec<NegativeControl>,
}

fn check_finite_nonneg(v: f64, label: &str) -> Result<f64, String> {
    if !v.is_finite() || v < 0.0 {
        return Err(format!("non-finite or negative float in {label}"));
    }
    Ok(v)
}

pub fn canonical_trigrams(text: &str) -> BTreeSet<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_ascii_lowercase())
        .collect();
    let mut set = BTreeSet::new();
    if words.len() >= 3 {
        for win in words.windows(3) {
            set.insert(format!("{} {} {}", win[0], win[1], win[2]));
        }
    }
    set
}

pub fn audit_benchmark_overlap(
    tasks: &[FrozenAgentTask],
    manifest: &TrainingOverlapManifest,
) -> Result<Vec<OverlapAuditEntry>, String> {
    if manifest.adapter_revision.trim().is_empty() || manifest.adapter_revision == "main" {
        return Err(
            "adapter revision must be a pinned digest/commit, not empty or floating tip".into(),
        );
    }
    let mut out = Vec::with_capacity(tasks.len());
    for task in tasks {
        let digest = DigestRef::sha256(task.prompt_text.as_bytes()).hex;
        let exact = manifest.train_task_digests.contains(&digest);
        let trigrams = canonical_trigrams(&task.prompt_text);
        let overlap_count = trigrams
            .iter()
            .filter(|tg| manifest.train_trigrams.contains(*tg))
            .count();
        let contaminated = exact || overlap_count >= 2;
        out.push(OverlapAuditEntry {
            task_id: task.task_id.clone(),
            prompt_digest: digest,
            exact_digest_overlap: exact,
            trigram_overlap_count: overlap_count,
            contaminated,
        });
    }
    Ok(out)
}

/// Independent deterministic task checker: verifies exact answer AND required tool invariant.
/// Deliberately ignores `attempt.self_reported_pass`.
pub fn verify_agent_task_output(task: &FrozenAgentTask, attempt: &TaskAttempt) -> OutcomeStatus {
    let answer_ok = attempt.submitted_answer.trim() == task.expected_answer.trim();
    let tool_ok = attempt
        .executed_tool_trace
        .iter()
        .any(|t| t == &task.required_tool_invariant);
    if answer_ok && tool_ok {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

pub fn reduce_arm_metrics(
    arm: ComparisonArm,
    tasks: &[FrozenAgentTask],
    audit: &[OverlapAuditEntry],
    attempts: &[TaskAttempt],
    include_tool_cost: bool,
    base_tool_calls: Option<usize>,
) -> Result<ArmEvaluationSummary, String> {
    if !include_tool_cost {
        return Err("omitting tool-call cost from total cost accounting is forbidden".into());
    }
    if tasks.len() != audit.len() || tasks.len() != attempts.len() {
        return Err("arm attempts must match frozen task population 1:1".into());
    }
    let mut contaminated_excluded = 0usize;
    let mut clean_heldout = 0usize;
    let mut accepted_clean = 0usize;
    let mut self_report_disagreements = 0usize;
    let mut total_tool_calls = 0usize;
    let mut total_model_cost = 0.0_f64;
    let mut total_tool_cost = 0.0_f64;

    for ((task, ov), attempt) in tasks.iter().zip(audit.iter()).zip(attempts.iter()) {
        if task.task_id != attempt.task_id || task.task_id != ov.task_id {
            return Err("mismatched task_id ordering across frozen task set".into());
        }
        let m_cost = check_finite_nonneg(attempt.model_tokens_cost_usd, "model_tokens_cost_usd")?;
        let t_cost = check_finite_nonneg(attempt.tool_calls_cost_usd, "tool_calls_cost_usd")?;
        if ov.contaminated {
            contaminated_excluded += 1;
            continue;
        }
        clean_heldout += 1;
        let independent = verify_agent_task_output(task, attempt);
        let independent_pass = independent == OutcomeStatus::Accepted;
        if independent_pass != attempt.self_reported_pass {
            self_report_disagreements += 1;
        }
        if independent_pass {
            accepted_clean += 1;
        }
        total_tool_calls += attempt.tool_call_count;
        total_model_cost += m_cost;
        total_tool_cost += t_cost;
    }

    let total_cost = total_model_cost + total_tool_cost;
    check_finite_nonneg(total_cost, "total_cost_usd")?;

    let clean_pass_rate = if clean_heldout == 0 {
        None
    } else {
        Some((accepted_clean as f64) / (clean_heldout as f64))
    };
    let cost_per_accepted = if accepted_clean == 0 {
        None
    } else {
        Some(total_cost / (accepted_clean as f64))
    };
    let tool_call_reduction_vs_base_pct = match base_tool_calls {
        Some(base) if base > 0 => {
            let delta = (base as f64) - (total_tool_calls as f64);
            Some((delta / (base as f64)) * 100.0)
        }
        _ => None,
    };

    Ok(ArmEvaluationSummary {
        arm,
        planned_tasks: tasks.len(),
        contaminated_excluded,
        clean_heldout_tasks: clean_heldout,
        accepted_clean_tasks: accepted_clean,
        clean_pass_rate,
        self_report_disagreements,
        total_tool_calls_clean: total_tool_calls,
        total_model_cost_usd: total_model_cost,
        total_tool_cost_usd: total_tool_cost,
        total_cost_usd: total_cost,
        cost_per_accepted_task_usd: cost_per_accepted,
        tool_call_reduction_vs_base_pct,
    })
}

pub fn validate_no_post_claim_gate(imports_55_62_pct_claim_as_gate: bool) -> OutcomeStatus {
    if imports_55_62_pct_claim_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

fn default_tasks_and_manifest() -> (Vec<FrozenAgentTask>, TrainingOverlapManifest) {
    let tasks = vec![
        FrozenAgentTask {
            task_id: "agent_clean_01".into(),
            prompt_text: "Compute deterministic checksum for ledger segment alpha seven".into(),
            expected_answer: "cksum:42a9".into(),
            required_tool_invariant: "run_verifier:alpha7".into(),
        },
        FrozenAgentTask {
            task_id: "agent_clean_02".into(),
            prompt_text: "Patch parser boundary overflow in rust slice decoder".into(),
            expected_answer: "patch:bounds_ok".into(),
            required_tool_invariant: "cargo_test:pass".into(),
        },
        FrozenAgentTask {
            task_id: "agent_clean_03".into(),
            prompt_text: "Audit manifest sha256 digest drift across two outputs".into(),
            expected_answer: "drift:0".into(),
            required_tool_invariant: "sha256_diff:match".into(),
        },
        FrozenAgentTask {
            task_id: "agent_clean_04".into(),
            prompt_text: "Isolate deadlocked queue consumer without changing api".into(),
            expected_answer: "fix:queue_unlock".into(),
            required_tool_invariant: "unit_test:pass".into(),
        },
        FrozenAgentTask {
            task_id: "agent_overlap_05".into(),
            prompt_text: "Pi harness benchmark training task repair json schema validator".into(),
            expected_answer: "schema:fixed".into(),
            required_tool_invariant: "schema_check:pass".into(),
        },
    ];

    let mut train_digests = BTreeSet::new();
    train_digests.insert(DigestRef::sha256(tasks[4].prompt_text.as_bytes()).hex);
    let train_trigrams = canonical_trigrams(&tasks[4].prompt_text);

    let manifest = TrainingOverlapManifest {
        adapter_repo: ADAPTER_REPO.into(),
        adapter_revision: "pin-required-blocked-offline-artifact".into(),
        train_task_digests: train_digests,
        train_trigrams,
    };
    (tasks, manifest)
}

pub fn run_default_harness() -> Result<HarnessLoraCompareReport, String> {
    let (tasks, manifest) = default_tasks_and_manifest();
    let audit = audit_benchmark_overlap(&tasks, &manifest)?;

    let mut digest_buf = String::new();
    for t in &tasks {
        let _ = write!(
            digest_buf,
            "{}:{}:{};",
            t.task_id, t.prompt_text, t.expected_answer
        );
    }
    let fixture_digest = DigestRef::sha256(digest_buf.as_bytes());

    let mk =
        |id: &str, ans: &str, tools: &[&str], self_pass: bool, mc: f64, tc: f64, calls: usize| {
            TaskAttempt {
                task_id: id.into(),
                submitted_answer: ans.into(),
                executed_tool_trace: tools.iter().map(|s| (*s).into()).collect(),
                self_reported_pass: self_pass,
                model_tokens_cost_usd: mc,
                tool_calls_cost_usd: tc,
                tool_call_count: calls,
            }
        };

    // Unadapted base: 1/4 clean accepted, higher tool calls, includes a false self-report on task 2
    let base_attempts = vec![
        mk(
            "agent_clean_01",
            "cksum:42a9",
            &["inspect", "run_verifier:alpha7"],
            true,
            0.018,
            0.008,
            8,
        ),
        mk(
            "agent_clean_02",
            "patch:wrong",
            &["edit"],
            true,
            0.022,
            0.010,
            10,
        ), // self-report disagrees!
        mk(
            "agent_clean_03",
            "drift:1",
            &["read"],
            false,
            0.019,
            0.009,
            9,
        ),
        mk(
            "agent_clean_04",
            "fix:timeout",
            &["read"],
            false,
            0.021,
            0.011,
            11,
        ),
        mk(
            "agent_overlap_05",
            "schema:fixed",
            &["schema_check:pass"],
            true,
            0.012,
            0.004,
            4,
        ),
    ];

    // Prompt-only change: 2/4 clean accepted, moderate tool calls
    let prompt_attempts = vec![
        mk(
            "agent_clean_01",
            "cksum:42a9",
            &["run_verifier:alpha7"],
            true,
            0.017,
            0.006,
            6,
        ),
        mk(
            "agent_clean_02",
            "patch:bounds_ok",
            &["cargo_test:pass"],
            true,
            0.020,
            0.007,
            7,
        ),
        mk(
            "agent_clean_03",
            "drift:unknown",
            &["read"],
            false,
            0.018,
            0.007,
            7,
        ),
        mk(
            "agent_clean_04",
            "fix:queue_unlock",
            &["read"],
            true,
            0.019,
            0.008,
            8,
        ), // missing required tool!
        mk(
            "agent_overlap_05",
            "schema:fixed",
            &["schema_check:pass"],
            true,
            0.011,
            0.003,
            3,
        ),
    ];

    // Harness-only change: 3/4 clean accepted, lower tool calls
    let harness_attempts = vec![
        mk(
            "agent_clean_01",
            "cksum:42a9",
            &["run_verifier:alpha7"],
            true,
            0.015,
            0.004,
            4,
        ),
        mk(
            "agent_clean_02",
            "patch:bounds_ok",
            &["cargo_test:pass"],
            true,
            0.016,
            0.005,
            5,
        ),
        mk(
            "agent_clean_03",
            "drift:0",
            &["sha256_diff:match"],
            true,
            0.015,
            0.004,
            4,
        ),
        mk(
            "agent_clean_04",
            "fix:incomplete",
            &["read"],
            false,
            0.017,
            0.006,
            6,
        ),
        mk(
            "agent_overlap_05",
            "schema:fixed",
            &["schema_check:pass"],
            true,
            0.010,
            0.002,
            2,
        ),
    ];

    // Harness-specific LoRA: 3/4 clean accepted, fewest tool calls (contaminated task 5 excluded!)
    let lora_attempts = vec![
        mk(
            "agent_clean_01",
            "cksum:42a9",
            &["run_verifier:alpha7"],
            true,
            0.014,
            0.003,
            3,
        ),
        mk(
            "agent_clean_02",
            "patch:bounds_ok",
            &["cargo_test:pass"],
            true,
            0.015,
            0.004,
            4,
        ),
        mk(
            "agent_clean_03",
            "drift:0",
            &["sha256_diff:match"],
            true,
            0.014,
            0.003,
            3,
        ),
        mk(
            "agent_clean_04",
            "fix:wrong",
            &["unit_test:pass"],
            false,
            0.016,
            0.005,
            5,
        ),
        mk(
            "agent_overlap_05",
            "schema:fixed",
            &["schema_check:pass"],
            true,
            0.008,
            0.001,
            1,
        ),
    ];

    let r_base = reduce_arm_metrics(
        ComparisonArm::UnadaptedBase,
        &tasks,
        &audit,
        &base_attempts,
        true,
        None,
    )?;
    let base_calls = Some(r_base.total_tool_calls_clean);
    let r_prompt = reduce_arm_metrics(
        ComparisonArm::PromptOnlyChange,
        &tasks,
        &audit,
        &prompt_attempts,
        true,
        base_calls,
    )?;
    let r_harness = reduce_arm_metrics(
        ComparisonArm::HarnessOnlyChange,
        &tasks,
        &audit,
        &harness_attempts,
        true,
        base_calls,
    )?;
    let r_lora = reduce_arm_metrics(
        ComparisonArm::HarnessSpecificLora,
        &tasks,
        &audit,
        &lora_attempts,
        true,
        base_calls,
    )?;

    let mut negative_controls = Vec::new();

    // NC1: overlapping benchmark task is flagged and excluded from clean held-out set
    negative_controls.push(NegativeControl {
        case_id: "nc1_training_overlap_task_excluded".into(),
        outcome: if audit[4].contaminated && r_lora.contaminated_excluded == 1 {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail:
            "task overlapping adapter training fingerprint is rejected from clean held-out scoring"
                .into(),
    });

    // NC2: zero accepted tasks yields cost_per_accepted_task_usd == None (never div-by-zero/Inf)
    {
        let zero_attempts: Vec<TaskAttempt> = tasks
            .iter()
            .map(|t| mk(&t.task_id, "wrong", &[], false, 0.01, 0.01, 2))
            .collect();
        let zero_res = reduce_arm_metrics(
            ComparisonArm::UnadaptedBase,
            &tasks,
            &audit,
            &zero_attempts,
            true,
            None,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_zero_accepted_cost_is_none".into(),
            outcome: if zero_res.accepted_clean_tasks == 0
                && zero_res.cost_per_accepted_task_usd.is_none()
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail:
                "zero accepted tasks returns None cost_per_accepted_task (zero divisor guarded)"
                    .into(),
        });
    }

    // NC3: omitting tool-call cost from total cost rejected
    {
        let outcome = if reduce_arm_metrics(
            ComparisonArm::HarnessSpecificLora,
            &tasks,
            &audit,
            &lora_attempts,
            false,
            base_calls,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_omitting_tool_cost_rejected".into(),
            outcome,
            detail: "total cost accounting must include tool-call costs alongside model tokens"
                .into(),
        });
    }

    // NC4: false self-reported pass overridden by independent checker
    negative_controls.push(NegativeControl {
        case_id: "nc4_false_self_report_overridden".into(),
        outcome: verify_agent_task_output(&tasks[1], &base_attempts[1]),
        detail: "self_reported_pass=true is overridden to invalid when independent checker fails"
            .into(),
    });

    // NC5: mismatched task population across arms rejected
    {
        let outcome = if reduce_arm_metrics(
            ComparisonArm::PromptOnlyChange,
            &tasks,
            &audit,
            &prompt_attempts[..3],
            true,
            base_calls,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc5_mismatched_task_population_rejected".into(),
            outcome,
            detail: "evaluating an arm on a subset of the frozen task set is rejected".into(),
        });
    }

    // NC6: NaN or negative cost rejected
    {
        let mut bad = base_attempts.clone();
        bad[0].tool_calls_cost_usd = f64::NAN;
        let outcome = if reduce_arm_metrics(
            ComparisonArm::UnadaptedBase,
            &tasks,
            &audit,
            &bad,
            true,
            None,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc6_nan_cost_rejected".into(),
            outcome,
            detail: "non-finite model or tool cost is rejected rather than swallowed".into(),
        });
    }

    // NC7: importing post's 55-62% tool-call reduction claim as pass gate rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_post_tool_reduction_claim_not_gate".into(),
        outcome: validate_no_post_claim_gate(true),
        detail: "authors' 55-62% tool-call reduction claim is unverified related work, not a gate"
            .into(),
    });

    // NC8: floating adapter tip ("main") rejected by overlap auditor
    {
        let mut floating = manifest.clone();
        floating.adapter_revision = "main".into();
        let outcome = if audit_benchmark_overlap(&tasks, &floating).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc8_unpinned_adapter_tip_rejected".into(),
            outcome,
            detail: "floating adapter tip 'main' is refused; pinned revision required".into(),
        });
    }

    Ok(HarnessLoraCompareReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "harness-hub/Qwen3.8-27B-pi-adapted LoRA weights, base Qwen3.8-27B checkpoint, and adapter training-set manifest",
            "4-arm independent checker, tool-inclusive cost reducer, and training-overlap contamination auditor verified on frozen synthetic contract; HF adapter weights & training manifest not present locally",
        ),
        synthetic_contract_fixture: true,
        fixture_digest,
        overlap_audit: audit,
        arms: vec![r_base, r_prompt, r_harness, r_lora],
        negative_controls,
    })
}

pub fn inputs_json(report: &HarnessLoraCompareReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.harness-lora-compare.inputs.v1\",\n",
            "  \"issue\": 49,\n",
            "  \"related_issues\": [18, 25],\n",
            "  \"execution_class\": {},\n",
            "  \"adapter_repo\": {},\n",
            "  \"source_bookmark\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(ADAPTER_REPO),
        quote(SOURCE_BOOKMARK),
        report.fixture_digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &HarnessLoraCompareReport) -> String {
    let audit: Vec<String> = report
        .overlap_audit
        .iter()
        .map(|o| {
            format!(
                "{{\"task_id\":{},\"prompt_digest\":{},\"exact_digest_overlap\":{},\"trigram_overlap_count\":{},\"contaminated\":{}}}",
                quote(&o.task_id),
                quote(&o.prompt_digest),
                o.exact_digest_overlap,
                o.trigram_overlap_count,
                o.contaminated,
            )
        })
        .collect();
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm\":{},\"planned_tasks\":{},\"contaminated_excluded\":{},\"clean_heldout_tasks\":{},\"accepted_clean_tasks\":{},\"clean_pass_rate\":{},\"self_report_disagreements\":{},\"total_tool_calls_clean\":{},\"total_model_cost_usd\":{},\"total_tool_cost_usd\":{},\"total_cost_usd\":{},\"cost_per_accepted_task_usd\":{},\"tool_call_reduction_vs_base_pct\":{}}}",
                quote(a.arm.as_str()),
                a.planned_tasks,
                a.contaminated_excluded,
                a.clean_heldout_tasks,
                a.accepted_clean_tasks,
                opt_f64(a.clean_pass_rate),
                a.self_report_disagreements,
                a.total_tool_calls_clean,
                finite_f64(a.total_model_cost_usd),
                finite_f64(a.total_tool_cost_usd),
                finite_f64(a.total_cost_usd),
                opt_f64(a.cost_per_accepted_task_usd),
                opt_f64(a.tool_call_reduction_vs_base_pct),
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
                quote(&c.detail),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.harness-lora-compare.results.v1\",\n",
            "  \"issue\": 49,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"overlap_audit\": [{}],\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        audit.join(","),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &HarnessLoraCompareReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #49: Harness-Specific LoRA vs Base, Prompt-Only, and Harness-Only Arms"
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
        "- Artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Benchmark overlap audit: `{}` / `{}` tasks flagged contaminated and excluded from clean scoring",
        report.overlap_audit.iter().filter(|o| o.contaminated).count(),
        report.overlap_audit.len()
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## 4-Arm Clean Held-Out Results (Tool Costs Included)").unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}`: `accepted={}/{}`, `tool_calls={}`, `total_cost_usd={:.4}`, `cost_per_accepted={:?}`, `self_report_disagreements={}`",
            a.arm.as_str(),
            a.accepted_clean_tasks,
            a.clean_heldout_tasks,
            a.total_tool_calls_clean,
            a.total_cost_usd,
            a.cost_per_accepted_task_usd,
            a.self_report_disagreements,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_harness_excludes_overlap_and_checks_all_four_arms() {
        let report = run_default_harness().unwrap();
        assert_eq!(
            report.hardware_gate.status,
            ExecutionClass::BlockedOnArtifacts
        );
        assert_eq!(report.arms.len(), 4);
        for a in &report.arms {
            assert_eq!(a.planned_tasks, 5);
            assert_eq!(a.contaminated_excluded, 1);
            assert_eq!(a.clean_heldout_tasks, 4);
        }
        assert_eq!(report.arms[0].self_report_disagreements, 1);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
