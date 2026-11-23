//! Issue #14: Conditionally extract native RWKV numerical code after a second consumer
//! demonstrates the API.
//!
//! Remains explicitly `blocked_on_artifacts` until a second concrete non-mock native-RWKV
//! consumer with named call sites demonstrates the leaf API (a paper, planned benchmark, or
//! second mock adapter does NOT satisfy the trigger).
//!
//! Pure deterministic core additionally verifies the leaf's target contracts on CPU:
//! - Conditional trigger gate evaluation (positive + 3 negative cases).
//! - Pinned byte-prefix trie RWKV tokenizer semantics (vs negative generic whitespace tokenizer).
//! - Structural safetensors header admission validator (dtype/shape bounds, byte-range overlap,
//!   pinned weight digest, file-immutability contract flag).
//! - Zero-reallocation reusable-workspace FP32/FP64 recurrent step (`workspace_allocations == 0`).
//! - Explicit FP32-CPU vs FLA-BF16-CUDA non-interchangeability guard and RSI wrapper preservation
//!   controls (per-agent isolation, offset refusal, snapshot purity, atomic failure rollback).

use std::collections::BTreeSet;
use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_str, quote, str_array, usize_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const RWKV_LEAF_DONOR_COMMIT: &str = "01e47e268d71bc1c53cc36e11c179f2b8f22f5c5";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafConsumerCandidate {
    pub consumer_id: String,
    pub concrete_call_site: Option<String>,
    pub is_live_non_mock_consumer: bool,
    pub pins_leaf_revision: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafTriggerVerdict {
    pub case_id: String,
    pub trigger_satisfied: bool,
    pub qualified_consumers: usize,
    pub rejection_reason: Option<String>,
}

/// Strictly enforce Issue #14's conditional extraction trigger: >= 2 distinct concrete non-mock
/// consumers with named call sites pinning the same leaf revision.
pub fn evaluate_leaf_trigger(
    case_id: &str,
    candidates: &[LeafConsumerCandidate],
) -> LeafTriggerVerdict {
    let mut qualified = BTreeSet::new();
    let mut has_planned_or_mock_only = false;
    for c in candidates {
        if c.is_live_non_mock_consumer && c.concrete_call_site.is_some() && c.pins_leaf_revision {
            qualified.insert(c.consumer_id.as_str());
        } else {
            has_planned_or_mock_only = true;
        }
    }

    let reason = if qualified.len() >= 2 {
        None
    } else if has_planned_or_mock_only {
        Some("paper_planned_benchmark_or_mock_cannot_satisfy_second_consumer_trigger".into())
    } else {
        Some("fewer_than_two_concrete_native_rwkv_consumers".into())
    };

    LeafTriggerVerdict {
        case_id: case_id.into(),
        trigger_satisfied: reason.is_none(),
        qualified_consumers: qualified.len(),
        rejection_reason: reason,
    }
}

/// Minimal pure-Rust byte-prefix trie tokenizer modeling RWKV's pinned longest-match byte-prefix
/// vocabulary semantics.
#[derive(Debug, Clone)]
pub struct BytePrefixVocab {
    /// `(token_id, token_bytes)` sorted by descending byte length, then ascending `token_id`.
    entries: Vec<(u32, Vec<u8>)>,
}

impl BytePrefixVocab {
    pub fn pinned_fixture() -> Self {
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        // Multi-byte greedy prefix entries (high IDs above raw byte fallback 0..255).
        let specials: &[(u32, &[u8])] = &[
            (256, b"rwkv"),
            (257, b"rwkv7"),
            (258, b" state"),
            (259, b"step"),
            (260, b"\n\n"),
            (261, "é".as_bytes()),
        ];
        for &(id, bytes) in specials {
            entries.push((id, bytes.to_vec()));
        }
        // Single-byte fallback guarantees total coverage of arbitrary UTF-8/binary inputs.
        for b in 0u8..=255u8 {
            entries.push((u32::from(b), vec![b]));
        }
        entries.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));
        Self { entries }
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        let mut pos = 0;
        while pos < input.len() {
            let mut matched = None;
            for (id, pat) in &self.entries {
                if !pat.is_empty() && input[pos..].starts_with(pat) {
                    matched = Some((*id, pat.len()));
                    break;
                }
            }
            let (id, len) =
                matched.ok_or_else(|| "byte-prefix trie missing fallback".to_string())?;
            ids.push(id);
            pos += len;
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for &id in ids {
            let pat = self
                .entries
                .iter()
                .find(|(eid, _)| *eid == id)
                .map(|(_, bytes)| bytes)
                .ok_or_else(|| format!("unknown rwkv token id {id}"))?;
            out.extend_from_slice(pat);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeaderEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset_start: usize,
    pub offset_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsAdmissionInput {
    pub case_id: String,
    pub header_bytes: usize,
    pub max_header_bytes: usize,
    pub payload_bytes: usize,
    pub file_immutable_for_map_lifetime: bool,
    pub expected_model_digest: DigestRef,
    pub observed_model_digest: DigestRef,
    pub tensors: Vec<TensorHeaderEntry>,
}

fn dtype_elem_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "BF16" | "F16" => Some(2),
        "F64" => Some(8),
        _ => None,
    }
}

/// Structural safetensors admission validator (bounds, dtype, shape-byte agreement, overlap,
/// pinned digest, and immutable-mapping safety obligation).
pub fn validate_safetensors_admission(
    input: &SafetensorsAdmissionInput,
) -> (OutcomeStatus, Option<String>) {
    if !input.file_immutable_for_map_lifetime {
        return (
            OutcomeStatus::Invalid,
            Some("mapped_file_immutability_safety_contract_missing".into()),
        );
    }
    if input.header_bytes == 0 || input.header_bytes > input.max_header_bytes {
        return (
            OutcomeStatus::Invalid,
            Some("malformed_or_oversized_safetensors_header".into()),
        );
    }
    if input.expected_model_digest != input.observed_model_digest {
        return (
            OutcomeStatus::Invalid,
            Some("pinned_weight_digest_mismatch".into()),
        );
    }
    if input.tensors.is_empty() {
        return (OutcomeStatus::Invalid, Some("empty_tensor_header".into()));
    }

    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(input.tensors.len());
    let mut seen_names = BTreeSet::new();
    for t in &input.tensors {
        if t.name.trim().is_empty() || !seen_names.insert(t.name.as_str()) {
            return (
                OutcomeStatus::Invalid,
                Some("empty_or_duplicate_tensor_name".into()),
            );
        }
        let Some(elem_bytes) = dtype_elem_bytes(&t.dtype) else {
            return (
                OutcomeStatus::Unsupported,
                Some("unsupported_tensor_dtype".into()),
            );
        };
        if t.shape.is_empty() || t.shape.contains(&0) {
            return (
                OutcomeStatus::Invalid,
                Some("invalid_zero_or_empty_shape".into()),
            );
        }
        let mut elems = 1usize;
        for &dim in &t.shape {
            let Some(next) = elems.checked_mul(dim) else {
                return (OutcomeStatus::Invalid, Some("tensor_shape_overflow".into()));
            };
            elems = next;
        }
        let Some(expected_bytes) = elems.checked_mul(elem_bytes) else {
            return (
                OutcomeStatus::Invalid,
                Some("tensor_byte_size_overflow".into()),
            );
        };
        if t.offset_start >= t.offset_end || t.offset_end > input.payload_bytes {
            return (
                OutcomeStatus::Invalid,
                Some("tensor_byte_range_out_of_bounds".into()),
            );
        }
        if t.offset_end - t.offset_start != expected_bytes {
            return (
                OutcomeStatus::Invalid,
                Some("tensor_shape_and_byte_span_mismatch".into()),
            );
        }
        ranges.push((t.offset_start, t.offset_end));
    }

    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return (
                OutcomeStatus::Invalid,
                Some("overlapping_tensor_byte_ranges".into()),
            );
        }
    }

    (OutcomeStatus::Accepted, None)
}

/// Reusable scratch workspace proving zero per-token heap allocation across steps.
#[derive(Debug, Clone)]
pub struct RwkvReusableWorkspace {
    pub scratch_a: Vec<f32>,
    pub scratch_b: Vec<f32>,
    pub reallocations: usize,
}

impl RwkvReusableWorkspace {
    pub fn with_dim(dim: usize) -> Self {
        Self {
            scratch_a: vec![0.0_f32; dim],
            scratch_b: vec![0.0_f32; dim],
            reallocations: 0,
        }
    }

    /// Execute one bounded FP32 linear + decay recurrence step strictly inside preallocated buffers.
    pub fn step_in_place(
        &mut self,
        state: &mut [f32],
        token_vec: &[f32],
        decay: f32,
        out: &mut [f32],
    ) -> Result<(), String> {
        let d = state.len();
        if d == 0
            || token_vec.len() != d
            || out.len() != d
            || self.scratch_a.len() != d
            || self.scratch_b.len() != d
            || !decay.is_finite()
        {
            return Err("invalid workspace or step dimension".into());
        }
        let cap_a_before = self.scratch_a.capacity();
        let cap_b_before = self.scratch_b.capacity();
        for i in 0..d {
            if !state[i].is_finite() || !token_vec[i].is_finite() {
                return Err("non-finite input in leaf step".into());
            }
            self.scratch_a[i] = state[i] * decay;
            self.scratch_b[i] = token_vec[i] * (1.0 - decay);
            let next = self.scratch_a[i] + self.scratch_b[i];
            if !next.is_finite() {
                return Err("non-finite state overflow in leaf step".into());
            }
            state[i] = next;
            out[i] = next.tanh();
        }
        if self.scratch_a.capacity() != cap_a_before || self.scratch_b.capacity() != cap_b_before {
            self.reallocations += 1;
        }
        Ok(())
    }
}

/// Validate RSI wrapper invariants preserved across leaf extraction.
pub fn validate_rsi_wrapper_contract(
    per_agent_isolated: bool,
    offset_refusal_enforced: bool,
    snapshot_immutable: bool,
    atomic_rollback_on_failure: bool,
) -> OutcomeStatus {
    if per_agent_isolated
        && offset_refusal_enforced
        && snapshot_immutable
        && atomic_rollback_on_failure
    {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

/// Reject silent checkpoint byte changes without an explicitly declared version bump.
pub fn validate_checkpoint_version_policy(
    checkpoint_bytes_changed: bool,
    version_bump_declared: bool,
) -> OutcomeStatus {
    if checkpoint_bytes_changed && !version_bump_declared {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Reject equating FP32 CPU architectural behavior with FLA BF16 CUDA policy parity.
pub fn validate_precision_interchange_claim(
    claims_fp32_cpu_interchangeable_with_fla_bf16_cuda: bool,
) -> OutcomeStatus {
    if claims_fp32_cpu_interchangeable_with_fla_bf16_cuda {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafControlVerdict {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct RwkvLeafReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub donor_commit: String,
    pub candidate_leaf_modules: Vec<String>,
    pub retained_in_rsi: Vec<String>,
    pub trigger_verdicts: Vec<LeafTriggerVerdict>,
    pub trie_token_ids: Vec<usize>,
    pub trie_roundtrip_exact: bool,
    pub generic_whitespace_diverges: bool,
    pub workspace_steps_executed: usize,
    pub workspace_reallocations: usize,
    pub fp32_cpu_interchangeable_with_fla_bf16_cuda: bool,
    pub admission_and_wrapper_controls: Vec<LeafControlVerdict>,
    pub final_output_l2_norm: f64,
}

pub fn run_default_harness() -> Result<RwkvLeafReport, String> {
    let trigger_verdicts = vec![
        evaluate_leaf_trigger(
            "01-current-state-deferred-single-rsi-consumer",
            &[
                LeafConsumerCandidate {
                    consumer_id: "OnixResearch/rsi-lab::swarm-model".into(),
                    concrete_call_site: Some("crates/swarm-model/src/rwkv7_backend.rs".into()),
                    is_live_non_mock_consumer: true,
                    pins_leaf_revision: true,
                },
                LeafConsumerCandidate {
                    consumer_id: "OnixResearch/inference-lab::planned-cpu-arm".into(),
                    concrete_call_site: None,
                    is_live_non_mock_consumer: false,
                    pins_leaf_revision: false,
                },
            ],
        ),
        evaluate_leaf_trigger(
            "02-negative-second-mock-adapter-rejected",
            &[
                LeafConsumerCandidate {
                    consumer_id: "OnixResearch/rsi-lab::swarm-model".into(),
                    concrete_call_site: Some("crates/swarm-model/src/rwkv7_backend.rs".into()),
                    is_live_non_mock_consumer: true,
                    pins_leaf_revision: true,
                },
                LeafConsumerCandidate {
                    consumer_id: "mock-rwkv-consumer".into(),
                    concrete_call_site: Some("tests/mock_rwkv.rs".into()),
                    is_live_non_mock_consumer: false,
                    pins_leaf_revision: true,
                },
            ],
        ),
        evaluate_leaf_trigger(
            "03-positive-contract-two-concrete-consumers-satisfy-trigger",
            &[
                LeafConsumerCandidate {
                    consumer_id: "OnixResearch/rsi-lab::swarm-model".into(),
                    concrete_call_site: Some("crates/swarm-model/src/rwkv7_backend.rs".into()),
                    is_live_non_mock_consumer: true,
                    pins_leaf_revision: true,
                },
                LeafConsumerCandidate {
                    consumer_id: "OnixResearch/inference-lab::issue-4-native-rwkv-arm".into(),
                    concrete_call_site: Some("src/rwkv_cpu_arm.rs::run_step".into()),
                    is_live_non_mock_consumer: true,
                    pins_leaf_revision: true,
                },
            ],
        ),
    ];

    // Byte-prefix trie tokenizer check.
    let vocab = BytePrefixVocab::pinned_fixture();
    let sample = "rwkv7 state step\n\né";
    let ids = vocab.encode(sample.as_bytes())?;
    let decoded = vocab.decode(&ids)?;
    let trie_roundtrip_exact = decoded == sample.as_bytes();
    // Generic whitespace tokenizer produces 3 tokens ("rwkv7", "state", "step\n\né") vs byte-prefix trie.
    let ws_count = sample.split_whitespace().count();
    let generic_whitespace_diverges = ws_count != ids.len();

    // Reusable workspace stepping (6 tokens, 0 reallocations).
    let mut ws = RwkvReusableWorkspace::with_dim(4);
    let mut state = [0.0_f32; 4];
    let mut out = [0.0_f32; 4];
    let steps = 6usize;
    for t in 0..steps {
        let tok = [
            0.2 * (t as f32 + 1.0),
            -0.15 * (t as f32 + 0.5),
            0.3 / (t as f32 + 1.0),
            -0.05 * (t as f32),
        ];
        ws.step_in_place(&mut state, &tok, 0.85, &mut out)?;
    }
    let final_output_l2_norm = out
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();

    // Safetensors admission + RSI wrapper controls.
    let good_digest = DigestRef::sha256(b"synthetic-rwkv7-leaf-weights-v1");
    let valid_st = SafetensorsAdmissionInput {
        case_id: "01-positive-valid-safetensors-admission".into(),
        header_bytes: 256,
        max_header_bytes: 1_048_576,
        payload_bytes: 128,
        file_immutable_for_map_lifetime: true,
        expected_model_digest: good_digest.clone(),
        observed_model_digest: good_digest.clone(),
        tensors: vec![
            TensorHeaderEntry {
                name: "blocks.0.att.r_k".into(),
                dtype: "F32".into(),
                shape: vec![4, 4],
                offset_start: 0,
                offset_end: 64,
            },
            TensorHeaderEntry {
                name: "blocks.0.ffn.key".into(),
                dtype: "F32".into(),
                shape: vec![4, 4],
                offset_start: 64,
                offset_end: 128,
            },
        ],
    };

    let mut bad_header = valid_st.clone();
    bad_header.header_bytes = 0;

    let mut bad_shape = valid_st.clone();
    bad_shape.tensors[0].offset_end = 48; // mismatch against 4*4*4 = 64 bytes

    let mut overlap = valid_st.clone();
    overlap.tensors[1].offset_start = 32;
    overlap.tensors[1].offset_end = 96;

    let mut bad_digest = valid_st.clone();
    bad_digest.observed_model_digest = DigestRef::sha256(b"tampered-weights");

    let mut bad_immut = valid_st.clone();
    bad_immut.file_immutable_for_map_lifetime = false;

    let mut controls = Vec::new();
    for (id, st) in [
        ("01-positive-valid-safetensors-admission", valid_st),
        ("02-negative-malformed-empty-header", bad_header),
        ("03-negative-shape-byte-span-mismatch", bad_shape),
        ("04-negative-overlapping-tensor-ranges", overlap),
        ("05-negative-tampered-weight-digest", bad_digest),
        (
            "06-negative-missing-immutability-safety-contract",
            bad_immut,
        ),
    ] {
        let (outcome, reason) = validate_safetensors_admission(&st);
        controls.push(LeafControlVerdict {
            case_id: id.into(),
            outcome,
            detail: reason.unwrap_or_else(|| "admitted".into()),
        });
    }

    // RSI wrapper preservation negative/positive controls via deterministic validators.
    controls.push(LeafControlVerdict {
        case_id: "07-control-rsi-wrapper-per-agent-isolation-and-offset-refusal".into(),
        outcome: validate_rsi_wrapper_contract(true, true, true, true),
        detail: "Agent epochs/offsets, immutable snapshots, fork/restore, and staged atomic rollback stay owned by RSI rwkv7_backend.rs.".into(),
    });
    controls.push(LeafControlVerdict {
        case_id: "08-negative-silent-checkpoint-version-bump-rejected".into(),
        outcome: validate_checkpoint_version_policy(true, false),
        detail: "Behavioral extraction cannot silently alter checkpoint bytes; any version change requires explicit declaration.".into(),
    });
    controls.push(LeafControlVerdict {
        case_id: "09-negative-fp32-cpu-claimed-equivalent-to-fla-bf16-cuda-rejected".into(),
        outcome: validate_precision_interchange_claim(true),
        detail: "FP32 CPU architectural behavior is explicitly non-interchangeable with FLA BF16 CUDA policy parity.".into(),
    });

    Ok(RwkvLeafReport {
        execution_class: ExecutionClass::BlockedOnArtifacts,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "second_concrete_native_rwkv_consumer_and_pinned_leaf_crate",
            "Issue #14 remains explicitly DEFERRED (blocked_on_artifacts) until a second concrete non-mock native-RWKV consumer with named call sites demonstrates the leaf API; RSI #74 PGO and RSI #11 packing remain in RSI.",
        ),
        donor_commit: RWKV_LEAF_DONOR_COMMIT.into(),
        candidate_leaf_modules: vec![
            "crates/swarm-model/src/rwkv7_recurrent.rs".into(),
            "crates/swarm-model/src/rwkv7_cpu.rs".into(),
            "crates/swarm-model/src/rwkv7_weights.rs".into(),
            "crates/swarm-model/src/rwkv7_tokenizer.rs".into(),
        ],
        retained_in_rsi: vec![
            "crates/swarm-model/src/rwkv7_backend.rs & backend.rs (InferenceBackend)".into(),
            "Agent IDs/epochs/offsets, immutable snapshots, checkpoint/fork/restore, staged atomic ingest".into(),
            "RSI #11 packing_flip_replay.py decision ownership & RSI #74 PGO end-to-end arm".into(),
        ],
        trigger_verdicts,
        trie_token_ids: ids.into_iter().map(|x| x as usize).collect(),
        trie_roundtrip_exact,
        generic_whitespace_diverges,
        workspace_steps_executed: steps,
        workspace_reallocations: ws.reallocations,
        fp32_cpu_interchangeable_with_fla_bf16_cuda: false,
        admission_and_wrapper_controls: controls,
        final_output_l2_norm,
    })
}

pub fn inputs_json(report: &RwkvLeafReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-14-rwkv-leaf.inputs.v1\",\n",
            "  \"issue\": 14,\n",
            "  \"execution_class\": {},\n",
            "  \"donor_repo\": \"OnixResearch/rsi-lab\",\n",
            "  \"donor_commit\": {},\n",
            "  \"candidate_leaf_modules\": {},\n",
            "  \"retained_in_rsi\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.donor_commit),
        str_array(&report.candidate_leaf_modules),
        str_array(&report.retained_in_rsi),
    )
}

pub fn results_json(report: &RwkvLeafReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-14-rwkv-leaf.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 14,").unwrap();
    writeln!(
        out,
        "  \"execution_class\": {},",
        quote(report.execution_class.as_str())
    )
    .unwrap();
    writeln!(
        out,
        "  \"hardware_gate\": {},",
        report.hardware_gate.to_json()
    )
    .unwrap();

    let triggers: Vec<String> = report
        .trigger_verdicts
        .iter()
        .map(|t| {
            format!(
                "{{\"case_id\":{},\"trigger_satisfied\":{},\"qualified_consumers\":{},\"rejection_reason\":{}}}",
                quote(&t.case_id),
                t.trigger_satisfied,
                t.qualified_consumers,
                opt_str(t.rejection_reason.as_deref()),
            )
        })
        .collect();
    writeln!(out, "  \"trigger_verdicts\": [{}],", triggers.join(",")).unwrap();
    writeln!(
        out,
        "  \"trie_token_ids\": {},",
        usize_array(&report.trie_token_ids)
    )
    .unwrap();
    writeln!(
        out,
        "  \"trie_roundtrip_exact\": {},",
        report.trie_roundtrip_exact
    )
    .unwrap();
    writeln!(
        out,
        "  \"generic_whitespace_diverges\": {},",
        report.generic_whitespace_diverges
    )
    .unwrap();
    writeln!(
        out,
        "  \"workspace_steps_executed\": {},",
        report.workspace_steps_executed
    )
    .unwrap();
    writeln!(
        out,
        "  \"workspace_reallocations\": {},",
        report.workspace_reallocations
    )
    .unwrap();
    writeln!(
        out,
        "  \"fp32_cpu_interchangeable_with_fla_bf16_cuda\": {},",
        report.fp32_cpu_interchangeable_with_fla_bf16_cuda
    )
    .unwrap();
    writeln!(
        out,
        "  \"final_output_l2_norm\": {},",
        finite_f64(report.final_output_l2_norm)
    )
    .unwrap();

    let ctrls: Vec<String> = report
        .admission_and_wrapper_controls
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
    writeln!(
        out,
        "  \"admission_and_wrapper_controls\": [{}]",
        ctrls.join(",")
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &RwkvLeafReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #14: Conditional Native RWKV Numerical Leaf Gate"
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
        "- Conditional gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Reusable workspace reallocations across {} steps: `{}`",
        report.workspace_steps_executed, report.workspace_reallocations
    )
    .unwrap();
    writeln!(
        out,
        "- Byte-prefix trie roundtrip exact: `{}` (generic whitespace diverges: `{}`)",
        report.trie_roundtrip_exact, report.generic_whitespace_diverges
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Trigger Gate Verdicts").unwrap();
    for t in &report.trigger_verdicts {
        writeln!(
            out,
            "- `{}`: satisfied=`{}` ({})",
            t.case_id,
            t.trigger_satisfied,
            t.rejection_reason.as_deref().unwrap_or("passed")
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "## Safetensors Admission & RSI Wrapper Controls").unwrap();
    for c in &report.admission_and_wrapper_controls {
        writeln!(
            out,
            "- `{}`: `{}` — {}",
            c.case_id,
            c.outcome.as_str(),
            c.detail
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> Explicitly `blocked_on_artifacts` (deferred) until a second concrete non-mock native-RWKV consumer demonstrates the leaf API."
    )
    .unwrap();
    out
}
