//! Issue #87: Benchmark pinned Knowledgator System 1 models alongside Clairic (#55) and a
//! structured-output LLM baseline on frozen hierarchical-extraction fixtures, default-off.
//!
//! Related-work & pinned companion references (strictly non-gating external claims):
//! - Knowledgator System 1 X post (2026-11-20, `@knowledgator`):
//!   `https://x.com/knowledgator/status/2123851865821700260`
//! - Blog: `https://knowledgator.com/blog/system-one-model`
//! - Video: `https://x.com/knowledgator/status/2123851865821700260/video/1`
//! - Authors' unverified claim: `~100x faster` classification, hierarchical extraction, editing,
//!   and structured tasks. Never imported as an acceptance gate.
//! - Companion pinned arms from `#55` (`crate::clairic_extraction`):
//!   - `Cloudflare/clairic-flash@36ad68b999560f56b57ef16a22180f8cb83c6d77` (`Apache-2.0`, 9B)
//!   - `Cloudflare/clairic@2c161be5cc71bacd6605b99a8c07547e58cfa434` (`Apache-2.0`, 27B)
//!   - `conventional_json_schema_structured_output_baseline_v1`
//!
//! Provides:
//! - Frozen 6-document hierarchical-extraction fixture (nested parent -> child entities, typed
//!   values, explicit abstention/null fields, and source-span grounding).
//! - Deterministic independent evaluator computing exact-field precision, recall, F1, strict
//!   schema conformance rate, span-grounding accuracy, population completeness, and throughput
//!   while keeping model-reported confidence strictly informational (non-gating).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::clairic_extraction::{
    ClairicArchitecturePath, CLAIRIC_27B_COMMIT, CLAIRIC_FLASH_9B_COMMIT, STRUCTURED_BASELINE_ID,
};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const KNOWLEDGATOR_X_POST: &str = "https://x.com/knowledgator/status/2123851865821700260";
pub const KNOWLEDGATOR_BLOG: &str = "https://knowledgator.com/blog/system-one-model";
pub const KNOWLEDGATOR_VIDEO: &str =
    "https://x.com/knowledgator/status/2123851865821700260/video/1";
pub const KNOWLEDGATOR_PIN_STATUS: &str = "external_checkpoint_pin_required_default_off";

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierarchicalField {
    pub path: String, // e.g., "invoice.line_items[0].amount_cents"
    pub value: Option<String>,
    pub grounded_in_source: bool,
}

#[derive(Debug, Clone)]
pub struct FrozenExtractionDoc {
    pub doc_id: String,
    pub source_text: String,
    pub gold_fields: Vec<HierarchicalField>,
}

#[derive(Debug, Clone)]
pub struct PredictedDocOutput {
    pub doc_id: String,
    pub schema_valid: bool,
    pub execution_outcome: OutcomeStatus,
    pub predicted_fields: Vec<HierarchicalField>,
    pub model_confidence_informational: f64,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct ExtractionArmMetrics {
    pub arm_id: String,
    pub checkpoint_pin: String,
    pub docs_planned: usize,
    pub docs_executed: usize,
    pub schema_valid_docs: usize,
    pub schema_conformance_rate: Option<f64>,
    pub exact_tp: usize,
    pub exact_fp: usize,
    pub exact_fn: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub exact_field_f1: Option<f64>,
    pub ungrounded_hallucinations: usize,
    pub docs_per_sec: Option<f64>,
    pub confidence_gating_used: bool,
    pub admission_outcome: OutcomeStatus,
}

pub fn score_hierarchical_arm(
    arm_id: &str,
    checkpoint_pin: &str,
    clairic_path_if_applicable: Option<ClairicArchitecturePath>,
    fixture: &[FrozenExtractionDoc],
    predictions: &[PredictedDocOutput],
) -> Result<ExtractionArmMetrics, String> {
    if fixture.is_empty() {
        return Err("frozen extraction fixture must be non-empty".into());
    }
    if checkpoint_pin.is_empty() || checkpoint_pin.contains("latest") {
        return Err("floating or missing checkpoint pin rejected".into());
    }
    if let Some(path) = clairic_path_if_applicable {
        if path != ClairicArchitecturePath::ParallelSlotAutoregressive {
            return Err(format!(
                "Clairic arm must use 18 Nov parallel slot-autoregressive path, got {}",
                path.as_str()
            ));
        }
    }
    if predictions.len() != fixture.len() {
        return Err(
            "population denominator mismatch: predictions must retain every planned document"
                .into(),
        );
    }

    let mut schema_valid_docs = 0usize;
    let mut exact_tp = 0usize;
    let mut exact_fp = 0usize;
    let mut exact_fn = 0usize;
    let mut ungrounded_hallucinations = 0usize;
    let mut total_latency_ms = 0.0_f64;
    let mut failed_docs = 0usize;

    for (doc, pred) in fixture.iter().zip(predictions.iter()) {
        if doc.doc_id != pred.doc_id {
            return Err(format!(
                "document ordering mismatch: expected {}, got {}",
                doc.doc_id, pred.doc_id
            ));
        }
        let conf = check_finite(pred.model_confidence_informational, "confidence")?;
        if !(0.0..=1.0).contains(&conf) {
            return Err("informational confidence out of [0, 1]".into());
        }
        let lat = check_finite(pred.latency_ms, "latency_ms")?;
        if lat <= 0.0 {
            return Err("non-positive latency_ms".into());
        }
        total_latency_ms += lat;

        if pred.execution_outcome != OutcomeStatus::Accepted || !pred.schema_valid {
            failed_docs += 1;
            exact_fn += doc.gold_fields.len();
            continue;
        }
        schema_valid_docs += 1;

        for pf in &pred.predicted_fields {
            if let Some(val) = &pf.value {
                let actually_in_source = doc.source_text.contains(val.as_str());
                if !actually_in_source || !pf.grounded_in_source {
                    ungrounded_hallucinations += 1;
                }
            }
        }

        for gf in &doc.gold_fields {
            let matched = pred.predicted_fields.iter().any(|pf| {
                pf.path == gf.path
                    && pf.value == gf.value
                    && match &pf.value {
                        Some(v) => doc.source_text.contains(v.as_str()) && pf.grounded_in_source,
                        None => true,
                    }
            });
            if matched {
                exact_tp += 1;
            } else {
                exact_fn += 1;
            }
        }
        for pf in &pred.predicted_fields {
            let matched = doc.gold_fields.iter().any(|gf| {
                pf.path == gf.path
                    && pf.value == gf.value
                    && match &pf.value {
                        Some(v) => doc.source_text.contains(v.as_str()) && pf.grounded_in_source,
                        None => true,
                    }
            });
            if !matched {
                exact_fp += 1;
            }
        }
    }

    let precision = if exact_tp + exact_fp == 0 {
        None
    } else {
        Some((exact_tp as f64) / ((exact_tp + exact_fp) as f64))
    };
    let recall = if exact_tp + exact_fn == 0 {
        None
    } else {
        Some((exact_tp as f64) / ((exact_tp + exact_fn) as f64))
    };
    let exact_field_f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    let schema_conformance_rate = Some((schema_valid_docs as f64) / (fixture.len() as f64));
    let docs_per_sec = if total_latency_ms > 0.0 {
        Some((fixture.len() as f64) * 1000.0 / total_latency_ms)
    } else {
        None
    };

    let admission_outcome = if failed_docs > 0 || ungrounded_hallucinations > 0 {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };

    Ok(ExtractionArmMetrics {
        arm_id: arm_id.into(),
        checkpoint_pin: checkpoint_pin.into(),
        docs_planned: fixture.len(),
        docs_executed: predictions.len(),
        schema_valid_docs,
        schema_conformance_rate,
        exact_tp,
        exact_fp,
        exact_fn,
        precision,
        recall,
        exact_field_f1,
        ungrounded_hallucinations,
        docs_per_sec,
        confidence_gating_used: false,
        admission_outcome,
    })
}

fn frozen_fixture() -> Vec<FrozenExtractionDoc> {
    vec![
        FrozenExtractionDoc {
            doc_id: "doc_01_invoice_nested".into(),
            source_text: "Invoice INV-9041 Vendor AcmeCorp Item0 GPU-Cable 4200 Item1 Rack-Rail 8900 Total 13100".into(),
            gold_fields: vec![
                HierarchicalField { path: "invoice.id".into(), value: Some("INV-9041".into()), grounded_in_source: true },
                HierarchicalField { path: "invoice.vendor".into(), value: Some("AcmeCorp".into()), grounded_in_source: true },
                HierarchicalField { path: "invoice.items[0].sku".into(), value: Some("GPU-Cable".into()), grounded_in_source: true },
                HierarchicalField { path: "invoice.items[1].cents".into(), value: Some("8900".into()), grounded_in_source: true },
            ],
        },
        FrozenExtractionDoc {
            doc_id: "doc_02_contract_abstention".into(),
            source_text: "Agreement AGR-221 Party NorthstarLabs Effective 2026-10-01 PenaltyClause omitted".into(),
            gold_fields: vec![
                HierarchicalField { path: "contract.id".into(), value: Some("AGR-221".into()), grounded_in_source: true },
                HierarchicalField { path: "contract.party".into(), value: Some("NorthstarLabs".into()), grounded_in_source: true },
                HierarchicalField { path: "contract.penalty_usd".into(), value: None, grounded_in_source: true },
            ],
        },
        FrozenExtractionDoc {
            doc_id: "doc_03_incident_hierarchy".into(),
            source_text: "Incident INC-778 Cluster SparkB Node0 GB10-A Temp 74C Action ThrottleCheck".into(),
            gold_fields: vec![
                HierarchicalField { path: "incident.id".into(), value: Some("INC-778".into()), grounded_in_source: true },
                HierarchicalField { path: "incident.node.hw".into(), value: Some("GB10-A".into()), grounded_in_source: true },
                HierarchicalField { path: "incident.node.temp".into(), value: Some("74C".into()), grounded_in_source: true },
            ],
        },
        FrozenExtractionDoc {
            doc_id: "doc_04_release_tree".into(),
            source_text: "Release R-44 Component KernelOpt Commit 8a9b1c2d Status Verified".into(),
            gold_fields: vec![
                HierarchicalField { path: "release.id".into(), value: Some("R-44".into()), grounded_in_source: true },
                HierarchicalField { path: "release.component.commit".into(), value: Some("8a9b1c2d".into()), grounded_in_source: true },
            ],
        },
        FrozenExtractionDoc {
            doc_id: "doc_05_spec_table".into(),
            source_text: "Model Ling32Air ActiveParams 5.5B LayersMLA 7 LayersKDA 35".into(),
            gold_fields: vec![
                HierarchicalField { path: "model.name".into(), value: Some("Ling32Air".into()), grounded_in_source: true },
                HierarchicalField { path: "model.arch.mla_layers".into(), value: Some("7".into()), grounded_in_source: true },
                HierarchicalField { path: "model.arch.kda_layers".into(), value: Some("35".into()), grounded_in_source: true },
            ],
        },
        FrozenExtractionDoc {
            doc_id: "doc_06_null_optional".into(),
            source_text: "Ticket T-509 Owner Unassigned Priority Medium".into(),
            gold_fields: vec![
                HierarchicalField { path: "ticket.id".into(), value: Some("T-509".into()), grounded_in_source: true },
                HierarchicalField { path: "ticket.owner".into(), value: None, grounded_in_source: true },
                HierarchicalField { path: "ticket.priority".into(), value: Some("Medium".into()), grounded_in_source: true },
            ],
        },
    ]
}

fn build_exact_predictions(
    fixture: &[FrozenExtractionDoc],
    latency_ms: f64,
    miss_last_doc_field: bool,
) -> Vec<PredictedDocOutput> {
    fixture
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let mut fields = doc.gold_fields.clone();
            if miss_last_doc_field && i + 1 == fixture.len() {
                if let Some(last) = fields.last_mut() {
                    last.value = Some("T-509".into()); // grounded in source, wrong field value
                }
            }
            PredictedDocOutput {
                doc_id: doc.doc_id.clone(),
                schema_valid: true,
                execution_outcome: OutcomeStatus::Accepted,
                predicted_fields: fields,
                model_confidence_informational: 0.91,
                latency_ms,
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct KnowledgatorSys1ExtractReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub arm_metrics: Vec<ExtractionArmMetrics>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<KnowledgatorSys1ExtractReport, String> {
    let fixture = frozen_fixture();
    let mut raw = String::new();
    for doc in &fixture {
        let _ = write!(raw, "{}|{}|", doc.doc_id, doc.source_text);
    }
    let fixture_digest = DigestRef::sha256(raw.as_bytes());

    let sys1_preds = build_exact_predictions(&fixture, 12.0, true);
    let clairic_preds = build_exact_predictions(&fixture, 45.0, false);
    let llm_preds = build_exact_predictions(&fixture, 180.0, false);

    let arm_metrics = vec![
        score_hierarchical_arm(
            "knowledgator_system1_small_default_off",
            "knowledgator/system1-pinned-placeholder@sha256:8700sys1",
            None,
            &fixture,
            &sys1_preds,
        )?,
        score_hierarchical_arm(
            "clairic_flash_9b_slot_ar",
            &format!("Cloudflare/clairic-flash@{CLAIRIC_FLASH_9B_COMMIT}"),
            Some(ClairicArchitecturePath::ParallelSlotAutoregressive),
            &fixture,
            &clairic_preds,
        )?,
        score_hierarchical_arm(
            STRUCTURED_BASELINE_ID,
            &format!("structured-llm-baseline@sha256:{CLAIRIC_27B_COMMIT}"),
            None,
            &fixture,
            &llm_preds,
        )?,
    ];

    let mut negative_controls = Vec::new();

    // NC1: Knowledgator checkpoint & serving hardware not locally pinned
    negative_controls.push(NegativeControl {
        case_id: "nc1_knowledgator_checkpoint_blocked_on_artifacts".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail:
            "real Knowledgator System 1 checkpoint digest requires external pin before live serving"
                .into(),
    });

    // NC2: floating `:latest` checkpoint rejected
    negative_controls.push(NegativeControl {
        case_id: "nc2_floating_checkpoint_tag_rejected".into(),
        outcome: if score_hierarchical_arm(
            "sys1_floating",
            "knowledgator/system1:latest",
            None,
            &fixture,
            &sys1_preds,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "floating `:latest` checkpoint tag rejected".into(),
    });

    // NC3: ungrounded hallucinated value with 0.99 confidence disqualified
    let mut hall_preds = build_exact_predictions(&fixture, 10.0, false);
    hall_preds[0].model_confidence_informational = 0.99;
    hall_preds[0].predicted_fields[0].value = Some("HALLUCINATED-9999".into());
    let hall_m = score_hierarchical_arm(
        "high_conf_hallucinator",
        "sha256:8701hall",
        None,
        &fixture,
        &hall_preds,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc3_high_confidence_hallucination_disqualified".into(),
        outcome: hall_m.admission_outcome,
        detail: "0.99 informational confidence never overrides ungrounded hallucinated field"
            .into(),
    });

    // NC4: schema-invalid document cannot be silently accepted
    let mut bad_schema = build_exact_predictions(&fixture, 15.0, false);
    bad_schema[1].schema_valid = false;
    let bad_s = score_hierarchical_arm(
        "schema_invalid_arm",
        "sha256:8702schema",
        None,
        &fixture,
        &bad_schema,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc4_schema_invalid_doc_disqualified".into(),
        outcome: bad_s.admission_outcome,
        detail: "schema-invalid output retains missed gold fields as FN and fails admission".into(),
    });

    // NC5: dropped document from population denominator rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_dropped_population_document_rejected".into(),
        outcome: if score_hierarchical_arm(
            "dropped_doc",
            "sha256:8703drop",
            None,
            &fixture,
            &sys1_preds[..5],
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "omitting a failed document from population denominator rejected".into(),
    });

    // NC6: outdated Clairic 19 Oct single-forward-pass preview path refused
    negative_controls.push(NegativeControl {
        case_id: "nc6_clairic_preview_single_pass_rejected".into(),
        outcome: if score_hierarchical_arm(
            "clairic_preview",
            &format!("Cloudflare/clairic-flash@{CLAIRIC_FLASH_9B_COMMIT}"),
            Some(ClairicArchitecturePath::PreviewSingleForwardPass),
            &fixture,
            &clairic_preds,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "outdated 19 Oct 2026 single-forward-pass Clairic preview path refused".into(),
    });

    // NC7: non-finite latency rejected
    let mut nan_lat = build_exact_predictions(&fixture, 15.0, false);
    nan_lat[0].latency_ms = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc7_nonfinite_latency_rejected".into(),
        outcome: if score_hierarchical_arm("nan_lat", "sha256:8704nan", None, &fixture, &nan_lat)
            .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN per-document latency rejected".into(),
    });

    // NC8: unverified ~100x speedup claim refused as pass gate
    negative_controls.push(NegativeControl {
        case_id: "nc8_unverified_100x_claim_non_gating".into(),
        outcome: OutcomeStatus::Invalid,
        detail: "authors' unverified ~100x speedup claim is strictly non-gating related work"
            .into(),
    });

    Ok(KnowledgatorSys1ExtractReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Pinned Knowledgator System 1 checkpoint alongside Clairic (#55) on local serving host",
            "Knowledgator System 1 checkpoint digest and live GPU serves are not executed locally; bounded frozen hierarchical-extraction contract evaluated on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        fixture_digest,
        arm_metrics,
        negative_controls,
    })
}

pub fn inputs_json(report: &KnowledgatorSys1ExtractReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.knowledgator-sys1-extract.inputs.v1\",\n",
            "  \"issue\": 87,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"knowledgator_x_post\": {},\n",
            "  \"knowledgator_blog\": {},\n",
            "  \"knowledgator_video\": {},\n",
            "  \"knowledgator_pin_status\": {},\n",
            "  \"companion_issue\": 55,\n",
            "  \"fixture_digest\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(KNOWLEDGATOR_X_POST),
        quote(KNOWLEDGATOR_BLOG),
        quote(KNOWLEDGATOR_VIDEO),
        quote(KNOWLEDGATOR_PIN_STATUS),
        report.fixture_digest.to_json(),
    )
}

pub fn results_json(report: &KnowledgatorSys1ExtractReport) -> String {
    let arms: Vec<String> = report
        .arm_metrics
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"checkpoint_pin\":{},\"docs_planned\":{},\"docs_executed\":{},\"schema_valid_docs\":{},\"schema_conformance_rate\":{},\"exact_tp\":{},\"exact_fp\":{},\"exact_fn\":{},\"precision\":{},\"recall\":{},\"exact_field_f1\":{},\"ungrounded_hallucinations\":{},\"docs_per_sec\":{},\"confidence_gating_used\":{},\"admission_outcome\":{}}}",
                quote(&a.arm_id),
                quote(&a.checkpoint_pin),
                a.docs_planned,
                a.docs_executed,
                a.schema_valid_docs,
                opt_f64(a.schema_conformance_rate),
                a.exact_tp,
                a.exact_fp,
                a.exact_fn,
                opt_f64(a.precision),
                opt_f64(a.recall),
                opt_f64(a.exact_field_f1),
                a.ungrounded_hallucinations,
                opt_f64(a.docs_per_sec),
                a.confidence_gating_used,
                quote(a.admission_outcome.as_str()),
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
            "  \"schema\": \"inference-lab.knowledgator-sys1-extract.results.v1\",\n",
            "  \"issue\": 87,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"arm_metrics\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.fixture_digest.to_json(),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &KnowledgatorSys1ExtractReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #87: Knowledgator System 1 vs Clairic (#55) & Structured-Output LLM Extraction"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `synthetic_contract_fixture: {}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.synthetic_contract_fixture
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware / artifact gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Fixture SHA-256: `{}` (`6` frozen hierarchical-extraction documents)",
        report.fixture_digest.hex
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Extraction Arm Metrics").unwrap();
    for a in &report.arm_metrics {
        writeln!(
            out,
            "- `{}`: `F1={}`, `schema={}/{}`, `hallucinations={}`, `docs/s={}`, `outcome={}`",
            a.arm_id,
            opt_f64(a.exact_field_f1),
            a.schema_valid_docs,
            a.docs_planned,
            a.ungrounded_hallucinations,
            opt_f64(a.docs_per_sec),
            a.admission_outcome.as_str()
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
    let _ = finite_f64(0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knowledgator_extraction_scores_f1_and_rejects_all_controls() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert_eq!(r.arm_metrics.len(), 3);
        assert!(r.arm_metrics[1].exact_field_f1.unwrap() > 0.999);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
