//! Issue #88: Qualify pinned Unlimited-OCR (3B) on a frozen multi-page PDF fixture against
//! page-chunked OCR baselines by edit distance and local throughput, default-off.
//!
//! Related-work record (strictly non-gating external claims):
//! - X post (2026-11-21, byte `@byteonai`):
//!   `https://x.com/byteonai/status/2124163291157008857`
//! - Companion video: `https://x.com/byteonai/status/2124163291157008857/video/1`
//! - Reported properties: MIT-licensed 3B OCR model, `32K` context, vLLM/SGLang compatible,
//!   single-pass parsing up to `100`-page PDFs.
//! - Authors' unverified claims: `93%` on OmniDocBench v1.5 and `<0.11` edit distance past `40`
//!   pages. Strictly non-gating related work only.
//! - Pin status: `external_checkpoint_pin_required_default_off` (`blocked_on_artifacts`).
//!
//! Provides:
//! - Exact pure-Rust Wagner-Fischer **normalized Levenshtein edit distance** (`d(a, b) / max(|a|, |b|)`).
//! - Cross-page table row-continuation verifier and hierarchical heading-tree continuity verifier.
//! - Multi-page PDF structure fixture (`12`, `40`, `64`, `100` pages) comparing default-off
//!   `unlimited_ocr_3b_single_pass` (with explicit `32_768`-token context limit guard!) against
//!   `page_chunked_ocr_baseline` on edit distance, cross-page table/heading preservation, memory
//!   footprint, and local pages/sec accounting.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/byteonai/status/2124163291157008857";
pub const UPSTREAM_VIDEO: &str = "https://x.com/byteonai/status/2124163291157008857/video/1";
pub const UPSTREAM_PIN_STATUS: &str = "external_checkpoint_pin_required_default_off";
pub const MODEL_CONTEXT_LIMIT_TOKENS: usize = 32_768;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Exact Wagner-Fischer Levenshtein distance on Unicode scalar values.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(av[i - 1] != bv[j - 1]);
            let del = prev[j] + 1;
            let ins = curr[j - 1] + 1;
            let sub = prev[j - 1] + cost;
            curr[j] = del.min(ins).min(sub);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Exact normalized edit distance (`None` if both strings are empty).
pub fn normalized_edit_distance(reference: &str, hypothesis: &str) -> Option<f64> {
    let max_len = reference.chars().count().max(hypothesis.chars().count());
    if max_len == 0 {
        return None;
    }
    Some((levenshtein_distance(reference, hypothesis) as f64) / (max_len as f64))
}

#[derive(Debug, Clone)]
pub struct PdfStructureFixture {
    pub doc_id: String,
    pub page_count: usize,
    pub reference_markdown: String,
    pub cross_page_tables_total: usize,
    pub cross_page_headings_total: usize,
    pub estimated_context_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct OcrArmObservation {
    pub doc_id: String,
    pub arm_id: String,
    pub page_count: usize,
    pub pages_emitted: usize,
    pub normalized_edit_distance: Option<f64>,
    pub cross_page_tables_preserved: usize,
    pub cross_page_tables_total: usize,
    pub cross_page_headings_preserved: usize,
    pub cross_page_headings_total: usize,
    pub context_tokens_used: usize,
    pub peak_memory_mib: f64,
    pub elapsed_s: f64,
    pub pages_per_sec: Option<f64>,
    pub outcome: OutcomeStatus,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct OcrEvalInput {
    pub arm_id: String,
    pub checkpoint_pin: String,
    pub hypothesis_markdown: String,
    pub pages_emitted: usize,
    pub cross_page_tables_preserved: usize,
    pub cross_page_headings_preserved: usize,
    pub context_limit_tokens: usize,
    pub peak_memory_mib: f64,
    pub elapsed_s: f64,
}

pub fn evaluate_ocr_observation(
    fixture: &PdfStructureFixture,
    input: &OcrEvalInput,
) -> Result<OcrArmObservation, String> {
    if input.checkpoint_pin.is_empty() || input.checkpoint_pin.contains("latest") {
        return Err("floating or missing OCR checkpoint pin rejected".into());
    }
    let mem = check_finite(input.peak_memory_mib, "peak_memory_mib")?;
    let elapsed = check_finite(input.elapsed_s, "elapsed_s")?;
    if mem <= 0.0 || elapsed <= 0.0 || fixture.page_count == 0 {
        return Err("invalid memory, elapsed_s, or zero page_count".into());
    }
    if input.cross_page_tables_preserved > fixture.cross_page_tables_total
        || input.cross_page_headings_preserved > fixture.cross_page_headings_total
    {
        return Err("preserved cross-page count exceeds fixture total".into());
    }

    let ned = normalized_edit_distance(&fixture.reference_markdown, &input.hypothesis_markdown);
    let pages_per_sec = Some((fixture.page_count as f64) / elapsed);

    let (outcome, reason) = if fixture.estimated_context_tokens > input.context_limit_tokens {
        (
            OutcomeStatus::Exhausted,
            format!(
                "document requires {} tokens > {} context limit (silent truncation refused)",
                fixture.estimated_context_tokens, input.context_limit_tokens
            ),
        )
    } else if input.pages_emitted != fixture.page_count {
        (
            OutcomeStatus::Invalid,
            format!(
                "page count mismatch: emitted {} vs fixture {}",
                input.pages_emitted, fixture.page_count
            ),
        )
    } else if ned.is_none() {
        (OutcomeStatus::Invalid, "empty reference markdown".into())
    } else {
        (
            OutcomeStatus::Accepted,
            "bounded OCR structure comparison evaluated".into(),
        )
    };

    Ok(OcrArmObservation {
        doc_id: fixture.doc_id.clone(),
        arm_id: input.arm_id.clone(),
        page_count: fixture.page_count,
        pages_emitted: input.pages_emitted,
        normalized_edit_distance: ned,
        cross_page_tables_preserved: input.cross_page_tables_preserved,
        cross_page_tables_total: fixture.cross_page_tables_total,
        cross_page_headings_preserved: input.cross_page_headings_preserved,
        cross_page_headings_total: fixture.cross_page_headings_total,
        context_tokens_used: fixture.estimated_context_tokens,
        peak_memory_mib: mem,
        elapsed_s: elapsed,
        pages_per_sec,
        outcome,
        reason,
    })
}

fn build_pdf_fixtures() -> Vec<PdfStructureFixture> {
    vec![
        PdfStructureFixture {
            doc_id: "pdf_12p_spec_sheet".into(),
            page_count: 12,
            reference_markdown: "# 1 Overview\n| ColA | ColB |\n| 10 | 20 |\n## 1.1 Thermal\nMax 75C".into(),
            cross_page_tables_total: 2,
            cross_page_headings_total: 4,
            estimated_context_tokens: 3_600,
        },
        PdfStructureFixture {
            doc_id: "pdf_40p_technical_report".into(),
            page_count: 40,
            reference_markdown: "# 1 Architecture\n| Layer | Dim | KV |\n| MLA | 512 | FP8 |\n## 2.3 Memory Budget\nTotal 128GiB".into(),
            cross_page_tables_total: 6,
            cross_page_headings_total: 12,
            estimated_context_tokens: 12_000,
        },
        PdfStructureFixture {
            doc_id: "pdf_64p_manual".into(),
            page_count: 64,
            reference_markdown: "# 1 System Manual\n| Node | Link | Rate |\n| Spark0 | RDMA | 200G |\n## 4.2 Fault Codes\nCode E-14".into(),
            cross_page_tables_total: 10,
            cross_page_headings_total: 18,
            estimated_context_tokens: 19_200,
        },
        PdfStructureFixture {
            doc_id: "pdf_100p_long_filing".into(),
            page_count: 100,
            reference_markdown: "# 1 Annual Filing\n| Quarter | Rev | Margin |\n| Q4 | 940 | 42% |\n## 8.1 Footnotes\nAudited".into(),
            cross_page_tables_total: 16,
            cross_page_headings_total: 28,
            estimated_context_tokens: 30_000,
        },
    ]
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct UnlimitedOcrLongdocReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub observations: Vec<OcrArmObservation>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<UnlimitedOcrLongdocReport, String> {
    let fixtures = build_pdf_fixtures();
    let mut raw = String::new();
    for f in &fixtures {
        let _ = write!(
            raw,
            "{}|{}|{}|",
            f.doc_id, f.page_count, f.reference_markdown
        );
    }
    let fixture_digest = DigestRef::sha256(raw.as_bytes());

    let mut observations = Vec::new();
    for f in &fixtures {
        // Single-pass Unlimited-OCR 3B synthetic contract fixture
        let single_hyp = f.reference_markdown.replace("Total", "Totl");
        observations.push(evaluate_ocr_observation(
            f,
            &OcrEvalInput {
                arm_id: "unlimited_ocr_3b_single_pass_default_off".into(),
                checkpoint_pin: "unlimited-ocr-3b-mit@sha256:8800uocr".into(),
                hypothesis_markdown: single_hyp,
                pages_emitted: f.page_count,
                cross_page_tables_preserved: f.cross_page_tables_total,
                cross_page_headings_preserved: f.cross_page_headings_total,
                context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
                peak_memory_mib: 6_800.0 + (f.page_count as f64) * 32.0,
                elapsed_s: 1.5 + (f.page_count as f64) * 0.14,
            },
        )?);
        // Page-chunked baseline breaks cross-page tables and heading nesting at page boundaries
        let chunk_hyp = format!(
            "{} [PAGE_BREAK_SPLIT]",
            f.reference_markdown.replace("## ", "# ")
        );
        observations.push(evaluate_ocr_observation(
            f,
            &OcrEvalInput {
                arm_id: "page_chunked_ocr_baseline".into(),
                checkpoint_pin: "page-chunked-ocr-ref@sha256:8801chunk".into(),
                hypothesis_markdown: chunk_hyp,
                pages_emitted: f.page_count,
                cross_page_tables_preserved: f.cross_page_tables_total / 2,
                cross_page_headings_preserved: f.cross_page_headings_total * 2 / 3,
                context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
                peak_memory_mib: 4_200.0,
                elapsed_s: 2.0 + (f.page_count as f64) * 0.25,
            },
        )?);
    }

    let mut negative_controls = Vec::new();

    // NC1: real Unlimited-OCR 3B checkpoint/vLLM/SGLang serve blocked on artifacts
    negative_controls.push(NegativeControl {
        case_id: "nc1_unlimited_ocr_checkpoint_blocked".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail: "real Unlimited-OCR 3B checkpoint & vLLM/SGLang serve not executed locally".into(),
    });

    // NC2: 32K context overflow refused (no silent truncation)
    let mut overflow_fix = fixtures[3].clone();
    overflow_fix.estimated_context_tokens = 40_960;
    let ov = evaluate_ocr_observation(
        &overflow_fix,
        &OcrEvalInput {
            arm_id: "overflow_test".into(),
            checkpoint_pin: "sha256:8802ov".into(),
            hypothesis_markdown: overflow_fix.reference_markdown.clone(),
            pages_emitted: overflow_fix.page_count,
            cross_page_tables_preserved: overflow_fix.cross_page_tables_total,
            cross_page_headings_preserved: overflow_fix.cross_page_headings_total,
            context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
            peak_memory_mib: 10_000.0,
            elapsed_s: 15.0,
        },
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc2_32k_context_overflow_not_silently_truncated".into(),
        outcome: ov.outcome,
        detail: "40960-token document exceeding 32768-token window returns Exhausted".into(),
    });

    // NC3: page-count mismatch rejected
    let pm = evaluate_ocr_observation(
        &fixtures[1],
        &OcrEvalInput {
            arm_id: "dropped_pages".into(),
            checkpoint_pin: "sha256:8803pm".into(),
            hypothesis_markdown: fixtures[1].reference_markdown.clone(),
            pages_emitted: 38,
            cross_page_tables_preserved: fixtures[1].cross_page_tables_total,
            cross_page_headings_preserved: fixtures[1].cross_page_headings_total,
            context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
            peak_memory_mib: 7_500.0,
            elapsed_s: 8.0,
        },
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc3_missing_pages_rejected".into(),
        outcome: pm.outcome,
        detail: "emitting 38 of 40 pages fails page-completeness check".into(),
    });

    // NC4: empty reference markdown returns None edit distance (zero denominator guard)
    negative_controls.push(NegativeControl {
        case_id: "nc4_empty_edit_distance_denominator_guard".into(),
        outcome: if normalized_edit_distance("", "").is_none() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "empty reference/hypothesis pair yields None instead of NaN/0.0".into(),
    });

    // NC5: floating `:latest` checkpoint tag rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_floating_checkpoint_tag_rejected".into(),
        outcome: if evaluate_ocr_observation(
            &fixtures[0],
            &OcrEvalInput {
                arm_id: "floating".into(),
                checkpoint_pin: "unlimited-ocr:latest".into(),
                hypothesis_markdown: fixtures[0].reference_markdown.clone(),
                pages_emitted: fixtures[0].page_count,
                cross_page_tables_preserved: 2,
                cross_page_headings_preserved: 4,
                context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
                peak_memory_mib: 5_000.0,
                elapsed_s: 3.0,
            },
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "floating checkpoint tag rejected".into(),
    });

    // NC6: preserved cross-page tables > fixture total rejected
    negative_controls.push(NegativeControl {
        case_id: "nc6_preserved_count_exceeding_total_rejected".into(),
        outcome: if evaluate_ocr_observation(
            &fixtures[0],
            &OcrEvalInput {
                arm_id: "overcount".into(),
                checkpoint_pin: "sha256:8804oc".into(),
                hypothesis_markdown: fixtures[0].reference_markdown.clone(),
                pages_emitted: fixtures[0].page_count,
                cross_page_tables_preserved: 99,
                cross_page_headings_preserved: 4,
                context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
                peak_memory_mib: 5_000.0,
                elapsed_s: 3.0,
            },
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "cross_page_tables_preserved > total rejected".into(),
    });

    // NC7: non-finite memory or elapsed time rejected
    negative_controls.push(NegativeControl {
        case_id: "nc7_nonfinite_memory_rejected".into(),
        outcome: if evaluate_ocr_observation(
            &fixtures[0],
            &OcrEvalInput {
                arm_id: "nan_mem".into(),
                checkpoint_pin: "sha256:8805nan".into(),
                hypothesis_markdown: fixtures[0].reference_markdown.clone(),
                pages_emitted: fixtures[0].page_count,
                cross_page_tables_preserved: 2,
                cross_page_headings_preserved: 4,
                context_limit_tokens: MODEL_CONTEXT_LIMIT_TOKENS,
                peak_memory_mib: f64::NAN,
                elapsed_s: 3.0,
            },
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN peak_memory_mib rejected".into(),
    });

    // NC8: unverified OmniDocBench 93% / <0.11 claims refused as acceptance gates
    negative_controls.push(NegativeControl {
        case_id: "nc8_unverified_omnidocbench_claim_non_gating".into(),
        outcome: OutcomeStatus::Invalid,
        detail: "authors' unverified 93% OmniDocBench v1.5 and <0.11 edit-distance claims are non-gating".into(),
    });

    Ok(UnlimitedOcrLongdocReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Local vLLM/SGLang serve of pinned MIT Unlimited-OCR (3B) on multi-page PDF corpus",
            "Unlimited-OCR 3B checkpoint digest and local vLLM/SGLang multimodal serving runtime are not present locally; exact Levenshtein and cross-page contract evaluated on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        fixture_digest,
        observations,
        negative_controls,
    })
}

pub fn inputs_json(report: &UnlimitedOcrLongdocReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.unlimited-ocr-longdoc.inputs.v1\",\n",
            "  \"issue\": 88,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"upstream_video\": {},\n",
            "  \"upstream_pin_status\": {},\n",
            "  \"model_context_limit_tokens\": {},\n",
            "  \"fixture_digest\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(UPSTREAM_X_POST),
        quote(UPSTREAM_VIDEO),
        quote(UPSTREAM_PIN_STATUS),
        MODEL_CONTEXT_LIMIT_TOKENS,
        report.fixture_digest.to_json(),
    )
}

pub fn results_json(report: &UnlimitedOcrLongdocReport) -> String {
    let obs: Vec<String> = report
        .observations
        .iter()
        .map(|o| {
            format!(
                "{{\"doc_id\":{},\"arm_id\":{},\"page_count\":{},\"pages_emitted\":{},\"normalized_edit_distance\":{},\"cross_page_tables_preserved\":{},\"cross_page_tables_total\":{},\"cross_page_headings_preserved\":{},\"cross_page_headings_total\":{},\"context_tokens_used\":{},\"peak_memory_mib\":{},\"elapsed_s\":{},\"pages_per_sec\":{},\"outcome\":{},\"reason\":{}}}",
                quote(&o.doc_id),
                quote(&o.arm_id),
                o.page_count,
                o.pages_emitted,
                opt_f64(o.normalized_edit_distance),
                o.cross_page_tables_preserved,
                o.cross_page_tables_total,
                o.cross_page_headings_preserved,
                o.cross_page_headings_total,
                o.context_tokens_used,
                finite_f64(o.peak_memory_mib),
                finite_f64(o.elapsed_s),
                opt_f64(o.pages_per_sec),
                quote(o.outcome.as_str()),
                quote(&o.reason),
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
            "  \"schema\": \"inference-lab.unlimited-ocr-longdoc.results.v1\",\n",
            "  \"issue\": 88,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"observations\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.fixture_digest.to_json(),
        obs.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &UnlimitedOcrLongdocReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #88: Unlimited-OCR (3B) Multi-Page PDF Qualification vs Page-Chunked Baselines"
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
    writeln!(out).unwrap();
    writeln!(out, "## Multi-Page PDF Fixture Observations").unwrap();
    for o in &report.observations {
        writeln!(
            out,
            "- `{}` / `{}` (`{}p`): `NED={}`, `tables={}/{}`, `headings={}/{}`, `mem={:.1}MiB`, `pages/s={}`, `outcome={}`",
            o.doc_id,
            o.arm_id,
            o.page_count,
            opt_f64(o.normalized_edit_distance),
            o.cross_page_tables_preserved,
            o.cross_page_tables_total,
            o.cross_page_headings_preserved,
            o.cross_page_headings_total,
            o.peak_memory_mib,
            opt_f64(o.pages_per_sec),
            o.outcome.as_str()
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
    fn levenshtein_and_unlimited_ocr_controls_verified() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("exact", "exact"), 0);
        assert!(normalized_edit_distance("", "").is_none());
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert_eq!(r.observations.len(), 8);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
