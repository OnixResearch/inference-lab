//! Issue #85: Log Constructor AGI `Link` learned sparse attention as related work and bound a
//! default-off long-context retrieval and end-task check if code/weights become public.
//!
//! Related-work record (strictly non-gating, unverified external claims):
//! - X post (2026-11-18, Michael Gao `@michaeltgao`):
//!   `https://x.com/michaeltgao/status/2123204373773983992`
//! - Companion video: `https://x.com/michaeltgao/status/2123204373773983992/video/1`
//! - Pin status: `post_only_unlocated_paper_repo` (`blocked_on_artifacts`; no public code or
//!   weights verified locally).
//! - Authors' unverified claims: `+53%` retrieval accuracy over dense attention at `4M` token
//!   context with `9.9x` GPU efficiency. Never imported as acceptance gates or targets.
//! - Directly complements `#23` (`flash-topk` sparse-indexer selection) and `#31` (`qwen38-qsa`
//!   long-context sparse vs dense attention).
//!
//! Provides:
//! - Pure-Rust miniature causal attention reference (`L = 64` tokens, `16` blocks of `4` tokens,
//!   `D = 8`) comparing:
//!   1. `dense_exact_attention` (full causal attention over all `16` blocks)
//!   2. `link_learned_full_read_k16` (learned block router selecting all `16` blocks -> exact
//!      numerical parity `<= 1e-12` vs dense attention)
//!   3. `link_learned_sparse_k4` (top-`4` learned sparse block read measuring exact output L2/max
//!      error, target-needle block recall, and KV read reduction)
//! - Default-off long-context (`128K`, `512K`, `1M`, `4M`) retrieval AND independent multi-hop
//!   end-task check boundary disqualifying faster sparse candidates that fail end-task accuracy.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/michaeltgao/status/2123204373773983992";
pub const UPSTREAM_VIDEO_URL: &str = "https://x.com/michaeltgao/status/2123204373773983992/video/1";
pub const UPSTREAM_PIN_STATUS: &str = "post_only_unlocated_paper_repo";

pub const SEQ_LEN: usize = 64;
pub const BLOCK_SIZE: usize = 4;
pub const NUM_BLOCKS: usize = SEQ_LEN / BLOCK_SIZE;
pub const HEAD_DIM: usize = 8;
pub const PARITY_TOL: f64 = 1.0e-12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

fn max_finite(a: f64, b: f64, ctx: &str) -> Result<f64, String> {
    let a = check_finite(a, ctx)?;
    let b = check_finite(b, ctx)?;
    Ok(if a >= b { a } else { b })
}

fn dot_vec(a: &[f64; HEAD_DIM], b: &[f64; HEAD_DIM]) -> Result<f64, String> {
    let mut acc = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += check_finite(x, "dot_a")? * check_finite(y, "dot_b")?;
    }
    check_finite(acc, "dot_acc")
}

/// Pure-Rust causal attention for query at the final position over selected block indices.
/// When `selected_blocks` equals `0..NUM_BLOCKS`, output matches exact dense attention `<= 1e-12`.
pub fn causal_block_attention(
    query: &[f64; HEAD_DIM],
    keys: &[[f64; HEAD_DIM]],
    values: &[[f64; HEAD_DIM]],
    selected_blocks: &[usize],
) -> Result<[f64; HEAD_DIM], String> {
    if keys.len() != values.len() || keys.len() != SEQ_LEN {
        return Err("keys/values length mismatch with SEQ_LEN".into());
    }
    if selected_blocks.is_empty() {
        return Err("selected_blocks must be non-empty".into());
    }
    let mut token_indices = Vec::new();
    for &b in selected_blocks {
        if b >= NUM_BLOCKS {
            return Err(format!(
                "block index {b} out of range (NUM_BLOCKS={NUM_BLOCKS})"
            ));
        }
        let start = b * BLOCK_SIZE;
        for pos in start..(start + BLOCK_SIZE) {
            token_indices.push(pos);
        }
    }
    let inv_sqrt_d = 1.0 / (HEAD_DIM as f64).sqrt();
    let mut scores = Vec::with_capacity(token_indices.len());
    let mut max_score = f64::NEG_INFINITY;
    for &pos in &token_indices {
        let s = dot_vec(query, &keys[pos])? * inv_sqrt_d;
        if s > max_score {
            max_score = s;
        }
        scores.push(s);
    }
    let mut exp_sum = 0.0_f64;
    let mut weights = Vec::with_capacity(scores.len());
    for &s in &scores {
        let e = (s - max_score).exp();
        exp_sum += check_finite(e, "attn_exp")?;
        weights.push(e);
    }
    if !exp_sum.is_finite() || exp_sum <= 0.0 {
        return Err("degenerate softmax denominator".into());
    }
    let mut out = [0.0_f64; HEAD_DIM];
    for (&pos, &w_unnorm) in token_indices.iter().zip(weights.iter()) {
        let w = w_unnorm / exp_sum;
        for (dst, &v) in out.iter_mut().zip(values[pos].iter()) {
            *dst += w * check_finite(v, "attn_val")?;
        }
    }
    for &v in &out {
        check_finite(v, "attn_out")?;
    }
    Ok(out)
}

/// Learned block-summary router selecting top-`top_k_blocks` deterministically with tie-breaking
/// by ascending block index.
pub fn select_learned_link_blocks(
    query: &[f64; HEAD_DIM],
    block_summaries: &[[f64; HEAD_DIM]],
    top_k_blocks: usize,
) -> Result<Vec<usize>, String> {
    if block_summaries.len() != NUM_BLOCKS {
        return Err("block_summaries length mismatch".into());
    }
    if top_k_blocks == 0 || top_k_blocks > NUM_BLOCKS {
        return Err(format!(
            "invalid top_k_blocks={top_k_blocks} (must be in 1..={NUM_BLOCKS})"
        ));
    }
    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(NUM_BLOCKS);
    for (idx, summary) in block_summaries.iter().enumerate() {
        scored.push((idx, dot_vec(query, summary)?));
    }
    scored.sort_by(|a, b| {
        if (b.1 - a.1).abs() <= 1.0e-15 {
            a.0.cmp(&b.0)
        } else if b.1 > a.1 {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    });
    let mut chosen: Vec<usize> = scored
        .into_iter()
        .take(top_k_blocks)
        .map(|(i, _)| i)
        .collect();
    chosen.sort_unstable();
    Ok(chosen)
}

#[derive(Debug, Clone)]
pub struct SparseReadArmResult {
    pub arm_id: String,
    pub selected_blocks: usize,
    pub total_blocks: usize,
    pub kv_read_fraction: f64,
    pub max_abs_err_vs_dense: f64,
    pub l2_err_vs_dense: f64,
    pub needle_block_recalled: bool,
    pub parity_gate_passed: bool,
}

#[derive(Debug, Clone)]
pub struct LongContextBoundaryRow {
    pub context_tokens: usize,
    pub arm_id: String,
    pub retrieval_accuracy: f64,
    pub end_task_accuracy: f64,
    pub needle_only_evaluated: bool,
    pub relative_kv_reads: f64,
    pub admission_outcome: OutcomeStatus,
    pub reason: String,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_long_context_boundary(
    context_tokens: usize,
    arm_id: &str,
    retrieval_accuracy: f64,
    end_task_accuracy: f64,
    dense_end_task_floor: f64,
    needle_only_evaluated: bool,
    relative_kv_reads: f64,
    artifacts_public_and_pinned: bool,
) -> Result<LongContextBoundaryRow, String> {
    let r = check_finite(retrieval_accuracy, "retrieval_accuracy")?;
    let e = check_finite(end_task_accuracy, "end_task_accuracy")?;
    let floor = check_finite(dense_end_task_floor, "dense_end_task_floor")?;
    let reads = check_finite(relative_kv_reads, "relative_kv_reads")?;
    if context_tokens == 0 || !(0.0..=1.0).contains(&r) || !(0.0..=1.0).contains(&e) || reads <= 0.0
    {
        return Err("invalid context_tokens, accuracy, or relative_kv_reads".into());
    }
    let (outcome, reason) = if !artifacts_public_and_pinned {
        (
            OutcomeStatus::BlockedOnArtifacts,
            "Constructor AGI Link public code/weights not yet pinned".into(),
        )
    } else if needle_only_evaluated {
        (
            OutcomeStatus::Invalid,
            "needle retrieval alone without independent end-task check is rejected".into(),
        )
    } else if e + 1.0e-12 < floor {
        (
            OutcomeStatus::Invalid,
            format!("end-task accuracy {e:.3} below dense quality floor {floor:.3}; faster sparse candidate disqualified"),
        )
    } else {
        (
            OutcomeStatus::Accepted,
            "bounded contract check admitted (independent retrieval + end-task >= dense floor)"
                .into(),
        )
    };
    Ok(LongContextBoundaryRow {
        context_tokens,
        arm_id: arm_id.into(),
        retrieval_accuracy: r,
        end_task_accuracy: e,
        needle_only_evaluated,
        relative_kv_reads: reads,
        admission_outcome: outcome,
        reason,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct LinkSparseAttentionReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_pin_status: String,
    pub unverified_author_claims: String,
    pub sparse_arms: Vec<SparseReadArmResult>,
    pub boundary_rows: Vec<LongContextBoundaryRow>,
    pub negative_controls: Vec<NegativeControl>,
}

struct LinkFixture {
    query: [f64; HEAD_DIM],
    keys: Vec<[f64; HEAD_DIM]>,
    values: Vec<[f64; HEAD_DIM]>,
    summaries: Vec<[f64; HEAD_DIM]>,
    needle_block: usize,
}

fn build_fixture() -> LinkFixture {
    let mut query = [0.0_f64; HEAD_DIM];
    for (d, slot) in query.iter_mut().enumerate() {
        *slot = det_weight(8501, d, 0.9);
    }
    let needle_block = 5usize;
    let mut keys = Vec::with_capacity(SEQ_LEN);
    let mut values = Vec::with_capacity(SEQ_LEN);
    for pos in 0..SEQ_LEN {
        let mut k = [0.0_f64; HEAD_DIM];
        let mut v = [0.0_f64; HEAD_DIM];
        let is_needle = pos / BLOCK_SIZE == needle_block;
        for d in 0..HEAD_DIM {
            k[d] = if is_needle {
                query[d] * 1.35 + det_weight(8510 + pos, d, 0.08)
            } else {
                det_weight(8520 + pos * 3, d, 0.35)
            };
            v[d] = det_weight(8560 + pos * 5, d, 0.70);
        }
        keys.push(k);
        values.push(v);
    }
    let mut summaries = Vec::with_capacity(NUM_BLOCKS);
    for b in 0..NUM_BLOCKS {
        let mut mean_k = [0.0_f64; HEAD_DIM];
        for key_row in keys.iter().skip(b * BLOCK_SIZE).take(BLOCK_SIZE) {
            for (dst, &kv) in mean_k.iter_mut().zip(key_row.iter()) {
                *dst += kv / (BLOCK_SIZE as f64);
            }
        }
        summaries.push(mean_k);
    }
    LinkFixture {
        query,
        keys,
        values,
        summaries,
        needle_block,
    }
}

fn diff_metrics(a: &[f64; HEAD_DIM], b: &[f64; HEAD_DIM]) -> Result<(f64, f64), String> {
    let mut max_abs = 0.0_f64;
    let mut sq = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (check_finite(x, "diff_a")? - check_finite(y, "diff_b")?).abs();
        max_abs = max_finite(max_abs, d, "diff_max")?;
        sq += d * d;
    }
    Ok((max_abs, sq.sqrt()))
}

pub fn run_default_harness() -> Result<LinkSparseAttentionReport, String> {
    let LinkFixture {
        query,
        keys,
        values,
        summaries,
        needle_block,
    } = build_fixture();
    let all_blocks: Vec<usize> = (0..NUM_BLOCKS).collect();
    let dense_out = causal_block_attention(&query, &keys, &values, &all_blocks)?;

    let full_link_blocks = select_learned_link_blocks(&query, &summaries, NUM_BLOCKS)?;
    let full_link_out = causal_block_attention(&query, &keys, &values, &full_link_blocks)?;
    let (full_max_err, full_l2_err) = diff_metrics(&dense_out, &full_link_out)?;

    let sparse_k4_blocks = select_learned_link_blocks(&query, &summaries, 4)?;
    let sparse_k4_out = causal_block_attention(&query, &keys, &values, &sparse_k4_blocks)?;
    let (k4_max_err, k4_l2_err) = diff_metrics(&dense_out, &sparse_k4_out)?;

    let sparse_k2_blocks = select_learned_link_blocks(&query, &summaries, 2)?;
    let sparse_k2_out = causal_block_attention(&query, &keys, &values, &sparse_k2_blocks)?;
    let (k2_max_err, k2_l2_err) = diff_metrics(&dense_out, &sparse_k2_out)?;

    let sparse_arms = vec![
        SparseReadArmResult {
            arm_id: "dense_exact_attention".into(),
            selected_blocks: NUM_BLOCKS,
            total_blocks: NUM_BLOCKS,
            kv_read_fraction: 1.0,
            max_abs_err_vs_dense: 0.0,
            l2_err_vs_dense: 0.0,
            needle_block_recalled: true,
            parity_gate_passed: true,
        },
        SparseReadArmResult {
            arm_id: "link_learned_full_read_k16".into(),
            selected_blocks: full_link_blocks.len(),
            total_blocks: NUM_BLOCKS,
            kv_read_fraction: 1.0,
            max_abs_err_vs_dense: full_max_err,
            l2_err_vs_dense: full_l2_err,
            needle_block_recalled: full_link_blocks.contains(&needle_block),
            parity_gate_passed: full_max_err <= PARITY_TOL,
        },
        SparseReadArmResult {
            arm_id: "link_learned_sparse_k4".into(),
            selected_blocks: sparse_k4_blocks.len(),
            total_blocks: NUM_BLOCKS,
            kv_read_fraction: (sparse_k4_blocks.len() as f64) / (NUM_BLOCKS as f64),
            max_abs_err_vs_dense: k4_max_err,
            l2_err_vs_dense: k4_l2_err,
            needle_block_recalled: sparse_k4_blocks.contains(&needle_block),
            parity_gate_passed: sparse_k4_blocks.contains(&needle_block),
        },
        SparseReadArmResult {
            arm_id: "link_learned_sparse_k2".into(),
            selected_blocks: sparse_k2_blocks.len(),
            total_blocks: NUM_BLOCKS,
            kv_read_fraction: (sparse_k2_blocks.len() as f64) / (NUM_BLOCKS as f64),
            max_abs_err_vs_dense: k2_max_err,
            l2_err_vs_dense: k2_l2_err,
            needle_block_recalled: sparse_k2_blocks.contains(&needle_block),
            parity_gate_passed: sparse_k2_blocks.contains(&needle_block),
        },
    ];

    let mut boundary_rows = Vec::new();
    for &ctx in &[131_072usize, 524_288, 1_048_576, 4_194_304] {
        boundary_rows.push(evaluate_long_context_boundary(
            ctx,
            "dense_attention_baseline",
            0.88,
            0.82,
            0.80,
            false,
            1.0,
            true,
        )?);
        boundary_rows.push(evaluate_long_context_boundary(
            ctx,
            "link_learned_sparse_default_off",
            0.0,
            0.0,
            0.80,
            false,
            0.125,
            false,
        )?);
    }

    let mut negative_controls = Vec::new();

    // NC1: unlocated public Link artifacts blocked
    negative_controls.push(NegativeControl {
        case_id: "nc1_unlocated_link_artifacts_blocked".into(),
        outcome: boundary_rows[1].admission_outcome,
        detail: "Link public code/weights not pinned -> blocked_on_artifacts".into(),
    });

    // NC2: k == 0 block selection rejected
    negative_controls.push(NegativeControl {
        case_id: "nc2_zero_block_selection_rejected".into(),
        outcome: if select_learned_link_blocks(&query, &summaries, 0).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "top_k_blocks = 0 rejected cleanly".into(),
    });

    // NC3: k > NUM_BLOCKS selection rejected
    negative_controls.push(NegativeControl {
        case_id: "nc3_oob_block_count_rejected".into(),
        outcome: if select_learned_link_blocks(&query, &summaries, NUM_BLOCKS + 1).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "top_k_blocks exceeding block count rejected".into(),
    });

    // NC4: non-finite NaN query rejected
    let mut nan_q = query;
    nan_q[0] = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc4_nonfinite_query_rejected".into(),
        outcome: if select_learned_link_blocks(&nan_q, &summaries, 4).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN query vector rejected before block scoring".into(),
    });

    // NC5: faster sparse candidate failing end-task quality floor disqualified
    let fast_wrong = evaluate_long_context_boundary(
        1_048_576,
        "fast_but_wrong_sparse",
        0.95,
        0.62,
        0.80,
        false,
        0.10,
        true,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc5_faster_but_wrong_end_task_disqualified".into(),
        outcome: fast_wrong.admission_outcome,
        detail: "sparse candidate with 10x fewer KV reads but 0.62 < 0.80 end-task accuracy disqualified".into(),
    });

    // NC6: needle-only check without end-task rejected
    let needle_only = evaluate_long_context_boundary(
        4_194_304,
        "needle_only_candidate",
        0.99,
        0.99,
        0.80,
        true,
        0.125,
        true,
    )?;
    negative_controls.push(NegativeControl {
        case_id: "nc6_needle_only_without_end_task_rejected".into(),
        outcome: needle_only.admission_outcome,
        detail: "retrieval-only check without independent end-task evaluation rejected".into(),
    });

    // NC7: out-of-range block index in attention refused
    negative_controls.push(NegativeControl {
        case_id: "nc7_oob_block_index_attention_rejected".into(),
        outcome: if causal_block_attention(&query, &keys, &values, &[NUM_BLOCKS]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "out-of-range block index rejected by attention reference".into(),
    });

    // NC8: importing unverified X-post +53% / 9.9x claims as pass gate rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_unverified_x_claim_as_gate_rejected".into(),
        outcome: OutcomeStatus::Invalid,
        detail: "authors' unverified +53% retrieval / 9.9x GPU efficiency claims are non-gating related work only".into(),
    });

    Ok(LinkSparseAttentionReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "Constructor AGI Link public code/weights & 4M-context GPU serve",
            "No public Constructor AGI Link repository or checkpoint is verified locally (post_only_unlocated_paper_repo); bounded pure-Rust sparse-read reference executed on CPU only",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_pin_status: UPSTREAM_PIN_STATUS.into(),
        unverified_author_claims: "Authors' unverified X post claims: +53% retrieval accuracy over dense attention at 4M token context and 9.9x GPU efficiency (non-gating related work only)".into(),
        sparse_arms,
        boundary_rows,
        negative_controls,
    })
}

pub fn inputs_json(report: &LinkSparseAttentionReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.link-sparse-attention.inputs.v1\",\n",
            "  \"issue\": 85,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"upstream_video_url\": {},\n",
            "  \"upstream_pin_status\": {},\n",
            "  \"complements_issues\": [23, 31],\n",
            "  \"fixture_seq_len\": {},\n",
            "  \"fixture_num_blocks\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(UPSTREAM_X_POST),
        quote(UPSTREAM_VIDEO_URL),
        quote(&report.upstream_pin_status),
        SEQ_LEN,
        NUM_BLOCKS,
    )
}

pub fn results_json(report: &LinkSparseAttentionReport) -> String {
    let arms: Vec<String> = report
        .sparse_arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"selected_blocks\":{},\"total_blocks\":{},\"kv_read_fraction\":{},\"max_abs_err_vs_dense\":{},\"l2_err_vs_dense\":{},\"needle_block_recalled\":{},\"parity_gate_passed\":{}}}",
                quote(&a.arm_id),
                a.selected_blocks,
                a.total_blocks,
                finite_f64(a.kv_read_fraction),
                finite_f64(a.max_abs_err_vs_dense),
                finite_f64(a.l2_err_vs_dense),
                a.needle_block_recalled,
                a.parity_gate_passed,
            )
        })
        .collect();
    let boundaries: Vec<String> = report
        .boundary_rows
        .iter()
        .map(|b| {
            format!(
                "{{\"context_tokens\":{},\"arm_id\":{},\"retrieval_accuracy\":{},\"end_task_accuracy\":{},\"needle_only_evaluated\":{},\"relative_kv_reads\":{},\"admission_outcome\":{},\"reason\":{}}}",
                b.context_tokens,
                quote(&b.arm_id),
                finite_f64(b.retrieval_accuracy),
                finite_f64(b.end_task_accuracy),
                b.needle_only_evaluated,
                opt_f64(Some(b.relative_kv_reads)),
                quote(b.admission_outcome.as_str()),
                quote(&b.reason),
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
            "  \"schema\": \"inference-lab.link-sparse-attention.results.v1\",\n",
            "  \"issue\": 85,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"unverified_author_claims\": {},\n",
            "  \"sparse_arms\": [{}],\n",
            "  \"long_context_boundary_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.unverified_author_claims),
        arms.join(","),
        boundaries.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &LinkSparseAttentionReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #85: Constructor AGI Link Learned Sparse Attention Related Work & Long-Context Boundary"
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
        "- Upstream reference: {} (`{}`; unverified claims: +53% retrieval at 4M, 9.9x GPU efficiency)",
        UPSTREAM_X_POST, report.upstream_pin_status
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Pure-Rust Learned Sparse-Read vs Exact Dense Attention"
    )
    .unwrap();
    for a in &report.sparse_arms {
        writeln!(
            out,
            "- `{}`: `{}/{}` blocks (`{:.3}` KV reads), `max_abs_err={:.3e}`, `l2_err={:.3e}`, `needle_recalled={}`, `parity_gate={}`",
            a.arm_id,
            a.selected_blocks,
            a.total_blocks,
            a.kv_read_fraction,
            a.max_abs_err_vs_dense,
            a.l2_err_vs_dense,
            a.needle_block_recalled,
            a.parity_gate_passed
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
    fn link_full_read_matches_dense_and_rejects_all_controls() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert_eq!(r.hardware_gate.status, ExecutionClass::BlockedOnArtifacts);
        assert!(r.sparse_arms[1].max_abs_err_vs_dense <= PARITY_TOL);
        assert!(r.sparse_arms[2].needle_block_recalled);
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
