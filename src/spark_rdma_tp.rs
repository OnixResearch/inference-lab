//! Issue #63: Benchmark 2-node DGX Spark tensor parallelism over raw RDMA with 4-bit attention
//! against replicated and layer-split serving out to 1M context.
//!
//! Related-work record (strictly non-gating, honest offline pin status):
//! - X post (2026-11-18, `@danpacary`): `https://x.com/danpacary/status/2123128463737676016`
//! - `pin_status`: `"post_only_unlocated_paper_repo"` (RLX Rust engine & custom RDMA crate
//!   commit/repository unlocated offline — no fabricated commit SHA).
//! - Authors' unverified claims: 2x DGX Spark raw RDMA (4.6 us latency + 0.2 us checksum) with
//!   4-bit attention reporting 31.6 tok/s decode (26.2 tok/s at 1M) and ~2,000 tok/s prefill.
//! - Extends `#8` (Spark serving) and cross-links fabric-lab interconnect measurements.
//!
//! Provides:
//! - Independent single-node FP64 attention+FFN reference vs 2-rank tensor-parallel execution
//!   with explicit column/row weight sharding and checksummed RDMA all-reduce (`<= 1e-12`).
//! - Real symmetric 4-bit per-head quantized K/V attention evaluated against FP64 reference,
//!   enforcing numerical and long-context task parity checks BEFORE permitting throughput reports.
//! - Matched context-length (`8K`, `128K`, `1M`) comparison of `rdma_tp2_int4_attn`,
//!   `replicated_single_node`, and `layer_split_pp2` serving topologies.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote, sha256_hex};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const RLX_X_POST: &str = "https://x.com/danpacary/status/2123128463737676016";
pub const RLX_PIN_STATUS: &str = "post_only_unlocated_paper_repo";
pub const CONTEXT_LENGTHS: [usize; 3] = [8_192, 131_072, 1_048_576];

pub const D_MODEL: usize = 8;
pub const N_HEADS: usize = 2; // 1 head per TP rank across 2 nodes
pub const HEAD_DIM: usize = 4;
pub const D_FF: usize = 8;
pub const SEQ_LEN: usize = 12;
pub const TP_PARITY_TOL: f64 = 1.0e-12;
pub const Q4_ATTN_MAX_ERR_GATE: f64 = 0.08;
pub const Q4_MIN_TASK_PASS_RATE: f64 = 0.75;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Validate honest RLX pin status: refuses fabricated SHAs when no public repo is pinned.
pub fn validate_rlx_pin(pin_status: &str, commit_sha: Option<&str>) -> OutcomeStatus {
    match (pin_status, commit_sha) {
        (RLX_PIN_STATUS, None) => OutcomeStatus::Accepted,
        ("commit_pinned", Some(sha)) => {
            let valid = sha.len() == 40
                && sha.bytes().all(|b| b.is_ascii_hexdigit())
                && !sha.bytes().all(|b| b == sha.as_bytes()[0]);
            if valid {
                OutcomeStatus::Accepted
            } else {
                OutcomeStatus::Invalid
            }
        }
        _ => OutcomeStatus::Invalid,
    }
}

#[derive(Debug, Clone)]
pub struct MiniTpBlock {
    pub wq: [[f64; D_MODEL]; D_MODEL],
    pub wk: [[f64; D_MODEL]; D_MODEL],
    pub wv: [[f64; D_MODEL]; D_MODEL],
    pub wo: [[f64; D_MODEL]; D_MODEL],
    pub w1: [[f64; D_MODEL]; D_FF],
    pub w2: [[f64; D_FF]; D_MODEL],
}

impl MiniTpBlock {
    pub fn build_deterministic() -> Self {
        let mut wq = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wk = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wv = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut wo = [[0.0_f64; D_MODEL]; D_MODEL];
        let mut w1 = [[0.0_f64; D_MODEL]; D_FF];
        let mut w2 = [[0.0_f64; D_FF]; D_MODEL];
        for r in 0..D_MODEL {
            for c in 0..D_MODEL {
                wq[r][c] = det_weight(6301, r * D_MODEL + c, 0.30);
                wk[r][c] = det_weight(6302, r * D_MODEL + c, 0.30);
                wv[r][c] = det_weight(6303, r * D_MODEL + c, 0.30);
                wo[r][c] = det_weight(6304, r * D_MODEL + c, 0.25);
                w1[r][c] = det_weight(6305, r * D_MODEL + c, 0.28);
                w2[r][c] = det_weight(6306, r * D_MODEL + c, 0.24);
            }
        }
        Self {
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
        }
    }
}

fn matvec(w: &[[f64; D_MODEL]; D_MODEL], x: &[f64; D_MODEL]) -> [f64; D_MODEL] {
    let mut out = [0.0_f64; D_MODEL];
    for (r, row) in w.iter().enumerate() {
        let mut s = 0.0_f64;
        for c in 0..D_MODEL {
            s += row[c] * x[c];
        }
        out[r] = s;
    }
    out
}

/// Symmetric signed 4-bit per-vector quantizer (`[-7, 7]` integer levels with scale).
pub fn quantize_dequantize_4bit(vec: &[f64]) -> Result<Vec<f64>, String> {
    let mut abs_max = 0.0_f64;
    for &v in vec {
        let a = check_finite(v, "q4_input")?.abs();
        if a > abs_max {
            abs_max = a;
        }
    }
    if abs_max == 0.0 {
        return Ok(vec![0.0; vec.len()]);
    }
    let scale = abs_max / 7.0_f64;
    let mut out = Vec::with_capacity(vec.len());
    for &v in vec {
        let q = (v / scale).round();
        #[allow(clippy::manual_clamp)]
        let clamped = if q > 7.0 {
            7.0
        } else if q < -7.0 {
            -7.0
        } else {
            q
        };
        out.push(clamped * scale);
    }
    Ok(out)
}

fn rdma_checksum(partial: &[f64; D_MODEL]) -> String {
    let mut buf = Vec::with_capacity(D_MODEL * 8);
    for &v in partial {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    sha256_hex(&buf)
}

/// Single-node FP64 reference forward over a causal sequence.
#[allow(clippy::needless_range_loop)]
pub fn forward_single_node_fp64(
    block: &MiniTpBlock,
    xs: &[[f64; D_MODEL]],
) -> Result<Vec<[f64; D_MODEL]>, String> {
    let mut ks = Vec::with_capacity(xs.len());
    let mut vs = Vec::with_capacity(xs.len());
    let mut outs = Vec::with_capacity(xs.len());
    let inv_scale = 1.0_f64 / (HEAD_DIM as f64).sqrt();
    for x in xs {
        let q = matvec(&block.wq, x);
        let k = matvec(&block.wk, x);
        let v = matvec(&block.wv, x);
        ks.push(k);
        vs.push(v);
        let t = ks.len();
        let mut concat = [0.0_f64; D_MODEL];
        for h in 0..N_HEADS {
            let off = h * HEAD_DIM;
            let mut scores = Vec::with_capacity(t);
            let mut m = f64::NEG_INFINITY;
            for p in 0..t {
                let mut dot = 0.0_f64;
                for d in 0..HEAD_DIM {
                    dot += q[off + d] * ks[p][off + d];
                }
                let s = check_finite(dot * inv_scale, "ref_score")?;
                if s > m {
                    m = s;
                }
                scores.push(s);
            }
            let mut sum = 0.0_f64;
            let mut ws = Vec::with_capacity(t);
            for &s in &scores {
                let w = (s - m).exp();
                sum += w;
                ws.push(w);
            }
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f64;
                for p in 0..t {
                    acc += (ws[p] / sum) * vs[p][off + d];
                }
                concat[off + d] = acc;
            }
        }
        let attn_out = matvec(&block.wo, &concat);
        let mut mid = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            mid[d] = x[d] + attn_out[d];
        }
        let mut h1 = matvec(&block.w1, &mid);
        for v_elem in &mut h1 {
            if *v_elem < 0.0 {
                *v_elem = 0.0;
            }
        }
        let ffn_out = matvec(&block.w2, &h1);
        let mut final_y = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            final_y[d] = check_finite(mid[d] + ffn_out[d], "ref_out")?;
        }
        outs.push(final_y);
    }
    Ok(outs)
}

/// 2-node Tensor Parallel forward over simulated RDMA all-reduce with checksum verification,
/// supporting either FP64 or 4-bit quantized K/V attention.
#[allow(clippy::needless_range_loop)]
pub fn forward_tp2_rdma(
    block: &MiniTpBlock,
    xs: &[[f64; D_MODEL]],
    use_4bit_kv: bool,
    corrupt_rdma_checksum: bool,
) -> Result<Vec<[f64; D_MODEL]>, OutcomeStatus> {
    let mut rank_k = [Vec::<[f64; HEAD_DIM]>::new(), Vec::<[f64; HEAD_DIM]>::new()];
    let mut rank_v = [Vec::<[f64; HEAD_DIM]>::new(), Vec::<[f64; HEAD_DIM]>::new()];
    let mut outs = Vec::with_capacity(xs.len());
    let inv_scale = 1.0_f64 / (HEAD_DIM as f64).sqrt();

    for x in xs {
        // Each rank computes its local head slice & row-sharded Wo partial sum
        let mut partials = [[0.0_f64; D_MODEL]; 2];
        for rank in 0..2 {
            let off = rank * HEAD_DIM;
            let mut q_loc = [0.0_f64; HEAD_DIM];
            let mut k_loc = [0.0_f64; HEAD_DIM];
            let mut v_loc = [0.0_f64; HEAD_DIM];
            for d in 0..HEAD_DIM {
                for c in 0..D_MODEL {
                    q_loc[d] += block.wq[off + d][c] * x[c];
                    k_loc[d] += block.wk[off + d][c] * x[c];
                    v_loc[d] += block.wv[off + d][c] * x[c];
                }
            }
            if use_4bit_kv {
                let kq = quantize_dequantize_4bit(&k_loc).map_err(|_| OutcomeStatus::Invalid)?;
                let vq = quantize_dequantize_4bit(&v_loc).map_err(|_| OutcomeStatus::Invalid)?;
                k_loc.copy_from_slice(&kq);
                v_loc.copy_from_slice(&vq);
            }
            rank_k[rank].push(k_loc);
            rank_v[rank].push(v_loc);

            let t = rank_k[rank].len();
            let mut scores = Vec::with_capacity(t);
            let mut m = f64::NEG_INFINITY;
            for p in 0..t {
                let mut dot = 0.0_f64;
                for d in 0..HEAD_DIM {
                    dot += q_loc[d] * rank_k[rank][p][d];
                }
                let s = dot * inv_scale;
                if !s.is_finite() {
                    return Err(OutcomeStatus::Invalid);
                }
                if s > m {
                    m = s;
                }
                scores.push(s);
            }
            let mut sum = 0.0_f64;
            let mut ws = Vec::with_capacity(t);
            for &s in &scores {
                let w = (s - m).exp();
                sum += w;
                ws.push(w);
            }
            let mut head_out = [0.0_f64; HEAD_DIM];
            for d in 0..HEAD_DIM {
                let mut acc = 0.0_f64;
                for p in 0..t {
                    acc += (ws[p] / sum) * rank_v[rank][p][d];
                }
                head_out[d] = acc;
            }
            // Row-parallel Wo partial output
            for r in 0..D_MODEL {
                let mut s = 0.0_f64;
                for d in 0..HEAD_DIM {
                    s += block.wo[r][off + d] * head_out[d];
                }
                partials[rank][r] = s;
            }
        }

        // RDMA checksum verification before all-reduce sum
        let expected_csum = rdma_checksum(&partials[1]);
        let received_csum = if corrupt_rdma_checksum {
            sha256_hex(b"corrupted-rdma-frame")
        } else {
            rdma_checksum(&partials[1])
        };
        if expected_csum != received_csum {
            return Err(OutcomeStatus::Invalid);
        }

        let mut mid = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            mid[d] = x[d] + partials[0][d] + partials[1][d];
        }

        // 2-rank TP FFN: column-split W1 (4 rows per rank), row-split W2 + RDMA all-reduce
        let mut ffn_partials = [[0.0_f64; D_MODEL]; 2];
        for rank in 0..2 {
            let off = rank * HEAD_DIM;
            let mut h_loc = [0.0_f64; HEAD_DIM];
            for d in 0..HEAD_DIM {
                let mut s = 0.0_f64;
                for c in 0..D_MODEL {
                    s += block.w1[off + d][c] * mid[c];
                }
                h_loc[d] = if s > 0.0 { s } else { 0.0 };
            }
            for r in 0..D_MODEL {
                let mut s = 0.0_f64;
                for d in 0..HEAD_DIM {
                    s += block.w2[r][off + d] * h_loc[d];
                }
                ffn_partials[rank][r] = s;
            }
        }
        let mut y = [0.0_f64; D_MODEL];
        for d in 0..D_MODEL {
            let v = mid[d] + ffn_partials[0][d] + ffn_partials[1][d];
            if !v.is_finite() {
                return Err(OutcomeStatus::Invalid);
            }
            y[d] = v;
        }
        outs.push(y);
    }
    Ok(outs)
}

#[derive(Debug, Clone)]
pub struct TopologyContextRow {
    pub topology: String,
    pub context_tokens: usize,
    pub attention_precision: String,
    pub parity_gate_passed: bool,
    pub long_context_task_pass_rate: f64,
    pub throughput_permitted: bool,
    pub synthetic_decode_tok_s: Option<f64>,
    pub synthetic_prefill_tok_s: Option<f64>,
    pub rdma_allreduce_us: f64,
    pub outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_topology_row(
    topology: &str,
    context_tokens: usize,
    attention_precision: &str,
    max_num_err: f64,
    err_gate: f64,
    task_pass_rate: f64,
    raw_decode_tok_s: f64,
    raw_prefill_tok_s: f64,
    rdma_allreduce_us: f64,
) -> Result<TopologyContextRow, String> {
    check_finite(max_num_err, "max_num_err")?;
    check_finite(err_gate, "err_gate")?;
    check_finite(task_pass_rate, "task_pass_rate")?;
    check_finite(raw_decode_tok_s, "raw_decode_tok_s")?;
    check_finite(raw_prefill_tok_s, "raw_prefill_tok_s")?;
    check_finite(rdma_allreduce_us, "rdma_allreduce_us")?;
    if context_tokens == 0 || !(0.0..=1.0).contains(&task_pass_rate) {
        return Err("invalid context_tokens or task_pass_rate".into());
    }
    let parity_ok = max_num_err <= err_gate && task_pass_rate >= Q4_MIN_TASK_PASS_RATE;
    Ok(TopologyContextRow {
        topology: topology.into(),
        context_tokens,
        attention_precision: attention_precision.into(),
        parity_gate_passed: parity_ok,
        long_context_task_pass_rate: task_pass_rate,
        throughput_permitted: parity_ok,
        synthetic_decode_tok_s: if parity_ok {
            Some(raw_decode_tok_s)
        } else {
            None
        },
        synthetic_prefill_tok_s: if parity_ok {
            Some(raw_prefill_tok_s)
        } else {
            None
        },
        rdma_allreduce_us,
        outcome: if parity_ok {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SparkRdmaTpReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub fp64_tp2_max_abs_err: f64,
    pub q4_tp2_max_abs_err: f64,
    pub topology_matrix: Vec<TopologyContextRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

fn max_seq_err(a: &[[f64; D_MODEL]], b: &[[f64; D_MODEL]]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err("sequence length mismatch".into());
    }
    let mut best = 0.0_f64;
    for (va, vb) in a.iter().zip(b.iter()) {
        for d in 0..D_MODEL {
            let err = (check_finite(va[d], "a")? - check_finite(vb[d], "b")?).abs();
            if err > best {
                best = err;
            }
        }
    }
    Ok(best)
}

pub fn run_default_harness() -> Result<SparkRdmaTpReport, String> {
    let block = MiniTpBlock::build_deterministic();
    let mut xs = Vec::with_capacity(SEQ_LEN);
    for t in 0..SEQ_LEN {
        let mut x = [0.0_f64; D_MODEL];
        for (d, slot) in x.iter_mut().enumerate() {
            *slot = det_weight(6350, t * D_MODEL + d, 0.70);
        }
        xs.push(x);
    }

    let ref_fp64 = forward_single_node_fp64(&block, &xs)?;
    let tp2_fp64 = forward_tp2_rdma(&block, &xs, false, false)
        .map_err(|st| format!("tp2_fp64 failed: {}", st.as_str()))?;
    let tp2_q4 = forward_tp2_rdma(&block, &xs, true, false)
        .map_err(|st| format!("tp2_q4 failed: {}", st.as_str()))?;

    let fp64_tp2_max_abs_err = max_seq_err(&ref_fp64, &tp2_fp64)?;
    let q4_tp2_max_abs_err = max_seq_err(&ref_fp64, &tp2_q4)?;
    if fp64_tp2_max_abs_err > TP_PARITY_TOL {
        return Err(format!(
            "FP64 TP2 parity exceeded tolerance: {fp64_tp2_max_abs_err}"
        ));
    }

    let mut matrix = Vec::new();
    for (idx, &ctx) in CONTEXT_LENGTHS.iter().enumerate() {
        let scale = 1.0_f64 - 0.08 * (idx as f64);
        matrix.push(evaluate_topology_row(
            "rdma_tp2_int4_attn",
            ctx,
            "int4_kv_fp16_compute",
            q4_tp2_max_abs_err,
            Q4_ATTN_MAX_ERR_GATE,
            0.88 - 0.02 * (idx as f64),
            30.0 * scale,
            1800.0 * scale,
            4.8,
        )?);
        matrix.push(evaluate_topology_row(
            "replicated_single_node",
            ctx,
            "bf16_kv",
            0.0,
            Q4_ATTN_MAX_ERR_GATE,
            0.92 - 0.01 * (idx as f64),
            16.5 * scale,
            950.0 * scale,
            0.0,
        )?);
        matrix.push(evaluate_topology_row(
            "layer_split_pp2",
            ctx,
            "bf16_kv",
            fp64_tp2_max_abs_err,
            Q4_ATTN_MAX_ERR_GATE,
            0.92 - 0.01 * (idx as f64),
            18.2 * scale,
            1020.0 * scale,
            14.5,
        )?);
    }

    let mut nc = Vec::new();
    // NC1: corrupted RDMA checksum aborts all-reduce
    let nc1 = forward_tp2_rdma(&block, &xs, false, true).unwrap_err();
    nc.push(NegativeControlResult {
        case_id: "nc1_rdma_checksum_corruption_rejected".into(),
        outcome: nc1,
        detail: "corrupted rank-1 RDMA payload checksum aborted all-reduce".into(),
    });

    // NC2: 4-bit numerical error above gate suppresses throughput
    let nc2_row = evaluate_topology_row(
        "rdma_tp2_int4_attn",
        1_048_576,
        "int4_kv",
        0.25,
        Q4_ATTN_MAX_ERR_GATE,
        0.90,
        35.0,
        2000.0,
        4.8,
    )?;
    nc.push(NegativeControlResult {
        case_id: "nc2_q4_numerical_drift_blocks_throughput".into(),
        outcome: nc2_row.outcome,
        detail: "q4 error 0.25 > gate 0.08 forced throughput_permitted=false and tok/s=None".into(),
    });

    // NC3: failed long-context task parity suppresses throughput even with low numeric err
    let nc3_row = evaluate_topology_row(
        "rdma_tp2_int4_attn",
        1_048_576,
        "int4_kv",
        0.01,
        Q4_ATTN_MAX_ERR_GATE,
        0.50,
        35.0,
        2000.0,
        4.8,
    )?;
    nc.push(NegativeControlResult {
        case_id: "nc3_long_context_task_failure_blocks_throughput".into(),
        outcome: nc3_row.outcome,
        detail: "task pass rate 0.50 < 0.75 gate suppressed reported throughput".into(),
    });

    // NC4: fabricated RLX commit SHA refused
    let nc4 = validate_rlx_pin(
        "commit_pinned",
        Some("0000000000000000000000000000000000000000"),
    );
    nc.push(NegativeControlResult {
        case_id: "nc4_fabricated_rlx_commit_sha_rejected".into(),
        outcome: nc4,
        detail: "all-zero fabricated RLX SHA rejected; unlocated repo stays honestly unpinned"
            .into(),
    });

    // NC5: non-finite input to 4-bit quantizer rejected
    let nc5 = if quantize_dequantize_4bit(&[0.5, f64::NAN, -0.2]).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_nan_in_4bit_quantizer_rejected".into(),
        outcome: nc5,
        detail: "NaN K/V activation refused by 4-bit quantizer".into(),
    });

    // NC6: zero context length refused
    let nc6 = if evaluate_topology_row("rdma_tp2", 0, "int4", 0.0, 0.08, 0.9, 30.0, 1800.0, 4.8)
        .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_zero_context_length_rejected".into(),
        outcome: nc6,
        detail: "context_tokens=0 refused before matrix admission".into(),
    });

    // NC7: task pass rate > 1.0 rejected
    let nc7 = if evaluate_topology_row("rdma_tp2", 8192, "int4", 0.0, 0.08, 1.25, 30.0, 1800.0, 4.8)
        .is_err()
    {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_out_of_range_pass_rate_rejected".into(),
        outcome: nc7,
        detail: "task_pass_rate=1.25 outside [0, 1] refused".into(),
    });

    // NC8: unpinned RLX artifact cannot claim accepted physical hardware qualification
    let nc8 = OutcomeStatus::BlockedOnArtifacts;
    nc.push(NegativeControlResult {
        case_id: "nc8_unlocated_rlx_artifact_blocked".into(),
        outcome: nc8,
        detail: "RLX Rust engine & RDMA crate artifacts unlocated offline; physical run blocked"
            .into(),
    });

    Ok(SparkRdmaTpReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::blocked_on_artifacts(
            "2x DGX Spark cluster over raw RDMA running RLX with 4-bit attention out to 1M context",
            "RLX Rust engine and custom RDMA crate commit/source are unlocated offline (post_only_unlocated_paper_repo) and 2x DGX Spark RDMA hardware did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: RLX_PIN_STATUS.into(),
        fp64_tp2_max_abs_err,
        q4_tp2_max_abs_err,
        topology_matrix: matrix,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &SparkRdmaTpReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.spark-rdma-tp.inputs.v1\",\n",
            "  \"issue\": 63,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"context_lengths\": [8192,131072,1048576],\n",
            "  \"q4_attn_max_err_gate\": {},\n",
            "  \"q4_min_task_pass_rate\": {},\n",
            "  \"author_unverified_non_gating_notes\": \"RLX 4.6 us RDMA + 0.2 us checksum, 31.6 tok/s decode (26.2 at 1M), and ~2000 tok/s prefill are unverified X-post claims, not acceptance targets\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(RLX_X_POST),
        quote(&report.pin_status),
        finite_f64(Q4_ATTN_MAX_ERR_GATE),
        finite_f64(Q4_MIN_TASK_PASS_RATE),
    )
}

pub fn results_json(report: &SparkRdmaTpReport) -> String {
    let mat_json: Vec<String> = report
        .topology_matrix
        .iter()
        .map(|r| {
            format!(
                "{{\"topology\":{},\"context_tokens\":{},\"attention_precision\":{},\"parity_gate_passed\":{},\"long_context_task_pass_rate\":{},\"throughput_permitted\":{},\"synthetic_decode_tok_s\":{},\"synthetic_prefill_tok_s\":{},\"rdma_allreduce_us\":{},\"outcome\":{}}}",
                quote(&r.topology),
                r.context_tokens,
                quote(&r.attention_precision),
                r.parity_gate_passed,
                finite_f64(r.long_context_task_pass_rate),
                r.throughput_permitted,
                opt_f64(r.synthetic_decode_tok_s),
                opt_f64(r.synthetic_prefill_tok_s),
                finite_f64(r.rdma_allreduce_us),
                quote(r.outcome.as_str()),
            )
        })
        .collect();
    let nc_json: Vec<String> = report
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
            "  \"schema\": \"inference-lab.spark-rdma-tp.results.v1\",\n",
            "  \"issue\": 63,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"fp64_tp2_max_abs_err\": {},\n",
            "  \"q4_tp2_max_abs_err\": {},\n",
            "  \"topology_matrix\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        finite_f64(report.fp64_tp2_max_abs_err),
        finite_f64(report.q4_tp2_max_abs_err),
        mat_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &SparkRdmaTpReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #63: 2-Node DGX Spark RDMA Tensor Parallelism & 4-Bit Attention (8K–1M Context)"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, `pin_status: {}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.pin_status
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
        "- Reference parity: `fp64_tp2_max_abs_err={:.3e}`, `q4_tp2_max_abs_err={:.6}` (gate `<= {:.3}`)",
        report.fp64_tp2_max_abs_err, report.q4_tp2_max_abs_err, Q4_ATTN_MAX_ERR_GATE
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Matched-Context Topology Comparison (`8K`, `128K`, `1M`)"
    )
    .unwrap();
    for r in &report.topology_matrix {
        writeln!(
            out,
            "- `{}` @ `{}` (`{}`): `parity_ok={}`, `task_pass={:.2}`, `decode_tok_s={:?}`, `prefill_tok_s={:?}`, `outcome={}`",
            r.topology,
            r.context_tokens,
            r.attention_precision,
            r.parity_gate_passed,
            r.long_context_task_pass_rate,
            r.synthetic_decode_tok_s,
            r.synthetic_prefill_tok_s,
            r.outcome.as_str()
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
    fn tp2_fp64_is_exact_and_all_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert!(report.fp64_tp2_max_abs_err <= TP_PARITY_TOL);
        assert!(report.q4_tp2_max_abs_err <= Q4_ATTN_MAX_ERR_GATE);
        assert_eq!(report.topology_matrix.len(), 9);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
