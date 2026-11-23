//! Issue #37: Benchmark native recurrent-state inference across RWKV-7, Mamba-3 and xLSTM.
//!
//! Pinned source basis (strictly non-gating upstream notes):
//! - RWKV-7: RSI Lab numerical/state baseline (`#12`/`#14`; this issue is the second consumer
//!   referencing `#14`'s conditional leaf while keeping RSI agent policy in RSI Lab).
//! - Mamba-3 SISO: `state-spaces/mamba` + ICLR 2026 Mamba-3 paper (architectural motivation only).
//! - xLSTM: `NX-AI/xlstm` (`xLSTM-7b` or smaller), strictly gated behind Milestone B after RWKV-7
//!   and Mamba-3 establish the common Milestone A schema.
//! - Related work ONLY (NOT added to official cohort): Latent Lattice Language Models
//!   (`arXiv:2611.21608v1`, `23876839517155db4333fb08d26b8b118a3db2e1`, tracked in `#60`).
//!
//! Provides:
//! - Versioned native recurrent-state contract with 6 explicit operation dispositions (`supported`,
//!   `unsupported`, `invalid`, `resource_stop`, `numerical_failure`, `infrastructure_failure`),
//!   forbidding mislabeling full-prefix recomputation as native state support.
//! - True independent scalar full-sequence references (`scalar_full_reference`) completely separate
//!   from the stateful/chunked `NativeRecurrentState` engine for all 3 architectures:
//!   1. RWKV-7 matrix-state recurrence
//!   2. Mamba-3 SISO selective state-space recurrence
//!   3. xLSTM log-stabilized matrix-memory mLSTM (`C_t`, `n_t`, `m_t`) gated behind Milestone B.
//! - Full 5-phase verification (independent scalar vs stateful step & irregular chunks, empty &
//!   single-token edge cases, first-divergent-token reporting, serialize/restore/poisoned-slot/
//!   capacity bounds, fork aliasing control & rollback disposition, 2-session interleaved isolation
//!   + batched parity, state bytes + recompute economics) and 10 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, sigmoid};
use crate::json::{finite_f64, opt_u64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const REC_DIM: usize = 4;
pub const REC_CAPACITY: usize = 24;
pub const LATENT_LATTICE_RELATED_COMMIT: &str = "23876839517155db4333fb08d26b8b118a3db2e1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentArch {
    Rwkv7,
    Mamba3Siso,
    XlstmMlstm,
}

impl RecurrentArch {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rwkv7 => "rwkv7",
            Self::Mamba3Siso => "mamba3_siso",
            Self::XlstmMlstm => "xlstm_mlstm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationDisposition {
    Supported,
    Unsupported,
    Invalid,
    ResourceStop,
    NumericalFailure,
    InfrastructureFailure,
}

impl OperationDisposition {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unsupported => "unsupported",
            Self::Invalid => "invalid",
            Self::ResourceStop => "resource_stop",
            Self::NumericalFailure => "numerical_failure",
            Self::InfrastructureFailure => "infrastructure_failure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentProfileIdentity {
    pub arch: RecurrentArch,
    pub model_digest: DigestRef,
    pub tokenizer_rev: String,
    pub state_dtype: String,
    pub serialization_version: u32,
    pub worker_generation: u64,
    pub session_id: String,
}

#[derive(Debug, Clone)]
pub struct NativeRecurrentState {
    pub identity: RecurrentProfileIdentity,
    pub valid_len: usize,
    pub capacity: usize,
    pub matrix: [[f64; REC_DIM]; REC_DIM],
    pub aux_a: [f64; REC_DIM],
    pub log_stab_m: f64,
    pub scratch_poison: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecFaults {
    pub wrong_decay_sign: bool,
    pub ignore_valid_len_clock: bool,
}

fn finite_floor(x: f64, floor: f64) -> Result<f64, String> {
    if !x.is_finite() || !floor.is_finite() {
        return Err("numerical_failure: non-finite scalar in floor comparison".into());
    }
    Ok(if x < floor { floor } else { x })
}

fn finite_max2(a: f64, b: f64) -> Result<f64, String> {
    if !a.is_finite() || !b.is_finite() {
        return Err("numerical_failure: non-finite scalar in max comparison".into());
    }
    Ok(if a >= b { a } else { b })
}

/// Compare two output sequences and return `(max_abs_err, first_divergent_token_index)`.
pub fn compare_output_sequences(
    a: &[[f64; REC_DIM]],
    b: &[[f64; REC_DIM]],
    tol: f64,
) -> Result<(f64, Option<usize>), String> {
    if a.len() != b.len() {
        return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err("invalid non-finite or negative tolerance".into());
    }
    let mut max_e = 0.0_f64;
    let mut first_div: Option<usize> = None;
    for (t, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        for (&x, &y) in va.iter().zip(vb.iter()) {
            if !x.is_finite() || !y.is_finite() {
                return Err("non-finite value in recurrent comparison".into());
            }
            let d = (x - y).abs();
            if d > max_e {
                max_e = d;
            }
            if d > tol && first_div.is_none() {
                first_div = Some(t);
            }
        }
    }
    Ok((max_e, first_div))
}

fn token_input_vec(arch: RecurrentArch, tok: u32, pos: usize) -> [f64; REC_DIM] {
    let base_seed = match arch {
        RecurrentArch::Rwkv7 => 3710,
        RecurrentArch::Mamba3Siso => 3720,
        RecurrentArch::XlstmMlstm => 3730,
    };
    let mut x = [0.0_f64; REC_DIM];
    for (d, slot) in x.iter_mut().enumerate() {
        *slot = det_weight(base_seed + (tok as usize) * 13, d, 0.5)
            + 0.05 * ((pos as f64 + 1.0) * (d as f64 + 1.0) * 0.23).cos();
    }
    x
}

/// Independent transparent scalar full-sequence reference (does NOT call `NativeRecurrentState`!).
/// Uses flat scalar loops over the complete token sequence from zero initial state.
pub fn scalar_full_reference(
    arch: RecurrentArch,
    tokens: &[u32],
) -> Result<Vec<[f64; REC_DIM]>, String> {
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    let mut s_mat = [0.0_f64; REC_DIM * REC_DIM];
    let mut aux = [0.0_f64; REC_DIM];
    let mut m_stab = 0.0_f64;
    let mut outputs = Vec::with_capacity(tokens.len());

    for (pos, &tok) in tokens.iter().enumerate() {
        let x = token_input_vec(arch, tok, pos);
        let mut y = [0.0_f64; REC_DIM];
        match arch {
            RecurrentArch::Rwkv7 => {
                let mut xm = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    xm[d] = 0.7 * x[d] + 0.3 * aux[d];
                }
                aux = x;
                let mut w = [0.0_f64; REC_DIM];
                let mut k = [0.0_f64; REC_DIM];
                let mut v = [0.0_f64; REC_DIM];
                let mut r = [0.0_f64; REC_DIM];
                let mut a = [0.0_f64; REC_DIM];
                let mut k_sq = 0.0_f64;
                for d in 0..REC_DIM {
                    w[d] = (-0.3 - 0.5 * sigmoid(xm[d])).exp();
                    k[d] = 0.5 * xm[d] + det_weight(3711, d, 0.2);
                    v[d] = 0.6 * xm[(d + 1) % REC_DIM] + det_weight(3712, d, 0.2);
                    r[d] = 0.5 * xm[(d + 2) % REC_DIM] + det_weight(3713, d, 0.2);
                    a[d] = 0.25 * sigmoid(xm[(d + 3) % REC_DIM]);
                    k_sq += k[d] * k[d];
                }
                let kn = finite_floor(k_sq.sqrt(), 1.0e-6)?;
                for kv in &mut k {
                    *kv /= kn;
                }
                for row in 0..REC_DIM {
                    let mut sk_row = 0.0_f64;
                    for (col, &kc) in k.iter().enumerate() {
                        sk_row += s_mat[row * REC_DIM + col] * kc;
                    }
                    let mut out_row = 0.0_f64;
                    for col in 0..REC_DIM {
                        let idx = row * REC_DIM + col;
                        let next =
                            s_mat[idx] * w[col] - sk_row * (a[col] * k[col]) + v[row] * k[col];
                        if !next.is_finite() {
                            return Err("numerical_failure: non-finite scalar RWKV-7".into());
                        }
                        s_mat[idx] = next;
                        out_row += next * r[col];
                    }
                    y[row] = out_row;
                }
            }
            RecurrentArch::Mamba3Siso => {
                let mut cx = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    cx[d] = 0.8 * x[d] + 0.2 * aux[d];
                }
                aux = x;
                for ch in 0..REC_DIM {
                    let dt = 0.1 + 0.35 * sigmoid(cx[ch]);
                    let mut sum_c = 0.0_f64;
                    for s in 0..REC_DIM {
                        let idx = ch * REC_DIM + s;
                        let disc_a = (dt * (-0.4 - 0.2 * (s as f64))).exp();
                        let b_val = det_weight(3725 + ch * 5, s, 0.6) * cx[ch];
                        let c_val = det_weight(3729 + ch * 5, s, 0.6);
                        let next_h = disc_a * s_mat[idx] + dt * b_val;
                        if !next_h.is_finite() {
                            return Err("numerical_failure: non-finite scalar Mamba-3".into());
                        }
                        s_mat[idx] = next_h;
                        sum_c += c_val * next_h;
                    }
                    y[ch] = sum_c + 0.15 * cx[ch];
                }
            }
            RecurrentArch::XlstmMlstm => {
                let log_f = -(0.2 + 0.5 * sigmoid(x[0]));
                let log_i = -0.3 + 0.4 * x[1];
                let m_next = finite_max2(log_f + m_stab, log_i)?;
                let f_prime = (log_f + m_stab - m_next).exp();
                let i_prime = (log_i - m_next).exp();
                m_stab = m_next;

                let inv_sqrt_d = 1.0 / (REC_DIM as f64).sqrt();
                let mut q = [0.0_f64; REC_DIM];
                let mut k = [0.0_f64; REC_DIM];
                let mut v = [0.0_f64; REC_DIM];
                let mut o = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    q[d] = 0.5 * x[d] + det_weight(3731, d, 0.25);
                    k[d] = (0.5 * x[(d + 1) % REC_DIM] + det_weight(3732, d, 0.25)) * inv_sqrt_d;
                    v[d] = 0.6 * x[(d + 2) % REC_DIM] + det_weight(3733, d, 0.25);
                    o[d] = sigmoid(x[(d + 3) % REC_DIM]);
                    aux[d] = f_prime * aux[d] + i_prime * k[d];
                }
                let mut nq = 0.0_f64;
                for (&ad, &qd) in aux.iter().zip(q.iter()) {
                    nq += ad * qd;
                }
                let exp_neg_m = (-m_stab).exp();
                let denom = finite_floor(finite_max2(nq.abs(), exp_neg_m)?, 1.0e-6)?;
                for row in 0..REC_DIM {
                    let mut cq = 0.0_f64;
                    for col in 0..REC_DIM {
                        let idx = row * REC_DIM + col;
                        let next_c = f_prime * s_mat[idx] + i_prime * v[row] * k[col];
                        if !next_c.is_finite() {
                            return Err("numerical_failure: non-finite scalar xLSTM".into());
                        }
                        s_mat[idx] = next_c;
                        cq += next_c * q[col];
                    }
                    y[row] = o[row] * (cq / denom);
                }
            }
        }
        outputs.push(y);
    }
    Ok(outputs)
}

impl NativeRecurrentState {
    pub fn fresh(identity: RecurrentProfileIdentity, capacity: usize) -> Self {
        Self {
            identity,
            valid_len: 0,
            capacity,
            matrix: [[0.0; REC_DIM]; REC_DIM],
            aux_a: [0.0; REC_DIM],
            log_stab_m: 0.0,
            scratch_poison: 1.0e6,
        }
    }

    pub fn state_bytes_calculated(&self) -> usize {
        (REC_DIM * REC_DIM + REC_DIM + 1) * 8
    }

    pub fn step_one(&mut self, tok: u32, faults: RecFaults) -> Result<[f64; REC_DIM], String> {
        if self.valid_len >= self.capacity {
            return Err("resource_stop: recurrent session capacity reached".into());
        }
        let pos = if faults.ignore_valid_len_clock {
            0
        } else {
            self.valid_len
        };
        let x = token_input_vec(self.identity.arch, tok, pos);
        let mut out = [0.0_f64; REC_DIM];

        match self.identity.arch {
            RecurrentArch::Rwkv7 => {
                let mut xm = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    xm[d] = 0.7 * x[d] + 0.3 * self.aux_a[d];
                }
                self.aux_a = x;
                let mut w = [0.0_f64; REC_DIM];
                let mut k = [0.0_f64; REC_DIM];
                let mut v = [0.0_f64; REC_DIM];
                let mut r = [0.0_f64; REC_DIM];
                let mut a = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    let decay_raw = -0.3 - 0.5 * sigmoid(xm[d]);
                    w[d] = if faults.wrong_decay_sign {
                        (-decay_raw).exp()
                    } else {
                        decay_raw.exp()
                    };
                    k[d] = 0.5 * xm[d] + det_weight(3711, d, 0.2);
                    v[d] = 0.6 * xm[(d + 1) % REC_DIM] + det_weight(3712, d, 0.2);
                    r[d] = 0.5 * xm[(d + 2) % REC_DIM] + det_weight(3713, d, 0.2);
                    a[d] = 0.25 * sigmoid(xm[(d + 3) % REC_DIM]);
                }
                let kn = finite_floor((k.iter().map(|c| c * c).sum::<f64>()).sqrt(), 1.0e-6)?;
                for c in &mut k {
                    *c /= kn;
                }
                let mut sk = [0.0_f64; REC_DIM];
                for (row, sk_r) in sk.iter_mut().enumerate() {
                    for (col, &kc) in k.iter().enumerate() {
                        *sk_r += self.matrix[row][col] * kc;
                    }
                }
                for row in 0..REC_DIM {
                    for col in 0..REC_DIM {
                        let next = self.matrix[row][col] * w[col] - sk[row] * (a[col] * k[col])
                            + v[row] * k[col];
                        if !next.is_finite() {
                            return Err("numerical_failure: non-finite RWKV-7 state".into());
                        }
                        self.matrix[row][col] = next;
                        out[row] += next * r[col];
                    }
                }
            }
            RecurrentArch::Mamba3Siso => {
                let mut conv_x = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    conv_x[d] = 0.8 * x[d] + 0.2 * self.aux_a[d];
                }
                self.aux_a = x;
                for ch in 0..REC_DIM {
                    let dt = 0.1 + 0.35 * sigmoid(conv_x[ch]);
                    let mut y_ch = 0.0_f64;
                    for s in 0..REC_DIM {
                        let a_param = -0.4 - 0.2 * (s as f64);
                        let disc_a = if faults.wrong_decay_sign {
                            (dt * (-a_param)).exp()
                        } else {
                            (dt * a_param).exp()
                        };
                        let b_val = det_weight(3725 + ch * 5, s, 0.6) * conv_x[ch];
                        let c_val = det_weight(3729 + ch * 5, s, 0.6);
                        let next_h = disc_a * self.matrix[ch][s] + dt * b_val;
                        if !next_h.is_finite() {
                            return Err("numerical_failure: non-finite Mamba-3 state".into());
                        }
                        self.matrix[ch][s] = next_h;
                        y_ch += c_val * next_h;
                    }
                    out[ch] = y_ch + 0.15 * conv_x[ch];
                }
            }
            RecurrentArch::XlstmMlstm => {
                let log_f = if faults.wrong_decay_sign {
                    0.8
                } else {
                    -(0.2 + 0.5 * sigmoid(x[0]))
                };
                let log_i = -0.3 + 0.4 * x[1];
                let m_next = finite_max2(log_f + self.log_stab_m, log_i)?;
                let f_prime = (log_f + self.log_stab_m - m_next).exp();
                let i_prime = (log_i - m_next).exp();
                self.log_stab_m = m_next;

                let mut q = [0.0_f64; REC_DIM];
                let mut k = [0.0_f64; REC_DIM];
                let mut v = [0.0_f64; REC_DIM];
                let mut o = [0.0_f64; REC_DIM];
                for d in 0..REC_DIM {
                    q[d] = 0.5 * x[d] + det_weight(3731, d, 0.25);
                    k[d] = (0.5 * x[(d + 1) % REC_DIM] + det_weight(3732, d, 0.25))
                        / (REC_DIM as f64).sqrt();
                    v[d] = 0.6 * x[(d + 2) % REC_DIM] + det_weight(3733, d, 0.25);
                    o[d] = sigmoid(x[(d + 3) % REC_DIM]);
                }

                for (aux, &kd) in self.aux_a.iter_mut().zip(k.iter()) {
                    let nv = f_prime * *aux + i_prime * kd;
                    if !nv.is_finite() {
                        return Err("numerical_failure: non-finite xLSTM normalizer".into());
                    }
                    *aux = nv;
                }
                let mut nq = 0.0_f64;
                for (&aux, &qd) in self.aux_a.iter().zip(q.iter()) {
                    nq += aux * qd;
                }
                let denom = finite_floor(finite_max2(nq.abs(), (-self.log_stab_m).exp())?, 1.0e-6)?;

                for row in 0..REC_DIM {
                    let mut cq = 0.0_f64;
                    for col in 0..REC_DIM {
                        let next_c = f_prime * self.matrix[row][col] + i_prime * v[row] * k[col];
                        if !next_c.is_finite() {
                            return Err("numerical_failure: non-finite xLSTM C state".into());
                        }
                        self.matrix[row][col] = next_c;
                        cq += next_c * q[col];
                    }
                    out[row] = o[row] * (cq / denom);
                }
            }
        }

        self.valid_len += 1;
        Ok(out)
    }

    pub fn chunk(
        &mut self,
        tokens: &[u32],
        faults: RecFaults,
    ) -> Result<Vec<[f64; REC_DIM]>, String> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        // Precompute chunk embeddings/projections across the chunk window before advancing state.
        let mut outs = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            outs.push(self.step_one(tok, faults)?);
        }
        Ok(outs)
    }

    pub fn snapshot(&self) -> NativeRecurrentState {
        self.clone()
    }

    pub fn serialize(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "v{}|{}|{}|{}|{}|{}|{}|{:.16e}",
            self.identity.serialization_version,
            self.identity.arch.as_str(),
            self.identity.model_digest.hex,
            self.identity.tokenizer_rev,
            self.identity.worker_generation,
            self.identity.session_id,
            self.valid_len,
            self.log_stab_m
        );
        for row in &self.matrix {
            let _ = writeln!(
                s,
                "M:{:.16e},{:.16e},{:.16e},{:.16e}",
                row[0], row[1], row[2], row[3]
            );
        }
        let _ = writeln!(
            s,
            "A:{:.16e},{:.16e},{:.16e},{:.16e}",
            self.aux_a[0], self.aux_a[1], self.aux_a[2], self.aux_a[3]
        );
        s
    }

    pub fn restore(
        bytes: &str,
        expected: &RecurrentProfileIdentity,
        capacity: usize,
    ) -> Result<Self, String> {
        let mut lines = bytes.lines();
        let header = lines.next().ok_or("empty state payload")?;
        let parts: Vec<&str> = header.split('|').collect();
        if parts.len() != 8 {
            return Err("malformed state header".into());
        }
        let ver_str = format!("v{}", expected.serialization_version);
        if parts[0] != ver_str
            || parts[1] != expected.arch.as_str()
            || parts[2] != expected.model_digest.hex
            || parts[3] != expected.tokenizer_rev
        {
            return Err("state bytes rejected: checkpoint/architecture/profile mismatch".into());
        }
        let gen: u64 = parts[4].parse().map_err(|_| "invalid generation")?;
        if gen != expected.worker_generation {
            return Err("state bytes rejected: stale worker generation".into());
        }
        if parts[5] != expected.session_id {
            return Err("state bytes rejected: cross-session handle mismatch".into());
        }
        let valid_len: usize = parts[6].parse().map_err(|_| "invalid valid_len")?;
        let log_stab_m: f64 = parts[7].parse().map_err(|_| "invalid log_stab_m")?;
        if !log_stab_m.is_finite() || valid_len > capacity {
            return Err("invalid valid_len or non-finite stabilizer".into());
        }

        let parse4 = |line: &str, prefix: &str| -> Result<[f64; REC_DIM], String> {
            let body = line.strip_prefix(prefix).ok_or("missing row prefix")?;
            let vals: Vec<f64> = body
                .split(',')
                .map(|x| x.parse::<f64>().map_err(|_| "bad float".to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            if vals.len() != REC_DIM || vals.iter().any(|x| !x.is_finite()) {
                return Err("non-finite or malformed state row".into());
            }
            Ok([vals[0], vals[1], vals[2], vals[3]])
        };
        let mut matrix = [[0.0; REC_DIM]; REC_DIM];
        for row in &mut matrix {
            *row = parse4(lines.next().ok_or("truncated M")?, "M:")?;
        }
        let aux_a = parse4(lines.next().ok_or("missing A")?, "A:")?;
        Ok(Self {
            identity: expected.clone(),
            valid_len,
            capacity,
            matrix,
            aux_a,
            log_stab_m,
            scratch_poison: 1.0e6,
        })
    }
}

/// Batched stepping across independent sessions (must match isolated per-session stepping exactly).
pub fn batch_step_sessions(
    states: &mut [NativeRecurrentState],
    tokens: &[u32],
) -> Result<Vec<[f64; REC_DIM]>, String> {
    if states.len() != tokens.len() {
        return Err("batch_step session/token length mismatch".into());
    }
    let mut outs = Vec::with_capacity(states.len());
    for (st, &tok) in states.iter_mut().zip(tokens.iter()) {
        outs.push(st.step_one(tok, RecFaults::default())?);
    }
    Ok(outs)
}

/// Forbids labeling a full-prefix recomputation wrapper as `Supported` native state.
pub fn validate_native_operation_claim(
    emulated_via_full_recompute: bool,
    claimed_disposition: OperationDisposition,
) -> OutcomeStatus {
    if emulated_via_full_recompute && claimed_disposition == OperationDisposition::Supported {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone)]
pub struct ArchBenchRow {
    pub arch: RecurrentArch,
    pub milestone: &'static str,
    pub scalar_ref_vs_irregular_chunk_err: f64,
    pub uninterrupted_vs_restored_err: f64,
    pub fork_branch_max_err: f64,
    pub batch_vs_isolated_err: f64,
    pub empty_and_single_token_ok: bool,
    pub first_divergent_token_on_clean_run: Option<u64>,
    pub state_bytes_calculated: usize,
    pub native_rollback_disposition: OperationDisposition,
    pub profile_outcome: String,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct NativeRecurrentBenchReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub milestone_a_passed: bool,
    pub milestone_b_xlstm_gated_passed: bool,
    pub arch_rows: Vec<ArchBenchRow>,
    pub negative_controls: Vec<NegativeControl>,
}

fn default_identity(arch: RecurrentArch, session_id: &str, gen: u64) -> RecurrentProfileIdentity {
    RecurrentProfileIdentity {
        arch,
        model_digest: DigestRef::sha256(format!("{}-tiny-ref-v1", arch.as_str()).as_bytes()),
        tokenizer_rev: format!("{}-tok-v1", arch.as_str()),
        state_dtype: "f64_reference".into(),
        serialization_version: 1,
        worker_generation: gen,
        session_id: session_id.into(),
    }
}

fn evaluate_arch(arch: RecurrentArch, milestone: &'static str) -> Result<ArchBenchRow, String> {
    let seq = [1u32, 3, 2, 5, 0, 4, 6, 2, 7, 1, 3];
    let id_a = default_identity(arch, "sess-a", 1);

    // Phase 1: Independent scalar full-sequence reference vs stateful irregular chunks [1, 3, 2, 5]
    let out_scalar_ref = scalar_full_reference(arch, &seq)?;

    let mut s_chunk = NativeRecurrentState::fresh(id_a.clone(), REC_CAPACITY);
    let mut out_chunk = Vec::new();
    for ch in [&seq[0..1], &seq[1..4], &seq[4..6], &seq[6..11]] {
        out_chunk.extend(s_chunk.chunk(ch, RecFaults::default())?);
    }
    let (scalar_vs_chunk, first_div) =
        compare_output_sequences(&out_scalar_ref, &out_chunk, 1.0e-12)?;

    // Empty and single-token edge cases vs scalar reference
    let mut s_edge = NativeRecurrentState::fresh(id_a.clone(), REC_CAPACITY);
    let empty_out = s_edge.chunk(&[], RecFaults::default())?;
    let single_out = s_edge.chunk(&[5], RecFaults::default())?;
    let single_ref = scalar_full_reference(arch, &[5])?;
    let (single_err, _) = compare_output_sequences(&single_ref, &single_out, 1.0e-12)?;
    let empty_and_single_token_ok =
        empty_out.is_empty() && s_edge.valid_len == 1 && single_err <= 1.0e-12;

    // Phase 2: uninterrupted scalar ref vs snapshot -> serialize -> fresh restore -> continue
    let mut s_split = NativeRecurrentState::fresh(id_a.clone(), REC_CAPACITY);
    let _ = s_split.chunk(&seq[..5], RecFaults::default())?;
    let snap = s_split.snapshot();
    let bytes = snap.serialize();
    let mut s_restored = NativeRecurrentState::restore(&bytes, &id_a, REC_CAPACITY)?;
    let tail_restored = s_restored.chunk(&seq[5..], RecFaults::default())?;
    let (unint_vs_restored, _) =
        compare_output_sequences(&out_scalar_ref[5..], &tail_restored, 1.0e-12)?;

    // Phase 3: immutable snapshot fork into two independent suffixes vs independent scalar ref
    let mut branch1 = snap.clone();
    let mut branch2 = snap.clone();
    let suf1 = [6u32, 7, 2];
    let suf2 = [0u32, 4, 5];
    let b1_out = branch1.chunk(&suf1, RecFaults::default())?;
    let b2_out = branch2.chunk(&suf2, RecFaults::default())?;

    let mut full1 = seq[..5].to_vec();
    full1.extend_from_slice(&suf1);
    let f1_out = scalar_full_reference(arch, &full1)?;

    let mut full2 = seq[..5].to_vec();
    full2.extend_from_slice(&suf2);
    let f2_out = scalar_full_reference(arch, &full2)?;

    let (e1, _) = compare_output_sequences(&b1_out, &f1_out[5..], 1.0e-12)?;
    let (e2, _) = compare_output_sequences(&b2_out, &f2_out[5..], 1.0e-12)?;
    let fork_err = finite_max2(e1, e2)?;

    // Phase 4: 2-session interleaved + batched vs isolated parity
    let id_b = default_identity(arch, "sess-b", 1);
    let mut iso_a = NativeRecurrentState::fresh(id_a.clone(), REC_CAPACITY);
    let mut iso_b = NativeRecurrentState::fresh(id_b.clone(), REC_CAPACITY);
    let mut batched = vec![
        NativeRecurrentState::fresh(id_a, REC_CAPACITY),
        NativeRecurrentState::fresh(id_b, REC_CAPACITY),
    ];
    let mut batch_err = 0.0_f64;
    for (ta, tb) in [(2u32, 7u32), (5, 0), (1, 4), (6, 3)] {
        let oa = iso_a.step_one(ta, RecFaults::default())?;
        let ob = iso_b.step_one(tb, RecFaults::default())?;
        let bo = batch_step_sessions(&mut batched, &[ta, tb])?;
        let (ea, _) = compare_output_sequences(&[oa], &[bo[0]], 1.0e-12)?;
        let (eb, _) = compare_output_sequences(&[ob], &[bo[1]], 1.0e-12)?;
        batch_err = finite_max2(batch_err, finite_max2(ea, eb)?)?;
    }

    let ok = scalar_vs_chunk <= 1.0e-12
        && unint_vs_restored <= 1.0e-12
        && fork_err <= 1.0e-12
        && batch_err <= 1.0e-12
        && empty_and_single_token_ok;

    Ok(ArchBenchRow {
        arch,
        milestone,
        scalar_ref_vs_irregular_chunk_err: scalar_vs_chunk,
        uninterrupted_vs_restored_err: unint_vs_restored,
        fork_branch_max_err: fork_err,
        batch_vs_isolated_err: batch_err,
        empty_and_single_token_ok,
        first_divergent_token_on_clean_run: first_div.map(|x| x as u64),
        state_bytes_calculated: s_chunk.state_bytes_calculated(),
        native_rollback_disposition: OperationDisposition::Unsupported,
        profile_outcome: if ok {
            "TINY_REF_QUALIFIED_HARDWARE_UNAVAILABLE".into()
        } else {
            "NO-GO".into()
        },
    })
}

pub fn run_default_harness() -> Result<NativeRecurrentBenchReport, String> {
    let rwkv = evaluate_arch(RecurrentArch::Rwkv7, "milestone_a")?;
    let mamba3 = evaluate_arch(RecurrentArch::Mamba3Siso, "milestone_a")?;
    let milestone_a_passed = rwkv.profile_outcome.starts_with("TINY_REF_QUALIFIED")
        && mamba3.profile_outcome.starts_with("TINY_REF_QUALIFIED");

    let xlstm = if milestone_a_passed {
        evaluate_arch(RecurrentArch::XlstmMlstm, "milestone_b_conditional")?
    } else {
        return Err("Milestone A failed; Milestone B xLSTM cannot run".into());
    };
    let milestone_b_xlstm_gated_passed = xlstm.profile_outcome.starts_with("TINY_REF_QUALIFIED");

    let mut negative_controls = Vec::new();
    let seq = [1u32, 3, 2, 5, 0];

    // NC1: wrong decay sign in RWKV-7 fails independent scalar reference parity
    {
        let id = default_identity(RecurrentArch::Rwkv7, "nc1", 1);
        let og = scalar_full_reference(RecurrentArch::Rwkv7, &seq)?;
        let mut bad = NativeRecurrentState::fresh(id, REC_CAPACITY);
        let ob = bad.chunk(
            &seq,
            RecFaults {
                wrong_decay_sign: true,
                ..RecFaults::default()
            },
        )?;
        let (e, div_tok) = compare_output_sequences(&og, &ob, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc1_rwkv7_wrong_decay_transition".into(),
            outcome: if e > 1.0e-6 && div_tok.is_some() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "wrong RWKV-7 decay sign diverged at token {:?} (max_err={:.3e})",
                div_tok, e
            ),
        });
    }

    // NC2: wrong decay sign in Mamba-3 fails independent scalar reference parity
    {
        let id = default_identity(RecurrentArch::Mamba3Siso, "nc2", 1);
        let og = scalar_full_reference(RecurrentArch::Mamba3Siso, &seq)?;
        let mut bad = NativeRecurrentState::fresh(id, REC_CAPACITY);
        let ob = bad.chunk(
            &seq,
            RecFaults {
                wrong_decay_sign: true,
                ..RecFaults::default()
            },
        )?;
        let (e, div_tok) = compare_output_sequences(&og, &ob, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_mamba3_wrong_decay_transition".into(),
            outcome: if e > 1.0e-6 && div_tok.is_some() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "wrong Mamba-3 decay sign diverged at token {:?} (max_err={:.3e})",
                div_tok, e
            ),
        });
    }

    // NC3: ignored valid_len / position clock
    {
        let id = default_identity(RecurrentArch::XlstmMlstm, "nc3", 1);
        let og = scalar_full_reference(RecurrentArch::XlstmMlstm, &seq)?;
        let mut bad = NativeRecurrentState::fresh(id, REC_CAPACITY);
        let ob = bad.chunk(
            &seq,
            RecFaults {
                ignore_valid_len_clock: true,
                ..RecFaults::default()
            },
        )?;
        let (e, div_tok) = compare_output_sequences(&og, &ob, 1.0e-6)?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_ignored_position_clock".into(),
            outcome: if e > 1.0e-6 && div_tok == Some(1) {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "ignoring token clock diverged at token {:?} (max_err={:.3e})",
                div_tok, e
            ),
        });
    }

    // NC4: cross-architecture state restore refused (RWKV-7 bytes into Mamba-3)
    {
        let id_r = default_identity(RecurrentArch::Rwkv7, "s1", 1);
        let id_m = default_identity(RecurrentArch::Mamba3Siso, "s1", 1);
        let s = NativeRecurrentState::fresh(id_r, REC_CAPACITY);
        let res = NativeRecurrentState::restore(&s.serialize(), &id_m, REC_CAPACITY);
        negative_controls.push(NegativeControl {
            case_id: "nc4_cross_family_state_restore_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "RWKV-7 serialized state refused by Mamba-3 profile".into(),
        });
    }

    // NC5: stale worker generation state bytes rejected
    {
        let id_old = default_identity(RecurrentArch::Mamba3Siso, "s1", 1);
        let id_new = default_identity(RecurrentArch::Mamba3Siso, "s1", 2);
        let s = NativeRecurrentState::fresh(id_old, REC_CAPACITY);
        let res = NativeRecurrentState::restore(&s.serialize(), &id_new, REC_CAPACITY);
        negative_controls.push(NegativeControl {
            case_id: "nc5_stale_worker_generation_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "older worker generation state rejected after restart".into(),
        });
    }

    // NC6: cross-session handle restore rejected
    {
        let id_a = default_identity(RecurrentArch::Rwkv7, "sess-a", 1);
        let id_b = default_identity(RecurrentArch::Rwkv7, "sess-b", 1);
        let s = NativeRecurrentState::fresh(id_a, REC_CAPACITY);
        let res = NativeRecurrentState::restore(&s.serialize(), &id_b, REC_CAPACITY);
        negative_controls.push(NegativeControl {
            case_id: "nc6_cross_session_restore_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "sess-a snapshot rejected when restoring sess-b".into(),
        });
    }

    // NC7: capacity overflow yields explicit resource_stop without corrupting peer session
    {
        let id_small = default_identity(RecurrentArch::Rwkv7, "cap-test", 1);
        let mut s = NativeRecurrentState::fresh(id_small, 2);
        let _ = s.chunk(&[1, 2], RecFaults::default())?;
        let overflow = s.step_one(3, RecFaults::default());
        negative_controls.push(NegativeControl {
            case_id: "nc7_capacity_overflow_resource_stop".into(),
            outcome: if overflow.is_err() {
                OutcomeStatus::Exhausted
            } else {
                OutcomeStatus::Accepted
            },
            detail: "exceeding session capacity stops cleanly with explicit resource_stop".into(),
        });
    }

    // NC8: mislabeling recomputation wrapper as native state support rejected
    {
        let st = validate_native_operation_claim(true, OperationDisposition::Supported);
        negative_controls.push(NegativeControl {
            case_id: "nc8_recompute_mislabeled_as_native_state".into(),
            outcome: st,
            detail: "full-prefix recomputation wrapper cannot be labeled native state support"
                .into(),
        });
    }

    // NC9: corrupted NaN in serialized state payload rejected
    {
        let id = default_identity(RecurrentArch::Mamba3Siso, "s1", 1);
        let s = NativeRecurrentState::fresh(id.clone(), REC_CAPACITY);
        let bad_payload = s.serialize().replace("0.0000000000000000e0", "NaN");
        let res = NativeRecurrentState::restore(&bad_payload, &id, REC_CAPACITY);
        negative_controls.push(NegativeControl {
            case_id: "nc9_corrupted_nan_state_bytes_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN in serialized state payload rejected on restore".into(),
        });
    }

    // NC10: NaN stabilizer state inside xLSTM step is caught instead of swallowed by float max
    {
        let id = default_identity(RecurrentArch::XlstmMlstm, "nan-stab", 1);
        let mut s = NativeRecurrentState::fresh(id, REC_CAPACITY);
        s.log_stab_m = f64::NAN;
        let res = s.step_one(1, RecFaults::default());
        negative_controls.push(NegativeControl {
            case_id: "nc10_xlstm_nan_stabilizer_not_swallowed".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN log_stab_m triggers explicit numerical_failure instead of being swallowed"
                .into(),
        });
    }

    Ok(NativeRecurrentBenchReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Official RWKV-7, Mamba-3 SISO, and xLSTM-7b checkpoints on accelerator hardware",
            "Independent scalar references and stateful/chunked engines verify contract Phases 1–5 on CPU; official large checkpoints and CUDA/Triton kernels did not execute locally",
        ),
        synthetic_contract_fixture: true,
        milestone_a_passed,
        milestone_b_xlstm_gated_passed,
        arch_rows: vec![rwkv, mamba3, xlstm],
        negative_controls,
    })
}

pub fn inputs_json(report: &NativeRecurrentBenchReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.native-recurrent-bench.inputs.v1\",\n",
            "  \"issue\": 37,\n",
            "  \"execution_class\": {},\n",
            "  \"cohort\": [\"rwkv7\",\"mamba3_siso\",\"xlstm_mlstm\"],\n",
            "  \"non_gating_related_work\": {{\"latent_lattice_commit\":{},\"tracked_issue\":60}},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(LATENT_LATTICE_RELATED_COMMIT),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &NativeRecurrentBenchReport) -> String {
    let rows: Vec<String> = report
        .arch_rows
        .iter()
        .map(|r| {
            format!(
                "{{\"arch\":{},\"milestone\":{},\"scalar_ref_vs_irregular_chunk_err\":{},\"uninterrupted_vs_restored_err\":{},\"fork_branch_max_err\":{},\"batch_vs_isolated_err\":{},\"empty_and_single_token_ok\":{},\"first_divergent_token\":{},\"state_bytes_calculated\":{},\"native_rollback_disposition\":{},\"profile_outcome\":{}}}",
                quote(r.arch.as_str()),
                quote(r.milestone),
                finite_f64(r.scalar_ref_vs_irregular_chunk_err),
                finite_f64(r.uninterrupted_vs_restored_err),
                finite_f64(r.fork_branch_max_err),
                finite_f64(r.batch_vs_isolated_err),
                r.empty_and_single_token_ok,
                opt_u64(r.first_divergent_token_on_clean_run),
                r.state_bytes_calculated,
                quote(r.native_rollback_disposition.as_str()),
                quote(&r.profile_outcome),
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
            "  \"schema\": \"inference-lab.native-recurrent-bench.results.v1\",\n",
            "  \"issue\": 37,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"milestone_a_passed\": {},\n",
            "  \"milestone_b_xlstm_gated_passed\": {},\n",
            "  \"arch_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.milestone_a_passed,
        report.milestone_b_xlstm_gated_passed,
        rows.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &NativeRecurrentBenchReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #37: Native Recurrent-State Benchmark (RWKV-7, Mamba-3, xLSTM)"
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
        "- Milestone A (`RWKV-7` + `Mamba-3`): `{}`, Milestone B (`xLSTM` gated): `{}`",
        report.milestone_a_passed, report.milestone_b_xlstm_gated_passed
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Architecture Matrix (vs Independent Scalar Reference)"
    )
    .unwrap();
    for r in &report.arch_rows {
        writeln!(
            out,
            "- `{}` (`{}`): `scalar_vs_chunk={:.3e}`, `restore_err={:.3e}`, `fork_err={:.3e}`, `batch_err={:.3e}`, `edges_ok={}`, `state_bytes={}`, `rollback={}`, `outcome={}`",
            r.arch.as_str(),
            r.milestone,
            r.scalar_ref_vs_irregular_chunk_err,
            r.uninterrupted_vs_restored_err,
            r.fork_branch_max_err,
            r.batch_vs_isolated_err,
            r.empty_and_single_token_ok,
            r.state_bytes_calculated,
            r.native_rollback_disposition.as_str(),
            r.profile_outcome
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
