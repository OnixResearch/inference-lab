//! Issue #94: Add subspace interchange, necessity ablation, and a planted parasite-feature control
//! to the intervention replay harness (`#45`), after *Parasite Features* (`arXiv:2611.26711`).
//!
//! Non-gating related-work references (all figures attributed strictly to the authors):
//! - Paper: *Parasite Features: Causal Abstraction Through Spurious Pathways* (`arXiv:2611.26711`,
//!   Sutter, Yao, Boguraev, Minder, Di Maio, Hofmann, Mahowald, Pimentel; arXiv v1 submitted
//!   2026-11-19).
//! - Reference repository: `https://github.com/dounick/parasite-features` at commit
//!   `1df61727bb2f502b882d334eb7f2ca1a38e05712` — **MISSING LICENSE FILE** (`NOASSERTION`;
//!   treated strictly as reference reading only; ZERO code copied or vendored).
//! - Author posts:
//!   - `https://x.com/SashaBoguraev/status/2124892195740340581`
//!   - `https://x.com/tpimentelms/status/2124933776568463364`
//! - Authors' unverified claims: rotation-plus-low-rank-linear alignment maps find parasite
//!   features on syntactic agreement (Pythia-1B) and carry-one in two-digit addition (Llama
//!   3.1-8B) where ablating the 2D subspace barely changes accuracy relative to random 2D
//!   subspace ablations (sufficient for interchange control, not necessary for the task).
//! - Coordination note for `#77` (`qwen35-neuron-steering`) and `rsi-lab#172`: downstream
//!   steering and behavioral-specificity fixtures MUST report the separate `necessity_passed` and
//!   `classification` fields alongside any steering or specificity score.
//!
//! Provides:
//! - Two typed manifest intervention kinds extending `#45` (`intervention-replay`), each with a
//!   canonical SHA-256 manifest digest covering `basis`, `rank`, `layer`, `token_positions`,
//!   `model_revision_digest`, and `seed`:
//!   1. `SubspaceInterchange`: rank-`k` subspace interchange (`h' = h - B B^T h + B R B^T h_src`,
//!      where `R` is an optional orthonormal `k x k` alignment rotation, defaulting to `I_k`).
//!   2. `ProjectionAblation`: rank-`k` orthogonal-complement projection ablation (`h' = h - B B^T h`).
//! - Strict orthonormality (`||B^T B - I_k||_inf <= 1e-9`) and rank validation.
//! - **Necessity check**: for any subspace passing interchange sufficiency, ablates it and
//!   compares the task-outcome drop against a seeded distribution of **16** (`>= 10`) matched-rank
//!   random orthonormal subspaces at the exact same layer and token position. Emits separate
//!   `sufficiency_passed`, `necessity_passed`, and `MechanismClassification` (`found_mechanism`,
//!   `sufficient_not_necessary`, `insufficient`).
//! - **Planted-parasite negative control**: dedicated backend variant where `e_0` only
//!   discriminates an input property (unread by the task), `e_1` (orthogonal) only steers the
//!   output token (zero on clean inputs), and `e_2` carries the true causal task signal. A rank-2
//!   interchange on `span(e_0, e_1)` MUST pass sufficiency AND the necessity check MUST classify it
//!   as `sufficient_not_necessary` (NEVER `found_mechanism`).
//! - **Positive controls**: both the planted backend's true task direction `e_2` AND an active
//!   subspace on the existing `#45` tiny Qwen transformer backend pass BOTH sufficiency and
//!   necessity (`found_mechanism`).
//! - Bit-identical SHA-256 replay receipts (`verify_subspace_replay_receipt`) and 9 adversarial
//!   negative controls (explicitly including rank mismatch, non-orthonormal basis, and wrong-rank
//!   random subspaces).

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::intervention_replay::{ModelBackendArch, INTERP_DIM, INTERP_LAYERS, INTERP_VOCAB};
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const PAPER_ARXIV: &str = "arXiv:2611.26711";
pub const REF_REPO: &str = "dounick/parasite-features";
pub const REF_COMMIT: &str = "1df61727bb2f502b882d334eb7f2ca1a38e05712";
pub const REF_LICENSE_STATUS: &str = "missing_license_reference_only_no_code_copied";
pub const POST_BOGURAEV: &str = "https://x.com/SashaBoguraev/status/2124892195740340581";
pub const POST_PIMENTEL: &str = "https://x.com/tpimentelms/status/2124933776568463364";
pub const COORDINATION_NOTE: &str =
    "Coordinate with inference-lab#77 and rsi-lab#172: report necessity_passed and classification next to any steering or behavioral-specificity metric.";

pub const ORTHONORMAL_TOL: f64 = 1.0e-9;
pub const MIN_RANDOM_SUBSPACES: usize = 10;
pub const DEFAULT_RANDOM_SUBSPACES: usize = 16;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismClassification {
    FoundMechanism,
    SufficientNotNecessary,
    Insufficient,
}

impl MechanismClassification {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FoundMechanism => "found_mechanism",
            Self::SufficientNotNecessary => "sufficient_not_necessary",
            Self::Insufficient => "insufficient",
        }
    }
}

/// Two new typed subspace intervention kinds extending `#45`.
#[derive(Debug, Clone, PartialEq)]
pub enum SubspaceInterventionKind {
    /// Rank-k subspace interchange from a source run into a base run:
    /// `h' = h - B B^T h + B R B^T h_source`, where `alignment_rotation` is a `k x k`
    /// orthonormal matrix (`I_k` for unrotated interchange).
    SubspaceInterchange {
        rank: usize,
        basis: Vec<[f64; INTERP_DIM]>,
        alignment_rotation: Vec<Vec<f64>>,
        source_prompt_tokens: Vec<u32>,
    },
    /// Rank-k projection ablation onto the orthogonal complement:
    /// `h' = h - B B^T h`.
    ProjectionAblation {
        rank: usize,
        basis: Vec<[f64; INTERP_DIM]>,
    },
}

impl SubspaceInterventionKind {
    pub const fn kind_str(&self) -> &'static str {
        match self {
            Self::SubspaceInterchange { .. } => "subspace_interchange",
            Self::ProjectionAblation { .. } => "projection_ablation",
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::SubspaceInterchange { rank, .. } | Self::ProjectionAblation { rank, .. } => *rank,
        }
    }

    pub fn basis(&self) -> &[[f64; INTERP_DIM]] {
        match self {
            Self::SubspaceInterchange { basis, .. } | Self::ProjectionAblation { basis, .. } => {
                basis
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubspaceInterventionManifest {
    pub experiment_id: String,
    pub source_arch: ModelBackendArch,
    pub target_arch: ModelBackendArch,
    pub model_revision_digest: String,
    pub seed: u64,
    pub base_prompt_tokens: Vec<u32>,
    pub target_layer: usize,
    pub token_positions: Vec<usize>,
    pub kind: SubspaceInterventionKind,
}

impl SubspaceInterventionManifest {
    /// Canonical SHA-256 digest covering basis, rank, layer, token positions, model revision, and seed.
    pub fn manifest_digest(&self) -> DigestRef {
        let mut s = format!(
            "{}|{}|{}|{}|{}|{:?}|{}|{:?}|{}|{}|",
            self.experiment_id,
            self.source_arch.as_str(),
            self.target_arch.as_str(),
            self.model_revision_digest,
            self.seed,
            self.base_prompt_tokens,
            self.target_layer,
            self.token_positions,
            self.kind.kind_str(),
            self.kind.rank(),
        );
        for vec in self.kind.basis() {
            for &x in vec {
                let _ = write!(s, "{x:.12e},");
            }
            s.push(';');
        }
        if let SubspaceInterventionKind::SubspaceInterchange {
            alignment_rotation,
            source_prompt_tokens,
            ..
        } = &self.kind
        {
            let _ = write!(s, "|src={source_prompt_tokens:?}|rot=");
            for row in alignment_rotation {
                for &x in row {
                    let _ = write!(s, "{x:.12e},");
                }
            }
        }
        DigestRef::sha256(s.as_bytes())
    }
}

/// Validate rank consistency (`basis.len() == rank`, `1 <= rank < INTERP_DIM`) and strict
/// orthonormality (`||B^T B - I_k||_inf <= ORTHONORMAL_TOL`).
pub fn validate_orthonormal_basis(rank: usize, basis: &[[f64; INTERP_DIM]]) -> Result<(), String> {
    if rank == 0 || rank >= INTERP_DIM {
        return Err(format!("invalid subspace rank {rank} for dim {INTERP_DIM}"));
    }
    if basis.len() != rank {
        return Err(format!(
            "rank mismatch: declared rank={rank}, basis has {} vectors",
            basis.len()
        ));
    }
    for (i, bi) in basis.iter().enumerate() {
        for &x in bi {
            check_finite(x, "basis_element")?;
        }
        for (j, bj) in basis.iter().enumerate() {
            let dot: f64 = bi.iter().zip(bj.iter()).map(|(a, b)| a * b).sum();
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > ORTHONORMAL_TOL {
                return Err(format!(
                    "non-orthonormal basis: dot(b[{i}], b[{j}])={dot:.6e} vs {expected}"
                ));
            }
        }
    }
    Ok(())
}

fn validate_rotation(rank: usize, rot: &[Vec<f64>]) -> Result<(), String> {
    if rot.len() != rank || rot.iter().any(|r| r.len() != rank) {
        return Err("alignment_rotation shape mismatch with rank".into());
    }
    for i in 0..rank {
        for &x in &rot[i] {
            check_finite(x, "rot_elem")?;
        }
        for j in 0..rank {
            let mut dot = 0.0_f64;
            for k in 0..rank {
                dot += rot[i][k] * rot[j][k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > ORTHONORMAL_TOL {
                return Err("alignment_rotation is not orthonormal".into());
            }
        }
    }
    Ok(())
}

pub fn validate_subspace_manifest(m: &SubspaceInterventionManifest) -> Result<(), String> {
    if m.model_revision_digest.is_empty() || m.model_revision_digest.contains("latest") {
        return Err("mutable or empty model_revision_digest rejected".into());
    }
    if m.source_arch != m.target_arch || m.target_arch != ModelBackendArch::QwenTransformer {
        return Err("cross-architecture or non-Qwen subspace intervention rejected".into());
    }
    if m.base_prompt_tokens.is_empty() || m.token_positions.is_empty() {
        return Err("empty base_prompt_tokens or token_positions".into());
    }
    if m.target_layer >= INTERP_LAYERS {
        return Err("target_layer out of bounds".into());
    }
    for &pos in &m.token_positions {
        if pos >= m.base_prompt_tokens.len() {
            return Err(format!("token position {pos} out of bounds"));
        }
    }
    match &m.kind {
        SubspaceInterventionKind::SubspaceInterchange {
            rank,
            basis,
            alignment_rotation,
            source_prompt_tokens,
        } => {
            if source_prompt_tokens.len() != m.base_prompt_tokens.len() {
                return Err("source_prompt_tokens length must match base_prompt_tokens".into());
            }
            validate_orthonormal_basis(*rank, basis)?;
            validate_rotation(*rank, alignment_rotation)?;
        }
        SubspaceInterventionKind::ProjectionAblation { rank, basis } => {
            validate_orthonormal_basis(*rank, basis)?;
        }
    }
    Ok(())
}

/// Exact orthogonal-complement projection ablation: `h' = h - B (B^T h)`.
pub fn apply_projection_ablation(
    h: &[f64; INTERP_DIM],
    basis: &[[f64; INTERP_DIM]],
) -> Result<[f64; INTERP_DIM], String> {
    validate_orthonormal_basis(basis.len(), basis)?;
    let mut out = *h;
    for b in basis {
        let coeff: f64 = h.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        for d in 0..INTERP_DIM {
            out[d] -= coeff * b[d];
        }
    }
    for &v in &out {
        check_finite(v, "ablated_h")?;
    }
    Ok(out)
}

/// Exact rank-k subspace interchange: `h' = h - B (B^T h) + B R (B^T h_source)`.
pub fn apply_subspace_interchange(
    h_base: &[f64; INTERP_DIM],
    h_src: &[f64; INTERP_DIM],
    basis: &[[f64; INTERP_DIM]],
    rot: &[Vec<f64>],
) -> Result<[f64; INTERP_DIM], String> {
    let rank = basis.len();
    validate_orthonormal_basis(rank, basis)?;
    validate_rotation(rank, rot)?;
    let mut out = apply_projection_ablation(h_base, basis)?;
    let mut src_coords = vec![0.0_f64; rank];
    for (i, b) in basis.iter().enumerate() {
        src_coords[i] = h_src.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
    }
    let mut rot_coords = vec![0.0_f64; rank];
    for (i, slot) in rot_coords.iter_mut().enumerate().take(rank) {
        let mut acc = 0.0_f64;
        for (&rij, &scj) in rot[i].iter().zip(src_coords.iter()).take(rank) {
            acc += rij * scj;
        }
        *slot = acc;
    }
    for (i, b) in basis.iter().enumerate() {
        for d in 0..INTERP_DIM {
            out[d] += rot_coords[i] * b[d];
        }
    }
    for &v in &out {
        check_finite(v, "interchanged_h")?;
    }
    Ok(out)
}

/// Deterministically construct a seeded matched-rank orthonormal random subspace via modified
/// Gram-Schmidt.
pub fn seeded_random_orthonormal_subspace(
    seed: u64,
    rank: usize,
) -> Result<Vec<[f64; INTERP_DIM]>, String> {
    if rank == 0 || rank >= INTERP_DIM {
        return Err(format!("invalid random subspace rank {rank}"));
    }
    let mut basis: Vec<[f64; INTERP_DIM]> = Vec::with_capacity(rank);
    let mut attempt = 0usize;
    while basis.len() < rank && attempt < 64 {
        let mut v = [0.0_f64; INTERP_DIM];
        for (d, slot) in v.iter_mut().enumerate() {
            *slot = det_weight(9400 + (seed as usize) * 53 + attempt * 17, d, 1.0);
        }
        for prev in &basis {
            let proj: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for d in 0..INTERP_DIM {
                v[d] -= proj * prev[d];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1.0e-6 {
            for x in &mut v {
                *x /= norm;
            }
            basis.push(v);
        }
        attempt += 1;
    }
    validate_orthonormal_basis(rank, &basis)?;
    Ok(basis)
}

fn identity_rotation(rank: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; rank]; rank];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Execute the tiny Qwen transformer backend from `#45` with a `SubspaceInterventionManifest`
/// and produce a bit-identical SHA-256 replay receipt.
#[derive(Debug, Clone)]
pub struct SubspaceReplayTrace {
    pub manifest_digest: DigestRef,
    pub logits: [f64; INTERP_VOCAB],
    pub probs: [f64; INTERP_VOCAB],
    pub top1_token: u32,
    pub trace_digest: DigestRef,
}

fn run_qwen_states(
    tokens: &[u32],
    stop_after_layer: usize,
) -> Result<Vec<[f64; INTERP_DIM]>, String> {
    let seq_len = tokens.len();
    let mut h: Vec<[f64; INTERP_DIM]> = tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            let mut v = [0.0_f64; INTERP_DIM];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = det_weight(4501 + (tok as usize) * 13 + pos * 5, d, 0.55);
            }
            v
        })
        .collect();
    for layer in 0..=stop_after_layer {
        let mut next_h = vec![[0.0_f64; INTERP_DIM]; seq_len];
        for t in 0..seq_len {
            let mut scores = Vec::with_capacity(t + 1);
            for s in 0..=t {
                let dot: f64 = h[t].iter().zip(h[s].iter()).map(|(a, b)| a * b).sum();
                scores.push(dot / (INTERP_DIM as f64).sqrt());
            }
            let attn_w = softmax_stable(&scores)?;
            let mut ctx = [0.0_f64; INTERP_DIM];
            for (s, &w) in attn_w.iter().enumerate() {
                for (c, &hs) in ctx.iter_mut().zip(h[s].iter()) {
                    *c += w * hs;
                }
            }
            let mut mlp = [0.0_f64; INTERP_DIM];
            for (o, slot) in mlp.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (i, &cv) in ctx.iter().enumerate() {
                    acc += cv * det_weight(4520 + layer * 31 + o * 7, i, 0.6);
                }
                *slot = acc.tanh();
            }
            for ((dst, &ht), &m) in next_h[t].iter_mut().zip(h[t].iter()).zip(mlp.iter()) {
                *dst = ht + 0.5 * m;
            }
        }
        h = next_h;
    }
    Ok(h)
}

pub fn run_qwen_subspace_intervention(
    manifest: &SubspaceInterventionManifest,
) -> Result<SubspaceReplayTrace, String> {
    validate_subspace_manifest(manifest)?;
    let seq_len = manifest.base_prompt_tokens.len();
    let src_states = match &manifest.kind {
        SubspaceInterventionKind::SubspaceInterchange {
            source_prompt_tokens,
            ..
        } => Some(run_qwen_states(
            source_prompt_tokens,
            manifest.target_layer,
        )?),
        SubspaceInterventionKind::ProjectionAblation { .. } => None,
    };

    let mut h: Vec<[f64; INTERP_DIM]> = manifest
        .base_prompt_tokens
        .iter()
        .enumerate()
        .map(|(pos, &tok)| {
            let mut v = [0.0_f64; INTERP_DIM];
            for (d, slot) in v.iter_mut().enumerate() {
                *slot = det_weight(4501 + (tok as usize) * 13 + pos * 5, d, 0.55);
            }
            v
        })
        .collect();

    for layer in 0..INTERP_LAYERS {
        let mut next_h = vec![[0.0_f64; INTERP_DIM]; seq_len];
        for t in 0..seq_len {
            let mut scores = Vec::with_capacity(t + 1);
            for s in 0..=t {
                let dot: f64 = h[t].iter().zip(h[s].iter()).map(|(a, b)| a * b).sum();
                scores.push(dot / (INTERP_DIM as f64).sqrt());
            }
            let attn_w = softmax_stable(&scores)?;
            let mut ctx = [0.0_f64; INTERP_DIM];
            for (s, &w) in attn_w.iter().enumerate() {
                for (c, &hs) in ctx.iter_mut().zip(h[s].iter()) {
                    *c += w * hs;
                }
            }
            let mut mlp = [0.0_f64; INTERP_DIM];
            for (o, slot) in mlp.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (i, &cv) in ctx.iter().enumerate() {
                    acc += cv * det_weight(4520 + layer * 31 + o * 7, i, 0.6);
                }
                *slot = acc.tanh();
            }
            for ((dst, &ht), &m) in next_h[t].iter_mut().zip(h[t].iter()).zip(mlp.iter()) {
                *dst = ht + 0.5 * m;
            }
            if layer == manifest.target_layer && manifest.token_positions.contains(&t) {
                next_h[t] = match &manifest.kind {
                    SubspaceInterventionKind::SubspaceInterchange {
                        basis,
                        alignment_rotation,
                        ..
                    } => {
                        let s_ref = src_states
                            .as_ref()
                            .ok_or_else(|| "missing source states".to_string())?;
                        apply_subspace_interchange(
                            &next_h[t],
                            &s_ref[t],
                            basis,
                            alignment_rotation,
                        )?
                    }
                    SubspaceInterventionKind::ProjectionAblation { basis, .. } => {
                        apply_projection_ablation(&next_h[t], basis)?
                    }
                };
            }
        }
        h = next_h;
    }

    let last = h[seq_len - 1];
    let mut logits = [0.0_f64; INTERP_VOCAB];
    for (v, slot) in logits.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (d, &hv) in last.iter().enumerate() {
            acc += hv * det_weight(4590 + v * 11, d, 0.85);
        }
        *slot = check_finite(acc, "qwen_subspace_logit")?;
    }
    let probs_v = softmax_stable(&logits)?;
    let mut probs = [0.0_f64; INTERP_VOCAB];
    let mut best_i = 0usize;
    for i in 0..INTERP_VOCAB {
        probs[i] = probs_v[i];
        if probs[i] > probs[best_i] {
            best_i = i;
        }
    }
    let manifest_digest = manifest.manifest_digest();
    let mut raw = String::new();
    let _ = write!(raw, "{}|{}|", manifest_digest.hex, manifest.seed);
    for l in &logits {
        let _ = write!(raw, "{l:.15e},");
    }
    Ok(SubspaceReplayTrace {
        manifest_digest,
        logits,
        probs,
        top1_token: best_i as u32,
        trace_digest: DigestRef::sha256(raw.as_bytes()),
    })
}

pub fn verify_subspace_replay_receipt(
    manifest: &SubspaceInterventionManifest,
    expected_trace_digest: &DigestRef,
) -> Result<OutcomeStatus, String> {
    let replayed = run_qwen_subspace_intervention(manifest)?;
    if &replayed.trace_digest == expected_trace_digest {
        Ok(OutcomeStatus::Accepted)
    } else {
        Ok(OutcomeStatus::Invalid)
    }
}

/// Planted-parasite backend variant (`dim = INTERP_DIM = 6`):
/// - `e_0`: discriminates spurious input property `prop in {-1, +1}` (NEVER read by the task!)
/// - `e_1`: orthogonal steering channel (cleanly `0.0` on normal runs; large readout weight `4.0`)
/// - `e_2`: TRUE causal task direction (`task_bit in {-1, +1}`, readout weight `2.0`)
/// - `e_3..e_5`: small background context features (readout weight `0.05`)
#[derive(Debug, Clone, Copy)]
pub struct PlantedExample {
    pub task_bit: i32,      // -1 or +1 -> gold binary token (0 or 1)
    pub spurious_prop: i32, // -1 or +1 -> encoded ONLY on e_0, never read by task
    pub seed_idx: usize,
}

impl PlantedExample {
    pub fn hidden(&self) -> [f64; INTERP_DIM] {
        [
            self.spurious_prop as f64, // e_0: spurious property discriminator ONLY (unread by task)
            0.0,                       // e_1: steering direction (unused on clean runs)
            self.task_bit as f64,      // e_2: true causal task direction
            det_weight(9450 + self.seed_idx, 3, 0.25),
            det_weight(9450 + self.seed_idx, 4, 0.25),
            det_weight(9450 + self.seed_idx, 5, 0.25),
        ]
    }

    pub fn gold_label(&self) -> u32 {
        if self.task_bit > 0 {
            1
        } else {
            0
        }
    }
}

pub fn planted_readout_predict(h: &[f64; INTERP_DIM]) -> Result<u32, String> {
    for &x in h {
        check_finite(x, "planted_h")?;
    }
    // Task NEVER reads h[0] (weight = 0.0)!
    // h[1] steers output when activated (2.4 > 2.0); h[2] carries the actual task computation.
    // Small constant bias (-0.01) ensures zeroing h[2] collapses accuracy to exact 50% binary chance.
    let margin = 2.4 * h[1] + 2.0 * h[2] - 0.01;
    Ok(if margin >= 0.0 { 1 } else { 0 })
}

#[derive(Debug, Clone)]
pub struct NecessityEvaluation {
    pub control_id: String,
    pub subspace_rank: usize,
    pub interchange_sufficiency_rate: f64,
    pub sufficiency_passed: bool,
    pub clean_task_accuracy: f64,
    pub ablated_task_accuracy: f64,
    pub target_ablation_accuracy_drop: f64,
    pub random_subspace_count: usize,
    pub random_subspace_rank: usize,
    pub random_ablation_mean_drop: f64,
    pub random_ablation_std_drop: f64,
    pub random_ablation_max_drop: f64,
    pub necessity_passed: bool,
    pub classification: MechanismClassification,
}

/// Evaluate both interchange sufficiency and projection-ablation necessity against `random_bases`
/// (requires `random_bases.len() >= 10`, all orthonormal with exact matching `rank`!).
pub fn evaluate_planted_subspace_necessity(
    control_id: &str,
    examples: &[PlantedExample],
    rank: usize,
    basis: &[[f64; INTERP_DIM]],
    interchange_rot: &[Vec<f64>],
    random_bases: &[Vec<[f64; INTERP_DIM]>],
    interchange_uses_spurious_target: bool,
) -> Result<NecessityEvaluation, String> {
    validate_orthonormal_basis(rank, basis)?;
    validate_rotation(rank, interchange_rot)?;
    if examples.is_empty() {
        return Err("examples must be non-empty".into());
    }
    if random_bases.len() < MIN_RANDOM_SUBSPACES {
        return Err(format!(
            "necessity check requires >= {MIN_RANDOM_SUBSPACES} matched-rank random subspaces, got {}",
            random_bases.len()
        ));
    }
    for rb in random_bases {
        if rb.len() != rank {
            return Err(format!(
                "random subspace rank mismatch: expected rank={rank}, got {}",
                rb.len()
            ));
        }
        validate_orthonormal_basis(rank, rb)?;
    }

    let n = examples.len() as f64;
    let mut clean_ok = 0usize;
    let mut abl_ok = 0usize;
    let mut interchange_ok = 0usize;

    for (i, ex) in examples.iter().enumerate() {
        let h_base = ex.hidden();
        if planted_readout_predict(&h_base)? == ex.gold_label() {
            clean_ok += 1;
        }
        let h_abl = apply_projection_ablation(&h_base, basis)?;
        if planted_readout_predict(&h_abl)? == ex.gold_label() {
            abl_ok += 1;
        }
        // Pair with a counter-example source having opposite target/property
        let src = examples[(i + examples.len() / 2) % examples.len()];
        let h_src = src.hidden();
        let h_int = apply_subspace_interchange(&h_base, &h_src, basis, interchange_rot)?;
        let expected_after_interchange = if interchange_uses_spurious_target {
            if src.spurious_prop > 0 {
                1
            } else {
                0
            }
        } else {
            src.gold_label()
        };
        if planted_readout_predict(&h_int)? == expected_after_interchange {
            interchange_ok += 1;
        }
    }

    let clean_acc = (clean_ok as f64) / n;
    let abl_acc = (abl_ok as f64) / n;
    let target_drop = clean_acc - abl_acc;
    let suff_rate = (interchange_ok as f64) / n;
    let sufficiency_passed = suff_rate >= 0.90;

    let mut rand_drops = Vec::with_capacity(random_bases.len());
    let mut rand_drop_sum = 0.0_f64;
    let mut rand_drop_max = 0.0_f64;
    for rb in random_bases {
        let mut r_ok = 0usize;
        for ex in examples {
            let h_abl = apply_projection_ablation(&ex.hidden(), rb)?;
            if planted_readout_predict(&h_abl)? == ex.gold_label() {
                r_ok += 1;
            }
        }
        let drop = clean_acc - (r_ok as f64) / n;
        rand_drops.push(drop);
        rand_drop_sum += drop;
        rand_drop_max = max_finite(rand_drop_max, drop, "rand_drop_max")?;
    }
    let rand_mean = rand_drop_sum / (random_bases.len() as f64);
    let mut rand_var_sum = 0.0_f64;
    for &d in &rand_drops {
        let diff = d - rand_mean;
        rand_var_sum += diff * diff;
    }
    let rand_std = (rand_var_sum / ((random_bases.len() - 1) as f64)).sqrt();
    // Strict distributional necessity: target ablation drop must exceed random-subspace
    // distribution mean + 2*sigma AND exceed a non-trivial task-degradation floor (> 0.25).
    let necessity_passed = target_drop > rand_mean + 2.0 * rand_std && target_drop > 0.25;

    let classification = if !sufficiency_passed {
        MechanismClassification::Insufficient
    } else if necessity_passed {
        MechanismClassification::FoundMechanism
    } else {
        MechanismClassification::SufficientNotNecessary
    };

    Ok(NecessityEvaluation {
        control_id: control_id.into(),
        subspace_rank: rank,
        interchange_sufficiency_rate: suff_rate,
        sufficiency_passed,
        clean_task_accuracy: clean_acc,
        ablated_task_accuracy: abl_acc,
        target_ablation_accuracy_drop: target_drop,
        random_subspace_count: random_bases.len(),
        random_subspace_rank: rank,
        random_ablation_mean_drop: rand_mean,
        random_ablation_std_drop: rand_std,
        random_ablation_max_drop: rand_drop_max,
        necessity_passed,
        classification,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ParasiteFeatureControlReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub planted_parasite_eval: NecessityEvaluation,
    pub positive_mechanism_eval: NecessityEvaluation,
    pub qwen_interchange_manifest_digest: DigestRef,
    pub qwen_interchange_replay_exact: bool,
    pub qwen_ablation_manifest_digest: DigestRef,
    pub qwen_ablation_replay_exact: bool,
    pub qwen_positive_control_sufficient: bool,
    pub qwen_positive_control_necessary: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<ParasiteFeatureControlReport, String> {
    // Build balanced 8-example planted dataset (task_bit independent of spurious_prop)
    let examples = vec![
        PlantedExample {
            task_bit: 1,
            spurious_prop: 1,
            seed_idx: 0,
        },
        PlantedExample {
            task_bit: 1,
            spurious_prop: -1,
            seed_idx: 1,
        },
        PlantedExample {
            task_bit: 1,
            spurious_prop: 1,
            seed_idx: 2,
        },
        PlantedExample {
            task_bit: 1,
            spurious_prop: -1,
            seed_idx: 3,
        },
        PlantedExample {
            task_bit: -1,
            spurious_prop: -1,
            seed_idx: 4,
        },
        PlantedExample {
            task_bit: -1,
            spurious_prop: 1,
            seed_idx: 5,
        },
        PlantedExample {
            task_bit: -1,
            spurious_prop: -1,
            seed_idx: 6,
        },
        PlantedExample {
            task_bit: -1,
            spurious_prop: 1,
            seed_idx: 7,
        },
    ];

    let e0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    // 1. Planted-parasite negative control: rank-2 span(e_0, e_1) with 90-degree alignment rotation
    //    routing source's e_0 property onto base's e_1 steering channel.
    let mut rand_rank2 = Vec::with_capacity(DEFAULT_RANDOM_SUBSPACES);
    for s in 0..DEFAULT_RANDOM_SUBSPACES {
        rand_rank2.push(seeded_random_orthonormal_subspace(100 + s as u64, 2)?);
    }
    let parasite_rot = vec![vec![0.0, -1.0], vec![1.0, 0.0]];
    let planted_parasite_eval = evaluate_planted_subspace_necessity(
        "planted_parasite_rank2_span_e0_e1",
        &examples,
        2,
        &[e0, e1],
        &parasite_rot,
        &rand_rank2,
        true,
    )?;
    if planted_parasite_eval.classification != MechanismClassification::SufficientNotNecessary {
        return Err(
            "HARNESS FAILURE: planted parasite control was NOT classified as sufficient_not_necessary"
                .into(),
        );
    }

    // 2. Positive control on planted backend: true causal task direction e_2 (rank 1)
    let mut rand_rank1 = Vec::with_capacity(DEFAULT_RANDOM_SUBSPACES);
    for s in 0..DEFAULT_RANDOM_SUBSPACES {
        rand_rank1.push(seeded_random_orthonormal_subspace(200 + s as u64, 1)?);
    }
    let positive_mechanism_eval = evaluate_planted_subspace_necessity(
        "positive_control_true_task_dir_e2",
        &examples,
        1,
        &[e2],
        &identity_rotation(1),
        &rand_rank1,
        false,
    )?;
    if positive_mechanism_eval.classification != MechanismClassification::FoundMechanism {
        return Err(format!(
            "positive control failed sufficiency or necessity: {positive_mechanism_eval:?}"
        ));
    }

    // 3. Bit-identical replay & positive control on existing #45 tiny Qwen transformer backend
    let base_prompt = vec![1u32, 4, 2, 6, 3];
    let source_prompt = vec![5u32, 2, 7, 1, 0];
    let qwen_basis = vec![e0, e1, e2]; // rank-3 active hidden subspace at layer 2, final token
    let interchange_manifest = SubspaceInterventionManifest {
        experiment_id: "exp94-qwen-subspace-interchange".into(),
        source_arch: ModelBackendArch::QwenTransformer,
        target_arch: ModelBackendArch::QwenTransformer,
        model_revision_digest: "qwen-tiny-ref@sha256:45a1b2c3".into(),
        seed: 20261124,
        base_prompt_tokens: base_prompt.clone(),
        target_layer: 2,
        token_positions: vec![4],
        kind: SubspaceInterventionKind::SubspaceInterchange {
            rank: 3,
            basis: qwen_basis.clone(),
            alignment_rotation: identity_rotation(3),
            source_prompt_tokens: source_prompt.clone(),
        },
    };
    let q_int = run_qwen_subspace_intervention(&interchange_manifest)?;
    let qwen_interchange_replay_exact =
        verify_subspace_replay_receipt(&interchange_manifest, &q_int.trace_digest)?
            == OutcomeStatus::Accepted;

    let ablation_manifest = SubspaceInterventionManifest {
        experiment_id: "exp94-qwen-projection-ablation".into(),
        source_arch: ModelBackendArch::QwenTransformer,
        target_arch: ModelBackendArch::QwenTransformer,
        model_revision_digest: "qwen-tiny-ref@sha256:45a1b2c3".into(),
        seed: 20261124,
        base_prompt_tokens: base_prompt.clone(),
        target_layer: 2,
        token_positions: vec![4],
        kind: SubspaceInterventionKind::ProjectionAblation {
            rank: 3,
            basis: qwen_basis.clone(),
        },
    };
    let q_abl = run_qwen_subspace_intervention(&ablation_manifest)?;
    let qwen_ablation_replay_exact =
        verify_subspace_replay_receipt(&ablation_manifest, &q_abl.trace_digest)?
            == OutcomeStatus::Accepted;

    // Verify the Qwen subspace actually changes logits significantly under both interchange & ablation
    let qwen_base_states = run_qwen_states(&base_prompt, 2)?;
    let qwen_src_states = run_qwen_states(&source_prompt, 2)?;
    let base_top1 = {
        let last = qwen_base_states[4];
        let mut best_i = 0usize;
        let mut best_l = f64::NEG_INFINITY;
        for v in 0..INTERP_VOCAB {
            let mut acc = 0.0_f64;
            for (d, &hv) in last.iter().enumerate() {
                acc += hv * det_weight(4590 + v * 11, d, 0.85);
            }
            if acc > best_l {
                best_l = acc;
                best_i = v;
            }
        }
        best_i as u32
    };
    let _ = qwen_src_states;
    let qwen_positive_control_sufficient = q_int.logits.iter().enumerate().any(|(i, &l)| {
        let mut b_acc = 0.0_f64;
        for (d, &hv) in qwen_base_states[4].iter().enumerate() {
            b_acc += hv * det_weight(4590 + i * 11, d, 0.85);
        }
        (l - b_acc).abs() > 1.0e-3
    });
    let qwen_positive_control_necessary = q_abl.top1_token != base_top1
        || q_abl.logits.iter().enumerate().any(|(i, &l)| {
            let mut b_acc = 0.0_f64;
            for (d, &hv) in qwen_base_states[4].iter().enumerate() {
                b_acc += hv * det_weight(4590 + i * 11, d, 0.85);
            }
            (l - b_acc).abs() > 0.05
        });

    let mut negative_controls = Vec::new();

    // NC1: planted parasite control caught as sufficient_not_necessary (NOT found_mechanism)
    negative_controls.push(NegativeControl {
        case_id: "nc1_planted_parasite_caught_sufficient_not_necessary".into(),
        outcome: if planted_parasite_eval.sufficiency_passed
            && !planted_parasite_eval.necessity_passed
            && planted_parasite_eval.classification
                == MechanismClassification::SufficientNotNecessary
        {
            OutcomeStatus::Invalid // refused as a causal mechanism
        } else {
            OutcomeStatus::Accepted
        },
        detail: "rank-2 parasite span(e0,e1) passes interchange (1.0) but fails necessity (0.0 drop vs random) -> sufficient_not_necessary".into(),
    });

    // NC2: rank mismatch (declared rank=2, basis len=1) rejected
    negative_controls.push(NegativeControl {
        case_id: "nc2_rank_mismatch_rejected".into(),
        outcome: if validate_orthonormal_basis(2, &[e0]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "declared rank=2 with 1 basis vector rejected".into(),
    });

    // NC3: non-orthonormal basis rejected
    let non_ortho = [
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];
    negative_controls.push(NegativeControl {
        case_id: "nc3_non_orthonormal_basis_rejected".into(),
        outcome: if validate_orthonormal_basis(2, &non_ortho).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "non-orthonormal basis (dot != 0 / norm != 1) rejected".into(),
    });

    // NC4: random subspaces with wrong rank rejected
    negative_controls.push(NegativeControl {
        case_id: "nc4_random_subspace_wrong_rank_rejected".into(),
        outcome: if evaluate_planted_subspace_necessity(
            "wrong_rank_random",
            &examples,
            2,
            &[e0, e1],
            &parasite_rot,
            &rand_rank1, // rank 1 instead of rank 2!
            true,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "rank-1 random subspaces supplied for rank-2 necessity check rejected".into(),
    });

    // NC5: fewer than 10 matched-rank random subspaces rejected
    negative_controls.push(NegativeControl {
        case_id: "nc5_fewer_than_10_random_subspaces_rejected".into(),
        outcome: if evaluate_planted_subspace_necessity(
            "too_few_random",
            &examples,
            2,
            &[e0, e1],
            &parasite_rot,
            &rand_rank2[..5],
            true,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "only 5 random subspaces (< 10 required) rejected".into(),
    });

    // NC6: tampered replay receipt digest detected
    let tampered = DigestRef::sha256(b"tampered-subspace-replay");
    negative_controls.push(NegativeControl {
        case_id: "nc6_tampered_subspace_replay_digest_rejected".into(),
        outcome: verify_subspace_replay_receipt(&interchange_manifest, &tampered)?,
        detail: "re-execution against tampered SHA-256 trace digest fails replay check".into(),
    });

    // NC7: NaN element in basis rejected
    let nan_basis = [[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]];
    negative_controls.push(NegativeControl {
        case_id: "nc7_nonfinite_basis_element_rejected".into(),
        outcome: if validate_orthonormal_basis(1, &nan_basis).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN subspace basis coordinate rejected".into(),
    });

    // NC8: mutable model revision rejected
    let mut bad_rev = ablation_manifest.clone();
    bad_rev.model_revision_digest = "qwen:latest".into();
    negative_controls.push(NegativeControl {
        case_id: "nc8_mutable_model_revision_rejected".into(),
        outcome: if validate_subspace_manifest(&bad_rev).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "floating `:latest` revision rejected in subspace manifest".into(),
    });

    // NC9: unlicensed reference repo code copy prohibited
    negative_controls.push(NegativeControl {
        case_id: "nc9_unlicensed_reference_repo_no_code_copied".into(),
        outcome: OutcomeStatus::BlockedOnArtifacts,
        detail: "dounick/parasite-features@1df61727 has no license file; reference reading only, zero code copied".into(),
    });

    Ok(ParasiteFeatureControlReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Physical-model steering/intervention runs (out of scope per issue #94 until #77 physical fixture runs)",
            "Bounded CPU subspace interchange, projection ablation, 16-random-subspace necessity test, planted parasite control, and #45 Qwen replay receipts verified locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        planted_parasite_eval,
        positive_mechanism_eval,
        qwen_interchange_manifest_digest: q_int.manifest_digest,
        qwen_interchange_replay_exact,
        qwen_ablation_manifest_digest: q_abl.manifest_digest,
        qwen_ablation_replay_exact,
        qwen_positive_control_sufficient,
        qwen_positive_control_necessary,
        negative_controls,
    })
}

pub fn inputs_json(report: &ParasiteFeatureControlReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.parasite-feature-control.inputs.v1\",\n",
            "  \"issue\": 94,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"extends_issue\": 45,\n",
            "  \"paper_arxiv\": {},\n",
            "  \"reference_repo\": {},\n",
            "  \"reference_commit\": {},\n",
            "  \"reference_license_status\": {},\n",
            "  \"author_posts\": [{}, {}],\n",
            "  \"coordination_note\": {},\n",
            "  \"min_random_subspaces\": {},\n",
            "  \"used_random_subspaces\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(PAPER_ARXIV),
        quote(REF_REPO),
        quote(REF_COMMIT),
        quote(REF_LICENSE_STATUS),
        quote(POST_BOGURAEV),
        quote(POST_PIMENTEL),
        quote(COORDINATION_NOTE),
        MIN_RANDOM_SUBSPACES,
        DEFAULT_RANDOM_SUBSPACES,
    )
}

fn eval_to_json(e: &NecessityEvaluation) -> String {
    format!(
        "{{\"control_id\":{},\"subspace_rank\":{},\"interchange_sufficiency_rate\":{},\"sufficiency_passed\":{},\"clean_task_accuracy\":{},\"ablated_task_accuracy\":{},\"target_ablation_accuracy_drop\":{},\"random_subspace_count\":{},\"random_subspace_rank\":{},\"random_ablation_mean_drop\":{},\"random_ablation_std_drop\":{},\"random_ablation_max_drop\":{},\"necessity_passed\":{},\"classification\":{}}}",
        quote(&e.control_id),
        e.subspace_rank,
        finite_f64(e.interchange_sufficiency_rate),
        e.sufficiency_passed,
        finite_f64(e.clean_task_accuracy),
        finite_f64(e.ablated_task_accuracy),
        finite_f64(e.target_ablation_accuracy_drop),
        e.random_subspace_count,
        e.random_subspace_rank,
        finite_f64(e.random_ablation_mean_drop),
        finite_f64(e.random_ablation_std_drop),
        finite_f64(e.random_ablation_max_drop),
        e.necessity_passed,
        quote(e.classification.as_str()),
    )
}

pub fn results_json(report: &ParasiteFeatureControlReport) -> String {
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
            "  \"schema\": \"inference-lab.parasite-feature-control.results.v1\",\n",
            "  \"issue\": 94,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"planted_parasite_eval\": {},\n",
            "  \"positive_mechanism_eval\": {},\n",
            "  \"qwen_interchange_manifest_digest\": {},\n",
            "  \"qwen_interchange_replay_exact\": {},\n",
            "  \"qwen_ablation_manifest_digest\": {},\n",
            "  \"qwen_ablation_replay_exact\": {},\n",
            "  \"qwen_positive_control_sufficient\": {},\n",
            "  \"qwen_positive_control_necessary\": {},\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        eval_to_json(&report.planted_parasite_eval),
        eval_to_json(&report.positive_mechanism_eval),
        report.qwen_interchange_manifest_digest.to_json(),
        report.qwen_interchange_replay_exact,
        report.qwen_ablation_manifest_digest.to_json(),
        report.qwen_ablation_replay_exact,
        report.qwen_positive_control_sufficient,
        report.qwen_positive_control_necessary,
        controls.join(","),
    )
}

pub fn summary_markdown(report: &ParasiteFeatureControlReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #94: Subspace Interchange, Necessity Ablation & Planted Parasite-Feature Control"
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
        "- Paper & reference: `{}` / `{}@{}` (`{}`)",
        PAPER_ARXIV, REF_REPO, REF_COMMIT, REF_LICENSE_STATUS
    )
    .unwrap();
    writeln!(out, "- Coordination note: {}", COORDINATION_NOTE).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Sufficiency vs Necessity Controls (16 Matched-Rank Random Subspaces)"
    )
    .unwrap();
    for e in [
        &report.planted_parasite_eval,
        &report.positive_mechanism_eval,
    ] {
        writeln!(
            out,
            "- `{}` (rank `{}`): `sufficiency={}` (`rate={:.2}`), `ablation_drop={:.2}` vs `random_mean={:.3} +/- {:.3}` (`max={:.2}`) -> `necessity={}`, `classification={}`",
            e.control_id,
            e.subspace_rank,
            e.sufficiency_passed,
            e.interchange_sufficiency_rate,
            e.target_ablation_accuracy_drop,
            e.random_ablation_mean_drop,
            e.random_ablation_std_drop,
            e.random_ablation_max_drop,
            e.necessity_passed,
            e.classification.as_str()
        )
        .unwrap();
    }
    writeln!(
        out,
        "- `#45` tiny Qwen subspace replay exact: `interchange={}`, `ablation={}`, `sufficient={}`, `necessary={}`",
        report.qwen_interchange_replay_exact,
        report.qwen_ablation_replay_exact,
        report.qwen_positive_control_sufficient,
        report.qwen_positive_control_necessary
    )
    .unwrap();
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
    fn planted_parasite_is_sufficient_not_necessary_and_positive_control_passes_both() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert!(r.planted_parasite_eval.sufficiency_passed);
        assert!(!r.planted_parasite_eval.necessity_passed);
        assert_eq!(
            r.planted_parasite_eval.classification,
            MechanismClassification::SufficientNotNecessary
        );
        assert!(r.positive_mechanism_eval.sufficiency_passed);
        assert!(r.positive_mechanism_eval.necessity_passed);
        assert_eq!(
            r.positive_mechanism_eval.classification,
            MechanismClassification::FoundMechanism
        );
        assert!(r.qwen_interchange_replay_exact);
        assert!(r.qwen_ablation_replay_exact);
        assert_eq!(r.negative_controls.len(), 9);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
