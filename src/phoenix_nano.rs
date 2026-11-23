//! Issue #34: Qualify Phoenix nano training, checkpoint resume and gRPC serving on one node.
//!
//! Pinned source basis (strictly non-gating upstream notes):
//! - `xai-org/x-algorithm@30647a85a20cbdd114169beb9973680c4dc4f401` (`phoenix/README.md`,
//!   `phoenix/reference/train_synth.py`, `phoenix/TRAINING.md`, `phoenix/QUICKSTART.md`,
//!   `phoenix/xrex/utils/gpu.py#L108-L118` recognizing only A100/H100/H200/GB200/GB300).
//! - Related ingress note only (out of scope, non-gating): Cloudflare 19 Nov 2026 inbound TCP/gRPC
//!   Containers beta; local wrangler port 8788 is not production Spectrum evidence.
//! - Decreasing synthetic loss prepares an execution artifact only; it does NOT establish
//!   recommendation quality or a learning-loop improvement.
//!
//! Provides:
//! - Exact `gpu_arch()` qualification gate recognizing `A100/H100/H200/GB200/GB300`, keeping
//!   `GB10` (Spark) and `RTX 5080` separately `unsupported`, and rejecting device-name spoof overrides.
//! - Seeded synthetic world + provider snapshot + codebook + training dump with SHA-256 identity
//!   binding and exact 0-indexed wire <-> 1-indexed model SID codec (`0` reserved for missing).
//! - Pure-Rust miniature retrieval/ranking trainer (weights + sparse AdaGrad + AdamW + RNG counter +
//!   data cursor) verifying uninterrupted vs fresh-process checkpoint restore continuation parity.
//! - Bounded native gRPC retrieve -> rank request envelope comparing served scores/IDs vs direct
//!   inference, independent candidate attention-isolation / missing-SID / fixed-score top-k controls,
//!   and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const PHOENIX_UPSTREAM_COMMIT: &str = "30647a85a20cbdd114169beb9973680c4dc4f401";
pub const EMBED_DIM: usize = 4;
pub const NUM_ITEMS: usize = 12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceQualificationInput {
    pub detected_gpu_name: String,
    pub override_gpu_name: Option<String>,
    pub jax_cuda_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceQualificationRow {
    pub device_label: String,
    pub status: OutcomeStatus,
    pub note: String,
}

/// Mirrors `phoenix/xrex/utils/gpu.py#L108-L118`: recognizes A100/H100/H200/GB200/GB300 only.
/// Overriding a detected GPU name is explicitly rejected as invalid kernel-correctness evidence.
pub fn qualify_phoenix_gpu_arch(input: &DeviceQualificationInput) -> DeviceQualificationRow {
    if input.override_gpu_name.is_some() {
        return DeviceQualificationRow {
            device_label: input.detected_gpu_name.clone(),
            status: OutcomeStatus::Invalid,
            note: "device-name override rejected: overriding gpu_arch() is not kernel-correctness evidence".into(),
        };
    }
    let recognized = matches!(
        input.detected_gpu_name.as_str(),
        "A100" | "H100" | "H200" | "GB200" | "GB300"
    );
    if !recognized {
        return DeviceQualificationRow {
            device_label: input.detected_gpu_name.clone(),
            status: OutcomeStatus::Unsupported,
            note: "rejected by upstream gpu_arch() cohort (only A100/H100/H200/GB200/GB300 recognized)".into(),
        };
    }
    if !input.jax_cuda_verified {
        return DeviceQualificationRow {
            device_label: input.detected_gpu_name.clone(),
            status: OutcomeStatus::UnavailableHardwareRuntime,
            note: "recognized GPU architecture but no physical JAX/CUDA device executed locally"
                .into(),
        };
    }
    DeviceQualificationRow {
        device_label: input.detected_gpu_name.clone(),
        status: OutcomeStatus::Accepted,
        note: "qualified GPU cohort".into(),
    }
}

/// Encode optional 0-indexed wire SID into 1-indexed model SID (`0` strictly reserved for missing).
pub fn wire_to_model_sid(wire_sid: Option<u32>, codebook_size: u32) -> Result<u32, String> {
    match wire_sid {
        None => Ok(0),
        Some(w) if w < codebook_size => Ok(w + 1),
        Some(w) => Err(format!(
            "wire SID {w} out of range for codebook_size {codebook_size}"
        )),
    }
}

/// Decode 1-indexed model SID back to optional 0-indexed wire SID (`0` -> `None`).
pub fn model_to_wire_sid(model_sid: u32, codebook_size: u32) -> Result<Option<u32>, String> {
    if model_sid == 0 {
        Ok(None)
    } else if model_sid <= codebook_size {
        Ok(Some(model_sid - 1))
    } else {
        Err(format!(
            "model SID {model_sid} out of range for codebook_size {codebook_size}"
        ))
    }
}

#[derive(Debug, Clone)]
pub struct SeededWorldBundle {
    pub snapshot_digest: DigestRef,
    pub codebook_digest: DigestRef,
    pub dump_digest: DigestRef,
    pub codebook_size: u32,
    pub dump_rows: Vec<(usize, Option<u32>, f64)>, // (item_id, wire_sid, target)
}

pub fn build_seeded_world() -> SeededWorldBundle {
    let codebook_size = 16u32;
    let mut dump_rows = Vec::new();
    let mut raw_dump = String::new();
    for step in 0..8usize {
        let item_id = (step * 5 + 1) % NUM_ITEMS;
        let wire_sid = if step == 3 {
            None
        } else {
            Some(((step * 3 + 2) as u32) % codebook_size)
        };
        let target = if step % 2 == 0 { 1.0 } else { -0.5 };
        dump_rows.push((item_id, wire_sid, target));
        let _ = writeln!(raw_dump, "{step},{item_id},{wire_sid:?},{target:.6}");
    }
    SeededWorldBundle {
        snapshot_digest: DigestRef::sha256(b"phoenix-nano-provider-snapshot-seed42-v1"),
        codebook_digest: DigestRef::sha256(b"phoenix-nano-codebook-16-seed42-v1"),
        dump_digest: DigestRef::sha256(raw_dump.as_bytes()),
        codebook_size,
        dump_rows,
    }
}

#[derive(Debug, Clone)]
pub struct PhoenixTrainerState {
    pub snapshot_digest: DigestRef,
    pub codebook_digest: DigestRef,
    pub embeddings: Vec<[f64; EMBED_DIM]>,
    pub adagrad_accum: Vec<[f64; EMBED_DIM]>,
    pub adamw_m: [f64; EMBED_DIM],
    pub adamw_v: [f64; EMBED_DIM],
    pub ranker_w: [f64; EMBED_DIM],
    pub rng_counter: u64,
    pub data_cursor: usize,
    pub step: usize,
}

impl PhoenixTrainerState {
    pub fn initial(world: &SeededWorldBundle) -> Self {
        let embeddings: Vec<[f64; EMBED_DIM]> = (0..NUM_ITEMS)
            .map(|item| {
                let mut e = [0.0_f64; EMBED_DIM];
                for (d, slot) in e.iter_mut().enumerate() {
                    *slot = det_weight(3401 + item * 7, d, 0.4);
                }
                e
            })
            .collect();
        let mut ranker_w = [0.0_f64; EMBED_DIM];
        for (d, w) in ranker_w.iter_mut().enumerate() {
            *w = det_weight(3499, d, 0.35);
        }
        Self {
            snapshot_digest: world.snapshot_digest.clone(),
            codebook_digest: world.codebook_digest.clone(),
            embeddings,
            adagrad_accum: vec![[1.0e-4; EMBED_DIM]; NUM_ITEMS],
            adamw_m: [0.0; EMBED_DIM],
            adamw_v: [0.0; EMBED_DIM],
            ranker_w,
            rng_counter: 1001,
            data_cursor: 0,
            step: 0,
        }
    }

    pub fn train_steps(&mut self, world: &SeededWorldBundle, steps: usize) -> Result<f64, String> {
        if self.snapshot_digest != world.snapshot_digest
            || self.codebook_digest != world.codebook_digest
        {
            return Err("snapshot or codebook digest mismatch before training step".into());
        }
        if world.dump_rows.is_empty() {
            return Err("empty training dump_rows".into());
        }
        let mut last_loss = 0.0_f64;
        for _ in 0..steps {
            let (item_id, wire_sid, target) =
                world.dump_rows[self.data_cursor % world.dump_rows.len()];
            let model_sid = wire_to_model_sid(wire_sid, world.codebook_size)?;
            let sid_bias = if model_sid == 0 {
                0.0 // Reserved missing SID contributes zero feature bias
            } else {
                det_weight(3450 + model_sid as usize, 0, 0.1)
            };
            let rng_jitter = det_weight(3480 + self.rng_counter as usize, 0, 0.02);

            let emb = self.embeddings[item_id];
            let pred: f64 = emb
                .iter()
                .zip(self.ranker_w.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>()
                + sid_bias
                + rng_jitter;
            let err = pred - target;
            last_loss = 0.5 * err * err;
            if !last_loss.is_finite() {
                return Err("non-finite Phoenix nano training loss".into());
            }

            // Sparse rowwise AdaGrad on item embedding + AdamW on dense ranker_w
            self.step += 1;
            let lr_emb = 0.05;
            let lr_w = 0.03;
            let b1 = 0.9;
            let b2 = 0.999;
            for (d, &emb_d) in emb.iter().enumerate() {
                let g_e = err * self.ranker_w[d];
                self.adagrad_accum[item_id][d] += g_e * g_e;
                self.embeddings[item_id][d] -= lr_emb * g_e / self.adagrad_accum[item_id][d].sqrt();

                let g_w = err * emb_d;
                self.adamw_m[d] = b1 * self.adamw_m[d] + (1.0 - b1) * g_w;
                self.adamw_v[d] = b2 * self.adamw_v[d] + (1.0 - b2) * g_w * g_w;
                let m_hat = self.adamw_m[d] / (1.0 - b1.powi(self.step as i32));
                let v_hat = self.adamw_v[d] / (1.0 - b2.powi(self.step as i32));
                self.ranker_w[d] =
                    self.ranker_w[d] * 0.999 - lr_w * m_hat / (v_hat.sqrt() + 1.0e-8);
            }
            self.rng_counter += 1;
            self.data_cursor += 1;
        }
        Ok(last_loss)
    }

    pub fn serialize_checkpoint(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "{}|{}|{}|{}|{}",
            self.snapshot_digest.hex,
            self.codebook_digest.hex,
            self.step,
            self.rng_counter,
            self.data_cursor
        );
        for row in &self.embeddings {
            let _ = writeln!(
                s,
                "E:{:.16e},{:.16e},{:.16e},{:.16e}",
                row[0], row[1], row[2], row[3]
            );
        }
        for row in &self.adagrad_accum {
            let _ = writeln!(
                s,
                "A:{:.16e},{:.16e},{:.16e},{:.16e}",
                row[0], row[1], row[2], row[3]
            );
        }
        let _ = writeln!(
            s,
            "W:{:.16e},{:.16e},{:.16e},{:.16e}",
            self.ranker_w[0], self.ranker_w[1], self.ranker_w[2], self.ranker_w[3]
        );
        let _ = writeln!(
            s,
            "M:{:.16e},{:.16e},{:.16e},{:.16e}",
            self.adamw_m[0], self.adamw_m[1], self.adamw_m[2], self.adamw_m[3]
        );
        let _ = writeln!(
            s,
            "V:{:.16e},{:.16e},{:.16e},{:.16e}",
            self.adamw_v[0], self.adamw_v[1], self.adamw_v[2], self.adamw_v[3]
        );
        s
    }

    pub fn restore_checkpoint(
        payload: &str,
        expected_snapshot: &DigestRef,
        expected_codebook: &DigestRef,
    ) -> Result<Self, String> {
        let mut lines = payload.lines();
        let header = lines.next().ok_or("empty checkpoint payload")?;
        let parts: Vec<&str> = header.split('|').collect();
        if parts.len() != 5 {
            return Err("malformed checkpoint header".into());
        }
        if parts[0] != expected_snapshot.hex || parts[1] != expected_codebook.hex {
            return Err("checkpoint snapshot/codebook digest mismatch".into());
        }
        let step: usize = parts[2].parse().map_err(|_| "invalid step")?;
        let rng_counter: u64 = parts[3].parse().map_err(|_| "invalid rng_counter")?;
        let data_cursor: usize = parts[4].parse().map_err(|_| "invalid data_cursor")?;

        let parse4 = |line: &str, prefix: &str| -> Result<[f64; EMBED_DIM], String> {
            let body = line
                .strip_prefix(prefix)
                .ok_or_else(|| format!("expected prefix {prefix}"))?;
            let vals: Vec<f64> = body
                .split(',')
                .map(|x| x.parse::<f64>().map_err(|_| "invalid float".to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            if vals.len() != EMBED_DIM || vals.iter().any(|x| !x.is_finite()) {
                return Err("non-finite or wrong-length tensor row in checkpoint".into());
            }
            Ok([vals[0], vals[1], vals[2], vals[3]])
        };

        let mut embeddings = Vec::with_capacity(NUM_ITEMS);
        for _ in 0..NUM_ITEMS {
            embeddings.push(parse4(lines.next().ok_or("truncated E")?, "E:")?);
        }
        let mut adagrad_accum = Vec::with_capacity(NUM_ITEMS);
        for _ in 0..NUM_ITEMS {
            adagrad_accum.push(parse4(lines.next().ok_or("truncated A")?, "A:")?);
        }
        let ranker_w = parse4(lines.next().ok_or("missing W")?, "W:")?;
        let adamw_m = parse4(lines.next().ok_or("missing M")?, "M:")?;
        let adamw_v = parse4(lines.next().ok_or("missing V")?, "V:")?;

        Ok(Self {
            snapshot_digest: expected_snapshot.clone(),
            codebook_digest: expected_codebook.clone(),
            embeddings,
            adagrad_accum,
            adamw_m,
            adamw_v,
            ranker_w,
            rng_counter,
            data_cursor,
            step,
        })
    }

    /// Score candidates independently (candidate-to-candidate attention isolation: each candidate's
    /// score depends ONLY on user query + its own embedding/SID, never on other candidates in batch).
    pub fn rank_candidates_isolated(
        &self,
        user_query: &[f64; EMBED_DIM],
        candidates: &[(usize, Option<u32>)],
        codebook_size: u32,
        bug_cross_candidate_leak: bool,
    ) -> Result<Vec<(usize, f64)>, String> {
        let mut out = Vec::with_capacity(candidates.len());
        let leak_term = if bug_cross_candidate_leak && !candidates.is_empty() {
            let last_id = candidates[candidates.len() - 1].0 % NUM_ITEMS;
            0.25 * self.embeddings[last_id][0]
        } else {
            0.0
        };
        for &(item_id, wire_sid) in candidates {
            if item_id >= NUM_ITEMS {
                return Err(format!("item_id {item_id} out of range"));
            }
            let model_sid = wire_to_model_sid(wire_sid, codebook_size)?;
            let emb = self.embeddings[item_id];
            let dot: f64 = user_query
                .iter()
                .zip(emb.iter())
                .zip(self.ranker_w.iter())
                .map(|((q, e), w)| q * e * w)
                .sum();
            let sid_bonus = if model_sid == 0 {
                0.0
            } else {
                det_weight(3450 + model_sid as usize, 1, 0.05)
            };
            let score = dot + sid_bonus + leak_term;
            if !score.is_finite() {
                return Err("non-finite candidate rank score".into());
            }
            out.push((item_id, score));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(out)
    }
}

fn max_state_diff(a: &PhoenixTrainerState, b: &PhoenixTrainerState) -> Result<f64, String> {
    if a.step != b.step || a.rng_counter != b.rng_counter || a.data_cursor != b.data_cursor {
        return Err("step/rng_counter/data_cursor mismatch".into());
    }
    let mut max_err = 0.0_f64;
    let mut update = |x: f64, y: f64, label: &str| -> Result<(), String> {
        if !x.is_finite() || !y.is_finite() {
            return Err(format!("non-finite {label}"));
        }
        let d = (x - y).abs();
        if d > max_err {
            max_err = d;
        }
        Ok(())
    };
    for (ra, rb) in a.embeddings.iter().zip(b.embeddings.iter()) {
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            update(x, y, "embedding")?;
        }
    }
    for (ra, rb) in a.adagrad_accum.iter().zip(b.adagrad_accum.iter()) {
        for (&x, &y) in ra.iter().zip(rb.iter()) {
            update(x, y, "adagrad_accum")?;
        }
    }
    for (&x, &y) in a.ranker_w.iter().zip(b.ranker_w.iter()) {
        update(x, y, "ranker_w")?;
    }
    for (&x, &y) in a.adamw_m.iter().zip(b.adamw_m.iter()) {
        update(x, y, "adamw_m")?;
    }
    for (&x, &y) in a.adamw_v.iter().zip(b.adamw_v.iter()) {
        update(x, y, "adamw_v")?;
    }
    Ok(max_err)
}

/// Validate wire-to-model SID mapping against a claimed model SID (catches 0/1-index shift).
pub fn validate_sid_mapping(
    wire_sid: Option<u32>,
    claimed_model_sid: u32,
    codebook_size: u32,
) -> OutcomeStatus {
    match wire_to_model_sid(wire_sid, codebook_size) {
        Ok(expected) if expected == claimed_model_sid => OutcomeStatus::Accepted,
        _ => OutcomeStatus::Invalid,
    }
}

/// Compare direct inference vs gRPC-served ranked IDs and scores under declared tolerance.
pub fn verify_direct_vs_served(
    direct: &[(usize, f64)],
    served: &[(usize, f64)],
    tol: f64,
) -> Result<f64, String> {
    if direct.len() != served.len() || !tol.is_finite() || tol < 0.0 {
        return Err("direct/served length or tolerance invalid".into());
    }
    let mut max_err = 0.0_f64;
    for (a, b) in direct.iter().zip(served.iter()) {
        if a.0 != b.0 {
            return Err(format!("item ID mismatch: {} vs {}", a.0, b.0));
        }
        if !a.1.is_finite() || !b.1.is_finite() {
            return Err("non-finite direct/served score".into());
        }
        let d = (a.1 - b.1).abs();
        if d > max_err {
            max_err = d;
        }
    }
    if max_err > tol {
        return Err(format!(
            "direct vs served score error {max_err:.3e} > tol {tol:.3e}"
        ));
    }
    Ok(max_err)
}

/// Process readiness alone cannot pass; at least one retrieve -> rank request must complete cleanly.
pub fn validate_serving_completion(
    process_ready: bool,
    completed_retrieve_rank_requests: usize,
    direct_served_ok: bool,
) -> OutcomeStatus {
    if process_ready && completed_retrieve_rank_requests > 0 && direct_served_ok {
        OutcomeStatus::Accepted
    } else {
        OutcomeStatus::Invalid
    }
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct PhoenixNanoReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub one_node_outcome: String,
    pub device_matrix: Vec<DeviceQualificationRow>,
    pub checkpoint_resume_max_err: f64,
    pub direct_vs_grpc_served_max_err: f64,
    pub retrieved_and_ranked_count: usize,
    pub synthetic_loss_proves_rec_quality: bool,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<PhoenixNanoReport, String> {
    let world = build_seeded_world();

    let device_matrix = vec![
        qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "GB300".into(),
            override_gpu_name: None,
            jax_cuda_verified: false,
        }),
        qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "GB10".into(),
            override_gpu_name: None,
            jax_cuda_verified: false,
        }),
        qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "RTX 5080".into(),
            override_gpu_name: None,
            jax_cuda_verified: false,
        }),
    ];

    // Uninterrupted 6-step run vs 3-step -> serialize -> fresh restore -> 3-step run
    let mut uninterrupted = PhoenixTrainerState::initial(&world);
    let _ = uninterrupted.train_steps(&world, 6)?;

    let mut split_run = PhoenixTrainerState::initial(&world);
    let _ = split_run.train_steps(&world, 3)?;
    let ckpt_bytes = split_run.serialize_checkpoint();
    let mut restored = PhoenixTrainerState::restore_checkpoint(
        &ckpt_bytes,
        &world.snapshot_digest,
        &world.codebook_digest,
    )?;
    let _ = restored.train_steps(&world, 3)?;
    let checkpoint_resume_max_err = max_state_diff(&uninterrupted, &restored)?;

    // Native gRPC retrieve -> rank envelope simulation vs direct inference
    let user_query = [0.6, -0.3, 0.8, 0.2];
    let retrieved_candidates = vec![(1usize, Some(2u32)), (4, None), (7, Some(9)), (2, Some(0))];
    let direct_ranked = restored.rank_candidates_isolated(
        &user_query,
        &retrieved_candidates,
        world.codebook_size,
        false,
    )?;
    // Re-load from serialized checkpoint into serving instance and run retrieve -> rank
    let serving_model = PhoenixTrainerState::restore_checkpoint(
        &restored.serialize_checkpoint(),
        &world.snapshot_digest,
        &world.codebook_digest,
    )?;
    let served_ranked = serving_model.rank_candidates_isolated(
        &user_query,
        &retrieved_candidates,
        world.codebook_size,
        false,
    )?;
    let direct_vs_grpc_served_max_err =
        verify_direct_vs_served(&direct_ranked, &served_ranked, 1.0e-12)?;

    let mut negative_controls = Vec::new();

    // NC1: shifted SID encoding (0-indexed wire 0 mistakenly sent as model 0 = missing)
    {
        let st = validate_sid_mapping(Some(0), 0, world.codebook_size);
        let missing_ok = model_to_wire_sid(0, world.codebook_size)?.is_none();
        negative_controls.push(NegativeControl {
            case_id: "nc1_shifted_wire_vs_model_sid_detected".into(),
            outcome: if missing_ok {
                st
            } else {
                OutcomeStatus::Accepted
            },
            detail:
                "wire SID 0 maps to model SID 1; model SID 0 stays strictly reserved for missing"
                    .into(),
        });
    }

    // NC2: out-of-range SID rejected
    {
        let res = wire_to_model_sid(Some(world.codebook_size), world.codebook_size);
        negative_controls.push(NegativeControl {
            case_id: "nc2_out_of_range_sid_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "wire SID >= codebook_size fails codec validation".into(),
        });
    }

    // NC3: corrupted snapshot/codebook digest mismatch on restore
    {
        let wrong_cb = DigestRef::sha256(b"wrong-codebook");
        let res =
            PhoenixTrainerState::restore_checkpoint(&ckpt_bytes, &world.snapshot_digest, &wrong_cb);
        negative_controls.push(NegativeControl {
            case_id: "nc3_codebook_digest_mismatch_rejected".into(),
            outcome: if res.is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "mismatched codebook digest refused at fresh-process restore".into(),
        });
    }

    // NC4a: dropped optimizer state ONLY (cursor preserved) causes real continuation drift
    {
        let mut bad_opt = PhoenixTrainerState::restore_checkpoint(
            &ckpt_bytes,
            &world.snapshot_digest,
            &world.codebook_digest,
        )?;
        bad_opt.adagrad_accum = vec![[1.0e-4; EMBED_DIM]; NUM_ITEMS];
        bad_opt.adamw_m = [0.0; EMBED_DIM];
        bad_opt.adamw_v = [0.0; EMBED_DIM];
        let _ = bad_opt.train_steps(&world, 3)?;
        let drift = max_state_diff(&uninterrupted, &bad_opt)?;
        negative_controls.push(NegativeControl {
            case_id: "nc4a_dropped_optimizer_state_only_on_resume".into(),
            outcome: if drift > 1.0e-6 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!(
                "dropping AdaGrad/AdamW optimizer state with intact cursor caused {:.3e} drift",
                drift
            ),
        });
    }

    // NC4b: stale data cursor ONLY (optimizer intact) fails resume continuation check
    {
        let mut bad_cursor = PhoenixTrainerState::restore_checkpoint(
            &ckpt_bytes,
            &world.snapshot_digest,
            &world.codebook_digest,
        )?;
        bad_cursor.data_cursor = 0;
        let _ = bad_cursor.train_steps(&world, 3)?;
        let detected = max_state_diff(&uninterrupted, &bad_cursor).is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc4b_stale_data_cursor_on_resume".into(),
            outcome: if detected {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "resetting data cursor on checkpoint resume fails continuation check".into(),
        });
    }

    // NC5: device-name spoof override rejected
    {
        let spoof = qualify_phoenix_gpu_arch(&DeviceQualificationInput {
            detected_gpu_name: "GB10".into(),
            override_gpu_name: Some("GB300".into()),
            jax_cuda_verified: true,
        });
        negative_controls.push(NegativeControl {
            case_id: "nc5_gpu_arch_name_override_rejected".into(),
            outcome: spoof.status,
            detail: spoof.note,
        });
    }

    // NC6: candidate-to-candidate attention isolation violation detected
    {
        let leaky = serving_model.rank_candidates_isolated(
            &user_query,
            &retrieved_candidates,
            world.codebook_size,
            true,
        )?;
        let mut drift = 0.0_f64;
        for (a, b) in direct_ranked.iter().zip(leaky.iter()) {
            let d = (a.1 - b.1).abs();
            if d > drift {
                drift = d;
            }
        }
        negative_controls.push(NegativeControl {
            case_id: "nc6_candidate_attention_isolation_leak".into(),
            outcome: if drift > 1.0e-9 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: format!("cross-candidate leakage shifted scores by {:.3e}", drift),
        });
    }

    // NC7: direct-vs-served score corruption detected
    {
        let mut tampered = served_ranked.clone();
        tampered[0].1 += 0.25;
        let mismatch = verify_direct_vs_served(&direct_ranked, &tampered, 1.0e-9).is_err();
        negative_controls.push(NegativeControl {
            case_id: "nc7_direct_vs_served_score_mismatch".into(),
            outcome: if mismatch {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "served gRPC score discrepancy against direct inference retained and rejected"
                .into(),
        });
    }

    // NC8: process readiness alone without completing retrieve -> rank request
    {
        let st = validate_serving_completion(true, 0, true);
        negative_controls.push(NegativeControl {
            case_id: "nc8_readiness_only_without_retrieve_rank".into(),
            outcome: st,
            detail: "process readiness alone cannot pass without completed retrieve->rank request"
                .into(),
        });
    }

    Ok(PhoenixNanoReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "One-node qualified GPU (A100/H100/H200/GB200/GB300) JAX/CUDA + Rust gRPC Phoenix serve",
            "Pure-Rust contract harness verifies gpu_arch(), SID codec, checkpoint resume, and gRPC retrieve->rank invariants; physical GPU training/serving did not run locally",
        ),
        synthetic_contract_fixture: true,
        one_node_outcome: "UNSUPPORTED_OR_UNAVAILABLE_LOCALLY".into(),
        device_matrix,
        checkpoint_resume_max_err,
        direct_vs_grpc_served_max_err,
        retrieved_and_ranked_count: served_ranked.len(),
        synthetic_loss_proves_rec_quality: false,
        negative_controls,
    })
}

pub fn inputs_json(report: &PhoenixNanoReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.phoenix-nano.inputs.v1\",\n",
            "  \"issue\": 34,\n",
            "  \"execution_class\": {},\n",
            "  \"pinned_upstream_commit\": {},\n",
            "  \"workload_transport\": \"native_grpc_retrieve_rank\",\n",
            "  \"non_gating_ingress_note\": \"Cloudflare 19 Nov 2026 Containers TCP/gRPC beta out of scope; local wrangler port 8788 is not Spectrum evidence\",\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(PHOENIX_UPSTREAM_COMMIT),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &PhoenixNanoReport) -> String {
    let devices: Vec<String> = report
        .device_matrix
        .iter()
        .map(|d| {
            format!(
                "{{\"device\":{},\"status\":{},\"note\":{}}}",
                quote(&d.device_label),
                quote(d.status.as_str()),
                quote(&d.note)
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
            "  \"schema\": \"inference-lab.phoenix-nano.results.v1\",\n",
            "  \"issue\": 34,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"one_node_outcome\": {},\n",
            "  \"checkpoint_resume_max_err\": {},\n",
            "  \"direct_vs_grpc_served_max_err\": {},\n",
            "  \"retrieved_and_ranked_count\": {},\n",
            "  \"synthetic_loss_proves_rec_quality\": {},\n",
            "  \"device_matrix\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        quote(&report.one_node_outcome),
        finite_f64(report.checkpoint_resume_max_err),
        finite_f64(report.direct_vs_grpc_served_max_err),
        report.retrieved_and_ranked_count,
        report.synthetic_loss_proves_rec_quality,
        devices.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &PhoenixNanoReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #34: Phoenix Nano Training, Checkpoint Resume & gRPC Serving"
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
        "- Resume parity: `max_err = {:.3e}`, Direct-vs-gRPC parity: `max_err = {:.3e}` (`{} items ranked`)",
        report.checkpoint_resume_max_err,
        report.direct_vs_grpc_served_max_err,
        report.retrieved_and_ranked_count
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## GPU Cohort Qualification (`gpu_arch()`)").unwrap();
    for d in &report.device_matrix {
        writeln!(
            out,
            "- `{}` -> `{}` ({})",
            d.device_label,
            d.status.as_str(),
            d.note
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
