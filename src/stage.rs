//! Issue #41: Versioned transformer stage contract and bounded Apple–Tenstorrent runner.
//!
//! Implements a real pure-Rust 4-layer dense causal transformer reference (`f64`: RMSNorm,
//! RoPE causal attention with stage-local KV cache, SwiGLU residual block, single greedy
//! sampler) to verify split vs unsplit numerical parity across full prefill, chunked prefill,
//! forced-token continuation, reset/reuse, and two distinct multi-stage placements.
//!
//! Enforces strict tensor-envelope validation, poisoned-buffer/stale-KV state transitions on
//! timeout/cancel, and explicit `UnavailableHardwareRuntime` rejection (zero silent CPU fallback)
//! when Apple GPU (Asahi / Metal-MLX) or Tenstorrent P150 execution is requested locally.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, quote, usize_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    ReferenceCpu,
    SimulatedSplit,
    AppleGpuAsahi,
    AppleGpuMetalMlx,
    TenstorrentP150,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReferenceCpu => "reference_cpu",
            Self::SimulatedSplit => "simulated_split",
            Self::AppleGpuAsahi => "apple_gpu_asahi",
            Self::AppleGpuMetalMlx => "apple_gpu_metal_mlx",
            Self::TenstorrentP150 => "tenstorrent_p150",
        }
    }

    pub const fn is_physical_hardware(self) -> bool {
        matches!(
            self,
            Self::AppleGpuAsahi | Self::AppleGpuMetalMlx | Self::TenstorrentP150
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDtype {
    F64Reference,
    F32,
    Bf16,
    Fp16,
}

impl TensorDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F64Reference => "f64_reference",
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Fp16 => "fp16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

impl ByteOrder {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LittleEndian => "little_endian",
            Self::BigEndian => "big_endian",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePhase {
    Prefill,
    ChunkedPrefill,
    Decode,
}

impl StagePhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::ChunkedPrefill => "chunked_prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    Free,
    HostOwned,
    InFlightDevice,
    Completed,
    PoisonedAfterTimeoutOrCancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvState {
    Empty,
    Valid { consumed_positions: usize },
    InvalidatedUnknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model_id: String,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAssignment {
    pub stage_id: String,
    pub backend: BackendKind,
    pub start_layer: usize,
    pub end_layer: usize, // exclusive
    pub owns_embedding: bool,
    pub owns_final_norm_and_logits: bool,
    pub owns_sampler: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunBinding {
    pub schema_version: String,
    pub checkpoint_digest: DigestRef,
    pub effective_weight_digest: DigestRef,
    pub tokenizer_digest: DigestRef,
    pub dtype: TensorDtype,
    pub byte_order: ByteOrder,
    pub runtime_build_id: String,
    pub config: ModelConfig,
    pub placement: Vec<StageAssignment>,
}

/// Validate placement manifest: contiguous layers covering `0..num_layers`,
/// exact single embedding owner, single final-norm/logit owner, single sampling owner,
/// and no silent CPU fallback if physical hardware is requested.
pub fn validate_binding(binding: &RunBinding) -> Result<(), (OutcomeStatus, String)> {
    if binding.placement.is_empty() {
        return Err((OutcomeStatus::Invalid, "empty_placement_manifest".into()));
    }
    let mut expected_layer = 0usize;
    let mut embed_owners = 0usize;
    let mut logit_owners = 0usize;
    let mut sampler_owners = 0usize;

    for stage in &binding.placement {
        if stage.backend.is_physical_hardware() {
            return Err((
                OutcomeStatus::UnavailableHardwareRuntime,
                format!(
                    "physical_backend_unavailable_no_silent_cpu_fallback:{}",
                    stage.backend.as_str()
                ),
            ));
        }
        if stage.start_layer >= stage.end_layer || stage.end_layer > binding.config.num_layers {
            return Err((
                OutcomeStatus::Invalid,
                format!(
                    "invalid_layer_interval:{}:{}..{}",
                    stage.stage_id, stage.start_layer, stage.end_layer
                ),
            ));
        }
        if stage.start_layer != expected_layer {
            return Err((
                OutcomeStatus::Invalid,
                format!(
                    "non_contiguous_or_overlapping_layer_interval:expected={expected_layer}:got={}",
                    stage.start_layer
                ),
            ));
        }
        expected_layer = stage.end_layer;
        if stage.owns_embedding {
            embed_owners += 1;
        }
        if stage.owns_final_norm_and_logits {
            logit_owners += 1;
        }
        if stage.owns_sampler {
            sampler_owners += 1;
        }
    }

    if expected_layer != binding.config.num_layers {
        return Err((
            OutcomeStatus::Invalid,
            format!(
                "incomplete_layer_coverage:covered={expected_layer}:required={}",
                binding.config.num_layers
            ),
        ));
    }
    if embed_owners != 1 || logit_owners != 1 || sampler_owners != 1 {
        return Err((
            OutcomeStatus::Invalid,
            format!(
                "invalid_single_owner_contract:embed={embed_owners}:logits={logit_owners}:sampler={sampler_owners}"
            ),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorEnvelope {
    pub schema_version: String,
    pub checkpoint_digest: DigestRef,
    pub effective_weight_digest: DigestRef,
    pub dtype: TensorDtype,
    pub byte_order: ByteOrder,
    pub logical_shape: Vec<usize>, // [batch, seq_len, hidden_dim]
    pub strides: Vec<usize>,
    pub padding: Vec<usize>,
    pub batch_id: u64,
    pub sequence_id: u64,
    pub valid_length: usize,
    pub positions: Vec<usize>,
    pub causal_mask: bool,
    pub request_generation: u64,
    pub stage_instance_id: String,
    pub phase: StagePhase,
    pub consumed_positions: usize,
    pub expected_next_position: usize,
    pub data: Vec<f64>,
    pub payload_digest: DigestRef,
}

impl TensorEnvelope {
    pub fn compute_payload_digest(data: &[f64]) -> DigestRef {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        DigestRef::sha256(&bytes)
    }
}

/// Deterministic frozen weights for a small 4-layer dense causal transformer.
#[derive(Debug, Clone)]
pub struct ToyTransformerWeights {
    pub config: ModelConfig,
    pub checkpoint_digest: DigestRef,
    pub effective_weight_digest: DigestRef,
    pub embed: Vec<f64>, // [vocab, hidden]
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f64>, // [hidden]
    pub lm_head: Vec<f64>,    // [vocab, hidden]
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f64>,
    pub wq: Vec<f64>, // [hidden, hidden]
    pub wk: Vec<f64>, // [hidden, hidden]
    pub wv: Vec<f64>, // [hidden, hidden]
    pub wo: Vec<f64>, // [hidden, hidden]
    pub ffn_norm: Vec<f64>,
    pub w_gate: Vec<f64>, // [ffn, hidden]
    pub w_up: Vec<f64>,   // [ffn, hidden]
    pub w_down: Vec<f64>, // [hidden, ffn]
}

fn deterministic_weight(seed: u64, idx: usize, scale: f64) -> f64 {
    let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
    unit * scale
}

pub fn frozen_dense_model() -> ToyTransformerWeights {
    let config = ModelConfig {
        model_id: "frozen-dense-4l-h8-v16-v1".into(),
        num_layers: 4,
        hidden_dim: 8,
        ffn_dim: 16,
        vocab_size: 16,
        norm_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let h = config.hidden_dim;
    let f = config.ffn_dim;
    let v = config.vocab_size;

    let embed: Vec<f64> = (0..v * h)
        .map(|i| deterministic_weight(100, i, 0.4))
        .collect();
    let mut layers = Vec::with_capacity(config.num_layers);
    for l in 0..config.num_layers {
        let base = 1000 + (l as u64) * 100;
        layers.push(LayerWeights {
            attn_norm: vec![1.0; h],
            wq: (0..h * h)
                .map(|i| deterministic_weight(base + 1, i, 0.25))
                .collect(),
            wk: (0..h * h)
                .map(|i| deterministic_weight(base + 2, i, 0.25))
                .collect(),
            wv: (0..h * h)
                .map(|i| deterministic_weight(base + 3, i, 0.25))
                .collect(),
            wo: (0..h * h)
                .map(|i| deterministic_weight(base + 4, i, 0.25))
                .collect(),
            ffn_norm: vec![1.0; h],
            w_gate: (0..f * h)
                .map(|i| deterministic_weight(base + 5, i, 0.25))
                .collect(),
            w_up: (0..f * h)
                .map(|i| deterministic_weight(base + 6, i, 0.25))
                .collect(),
            w_down: (0..h * f)
                .map(|i| deterministic_weight(base + 7, i, 0.25))
                .collect(),
        });
    }
    let final_norm = vec![1.0; h];
    let lm_head: Vec<f64> = (0..v * h)
        .map(|i| deterministic_weight(9000, i, 0.35))
        .collect();

    let mut digest_bytes = Vec::new();
    for val in embed.iter().chain(final_norm.iter()).chain(lm_head.iter()) {
        digest_bytes.extend_from_slice(&val.to_le_bytes());
    }
    for l in &layers {
        for val in
            l.wq.iter()
                .chain(l.wk.iter())
                .chain(l.wv.iter())
                .chain(l.wo.iter())
                .chain(l.w_gate.iter())
                .chain(l.w_up.iter())
                .chain(l.w_down.iter())
        {
            digest_bytes.extend_from_slice(&val.to_le_bytes());
        }
    }
    let checkpoint_digest = DigestRef::sha256(&digest_bytes);
    let effective_weight_digest = checkpoint_digest.clone();

    ToyTransformerWeights {
        config,
        checkpoint_digest,
        effective_weight_digest,
        embed,
        layers,
        final_norm,
        lm_head,
    }
}

fn rms_norm(x: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / (x.len() as f64);
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(v, g)| v * inv * g)
        .collect()
}

fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for (r, slot) in out.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut sum = 0.0;
        for (a, b) in row.iter().zip(x.iter()) {
            sum += a * b;
        }
        *slot = sum;
    }
    out
}

fn apply_rope(v: &mut [f64], pos: usize, theta: f64) {
    let d = v.len();
    let mut i = 0;
    while i + 1 < d {
        let exponent = (i as f64) / (d as f64);
        let freq = (pos as f64) / theta.powf(exponent);
        let (sin_f, cos_f) = freq.sin_cos();
        let x0 = v[i];
        let x1 = v[i + 1];
        v[i] = x0 * cos_f - x1 * sin_f;
        v[i + 1] = x0 * sin_f + x1 * cos_f;
        i += 2;
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// One stateful stage holding resident layer-local attention KV caches and explicit buffer state.
#[derive(Debug, Clone)]
pub struct StageRuntime {
    pub assignment: StageAssignment,
    pub binding_checkpoint: DigestRef,
    pub binding_effective_weight: DigestRef,
    pub dtype: TensorDtype,
    pub byte_order: ByteOrder,
    pub active_generation: u64,
    pub buffer_state: BufferState,
    pub kv_state: KvState,
    // Per assigned layer: Vec of K vectors and V vectors indexed by token position.
    pub kv_keys: Vec<Vec<Vec<f64>>>,
    pub kv_vals: Vec<Vec<Vec<f64>>>,
}

impl StageRuntime {
    pub fn initialize(binding: &RunBinding, assignment: StageAssignment) -> Self {
        let count = assignment.end_layer - assignment.start_layer;
        Self {
            assignment,
            binding_checkpoint: binding.checkpoint_digest.clone(),
            binding_effective_weight: binding.effective_weight_digest.clone(),
            dtype: binding.dtype,
            byte_order: binding.byte_order,
            active_generation: 1,
            buffer_state: BufferState::Free,
            kv_state: KvState::Empty,
            kv_keys: vec![Vec::new(); count],
            kv_vals: vec![Vec::new(); count],
        }
    }

    pub fn reset(&mut self, next_generation: u64) {
        self.active_generation = next_generation;
        self.buffer_state = BufferState::Free;
        self.kv_state = KvState::Empty;
        for k in &mut self.kv_keys {
            k.clear();
        }
        for v in &mut self.kv_vals {
            v.clear();
        }
    }

    /// Simulate a timeout or cancellation while a device buffer is in flight:
    /// poisons the buffer and invalidates KV state so neither can be silently reused.
    pub fn record_timeout_or_cancel(&mut self) {
        self.buffer_state = BufferState::PoisonedAfterTimeoutOrCancel;
        self.kv_state = KvState::InvalidatedUnknown;
    }

    pub fn step(
        &mut self,
        weights: &ToyTransformerWeights,
        input: &TensorEnvelope,
    ) -> Result<TensorEnvelope, (OutcomeStatus, String)> {
        if self.assignment.backend.is_physical_hardware() {
            return Err((
                OutcomeStatus::UnavailableHardwareRuntime,
                format!(
                    "physical_backend_unavailable:{}",
                    self.assignment.backend.as_str()
                ),
            ));
        }
        if self.buffer_state == BufferState::PoisonedAfterTimeoutOrCancel {
            return Err((
                OutcomeStatus::Invalid,
                "poisoned_buffer_cannot_be_reused_after_timeout_or_cancel".into(),
            ));
        }
        if self.kv_state == KvState::InvalidatedUnknown {
            return Err((
                OutcomeStatus::Invalid,
                "stale_or_invalidated_kv_cannot_continue_without_reset".into(),
            ));
        }
        if input.checkpoint_digest != self.binding_checkpoint
            || input.effective_weight_digest != self.binding_effective_weight
        {
            return Err((
                OutcomeStatus::Invalid,
                "checkpoint_or_effective_weight_digest_mismatch".into(),
            ));
        }
        if input.dtype != self.dtype || input.byte_order != self.byte_order {
            return Err((
                OutcomeStatus::Invalid,
                "dtype_or_byte_order_mismatch".into(),
            ));
        }
        if input.request_generation != self.active_generation {
            return Err((
                OutcomeStatus::Invalid,
                format!(
                    "stale_request_generation:expected={}:got={}",
                    self.active_generation, input.request_generation
                ),
            ));
        }
        if input.logical_shape.len() != 3 {
            return Err((
                OutcomeStatus::Invalid,
                "expected_3d_activation_shape".into(),
            ));
        }
        let batch = input.logical_shape[0];
        let seq_len = input.logical_shape[1];
        let hidden = input.logical_shape[2];
        if batch != 1 || hidden != weights.config.hidden_dim {
            return Err((
                OutcomeStatus::Invalid,
                "incompatible_batch_or_hidden_dim".into(),
            ));
        }
        let expected_strides = vec![seq_len * hidden, hidden, 1];
        if input.strides != expected_strides {
            return Err((OutcomeStatus::Invalid, "incompatible_stride_layout".into()));
        }
        let required_elems = batch * seq_len * hidden;
        if input.data.len() < required_elems {
            return Err((OutcomeStatus::Invalid, "truncated_tensor_buffer".into()));
        }
        if input.payload_digest != TensorEnvelope::compute_payload_digest(&input.data) {
            return Err((
                OutcomeStatus::Invalid,
                "payload_digest_integrity_mismatch".into(),
            ));
        }
        let current_consumed = match self.kv_state {
            KvState::Empty => 0,
            KvState::Valid { consumed_positions } => consumed_positions,
            KvState::InvalidatedUnknown => unreachable!(),
        };
        if input.expected_next_position != current_consumed || input.positions.len() != seq_len {
            return Err((
                OutcomeStatus::Invalid,
                format!(
                    "wrong_token_position:expected={current_consumed}:got={}",
                    input.expected_next_position
                ),
            ));
        }
        for (idx, &p) in input.positions.iter().enumerate() {
            if p != current_consumed + idx {
                return Err((
                    OutcomeStatus::Invalid,
                    format!(
                        "non_monotonic_position:expected={}:got={p}",
                        current_consumed + idx
                    ),
                ));
            }
        }

        self.buffer_state = BufferState::InFlightDevice;
        let mut hidden_states: Vec<Vec<f64>> = (0..seq_len)
            .map(|t| input.data[t * hidden..(t + 1) * hidden].to_vec())
            .collect();

        for local_idx in 0..(self.assignment.end_layer - self.assignment.start_layer) {
            let layer_idx = self.assignment.start_layer + local_idx;
            let lw = &weights.layers[layer_idx];
            let mut next_states = Vec::with_capacity(seq_len);

            for (t, h_in) in hidden_states.iter().enumerate() {
                let pos = input.positions[t];
                let normed = rms_norm(h_in, &lw.attn_norm, weights.config.norm_eps);
                let mut q = matvec(&lw.wq, hidden, hidden, &normed);
                let mut k = matvec(&lw.wk, hidden, hidden, &normed);
                let v = matvec(&lw.wv, hidden, hidden, &normed);
                apply_rope(&mut q, pos, weights.config.rope_theta);
                apply_rope(&mut k, pos, weights.config.rope_theta);

                self.kv_keys[local_idx].push(k);
                self.kv_vals[local_idx].push(v);

                let kv_len = self.kv_keys[local_idx].len();
                let scale = 1.0 / (hidden as f64).sqrt();
                let mut scores = Vec::with_capacity(kv_len);
                let mut max_s = f64::NEG_INFINITY;
                for prev_k in &self.kv_keys[local_idx] {
                    let dot: f64 = q.iter().zip(prev_k.iter()).map(|(a, b)| a * b).sum();
                    let s = dot * scale;
                    if s > max_s {
                        max_s = s;
                    }
                    scores.push(s);
                }
                let mut weights_exp = Vec::with_capacity(kv_len);
                let mut sum_exp = 0.0;
                for s in scores {
                    let e = (s - max_s).exp();
                    sum_exp += e;
                    weights_exp.push(e);
                }
                let mut ctx = vec![0.0; hidden];
                for (w_att, prev_v) in weights_exp.iter().zip(self.kv_vals[local_idx].iter()) {
                    let prob = *w_att / sum_exp;
                    for (c, pv) in ctx.iter_mut().zip(prev_v.iter()) {
                        *c += prob * pv;
                    }
                }
                let attn_proj = matvec(&lw.wo, hidden, hidden, &ctx);
                let post_attn: Vec<f64> = h_in
                    .iter()
                    .zip(attn_proj.iter())
                    .map(|(a, b)| a + b)
                    .collect();

                let ffn_in = rms_norm(&post_attn, &lw.ffn_norm, weights.config.norm_eps);
                let gate = matvec(&lw.w_gate, weights.config.ffn_dim, hidden, &ffn_in);
                let up = matvec(&lw.w_up, weights.config.ffn_dim, hidden, &ffn_in);
                let gated: Vec<f64> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(g, u)| silu(*g) * u)
                    .collect();
                let ffn_out = matvec(&lw.w_down, hidden, weights.config.ffn_dim, &gated);
                let post_ffn: Vec<f64> = post_attn
                    .iter()
                    .zip(ffn_out.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                next_states.push(post_ffn);
            }
            hidden_states = next_states;
        }

        let new_consumed = current_consumed + seq_len;
        self.kv_state = KvState::Valid {
            consumed_positions: new_consumed,
        };
        self.buffer_state = BufferState::Completed;

        let mut flat = Vec::with_capacity(seq_len * hidden);
        for row in hidden_states {
            flat.extend_from_slice(&row);
        }
        let payload_digest = TensorEnvelope::compute_payload_digest(&flat);
        Ok(TensorEnvelope {
            schema_version: input.schema_version.clone(),
            checkpoint_digest: input.checkpoint_digest.clone(),
            effective_weight_digest: input.effective_weight_digest.clone(),
            dtype: input.dtype,
            byte_order: input.byte_order,
            logical_shape: vec![1, seq_len, hidden],
            strides: expected_strides,
            padding: vec![0, 0, 0],
            batch_id: input.batch_id,
            sequence_id: input.sequence_id,
            valid_length: seq_len,
            positions: input.positions.clone(),
            causal_mask: true,
            request_generation: input.request_generation,
            stage_instance_id: self.assignment.stage_id.clone(),
            phase: input.phase,
            consumed_positions: new_consumed,
            expected_next_position: input.expected_next_position,
            data: flat,
            payload_digest,
        })
    }
}

fn embed_tokens(
    weights: &ToyTransformerWeights,
    binding: &RunBinding,
    tokens: &[usize],
    start_pos: usize,
    generation: u64,
    phase: StagePhase,
) -> TensorEnvelope {
    let h = weights.config.hidden_dim;
    let seq_len = tokens.len();
    let mut data = Vec::with_capacity(seq_len * h);
    for &tok in tokens {
        let id = tok % weights.config.vocab_size;
        data.extend_from_slice(&weights.embed[id * h..(id + 1) * h]);
    }
    let payload_digest = TensorEnvelope::compute_payload_digest(&data);
    TensorEnvelope {
        schema_version: binding.schema_version.clone(),
        checkpoint_digest: binding.checkpoint_digest.clone(),
        effective_weight_digest: binding.effective_weight_digest.clone(),
        dtype: binding.dtype,
        byte_order: binding.byte_order,
        logical_shape: vec![1, seq_len, h],
        strides: vec![seq_len * h, h, 1],
        padding: vec![0, 0, 0],
        batch_id: 1,
        sequence_id: 1,
        valid_length: seq_len,
        positions: (start_pos..start_pos + seq_len).collect(),
        causal_mask: true,
        request_generation: generation,
        stage_instance_id: "embedding-host".into(),
        phase,
        consumed_positions: start_pos,
        expected_next_position: start_pos,
        data,
        payload_digest,
    }
}

fn compute_last_token_logits(weights: &ToyTransformerWeights, env: &TensorEnvelope) -> Vec<f64> {
    let h = weights.config.hidden_dim;
    let last_offset = (env.valid_length - 1) * h;
    let last_hidden = &env.data[last_offset..last_offset + h];
    let normed = rms_norm(last_hidden, &weights.final_norm, weights.config.norm_eps);
    matvec(&weights.lm_head, weights.config.vocab_size, h, &normed)
}

fn greedy_argmax(logits: &[f64]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlacementParityResult {
    pub placement_name: String,
    pub mode: String,
    pub max_boundary_abs_error: f64,
    pub max_logit_abs_error: f64,
    pub tolerance: f64,
    pub passed: bool,
    pub sampled_tokens: Vec<usize>,
    pub matches_reference_tokens: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegativeControlOutcome {
    pub control_name: String,
    pub rejected: bool,
    pub status: OutcomeStatus,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageContractReport {
    pub execution_class: ExecutionClass,
    pub hardware_gates: Vec<HardwareGate>,
    pub model_id: String,
    pub checkpoint_digest: DigestRef,
    pub reference_sampled_tokens: Vec<usize>,
    pub parity_checks: Vec<PlacementParityResult>,
    pub negative_controls: Vec<NegativeControlOutcome>,
}

pub fn make_binding(
    weights: &ToyTransformerWeights,
    placement: Vec<StageAssignment>,
) -> RunBinding {
    RunBinding {
        schema_version: "inference-lab.transformer-stage.v1".into(),
        checkpoint_digest: weights.checkpoint_digest.clone(),
        effective_weight_digest: weights.effective_weight_digest.clone(),
        tokenizer_digest: DigestRef::sha256(b"inference-lab:stage-toy-tokenizer:v1"),
        dtype: TensorDtype::F64Reference,
        byte_order: ByteOrder::LittleEndian,
        runtime_build_id: "pure-rust-reference-0.1.0".into(),
        config: weights.config.clone(),
        placement,
    }
}

struct PipelineOutput {
    final_activation: Vec<f64>,
    all_logits: Vec<f64>,
    sampled_tokens: Vec<usize>,
    stages: Vec<StageRuntime>,
}

struct ParityCaseSpec<'a> {
    name: &'a str,
    mode: &'a str,
    binding: &'a RunBinding,
    chunks: Vec<&'a [usize]>,
    generation: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetReuseOutcome {
    pub populated_kv_len: usize,
    pub cleared_kv_len: usize,
    pub final_activation: Vec<f64>,
    pub all_logits: Vec<f64>,
    pub sampled_tokens: Vec<usize>,
}

struct StageRunOutput {
    final_activation: Vec<f64>,
    all_logits: Vec<f64>,
    sampled_tokens: Vec<usize>,
}

fn run_on_stages(
    weights: &ToyTransformerWeights,
    binding: &RunBinding,
    stages: &mut [StageRuntime],
    prompt_chunks: &[&[usize]],
    forced_decode_tokens: &[usize],
    generation: u64,
) -> Result<StageRunOutput, (OutcomeStatus, String)> {
    let mut pos = 0usize;
    let mut last_env = None;
    for (chunk_idx, chunk) in prompt_chunks.iter().enumerate() {
        let phase = if prompt_chunks.len() > 1 {
            StagePhase::ChunkedPrefill
        } else {
            StagePhase::Prefill
        };
        let mut env = embed_tokens(weights, binding, chunk, pos, generation, phase);
        for st in stages.iter_mut() {
            env = st.step(weights, &env)?;
        }
        pos += chunk.len();
        if chunk_idx + 1 == prompt_chunks.len() {
            last_env = Some(env);
        }
    }

    let prefill_env = last_env.expect("prompt_chunks non-empty");
    let mut sampled = Vec::new();
    let mut all_logits = compute_last_token_logits(weights, &prefill_env);
    sampled.push(greedy_argmax(&all_logits));

    let mut final_activation = prefill_env.data.clone();
    for &tok in forced_decode_tokens {
        let mut env = embed_tokens(
            weights,
            binding,
            &[tok],
            pos,
            generation,
            StagePhase::Decode,
        );
        for st in stages.iter_mut() {
            env = st.step(weights, &env)?;
        }
        pos += 1;
        let step_logits = compute_last_token_logits(weights, &env);
        sampled.push(greedy_argmax(&step_logits));
        all_logits.extend_from_slice(&step_logits);
        final_activation = env.data;
    }

    Ok(StageRunOutput {
        final_activation,
        all_logits,
        sampled_tokens: sampled,
    })
}

fn run_pipeline(
    weights: &ToyTransformerWeights,
    binding: &RunBinding,
    prompt_chunks: &[&[usize]],
    forced_decode_tokens: &[usize],
    generation: u64,
) -> Result<PipelineOutput, (OutcomeStatus, String)> {
    validate_binding(binding)?;
    let mut stages: Vec<StageRuntime> = binding
        .placement
        .iter()
        .cloned()
        .map(|a| {
            let mut st = StageRuntime::initialize(binding, a);
            st.reset(generation);
            st
        })
        .collect();

    let run_out = run_on_stages(
        weights,
        binding,
        &mut stages,
        prompt_chunks,
        forced_decode_tokens,
        generation,
    )?;

    Ok(PipelineOutput {
        final_activation: run_out.final_activation,
        all_logits: run_out.all_logits,
        sampled_tokens: run_out.sampled_tokens,
        stages,
    })
}

/// Exercise real reset & reuse: run generation 1 on the given stages (populating their
/// layer-local KV caches with a distinct prompt), verify KV is populated, call `reset(2)`
/// on those same stage instances, verify KV is cleared, then run generation 2 and return output.
pub fn run_reset_reuse_sequence(
    weights: &ToyTransformerWeights,
    binding: &RunBinding,
    gen1_prompt: &[usize],
    gen2_prompt: &[usize],
    gen2_forced_decode: &[usize],
) -> Result<ResetReuseOutcome, (OutcomeStatus, String)> {
    validate_binding(binding)?;
    let mut stages: Vec<StageRuntime> = binding
        .placement
        .iter()
        .cloned()
        .map(|a| StageRuntime::initialize(binding, a))
        .collect();

    let _ = run_on_stages(weights, binding, &mut stages, &[gen1_prompt], &[], 1)?;
    let populated_kv_len = stages[0].kv_keys[0].len();

    for st in &mut stages {
        st.reset(2);
    }
    let cleared_kv_len = stages[0].kv_keys[0].len();

    let run_out = run_on_stages(
        weights,
        binding,
        &mut stages,
        &[gen2_prompt],
        gen2_forced_decode,
        2,
    )?;

    Ok(ResetReuseOutcome {
        populated_kv_len,
        cleared_kv_len,
        final_activation: run_out.final_activation,
        all_logits: run_out.all_logits,
        sampled_tokens: run_out.sampled_tokens,
    })
}

/// Execute the full Issue #41 bounded contract harness:
/// unsplit reference, 2-stage full/chunked prefill + decode, 3-stage split, populated-KV reset/reuse,
/// and 11 adversarial negative controls including zero silent CPU fallback for hardware backends.
pub fn run_default_harness() -> Result<StageContractReport, String> {
    let weights = frozen_dense_model();
    let tol = 1e-12;
    let prompt: [usize; 4] = [2, 7, 11, 4];
    let forced_decode: [usize; 2] = [9, 3];

    let unsplit_binding = make_binding(
        &weights,
        vec![StageAssignment {
            stage_id: "stage-all-0-4".into(),
            backend: BackendKind::ReferenceCpu,
            start_layer: 0,
            end_layer: 4,
            owns_embedding: true,
            owns_final_norm_and_logits: true,
            owns_sampler: true,
        }],
    );

    let split2_binding = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "stage-a-0-2".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 0,
                end_layer: 2,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "stage-b-2-4".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 2,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );

    let split3_binding = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "stage-a-0-1".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 0,
                end_layer: 1,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "stage-b-1-3".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 1,
                end_layer: 3,
                owns_embedding: false,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "stage-c-3-4".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 3,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );

    let ref_out = run_pipeline(&weights, &unsplit_binding, &[&prompt], &forced_decode, 1)
        .map_err(|(_, e)| e)?;

    let mut parity_checks = Vec::new();
    let cases = [
        ParityCaseSpec {
            name: "split-2stage-0..2-2..4",
            mode: "full_prefill_plus_forced_decode",
            binding: &split2_binding,
            chunks: vec![&prompt[..]],
            generation: 1,
        },
        ParityCaseSpec {
            name: "split-2stage-0..2-2..4",
            mode: "chunked_prefill_2+2_plus_forced_decode",
            binding: &split2_binding,
            chunks: vec![&prompt[0..2], &prompt[2..4]],
            generation: 1,
        },
        ParityCaseSpec {
            name: "split-3stage-0..1-1..3-3..4",
            mode: "full_prefill_plus_forced_decode",
            binding: &split3_binding,
            chunks: vec![&prompt[..]],
            generation: 1,
        },
    ];

    for spec in cases {
        let out = run_pipeline(
            &weights,
            spec.binding,
            &spec.chunks,
            &forced_decode,
            spec.generation,
        )
        .map_err(|(_, e)| e)?;
        let _ = out.stages.len();
        let act_err = max_abs_diff(&ref_out.final_activation, &out.final_activation);
        let logit_err = max_abs_diff(&ref_out.all_logits, &out.all_logits);
        let matches_tokens = out.sampled_tokens == ref_out.sampled_tokens;
        parity_checks.push(PlacementParityResult {
            placement_name: spec.name.into(),
            mode: spec.mode.into(),
            max_boundary_abs_error: act_err,
            max_logit_abs_error: logit_err,
            tolerance: tol,
            passed: act_err <= tol && logit_err <= tol && matches_tokens,
            sampled_tokens: out.sampled_tokens,
            matches_reference_tokens: matches_tokens,
        });
    }

    // Real populated-KV reset/reuse check on the SAME stage instances:
    // run distinct generation-1 prompt [5, 13, 1, 8], verify KV populated (4), reset(2), verify cleared (0), run gen 2.
    let reuse = run_reset_reuse_sequence(
        &weights,
        &split2_binding,
        &[5, 13, 1, 8],
        &prompt,
        &forced_decode,
    )
    .map_err(|(_, e)| e)?;
    let reuse_act_err = max_abs_diff(&ref_out.final_activation, &reuse.final_activation);
    let reuse_logit_err = max_abs_diff(&ref_out.all_logits, &reuse.all_logits);
    let reuse_matches = reuse.sampled_tokens == ref_out.sampled_tokens;
    parity_checks.push(PlacementParityResult {
        placement_name: "split-2stage-reset-reuse".into(),
        mode: "populated_kv_gen1_reset_then_gen2_full_prefill_plus_decode".into(),
        max_boundary_abs_error: reuse_act_err,
        max_logit_abs_error: reuse_logit_err,
        tolerance: tol,
        passed: reuse.populated_kv_len == 4
            && reuse.cleared_kv_len == 0
            && reuse_act_err <= tol
            && reuse_logit_err <= tol
            && reuse_matches,
        sampled_tokens: reuse.sampled_tokens,
        matches_reference_tokens: reuse_matches,
    });

    // Adversarial negative controls
    let mut negative_controls = Vec::new();
    let push_neg = |list: &mut Vec<NegativeControlOutcome>,
                    name: &str,
                    res: Result<(), (OutcomeStatus, String)>| match res {
        Ok(()) => list.push(NegativeControlOutcome {
            control_name: name.into(),
            rejected: false,
            status: OutcomeStatus::Accepted,
            reason: "unexpectedly_accepted".into(),
        }),
        Err((status, reason)) => list.push(NegativeControlOutcome {
            control_name: name.into(),
            rejected: true,
            status,
            reason,
        }),
    };

    // 1. Non-contiguous / overlapping layer interval
    let bad_interval = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "s0".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 0,
                end_layer: 2,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "s1".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 1,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );
    push_neg(
        &mut negative_controls,
        "neg-overlapping-layer-interval",
        validate_binding(&bad_interval),
    );

    // 2. Multiple sampling owners
    let multi_sampler = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "s0".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 0,
                end_layer: 2,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: true,
            },
            StageAssignment {
                stage_id: "s1".into(),
                backend: BackendKind::SimulatedSplit,
                start_layer: 2,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );
    push_neg(
        &mut negative_controls,
        "neg-multiple-sampling-owners",
        validate_binding(&multi_sampler),
    );

    // 3. Checkpoint / dtype / layout mismatch
    let mut stage_rt =
        StageRuntime::initialize(&split2_binding, split2_binding.placement[0].clone());
    let mut bad_ckpt_env = embed_tokens(&weights, &split2_binding, &[2], 0, 1, StagePhase::Prefill);
    bad_ckpt_env.checkpoint_digest = DigestRef::sha256(b"different-checkpoint");
    push_neg(
        &mut negative_controls,
        "neg-checkpoint-digest-mismatch",
        stage_rt.step(&weights, &bad_ckpt_env).map(|_| ()),
    );

    let mut bad_dtype_env =
        embed_tokens(&weights, &split2_binding, &[2], 0, 1, StagePhase::Prefill);
    bad_dtype_env.dtype = TensorDtype::Bf16;
    push_neg(
        &mut negative_controls,
        "neg-dtype-mismatch",
        stage_rt.step(&weights, &bad_dtype_env).map(|_| ()),
    );

    // 4. Stale request generation
    let stale_gen_env = embed_tokens(&weights, &split2_binding, &[2], 0, 99, StagePhase::Prefill);
    push_neg(
        &mut negative_controls,
        "neg-stale-request-generation",
        stage_rt.step(&weights, &stale_gen_env).map(|_| ()),
    );

    // 5. Wrong token position
    let wrong_pos_env = embed_tokens(&weights, &split2_binding, &[2], 3, 1, StagePhase::Decode);
    push_neg(
        &mut negative_controls,
        "neg-wrong-token-position",
        stage_rt.step(&weights, &wrong_pos_env).map(|_| ()),
    );

    // 6. Truncated tensor buffer
    let mut trunc_env = embed_tokens(
        &weights,
        &split2_binding,
        &[2, 7],
        0,
        1,
        StagePhase::Prefill,
    );
    trunc_env.data.truncate(5);
    trunc_env.payload_digest = TensorEnvelope::compute_payload_digest(&trunc_env.data);
    push_neg(
        &mut negative_controls,
        "neg-truncated-buffer",
        stage_rt.step(&weights, &trunc_env).map(|_| ()),
    );

    // 7. Incompatible stride / layout mismatch
    let mut bad_layout_env = embed_tokens(
        &weights,
        &split2_binding,
        &[2, 7],
        0,
        1,
        StagePhase::Prefill,
    );
    bad_layout_env.strides = vec![1, 8, 16];
    push_neg(
        &mut negative_controls,
        "neg-incompatible-stride-layout",
        stage_rt.step(&weights, &bad_layout_env).map(|_| ()),
    );

    // 8. Corrupted payload / invalid completion digest integrity mismatch
    let mut bad_digest_env =
        embed_tokens(&weights, &split2_binding, &[2], 0, 1, StagePhase::Prefill);
    bad_digest_env.data[0] += 1.0;
    push_neg(
        &mut negative_controls,
        "neg-payload-digest-integrity-mismatch",
        stage_rt.step(&weights, &bad_digest_env).map(|_| ()),
    );

    // 9. Timed-out / cancelled stage cannot reuse poisoned buffer or stale KV
    let mut poisoned_rt =
        StageRuntime::initialize(&split2_binding, split2_binding.placement[0].clone());
    poisoned_rt.record_timeout_or_cancel();
    let valid_env = embed_tokens(&weights, &split2_binding, &[2], 0, 1, StagePhase::Prefill);
    push_neg(
        &mut negative_controls,
        "neg-timeout-poisoned-buffer-and-stale-kv",
        poisoned_rt.step(&weights, &valid_env).map(|_| ()),
    );

    // 8. Requesting Apple GPU + Tenstorrent P150 must fail with UnavailableHardwareRuntime (no silent CPU fallback!)
    let hw_binding = make_binding(
        &weights,
        vec![
            StageAssignment {
                stage_id: "apple-m2-ultra-0-2".into(),
                backend: BackendKind::AppleGpuMetalMlx,
                start_layer: 0,
                end_layer: 2,
                owns_embedding: true,
                owns_final_norm_and_logits: false,
                owns_sampler: false,
            },
            StageAssignment {
                stage_id: "tt-p150-2-4".into(),
                backend: BackendKind::TenstorrentP150,
                start_layer: 2,
                end_layer: 4,
                owns_embedding: false,
                owns_final_norm_and_logits: true,
                owns_sampler: true,
            },
        ],
    );
    push_neg(
        &mut negative_controls,
        "neg-no-silent-cpu-fallback-for-apple-tt-hardware",
        validate_binding(&hw_binding),
    );

    Ok(StageContractReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gates: vec![
            HardwareGate::unavailable(
                "apple-m2-ultra-gpu-asahi-or-metal-mlx",
                "Split CPU reference validates the stage contract and numerical parity only; physical Apple GPU execution requires an M2 Ultra host.",
            ),
            HardwareGate::unavailable(
                "tenstorrent-p150-worker",
                "Physical Tenstorrent stage execution is owned by TT-Onix (#48-#50) and is unavailable on this host.",
            ),
        ],
        model_id: weights.config.model_id,
        checkpoint_digest: weights.checkpoint_digest,
        reference_sampled_tokens: ref_out.sampled_tokens,
        parity_checks,
        negative_controls,
    })
}

pub fn inputs_json(report: &StageContractReport) -> String {
    format!(
        "{{\"schema\":\"inference-lab.stage-inputs.v1\",\"issue\":41,\"model_id\":{},\"checkpoint_digest\":{},\"dtype\":\"f64_reference\",\"byte_order\":\"little_endian\",\"num_layers\":4,\"hidden_dim\":8,\"ffn_dim\":16,\"vocab_size\":16,\"prompt_tokens\":[2,7,11,4],\"forced_decode_tokens\":[9,3],\"tolerance\":1e-12}}\n",
        quote(&report.model_id),
        report.checkpoint_digest.to_json()
    )
}

pub fn results_json(report: &StageContractReport) -> String {
    let gates: Vec<String> = report
        .hardware_gates
        .iter()
        .map(HardwareGate::to_json)
        .collect();
    let parity: Vec<String> = report
        .parity_checks
        .iter()
        .map(|p| {
            format!(
                "{{\"placement_name\":{},\"mode\":{},\"max_boundary_abs_error\":{},\"max_logit_abs_error\":{},\"tolerance\":{},\"passed\":{},\"sampled_tokens\":{},\"matches_reference_tokens\":{}}}",
                quote(&p.placement_name),
                quote(&p.mode),
                finite_f64(p.max_boundary_abs_error),
                finite_f64(p.max_logit_abs_error),
                finite_f64(p.tolerance),
                p.passed,
                usize_array(&p.sampled_tokens),
                p.matches_reference_tokens
            )
        })
        .collect();
    let negs: Vec<String> = report
        .negative_controls
        .iter()
        .map(|n| {
            format!(
                "{{\"control_name\":{},\"rejected\":{},\"status\":{},\"reason\":{}}}",
                quote(&n.control_name),
                n.rejected,
                quote(n.status.as_str()),
                quote(&n.reason)
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.stage-report.v1\",\"issue\":41,\"package_version\":{},\"execution_class\":{},\"hardware_gates\":[{}],\"model_id\":{},\"checkpoint_digest\":{},\"reference_sampled_tokens\":{},\"parity_checks\":[{}],\"negative_controls\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(report.execution_class.as_str()),
        gates.join(","),
        quote(&report.model_id),
        report.checkpoint_digest.to_json(),
        usize_array(&report.reference_sampled_tokens),
        parity.join(",\n"),
        negs.join(",\n")
    )
}

pub fn summary_markdown(report: &StageContractReport) -> String {
    let mut out = format!(
        "# Issue #41 — Versioned transformer stage contract report\n\n\
         - **Execution class**: `{}`\n\
         - **Model**: `{}` (`sha256:{}`)\n\
         - **Apple M2 Ultra / Tenstorrent P150**: `unavailable_hardware_runtime` (zero silent CPU fallback)\n\n\
         ## Split vs unsplit numerical parity (reference `f64`)\n\n\
         | Placement | Mode | Max activation err | Max logit err | Passed | Matches reference tokens |\n\
         | --- | --- | --- | --- | --- | --- |\n",
        report.execution_class.as_str(),
        report.model_id,
        &report.checkpoint_digest.hex[..16]
    );
    for p in &report.parity_checks {
        writeln!(
            out,
            "| `{}` | `{}` | {} | {} | `{}` | `{}` |",
            p.placement_name,
            p.mode,
            finite_f64(p.max_boundary_abs_error),
            finite_f64(p.max_logit_abs_error),
            p.passed,
            p.matches_reference_tokens
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\n## Adversarial negative controls\n\n\
         | Control | Rejected | Outcome | Reason |\n\
         | --- | --- | --- | --- |\n",
    );
    for n in &report.negative_controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` |",
            n.control_name,
            n.rejected,
            n.status.as_str(),
            n.reason
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nA split CPU reference establishes contract and numerical equivalence only; it does not \
         qualify physical Apple M2 Ultra or Tenstorrent P150 execution.\n",
    );
    out
}
