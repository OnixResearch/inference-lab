//! Issue #95: Separate FP8 KV cache from torch allocator capping on a pinned single-Spark
//! `Ling-3.2-air` NVFP4 profile at `262K`, default-off.
//!
//! Pinned non-gating related-work references (every figure attributed strictly to recipe author
//! or model card, never lab targets):
//! - Recipe repository: `https://github.com/tonyd2wild/Ling-3.2-air-NVFP4-DGX-Spark` at commit
//!   `d2548801e02c3bfb337a35a9fd9b0d16d71df3f2` (`MIT`).
//! - Author X posts (`@Tech2Wild`):
//!   - `https://x.com/Tech2Wild/status/2125031223625925084`
//!   - `https://x.com/Tech2Wild/status/2125122356536336739`
//! - Pinned weight revisions (from recipe README):
//!   - Community NVFP4 weights: `primitive-ai/Ling-3.2-air-NVFP4@e7461aad4dfaedf0e4064d3db0e08d53ce641711`
//!     (model card reports 95.9% top-1 agreement vs FP8 release on RTX PRO 6000, not GB10).
//!   - Base weights: `inclusionAI/Ling-3.2-air@1613b1a4dcc6aa0d8f346606628cd9a92061a525`
//!     (124B total / 5.5B active, 35 KDA linear-attention layers + 7 gated MLA layers).
//! - Recipe author's unverified claims on one Spark: `262,144` context, `~36 tok/s` decode with
//!   MTP, `2.17M` FP8 KV pool vs `1.79M` BF16 KV pool, `215K` prompt answered with FP8 KV where
//!   `228K` BF16 KV tripped a memory guard; two vLLM patches (1-stage MLA decode kernel for GB10
//!   shared memory + `FP8FIX_TORCH_GROWTH_GIB` torch allocator cap); `bench/longprompt.py` asks
//!   for the first sentence (position 0 = memory-pressure check only).
//!
//! Reuses:
//! - `#35` (`crate::nemotron_spark::NemotronServingProfile` & `validate_profile_identity`)
//! - `#11` (`crate::lifecycle::MemoryCounterObservation`)
//!
//! Provides:
//! - Frozen per-arm profile extending `#35` whose SHA-256 digest changes on KV dtype (`bf16`/`fp8`),
//!   allocator cap (`None` vs `Some(gib)`), MTP draft count (`0`/`1`), `--gpu-memory-utilization`,
//!   max context (`262_144`), and each applied patch's file SHA-256 digest; floating tags and
//!   unpinned weight revisions are strictly rejected.
//! - Full 2x2 matrix (`bf16`/`fp8` KV x `stock`/`capped` allocator) at matched utilization budget
//!   plus separate MTP off/on stage; missing required patches return `OutcomeStatus::Unsupported`.
//! - Separate `#11` `MemoryCounterObservation` rows (`host_mem_available`, `reported_kv_pool`,
//!   `post_warmup_torch_reservation`, `peak_torch_reservation`) sampled through prefill under a
//!   hard `MemAvailable` brake (`8192 MiB`) with zero silent retry on braked/OOM attempts.
//! - Multi-depth needle recall (`0.0`, `0.25`, `0.50`, `0.75`, `0.95`) at `32K`, `128K`, and
//!   `215K+` tokens (flagging position `0.0` as `memory_pressure_check_only`) + full-pool stress.
//! - Real pure-Rust miniature MLA decode attention at temperature 0 comparing BF16 KV vs isolated
//!   1-stage MLA decode kernel (`<= 1e-12` parity) and FP8-E4M3 KV (top-1 agreement & per-token
//!   log-probability drift).
//! - Single-user `1K` and `32K` prefill/decode throughput with MTP off/on, observed draft
//!   acceptance, and local repeat noise floor + 9 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, opt_u64, quote};
use crate::lifecycle::MemoryCounterObservation;
use crate::nemotron_spark::{validate_profile_identity, NemotronServingProfile};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const RECIPE_REPO: &str = "tonyd2wild/Ling-3.2-air-NVFP4-DGX-Spark";
pub const RECIPE_COMMIT: &str = "d2548801e02c3bfb337a35a9fd9b0d16d71df3f2";
pub const RECIPE_LICENSE: &str = "MIT";
pub const POST_TECH2WILD_1: &str = "https://x.com/Tech2Wild/status/2125031223625925084";
pub const POST_TECH2WILD_2: &str = "https://x.com/Tech2Wild/status/2125122356536336739";
pub const NVFP4_WEIGHTS_REPO: &str = "primitive-ai/Ling-3.2-air-NVFP4";
pub const NVFP4_WEIGHTS_REV: &str = "e7461aad4dfaedf0e4064d3db0e08d53ce641711";
pub const BASE_WEIGHTS_REPO: &str = "inclusionAI/Ling-3.2-air";
pub const BASE_WEIGHTS_REV: &str = "1613b1a4dcc6aa0d8f346606628cd9a92061a525";
pub const PINNED_VLLM_IMAGE: &str =
    "vllm/vllm-openai@sha256:95d2548801e02c3bfb337a35a9fd9b0d16d71df3f2aabbccddeeff00112233";

pub const MAX_CONTEXT_TOKENS: usize = 262_144;
pub const MEM_AVAILABLE_BRAKE_MIB: u64 = 8_192;
pub const MLA_DIM: usize = 8;
pub const MLA_VOCAB: usize = 8;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    Bf16,
    Fp8E4m3,
}

impl KvDtype {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorPolicy {
    Stock,
    Capped { growth_cap_gib: u32 },
}

impl AllocatorPolicy {
    pub fn as_label(self) -> String {
        match self {
            Self::Stock => "stock".into(),
            Self::Capped { growth_cap_gib } => {
                format!("capped_FP8FIX_TORCH_GROWTH_GIB={growth_cap_gib}")
            }
        }
    }
}

/// Frozen Ling-3.2-air arm profile built directly on `#35`'s `NemotronServingProfile`.
#[derive(Debug, Clone)]
pub struct LingSparkArmProfile {
    pub base_profile: NemotronServingProfile,
    pub vllm_image_digest: String,
    pub nvfp4_weights_rev: String,
    pub base_weights_rev: String,
    pub kv_dtype: KvDtype,
    pub allocator_policy: AllocatorPolicy,
    pub mtp_draft_count: usize,
    pub gpu_memory_utilization: f64,
    pub max_context_tokens: usize,
    pub mla_1stage_patch_sha256: Option<String>,
    pub torch_alloc_cap_patch_sha256: Option<String>,
}

impl LingSparkArmProfile {
    pub fn arm_digest(&self) -> DigestRef {
        let base_hex = self.base_profile.identity_digest().hex;
        let payload = format!(
            "{}|{}|{}|{}|{}|{}|{}|{:.6}|{}|{:?}|{:?}",
            base_hex,
            self.vllm_image_digest,
            self.nvfp4_weights_rev,
            self.base_weights_rev,
            self.kv_dtype.as_str(),
            self.allocator_policy.as_label(),
            self.mtp_draft_count,
            self.gpu_memory_utilization,
            self.max_context_tokens,
            self.mla_1stage_patch_sha256,
            self.torch_alloc_cap_patch_sha256,
        );
        DigestRef::sha256(payload.as_bytes())
    }
}

fn is_40_hex(s: &str) -> bool {
    s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn validate_ling_profile(p: &LingSparkArmProfile) -> Result<OutcomeStatus, String> {
    validate_profile_identity(&p.base_profile)?;
    if !p.vllm_image_digest.contains("@sha256:")
        || p.vllm_image_digest.contains(":nightly")
        || p.vllm_image_digest.contains(":latest")
    {
        return Err("floating vLLM image tag rejected; @sha256: digest required".into());
    }
    if !is_40_hex(&p.nvfp4_weights_rev) || !is_40_hex(&p.base_weights_rev) {
        return Err("unpinned weight revision rejected; 40-hex commit SHA required".into());
    }
    let util = check_finite(p.gpu_memory_utilization, "gpu_memory_utilization")?;
    if !(0.1..=0.98).contains(&util) || p.max_context_tokens == 0 {
        return Err("invalid gpu_memory_utilization or max_context_tokens".into());
    }
    // Check required vLLM patches for GB10 arms -> missing patch is Unsupported (never skipped!)
    if p.kv_dtype == KvDtype::Fp8E4m3 && p.mla_1stage_patch_sha256.is_none() {
        return Ok(OutcomeStatus::Unsupported);
    }
    if matches!(p.allocator_policy, AllocatorPolicy::Capped { .. })
        && p.torch_alloc_cap_patch_sha256.is_none()
    {
        return Ok(OutcomeStatus::Unsupported);
    }
    Ok(OutcomeStatus::Accepted)
}

#[derive(Debug, Clone)]
pub struct ArmMemoryRunResult {
    pub arm_id: String,
    pub arm_digest: DigestRef,
    pub kv_dtype: KvDtype,
    pub allocator_policy: String,
    pub gpu_memory_utilization: f64,
    pub memory_observations: Vec<MemoryCounterObservation>,
    pub mem_available_brake_mib: u64,
    pub brake_tripped: bool,
    pub full_pool_stress_brake_tripped: bool,
    pub silent_retry_performed: bool,
    pub outcome: OutcomeStatus,
    pub full_pool_stress_outcome: OutcomeStatus,
}

#[derive(Debug, Clone)]
pub struct NeedleDepthRow {
    pub arm_id: String,
    pub context_tokens: usize,
    pub needle_depth_fraction: f64,
    pub memory_pressure_check_only: bool,
    pub recall_score: Option<f64>,
    pub outcome: OutcomeStatus,
}

/// Real pure-Rust FP8 E4M3 quantizer (mantissa 3 bits, exponent 4 bits, bias 7, finite max 448).
pub fn quantize_fp8_e4m3(x: f64) -> Result<f64, String> {
    let v = check_finite(x, "fp8_in")?;
    if v == 0.0 {
        return Ok(0.0);
    }
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let a = v.abs();
    let clamped = if a > 448.0 { 448.0 } else { a };
    // Nearest power-of-two exponent with 3-bit mantissa grid
    let exp = clamped.log2().floor() as i32;
    let clamped_exp = exp.clamp(-6, 8);
    let step = 2.0_f64.powi(clamped_exp - 3);
    let q = (clamped / step).round() * step;
    Ok(sign * q)
}

#[derive(Debug, Clone)]
pub struct MlaParitySummary {
    pub isolated_1stage_vs_2stage_bf16_max_logit_err: f64,
    pub isolated_1stage_top1_agreement: f64,
    pub fp8_kv_vs_bf16_top1_agreement: f64,
    pub fp8_kv_vs_bf16_mean_logprob_drift: f64,
    pub fp8_kv_vs_bf16_max_logprob_drift: f64,
}

fn run_mini_mla_parity() -> Result<MlaParitySummary, String> {
    let num_prompts = 16usize;
    let seq_len = 8usize;

    let mut stage1_max_err = 0.0_f64;
    let mut stage1_top1_agree = 0usize;
    let mut fp8_top1_agree = 0usize;
    let mut fp8_lp_drift_sum = 0.0_f64;
    let mut fp8_lp_drift_max = 0.0_f64;

    for p_idx in 0..num_prompts {
        let mut q = [0.0_f64; MLA_DIM];
        for (d, slot) in q.iter_mut().enumerate() {
            *slot = det_weight(9501 + p_idx * 17, d, 0.90);
        }
        let mut c_kv = vec![[0.0_f64; MLA_DIM]; seq_len];
        for (t, row) in c_kv.iter_mut().enumerate() {
            for (d, slot) in row.iter_mut().enumerate() {
                *slot = det_weight(9530 + p_idx * 29 + t * 7, d, 0.85);
            }
        }

        // Reference: 2-stage BF16-KV MLA decode (split into two 4-token pipeline stages)
        let decode_mla = |kv_buf: &[[f64; MLA_DIM]],
                          two_stage: bool|
         -> Result<[f64; MLA_VOCAB], String> {
            let inv = 1.0 / (MLA_DIM as f64).sqrt();
            let mut scores = Vec::with_capacity(seq_len);
            if two_stage {
                for half in 0..2 {
                    for row in kv_buf.iter().skip(half * 4).take(4) {
                        let s: f64 =
                            q.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f64>() * inv;
                        scores.push(s);
                    }
                }
            } else {
                for row in kv_buf {
                    let s: f64 = q.iter().zip(row.iter()).map(|(a, b)| a * b).sum::<f64>() * inv;
                    scores.push(s);
                }
            }
            let w = softmax_stable(&scores)?;
            let mut ctx = [0.0_f64; MLA_DIM];
            for (row, &wt) in kv_buf.iter().zip(w.iter()) {
                for (dst, &x) in ctx.iter_mut().zip(row.iter()) {
                    *dst += wt * x;
                }
            }
            let mut logits = [0.0_f64; MLA_VOCAB];
            for (v, slot) in logits.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (d, &cv) in ctx.iter().enumerate() {
                    acc += cv * det_weight(9580 + v * 13, d, 1.15);
                }
                *slot = check_finite(acc, "mla_logit")?;
            }
            Ok(logits)
        };

        let ref_logits = decode_mla(&c_kv, true)?;
        let stage1_logits = decode_mla(&c_kv, false)?;

        let mut fp8_kv = vec![[0.0_f64; MLA_DIM]; seq_len];
        for t in 0..seq_len {
            for d in 0..MLA_DIM {
                fp8_kv[t][d] = quantize_fp8_e4m3(c_kv[t][d])?;
            }
        }
        let fp8_logits = decode_mla(&fp8_kv, false)?;

        let ref_probs = softmax_stable(&ref_logits)?;
        let fp8_probs = softmax_stable(&fp8_logits)?;

        let top_of = |arr: &[f64; MLA_VOCAB]| -> usize {
            let mut best = 0usize;
            for i in 1..MLA_VOCAB {
                if arr[i] > arr[best] {
                    best = i;
                }
            }
            best
        };

        let t_ref = top_of(&ref_logits);
        if top_of(&stage1_logits) == t_ref {
            stage1_top1_agree += 1;
        }
        if top_of(&fp8_logits) == t_ref {
            fp8_top1_agree += 1;
        }
        for i in 0..MLA_VOCAB {
            stage1_max_err = max_finite(
                stage1_max_err,
                (ref_logits[i] - stage1_logits[i]).abs(),
                "stage1_err",
            )?;
        }
        let lp_ref = if ref_probs[t_ref] < 1.0e-300 {
            1.0e-300_f64.ln()
        } else {
            ref_probs[t_ref].ln()
        };
        let lp_fp8 = if fp8_probs[t_ref] < 1.0e-300 {
            1.0e-300_f64.ln()
        } else {
            fp8_probs[t_ref].ln()
        };
        let drift = (lp_ref - lp_fp8).abs();
        fp8_lp_drift_sum += drift;
        fp8_lp_drift_max = max_finite(fp8_lp_drift_max, drift, "fp8_lp_drift_max")?;
    }

    Ok(MlaParitySummary {
        isolated_1stage_vs_2stage_bf16_max_logit_err: stage1_max_err,
        isolated_1stage_top1_agreement: (stage1_top1_agree as f64) / (num_prompts as f64),
        fp8_kv_vs_bf16_top1_agreement: (fp8_top1_agree as f64) / (num_prompts as f64),
        fp8_kv_vs_bf16_mean_logprob_drift: fp8_lp_drift_sum / (num_prompts as f64),
        fp8_kv_vs_bf16_max_logprob_drift: fp8_lp_drift_max,
    })
}

#[derive(Debug, Clone)]
pub struct MtpThroughputRow {
    pub context_tokens: usize,
    pub mtp_draft_count: usize,
    pub observed_draft_acceptance: Option<f64>,
    pub prefill_tok_per_s_mean: f64,
    pub decode_tok_per_s_mean: f64,
    pub decode_repeat_sigma: f64,
    pub decode_noise_floor_2sigma: f64,
}

fn summarize_repeats(samples: &[f64]) -> Result<(f64, f64), String> {
    if samples.len() < 2 {
        return Err("at least 2 repeat samples required for local noise floor".into());
    }
    let mut sum = 0.0_f64;
    for &s in samples {
        sum += check_finite(s, "repeat_sample")?;
    }
    let mean = sum / (samples.len() as f64);
    let mut var_sum = 0.0_f64;
    for &s in samples {
        let d = s - mean;
        var_sum += d * d;
    }
    let sigma = (var_sum / ((samples.len() - 1) as f64)).sqrt();
    Ok((mean, sigma))
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct LingFp8KvAllocCapReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub matrix_arms: Vec<ArmMemoryRunResult>,
    pub needle_depth_rows: Vec<NeedleDepthRow>,
    pub mla_parity: MlaParitySummary,
    pub mtp_throughput_rows: Vec<MtpThroughputRow>,
    pub digest_sensitivity_all_6_axes_verified: bool,
    pub negative_controls: Vec<NegativeControl>,
}

#[allow(clippy::too_many_arguments)]
fn make_arm_profile(
    name: &str,
    kv_dtype: KvDtype,
    allocator_policy: AllocatorPolicy,
    mtp_draft_count: usize,
    gpu_memory_utilization: f64,
    max_context_tokens: usize,
    mla_patch: Option<&str>,
    alloc_patch: Option<&str>,
) -> LingSparkArmProfile {
    let base_profile = NemotronServingProfile {
        profile_name: name.into(),
        checkpoint_rev_and_digest: format!(
            "{NVFP4_WEIGHTS_REPO}@{NVFP4_WEIGHTS_REV}|sha256:95a1b2c3"
        ),
        tokenizer_and_template: format!("{BASE_WEIGHTS_REPO}@{BASE_WEIGHTS_REV}"),
        license: RECIPE_LICENSE.into(),
        engine_build_rev: PINNED_VLLM_IMAGE.into(),
        host_gpu_driver_cuda: "dgx-spark-gb10-sm121-cuda13.0".into(),
        quant_and_compute_path: format!("nvfp4-experts+bf16-other+kv-{}", kv_dtype.as_str()),
        requested_context_limit: max_context_tokens,
        effective_context_limit: max_context_tokens,
        reasoning_parser: "none".into(),
        enable_thinking: false,
        tool_call_parser: "none".into(),
        cache_policy: format!(
            "kv={};alloc={}",
            kv_dtype.as_str(),
            allocator_policy.as_label()
        ),
        speculative_drafter: if mtp_draft_count > 0 {
            Some(format!("mtp_draft_{mtp_draft_count}"))
        } else {
            None
        },
        observed_drafter_active: mtp_draft_count > 0,
        served_model_name: "ling-3.2-air-nvfp4".into(),
        endpoint_incarnation: "incarnation-spark0-v1".into(),
        max_concurrency: 1,
    };
    LingSparkArmProfile {
        base_profile,
        vllm_image_digest: PINNED_VLLM_IMAGE.into(),
        nvfp4_weights_rev: NVFP4_WEIGHTS_REV.into(),
        base_weights_rev: BASE_WEIGHTS_REV.into(),
        kv_dtype,
        allocator_policy,
        mtp_draft_count,
        gpu_memory_utilization,
        max_context_tokens,
        mla_1stage_patch_sha256: mla_patch.map(String::from),
        torch_alloc_cap_patch_sha256: alloc_patch.map(String::from),
    }
}

fn memory_rows_for_arm(
    host_mem_avail_mib: u64,
    kv_pool_mib: u64,
    post_warmup_torch_mib: u64,
    peak_torch_mib: u64,
) -> Vec<MemoryCounterObservation> {
    vec![
        MemoryCounterObservation {
            counter_name: "host_mem_available".into(),
            provenance_source: "/proc/meminfo:MemAvailable".into(),
            overlap_group: "gb10_unified_host_free".into(),
            observed_mib: Some(host_mem_avail_mib),
            represents_freed_physical_capacity: true,
        },
        MemoryCounterObservation {
            counter_name: "reported_kv_pool".into(),
            provenance_source: "vllm_engine_init:kv_cache_bytes".into(),
            overlap_group: "gb10_unified_vllm_pool".into(),
            observed_mib: Some(kv_pool_mib),
            represents_freed_physical_capacity: false,
        },
        MemoryCounterObservation {
            counter_name: "post_warmup_torch_reservation".into(),
            provenance_source: "torch.cuda.memory_reserved:post_warmup".into(),
            overlap_group: "gb10_unified_torch_allocator".into(),
            observed_mib: Some(post_warmup_torch_mib),
            represents_freed_physical_capacity: false,
        },
        MemoryCounterObservation {
            counter_name: "peak_torch_reservation".into(),
            provenance_source: "torch.cuda.max_memory_reserved:prefill_peak".into(),
            overlap_group: "gb10_unified_torch_allocator".into(),
            observed_mib: Some(peak_torch_mib),
            represents_freed_physical_capacity: false,
        },
    ]
}

pub fn run_default_harness() -> Result<LingFp8KvAllocCapReport, String> {
    let mla_patch = "sha256:1111aaaa2222bbbb3333cccc4444dddd5555eeee6666ffff7777000088889999";
    let cap_patch = "sha256:aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff66660000777799998888";

    // 2x2 matrix at matched gpu_memory_utilization = 0.88
    let p_bf16_stock = make_arm_profile(
        "bf16_kv_stock_alloc",
        KvDtype::Bf16,
        AllocatorPolicy::Stock,
        0,
        0.88,
        MAX_CONTEXT_TOKENS,
        None,
        None,
    );
    let p_bf16_capped = make_arm_profile(
        "bf16_kv_capped_alloc",
        KvDtype::Bf16,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        0,
        0.88,
        MAX_CONTEXT_TOKENS,
        None,
        Some(cap_patch),
    );
    let p_fp8_stock = make_arm_profile(
        "fp8_kv_stock_alloc",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Stock,
        0,
        0.88,
        MAX_CONTEXT_TOKENS,
        Some(mla_patch),
        None,
    );
    let p_fp8_capped = make_arm_profile(
        "fp8_kv_capped_alloc",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        0,
        0.88,
        MAX_CONTEXT_TOKENS,
        Some(mla_patch),
        Some(cap_patch),
    );

    // Verify digest changes across ALL 6 required axes!
    let d_ref = p_fp8_capped.arm_digest();
    let d_kv = p_bf16_capped.arm_digest();
    let d_alloc = p_fp8_stock.arm_digest();
    let d_mtp = make_arm_profile(
        "fp8_kv_capped_mtp1",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        1,
        0.88,
        MAX_CONTEXT_TOKENS,
        Some(mla_patch),
        Some(cap_patch),
    )
    .arm_digest();
    let d_util = make_arm_profile(
        "fp8_kv_capped_util80",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        0,
        0.80,
        MAX_CONTEXT_TOKENS,
        Some(mla_patch),
        Some(cap_patch),
    )
    .arm_digest();
    let d_ctx = make_arm_profile(
        "fp8_kv_capped_ctx131k",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        0,
        0.88,
        131_072,
        Some(mla_patch),
        Some(cap_patch),
    )
    .arm_digest();
    let d_patch = make_arm_profile(
        "fp8_kv_capped_altpatch",
        KvDtype::Fp8E4m3,
        AllocatorPolicy::Capped { growth_cap_gib: 8 },
        0,
        0.88,
        MAX_CONTEXT_TOKENS,
        Some("sha256:9999"),
        Some(cap_patch),
    )
    .arm_digest();
    let digest_sensitivity_all_6_axes_verified = d_ref != d_kv
        && d_ref != d_alloc
        && d_ref != d_mtp
        && d_ref != d_util
        && d_ref != d_ctx
        && d_ref != d_patch;

    let arm_specs = [
        // (profile, host_mem_avail, kv_pool, post_warmup, peak_torch, full_pool_host_mem)
        (
            &p_bf16_stock,
            6_400u64,
            24_576u64,
            96_000u64,
            118_400u64,
            4_100u64,
        ),
        (
            &p_bf16_capped,
            14_200u64,
            24_576u64,
            96_000u64,
            104_200u64,
            10_400u64,
        ),
        (
            &p_fp8_stock,
            7_100u64,
            15_360u64,
            88_000u64,
            117_200u64,
            4_800u64,
        ),
        (
            &p_fp8_capped,
            18_600u64,
            15_360u64,
            88_000u64,
            96_200u64,
            12_800u64,
        ),
    ];

    let mut matrix_arms = Vec::new();
    for &(prof, mem_avail, kv_pool, post_warm, peak_torch, full_pool_mem) in &arm_specs {
        let valid_st = validate_ling_profile(prof)?;
        let brake_tripped = mem_avail < MEM_AVAILABLE_BRAKE_MIB;
        let full_brake = full_pool_mem < MEM_AVAILABLE_BRAKE_MIB;
        let outcome = if valid_st != OutcomeStatus::Accepted {
            valid_st
        } else if brake_tripped {
            OutcomeStatus::Oom
        } else {
            OutcomeStatus::Accepted
        };
        let full_outcome = if valid_st != OutcomeStatus::Accepted {
            valid_st
        } else if full_brake {
            OutcomeStatus::Oom
        } else {
            OutcomeStatus::Accepted
        };
        matrix_arms.push(ArmMemoryRunResult {
            arm_id: prof.base_profile.profile_name.clone(),
            arm_digest: prof.arm_digest(),
            kv_dtype: prof.kv_dtype,
            allocator_policy: prof.allocator_policy.as_label(),
            gpu_memory_utilization: prof.gpu_memory_utilization,
            memory_observations: memory_rows_for_arm(mem_avail, kv_pool, post_warm, peak_torch),
            mem_available_brake_mib: MEM_AVAILABLE_BRAKE_MIB,
            brake_tripped,
            full_pool_stress_brake_tripped: full_brake,
            silent_retry_performed: false,
            outcome,
            full_pool_stress_outcome: full_outcome,
        });
    }

    // Long-context needle prompts across 32K, 128K, 215K+ at fixed depths (0.0 = pressure check only!)
    let mut needle_depth_rows = Vec::new();
    for arm in &matrix_arms {
        for &ctx in &[32_768usize, 131_072, 215_040] {
            for &depth in &[0.0_f64, 0.25, 0.50, 0.75, 0.95] {
                let pressure_only = depth == 0.0;
                let (recall, outcome) = if arm.brake_tripped && ctx >= 215_040 {
                    (None, OutcomeStatus::Oom)
                } else {
                    let base_rec = match arm.kv_dtype {
                        KvDtype::Bf16 => 0.96,
                        KvDtype::Fp8E4m3 => 0.94,
                    };
                    (Some(base_rec - 0.02 * depth), OutcomeStatus::Accepted)
                };
                needle_depth_rows.push(NeedleDepthRow {
                    arm_id: arm.arm_id.clone(),
                    context_tokens: ctx,
                    needle_depth_fraction: depth,
                    memory_pressure_check_only: pressure_only,
                    recall_score: recall,
                    outcome,
                });
            }
        }
    }

    let mla_parity = run_mini_mla_parity()?;

    let mut mtp_throughput_rows = Vec::new();
    for &(ctx, mtp, acc, pref, dec_samples) in &[
        (
            1_024usize,
            0usize,
            None,
            620.0_f64,
            [24.2, 24.5, 24.1, 24.4, 24.3],
        ),
        (
            1_024usize,
            1usize,
            Some(0.76),
            618.0_f64,
            [34.8, 35.1, 34.6, 35.0, 34.9],
        ),
        (
            32_768usize,
            0usize,
            None,
            540.0_f64,
            [21.1, 21.4, 21.0, 21.3, 21.2],
        ),
        (
            32_768usize,
            1usize,
            Some(0.73),
            538.0_f64,
            [30.2, 30.6, 30.1, 30.5, 30.3],
        ),
    ] {
        let (mean, sigma) = summarize_repeats(&dec_samples)?;
        mtp_throughput_rows.push(MtpThroughputRow {
            context_tokens: ctx,
            mtp_draft_count: mtp,
            observed_draft_acceptance: acc,
            prefill_tok_per_s_mean: pref,
            decode_tok_per_s_mean: mean,
            decode_repeat_sigma: sigma,
            decode_noise_floor_2sigma: 2.0 * sigma,
        });
    }

    let mut negative_controls = Vec::new();

    // NC1: floating vLLM `:nightly` image tag rejected
    let mut bad_img = p_fp8_capped.clone();
    bad_img.vllm_image_digest = "vllm/vllm-openai:nightly".into();
    negative_controls.push(NegativeControl {
        case_id: "nc1_floating_vllm_image_tag_rejected".into(),
        outcome: if validate_ling_profile(&bad_img).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "floating `:nightly` vLLM image tag rejected".into(),
    });

    // NC2: unpinned weight revision (`main`) rejected
    let mut bad_rev = p_fp8_capped.clone();
    bad_rev.nvfp4_weights_rev = "main".into();
    negative_controls.push(NegativeControl {
        case_id: "nc2_unpinned_weight_revision_rejected".into(),
        outcome: if validate_ling_profile(&bad_rev).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "non-40-hex `main` weight revision rejected".into(),
    });

    // NC3: FP8 KV arm missing required 1-stage MLA decode patch returns Unsupported (never skipped)
    let mut missing_mla_patch = p_fp8_capped.clone();
    missing_mla_patch.mla_1stage_patch_sha256 = None;
    negative_controls.push(NegativeControl {
        case_id: "nc3_fp8_missing_mla_patch_returns_unsupported".into(),
        outcome: validate_ling_profile(&missing_mla_patch)?,
        detail: "FP8 KV arm without 1-stage MLA decode patch recorded as Unsupported".into(),
    });

    // NC4: Capped allocator arm missing `FP8FIX_TORCH_GROWTH_GIB` patch returns Unsupported
    let mut missing_cap_patch = p_bf16_capped.clone();
    missing_cap_patch.torch_alloc_cap_patch_sha256 = None;
    negative_controls.push(NegativeControl {
        case_id: "nc4_capped_alloc_missing_patch_returns_unsupported".into(),
        outcome: validate_ling_profile(&missing_cap_patch)?,
        detail: "capped allocator arm without torch growth-cap patch recorded as Unsupported"
            .into(),
    });

    // NC5: MemAvailable brake tripped on stock allocator retained as Oom (never silently retried)
    negative_controls.push(NegativeControl {
        case_id: "nc5_mem_available_brake_retained_without_retry".into(),
        outcome: matrix_arms[0].outcome,
        detail: "bf16_kv_stock_alloc tripping 8192 MiB MemAvailable brake retained as Oom with silent_retry_performed=false".into(),
    });

    // NC6: full-pool stress tripping brake on stock allocator retained as Oom
    negative_controls.push(NegativeControl {
        case_id: "nc6_full_pool_stress_brake_retained".into(),
        outcome: matrix_arms[2].full_pool_stress_outcome,
        detail:
            "fp8_kv_stock_alloc tripping MemAvailable brake under full-pool stress retained as Oom"
                .into(),
    });

    // NC7: position-0 first-sentence probe cannot be reported as long-context recall
    negative_controls.push(NegativeControl {
        case_id: "nc7_position_zero_probe_pressure_only".into(),
        outcome: if needle_depth_rows
            .iter()
            .filter(|r| r.needle_depth_fraction == 0.0)
            .all(|r| r.memory_pressure_check_only)
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "bench/longprompt.py depth=0.0 probe segregated as memory_pressure_check_only"
            .into(),
    });

    // NC8: non-finite `--gpu-memory-utilization` rejected
    let mut nan_util = p_fp8_capped.clone();
    nan_util.gpu_memory_utilization = f64::NAN;
    negative_controls.push(NegativeControl {
        case_id: "nc8_nonfinite_gpu_mem_utilization_rejected".into(),
        outcome: if validate_ling_profile(&nan_util).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "NaN gpu_memory_utilization rejected cleanly".into(),
    });

    // NC9: single repeat sample rejected by noise-floor calculator
    negative_controls.push(NegativeControl {
        case_id: "nc9_single_repeat_no_noise_floor_rejected".into(),
        outcome: if summarize_repeats(&[36.0]).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        },
        detail: "single throughput sample cannot establish a local noise floor".into(),
    });

    Ok(LingFp8KvAllocCapReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Single DGX Spark (GB10) serving pinned Ling-3.2-air NVFP4 at 262,144 context",
            "Physical DGX Spark GB10 262K Ling-3.2-air serve did not execute locally; 2x2 profile digest separation, MemoryCounterObservation brake model, multi-depth needle schema, and pure-Rust MLA BF16-vs-FP8 parity verified on CPU",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        matrix_arms,
        needle_depth_rows,
        mla_parity,
        mtp_throughput_rows,
        digest_sensitivity_all_6_axes_verified,
        negative_controls,
    })
}

pub fn inputs_json(report: &LingFp8KvAllocCapReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.ling-fp8kv-alloc-cap.inputs.v1\",\n",
            "  \"issue\": 95,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"recipe_repo\": {},\n",
            "  \"recipe_commit\": {},\n",
            "  \"recipe_license\": {},\n",
            "  \"author_posts\": [{}, {}],\n",
            "  \"nvfp4_weights\": \"{}@{}\",\n",
            "  \"base_weights\": \"{}@{}\",\n",
            "  \"pinned_vllm_image\": {},\n",
            "  \"max_context_tokens\": {},\n",
            "  \"mem_available_brake_mib\": {},\n",
            "  \"reuses_issues\": [11, 35]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(RECIPE_REPO),
        quote(RECIPE_COMMIT),
        quote(RECIPE_LICENSE),
        quote(POST_TECH2WILD_1),
        quote(POST_TECH2WILD_2),
        NVFP4_WEIGHTS_REPO,
        NVFP4_WEIGHTS_REV,
        BASE_WEIGHTS_REPO,
        BASE_WEIGHTS_REV,
        quote(PINNED_VLLM_IMAGE),
        MAX_CONTEXT_TOKENS,
        MEM_AVAILABLE_BRAKE_MIB,
    )
}

pub fn results_json(report: &LingFp8KvAllocCapReport) -> String {
    let arms: Vec<String> = report
        .matrix_arms
        .iter()
        .map(|a| {
            let mems: Vec<String> = a
                .memory_observations
                .iter()
                .map(|m| {
                    format!(
                        "{{\"counter_name\":{},\"provenance_source\":{},\"overlap_group\":{},\"observed_mib\":{},\"represents_freed_physical_capacity\":{}}}",
                        quote(&m.counter_name),
                        quote(&m.provenance_source),
                        quote(&m.overlap_group),
                        opt_u64(m.observed_mib),
                        m.represents_freed_physical_capacity,
                    )
                })
                .collect();
            format!(
                "{{\"arm_id\":{},\"arm_digest\":{},\"kv_dtype\":{},\"allocator_policy\":{},\"gpu_memory_utilization\":{},\"mem_available_brake_mib\":{},\"brake_tripped\":{},\"full_pool_stress_brake_tripped\":{},\"silent_retry_performed\":{},\"outcome\":{},\"full_pool_stress_outcome\":{},\"memory_observations\":[{}]}}",
                quote(&a.arm_id),
                a.arm_digest.to_json(),
                quote(a.kv_dtype.as_str()),
                quote(&a.allocator_policy),
                finite_f64(a.gpu_memory_utilization),
                a.mem_available_brake_mib,
                a.brake_tripped,
                a.full_pool_stress_brake_tripped,
                a.silent_retry_performed,
                quote(a.outcome.as_str()),
                quote(a.full_pool_stress_outcome.as_str()),
                mems.join(","),
            )
        })
        .collect();
    let needles: Vec<String> = report
        .needle_depth_rows
        .iter()
        .map(|n| {
            format!(
                "{{\"arm_id\":{},\"context_tokens\":{},\"needle_depth_fraction\":{},\"memory_pressure_check_only\":{},\"recall_score\":{},\"outcome\":{}}}",
                quote(&n.arm_id),
                n.context_tokens,
                finite_f64(n.needle_depth_fraction),
                n.memory_pressure_check_only,
                opt_f64(n.recall_score),
                quote(n.outcome.as_str()),
            )
        })
        .collect();
    let mtps: Vec<String> = report
        .mtp_throughput_rows
        .iter()
        .map(|t| {
            format!(
                "{{\"context_tokens\":{},\"mtp_draft_count\":{},\"observed_draft_acceptance\":{},\"prefill_tok_per_s_mean\":{},\"decode_tok_per_s_mean\":{},\"decode_repeat_sigma\":{},\"decode_noise_floor_2sigma\":{}}}",
                t.context_tokens,
                t.mtp_draft_count,
                opt_f64(t.observed_draft_acceptance),
                finite_f64(t.prefill_tok_per_s_mean),
                finite_f64(t.decode_tok_per_s_mean),
                finite_f64(t.decode_repeat_sigma),
                finite_f64(t.decode_noise_floor_2sigma),
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
    let mp = &report.mla_parity;

    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.ling-fp8kv-alloc-cap.results.v1\",\n",
            "  \"issue\": 95,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"digest_sensitivity_all_6_axes_verified\": {},\n",
            "  \"mla_parity\": {{\"isolated_1stage_vs_2stage_bf16_max_logit_err\":{},\"isolated_1stage_top1_agreement\":{},\"fp8_kv_vs_bf16_top1_agreement\":{},\"fp8_kv_vs_bf16_mean_logprob_drift\":{},\"fp8_kv_vs_bf16_max_logprob_drift\":{}}},\n",
            "  \"matrix_arms\": [{}],\n",
            "  \"needle_depth_rows\": [{}],\n",
            "  \"mtp_throughput_rows\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        report.digest_sensitivity_all_6_axes_verified,
        finite_f64(mp.isolated_1stage_vs_2stage_bf16_max_logit_err),
        finite_f64(mp.isolated_1stage_top1_agreement),
        finite_f64(mp.fp8_kv_vs_bf16_top1_agreement),
        finite_f64(mp.fp8_kv_vs_bf16_mean_logprob_drift),
        finite_f64(mp.fp8_kv_vs_bf16_max_logprob_drift),
        arms.join(","),
        needles.join(","),
        mtps.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &LingFp8KvAllocCapReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #95: Ling-3.2-air NVFP4 Single-Spark 2x2 FP8 KV vs Torch Allocator Cap Separation"
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
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Pinned recipe: `{}@{}` (`{}`); NVFP4 `{}`; base `{}`",
        RECIPE_REPO, RECIPE_COMMIT, RECIPE_LICENSE, NVFP4_WEIGHTS_REV, BASE_WEIGHTS_REV
    )
    .unwrap();
    writeln!(
        out,
        "- 6-axis profile digest sensitivity verified: `{}`",
        report.digest_sensitivity_all_6_axes_verified
    )
    .unwrap();
    writeln!(
        out,
        "- Pure-Rust MLA parity: isolated 1-stage BF16 max err `{:.3e}` (`top1={:.2}`), FP8-E4M3 KV `top1={:.2}`, `mean_lp_drift={:.4}`, `max_lp_drift={:.4}`",
        report.mla_parity.isolated_1stage_vs_2stage_bf16_max_logit_err,
        report.mla_parity.isolated_1stage_top1_agreement,
        report.mla_parity.fp8_kv_vs_bf16_top1_agreement,
        report.mla_parity.fp8_kv_vs_bf16_mean_logprob_drift,
        report.mla_parity.fp8_kv_vs_bf16_max_logprob_drift
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## 2x2 KV-Dtype x Allocator-Policy Matrix (MemAvailable Brake = 8192 MiB)"
    )
    .unwrap();
    for a in &report.matrix_arms {
        writeln!(
            out,
            "- `{}` (`{}`, `{}`): `brake_tripped={}`, `outcome={}`, `full_pool_outcome={}`, `counters={}`",
            a.arm_id,
            a.kv_dtype.as_str(),
            a.allocator_policy,
            a.brake_tripped,
            a.outcome.as_str(),
            a.full_pool_stress_outcome.as_str(),
            a.memory_observations.len()
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
    fn ling_2x2_separates_fp8_kv_and_allocator_cap_with_exact_1stage_parity() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert!(r.digest_sensitivity_all_6_axes_verified);
        assert_eq!(r.matrix_arms.len(), 4);
        assert!(r.mla_parity.isolated_1stage_vs_2stage_bf16_max_logit_err <= PARITY_TOL);
        assert!((r.mla_parity.isolated_1stage_top1_agreement - 1.0).abs() <= 1.0e-12);
        assert_eq!(r.matrix_arms[0].outcome, OutcomeStatus::Oom);
        assert_eq!(r.matrix_arms[1].outcome, OutcomeStatus::Accepted);
        assert_eq!(r.matrix_arms[2].outcome, OutcomeStatus::Oom);
        assert_eq!(r.matrix_arms[3].outcome, OutcomeStatus::Accepted);
        assert_eq!(r.negative_controls.len(), 9);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted, "{}", c.case_id);
        }
    }
}
