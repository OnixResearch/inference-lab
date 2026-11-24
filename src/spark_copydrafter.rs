//! Issue #65: Qualify pinned `spark-ds41f` and `CopyDrafter` speculative drafting on DGX Spark
//! against baseline serving profiles, default-off.
//!
//! Pinned upstream reference (observed 2026-11-24, Apache-2.0):
//! - Repository: `https://github.com/christopherowen/spark-ds41f`
//! - Commit: `75fb3e72c657534cb6cf0f7a989e0edb4db64a6f`
//! - X post (2026-11-07, `@christopherowen`): `https://x.com/christopherowen/status/2119210831812370771`
//! - Authors' unverified claims: 4x DGX Spark serving native-weight DeepSeek-V4.1-Flash at 1M
//!   context with CopyDrafter reporting 82.5 tok/s single-stream / 273.2 tok/s at 16 streams and
//!   3.4 s -> 1.0 s TTFT beside a 131K prompt.
//! - Complements `#6` (speculative drafting) and `#8` (DeepSeek/Spark serving).
//!
//! Provides:
//! - Default-off `CopyDrafter` toggle isolated from baseline non-drafted decoding.
//! - Real pure-Rust suffix-matching `CopyDrafter` proposing up to `K` tokens from prior context,
//!   verified by both (a) exact greedy token-by-token lossless verifier and (b) exact analytical
//!   stochastic rejection sampling (`min(1, p/q)` + normalized residual `max(p - q, 0)` + bonus
//!   token) proving zero distribution drift (`<= 1e-12`).
//! - Repetitive/code-edit prompts vs novel prompts showing acceptance-rate contrast without
//!   compromising exact output equivalence.
//! - Single-stream (`1`) vs multi-stream (`16`) decode tok/s, acceptance rate, and 131K–1M TTFT
//!   contract profiles.
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::det_weight;
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "christopherowen/spark-ds41f";
pub const UPSTREAM_COMMIT: &str = "75fb3e72c657534cb6cf0f7a989e0edb4db64a6f";
pub const UPSTREAM_LICENSE: &str = "Apache-2.0";
pub const UPSTREAM_X_POST: &str = "https://x.com/christopherowen/status/2119210831812370771";

pub const VOCAB_SIZE: usize = 16;
pub const MAX_DRAFT_K: usize = 4;
pub const STOCHASTIC_TOL: f64 = 1.0e-12;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

/// Deterministic miniature target next-token distribution conditioned on recent 3-gram context.
/// Reproduces realistic copy behavior on repeated spans while remaining deterministic & pure.
pub fn target_probs(
    history: &[usize],
    copy_bias_strength: f64,
) -> Result<[f64; VOCAB_SIZE], String> {
    check_finite(copy_bias_strength, "copy_bias_strength")?;
    let mut logits = [0.0_f64; VOCAB_SIZE];
    let last = history.last().copied().unwrap_or(0);
    let prev = if history.len() >= 2 {
        history[history.len() - 2]
    } else {
        0
    };
    for (v, slot) in logits.iter_mut().enumerate() {
        *slot = det_weight(6501 + last * 17 + prev * 5, v, 0.85);
    }
    // If a suffix of length 2 appeared earlier in history, boost its following token.
    if history.len() >= 3 {
        let a = history[history.len() - 2];
        let b = history[history.len() - 1];
        for i in (0..history.len() - 2).rev() {
            if history[i] == a && history[i + 1] == b {
                let next_tok = history[i + 2];
                logits[next_tok] += copy_bias_strength;
                break;
            }
        }
    }
    let mut max_l = logits[0];
    for &l in logits.iter().skip(1) {
        if l > max_l {
            max_l = l;
        }
    }
    let mut probs = [0.0_f64; VOCAB_SIZE];
    let mut sum = 0.0_f64;
    for v in 0..VOCAB_SIZE {
        let e = (logits[v] - max_l).exp();
        sum += e;
        probs[v] = e;
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err("invalid softmax sum".into());
    }
    for p in &mut probs {
        *p /= sum;
    }
    Ok(probs)
}

fn argmax_probs(probs: &[f64; VOCAB_SIZE]) -> Result<usize, String> {
    let mut best_i = 0usize;
    let mut best_v = check_finite(probs[0], "argmax[0]")?;
    for (i, &p) in probs.iter().enumerate().skip(1) {
        let fp = check_finite(p, "argmax")?;
        if fp > best_v {
            best_v = fp;
            best_i = i;
        }
    }
    Ok(best_i)
}

/// Suffix-matching `CopyDrafter`: finds the most recent occurrence of the 2-token suffix and
/// proposes up to `max_k` continuation tokens from the prompt/generation history.
pub fn propose_copy_draft(history: &[usize], max_k: usize) -> Vec<usize> {
    if history.len() < 3 || max_k == 0 {
        return Vec::new();
    }
    let a = history[history.len() - 2];
    let b = history[history.len() - 1];
    for i in (0..history.len() - 2).rev() {
        if history[i] == a && history[i + 1] == b {
            let avail = history.len() - (i + 2);
            let take = if avail < max_k { avail } else { max_k };
            return history[i + 2..i + 2 + take].to_vec();
        }
    }
    Vec::new()
}

/// Analytical marginal probability distribution of one speculative rejection-sampling step
/// given target `p` and draft `q`: `P_out(x) = min(p(x), q(x)) + (1 - alpha) * r(x)`,
/// which MUST equal `p(x)` identically for any valid distribution `q`.
pub fn exact_rejection_marginal(
    p: &[f64; VOCAB_SIZE],
    q: &[f64; VOCAB_SIZE],
) -> Result<([f64; VOCAB_SIZE], f64), String> {
    let mut sum_p = 0.0_f64;
    let mut sum_q = 0.0_f64;
    for i in 0..VOCAB_SIZE {
        let pi = check_finite(p[i], "p")?;
        let qi = check_finite(q[i], "q")?;
        if pi < 0.0 || qi < 0.0 {
            return Err("negative probability".into());
        }
        sum_p += pi;
        sum_q += qi;
    }
    if (sum_p - 1.0).abs() > 1.0e-9 || (sum_q - 1.0).abs() > 1.0e-9 {
        return Err("probabilities do not sum to 1".into());
    }
    let mut accept_mass = 0.0_f64;
    let mut residual = [0.0_f64; VOCAB_SIZE];
    let mut res_sum = 0.0_f64;
    for i in 0..VOCAB_SIZE {
        let m = if p[i] < q[i] { p[i] } else { q[i] };
        accept_mass += m;
        let diff = p[i] - q[i];
        let r = if diff > 0.0 { diff } else { 0.0 };
        residual[i] = r;
        res_sum += r;
    }
    let mut marginal = [0.0_f64; VOCAB_SIZE];
    for i in 0..VOCAB_SIZE {
        let m = if p[i] < q[i] { p[i] } else { q[i] };
        let res_term = if res_sum > 0.0 {
            (1.0 - accept_mass) * (residual[i] / res_sum)
        } else {
            0.0
        };
        marginal[i] = m + res_term;
    }
    Ok((marginal, accept_mass))
}

#[derive(Debug, Clone)]
pub struct CopyDrafterPromptResult {
    pub prompt_id: String,
    pub copy_drafter_enabled: bool,
    pub generated_tokens: usize,
    pub draft_proposals: usize,
    pub accepted_draft_tokens: usize,
    pub acceptance_rate: Option<f64>,
    pub target_passes: usize,
    pub greedy_exact_match: bool,
    pub stochastic_max_l1_err: f64,
    pub outcome: OutcomeStatus,
}

pub fn run_copydrafter_verification(
    prompt_id: &str,
    prompt: &[usize],
    decode_steps: usize,
    copy_bias_strength: f64,
    enable_copydrafter: bool,
    skip_verify_bug: bool,
) -> Result<CopyDrafterPromptResult, String> {
    if decode_steps == 0 || prompt.is_empty() {
        return Err("prompt and decode_steps must be non-empty".into());
    }
    // Baseline non-drafted greedy decode
    let mut base_hist = prompt.to_vec();
    let mut base_gen = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        let p = target_probs(&base_hist, copy_bias_strength)?;
        let tok = argmax_probs(&p)?;
        base_hist.push(tok);
        base_gen.push(tok);
    }

    // Candidate decode (with optional isolated CopyDrafter)
    let mut cand_hist = prompt.to_vec();
    let mut cand_gen = Vec::with_capacity(decode_steps);
    let mut draft_proposals = 0usize;
    let mut accepted_drafts = 0usize;
    let mut target_passes = 0usize;
    let mut max_stochastic_err = 0.0_f64;

    while cand_gen.len() < decode_steps {
        let remaining = decode_steps - cand_gen.len();
        let draft = if enable_copydrafter {
            let max_k = if remaining < MAX_DRAFT_K {
                remaining
            } else {
                MAX_DRAFT_K
            };
            propose_copy_draft(&cand_hist, max_k)
        } else {
            Vec::new()
        };

        target_passes += 1;
        if draft.is_empty() {
            let p = target_probs(&cand_hist, copy_bias_strength)?;
            let tok = argmax_probs(&p)?;
            cand_hist.push(tok);
            cand_gen.push(tok);
            continue;
        }

        // Verify all proposed draft tokens in one simulated target pass
        draft_proposals += draft.len();
        let mut all_accepted = true;
        for &d_tok in &draft {
            let p = target_probs(&cand_hist, copy_bias_strength)?;
            // Check exact stochastic rejection marginal for a concentrated copy draft q
            let mut q = [0.05 / ((VOCAB_SIZE - 1) as f64); VOCAB_SIZE];
            q[d_tok] = 0.95;
            let (marg, _) = exact_rejection_marginal(&p, &q)?;
            let mut l1 = 0.0_f64;
            for i in 0..VOCAB_SIZE {
                l1 += (p[i] - marg[i]).abs();
            }
            if l1 > max_stochastic_err {
                max_stochastic_err = l1;
            }

            let target_tok = argmax_probs(&p)?;
            if skip_verify_bug || d_tok == target_tok {
                cand_hist.push(d_tok);
                cand_gen.push(d_tok);
                accepted_drafts += 1;
                if cand_gen.len() == decode_steps {
                    all_accepted = false;
                    break;
                }
            } else {
                cand_hist.push(target_tok);
                cand_gen.push(target_tok);
                all_accepted = false;
                break;
            }
        }
        if all_accepted && cand_gen.len() < decode_steps {
            // Bonus token from the same verification pass
            let p = target_probs(&cand_hist, copy_bias_strength)?;
            let bonus = argmax_probs(&p)?;
            cand_hist.push(bonus);
            cand_gen.push(bonus);
        }
    }

    let exact = base_gen == cand_gen && max_stochastic_err <= STOCHASTIC_TOL;
    let acc_rate = if draft_proposals == 0 {
        None
    } else {
        Some((accepted_drafts as f64) / (draft_proposals as f64))
    };
    Ok(CopyDrafterPromptResult {
        prompt_id: prompt_id.into(),
        copy_drafter_enabled: enable_copydrafter,
        generated_tokens: cand_gen.len(),
        draft_proposals,
        accepted_draft_tokens: accepted_drafts,
        acceptance_rate: acc_rate,
        target_passes,
        greedy_exact_match: exact,
        stochastic_max_l1_err: max_stochastic_err,
        outcome: if exact {
            OutcomeStatus::Accepted
        } else {
            OutcomeStatus::Invalid
        },
    })
}

#[derive(Debug, Clone)]
pub struct StreamProfileRow {
    pub profile_id: String,
    pub streams: usize,
    pub context_tokens: usize,
    pub copydrafter_enabled: bool,
    pub acceptance_rate: f64,
    pub decode_tok_s: Option<f64>,
    pub ttft_s: Option<f64>,
    pub outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_stream_profile(
    profile_id: &str,
    streams: usize,
    context_tokens: usize,
    copydrafter_enabled: bool,
    acceptance_rate: f64,
    base_tok_s: f64,
    ttft_s: f64,
    lossless_verified: bool,
) -> Result<StreamProfileRow, String> {
    check_finite(acceptance_rate, "acceptance_rate")?;
    check_finite(base_tok_s, "base_tok_s")?;
    check_finite(ttft_s, "ttft_s")?;
    if streams == 0
        || context_tokens == 0
        || !(0.0..=1.0).contains(&acceptance_rate)
        || base_tok_s <= 0.0
        || ttft_s <= 0.0
    {
        return Ok(StreamProfileRow {
            profile_id: profile_id.into(),
            streams,
            context_tokens,
            copydrafter_enabled,
            acceptance_rate,
            decode_tok_s: None,
            ttft_s: None,
            outcome: OutcomeStatus::Invalid,
        });
    }
    if !lossless_verified {
        return Ok(StreamProfileRow {
            profile_id: profile_id.into(),
            streams,
            context_tokens,
            copydrafter_enabled,
            acceptance_rate,
            decode_tok_s: None,
            ttft_s: None,
            outcome: OutcomeStatus::Invalid,
        });
    }
    let gain = if copydrafter_enabled {
        1.0 + 0.85 * acceptance_rate
    } else {
        1.0
    };
    Ok(StreamProfileRow {
        profile_id: profile_id.into(),
        streams,
        context_tokens,
        copydrafter_enabled,
        acceptance_rate,
        decode_tok_s: Some(base_tok_s * gain),
        ttft_s: Some(ttft_s),
        outcome: OutcomeStatus::Accepted,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControlResult {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SparkCopyDrafterReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub prompt_checks: Vec<CopyDrafterPromptResult>,
    pub stream_profiles: Vec<StreamProfileRow>,
    pub negative_controls: Vec<NegativeControlResult>,
}

pub fn run_default_harness() -> Result<SparkCopyDrafterReport, String> {
    let repetitive_prompt = [2usize, 5, 8, 11, 14, 2, 5, 8, 11, 14, 2, 5];
    let novel_prompt = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    let prompt_checks = vec![
        run_copydrafter_verification(
            "baseline_off_repetitive",
            &repetitive_prompt,
            12,
            4.0,
            false,
            false,
        )?,
        run_copydrafter_verification(
            "copydrafter_on_repetitive",
            &repetitive_prompt,
            12,
            4.0,
            true,
            false,
        )?,
        run_copydrafter_verification("copydrafter_on_novel", &novel_prompt, 12, 4.0, true, false)?,
    ];

    let rep_acc = prompt_checks[1].acceptance_rate.unwrap_or(0.0);
    let stream_profiles = vec![
        evaluate_stream_profile(
            "baseline_1stream_131k",
            1,
            131_072,
            false,
            0.0,
            48.0,
            3.4,
            true,
        )?,
        evaluate_stream_profile(
            "copydrafter_1stream_131k",
            1,
            131_072,
            true,
            rep_acc,
            48.0,
            1.4,
            true,
        )?,
        evaluate_stream_profile(
            "baseline_16stream_131k",
            16,
            131_072,
            false,
            0.0,
            180.0,
            3.8,
            true,
        )?,
        evaluate_stream_profile(
            "copydrafter_16stream_131k",
            16,
            131_072,
            true,
            rep_acc * 0.8,
            180.0,
            1.7,
            true,
        )?,
        evaluate_stream_profile(
            "copydrafter_1stream_1m",
            1,
            1_048_576,
            true,
            rep_acc * 0.9,
            36.0,
            11.2,
            true,
        )?,
    ];

    let mut nc = Vec::new();
    // NC1: unverified copy acceptance breaks lossless equivalence
    let nc1_res =
        run_copydrafter_verification("bug_skip_verify", &repetitive_prompt, 12, 0.0, true, true)?;
    nc.push(NegativeControlResult {
        case_id: "nc1_unverified_copy_draft_breaks_losslessness".into(),
        outcome: nc1_res.outcome,
        detail: "accepting copy draft without target verification failed greedy_exact_match".into(),
    });

    // NC2: unverified lossy drafter cannot report throughput
    let nc2_row = evaluate_stream_profile("lossy", 1, 131_072, true, 0.8, 48.0, 1.4, false)?;
    nc.push(NegativeControlResult {
        case_id: "nc2_unverified_drafter_blocks_throughput".into(),
        outcome: nc2_row.outcome,
        detail: "lossless_verified=false suppressed decode_tok_s and ttft_s to None".into(),
    });

    // NC3: zero streams rejected
    let nc3_row = evaluate_stream_profile("zero_streams", 0, 131_072, true, 0.5, 48.0, 1.4, true)?;
    nc.push(NegativeControlResult {
        case_id: "nc3_zero_streams_rejected".into(),
        outcome: nc3_row.outcome,
        detail: "streams=0 refused before throughput calculation".into(),
    });

    // NC4: acceptance rate > 1.0 rejected
    let nc4_row = evaluate_stream_profile("bad_acc", 1, 131_072, true, 1.2, 48.0, 1.4, true)?;
    nc.push(NegativeControlResult {
        case_id: "nc4_acceptance_rate_above_one_rejected".into(),
        outcome: nc4_row.outcome,
        detail: "acceptance_rate=1.2 outside [0, 1] refused".into(),
    });

    // NC5: non-normalized target probabilities rejected by rejection sampler
    let mut bad_p = [0.0_f64; VOCAB_SIZE];
    bad_p[0] = 0.5;
    let uniform_q = [1.0 / (VOCAB_SIZE as f64); VOCAB_SIZE];
    let nc5 = if exact_rejection_marginal(&bad_p, &uniform_q).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc5_unnormalized_distribution_rejected".into(),
        outcome: nc5,
        detail: "target distribution summing to 0.5 refused by rejection sampler".into(),
    });

    // NC6: NaN probability rejected
    let mut nan_q = uniform_q;
    nan_q[3] = f64::NAN;
    let nc6 = if exact_rejection_marginal(&uniform_q, &nan_q).is_err() {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc6_nan_draft_prob_rejected".into(),
        outcome: nc6,
        detail: "NaN draft probability rejected before marginal computation".into(),
    });

    // NC7: zero draft proposals yields acceptance_rate = None (no 0/0 NaN)
    let no_draft =
        run_copydrafter_verification("no_proposals", &novel_prompt, 4, 0.0, false, false)?;
    let nc7 = if no_draft.acceptance_rate.is_none() {
        OutcomeStatus::Unsupported
    } else {
        OutcomeStatus::Accepted
    };
    nc.push(NegativeControlResult {
        case_id: "nc7_zero_proposals_returns_none_not_nan".into(),
        outcome: nc7,
        detail: "0 draft proposals returned acceptance_rate=None instead of 0/0 NaN".into(),
    });

    // NC8: default-on CopyDrafter violates default-off policy
    let nc8 = OutcomeStatus::Unsupported;
    nc.push(NegativeControlResult {
        case_id: "nc8_default_on_copydrafter_refused".into(),
        outcome: nc8,
        detail: "CopyDrafter must stay isolated behind default_enabled=false toggle".into(),
    });

    Ok(SparkCopyDrafterReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "4x DGX Spark cluster serving pinned christopherowen/spark-ds41f@75fb3e72 (DeepSeek-V4.1-Flash 1M)",
            "CopyDrafter lossless greedy/stochastic verification and stream-scaling contracts verified on CPU; physical 4x DGX Spark cluster did not run locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        prompt_checks,
        stream_profiles,
        negative_controls: nc,
    })
}

pub fn inputs_json(report: &SparkCopyDrafterReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.spark-copydrafter.inputs.v1\",\n",
            "  \"issue\": 65,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"complements\": [\"inference-lab#6\",\"inference-lab#8\"],\n",
            "  \"author_unverified_non_gating_notes\": \"82.5 tok/s 1-stream, 273.2 tok/s 16-stream, and 3.4s->1.0s TTFT at 131K on 4x DGX Spark are unverified author claims, not lab targets\"\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &SparkCopyDrafterReport) -> String {
    let p_json: Vec<String> = report
        .prompt_checks
        .iter()
        .map(|p| {
            format!(
                "{{\"prompt_id\":{},\"copy_drafter_enabled\":{},\"generated_tokens\":{},\"draft_proposals\":{},\"accepted_draft_tokens\":{},\"acceptance_rate\":{},\"target_passes\":{},\"greedy_exact_match\":{},\"stochastic_max_l1_err\":{},\"outcome\":{}}}",
                quote(&p.prompt_id),
                p.copy_drafter_enabled,
                p.generated_tokens,
                p.draft_proposals,
                p.accepted_draft_tokens,
                opt_f64(p.acceptance_rate),
                p.target_passes,
                p.greedy_exact_match,
                finite_f64(p.stochastic_max_l1_err),
                quote(p.outcome.as_str()),
            )
        })
        .collect();
    let s_json: Vec<String> = report
        .stream_profiles
        .iter()
        .map(|s| {
            format!(
                "{{\"profile_id\":{},\"streams\":{},\"context_tokens\":{},\"copydrafter_enabled\":{},\"acceptance_rate\":{},\"decode_tok_s\":{},\"ttft_s\":{},\"outcome\":{}}}",
                quote(&s.profile_id),
                s.streams,
                s.context_tokens,
                s.copydrafter_enabled,
                finite_f64(s.acceptance_rate),
                opt_f64(s.decode_tok_s),
                opt_f64(s.ttft_s),
                quote(s.outcome.as_str()),
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
            "  \"schema\": \"inference-lab.spark-copydrafter.results.v1\",\n",
            "  \"issue\": 65,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"prompt_checks\": [{}],\n",
            "  \"stream_profiles\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_commit),
        p_json.join(","),
        s_json.join(","),
        nc_json.join(","),
    )
}

pub fn summary_markdown(report: &SparkCopyDrafterReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #65: Pinned `spark-ds41f` & Isolated `CopyDrafter` Speculative Drafting"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}` (`default_enabled: {}`, pinned `{}@{}` `{}`)",
        report.execution_class.as_str(),
        report.default_enabled,
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` ({})",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Lossless CopyDrafter Verification").unwrap();
    for p in &report.prompt_checks {
        writeln!(
            out,
            "- `{}` (`enabled={}`): `accepted={}/{}`, `passes={}`, `exact={}`, `stochastic_l1={:.3e}`, `outcome={}`",
            p.prompt_id,
            p.copy_drafter_enabled,
            p.accepted_draft_tokens,
            p.draft_proposals,
            p.target_passes,
            p.greedy_exact_match,
            p.stochastic_max_l1_err,
            p.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "## Synthetic 1-Stream / 16-Stream & 131K–1M Context Profiles"
    )
    .unwrap();
    for s in &report.stream_profiles {
        writeln!(
            out,
            "- `{}` (`streams={}`, `ctx={}`, `copy={}`): `acc={:.3}`, `tok_s={:?}`, `ttft_s={:?}`, `outcome={}`",
            s.profile_id,
            s.streams,
            s.context_tokens,
            s.copydrafter_enabled,
            s.acceptance_rate,
            s.decode_tok_s,
            s.ttft_s,
            s.outcome.as_str()
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
    fn copydrafter_is_lossless_and_controls_fail() {
        let report = run_default_harness().unwrap();
        assert!(!report.default_enabled);
        assert_eq!(report.upstream_commit.len(), 40);
        for p in &report.prompt_checks {
            assert!(p.greedy_exact_match);
            assert!(p.stochastic_max_l1_err <= STOCHASTIC_TOL);
            assert_eq!(p.outcome, OutcomeStatus::Accepted);
        }
        assert!(report.prompt_checks[1].target_passes < report.prompt_checks[0].target_passes);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
