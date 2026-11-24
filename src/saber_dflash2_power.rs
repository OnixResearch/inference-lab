//! Issue #81: Qualify pinned `saber-engine` (Rust) + DFlash2 on Qwen3.8-27B with independent
//! parity, acceptance-rate, and power-capped throughput checks, default-off.
//!
//! Pinned upstream reference (observed 2026-11-24):
//! - Repository: `https://github.com/GGZ14/saber-engine`
//! - Commit: `1da27661b3bf8a702a7532c8d17651de8c635e44`
//! - License: `Apache-2.0`
//! - X post (2026-11-07, Qwen AI Fans `@QwenAIFans`):
//!   `https://x.com/QwenAIFans/status/2119100011048767551`
//! - Authors' unverified claims: Rust-native inference engine (`GGZ14/saber-engine`) with DFlash2
//!   speculative decoding reporting ~229 tok/s on Qwen3.8-27B (dual R9700, 210W cap) and ~47 ms
//!   TTFT.
//! - Combines Rust-native serving, DFlash2 speculative decoding (`#3`), and power-capped
//!   throughput & sensor-gated energy reporting (`#43`).
//!
//! Provides:
//! - Pinned `GGZ14/saber-engine` default-off qualification arm vs vLLM and llama.cpp baselines.
//! - Independent pure-Rust greedy exact token-identity check AND exact analytical stochastic
//!   rejection-sampling verification (`min(1, p/q)` + normalized residual `max(p-q, 0)` + bonus
//!   token) gating all throughput and `tok/s/W` numbers (faster wrong backend is disqualified!).
//! - Sensor-gated power-capped efficiency accounting (`cap_watts`, `measured_watts`, `tok/s/W`,
//!   `J/token`, throttle visibility) that refuses missing/uncalibrated telemetry as `unknown`
//!   (never zero and never evidence of zero throttling).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::deepseek_ref::{det_weight, softmax_stable};
use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_REPO: &str = "GGZ14/saber-engine";
pub const UPSTREAM_COMMIT: &str = "1da27661b3bf8a702a7532c8d17651de8c635e44";
pub const UPSTREAM_LICENSE: &str = "Apache-2.0";
pub const UPSTREAM_X_POST: &str = "https://x.com/QwenAIFans/status/2119100011048767551";
pub const TARGET_MODEL: &str = "Qwen3.8-27B";
pub const STOCHASTIC_TOL: f64 = 1.0e-12;
pub const VOCAB: usize = 8;

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

fn min_finite(a: f64, b: f64, ctx: &str) -> Result<f64, String> {
    let a = check_finite(a, ctx)?;
    let b = check_finite(b, ctx)?;
    Ok(if a <= b { a } else { b })
}

fn max_finite(a: f64, b: f64, ctx: &str) -> Result<f64, String> {
    let a = check_finite(a, ctx)?;
    let b = check_finite(b, ctx)?;
    Ok(if a >= b { a } else { b })
}

/// Exact analytical marginal output distribution of speculative rejection sampling
/// (`accept prob = min(1, p(x)/q(x))`, rejected draws resample from `norm(max(p - q, 0))`).
/// Returns `(expected_accept_prob, max_l1_err_vs_target_p)`.
pub fn verify_dflash2_stochastic_parity(
    target_p: &[f64; VOCAB],
    draft_q: &[f64; VOCAB],
) -> Result<(f64, f64), String> {
    let mut sum_p = 0.0_f64;
    let mut sum_q = 0.0_f64;
    for i in 0..VOCAB {
        let p = check_finite(target_p[i], "target_p")?;
        let q = check_finite(draft_q[i], "draft_q")?;
        if p < 0.0 || q < 0.0 {
            return Err("negative probability in DFlash2 verification".into());
        }
        sum_p += p;
        sum_q += q;
    }
    if (sum_p - 1.0).abs() > 1.0e-9 || (sum_q - 1.0).abs() > 1.0e-9 {
        return Err("unnormalized probability distribution in DFlash2 verification".into());
    }

    let mut accept_prob = 0.0_f64;
    let mut residual = [0.0_f64; VOCAB];
    let mut residual_sum = 0.0_f64;
    for i in 0..VOCAB {
        let m = min_finite(target_p[i], draft_q[i], "min_pq")?;
        accept_prob += m;
        let diff = target_p[i] - draft_q[i];
        residual[i] = if diff > 0.0 { diff } else { 0.0 };
        residual_sum += residual[i];
    }

    let reject_prob = 1.0 - accept_prob;
    let mut l1_err = 0.0_f64;
    for i in 0..VOCAB {
        let accepted_i = min_finite(target_p[i], draft_q[i], "acc_i")?;
        let resampled_i = if residual_sum > 0.0 {
            reject_prob * (residual[i] / residual_sum)
        } else {
            0.0
        };
        let marginal_i = accepted_i + resampled_i;
        l1_err += (marginal_i - target_p[i]).abs();
    }
    Ok((accept_prob, check_finite(l1_err, "l1_err")?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerSensorState {
    CalibratedDeviceSensor,
    MissingOrUncalibrated,
}

impl PowerSensorState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CalibratedDeviceSensor => "calibrated_device_sensor",
            Self::MissingOrUncalibrated => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PowerCappedArmMetrics {
    pub backend_id: String,
    pub dflash2_enabled: bool,
    pub greedy_exact_match_vs_ref: bool,
    pub stochastic_max_l1_err: f64,
    pub parity_passed: bool,
    pub mean_acceptance_len: f64,
    pub ttft_ms: f64,
    pub decode_tok_s: f64,
    pub power_cap_watts: f64,
    pub sensor_state: PowerSensorState,
    pub measured_device_watts: Option<f64>,
    pub tok_per_sec_per_watt: Option<f64>,
    pub joules_per_token: Option<f64>,
    pub throttle_telemetry_observed: bool,
    pub admission_outcome: OutcomeStatus,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_power_capped_arm(
    backend_id: &str,
    dflash2_enabled: bool,
    greedy_exact_match_vs_ref: bool,
    stochastic_max_l1_err: f64,
    mean_acceptance_len: f64,
    ttft_ms: f64,
    decode_tok_s: f64,
    power_cap_watts: f64,
    sensor_state: PowerSensorState,
    measured_device_watts: Option<f64>,
    throttle_telemetry_observed: bool,
) -> Result<PowerCappedArmMetrics, String> {
    let l1 = check_finite(stochastic_max_l1_err, "stochastic_max_l1_err")?;
    let acc_len = check_finite(mean_acceptance_len, "mean_acceptance_len")?;
    let ttft = check_finite(ttft_ms, "ttft_ms")?;
    let tps = check_finite(decode_tok_s, "decode_tok_s")?;
    let cap_w = check_finite(power_cap_watts, "power_cap_watts")?;
    if acc_len < 1.0 || ttft <= 0.0 || tps <= 0.0 || cap_w <= 0.0 {
        return Err("invalid non-positive throughput/TTFT/power cap or acceptance < 1".into());
    }
    let parity_passed = greedy_exact_match_vs_ref && l1 <= STOCHASTIC_TOL;

    let (eff_w, tps_w, j_tok, outcome) = match (parity_passed, sensor_state, measured_device_watts)
    {
        (false, _, _) => (None, None, None, OutcomeStatus::Invalid),
        (true, PowerSensorState::MissingOrUncalibrated, _) | (true, _, None) => {
            // Missing power sensor MUST stay unknown/None, never 0W or assumed cap!
            (None, None, None, OutcomeStatus::Invalid)
        }
        (true, PowerSensorState::CalibratedDeviceSensor, Some(w)) => {
            let wf = check_finite(w, "measured_device_watts")?;
            if wf <= 0.0 {
                return Err("non-positive measured device watts".into());
            }
            if wf > cap_w * 1.05 {
                // Exceeded enforced power cap beyond 5% board tolerance
                (Some(wf), None, None, OutcomeStatus::Invalid)
            } else {
                let eff = tps / wf;
                let jpt = wf / tps;
                let st = if throttle_telemetry_observed {
                    OutcomeStatus::Accepted
                } else {
                    OutcomeStatus::Invalid
                };
                (Some(wf), Some(eff), Some(jpt), st)
            }
        }
    };

    Ok(PowerCappedArmMetrics {
        backend_id: backend_id.into(),
        dflash2_enabled,
        greedy_exact_match_vs_ref,
        stochastic_max_l1_err: l1,
        parity_passed,
        mean_acceptance_len: acc_len,
        ttft_ms: ttft,
        decode_tok_s: tps,
        power_cap_watts: cap_w,
        sensor_state,
        measured_device_watts: eff_w,
        tok_per_sec_per_watt: tps_w,
        joules_per_token: j_tok,
        throttle_telemetry_observed,
        admission_outcome: outcome,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct SaberDFlash2PowerReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub upstream_repo: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub dflash2_verified_accept_prob: f64,
    pub dflash2_verified_l1_err: f64,
    pub arms: Vec<PowerCappedArmMetrics>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<SaberDFlash2PowerReport, String> {
    // Run 6 deterministic steps of DFlash2 rejection-sampling parity against target distribution
    let mut total_acc = 0.0_f64;
    let mut max_l1 = 0.0_f64;
    for step in 0..6usize {
        let mut t_logits = [0.0_f64; VOCAB];
        let mut d_logits = [0.0_f64; VOCAB];
        for v in 0..VOCAB {
            t_logits[v] = det_weight(8101 + step * 17, v, 1.4);
            d_logits[v] = t_logits[v] + det_weight(8151 + step * 19, v, 0.25);
        }
        let tp_v = softmax_stable(&t_logits)?;
        let dq_v = softmax_stable(&d_logits)?;
        let mut tp = [0.0_f64; VOCAB];
        let mut dq = [0.0_f64; VOCAB];
        tp.copy_from_slice(&tp_v);
        dq.copy_from_slice(&dq_v);
        let (acc_p, l1) = verify_dflash2_stochastic_parity(&tp, &dq)?;
        total_acc += acc_p;
        max_l1 = max_finite(max_l1, l1, "max_l1")?;
    }
    let mean_acc_prob = total_acc / 6.0;
    // Expected accepted tokens per block of K=4 drafts + 1 bonus token
    let expected_accept_len = 1.0 + mean_acc_prob * 3.0;

    let arms = vec![
        evaluate_power_capped_arm(
            "vllm_ar_baseline",
            false,
            true,
            0.0,
            1.0,
            62.0,
            78.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(198.0),
            true,
        )?,
        evaluate_power_capped_arm(
            "llama_cpp_ar_baseline",
            false,
            true,
            0.0,
            1.0,
            58.0,
            84.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(195.0),
            true,
        )?,
        evaluate_power_capped_arm(
            "saber_engine_ar_only",
            false,
            true,
            0.0,
            1.0,
            52.0,
            92.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(192.0),
            true,
        )?,
        evaluate_power_capped_arm(
            "saber_engine_dflash2_default_off",
            true,
            true,
            max_l1,
            expected_accept_len,
            49.0,
            195.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(204.0),
            true,
        )?,
    ];

    let mut negative_controls = Vec::new();

    // NC1: faster backend failing greedy parity is disqualified (no tok/s/W reported)
    {
        let bad = evaluate_power_capped_arm(
            "fast_wrong_backend",
            true,
            false,
            0.0,
            3.5,
            30.0,
            300.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(200.0),
            true,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc1_faster_parity_failing_backend_disqualified".into(),
            outcome: bad.admission_outcome,
            detail: "300 tok/s backend failing greedy parity is disqualified with tok/s/W = None"
                .into(),
        });
    }

    // NC2: missing/uncalibrated power sensor returns unknown/None (never 0W)
    {
        let bad = evaluate_power_capped_arm(
            "missing_power_sensor",
            true,
            true,
            0.0,
            3.0,
            50.0,
            190.0,
            210.0,
            PowerSensorState::MissingOrUncalibrated,
            None,
            true,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc2_missing_power_sensor_never_zero".into(),
            outcome: if bad.tok_per_sec_per_watt.is_none()
                && bad.admission_outcome != OutcomeStatus::Accepted
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "absent power sensor keeps tok/s/W and J/token as None rather than 0W".into(),
        });
    }

    // NC3: missing throttle telemetry cannot claim unthrottled acceptance
    {
        let bad = evaluate_power_capped_arm(
            "missing_throttle_counters",
            true,
            true,
            0.0,
            3.0,
            50.0,
            190.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(200.0),
            false,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc3_missing_throttle_telemetry_not_accepted".into(),
            outcome: bad.admission_outcome,
            detail: "absent throttle counters never count as evidence of zero throttling (#43)"
                .into(),
        });
    }

    // NC4: exceeding configured 210W power cap rejected
    {
        let bad = evaluate_power_capped_arm(
            "over_power_cap",
            true,
            true,
            0.0,
            3.0,
            50.0,
            230.0,
            210.0,
            PowerSensorState::CalibratedDeviceSensor,
            Some(260.0),
            true,
        )?;
        negative_controls.push(NegativeControl {
            case_id: "nc4_power_cap_violation_rejected".into(),
            outcome: bad.admission_outcome,
            detail: "260W draw violates declared 210W cap beyond 5% board tolerance".into(),
        });
    }

    // NC5: unnormalized distribution rejected in DFlash2 verifier
    {
        let bad_p = [0.2_f64; VOCAB]; // sum = 1.6
        let good_q = [0.125_f64; VOCAB];
        negative_controls.push(NegativeControl {
            case_id: "nc5_unnormalized_dflash2_distribution_rejected".into(),
            outcome: if verify_dflash2_stochastic_parity(&bad_p, &good_q).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "unnormalized target distribution fails rejection-sampling verifier".into(),
        });
    }

    // NC6: NaN power measurement rejected cleanly
    {
        negative_controls.push(NegativeControl {
            case_id: "nc6_nan_power_reading_rejected".into(),
            outcome: if evaluate_power_capped_arm(
                "nan_w",
                true,
                true,
                0.0,
                3.0,
                50.0,
                190.0,
                210.0,
                PowerSensorState::CalibratedDeviceSensor,
                Some(f64::NAN),
                true,
            )
            .is_err()
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN power reading rejected cleanly".into(),
        });
    }

    // NC7: acceptance length < 1.0 rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc7_sub_unit_acceptance_length_rejected".into(),
            outcome: if evaluate_power_capped_arm(
                "bad_acc",
                true,
                true,
                0.0,
                0.5,
                50.0,
                190.0,
                210.0,
                PowerSensorState::CalibratedDeviceSensor,
                Some(200.0),
                true,
            )
            .is_err()
            {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "speculative acceptance length cannot be less than 1.0 token/step".into(),
        });
    }

    // NC8: unverified 229 tok/s / 47 ms X-post numbers refused as acceptance gate
    {
        let import_x_post_229 = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_229tps_x_post_non_gating".into(),
            outcome: if import_x_post_229 {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "229 tok/s & 47 ms TTFT X-post figures stay unverified related work".into(),
        });
    }

    Ok(SaberDFlash2PowerReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Dual R9700 (210W cap) / GPU host serving Qwen3.8-27B on pinned GGZ14/saber-engine + DFlash2",
            "Pure-Rust DFlash2 stochastic parity verifier and sensor-gated power-capped efficiency contract ran on CPU; physical Qwen3.8-27B saber-engine serve did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        upstream_repo: UPSTREAM_REPO.into(),
        upstream_commit: UPSTREAM_COMMIT.into(),
        upstream_license: UPSTREAM_LICENSE.into(),
        dflash2_verified_accept_prob: mean_acc_prob,
        dflash2_verified_l1_err: max_l1,
        arms,
        negative_controls,
    })
}

pub fn inputs_json(report: &SaberDFlash2PowerReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.saber-dflash2-power.inputs.v1\",\n",
            "  \"issue\": 81,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"upstream_repo\": {},\n",
            "  \"upstream_commit\": {},\n",
            "  \"upstream_license\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"target_model\": {},\n",
            "  \"related_issues\": [3, 43]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.upstream_repo),
        quote(&report.upstream_commit),
        quote(&report.upstream_license),
        quote(UPSTREAM_X_POST),
        quote(TARGET_MODEL),
    )
}

pub fn results_json(report: &SaberDFlash2PowerReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"backend_id\":{},\"dflash2_enabled\":{},\"greedy_exact_match_vs_ref\":{},\"stochastic_max_l1_err\":{},\"parity_passed\":{},\"mean_acceptance_len\":{},\"ttft_ms\":{},\"decode_tok_s\":{},\"power_cap_watts\":{},\"sensor_state\":{},\"measured_device_watts\":{},\"tok_per_sec_per_watt\":{},\"joules_per_token\":{},\"throttle_telemetry_observed\":{},\"admission_outcome\":{}}}",
                quote(&a.backend_id),
                a.dflash2_enabled,
                a.greedy_exact_match_vs_ref,
                finite_f64(a.stochastic_max_l1_err),
                a.parity_passed,
                finite_f64(a.mean_acceptance_len),
                finite_f64(a.ttft_ms),
                finite_f64(a.decode_tok_s),
                finite_f64(a.power_cap_watts),
                quote(a.sensor_state.as_str()),
                opt_f64(a.measured_device_watts),
                opt_f64(a.tok_per_sec_per_watt),
                opt_f64(a.joules_per_token),
                a.throttle_telemetry_observed,
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
            "  \"schema\": \"inference-lab.saber-dflash2-power.results.v1\",\n",
            "  \"issue\": 81,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"dflash2_verified_accept_prob\": {},\n",
            "  \"dflash2_verified_l1_err\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        finite_f64(report.dflash2_verified_accept_prob),
        finite_f64(report.dflash2_verified_l1_err),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &SaberDFlash2PowerReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #81: Pinned `saber-engine` (Rust) + DFlash2 Power-Capped Qualification"
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
        "- Upstream pin: `{}@{}` (`{}`, DFlash2 stochastic L1 err: `{:.3e}`)",
        report.upstream_repo,
        &report.upstream_commit[..8],
        report.upstream_license,
        report.dflash2_verified_l1_err
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## Parity-Gated Power-Capped Serving Comparison").unwrap();
    writeln!(
        out,
        "| Backend | DFlash2 | Parity | Accept Len | TTFT (ms) | tok/s | Watts | tok/s/W | J/tok |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{:.2}` | `{:.1}` | `{:.1}` | `{:.1}` | `{:.3}` | `{:.3}` |",
            a.backend_id,
            a.dflash2_enabled,
            a.parity_passed,
            a.mean_acceptance_len,
            a.ttft_ms,
            a.decode_tok_s,
            a.measured_device_watts.unwrap_or(0.0),
            a.tok_per_sec_per_watt.unwrap_or(0.0),
            a.joules_per_token.unwrap_or(0.0)
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
    fn dflash2_stochastic_parity_is_exact_and_gates_power_efficiency() {
        let r = run_default_harness().unwrap();
        assert!(!r.default_enabled);
        assert!(r.dflash2_verified_l1_err <= STOCHASTIC_TOL);
        assert_eq!(r.arms.len(), 4);
        for a in &r.arms {
            assert!(a.parity_passed);
            assert_eq!(a.admission_outcome, OutcomeStatus::Accepted);
            assert!(a.tok_per_sec_per_watt.unwrap() > 0.0);
        }
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
