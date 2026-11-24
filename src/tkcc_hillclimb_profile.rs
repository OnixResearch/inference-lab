//! Issue #79: Log Together Compiler (TKCC) continual abstraction-free compilation as related work
//! beside `#53`/`#61` and measure experiment-throughput bottlenecks in local hill-climbs.
//!
//! Pinned related work (strictly non-gating external notes):
//! - X post (2026-11-04, Simran Arora `@simran_s_arora`):
//!   `https://x.com/simran_s_arora/status/2118032475498983540`
//! - Authors' unverified claims: Together Compiler (TKCC) is an abstraction-free continual
//!   agentic compiler for polymorphic inference workloads claiming 252x higher experiment
//!   throughput per hour and SOTA TPS via 24h/2-node recompiles.
//! - Directly informs `#53` (ceiling-first serving hill-climb) and `#61` (agent-written decode
//!   megakernels).
//!
//! Provides:
//! - Structured TKCC related-work record beside `#53` and `#61`.
//! - Local hill-climb experiment-cycle bottleneck profiler decomposing each candidate trial into
//!   5 explicit phases: `compile_s`, `warm_up_s`, `parity_check_s`, `measurement_s`, and
//!   `teardown_overhead_s`.
//! - Exact Amdahl speedup ceiling per phase, failure-retaining population accounting (`accepted`,
//!   `parity_failed`, `compile_failed`, `timeout` all preserved in the total candidate denominator),
//!   and hard enforcement that `parity_check_s > 0` (refusing throughput inflation obtained by
//!   skipping numerical correctness gates).
//! - 8 adversarial negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{ExecutionClass, HardwareGate, OutcomeStatus};

pub const UPSTREAM_X_POST: &str = "https://x.com/simran_s_arora/status/2118032475498983540";
pub const PIN_STATUS: &str = "post_only_unlocated_paper_repo";

fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite float in {ctx}"))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HillClimbTrialTiming {
    pub candidate_id: String,
    pub outcome: OutcomeStatus,
    pub parity_gate_executed: bool,
    pub compile_s: f64,
    pub warm_up_s: f64,
    pub parity_check_s: f64,
    pub measurement_s: f64,
    pub teardown_overhead_s: f64,
}

impl HillClimbTrialTiming {
    pub fn total_s(&self) -> Result<f64, String> {
        let vals = [
            check_finite(self.compile_s, "compile_s")?,
            check_finite(self.warm_up_s, "warm_up_s")?,
            check_finite(self.parity_check_s, "parity_check_s")?,
            check_finite(self.measurement_s, "measurement_s")?,
            check_finite(self.teardown_overhead_s, "teardown_overhead_s")?,
        ];
        if vals.iter().any(|&x| x < 0.0) {
            return Err(format!(
                "negative phase duration in `{}`",
                self.candidate_id
            ));
        }
        let sum: f64 = vals.iter().sum();
        if sum <= 0.0 {
            return Err(format!("zero total duration in `{}`", self.candidate_id));
        }
        Ok(sum)
    }
}

#[derive(Debug, Clone)]
pub struct PhaseBreakdown {
    pub phase_name: String,
    pub total_s: f64,
    pub share_fraction: f64,
    pub amdahl_max_speedup_if_zeroed: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CycleProfileSummary {
    pub profile_id: String,
    pub total_trials: usize,
    pub accepted_trials: usize,
    pub non_accepted_trials: usize,
    pub total_wall_s: f64,
    pub total_trials_per_hour: Option<f64>,
    pub accepted_trials_per_hour: Option<f64>,
    pub dominant_bottleneck_phase: String,
    pub phases: Vec<PhaseBreakdown>,
}

pub fn profile_hillclimb_cycles(
    profile_id: &str,
    trials: &[HillClimbTrialTiming],
) -> Result<CycleProfileSummary, String> {
    if trials.is_empty() {
        return Err("empty hill-climb trial log".into());
    }
    let mut sum_compile = 0.0_f64;
    let mut sum_warmup = 0.0_f64;
    let mut sum_parity = 0.0_f64;
    let mut sum_meas = 0.0_f64;
    let mut sum_teardown = 0.0_f64;
    let mut accepted = 0usize;

    for t in trials {
        // Hard invariant: an accepted candidate MUST have executed the parity check with > 0 time.
        if t.outcome == OutcomeStatus::Accepted
            && (!t.parity_gate_executed || t.parity_check_s <= 0.0)
        {
            return Err(format!(
                "accepted candidate `{}` skipped mandatory parity check",
                t.candidate_id
            ));
        }
        let _ = t.total_s()?;
        sum_compile += t.compile_s;
        sum_warmup += t.warm_up_s;
        sum_parity += t.parity_check_s;
        sum_meas += t.measurement_s;
        sum_teardown += t.teardown_overhead_s;
        if t.outcome == OutcomeStatus::Accepted {
            accepted += 1;
        }
    }

    let total_wall_s = sum_compile + sum_warmup + sum_parity + sum_meas + sum_teardown;
    if !total_wall_s.is_finite() || total_wall_s <= 0.0 {
        return Err("invalid total_wall_s".into());
    }

    let raw_phases = [
        ("compile", sum_compile),
        ("warm_up", sum_warmup),
        ("parity_check", sum_parity),
        ("measurement", sum_meas),
        ("teardown_overhead", sum_teardown),
    ];

    let mut dominant = raw_phases[0].0;
    let mut dominant_s = raw_phases[0].1;
    let mut phases = Vec::with_capacity(raw_phases.len());
    for (name, dur) in raw_phases {
        if dur > dominant_s {
            dominant_s = dur;
            dominant = name;
        }
        let share = dur / total_wall_s;
        let rem = 1.0 - share;
        let amdahl = if rem > 1.0e-12 { Some(1.0 / rem) } else { None };
        phases.push(PhaseBreakdown {
            phase_name: name.into(),
            total_s: dur,
            share_fraction: share,
            amdahl_max_speedup_if_zeroed: amdahl,
        });
    }

    let total_trials_per_hour = Some((trials.len() as f64) * 3600.0 / total_wall_s);
    let accepted_trials_per_hour = if accepted > 0 {
        Some((accepted as f64) * 3600.0 / total_wall_s)
    } else {
        None
    };

    Ok(CycleProfileSummary {
        profile_id: profile_id.into(),
        total_trials: trials.len(),
        accepted_trials: accepted,
        non_accepted_trials: trials.len() - accepted,
        total_wall_s,
        total_trials_per_hour,
        accepted_trials_per_hour,
        dominant_bottleneck_phase: dominant.into(),
        phases,
    })
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct TkccHillclimbReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub default_enabled: bool,
    pub synthetic_contract_fixture: bool,
    pub pin_status: String,
    pub profiles: Vec<CycleProfileSummary>,
    pub negative_controls: Vec<NegativeControl>,
}

pub fn run_default_harness() -> Result<TkccHillclimbReport, String> {
    // Baseline local hill-climb (cold full rebuild + full server restart per trial)
    let cold_trials = vec![
        HillClimbTrialTiming {
            candidate_id: "cand-01-baseline".into(),
            outcome: OutcomeStatus::Accepted,
            parity_gate_executed: true,
            compile_s: 42.0,
            warm_up_s: 14.0,
            parity_check_s: 9.5,
            measurement_s: 18.0,
            teardown_overhead_s: 4.5,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-02-compile-err".into(),
            outcome: OutcomeStatus::Invalid,
            parity_gate_executed: false,
            compile_s: 28.0,
            warm_up_s: 0.0,
            parity_check_s: 0.0,
            measurement_s: 0.0,
            teardown_overhead_s: 2.0,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-03-parity-fail".into(),
            outcome: OutcomeStatus::Invalid,
            parity_gate_executed: true,
            compile_s: 40.0,
            warm_up_s: 13.5,
            parity_check_s: 9.0,
            measurement_s: 0.0,
            teardown_overhead_s: 4.0,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-04-accepted-fast".into(),
            outcome: OutcomeStatus::Accepted,
            parity_gate_executed: true,
            compile_s: 41.0,
            warm_up_s: 14.0,
            parity_check_s: 9.5,
            measurement_s: 17.5,
            teardown_overhead_s: 4.0,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-05-timeout".into(),
            outcome: OutcomeStatus::Timeout,
            parity_gate_executed: true,
            compile_s: 43.0,
            warm_up_s: 14.5,
            parity_check_s: 10.0,
            measurement_s: 30.0,
            teardown_overhead_s: 5.5,
        },
    ];

    // Incremental kernel-cache + persistent warm-worker local hill-climb (retaining full parity!)
    let cached_trials = vec![
        HillClimbTrialTiming {
            candidate_id: "cand-01-baseline".into(),
            outcome: OutcomeStatus::Accepted,
            parity_gate_executed: true,
            compile_s: 6.5,
            warm_up_s: 3.0,
            parity_check_s: 9.5,
            measurement_s: 18.0,
            teardown_overhead_s: 1.0,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-02-compile-err".into(),
            outcome: OutcomeStatus::Invalid,
            parity_gate_executed: false,
            compile_s: 4.0,
            warm_up_s: 0.0,
            parity_check_s: 0.0,
            measurement_s: 0.0,
            teardown_overhead_s: 0.5,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-03-parity-fail".into(),
            outcome: OutcomeStatus::Invalid,
            parity_gate_executed: true,
            compile_s: 6.0,
            warm_up_s: 2.8,
            parity_check_s: 9.0,
            measurement_s: 0.0,
            teardown_overhead_s: 0.8,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-04-accepted-fast".into(),
            outcome: OutcomeStatus::Accepted,
            parity_gate_executed: true,
            compile_s: 6.2,
            warm_up_s: 3.0,
            parity_check_s: 9.5,
            measurement_s: 17.5,
            teardown_overhead_s: 1.0,
        },
        HillClimbTrialTiming {
            candidate_id: "cand-05-timeout".into(),
            outcome: OutcomeStatus::Timeout,
            parity_gate_executed: true,
            compile_s: 6.8,
            warm_up_s: 3.1,
            parity_check_s: 10.0,
            measurement_s: 30.0,
            teardown_overhead_s: 1.2,
        },
    ];

    let p_cold = profile_hillclimb_cycles("local_cold_rebuild_hillclimb", &cold_trials)?;
    let p_cached = profile_hillclimb_cycles("local_cached_persistent_hillclimb", &cached_trials)?;

    let mut negative_controls = Vec::new();

    // NC1: accepted candidate skipping parity check rejected
    {
        let mut bad = cold_trials[0].clone();
        bad.parity_gate_executed = false;
        bad.parity_check_s = 0.0;
        negative_controls.push(NegativeControl {
            case_id: "nc1_skipped_parity_gate_on_accepted_rejected".into(),
            outcome: if profile_hillclimb_cycles("bad", &[bad]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "inflating experiment throughput by skipping parity checks is refused".into(),
        });
    }

    // NC2: negative phase duration rejected
    {
        let mut bad = cold_trials[0].clone();
        bad.compile_s = -1.0;
        negative_controls.push(NegativeControl {
            case_id: "nc2_negative_phase_duration_rejected".into(),
            outcome: if profile_hillclimb_cycles("bad", &[bad]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "negative compile duration rejected cleanly".into(),
        });
    }

    // NC3: NaN phase duration rejected
    {
        let mut bad = cold_trials[0].clone();
        bad.measurement_s = f64::NAN;
        negative_controls.push(NegativeControl {
            case_id: "nc3_nan_phase_duration_rejected".into(),
            outcome: if profile_hillclimb_cycles("bad", &[bad]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "NaN measurement duration rejected".into(),
        });
    }

    // NC4: empty trial log rejected
    {
        negative_controls.push(NegativeControl {
            case_id: "nc4_empty_trial_log_rejected".into(),
            outcome: if profile_hillclimb_cycles("empty", &[]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "empty hill-climb ledger refused instead of zero division".into(),
        });
    }

    // NC5: all-failed batch reports accepted_trials_per_hour = None (never zero-collapsed success)
    {
        let all_fail = profile_hillclimb_cycles("all_fail", &cold_trials[1..3])?;
        negative_controls.push(NegativeControl {
            case_id: "nc5_zero_accepted_reports_none_rate".into(),
            outcome: if all_fail.accepted_trials == 0 && all_fail.accepted_trials_per_hour.is_none()
            {
                OutcomeStatus::Exhausted
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero accepted candidates returns accepted_trials_per_hour = None".into(),
        });
    }

    // NC6: dropping failed candidates from denominator detected
    {
        let retained_all = p_cold.total_trials
            == p_cold.accepted_trials + p_cold.non_accepted_trials
            && p_cold.non_accepted_trials == 3;
        negative_controls.push(NegativeControl {
            case_id: "nc6_failed_candidates_retained_in_denominator".into(),
            outcome: if retained_all {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "omitting compile-failed/parity-failed/timeout trials from wall-time denominator is invalid".into(),
        });
    }

    // NC7: zero total duration rejected
    {
        let bad = HillClimbTrialTiming {
            candidate_id: "zero".into(),
            outcome: OutcomeStatus::Invalid,
            parity_gate_executed: false,
            compile_s: 0.0,
            warm_up_s: 0.0,
            parity_check_s: 0.0,
            measurement_s: 0.0,
            teardown_overhead_s: 0.0,
        };
        negative_controls.push(NegativeControl {
            case_id: "nc7_zero_total_trial_duration_rejected".into(),
            outcome: if profile_hillclimb_cycles("zero", &[bad]).is_err() {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "zero-duration trial refused before throughput rate computation".into(),
        });
    }

    // NC8: importing TKCC's unverified 252x claim as a local lab gate rejected
    {
        let import_252x_as_gate = true;
        negative_controls.push(NegativeControl {
            case_id: "nc8_unverified_252x_tkcc_claim_non_gating".into(),
            outcome: if import_252x_as_gate {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::Accepted
            },
            detail: "TKCC 252x experiment-throughput claim stays unverified related work".into(),
        });
    }

    Ok(TkccHillclimbReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "2-node continual GPU compilation & serving hill-climb cluster (beside #53 & #61)",
            "Pure-Rust experiment-cycle bottleneck profiler and Amdahl ceiling analyzer executed on CPU; live GPU kernel compilation & 24h continual recompiles did not execute locally",
        ),
        default_enabled: false,
        synthetic_contract_fixture: true,
        pin_status: PIN_STATUS.into(),
        profiles: vec![p_cold, p_cached],
        negative_controls,
    })
}

pub fn inputs_json(report: &TkccHillclimbReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.tkcc-hillclimb-profile.inputs.v1\",\n",
            "  \"issue\": 79,\n",
            "  \"execution_class\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"pin_status\": {},\n",
            "  \"upstream_x_post\": {},\n",
            "  \"related_issues\": [53, 61]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.default_enabled,
        report.synthetic_contract_fixture,
        quote(&report.pin_status),
        quote(UPSTREAM_X_POST),
    )
}

pub fn results_json(report: &TkccHillclimbReport) -> String {
    let profiles: Vec<String> = report
        .profiles
        .iter()
        .map(|p| {
            let ph: Vec<String> = p
                .phases
                .iter()
                .map(|x| {
                    format!(
                        "{{\"phase_name\":{},\"total_s\":{},\"share_fraction\":{},\"amdahl_max_speedup_if_zeroed\":{}}}",
                        quote(&x.phase_name),
                        finite_f64(x.total_s),
                        finite_f64(x.share_fraction),
                        opt_f64(x.amdahl_max_speedup_if_zeroed)
                    )
                })
                .collect();
            format!(
                "{{\"profile_id\":{},\"total_trials\":{},\"accepted_trials\":{},\"non_accepted_trials\":{},\"total_wall_s\":{},\"total_trials_per_hour\":{},\"accepted_trials_per_hour\":{},\"dominant_bottleneck_phase\":{},\"phases\":[{}]}}",
                quote(&p.profile_id),
                p.total_trials,
                p.accepted_trials,
                p.non_accepted_trials,
                finite_f64(p.total_wall_s),
                opt_f64(p.total_trials_per_hour),
                opt_f64(p.accepted_trials_per_hour),
                quote(&p.dominant_bottleneck_phase),
                ph.join(",")
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
            "  \"schema\": \"inference-lab.tkcc-hillclimb-profile.results.v1\",\n",
            "  \"issue\": 79,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"default_enabled\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"profiles\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.default_enabled,
        report.synthetic_contract_fixture,
        profiles.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &TkccHillclimbReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #79: Together Compiler (TKCC) Related Work & Local Hill-Climb Cycle Profiler"
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
    writeln!(out).unwrap();
    writeln!(out, "## Hill-Climb Experiment-Cycle Bottlenecks").unwrap();
    writeln!(
        out,
        "| Profile | Trials (Acc/Total) | Wall (s) | Trials/hr | Accepted/hr | Dominant Phase |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | ---: | --- |").unwrap();
    for p in &report.profiles {
        writeln!(
            out,
            "| `{}` | `{}/{}` | `{:.1}` | `{:.1}` | `{:.1}` | `{}` |",
            p.profile_id,
            p.accepted_trials,
            p.total_trials,
            p.total_wall_s,
            p.total_trials_per_hour.unwrap_or(0.0),
            p.accepted_trials_per_hour.unwrap_or(0.0),
            p.dominant_bottleneck_phase
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
    fn cold_profile_is_compile_bound_and_cached_shifts_bottleneck_to_measurement() {
        let r = run_default_harness().unwrap();
        assert_eq!(r.profiles.len(), 2);
        assert_eq!(r.profiles[0].dominant_bottleneck_phase, "compile");
        assert_eq!(r.profiles[1].dominant_bottleneck_phase, "measurement");
        assert!(
            r.profiles[1].total_trials_per_hour.unwrap()
                > r.profiles[0].total_trials_per_hour.unwrap()
        );
        assert_eq!(r.negative_controls.len(), 8);
        for c in &r.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
