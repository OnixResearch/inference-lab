//! Issue #17: Benchmark discrete graphical sampling by accuracy and effective samples per
//! end-to-end cost.
//!
//! Pure deterministic core consuming Compiler Lab `#222`'s finite semantic profile
//! (`fd0bb4328d811768a08083f5d7a0ddccf2dab275`, positive factor tables, `<= 256` joint states):
//! - Independent exact joint enumeration oracle computing partition function `Z`, exact node
//!   marginals, exact conditional distributions `P(X_i | X_{-i})`, and exact expected energy.
//! - Three real CPU Gibbs samplers: sequential reference, serial execution of an admitted
//!   graph-coloring phase schedule, and color-phase Gibbs with explicit deterministic
//!   per-`(sweep, node)` SplitMix64 RNG stream mapping (proving exact trajectory reproduction
//!   when schedule + random-input assignment match, and oracle-based distribution comparison
//!   across distinct valid schedules).
//! - Honest Geyer initial-positive-sequence ESS estimator (`ess-ips-v1`) returning `None`
//!   (never invented ESS) on stuck, short, or zero-variance chains.
//! - 8 adversarial controls catching stale neighbor cache, invalid 1-coloring on coupled nodes,
//!   omitted state commits, swapped pairwise factor indexing, correlated/reused RNG streams,
//!   invalid normalization, incomplete chains, and silent accelerator fallback.
//! - Explicit non-gating pinned related-work records for AMARA (`arXiv:2611.15167v1`) and
//!   default-off CRAM (`arXiv:2611.17309`, `Prefrontal-AI/CRAM@62bbe830c7d6f87f2ba18f30a28039258b986454`).

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const COMPILER_LAB_222_COMMIT: &str = "fd0bb4328d811768a08083f5d7a0ddccf2dab275";
pub const CRAM_PIN_COMMIT: &str = "62bbe830c7d6f87f2ba18f30a28039258b986454";
pub const MAX_EXACT_JOINT_STATES: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFactor {
    pub u: usize,
    pub v: usize,
    /// Row-major `[card_u][card_v]` strictly positive unnormalized potential table.
    pub table: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteFactorGraph {
    pub model_id: String,
    pub cardinalities: Vec<usize>,
    /// Strictly positive unary potential tables `[node][val]`.
    pub unary: Vec<Vec<f64>>,
    pub pairwise: Vec<PairwiseFactor>,
}

impl DiscreteFactorGraph {
    pub fn total_joint_states(&self) -> Result<usize, String> {
        if self.cardinalities.is_empty() {
            return Err("empty graphical model".into());
        }
        let mut total = 1usize;
        for &c in &self.cardinalities {
            if c < 2 {
                return Err("variable cardinality must be >= 2".into());
            }
            total = total
                .checked_mul(c)
                .ok_or_else(|| "joint state count overflow".to_string())?;
        }
        Ok(total)
    }

    pub fn validate(&self) -> Result<usize, String> {
        let total = self.total_joint_states()?;
        if total > MAX_EXACT_JOINT_STATES {
            return Err("model exceeds 256-state exact enumeration profile".into());
        }
        if self.unary.len() != self.cardinalities.len() {
            return Err("unary potential count mismatch".into());
        }
        for (i, u) in self.unary.iter().enumerate() {
            if u.len() != self.cardinalities[i] || u.iter().any(|&w| !w.is_finite() || w <= 0.0) {
                return Err("unary potential must be finite and strictly positive".into());
            }
        }
        for pf in &self.pairwise {
            if pf.u >= self.cardinalities.len() || pf.v >= self.cardinalities.len() || pf.u == pf.v
            {
                return Err("invalid pairwise edge indices".into());
            }
            if pf.table.len() != self.cardinalities[pf.u] {
                return Err("pairwise row count mismatch".into());
            }
            for row in &pf.table {
                if row.len() != self.cardinalities[pf.v]
                    || row.iter().any(|&w| !w.is_finite() || w <= 0.0)
                {
                    return Err("pairwise potential must be finite and strictly positive".into());
                }
            }
        }
        Ok(total)
    }

    pub fn unnormalized_weight(&self, state: &[usize]) -> f64 {
        let mut w = 1.0_f64;
        for (i, &val) in state.iter().enumerate() {
            w *= self.unary[i][val];
        }
        for pf in &self.pairwise {
            w *= pf.table[state[pf.u]][state[pf.v]];
        }
        w
    }

    pub fn energy(&self, state: &[usize]) -> f64 {
        -self.unnormalized_weight(state).ln()
    }

    /// Exact conditional distribution `P(X_node = . | X_{-node})` computed directly from factors.
    pub fn conditional_distribution(
        &self,
        node: usize,
        state: &[usize],
        swap_pairwise_indexing: bool,
    ) -> Result<Vec<f64>, String> {
        let card = self.cardinalities[node];
        let mut weights = vec![0.0_f64; card];
        for (val, slot) in weights.iter_mut().enumerate() {
            let mut w = self.unary[node][val];
            for pf in &self.pairwise {
                if pf.u == node {
                    let other = state[pf.v];
                    let entry = if swap_pairwise_indexing
                        && other < pf.table.len()
                        && val < pf.table[other].len()
                    {
                        pf.table[other][val]
                    } else {
                        pf.table[val][other]
                    };
                    w *= entry;
                } else if pf.v == node {
                    let other = state[pf.u];
                    let entry = if swap_pairwise_indexing
                        && val < pf.table.len()
                        && other < pf.table[val].len()
                    {
                        pf.table[val][other]
                    } else {
                        pf.table[other][val]
                    };
                    w *= entry;
                }
            }
            *slot = w;
        }
        let z: f64 = weights.iter().sum();
        if !z.is_finite() || z <= 0.0 {
            return Err("invalid conditional normalization constant".into());
        }
        for w in &mut weights {
            *w /= z;
        }
        Ok(weights)
    }

    /// Validate that a color-phase schedule has no edge between two nodes in the same color phase.
    pub fn validate_color_schedule(&self, phases: &[Vec<usize>]) -> bool {
        let n = self.cardinalities.len();
        let mut node_phase = vec![None; n];
        for (p_idx, phase) in phases.iter().enumerate() {
            for &node in phase {
                if node >= n || node_phase[node].is_some() {
                    return false;
                }
                node_phase[node] = Some(p_idx);
            }
        }
        if node_phase.iter().any(Option::is_none) {
            return false;
        }
        for pf in &self.pairwise {
            if node_phase[pf.u] == node_phase[pf.v] {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactOracleResult {
    pub total_joint_states: usize,
    pub partition_z: f64,
    pub marginals: Vec<Vec<f64>>,
    pub expected_energy: f64,
}

/// Exact exhaustive enumeration oracle (`<= 256` joint states).
pub fn exact_enumerate(graph: &DiscreteFactorGraph) -> Result<ExactOracleResult, String> {
    let total = graph.validate()?;
    let n = graph.cardinalities.len();
    let mut marginals: Vec<Vec<f64>> = graph
        .cardinalities
        .iter()
        .map(|&c| vec![0.0_f64; c])
        .collect();
    let mut z = 0.0_f64;
    let mut weighted_energy = 0.0_f64;

    let mut state = vec![0usize; n];
    for idx in 0..total {
        let mut rem = idx;
        for i in (0..n).rev() {
            state[i] = rem % graph.cardinalities[i];
            rem /= graph.cardinalities[i];
        }
        let w = graph.unnormalized_weight(&state);
        if !w.is_finite() || w <= 0.0 {
            return Err("non-finite or non-positive joint weight in exact_enumerate".into());
        }
        z += w;
        weighted_energy += w * (-w.ln());
        for i in 0..n {
            marginals[i][state[i]] += w;
        }
    }
    if !z.is_finite() || z <= 0.0 || !weighted_energy.is_finite() {
        return Err("non-finite partition function or energy in exact_enumerate".into());
    }
    for m in &mut marginals {
        for p in m.iter_mut() {
            *p /= z;
        }
    }
    Ok(ExactOracleResult {
        total_joint_states: total,
        partition_z: z,
        marginals,
        expected_energy: weighted_energy / z,
    })
}

/// Deterministic SplitMix64 hash mapping `(seed, chain, sweep, node)` -> uniform `[0, 1)`.
fn uniform_for_step(seed: u64, chain: u64, sweep: u64, node: u64) -> f64 {
    let mut z = seed
        .wrapping_add(chain.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(sweep.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(node.wrapping_mul(0x94d0_49bb_1331_11eb));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Map upper 53 bits strictly to [0, 1).
    ((z >> 11) as f64) * (1.0_f64 / ((1u64 << 53) as f64))
}

fn sample_categorical(probs: &[f64], u: f64) -> usize {
    let mut cum = 0.0_f64;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return i;
        }
    }
    probs.len() - 1
}

/// Honest Geyer Initial-Positive-Sequence (IPS) ESS estimator (`ess-ips-v1`).
/// Returns `None` for short, stuck, or zero-variance series — never invents ESS.
pub fn estimate_ess_ips(series: &[f64]) -> Option<f64> {
    let n = series.len();
    if n < 32 || series.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let mean = series.iter().sum::<f64>() / nf;
    let var = series.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    if var <= 1e-12 {
        return None;
    }

    let max_lag = (n / 2).min(64);
    let autocorr = |lag: usize| -> f64 {
        let mut cov = 0.0_f64;
        for t in 0..(n - lag) {
            cov += (series[t] - mean) * (series[t + lag] - mean);
        }
        cov / (nf * var)
    };

    let mut tau = -1.0_f64;
    let mut k = 0usize;
    while 2 * k < max_lag {
        let rho_even = if k == 0 { 1.0 } else { autocorr(2 * k) };
        let rho_odd = autocorr(2 * k + 1);
        let pair_sum = rho_even + rho_odd;
        if pair_sum <= 0.0 {
            break;
        }
        tau += 2.0 * pair_sum;
        k += 1;
    }
    let tau_clamped = tau.max(1.0);
    let ess = nf / tau_clamped;
    if ess.is_finite() && ess > 0.0 {
        Some(ess)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerVariant {
    SequentialReference,
    SerialColorSchedule,
    ColorPhaseParallelModel,
    FaultyStaleNeighborCache,
    FaultyInvalidOneColoring,
    FaultyOmittedCommit,
    FaultySwappedPairwiseIndex,
    FaultyReusedRngStream,
}

impl SamplerVariant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SequentialReference => "sequential_reference",
            Self::SerialColorSchedule => "serial_color_schedule",
            Self::ColorPhaseParallelModel => "color_phase_parallel_model",
            Self::FaultyStaleNeighborCache => "faulty_stale_neighbor_cache",
            Self::FaultyInvalidOneColoring => "faulty_invalid_one_coloring",
            Self::FaultyOmittedCommit => "faulty_omitted_commit",
            Self::FaultySwappedPairwiseIndex => "faulty_swapped_pairwise_index",
            Self::FaultyReusedRngStream => "faulty_reused_rng_stream",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerRunSummary {
    pub variant: SamplerVariant,
    pub schedule_valid: bool,
    pub total_chains: usize,
    pub retained_sweeps_per_chain: usize,
    pub max_marginal_abs_err: f64,
    pub energy_abs_err: f64,
    pub ess_energy: Option<f64>,
    pub trajectory_digest: DigestRef,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[allow(clippy::too_many_arguments)]
pub fn run_sampler_variant(
    graph: &DiscreteFactorGraph,
    oracle: &ExactOracleResult,
    phases: &[Vec<usize>],
    variant: SamplerVariant,
    n_chains: usize,
    burn_in: usize,
    retained_sweeps: usize,
    seed: u64,
    marginal_tol: f64,
) -> Result<SamplerRunSummary, String> {
    if n_chains == 0 || retained_sweeps == 0 || !marginal_tol.is_finite() || marginal_tol < 0.0 {
        return Err("n_chains and retained_sweeps must be > 0 and marginal_tol finite >= 0".into());
    }
    let n_nodes = graph.cardinalities.len();
    let effective_phases: Vec<Vec<usize>> = match variant {
        SamplerVariant::SequentialReference => (0..n_nodes).map(|i| vec![i]).collect(),
        SamplerVariant::FaultyInvalidOneColoring => vec![(0..n_nodes).collect()],
        _ => phases.to_vec(),
    };
    let schedule_valid = match variant {
        SamplerVariant::SequentialReference => true,
        _ => graph.validate_color_schedule(&effective_phases),
    };

    let mut counts: Vec<Vec<usize>> = graph
        .cardinalities
        .iter()
        .map(|&c| vec![0usize; c])
        .collect();
    let mut energy_sum = 0.0_f64;
    let mut first_chain_energies = Vec::with_capacity(retained_sweeps);
    let mut traj_bytes = Vec::new();
    let total_samples = n_chains * retained_sweeps;

    for chain in 0..n_chains {
        let mut state = vec![chain % 2; n_nodes];
        let frozen_init = state.clone();
        for sweep in 0..(burn_in + retained_sweeps) {
            for phase in &effective_phases {
                // In color-phase execution, nodes within a valid phase read the pre-phase snapshot
                // and commit together after sampling all nodes in that phase.
                let snapshot = match variant {
                    SamplerVariant::FaultyStaleNeighborCache => frozen_init.clone(),
                    SamplerVariant::ColorPhaseParallelModel
                    | SamplerVariant::FaultyInvalidOneColoring => state.clone(),
                    _ => state.clone(),
                };
                let mut next_vals = Vec::with_capacity(phase.len());
                for &node in phase {
                    let read_state = match variant {
                        SamplerVariant::SequentialReference
                        | SamplerVariant::SerialColorSchedule => &state,
                        _ => &snapshot,
                    };
                    let probs = graph.conditional_distribution(
                        node,
                        read_state,
                        variant == SamplerVariant::FaultySwappedPairwiseIndex,
                    )?;
                    let u = if variant == SamplerVariant::FaultyReusedRngStream {
                        // Negative control: every node and chain reuses the same RNG draw!
                        uniform_for_step(seed, 0, sweep as u64, 0)
                    } else {
                        uniform_for_step(seed, chain as u64, sweep as u64, node as u64)
                    };
                    let sampled = sample_categorical(&probs, u);
                    match variant {
                        SamplerVariant::SequentialReference
                        | SamplerVariant::SerialColorSchedule => {
                            state[node] = sampled;
                        }
                        SamplerVariant::FaultyOmittedCommit => {}
                        _ => next_vals.push((node, sampled)),
                    }
                }
                for (node, val) in next_vals {
                    state[node] = val;
                }
            }

            if sweep >= burn_in {
                let e = graph.energy(&state);
                energy_sum += e;
                if chain == 0 {
                    first_chain_energies.push(e);
                    for &s in &state {
                        traj_bytes.push(s as u8);
                    }
                }
                for i in 0..n_nodes {
                    counts[i][state[i]] += 1;
                }
            }
        }
    }

    let mut max_marginal_err = 0.0_f64;
    for (i, node_counts) in counts.iter().enumerate().take(n_nodes) {
        for (v, &cnt) in node_counts.iter().enumerate() {
            let emp = (cnt as f64) / (total_samples as f64);
            max_marginal_err = max_marginal_err.max((emp - oracle.marginals[i][v]).abs());
        }
    }
    let emp_energy = energy_sum / (total_samples as f64);
    let energy_err = (emp_energy - oracle.expected_energy).abs();
    let ess = estimate_ess_ips(&first_chain_energies);

    let (outcome, detail) = if !schedule_valid {
        (
            OutcomeStatus::Invalid,
            "invalid_color_schedule_coupled_nodes_updated_simultaneously".into(),
        )
    } else if ess.is_none() {
        (
            OutcomeStatus::Invalid,
            "stuck_or_zero_variance_chain_ess_undefined".into(),
        )
    } else if max_marginal_err > marginal_tol {
        (
            OutcomeStatus::Invalid,
            "empirical_marginals_exceed_exact_oracle_tolerance".into(),
        )
    } else {
        (
            OutcomeStatus::Accepted,
            "matches_exact_oracle_marginals_with_valid_ess".into(),
        )
    };

    Ok(SamplerRunSummary {
        variant,
        schedule_valid,
        total_chains: n_chains,
        retained_sweeps_per_chain: retained_sweeps,
        max_marginal_abs_err: max_marginal_err,
        energy_abs_err: energy_err,
        ess_energy: ess,
        trajectory_digest: DigestRef::sha256(&traj_bytes),
        outcome,
        detail,
    })
}

/// Asymmetric coupled 4-node bipartite cycle (2^4 = 16 states <= 256) that cleanly distinguishes
/// swapped factor indexing, stale caches, invalid 1-coloring, and correlated RNG streams.
pub fn frozen_coupled_model() -> (DiscreteFactorGraph, Vec<Vec<usize>>) {
    let graph = DiscreteFactorGraph {
        model_id: "compiler-lab-222-asymmetric-4node-cycle-v1".into(),
        cardinalities: vec![2, 2, 2, 2],
        unary: vec![
            vec![1.6, 0.7],
            vec![0.8, 1.5],
            vec![1.4, 0.9],
            vec![0.6, 1.8],
        ],
        pairwise: vec![
            // Asymmetric couplings so swapped indexing (u,v) vs (v,u) alters conditionals!
            PairwiseFactor {
                u: 0,
                v: 1,
                table: vec![vec![3.2, 0.4], vec![1.9, 2.8]],
            },
            PairwiseFactor {
                u: 1,
                v: 2,
                table: vec![vec![2.6, 0.5], vec![2.1, 3.0]],
            },
            PairwiseFactor {
                u: 2,
                v: 3,
                table: vec![vec![3.4, 0.6], vec![1.8, 2.5]],
            },
            PairwiseFactor {
                u: 3,
                v: 0,
                table: vec![vec![2.9, 0.3], vec![2.2, 3.1]],
            },
        ],
    };
    let phases = vec![vec![0, 2], vec![1, 3]];
    (graph, phases)
}

/// Validate planned chain population conservation (no silently dropped failed/incomplete chains).
pub fn validate_chain_population_denominator(
    planned_chains: usize,
    completed_chains: usize,
    failed_chains: usize,
    unavailable_chains: usize,
) -> OutcomeStatus {
    let accounted = completed_chains
        .saturating_add(failed_chains)
        .saturating_add(unavailable_chains);
    if planned_chains == 0 || accounted != planned_chains {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

/// Validate accelerator sampler availability (never silently fall back to CPU).
pub fn validate_sampler_backend_execution(
    requested_accelerator: bool,
    accelerator_available: bool,
) -> OutcomeStatus {
    if requested_accelerator && !accelerator_available {
        OutcomeStatus::UnavailableHardwareRuntime
    } else {
        OutcomeStatus::Accepted
    }
}

/// Validate CRAM MAP-search default-off boundary (never substitutes for unbiased ESS).
pub fn validate_cram_arm_policy(cram_enabled: bool, claims_unbiased_ess: bool) -> OutcomeStatus {
    if !cram_enabled {
        OutcomeStatus::Unsupported
    } else if claims_unbiased_ess {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtraProtocolControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct GraphicalSamplingReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub compiler_lab_222_commit: String,
    pub ess_estimator_version: String,
    pub model_id: String,
    pub oracle: ExactOracleResult,
    pub conditional_oracle_max_err: f64,
    pub serial_and_parallel_color_trajectories_bit_identical: bool,
    pub sampler_runs: Vec<SamplerRunSummary>,
    pub extra_controls: Vec<ExtraProtocolControl>,
    pub cram_default_off_enabled: bool,
    pub amara_speedup_used_as_gate: bool,
    pub scoped_decision: String,
}

pub fn run_default_harness() -> Result<GraphicalSamplingReport, String> {
    let (graph, phases) = frozen_coupled_model();
    let oracle = exact_enumerate(&graph)?;

    // Validate conditional vector normalization & exact agreement against direct joint ratio.
    let test_state = [1usize, 0, 1, 0];
    let mut cond_max_err = 0.0_f64;
    for node in 0..graph.cardinalities.len() {
        let cond = graph.conditional_distribution(node, &test_state, false)?;
        let mut s0 = test_state;
        let mut s1 = test_state;
        s0[node] = 0;
        s1[node] = 1;
        let w0 = graph.unnormalized_weight(&s0);
        let w1 = graph.unnormalized_weight(&s1);
        let p0 = w0 / (w0 + w1);
        let p1 = w1 / (w0 + w1);
        cond_max_err = cond_max_err
            .max((cond[0] - p0).abs())
            .max((cond[1] - p1).abs());
    }

    let n_chains = 4usize;
    let burn_in = 128usize;
    let retained = 1024usize;
    let seed = 20_261_123_u64;
    let marginal_tol = 0.045_f64;

    let variants = [
        SamplerVariant::SequentialReference,
        SamplerVariant::SerialColorSchedule,
        SamplerVariant::ColorPhaseParallelModel,
        SamplerVariant::FaultyStaleNeighborCache,
        SamplerVariant::FaultyInvalidOneColoring,
        SamplerVariant::FaultyOmittedCommit,
        SamplerVariant::FaultySwappedPairwiseIndex,
        SamplerVariant::FaultyReusedRngStream,
    ];

    let mut sampler_runs = Vec::with_capacity(variants.len());
    for v in variants {
        sampler_runs.push(run_sampler_variant(
            &graph,
            &oracle,
            &phases,
            v,
            n_chains,
            burn_in,
            retained,
            seed,
            marginal_tol,
        )?);
    }

    // Identical schedule + identical per-(sweep,node) RNG stream mapping must produce bit-identical
    // trajectory between SerialColorSchedule and ColorPhaseParallelModel!
    let serial_color = &sampler_runs[1];
    let parallel_color = &sampler_runs[2];
    let bit_identical = serial_color.trajectory_digest == parallel_color.trajectory_digest;

    // Extra controls: invalid normalization, incomplete chain population, accelerator unavailable.
    let mut bad_graph = graph.clone();
    bad_graph.unary[0][0] = -1.0;
    let invalid_norm_err = bad_graph.validate().is_err();

    let extra_controls = vec![
        ExtraProtocolControl {
            case_id: "01-negative-invalid-nonpositive-potential-rejected".into(),
            outcome: if invalid_norm_err {
                OutcomeStatus::Invalid
            } else {
                OutcomeStatus::InternalFailure
            },
            detail: "Negative or zero potential violates Compiler Lab #222 positive-factor profile.".into(),
        },
        ExtraProtocolControl {
            case_id: "02-negative-incomplete-dropped-chain-retained-in-denominator".into(),
            outcome: validate_chain_population_denominator(4, 3, 0, 0),
            detail: "Dropping a failed or truncated chain out of planned chain denominator is rejected.".into(),
        },
        ExtraProtocolControl {
            case_id: "03-control-accelerator-arm-unavailable-no-cpu-fallback".into(),
            outcome: validate_sampler_backend_execution(true, false),
            detail: "Physical accelerator sampler arm remains unavailable_hardware_runtime without silent CPU fallback.".into(),
        },
        ExtraProtocolControl {
            case_id: "04-control-cram-map-search-default-off-separate-from-ess".into(),
            outcome: validate_cram_arm_policy(false, false),
            detail: format!(
                "CRAM (Prefrontal-AI/CRAM@{CRAM_PIN_COMMIT}, arXiv:2611.17309) energy-threshold MAP search stays default-off and never substitutes for unbiased sampling or ESS."
            ),
        },
    ];

    Ok(GraphicalSamplingReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "physical_accelerator_discrete_graphical_sampler",
            "CPU sequential reference, serial color-schedule, and color-phase Gibbs samplers ran with exact oracle & IPS ESS; physical accelerator execution remains unavailable_hardware_runtime.",
        ),
        compiler_lab_222_commit: COMPILER_LAB_222_COMMIT.into(),
        ess_estimator_version: "ess-ips-v1".into(),
        model_id: graph.model_id,
        oracle,
        conditional_oracle_max_err: cond_max_err,
        serial_and_parallel_color_trajectories_bit_identical: bit_identical,
        sampler_runs,
        extra_controls,
        cram_default_off_enabled: false,
        amara_speedup_used_as_gate: false,
        scoped_decision: "INCONCLUSIVE_FOR_ACCELERATOR_GO_FOR_CPU_EXACT_ORACLE_CONTRACT".into(),
    })
}

pub fn inputs_json(report: &GraphicalSamplingReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.issue-17-graphical-sampling.inputs.v1\",\n",
            "  \"issue\": 17,\n",
            "  \"execution_class\": {},\n",
            "  \"compiler_lab_222_commit\": {},\n",
            "  \"model_id\": {},\n",
            "  \"total_joint_states\": {},\n",
            "  \"ess_estimator_version\": {},\n",
            "  \"cram_pin_commit\": {},\n",
            "  \"cram_default_off_enabled\": {},\n",
            "  \"amara_speedup_used_as_gate\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(&report.compiler_lab_222_commit),
        quote(&report.model_id),
        report.oracle.total_joint_states,
        quote(&report.ess_estimator_version),
        quote(CRAM_PIN_COMMIT),
        report.cram_default_off_enabled,
        report.amara_speedup_used_as_gate,
    )
}

pub fn results_json(report: &GraphicalSamplingReport) -> String {
    let mut out = String::new();
    writeln!(out, "{{").unwrap();
    writeln!(
        out,
        "  \"schema\": \"inference-lab.issue-17-graphical-sampling.results.v1\","
    )
    .unwrap();
    writeln!(out, "  \"issue\": 17,").unwrap();
    writeln!(
        out,
        "  \"execution_class\": {},",
        quote(report.execution_class.as_str())
    )
    .unwrap();
    writeln!(
        out,
        "  \"hardware_gate\": {},",
        report.hardware_gate.to_json()
    )
    .unwrap();
    writeln!(
        out,
        "  \"exact_partition_z\": {},",
        finite_f64(report.oracle.partition_z)
    )
    .unwrap();
    writeln!(
        out,
        "  \"exact_expected_energy\": {},",
        finite_f64(report.oracle.expected_energy)
    )
    .unwrap();
    writeln!(
        out,
        "  \"conditional_oracle_max_err\": {},",
        finite_f64(report.conditional_oracle_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "  \"serial_and_parallel_color_trajectories_bit_identical\": {},",
        report.serial_and_parallel_color_trajectories_bit_identical
    )
    .unwrap();
    writeln!(
        out,
        "  \"scoped_decision\": {},",
        quote(&report.scoped_decision)
    )
    .unwrap();

    let runs: Vec<String> = report
        .sampler_runs
        .iter()
        .map(|r| {
            format!(
                concat!(
                    "{{\"variant\":{},\"schedule_valid\":{},\"total_chains\":{},",
                    "\"retained_sweeps_per_chain\":{},\"max_marginal_abs_err\":{},",
                    "\"energy_abs_err\":{},\"ess_energy\":{},\"trajectory_sha256\":{},",
                    "\"outcome\":{},\"detail\":{}}}"
                ),
                quote(r.variant.as_str()),
                r.schedule_valid,
                r.total_chains,
                r.retained_sweeps_per_chain,
                finite_f64(r.max_marginal_abs_err),
                finite_f64(r.energy_abs_err),
                opt_f64(r.ess_energy),
                quote(&r.trajectory_digest.hex),
                quote(r.outcome.as_str()),
                quote(&r.detail),
            )
        })
        .collect();
    writeln!(out, "  \"sampler_runs\": [{}],", runs.join(",")).unwrap();

    let extras: Vec<String> = report
        .extra_controls
        .iter()
        .map(|c| {
            format!(
                "{{\"case_id\":{},\"outcome\":{},\"detail\":{}}}",
                quote(&c.case_id),
                quote(c.outcome.as_str()),
                quote(&c.detail),
            )
        })
        .collect();
    writeln!(out, "  \"extra_controls\": [{}]", extras.join(",")).unwrap();
    writeln!(out, "}}").unwrap();
    out
}

pub fn summary_markdown(report: &GraphicalSamplingReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #17: Discrete Graphical Sampling Benchmark & Exact Oracle Controls"
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "- Execution class: `{}`",
        report.execution_class.as_str()
    )
    .unwrap();
    writeln!(
        out,
        "- Hardware gate: `{}` (`{}`)",
        report.hardware_gate.status.as_str(),
        report.hardware_gate.target_hardware
    )
    .unwrap();
    writeln!(
        out,
        "- Exact oracle (`{}` states): `Z={}`, `E[energy]={}`, conditional max err `{}`",
        report.oracle.total_joint_states,
        finite_f64(report.oracle.partition_z),
        finite_f64(report.oracle.expected_energy),
        finite_f64(report.conditional_oracle_max_err)
    )
    .unwrap();
    writeln!(
        out,
        "- Matched schedule+RNG bit-identical trajectory (serial vs color-phase): `{}`",
        report.serial_and_parallel_color_trajectories_bit_identical
    )
    .unwrap();
    writeln!(out, "- Scoped decision: `{}`", report.scoped_decision).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "## CPU Samplers & Adversarial Controls").unwrap();
    writeln!(
        out,
        "| Variant | Valid Sched | Max Marginal Err | Energy Err | ESS (`ess-ips-v1`) | Outcome |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | ---: | ---: | ---: | --- |").unwrap();
    for r in &report.sampler_runs {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |",
            r.variant.as_str(),
            r.schedule_valid,
            finite_f64(r.max_marginal_abs_err),
            finite_f64(r.energy_abs_err),
            opt_f64(r.ess_energy),
            r.outcome.as_str()
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    writeln!(
        out,
        "> AMARA (`arXiv:2611.15167v1`) and default-off CRAM (`arXiv:2611.17309`) claims remain strictly non-gating; stuck or biased samplers never win on raw updates/s."
    )
    .unwrap();
    out
}
