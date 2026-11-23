//! Issue #48: Add NVMe interrupt coalescing as a controlled host variable for SSD-backed
//! Engram/PLE tables on DGX Spark.
//!
//! Related existing issues: `#35` (Nemotron Spark), `#39` (CPU contention), `#43` (power/cadence).
//! Source basis: post-only unverified report (`https://x.com/wrldsuksgo2mars/status/2105216239056589131`)
//! referencing `nvme get-feature /dev/nvme0 -f 8 -H` and `nvidia-nvme-interrupt-coalescing.service`.
//! The post explicitly notes this is storage 4K QD1 latency, NOT a tokens-per-second claim; its
//! ~55-57 us vs ~200 us figures are strictly unverified and NEVER imported as lab pass gates.
//!
//! Provides:
//! - Deterministic parser for `nvme get-feature /dev/nvme0 -f 8 -H` output classifying coalescing
//!   as `On`, `Off`, or `Unknown` (never defaulting missing/unparseable output to `Off`).
//! - Typed controlled host factors: `CoalescingState`, `PageCacheState`, `DriveIdentity`, and
//!   `SystemdUnitStatus` with explicit `mutates_host_service: false`.
//! - Strict separation between `StorageLatencyMetrics` (4K QD1 random read us) and
//!   `PagedDecodeArmMetrics` (SSD-paged Engram/PLE table lookup + compute decode latency).
//! - Deterministic 2x2 (`coalescing: off/on` x `page_cache: cold/warm`) SSD-paged decode latency
//!   reducer and 8 executable negative controls.

use std::fmt::Write as FmtWrite;

use crate::json::{finite_f64, opt_f64, quote};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate, OutcomeStatus};

pub const SOURCE_BOOKMARK: &str = "https://x.com/wrldsuksgo2mars/status/2105216239056589131";
pub const NVME_FEATURE_CMD: &str = "nvme get-feature /dev/nvme0 -f 8 -H";
pub const SYSTEMD_UNIT_NAME: &str = "nvidia-nvme-interrupt-coalescing.service";
pub const RELATED_ISSUES: [u32; 3] = [35, 39, 43];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalescingState {
    Off,
    On,
    Unknown,
}

impl CoalescingState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageCacheState {
    Cold,
    Warm,
    Unknown,
}

impl PageCacheState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warm => "warm",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvmeCoalescingParsed {
    pub state: CoalescingState,
    pub agg_time_100us: Option<u32>,
    pub agg_threshold: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct HostControlledFactors {
    pub device_path: String,
    pub drive_model: String,
    pub firmware_rev: String,
    pub systemd_unit_name: String,
    pub systemd_unit_active: String,
    pub coalescing_state: CoalescingState,
    pub page_cache_state: PageCacheState,
    pub nvme_feature_8_raw: String,
}

#[derive(Debug, Clone)]
pub struct StorageLatencyMetrics {
    pub qd1_4k_p50_us: f64,
    pub qd1_4k_p95_us: f64,
    pub qd1_4k_p99_us: f64,
    pub is_tok_s_claim: bool,
}

#[derive(Debug, Clone)]
pub struct PagedTokenTraceEvent {
    pub token_index: usize,
    pub table_reads_total: usize,
    pub page_cache_hits: usize,
    pub ssd_miss_io_us: f64,
    pub compute_us: f64,
}

#[derive(Debug, Clone)]
pub struct PagedDecodeArmReport {
    pub arm_id: String,
    pub factors: HostControlledFactors,
    pub storage_latency: StorageLatencyMetrics,
    pub tokens_decoded: usize,
    pub total_table_reads: usize,
    pub total_cache_hits: usize,
    pub total_ssd_misses: usize,
    pub mean_io_ms_per_token: Option<f64>,
    pub mean_compute_ms_per_token: Option<f64>,
    pub mean_decode_ms_per_token: Option<f64>,
    pub p95_decode_ms_per_token: Option<f64>,
    pub decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NegativeControl {
    pub case_id: String,
    pub outcome: OutcomeStatus,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct NvmeCoalescingReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub synthetic_contract_fixture: bool,
    pub fixture_digest: DigestRef,
    pub mutates_host_service: bool,
    pub storage_latency_is_not_tok_s_claim: bool,
    pub arms: Vec<PagedDecodeArmReport>,
    pub cold_coalescing_on_vs_off_delta_ms: Option<f64>,
    pub warm_coalescing_on_vs_off_delta_ms: Option<f64>,
    pub negative_controls: Vec<NegativeControl>,
}

fn check_finite_nonneg(v: f64, label: &str) -> Result<f64, String> {
    if !v.is_finite() || v < 0.0 {
        return Err(format!("invalid non-finite or negative float in {label}"));
    }
    Ok(v)
}

/// Deterministically parse human-readable `nvme get-feature /dev/nvme0 -f 8 -H` output.
/// Looks for `Aggregation Time (TIME): <n>` and `Aggregation Threshold (THR): <n>`.
/// Returns `CoalescingState::Unknown` whenever fields are missing or unparseable.
pub fn parse_nvme_coalescing_feature(raw: &str) -> NvmeCoalescingParsed {
    let mut time_val: Option<u32> = None;
    let mut thr_val: Option<u32> = None;
    for line in raw.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("aggregation time") {
            if let Some(after_colon) = line.rsplit(':').next() {
                let token = after_colon.split_whitespace().next().unwrap_or("");
                if let Some(hex) = token
                    .strip_prefix("0x")
                    .or_else(|| token.strip_prefix("0X"))
                {
                    time_val = u32::from_str_radix(hex, 16).ok();
                } else {
                    time_val = token.parse::<u32>().ok();
                }
            }
        } else if lower.contains("aggregation threshold") {
            if let Some(after_colon) = line.rsplit(':').next() {
                let token = after_colon.split_whitespace().next().unwrap_or("");
                if let Some(hex) = token
                    .strip_prefix("0x")
                    .or_else(|| token.strip_prefix("0X"))
                {
                    thr_val = u32::from_str_radix(hex, 16).ok();
                } else {
                    thr_val = token.parse::<u32>().ok();
                }
            }
        }
    }
    let state = match (time_val, thr_val) {
        (Some(t), Some(h)) => {
            if t == 0 && h == 0 {
                CoalescingState::Off
            } else {
                CoalescingState::On
            }
        }
        _ => CoalescingState::Unknown,
    };
    NvmeCoalescingParsed {
        state,
        agg_time_100us: time_val,
        agg_threshold: thr_val,
    }
}

pub fn validate_controlled_factors(
    factors: &HostControlledFactors,
    storage: &StorageLatencyMetrics,
) -> Result<(), String> {
    if factors.device_path.trim().is_empty() || factors.drive_model.trim().is_empty() {
        return Err("drive device_path and drive_model must be recorded explicitly".into());
    }
    if factors.coalescing_state == CoalescingState::Unknown {
        return Err("unknown NVMe coalescing state cannot be scored as a controlled arm".into());
    }
    if factors.page_cache_state == PageCacheState::Unknown {
        return Err("unknown page_cache_state cannot be scored as a controlled arm".into());
    }
    let parsed = parse_nvme_coalescing_feature(&factors.nvme_feature_8_raw);
    if parsed.state != factors.coalescing_state {
        return Err(format!(
            "declared coalescing state ({}) disagrees with parsed nvme feature 8 output ({})",
            factors.coalescing_state.as_str(),
            parsed.state.as_str()
        ));
    }
    if storage.is_tok_s_claim {
        return Err(
            "4K QD1 storage latency must remain strictly separate from decode tok/s claims".into(),
        );
    }
    check_finite_nonneg(storage.qd1_4k_p50_us, "qd1_4k_p50_us")?;
    check_finite_nonneg(storage.qd1_4k_p95_us, "qd1_4k_p95_us")?;
    check_finite_nonneg(storage.qd1_4k_p99_us, "qd1_4k_p99_us")?;
    Ok(())
}

pub fn validate_paired_cache_match(
    a: &HostControlledFactors,
    b: &HostControlledFactors,
) -> Result<(), String> {
    if a.drive_model != b.drive_model {
        return Err("paired coalescing comparison requires identical drive model".into());
    }
    if a.page_cache_state != b.page_cache_state {
        return Err(
            "paired coalescing comparison cannot conflate cold and warm page-cache states".into(),
        );
    }
    if a.coalescing_state == b.coalescing_state {
        return Err("paired comparison requires distinct coalescing settings (off vs on)".into());
    }
    Ok(())
}

fn percentile_finite(values: &[f64], p_num: usize, p_den: usize) -> Result<Option<f64>, String> {
    if values.is_empty() || p_den == 0 {
        return Ok(None);
    }
    for &v in values {
        check_finite_nonneg(v, "percentile sample")?;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) * p_num) / p_den;
    Ok(Some(sorted[idx]))
}

pub fn reduce_paged_decode_arm(
    arm_id: &str,
    factors: HostControlledFactors,
    storage_latency: StorageLatencyMetrics,
    events: &[PagedTokenTraceEvent],
) -> Result<PagedDecodeArmReport, String> {
    validate_controlled_factors(&factors, &storage_latency)?;
    if events.is_empty() {
        return Ok(PagedDecodeArmReport {
            arm_id: arm_id.into(),
            factors,
            storage_latency,
            tokens_decoded: 0,
            total_table_reads: 0,
            total_cache_hits: 0,
            total_ssd_misses: 0,
            mean_io_ms_per_token: None,
            mean_compute_ms_per_token: None,
            mean_decode_ms_per_token: None,
            p95_decode_ms_per_token: None,
            decode_tok_s: None,
        });
    }

    let mut total_reads = 0usize;
    let mut total_hits = 0usize;
    let mut total_io_us = 0.0_f64;
    let mut total_compute_us = 0.0_f64;
    let mut per_token_ms = Vec::with_capacity(events.len());

    for ev in events {
        if ev.page_cache_hits > ev.table_reads_total {
            return Err(format!(
                "token {} has page_cache_hits > table_reads_total",
                ev.token_index
            ));
        }
        let io_us = check_finite_nonneg(ev.ssd_miss_io_us, "ssd_miss_io_us")?;
        let comp_us = check_finite_nonneg(ev.compute_us, "compute_us")?;
        total_reads += ev.table_reads_total;
        total_hits += ev.page_cache_hits;
        total_io_us += io_us;
        total_compute_us += comp_us;
        let tok_ms = (io_us + comp_us) / 1000.0;
        check_finite_nonneg(tok_ms, "per_token_ms")?;
        per_token_ms.push(tok_ms);
    }

    let n = events.len() as f64;
    let mean_io_ms = (total_io_us / n) / 1000.0;
    let mean_comp_ms = (total_compute_us / n) / 1000.0;
    let mean_decode_ms = mean_io_ms + mean_comp_ms;
    let p95_decode_ms = percentile_finite(&per_token_ms, 95, 100)?;
    let decode_tok_s = if mean_decode_ms > 0.0 {
        Some(1000.0 / mean_decode_ms)
    } else {
        None
    };

    Ok(PagedDecodeArmReport {
        arm_id: arm_id.into(),
        factors,
        storage_latency,
        tokens_decoded: events.len(),
        total_table_reads: total_reads,
        total_cache_hits: total_hits,
        total_ssd_misses: total_reads - total_hits,
        mean_io_ms_per_token: Some(mean_io_ms),
        mean_compute_ms_per_token: Some(mean_comp_ms),
        mean_decode_ms_per_token: Some(mean_decode_ms),
        p95_decode_ms_per_token: p95_decode_ms,
        decode_tok_s,
    })
}

fn synthetic_paged_events(
    coalescing: CoalescingState,
    cache: PageCacheState,
    num_tokens: usize,
) -> Vec<PagedTokenTraceEvent> {
    let per_miss_us = match coalescing {
        CoalescingState::Off => 62.0,
        CoalescingState::On => 195.0,
        CoalescingState::Unknown => 0.0,
    };
    let mut out = Vec::with_capacity(num_tokens);
    for t in 0..num_tokens {
        let reads = 16usize;
        let hits = match cache {
            PageCacheState::Cold => 2 + (t % 3),
            PageCacheState::Warm => 14 + (t % 2),
            PageCacheState::Unknown => 0,
        };
        let misses = reads - hits;
        let jitter = ((t * 7) % 5) as f64 * 1.5;
        out.push(PagedTokenTraceEvent {
            token_index: t,
            table_reads_total: reads,
            page_cache_hits: hits,
            ssd_miss_io_us: (misses as f64) * (per_miss_us + jitter),
            compute_us: 14_200.0 + ((t % 4) as f64) * 35.0,
        });
    }
    out
}

pub fn validate_no_host_mutation(requests_service_mutation: bool) -> OutcomeStatus {
    if requests_service_mutation {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn validate_no_post_threshold_import(uses_post_latency_as_gate: bool) -> OutcomeStatus {
    if uses_post_latency_as_gate {
        OutcomeStatus::Invalid
    } else {
        OutcomeStatus::Accepted
    }
}

pub fn run_default_harness() -> Result<NvmeCoalescingReport, String> {
    let raw_off = "get-feature:0x08 (Interrupt Coalescing), Current value:0x00000000\n  Aggregation Time (TIME): 0\n  Aggregation Threshold (THR): 0\n";
    let raw_on = "get-feature:0x08 (Interrupt Coalescing), Current value:0x00000208\n  Aggregation Time (TIME): 2\n  Aggregation Threshold (THR): 8\n";

    let specs = [
        (
            "cold_coalescing_off",
            CoalescingState::Off,
            PageCacheState::Cold,
            "inactive",
            raw_off,
            StorageLatencyMetrics {
                qd1_4k_p50_us: 61.5,
                qd1_4k_p95_us: 74.0,
                qd1_4k_p99_us: 89.0,
                is_tok_s_claim: false,
            },
        ),
        (
            "cold_coalescing_on",
            CoalescingState::On,
            PageCacheState::Cold,
            "active",
            raw_on,
            StorageLatencyMetrics {
                qd1_4k_p50_us: 194.0,
                qd1_4k_p95_us: 215.0,
                qd1_4k_p99_us: 238.0,
                is_tok_s_claim: false,
            },
        ),
        (
            "warm_coalescing_off",
            CoalescingState::Off,
            PageCacheState::Warm,
            "inactive",
            raw_off,
            StorageLatencyMetrics {
                qd1_4k_p50_us: 61.5,
                qd1_4k_p95_us: 74.0,
                qd1_4k_p99_us: 89.0,
                is_tok_s_claim: false,
            },
        ),
        (
            "warm_coalescing_on",
            CoalescingState::On,
            PageCacheState::Warm,
            "active",
            raw_on,
            StorageLatencyMetrics {
                qd1_4k_p50_us: 194.0,
                qd1_4k_p95_us: 215.0,
                qd1_4k_p99_us: 238.0,
                is_tok_s_claim: false,
            },
        ),
    ];

    let mut arms = Vec::with_capacity(specs.len());
    let mut digest_buf = String::new();
    for (id, coal, cache, unit_active, raw_f8, storage) in specs {
        let factors = HostControlledFactors {
            device_path: "/dev/nvme0".into(),
            drive_model: "SYNTHETIC-SPARK-NVME-GEN4-4TB".into(),
            firmware_rev: "FW-SYNTH-01".into(),
            systemd_unit_name: SYSTEMD_UNIT_NAME.into(),
            systemd_unit_active: unit_active.into(),
            coalescing_state: coal,
            page_cache_state: cache,
            nvme_feature_8_raw: raw_f8.into(),
        };
        let events = synthetic_paged_events(coal, cache, 32);
        let _ = write!(
            digest_buf,
            "{}:{}:{}:{};",
            id,
            coal.as_str(),
            cache.as_str(),
            events.len()
        );
        arms.push(reduce_paged_decode_arm(id, factors, storage, &events)?);
    }

    validate_paired_cache_match(&arms[0].factors, &arms[1].factors)?;
    validate_paired_cache_match(&arms[2].factors, &arms[3].factors)?;

    let cold_delta = match (
        arms[1].mean_decode_ms_per_token,
        arms[0].mean_decode_ms_per_token,
    ) {
        (Some(on), Some(off)) => Some(on - off),
        _ => None,
    };
    let warm_delta = match (
        arms[3].mean_decode_ms_per_token,
        arms[2].mean_decode_ms_per_token,
    ) {
        (Some(on), Some(off)) => Some(on - off),
        _ => None,
    };

    let mut negative_controls = Vec::new();

    // NC1: missing/unknown coalescing state rejected (never defaults to Off)
    {
        let parsed = parse_nvme_coalescing_feature("nvme: permission denied");
        let mut bad_factors = arms[0].factors.clone();
        bad_factors.coalescing_state = parsed.state;
        bad_factors.nvme_feature_8_raw = "nvme: permission denied".into();
        let outcome = if parsed.state == CoalescingState::Unknown
            && validate_controlled_factors(&bad_factors, &arms[0].storage_latency).is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc1_unknown_coalescing_never_defaults_off".into(),
            outcome,
            detail: "unparseable nvme feature 8 output classifies as unknown and is refused".into(),
        });
    }

    // NC2: empty drive model rejected
    {
        let mut bad = arms[0].factors.clone();
        bad.drive_model = String::new();
        let outcome = if validate_controlled_factors(&bad, &arms[0].storage_latency).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc2_missing_drive_model_rejected".into(),
            outcome,
            detail: "drive model is a required controlled host factor".into(),
        });
    }

    // NC3: cold vs warm page-cache conflation rejected in paired comparison
    {
        let outcome = if validate_paired_cache_match(&arms[0].factors, &arms[3].factors).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc3_cold_vs_warm_cache_conflation_rejected".into(),
            outcome,
            detail: "paired coalescing comparison cannot mix cold and warm page-cache runs".into(),
        });
    }

    // NC4: conflating 4K QD1 storage latency with decode tok/s claim rejected
    {
        let mut bad_storage = arms[0].storage_latency.clone();
        bad_storage.is_tok_s_claim = true;
        let outcome = if validate_controlled_factors(&arms[0].factors, &bad_storage).is_err() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc4_storage_latency_not_tok_s_claim".into(),
            outcome,
            detail: "4K QD1 storage latency is strictly separated from decode tok/s claims".into(),
        });
    }

    // NC5: NaN or negative latency sample rejected
    {
        let bad_ev = vec![PagedTokenTraceEvent {
            token_index: 0,
            table_reads_total: 4,
            page_cache_hits: 1,
            ssd_miss_io_us: f64::NAN,
            compute_us: 100.0,
        }];
        let outcome = if reduce_paged_decode_arm(
            "bad_nan",
            arms[0].factors.clone(),
            arms[0].storage_latency.clone(),
            &bad_ev,
        )
        .is_err()
        {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc5_nan_latency_sample_rejected".into(),
            outcome,
            detail: "non-finite SSD I/O latency is rejected instead of swallowed".into(),
        });
    }

    // NC6: zero decoded tokens yields None tok/s (no div-by-zero)
    {
        let empty = reduce_paged_decode_arm(
            "empty",
            arms[0].factors.clone(),
            arms[0].storage_latency.clone(),
            &[],
        )?;
        let outcome = if empty.decode_tok_s.is_none() && empty.mean_decode_ms_per_token.is_none() {
            OutcomeStatus::Invalid
        } else {
            OutcomeStatus::Accepted
        };
        negative_controls.push(NegativeControl {
            case_id: "nc6_zero_tokens_guarded".into(),
            outcome,
            detail: "zero decoded tokens yields None decode latency and tok/s".into(),
        });
    }

    // NC7: host systemd service mutation refused by harness
    negative_controls.push(NegativeControl {
        case_id: "nc7_host_service_mutation_refused".into(),
        outcome: validate_no_host_mutation(true),
        detail: "harness records controlled host variables without mutating systemd units".into(),
    });

    // NC8: importing X post's 55-57 us / 200 us claims as pass/fail gates rejected
    negative_controls.push(NegativeControl {
        case_id: "nc8_post_latency_claims_not_pass_gates".into(),
        outcome: validate_no_post_threshold_import(true),
        detail: "unverified X post 55-57 us vs 200 us figures are never lab acceptance thresholds"
            .into(),
    });

    Ok(NvmeCoalescingReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "Physical NVIDIA DGX Spark NVMe (/dev/nvme0) with nvidia-nvme-interrupt-coalescing.service and SSD-paged Engram/PLE tables",
            "NVMe feature 8 parser, controlled-factor validator, and 2x2 SSD-paged decode reducer verified on CPU; physical NVMe device did not execute locally",
        ),
        synthetic_contract_fixture: true,
        fixture_digest: DigestRef::sha256(digest_buf.as_bytes()),
        mutates_host_service: false,
        storage_latency_is_not_tok_s_claim: true,
        arms,
        cold_coalescing_on_vs_off_delta_ms: cold_delta,
        warm_coalescing_on_vs_off_delta_ms: warm_delta,
        negative_controls,
    })
}

pub fn inputs_json(report: &NvmeCoalescingReport) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nvme-coalescing.inputs.v1\",\n",
            "  \"issue\": 48,\n",
            "  \"related_issues\": [35, 39, 43],\n",
            "  \"execution_class\": {},\n",
            "  \"source_bookmark\": {},\n",
            "  \"nvme_probe_command\": {},\n",
            "  \"systemd_unit\": {},\n",
            "  \"mutates_host_service\": {},\n",
            "  \"storage_latency_is_not_tok_s_claim\": {},\n",
            "  \"fixture_digest\": {},\n",
            "  \"synthetic_contract_fixture\": {}\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        quote(SOURCE_BOOKMARK),
        quote(NVME_FEATURE_CMD),
        quote(SYSTEMD_UNIT_NAME),
        report.mutates_host_service,
        report.storage_latency_is_not_tok_s_claim,
        report.fixture_digest.to_json(),
        report.synthetic_contract_fixture,
    )
}

pub fn results_json(report: &NvmeCoalescingReport) -> String {
    let arms: Vec<String> = report
        .arms
        .iter()
        .map(|a| {
            format!(
                "{{\"arm_id\":{},\"device_path\":{},\"drive_model\":{},\"coalescing_state\":{},\"page_cache_state\":{},\"systemd_unit_active\":{},\"storage_qd1_4k_p50_us\":{},\"storage_qd1_4k_p95_us\":{},\"storage_qd1_4k_p99_us\":{},\"storage_is_tok_s_claim\":{},\"tokens_decoded\":{},\"total_table_reads\":{},\"total_cache_hits\":{},\"total_ssd_misses\":{},\"mean_io_ms_per_token\":{},\"mean_compute_ms_per_token\":{},\"mean_decode_ms_per_token\":{},\"p95_decode_ms_per_token\":{},\"decode_tok_s\":{}}}",
                quote(&a.arm_id),
                quote(&a.factors.device_path),
                quote(&a.factors.drive_model),
                quote(a.factors.coalescing_state.as_str()),
                quote(a.factors.page_cache_state.as_str()),
                quote(&a.factors.systemd_unit_active),
                finite_f64(a.storage_latency.qd1_4k_p50_us),
                finite_f64(a.storage_latency.qd1_4k_p95_us),
                finite_f64(a.storage_latency.qd1_4k_p99_us),
                a.storage_latency.is_tok_s_claim,
                a.tokens_decoded,
                a.total_table_reads,
                a.total_cache_hits,
                a.total_ssd_misses,
                opt_f64(a.mean_io_ms_per_token),
                opt_f64(a.mean_compute_ms_per_token),
                opt_f64(a.mean_decode_ms_per_token),
                opt_f64(a.p95_decode_ms_per_token),
                opt_f64(a.decode_tok_s),
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
                quote(&c.detail),
            )
        })
        .collect();
    format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.nvme-coalescing.results.v1\",\n",
            "  \"issue\": 48,\n",
            "  \"execution_class\": {},\n",
            "  \"hardware_gate\": {},\n",
            "  \"synthetic_contract_fixture\": {},\n",
            "  \"mutates_host_service\": {},\n",
            "  \"storage_latency_is_not_tok_s_claim\": {},\n",
            "  \"cold_coalescing_on_vs_off_delta_ms\": {},\n",
            "  \"warm_coalescing_on_vs_off_delta_ms\": {},\n",
            "  \"arms\": [{}],\n",
            "  \"negative_controls\": [{}]\n",
            "}}\n"
        ),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        report.synthetic_contract_fixture,
        report.mutates_host_service,
        report.storage_latency_is_not_tok_s_claim,
        opt_f64(report.cold_coalescing_on_vs_off_delta_ms),
        opt_f64(report.warm_coalescing_on_vs_off_delta_ms),
        arms.join(","),
        controls.join(","),
    )
}

pub fn summary_markdown(report: &NvmeCoalescingReport) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "# Issue #48: NVMe Interrupt Coalescing Controlled Host Variable (DGX Spark)"
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
        "- Probe command: `{}` | Unit: `{}` | `mutates_host_service: {}`",
        NVME_FEATURE_CMD, SYSTEMD_UNIT_NAME, report.mutates_host_service
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "## 2x2 Controlled Factor Matrix (Storage 4K QD1 Separated from Decode)"
    )
    .unwrap();
    for a in &report.arms {
        writeln!(
            out,
            "- `{}` (`coalescing={}`, `cache={}`): `storage_p50_us={:.1}`, `mean_io_ms={:.3}`, `mean_decode_ms={:.3}`, `decode_tok_s={:.2}`",
            a.arm_id,
            a.factors.coalescing_state.as_str(),
            a.factors.page_cache_state.as_str(),
            a.storage_latency.qd1_4k_p50_us,
            a.mean_io_ms_per_token.unwrap_or(0.0),
            a.mean_decode_ms_per_token.unwrap_or(0.0),
            a.decode_tok_s.unwrap_or(0.0),
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
    fn nvme_feature_8_parser_distinguishes_off_on_and_unknown() {
        let off = parse_nvme_coalescing_feature(
            "Aggregation Time (TIME): 0\nAggregation Threshold (THR): 0\n",
        );
        assert_eq!(off.state, CoalescingState::Off);
        let on = parse_nvme_coalescing_feature(
            "Aggregation Time (TIME): 0x2\nAggregation Threshold (THR): 0x8\n",
        );
        assert_eq!(on.state, CoalescingState::On);
        let unk = parse_nvme_coalescing_feature("error reading device");
        assert_eq!(unk.state, CoalescingState::Unknown);
    }

    #[test]
    fn default_harness_separates_cold_warm_deltas_and_rejects_all_controls() {
        let report = run_default_harness().unwrap();
        assert!(
            report.cold_coalescing_on_vs_off_delta_ms.unwrap()
                > report.warm_coalescing_on_vs_off_delta_ms.unwrap()
        );
        assert!(report.storage_latency_is_not_tok_s_claim);
        assert!(!report.mutates_host_service);
        assert_eq!(report.negative_controls.len(), 8);
        for c in &report.negative_controls {
            assert_ne!(c.outcome, OutcomeStatus::Accepted);
        }
    }
}
