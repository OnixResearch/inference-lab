//! Issue #40: Host and GPU trace correlation qualification before attributing inference stalls.
//!
//! Enforces run/module identity binding, clock-domain mapping with uncertainty bounds,
//! scoped `(context_id, stream_id, correlation_id)` joins, batched-kernel non-attribution,
//! event-loss completeness checks, bounded deferred metadata association, and explicit
//! negative controls demonstrating that incomplete traces cannot substantiate causal paths.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use crate::json::{opt_i64, opt_str, opt_u64, quote, str_array};
use crate::provenance::{DigestRef, ExecutionClass, HardwareGate};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockDomain {
    CpuKtimeNs,
    DeviceGlobalTimerNs,
    ClientMonotonicNs,
}

impl ClockDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuKtimeNs => "cpu_ktime_ns",
            Self::DeviceGlobalTimerNs => "device_global_timer_ns",
            Self::ClientMonotonicNs => "client_monotonic_ns",
        }
    }
}

/// Calibrated mapping from a source clock domain into `CpuKtimeNs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClockMapping {
    pub source_domain: ClockDomain,
    pub target_domain: ClockDomain,
    pub offset_ns: i64,
    pub calibration_interval_ns: u64,
    pub uncertainty_ns: u64,
}

impl ClockMapping {
    pub fn convert(&self, ts_ns: i64) -> i64 {
        ts_ns.saturating_add(self.offset_ns)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRunBinding {
    pub run_id: String,
    pub workload_id: String,
    pub process_pid: u32,
    pub engine_module_digest: DigestRef,
    pub device_id: u32,
    pub driver_version: String,
    pub runtime_version: String,
    pub exporter_version: String,
    pub native_artifact_digest: DigestRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceMode {
    Untraced,
    NativeDeviceTraced,
    LinuxEventTraced,
    Combined,
}

impl TraceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Untraced => "untraced",
            Self::NativeDeviceTraced => "native_device_traced",
            Self::LinuxEventTraced => "linux_event_traced",
            Self::Combined => "combined",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostApiEvent {
    pub event_id: String,
    pub run_id: String,
    pub module_digest: DigestRef,
    pub api_name: String,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u64,
    pub cuda_graph_id: Option<u64>,
    pub domain: ClockDomain,
    pub start_ns: i64,
    pub end_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceKernelEvent {
    pub event_id: String,
    pub run_id: String,
    pub device_id: u32,
    pub kernel_name: String,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u64,
    pub cuda_graph_id: Option<u64>,
    pub domain: ClockDomain,
    pub start_ns: i64,
    pub end_ns: i64,
    pub batched_request_count: u32,
    pub explicit_request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxSchedEvent {
    pub event_id: String,
    pub tid: u32,
    pub state: String, // "runnable" | "off_cpu"
    pub domain: ClockDomain,
    pub start_ns: i64,
    pub end_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestPhaseEvent {
    pub request_id: String,
    pub phase: String,
    pub domain: ClockDomain,
    pub start_ns: i64,
    pub end_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSseEvent {
    pub request_id: String,
    pub chunk_index: u32,
    pub domain: ClockDomain,
    pub arrival_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeferredMetadataRecord {
    pub record_id: String,
    pub correlation_id: u64,
    pub context_id: u32,
    pub stream_id: u32,
    pub arrival_delay_ns: u64,
    pub label: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectorCountStatus {
    SupportedZero,
    SupportedNonZero(u64),
    MissingData,
    UnsupportedCollector,
}

impl CollectorCountStatus {
    pub const fn status_str(self) -> &'static str {
        match self {
            Self::SupportedZero => "supported_zero",
            Self::SupportedNonZero(_) => "supported_nonzero",
            Self::MissingData => "missing_data",
            Self::UnsupportedCollector => "unsupported_collector",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletenessReport {
    pub dropped_events: CollectorCountStatus,
    pub buffer_overflows: u64,
    pub incomplete_buffers: u64,
    pub excluded_apis: Vec<String>,
    pub attach_detach_failures: u64,
    pub unsupported_collectors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceBundle {
    pub control_name: String,
    pub binding: TraceRunBinding,
    pub trace_mode: TraceMode,
    pub clock_mappings: Vec<ClockMapping>,
    pub deferred_metadata_window_ns: u64,
    pub host_events: Vec<HostApiEvent>,
    pub device_events: Vec<DeviceKernelEvent>,
    pub linux_sched_events: Vec<LinuxSchedEvent>,
    pub request_phases: Vec<RequestPhaseEvent>,
    pub client_sse_events: Vec<ClientSseEvent>,
    pub deferred_metadata: Vec<DeferredMetadataRecord>,
    pub completeness: CompletenessReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinedAttributionRow {
    pub device_event_id: String,
    pub host_event_id: Option<String>,
    pub kernel_name: String,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u64,
    pub cuda_graph_id: Option<u64>,
    pub host_enqueue_start_ktime_ns: Option<i64>,
    pub device_start_ktime_ns: Option<i64>,
    pub launch_to_device_delta_ns: Option<i64>,
    pub uncertainty_ns: Option<u64>,
    pub request_attribution: String,
    pub metadata_label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEvent {
    pub event_kind: String,
    pub source_event_id: String,
    pub label: String,
    pub source_clock_domain: ClockDomain,
    pub units: String,
    pub ktime_start_ns: Option<i64>,
    pub ktime_end_ns: Option<i64>,
    pub uncertainty_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceQualificationOutcome {
    pub control_name: String,
    pub causal_path_substantiated: bool,
    pub status: String,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub unresolved_metadata_ids: Vec<String>,
    pub unsupported_conclusions: Vec<String>,
    pub joined_rows: Vec<JoinedAttributionRow>,
    pub timeline: Vec<TimelineEvent>,
}

/// Qualify one trace bundle against strict identity, clock-domain, correlation-scope,
/// completeness, and deferred-metadata rules.
pub fn qualify_trace_bundle(bundle: &TraceBundle) -> TraceQualificationOutcome {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut unresolved_metadata_ids = Vec::new();
    let mut unsupported_conclusions = Vec::new();

    // 1. Completeness checks: any dropped events, overflow, incomplete buffer, or attach failure
    // prevents substantiating a complete causal path.
    match bundle.completeness.dropped_events {
        CollectorCountStatus::SupportedZero => {}
        CollectorCountStatus::SupportedNonZero(n) => {
            errors.push(format!("dropped_events_nonzero:{n}"));
        }
        CollectorCountStatus::MissingData => {
            errors.push("dropped_events_counter_missing_data".into());
        }
        CollectorCountStatus::UnsupportedCollector => {
            errors.push("dropped_events_unsupported_collector".into());
        }
    }
    if bundle.completeness.buffer_overflows > 0 {
        errors.push(format!(
            "buffer_overflows_nonzero:{}",
            bundle.completeness.buffer_overflows
        ));
    }
    if bundle.completeness.incomplete_buffers > 0 {
        errors.push(format!(
            "incomplete_buffers_nonzero:{}",
            bundle.completeness.incomplete_buffers
        ));
    }
    if bundle.completeness.attach_detach_failures > 0 {
        errors.push(format!(
            "attach_detach_failures_nonzero:{}",
            bundle.completeness.attach_detach_failures
        ));
    }
    if !bundle.completeness.unsupported_collectors.is_empty() {
        unsupported_conclusions.push(format!(
            "unsupported_collectors:{}",
            bundle.completeness.unsupported_collectors.join(",")
        ));
    }

    // Helper: map timestamp into CpuKtimeNs ONLY if domain matches or calibrated mapping exists.
    let map_to_ktime = |domain: ClockDomain, ts: i64| -> Result<(i64, u64), String> {
        if domain == ClockDomain::CpuKtimeNs {
            return Ok((ts, 0));
        }
        for m in &bundle.clock_mappings {
            if m.source_domain == domain && m.target_domain == ClockDomain::CpuKtimeNs {
                return Ok((m.convert(ts), m.uncertainty_ns));
            }
        }
        Err(format!("missing_clock_mapping_from_{}", domain.as_str()))
    };

    // 2. Index host events by scoped key (context_id, stream_id, correlation_id), detecting duplicates/order/binding issues.
    type ScopeKey = (u32, u32, u64);
    let mut host_by_scope: BTreeMap<ScopeKey, &HostApiEvent> = BTreeMap::new();
    let mut seen_host_ids = BTreeSet::new();
    let mut last_host_start: Option<i64> = None;

    for ev in &bundle.host_events {
        if ev.run_id != bundle.binding.run_id {
            errors.push(format!("wrong_run_id_on_host_event:{}", ev.event_id));
            continue;
        }
        if ev.module_digest != bundle.binding.engine_module_digest {
            errors.push(format!("wrong_module_digest_on_host_event:{}", ev.event_id));
            continue;
        }
        if !seen_host_ids.insert(ev.event_id.clone()) {
            warnings.push(format!("duplicate_host_event_id:{}", ev.event_id));
            continue;
        }
        if let Some(prev) = last_host_start {
            if ev.start_ns < prev {
                warnings.push(format!("out_of_order_host_event:{}", ev.event_id));
            }
        }
        last_host_start = Some(ev.start_ns);
        let key = (ev.context_id, ev.stream_id, ev.correlation_id);
        if host_by_scope.insert(key, ev).is_some() {
            errors.push(format!(
                "correlation_id_collision_within_scope:ctx={}:stream={}:corr={}",
                ev.context_id, ev.stream_id, ev.correlation_id
            ));
        }
    }

    // 3. Deferred metadata resolution within bounded window; late records stay unresolved.
    let mut resolved_meta: BTreeMap<ScopeKey, String> = BTreeMap::new();
    for meta in &bundle.deferred_metadata {
        if meta.arrival_delay_ns > bundle.deferred_metadata_window_ns {
            unresolved_metadata_ids.push(meta.record_id.clone());
            warnings.push(format!(
                "late_metadata_unresolved_not_snapped:{}",
                meta.record_id
            ));
        } else {
            let key = (meta.context_id, meta.stream_id, meta.correlation_id);
            resolved_meta.insert(key, meta.label.clone());
        }
    }

    // 4. Join device events strictly on scoped key and calibrated clocks.
    let mut joined_rows = Vec::with_capacity(bundle.device_events.len());
    for dev in &bundle.device_events {
        if dev.run_id != bundle.binding.run_id {
            errors.push(format!("wrong_run_id_on_device_event:{}", dev.event_id));
            continue;
        }
        if dev.device_id != bundle.binding.device_id {
            errors.push(format!(
                "swapped_or_wrong_device_id:expected={}:observed={}:event={}",
                bundle.binding.device_id, dev.device_id, dev.event_id
            ));
            continue;
        }

        let scope_key = (dev.context_id, dev.stream_id, dev.correlation_id);
        let matched_host = host_by_scope.get(&scope_key).copied();
        if matched_host.is_none() {
            errors.push(format!(
                "unmatched_device_event_scope_or_swapped_stream:{}",
                dev.event_id
            ));
        }

        let dev_mapped = map_to_ktime(dev.domain, dev.start_ns);
        let host_mapped = matched_host.map(|h| map_to_ktime(h.domain, h.start_ns));

        let (host_start_ktime, dev_start_ktime, delta_ns, uncertainty_ns) =
            match (host_mapped, dev_mapped) {
                (Some(Ok((hs, hu))), Ok((ds, du))) => {
                    let delta = ds.saturating_sub(hs);
                    if delta < -(hu.saturating_add(du) as i64) {
                        errors.push(format!(
                            "impossible_negative_launch_delta_clock_mismatch:{}",
                            dev.event_id
                        ));
                    }
                    (Some(hs), Some(ds), Some(delta), Some(hu.saturating_add(du)))
                }
                (Some(Err(e)), _) | (_, Err(e)) => {
                    errors.push(format!("clock_domain_error:{}:{e}", dev.event_id));
                    (None, None, None, None)
                }
                (None, Ok((ds, du))) => (None, Some(ds), None, Some(du)),
            };

        let request_attribution = if dev.batched_request_count > 1 {
            unsupported_conclusions.push(format!(
                "batched_kernel_per_request_attribution_unavailable:{}:batch={}",
                dev.event_id, dev.batched_request_count
            ));
            "unavailable_batched_multi_request".into()
        } else if let Some(req_id) = &dev.explicit_request_id {
            req_id.clone()
        } else {
            "unavailable_no_request_annotation".into()
        };

        joined_rows.push(JoinedAttributionRow {
            device_event_id: dev.event_id.clone(),
            host_event_id: matched_host.map(|h| h.event_id.clone()),
            kernel_name: dev.kernel_name.clone(),
            context_id: dev.context_id,
            stream_id: dev.stream_id,
            correlation_id: dev.correlation_id,
            cuda_graph_id: dev.cuda_graph_id,
            host_enqueue_start_ktime_ns: host_start_ktime,
            device_start_ktime_ns: dev_start_ktime,
            launch_to_device_delta_ns: delta_ns,
            uncertainty_ns,
            request_attribution,
            metadata_label: resolved_meta.get(&scope_key).cloned(),
        });
    }

    // 5. Build compact multi-layer timeline validating clock domains across ALL event streams
    // (host API, device execution, Linux sched, request phases, and client SSE arrivals).
    let mut timeline = Vec::new();
    for ev in &bundle.host_events {
        if ev.run_id != bundle.binding.run_id
            || ev.module_digest != bundle.binding.engine_module_digest
        {
            continue;
        }
        match (
            map_to_ktime(ev.domain, ev.start_ns),
            map_to_ktime(ev.domain, ev.end_ns),
        ) {
            (Ok((s, u)), Ok((e, _))) => timeline.push(TimelineEvent {
                event_kind: "host_api_enqueue".into(),
                source_event_id: ev.event_id.clone(),
                label: ev.api_name.clone(),
                source_clock_domain: ev.domain,
                units: "ns".into(),
                ktime_start_ns: Some(s),
                ktime_end_ns: Some(e),
                uncertainty_ns: Some(u),
            }),
            (Err(err), _) | (_, Err(err)) => {
                errors.push(format!("clock_domain_error_host:{}:{err}", ev.event_id));
            }
        }
    }
    for dev in &bundle.device_events {
        if dev.run_id != bundle.binding.run_id || dev.device_id != bundle.binding.device_id {
            continue;
        }
        if let (Ok((s, u)), Ok((e, _))) = (
            map_to_ktime(dev.domain, dev.start_ns),
            map_to_ktime(dev.domain, dev.end_ns),
        ) {
            timeline.push(TimelineEvent {
                event_kind: "device_execution_or_copy".into(),
                source_event_id: dev.event_id.clone(),
                label: dev.kernel_name.clone(),
                source_clock_domain: dev.domain,
                units: "ns".into(),
                ktime_start_ns: Some(s),
                ktime_end_ns: Some(e),
                uncertainty_ns: Some(u),
            });
        }
    }
    for sched in &bundle.linux_sched_events {
        match (
            map_to_ktime(sched.domain, sched.start_ns),
            map_to_ktime(sched.domain, sched.end_ns),
        ) {
            (Ok((s, u)), Ok((e, _))) => timeline.push(TimelineEvent {
                event_kind: "linux_sched".into(),
                source_event_id: sched.event_id.clone(),
                label: format!("tid={}:{}", sched.tid, sched.state),
                source_clock_domain: sched.domain,
                units: "ns".into(),
                ktime_start_ns: Some(s),
                ktime_end_ns: Some(e),
                uncertainty_ns: Some(u),
            }),
            (Err(err), _) | (_, Err(err)) => {
                errors.push(format!(
                    "clock_domain_error_linux_sched:{}:{err}",
                    sched.event_id
                ));
            }
        }
    }
    for phase in &bundle.request_phases {
        match (
            map_to_ktime(phase.domain, phase.start_ns),
            map_to_ktime(phase.domain, phase.end_ns),
        ) {
            (Ok((s, u)), Ok((e, _))) => timeline.push(TimelineEvent {
                event_kind: "request_phase".into(),
                source_event_id: format!("{}:{}", phase.request_id, phase.phase),
                label: phase.phase.clone(),
                source_clock_domain: phase.domain,
                units: "ns".into(),
                ktime_start_ns: Some(s),
                ktime_end_ns: Some(e),
                uncertainty_ns: Some(u),
            }),
            (Err(err), _) | (_, Err(err)) => {
                errors.push(format!(
                    "clock_domain_error_request_phase:{}:{err}",
                    phase.request_id
                ));
            }
        }
    }
    for sse in &bundle.client_sse_events {
        match map_to_ktime(sse.domain, sse.arrival_ns) {
            Ok((s, u)) => timeline.push(TimelineEvent {
                event_kind: "client_sse_arrival".into(),
                source_event_id: format!("{}:chunk-{}", sse.request_id, sse.chunk_index),
                label: format!("sse_chunk_{}", sse.chunk_index),
                source_clock_domain: sse.domain,
                units: "ns".into(),
                ktime_start_ns: Some(s),
                ktime_end_ns: Some(s),
                uncertainty_ns: Some(u),
            }),
            Err(err) => {
                errors.push(format!(
                    "clock_domain_error_client_sse:{}:chunk-{}:{err}",
                    sse.request_id, sse.chunk_index
                ));
            }
        }
    }
    timeline.sort_by_key(|ev| {
        (
            ev.ktime_start_ns.unwrap_or(i64::MAX),
            ev.event_kind.clone(),
            ev.source_event_id.clone(),
        )
    });

    let causal_path_substantiated = errors.is_empty() && unresolved_metadata_ids.is_empty();
    let status = if causal_path_substantiated {
        "qualified_positive_control".into()
    } else {
        "incomplete_cannot_substantiate".into()
    };

    TraceQualificationOutcome {
        control_name: bundle.control_name.clone(),
        causal_path_substantiated,
        status,
        errors,
        warnings,
        unresolved_metadata_ids,
        unsupported_conclusions,
        joined_rows,
        timeline,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerturbationRow {
    pub mode: TraceMode,
    pub elapsed_logical_units: u64,
    pub trace_bytes: u64,
    pub ranking_eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContractReport {
    pub execution_class: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub controls: Vec<TraceQualificationOutcome>,
    pub perturbation_matrix: Vec<PerturbationRow>,
}

fn base_positive_bundle() -> TraceBundle {
    let mod_digest = DigestRef::sha256(b"inference-lab:cupti-control-kernel:v1");
    let binding = TraceRunBinding {
        run_id: "run-nvidia-control-01".into(),
        workload_id: "known-kernel-copy-sync-v1".into(),
        process_pid: 4242,
        engine_module_digest: mod_digest.clone(),
        device_id: 0,
        driver_version: "560.35.03-contract-fixture".into(),
        runtime_version: "cuda-12.6-contract-fixture".into(),
        exporter_version: "nsys-2025.1-contract-fixture".into(),
        native_artifact_digest: DigestRef::sha256(b"native-qdrep-export-fixture-v1"),
    };
    TraceBundle {
        control_name: "positive-known-kernel-copy-sync".into(),
        binding,
        trace_mode: TraceMode::Combined,
        clock_mappings: vec![
            ClockMapping {
                source_domain: ClockDomain::DeviceGlobalTimerNs,
                target_domain: ClockDomain::CpuKtimeNs,
                offset_ns: 1_000_000,
                calibration_interval_ns: 10_000_000,
                uncertainty_ns: 250,
            },
            ClockMapping {
                source_domain: ClockDomain::ClientMonotonicNs,
                target_domain: ClockDomain::CpuKtimeNs,
                offset_ns: 0,
                calibration_interval_ns: 10_000_000,
                uncertainty_ns: 100,
            },
        ],
        deferred_metadata_window_ns: 5_000,
        host_events: vec![
            HostApiEvent {
                event_id: "host-h2d-1".into(),
                run_id: "run-nvidia-control-01".into(),
                module_digest: mod_digest.clone(),
                api_name: "cudaMemcpyAsync".into(),
                context_id: 1,
                stream_id: 7,
                correlation_id: 101,
                cuda_graph_id: None,
                domain: ClockDomain::CpuKtimeNs,
                start_ns: 10_000_000,
                end_ns: 10_002_000,
            },
            HostApiEvent {
                event_id: "host-gemm-2".into(),
                run_id: "run-nvidia-control-01".into(),
                module_digest: mod_digest.clone(),
                api_name: "cudaGraphLaunch".into(),
                context_id: 1,
                stream_id: 7,
                correlation_id: 102,
                cuda_graph_id: Some(9001),
                domain: ClockDomain::CpuKtimeNs,
                start_ns: 10_005_000,
                end_ns: 10_006_500,
            },
            // Same correlation_id=102 in a different context (context_id=2) tests scoped identity!
            HostApiEvent {
                event_id: "host-ctx2-batched-3".into(),
                run_id: "run-nvidia-control-01".into(),
                module_digest: mod_digest,
                api_name: "cudaLaunchKernel".into(),
                context_id: 2,
                stream_id: 8,
                correlation_id: 102,
                cuda_graph_id: None,
                domain: ClockDomain::CpuKtimeNs,
                start_ns: 10_020_000,
                end_ns: 10_021_000,
            },
        ],
        device_events: vec![
            DeviceKernelEvent {
                event_id: "dev-h2d-1".into(),
                run_id: "run-nvidia-control-01".into(),
                device_id: 0,
                kernel_name: "memcpy_h2d_prefill".into(),
                context_id: 1,
                stream_id: 7,
                correlation_id: 101,
                cuda_graph_id: None,
                domain: ClockDomain::DeviceGlobalTimerNs,
                start_ns: 9_003_000,
                end_ns: 9_008_000,
                batched_request_count: 1,
                explicit_request_id: Some("req-01".into()),
            },
            DeviceKernelEvent {
                event_id: "dev-gemm-2".into(),
                run_id: "run-nvidia-control-01".into(),
                device_id: 0,
                kernel_name: "graph_decode_gemm".into(),
                context_id: 1,
                stream_id: 7,
                correlation_id: 102,
                cuda_graph_id: Some(9001),
                domain: ClockDomain::DeviceGlobalTimerNs,
                start_ns: 9_009_000,
                end_ns: 9_019_000,
                batched_request_count: 1,
                explicit_request_id: Some("req-01".into()),
            },
            DeviceKernelEvent {
                event_id: "dev-batched-3".into(),
                run_id: "run-nvidia-control-01".into(),
                device_id: 0,
                kernel_name: "batched_attention_decode".into(),
                context_id: 2,
                stream_id: 8,
                correlation_id: 102,
                cuda_graph_id: None,
                domain: ClockDomain::DeviceGlobalTimerNs,
                start_ns: 9_022_000,
                end_ns: 9_035_000,
                batched_request_count: 4,
                explicit_request_id: None,
            },
        ],
        linux_sched_events: vec![LinuxSchedEvent {
            event_id: "sched-1".into(),
            tid: 4243,
            state: "runnable".into(),
            domain: ClockDomain::CpuKtimeNs,
            start_ns: 9_999_000,
            end_ns: 10_010_000,
        }],
        request_phases: vec![RequestPhaseEvent {
            request_id: "req-01".into(),
            phase: "decode".into(),
            domain: ClockDomain::CpuKtimeNs,
            start_ns: 10_000_000,
            end_ns: 10_025_000,
        }],
        client_sse_events: vec![ClientSseEvent {
            request_id: "req-01".into(),
            chunk_index: 0,
            domain: ClockDomain::ClientMonotonicNs,
            arrival_ns: 10_026_000,
        }],
        deferred_metadata: vec![DeferredMetadataRecord {
            record_id: "meta-1".into(),
            correlation_id: 102,
            context_id: 1,
            stream_id: 7,
            arrival_delay_ns: 1_200,
            label: "nvtx:decode_step_0".into(),
        }],
        completeness: CompletenessReport {
            dropped_events: CollectorCountStatus::SupportedZero,
            buffer_overflows: 0,
            incomplete_buffers: 0,
            excluded_apis: Vec::new(),
            attach_detach_failures: 0,
            unsupported_collectors: Vec::new(),
        },
    }
}

/// Run the positive control plus all 7 adversarial negative controls required by Issue #40.
pub fn run_default_harness() -> TraceContractReport {
    let positive = base_positive_bundle();

    // Negative 1: wrong run & module identity
    let mut wrong_identity = base_positive_bundle();
    wrong_identity.control_name = "neg-wrong-run-and-module-identity".into();
    wrong_identity.host_events[0].run_id = "run-other-99".into();
    wrong_identity.host_events[1].module_digest = DigestRef::sha256(b"wrong-module-binary");

    // Negative 2: unmapped clock domain (forbids raw CPU ktime vs device global timer subtraction)
    let mut wrong_clock = base_positive_bundle();
    wrong_clock.control_name = "neg-unmapped-clock-domain".into();
    wrong_clock.clock_mappings.clear();

    // Negative 3: swapped stream/device IDs
    let mut swapped_ids = base_positive_bundle();
    swapped_ids.control_name = "neg-swapped-stream-and-device-ids".into();
    swapped_ids.device_events[0].stream_id = 99;
    swapped_ids.device_events[1].device_id = 1;

    // Negative 4: duplicate & out-of-order records + correlation collision in same scope
    let mut dup_collision = base_positive_bundle();
    dup_collision.control_name = "neg-duplicate-out-of-order-and-collision".into();
    let mut dup = dup_collision.host_events[1].clone();
    dup.start_ns = 9_990_000; // out of order
    dup_collision.host_events.push(dup);
    let mut collide = dup_collision.host_events[0].clone();
    collide.event_id = "host-colliding-corr".into();
    dup_collision.host_events.push(collide);

    // Negative 5: buffer loss / dropped events
    let mut buffer_loss = base_positive_bundle();
    buffer_loss.control_name = "neg-dropped-events-and-overflow".into();
    buffer_loss.completeness.dropped_events = CollectorCountStatus::SupportedNonZero(17);
    buffer_loss.completeness.buffer_overflows = 1;

    // Negative 6: missing dropped-event counter (missing data != zero!)
    let mut missing_counter = base_positive_bundle();
    missing_counter.control_name = "neg-missing-drop-counter-not-zero".into();
    missing_counter.completeness.dropped_events = CollectorCountStatus::MissingData;

    // Negative 7: late metadata beyond deferred bound (must stay unresolved, never snapped)
    let mut late_meta = base_positive_bundle();
    late_meta.control_name = "neg-late-metadata-unresolved".into();
    late_meta.deferred_metadata.push(DeferredMetadataRecord {
        record_id: "meta-late-99".into(),
        correlation_id: 101,
        context_id: 1,
        stream_id: 7,
        arrival_delay_ns: 25_000,
        label: "nvtx:too_late".into(),
    });

    let controls = vec![
        qualify_trace_bundle(&positive),
        qualify_trace_bundle(&wrong_identity),
        qualify_trace_bundle(&wrong_clock),
        qualify_trace_bundle(&swapped_ids),
        qualify_trace_bundle(&dup_collision),
        qualify_trace_bundle(&buffer_loss),
        qualify_trace_bundle(&missing_counter),
        qualify_trace_bundle(&late_meta),
    ];

    // Performance ranking uses separate uninstrumented trials ONLY.
    let perturbation_matrix = vec![
        PerturbationRow {
            mode: TraceMode::Untraced,
            elapsed_logical_units: 1000,
            trace_bytes: 0,
            ranking_eligible: true,
        },
        PerturbationRow {
            mode: TraceMode::NativeDeviceTraced,
            elapsed_logical_units: 1042,
            trace_bytes: 16_384,
            ranking_eligible: false,
        },
        PerturbationRow {
            mode: TraceMode::LinuxEventTraced,
            elapsed_logical_units: 1018,
            trace_bytes: 8_192,
            ranking_eligible: false,
        },
        PerturbationRow {
            mode: TraceMode::Combined,
            elapsed_logical_units: 1068,
            trace_bytes: 24_576,
            ranking_eligible: false,
        },
    ];

    TraceContractReport {
        execution_class: ExecutionClass::BoundedContractHarness,
        hardware_gate: HardwareGate::unavailable(
            "nvidia-gpu-with-nsight-systems-and-cupti",
            "Bounded contract harness verifies trace correlation, clock mappings, and negative controls; no physical NVIDIA GPU or Nsight/CUPTI capture ran locally.",
        ),
        controls,
        perturbation_matrix,
    }
}

pub fn inputs_json() -> String {
    let pos = base_positive_bundle();
    format!(
        "{{\"schema\":\"inference-lab.trace-inputs.v1\",\"issue\":40,\"workload_id\":{},\"run_id\":{},\"engine_module_digest\":{},\"native_artifact_digest\":{},\"deferred_metadata_window_ns\":{},\"controls_count\":8,\"note\":\"iaprof targets Intel i915/Xe and is explicitly excluded from the NVIDIA CUPTI/Nsight profile\"}}\n",
        quote(&pos.binding.workload_id),
        quote(&pos.binding.run_id),
        pos.binding.engine_module_digest.to_json(),
        pos.binding.native_artifact_digest.to_json(),
        pos.deferred_metadata_window_ns
    )
}

pub fn results_json(report: &TraceContractReport) -> String {
    let controls: Vec<String> = report
        .controls
        .iter()
        .map(|c| {
            let rows: Vec<String> = c
                .joined_rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"device_event_id\":{},\"host_event_id\":{},\"kernel_name\":{},\"context_id\":{},\"stream_id\":{},\"correlation_id\":{},\"cuda_graph_id\":{},\"host_enqueue_start_ktime_ns\":{},\"device_start_ktime_ns\":{},\"launch_to_device_delta_ns\":{},\"uncertainty_ns\":{},\"request_attribution\":{},\"metadata_label\":{}}}",
                        quote(&r.device_event_id),
                        opt_str(r.host_event_id.as_deref()),
                        quote(&r.kernel_name),
                        r.context_id,
                        r.stream_id,
                        r.correlation_id,
                        opt_u64(r.cuda_graph_id),
                        opt_i64(r.host_enqueue_start_ktime_ns),
                        opt_i64(r.device_start_ktime_ns),
                        opt_i64(r.launch_to_device_delta_ns),
                        opt_u64(r.uncertainty_ns),
                        quote(&r.request_attribution),
                        opt_str(r.metadata_label.as_deref())
                    )
                })
                .collect();
            let timeline: Vec<String> = c
                .timeline
                .iter()
                .map(|t| {
                    format!(
                        "{{\"event_kind\":{},\"source_event_id\":{},\"label\":{},\"source_clock_domain\":{},\"units\":{},\"ktime_start_ns\":{},\"ktime_end_ns\":{},\"uncertainty_ns\":{}}}",
                        quote(&t.event_kind),
                        quote(&t.source_event_id),
                        quote(&t.label),
                        quote(t.source_clock_domain.as_str()),
                        quote(&t.units),
                        opt_i64(t.ktime_start_ns),
                        opt_i64(t.ktime_end_ns),
                        opt_u64(t.uncertainty_ns)
                    )
                })
                .collect();
            format!(
                "{{\"control_name\":{},\"causal_path_substantiated\":{},\"status\":{},\"errors\":{},\"warnings\":{},\"unresolved_metadata_ids\":{},\"unsupported_conclusions\":{},\"joined_rows\":[{}],\"timeline\":[{}]}}",
                quote(&c.control_name),
                c.causal_path_substantiated,
                quote(&c.status),
                str_array(&c.errors),
                str_array(&c.warnings),
                str_array(&c.unresolved_metadata_ids),
                str_array(&c.unsupported_conclusions),
                rows.join(","),
                timeline.join(",")
            )
        })
        .collect();
    let pert: Vec<String> = report
        .perturbation_matrix
        .iter()
        .map(|p| {
            format!(
                "{{\"mode\":{},\"elapsed_logical_units\":{},\"trace_bytes\":{},\"ranking_eligible\":{}}}",
                quote(p.mode.as_str()),
                p.elapsed_logical_units,
                p.trace_bytes,
                p.ranking_eligible
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.trace-report.v1\",\"issue\":40,\"package_version\":{},\"execution_class\":{},\"hardware_gate\":{},\"controls\":[{}],\"perturbation_matrix\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(report.execution_class.as_str()),
        report.hardware_gate.to_json(),
        controls.join(",\n"),
        pert.join(",\n")
    )
}

pub fn summary_markdown(report: &TraceContractReport) -> String {
    let mut out = format!(
        "# Issue #40 — Host and GPU trace correlation qualification\n\n\
         - **Execution class**: `{}`\n\
         - **Physical hardware gate**: `{}`\n\n\
         ## Positive and negative control outcomes\n\n\
         | Control | Causal path substantiated | Status | Errors | Unresolved metadata | Unsupported conclusions |\n\
         | --- | --- | --- | --- | --- | --- |\n",
        report.execution_class.as_str(),
        report.hardware_gate.status.as_str()
    );
    for c in &report.controls {
        writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} | {} | {} |",
            c.control_name,
            c.causal_path_substantiated,
            c.status,
            c.errors.len(),
            c.unresolved_metadata_ids.len(),
            c.unsupported_conclusions.len()
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nOnly the calibrated, lossless positive control substantiates a causal path, while its \
         multi-request batched kernel still keeps per-request attribution explicitly `unavailable`. \
         Performance ranking remains restricted to uninstrumented trials.\n",
    );
    out
}
