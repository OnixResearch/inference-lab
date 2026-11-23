# Issue #39: Host CPU Contention & Topology-Aware Placement

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`)
- Hardware gate: `unavailable_hardware_runtime` (Physical Linux / DGX Spark host CPU contention & GPU serving trials)
- Blink workload reproducible from paper: `false` (Blink (arXiv:2604.07609) Fig. 1 interference workload & CPU budget are not fully specified for exact reproduction; gap recorded rather than invented)
- Conditional `sched_ext` follow-on triggered: `false`

## Synthetic Contract Contention Matrix (Blink-Comparable Form)
- `default_placement` @ `idle`: `ratio=1.000`, `p99_ttft=48.0ms`, `p99_tpot=14.0ms`, `bg_work=0` (`starved=false`) — idle baseline
- `default_placement` @ `moderate_cpu_load`: `ratio=0.775`, `p99_ttft=96.0ms`, `p99_tpot=21.0ms`, `bg_work=100` (`starved=false`) — moderate contention degrades tail
- `default_placement` @ `heavy_cpu_load`: `ratio=0.362`, `p99_ttft=310.0ms`, `p99_tpot=49.0ms`, `bg_work=192` (`starved=false`) — heavy contention triggers host launch stall
- `static_topology_affinity` @ `idle`: `ratio=1.000`, `p99_ttft=50.0ms`, `p99_tpot=14.5ms`, `bg_work=0` (`starved=false`) — retained idle regression from reduced core pool
- `static_topology_affinity` @ `moderate_cpu_load`: `ratio=0.941`, `p99_ttft=62.0ms`, `p99_tpot=16.0ms`, `bg_work=96` (`starved=false`) — affinity recovers tail with comparable background work
- `static_topology_affinity` @ `heavy_cpu_load`: `ratio=0.825`, `p99_ttft=88.0ms`, `p99_tpot=19.5ms`, `bg_work=184` (`starved=false`) — affinity limits heavy-load degradation
- `host_light_decode_cuda_graphs` @ `idle`: `ratio=1.000`, `p99_ttft=47.0ms`, `p99_tpot=13.5ms`, `bg_work=0` (`starved=false`) — CUDA-graph host-light idle
- `host_light_decode_cuda_graphs` @ `moderate_cpu_load`: `ratio=0.968`, `p99_ttft=54.0ms`, `p99_tpot=14.2ms`, `bg_work=98` (`starved=false`) — host-light separates per-token host path
- `host_light_decode_cuda_graphs` @ `heavy_cpu_load`: `ratio=0.923`, `p99_ttft=68.0ms`, `p99_tpot=15.6ms`, `bg_work=190` (`starved=false`) — host-light minimizes heavy-load tail impact

## Negative Controls (8 / 8 rejected)
- `nc1_transplanted_blog_cpu_mask_rejected` -> `invalid` (blog-specific CPU masks cannot be transplanted across hosts)
- `nc2_unlabeled_combined_affinity_priority_irq` -> `invalid` (affinity, priority, and IRQ must be separated into distinct labeled arms)
- `nc3_cuda_api_irq_not_causal_gpu_idle` -> `invalid` (CUDA API duration / off-CPU / IRQ overlap alone cannot establish GPU idle or causal stall)
- `nc4_mean_itl_relabeled_as_token_gap_percentile` -> `invalid` (per-call mean ITL must never be relabeled as a within-stream token-gap percentile)
- `nc5_bpf_verifier_load_not_inference_slo` -> `invalid` (successful BPF verifier load for sched_ext is not an inference SLO result)
- `nc6_missing_traced_vs_untraced_overhead_check` -> `invalid` (traced and untraced runs must be compared to bound observer overhead)
- `nc7_background_starvation_recovery_flagged` -> `invalid` (recovery achieved by starving background work (<70% progress) is flagged)
- `nc8_nonfinite_throughput_rejected` -> `invalid` (NaN throughput fails colocated/isolated ratio computation)
