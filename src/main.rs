//! Imperative shell: CLI parsing, deterministic artifact serialization, fresh-directory
//! publication with empty `COMPLETE` marker written last.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use inference_lab::{
    activation_injection_controls, adaptive_spec, adv_spec_reasoning, alora_prefix_cache,
    anchor_label_drafter, apple_nvidia_sidecar, batch2, batch3, batch4, batch5, batch6, batch7,
    batch8, batch9, biollm_protocol, blink_cpu_free_serving, clairic_extraction, compact_hybrids,
    cpu_contention, decode_megakernels, deepseek_ref, deepseek_spark, diffusion,
    disagg_kv_transfer, djev_protocol, drafter_opd_replay, drafting, exl3_longctx_quant,
    expert_placement, flash_topk, ggml_ops, graphical_sampling, harness_lora_compare,
    hetero_rollout, hnet_byte_modeling, hypertune_kv_adapters, inkling_mini_serve,
    inkling_residuals, intervention_replay, k2_uno_mova, kernel_opt, knowledgator_sys1_extract,
    kv_compression, kv_routing, latent_feedback, learned_compiler, lfm3_d1_decision, lifecycle,
    ling_fp8kv_alloc_cap, link_sparse_attention, linquant_recurrent, loop_kv_compression, macaq_r,
    maya_ik2_serve, mimo_hysparse2_kv, mit_qda_quant, mla_comparison, moe_spec_roofline,
    mtp_decoder, native_recurrent_bench, nemotron_spark, nvfp4_mxfp8, nvme_coalescing,
    parasite_feature_control, pdl_breakable_graphs, perseus_mpk_rdma, phoenix_nano, power,
    proof_completion, qsa_state_coherence, quark_eval, qwen35_neuron_steering, qwen38_longbook_gdn,
    qwen38_qsa, qwen38_quant, related_work, rsi_bench, rwkv_ggml, rwkv_leaf, saber_dflash2_power,
    seed, serving_bench, serving_hill_climb, silicon_codesign_audit, simplehc_mhc,
    spark_copydrafter, spark_engines, spark_kv_checkpoint, spark_rdma_tp, sparse_moe,
    split_m2_p150, stage, structured_read, tensorlean_kernel_verify, tiered_eval,
    tkcc_hillclimb_profile, tokenizer, trace, ttf_ternary_qwen, tufazip_online_codec,
    unlimited_ocr_longdoc, varloop_budget_protocol, vision_infusion,
};

const HELP: &str = "\
Inference Lab: bounded inference contract and measurement harnesses

Usage:
  inference-lab <command> --out NEW_DIRECTORY
  inference-lab --help

Bootstrap & Batch 1 commands:
  benchmark           Bootstrap serving/KV-budget contract seed (serving-contract-v1)
  diffusion           Issue #1 bounded diffusion proposal contract, EBM selection & tree interchange
  trace               Issue #40 host & GPU trace correlation qualification + negative controls
  stage               Issue #41 versioned transformer stage contract & 4-layer split reference
  power               Issue #43 power & throttle cadence qualification + protocol controls
  tokenizer           Issue #52 BPE & chat-template conformance + disqualifying parity checker
  related-work        Issues #56, #58, #60 pinned default-off related-work gate receipts

Batch 2 commands (#2-#11):
  proof-completion    Issue #2 masked proof completion through Compiler Lab's frozen gate
  drafting            Issue #3 diffusion drafting conformance & exact rejection-sampling parity
  kernel-opt          Issue #4 reference-assisted vs spec-only kernel optimization gate ordering
  kv-routing          Issue #5 session KV routing, bounded soft affinity & RDMA compatibility
  adaptive-spec       Issue #6 adaptive speculative graph & concurrency characterization
  serving-bench       Issue #7 portable serving benchmark extraction, branch-trace & RLM profiles
  deepseek-spark      Issue #8 DeepSeek/Spark migration, repaired counters & semantic quality checks
  learned-compiler    Issue #9 learned-compiler recovery inventory (blocked_on_artifacts) & controls
  structured-read     Issue #10 conditional shared structured-read gate (blocked_on_artifacts)
  lifecycle           Issue #11 centralized model-lifecycle protocol & Spark adapter controls
  batch2-suite        Run all Batch 2 harnesses (#2-#11) with SHA-256 manifest.json & COMPLETE last

Batch 3 commands (#12-#21):
  rsi-bench           Issue #12 standalone RSI inference benchmark consolidation & RSI #82 gate
  djev-protocol       Issue #13 reusable djev protocol client split & closed RSI #42 cache identity
  rwkv-leaf           Issue #14 conditional native RWKV numerical leaf gate (blocked_on_artifacts)
  ggml-ops            Issue #15 pinned ggml test-backend-ops 4-axis capability & quant reference
  rwkv-ggml           Issue #16 ggml RWKV-7 recurrence & state-continuation parity vs native RSI
  graphical-sampling  Issue #17 discrete graphical sampling exact oracle, color-phase Gibbs & ESS
  qwen38-quant        Issue #18 Qwen3.8-27B quant/serving profile, MACaQ, QSA, Monoglot & QDA KL
  deepseek-ref        Issue #19 corrected pinned DeepSeek-style reference & 5 defect regressions
  mla-comparison      Issue #20 explicit vs absorbed MLA parity, sentinel poisoning & memory bytes
  sparse-moe          Issue #21 sparse MoE dispatch vs dense/tiny reference & analytical gradients
  batch3-suite        Run all Batch 3 harnesses (#12-#21) with SHA-256 manifest.json & COMPLETE last

Batch 4 commands (#22-#31):
  mtp-decoder         Issue #22 small MTP draft-and-verify decoder, poisoned caches & stochastic parity
  flash-topk          Issue #23 FlashTopK sparse-indexer reference, SM90/100 gate & recovery checks
  macaq-r             Issue #24 MACaQ-R native MoE K-quant refinement, artifact inspection & rescoring
  hetero-rollout      Issue #25 heterogeneous rollout serving, versioned HTTP LoRA & fleet reconciliation
  nvfp4-mxfp8         Issue #26 NVFP4 fwd / MXFP8 bwd numerics, TE SM120/121 gate & Fabric #6 codec
  spark-engines       Issue #27 Hyperflare & TensorRT Edge-LLM Spark matrix & KV+recurrent continuation
  kv-compression      Issue #28 KVPress long-context compression, RULER diagnostics & grounded end task
  expert-placement    Issue #29 RustingStrata vs llama.cpp expert placement, warm states & paired reducer
  vision-infusion     Issue #30 matched-capacity vision encoders vs early in-fusion & packed parity
  qwen38-qsa          Issue #31 Qwen3.8-27B 4-token block QSA vs dense 2x2 matrix & long-context tasks
  batch4-suite        Run all Batch 4 harnesses (#22-#31) with SHA-256 manifest.json & COMPLETE last

Batch 5 commands (#32-#39, #42, #44-#46):
  qsa-state-coherence     Issue #32 QSA cache reuse & target/indexer/GDN/MTP state coherence
  inkling-residuals       Issue #33 Inkling 6-layer block attention residuals & dynamic/static sinks
  phoenix-nano            Issue #34 Phoenix nano gpu_arch(), SID codec, resume & gRPC retrieve->rank
  nemotron-spark          Issue #35 Nemotron 3.5 Lightning & Bolt DGX Spark serving profiles
  compact-hybrids         Issue #36 Compact hybrid recurrent profile (Granite/Jamba2/Falcon H1)
  native-recurrent-bench  Issue #37 Native recurrent-state bench (RWKV-7, Mamba-3 SISO, xLSTM)
  tiered-eval             Issue #38 Tiered persistent-engine evaluation with cold final replay
  cpu-contention          Issue #39 Host CPU contention, topology placement & Blink-comparable ratios
  split-m2-p150           Issue #42 M2 Ultra + P150/P150x2 dense & disjoint-MoE split qualification
  quark-eval              Issue #44 Quark Base/Zero local-window/global-KV parity & context costs
  intervention-replay     Issue #45 Reproducible activation intervention & deterministic replay harness
  biollm-protocol         Issue #46 Equal-budget 5-arm BioLLM adapter protocol & matched controls
  batch5-suite            Run all Batch 5 harnesses (#32-#39, #42, #44-#46) with manifest & COMPLETE last

Batch 6 commands (#47-#51, #53-#55, #57, #59, #61):
  adv-spec-reasoning      Issue #47 Adversarial speculative reasoning drafts vs unbiased drafts & slot noise
  nvme-coalescing         Issue #48 NVMe interrupt coalescing controlled host variable for SSD-paged tables
  harness-lora-compare    Issue #49 Harness-specific LoRA vs base/prompt/harness & overlap contamination audit
  latent-feedback         Issue #50 Data-limited latent feedback vs matched-FLOP extra-epoch control
  simplehc-mhc            Issue #51 SimpleHC vs DeepSeek mHC (Sinkhorn-Knopp) overhead & loss beside #33
  serving-hill-climb      Issue #53 Default-off ceiling-first, interleaved-control serving hill-climb
  tufazip-online-codec    Issue #54 Default-off tufazip online range codec & 2xB200 CUDA 13.0 hardware gate
  clairic-extraction      Issue #55 Default-off pinned Clairic/Clairic-flash typed extraction vs baseline
  inkling-mini-serve      Issue #57 Default-off pinned Inkling 2 Mini local serve & 48-layer looped MoE parity
  maya-ik2-serve          Issue #59 Default-off pinned maya-ik2 24 GB sm_86 serve & packed 4-bit unpack parity
  decode-megakernels      Issue #61 Persistent decode megakernels vs multi-launch reference & batch scaling
  batch6-suite            Run all Batch 6 harnesses (#47-#51, #53-#55, #57, #59, #61) with manifest & COMPLETE last

Batch 7 commands (#62-#73):
  disagg-kv-transfer      Issue #62 Disaggregated Spark-CUDA prefill to Apple-Metal decode 10GbE KV transfer
  spark-rdma-tp           Issue #63 2-node DGX Spark raw-RDMA TP & 4-bit attention vs replicated/layer-split (1M)
  spark-kv-checkpoint     Issue #64 2-Spark KV sharding & disk-persisted KV checkpoint resume (GLM-5.3-Flash)
  spark-copydrafter       Issue #65 Pinned spark-ds41f & isolated CopyDrafter lossless speculative drafting
  hypertune-kv-adapters   Issue #66 Pinned Hypertune soft-prefill paged KV adapters vs heterogeneous batched LoRA
  qwen38-longbook-gdn     Issue #67 Pinned Qwen3.8-27B-Longbook27-v2 (all-64-layer GDN) vs base Qwen3.8-27B (262K)
  linquant-recurrent      Issue #68 LinQuant algebraic recurrent prefill rewrite error & throughput on GDN/Mamba-2
  varloop-budget-protocol Issue #69 Pinned TML-d24-varloop-research & 2DSD equal-cost loop vs thinking protocol
  ttf-ternary-qwen        Issue #70 Pinned ttf-home 1.6-bpw ternary Qwen3.8-27B vs 4-bit GGUF & FP16 controls
  drafter-opd-replay      Issue #71 On-policy Draft-OPD & D-PACE rejection weighting on noise-floor-gated replay
  k2-uno-mova             Issue #72 K2 Horizon Uno draftless lossless adapter & MoVA/softplus gating vs AR/MTP
  lfm3-d1-decision        Issue #73 Pinned LFM3-1.7B tri-hybrid (256k) & d1-2B single-pass typed decision model
  batch7-suite            Run all Batch 7 harnesses (#62-#73) with SHA-256 manifest & COMPLETE last

Batch 8 commands (#74-#84):
  pdl-breakable-graphs    Issue #74 PDL kernel overlap & breakable prefill CUDA graphs on flash-MoE fixture
  tensorlean-kernel-verify Issue #75 TensorLean formal-verification boundary & seeded kernel-bug coverage
  blink-cpu-free-serving  Issue #76 Blink CPU-free RDMA serving & host-CPU contention resilience vs vLLM/SGLang
  qwen35-neuron-steering  Issue #77 Qwen3.5-4B exact two-neuron down_proj steering regression fixture (#45)
  hnet-byte-modeling      Issue #78 H-Net v2 byte-level dynamic chunking parity & tokenizer-free edge cases
  tkcc-hillclimb-profile  Issue #79 Together Compiler (TKCC) related work & local hill-climb cycle profiler
  exl3-longctx-quant      Issue #80 Pinned EXL3 2.25-bpw trellis quantizer & 128K-1M reasoning/coding checks
  saber-dflash2-power     Issue #81 Pinned saber-engine (Rust) + DFlash2 parity & power-capped tok/s/W (#43)
  mimo-hysparse2-kv       Issue #82 MiMo HySparse2 causal encoder-decoder KV projection parity & 1M ablation
  apple-nvidia-sidecar    Issue #83 Pinned apple-nvidia-sidecar Apple-host + NVIDIA eGPU split parity (#42)
  silicon-codesign-audit  Issue #84 arXiv:2611.18859 ML/silicon co-design pre-silicon roofline/energy audit
  batch8-suite            Run all Batch 8 harnesses (#74-#84) with SHA-256 manifest & COMPLETE last

Batch 9 commands (#85-#95):
  link-sparse-attention   Issue #85 Constructor AGI Link learned sparse attention & 4M retrieval/end-task boundary
  mit-qda-quant           Issue #86 MIT QDA mixed-precision recipe vs pinned manual/GGUF/NVFP4 baselines
  knowledgator-sys1-extract Issue #87 Knowledgator System 1 extraction alongside Clairic (#55) & LLM baseline
  unlimited-ocr-longdoc   Issue #88 Unlimited-OCR (3B) multi-page PDF edit distance & cross-page preservation
  loop-kv-compression     Issue #89 ResidualQuant INT2 & LoopVQ sub-1-bit cross-loop KV compression (#69)
  alora-prefix-cache      Issue #90 Activated LoRA (aLoRA) delayed-suffix vs Hypertune (#66) prefix caching
  perseus-mpk-rdma        Issue #91 Perseus decoupled RDMA signaling & MPK persistent kernel beside #61/#63/#76
  moe-spec-roofline       Issue #92 Sparse-MoE speculative verification roofline & temporal routing overlap
  anchor-label-drafter    Issue #93 Off-policy Anchor-Label Relabelling vs Draft-OPD & D-PACE (#71)
  parasite-feature-control Issue #94 Parasite Features subspace interchange, necessity ablation & planted control (#45)
  ling-fp8kv-alloc-cap    Issue #95 Single-Spark Ling-3.2-air NVFP4 2x2 FP8-KV vs torch allocator cap (262K)
  batch9-suite            Run all Batch 9 harnesses (#85-#95) with SHA-256 manifest & COMPLETE last

Issue #96 commands:
  activation-injection-controls Issue #96 Default-off concept-injection anomaly sweep, SAE refusal/truthfulness controls & non-circularity guard (#45, #77, #94)

Rules:
  The output directory must not exist; its parent directory must exist.
  Writes inputs.json, results.json, summary.md, and finally an empty COMPLETE marker.
  All harnesses run as deterministic bounded_contract_harness or blocked_on_artifacts checks;
  no physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, FPGA, or live serve runs locally.
";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Help,
    Benchmark(PathBuf),
    Diffusion(PathBuf),
    Trace(PathBuf),
    Stage(PathBuf),
    Power(PathBuf),
    Tokenizer(PathBuf),
    RelatedWork(PathBuf),
    ProofCompletion(PathBuf),
    Drafting(PathBuf),
    KernelOpt(PathBuf),
    KvRouting(PathBuf),
    AdaptiveSpec(PathBuf),
    ServingBench(PathBuf),
    DeepSeekSpark(PathBuf),
    LearnedCompiler(PathBuf),
    StructuredRead(PathBuf),
    Lifecycle(PathBuf),
    Batch2Suite(PathBuf),
    RsiBench(PathBuf),
    DjevProtocol(PathBuf),
    RwkvLeaf(PathBuf),
    GgmlOps(PathBuf),
    RwkvGgml(PathBuf),
    GraphicalSampling(PathBuf),
    Qwen38Quant(PathBuf),
    DeepSeekRef(PathBuf),
    MlaComparison(PathBuf),
    SparseMoe(PathBuf),
    Batch3Suite(PathBuf),
    MtpDecoder(PathBuf),
    FlashTopK(PathBuf),
    MacaqR(PathBuf),
    HeteroRollout(PathBuf),
    Nvfp4Mxfp8(PathBuf),
    SparkEngines(PathBuf),
    KvCompression(PathBuf),
    ExpertPlacement(PathBuf),
    VisionInfusion(PathBuf),
    Qwen38Qsa(PathBuf),
    Batch4Suite(PathBuf),
    QsaStateCoherence(PathBuf),
    InklingResiduals(PathBuf),
    PhoenixNano(PathBuf),
    NemotronSpark(PathBuf),
    CompactHybrids(PathBuf),
    NativeRecurrentBench(PathBuf),
    TieredEval(PathBuf),
    CpuContention(PathBuf),
    SplitM2P150(PathBuf),
    QuarkEval(PathBuf),
    InterventionReplay(PathBuf),
    BioLlmProtocol(PathBuf),
    Batch5Suite(PathBuf),
    AdvSpecReasoning(PathBuf),
    NvmeCoalescing(PathBuf),
    HarnessLoraCompare(PathBuf),
    LatentFeedback(PathBuf),
    SimpleHcMhc(PathBuf),
    ServingHillClimb(PathBuf),
    TufazipOnlineCodec(PathBuf),
    ClairicExtraction(PathBuf),
    InklingMiniServe(PathBuf),
    MayaIk2Serve(PathBuf),
    DecodeMegakernels(PathBuf),
    Batch6Suite(PathBuf),
    DisaggKvTransfer(PathBuf),
    SparkRdmaTp(PathBuf),
    SparkKvCheckpoint(PathBuf),
    SparkCopyDrafter(PathBuf),
    HypertuneKvAdapters(PathBuf),
    Qwen38LongbookGdn(PathBuf),
    LinQuantRecurrent(PathBuf),
    VarloopBudgetProtocol(PathBuf),
    TtfTernaryQwen(PathBuf),
    DrafterOpdReplay(PathBuf),
    K2UnoMova(PathBuf),
    Lfm3D1Decision(PathBuf),
    Batch7Suite(PathBuf),
    PdlBreakableGraphs(PathBuf),
    TensorLeanKernelVerify(PathBuf),
    BlinkCpuFreeServing(PathBuf),
    Qwen35NeuronSteering(PathBuf),
    HNetByteModeling(PathBuf),
    TkccHillclimbProfile(PathBuf),
    Exl3LongCtxQuant(PathBuf),
    SaberDFlash2Power(PathBuf),
    MimoHySparse2Kv(PathBuf),
    AppleNvidiaSidecar(PathBuf),
    SiliconCoDesignAudit(PathBuf),
    Batch8Suite(PathBuf),
    LinkSparseAttention(PathBuf),
    MitQdaQuant(PathBuf),
    KnowledgatorSys1Extract(PathBuf),
    UnlimitedOcrLongdoc(PathBuf),
    LoopKvCompression(PathBuf),
    AloraPrefixCache(PathBuf),
    PerseusMpkRdma(PathBuf),
    MoeSpecRoofline(PathBuf),
    AnchorLabelDrafter(PathBuf),
    ParasiteFeatureControl(PathBuf),
    LingFp8KvAllocCap(PathBuf),
    Batch9Suite(PathBuf),
    ActivationInjectionControls(PathBuf),
}

fn parse(args: &[String]) -> Result<Command, String> {
    match args {
        [arg] if arg == "--help" || arg == "-h" => Ok(Command::Help),
        [command, flag, path] if flag == "--out" && !path.is_empty() && !path.starts_with('-') => {
            let out = PathBuf::from(path);
            match command.as_str() {
                "benchmark" => Ok(Command::Benchmark(out)),
                "diffusion" => Ok(Command::Diffusion(out)),
                "trace" => Ok(Command::Trace(out)),
                "stage" => Ok(Command::Stage(out)),
                "power" => Ok(Command::Power(out)),
                "tokenizer" => Ok(Command::Tokenizer(out)),
                "related-work" => Ok(Command::RelatedWork(out)),
                "proof-completion" => Ok(Command::ProofCompletion(out)),
                "drafting" => Ok(Command::Drafting(out)),
                "kernel-opt" => Ok(Command::KernelOpt(out)),
                "kv-routing" => Ok(Command::KvRouting(out)),
                "adaptive-spec" => Ok(Command::AdaptiveSpec(out)),
                "serving-bench" => Ok(Command::ServingBench(out)),
                "deepseek-spark" => Ok(Command::DeepSeekSpark(out)),
                "learned-compiler" => Ok(Command::LearnedCompiler(out)),
                "structured-read" => Ok(Command::StructuredRead(out)),
                "lifecycle" => Ok(Command::Lifecycle(out)),
                "batch2-suite" => Ok(Command::Batch2Suite(out)),
                "rsi-bench" => Ok(Command::RsiBench(out)),
                "djev-protocol" => Ok(Command::DjevProtocol(out)),
                "rwkv-leaf" => Ok(Command::RwkvLeaf(out)),
                "ggml-ops" => Ok(Command::GgmlOps(out)),
                "rwkv-ggml" => Ok(Command::RwkvGgml(out)),
                "graphical-sampling" => Ok(Command::GraphicalSampling(out)),
                "qwen38-quant" => Ok(Command::Qwen38Quant(out)),
                "deepseek-ref" => Ok(Command::DeepSeekRef(out)),
                "mla-comparison" => Ok(Command::MlaComparison(out)),
                "sparse-moe" => Ok(Command::SparseMoe(out)),
                "batch3-suite" => Ok(Command::Batch3Suite(out)),
                "mtp-decoder" => Ok(Command::MtpDecoder(out)),
                "flash-topk" => Ok(Command::FlashTopK(out)),
                "macaq-r" => Ok(Command::MacaqR(out)),
                "hetero-rollout" => Ok(Command::HeteroRollout(out)),
                "nvfp4-mxfp8" => Ok(Command::Nvfp4Mxfp8(out)),
                "spark-engines" => Ok(Command::SparkEngines(out)),
                "kv-compression" => Ok(Command::KvCompression(out)),
                "expert-placement" => Ok(Command::ExpertPlacement(out)),
                "vision-infusion" => Ok(Command::VisionInfusion(out)),
                "qwen38-qsa" => Ok(Command::Qwen38Qsa(out)),
                "batch4-suite" => Ok(Command::Batch4Suite(out)),
                "qsa-state-coherence" => Ok(Command::QsaStateCoherence(out)),
                "inkling-residuals" => Ok(Command::InklingResiduals(out)),
                "phoenix-nano" => Ok(Command::PhoenixNano(out)),
                "nemotron-spark" => Ok(Command::NemotronSpark(out)),
                "compact-hybrids" => Ok(Command::CompactHybrids(out)),
                "native-recurrent-bench" => Ok(Command::NativeRecurrentBench(out)),
                "tiered-eval" => Ok(Command::TieredEval(out)),
                "cpu-contention" => Ok(Command::CpuContention(out)),
                "split-m2-p150" => Ok(Command::SplitM2P150(out)),
                "quark-eval" => Ok(Command::QuarkEval(out)),
                "intervention-replay" => Ok(Command::InterventionReplay(out)),
                "biollm-protocol" => Ok(Command::BioLlmProtocol(out)),
                "batch5-suite" => Ok(Command::Batch5Suite(out)),
                "adv-spec-reasoning" => Ok(Command::AdvSpecReasoning(out)),
                "nvme-coalescing" => Ok(Command::NvmeCoalescing(out)),
                "harness-lora-compare" => Ok(Command::HarnessLoraCompare(out)),
                "latent-feedback" => Ok(Command::LatentFeedback(out)),
                "simplehc-mhc" => Ok(Command::SimpleHcMhc(out)),
                "serving-hill-climb" => Ok(Command::ServingHillClimb(out)),
                "tufazip-online-codec" => Ok(Command::TufazipOnlineCodec(out)),
                "clairic-extraction" => Ok(Command::ClairicExtraction(out)),
                "inkling-mini-serve" => Ok(Command::InklingMiniServe(out)),
                "maya-ik2-serve" => Ok(Command::MayaIk2Serve(out)),
                "decode-megakernels" => Ok(Command::DecodeMegakernels(out)),
                "batch6-suite" => Ok(Command::Batch6Suite(out)),
                "disagg-kv-transfer" => Ok(Command::DisaggKvTransfer(out)),
                "spark-rdma-tp" => Ok(Command::SparkRdmaTp(out)),
                "spark-kv-checkpoint" => Ok(Command::SparkKvCheckpoint(out)),
                "spark-copydrafter" => Ok(Command::SparkCopyDrafter(out)),
                "hypertune-kv-adapters" => Ok(Command::HypertuneKvAdapters(out)),
                "qwen38-longbook-gdn" => Ok(Command::Qwen38LongbookGdn(out)),
                "linquant-recurrent" => Ok(Command::LinQuantRecurrent(out)),
                "varloop-budget-protocol" => Ok(Command::VarloopBudgetProtocol(out)),
                "ttf-ternary-qwen" => Ok(Command::TtfTernaryQwen(out)),
                "drafter-opd-replay" => Ok(Command::DrafterOpdReplay(out)),
                "k2-uno-mova" => Ok(Command::K2UnoMova(out)),
                "lfm3-d1-decision" => Ok(Command::Lfm3D1Decision(out)),
                "batch7-suite" => Ok(Command::Batch7Suite(out)),
                "pdl-breakable-graphs" => Ok(Command::PdlBreakableGraphs(out)),
                "tensorlean-kernel-verify" => Ok(Command::TensorLeanKernelVerify(out)),
                "blink-cpu-free-serving" => Ok(Command::BlinkCpuFreeServing(out)),
                "qwen35-neuron-steering" => Ok(Command::Qwen35NeuronSteering(out)),
                "hnet-byte-modeling" => Ok(Command::HNetByteModeling(out)),
                "tkcc-hillclimb-profile" => Ok(Command::TkccHillclimbProfile(out)),
                "exl3-longctx-quant" => Ok(Command::Exl3LongCtxQuant(out)),
                "saber-dflash2-power" => Ok(Command::SaberDFlash2Power(out)),
                "mimo-hysparse2-kv" => Ok(Command::MimoHySparse2Kv(out)),
                "apple-nvidia-sidecar" => Ok(Command::AppleNvidiaSidecar(out)),
                "silicon-codesign-audit" => Ok(Command::SiliconCoDesignAudit(out)),
                "batch8-suite" => Ok(Command::Batch8Suite(out)),
                "link-sparse-attention" => Ok(Command::LinkSparseAttention(out)),
                "mit-qda-quant" => Ok(Command::MitQdaQuant(out)),
                "knowledgator-sys1-extract" => Ok(Command::KnowledgatorSys1Extract(out)),
                "unlimited-ocr-longdoc" => Ok(Command::UnlimitedOcrLongdoc(out)),
                "loop-kv-compression" => Ok(Command::LoopKvCompression(out)),
                "alora-prefix-cache" => Ok(Command::AloraPrefixCache(out)),
                "perseus-mpk-rdma" => Ok(Command::PerseusMpkRdma(out)),
                "moe-spec-roofline" => Ok(Command::MoeSpecRoofline(out)),
                "anchor-label-drafter" => Ok(Command::AnchorLabelDrafter(out)),
                "parasite-feature-control" => Ok(Command::ParasiteFeatureControl(out)),
                "ling-fp8kv-alloc-cap" => Ok(Command::LingFp8KvAllocCap(out)),
                "batch9-suite" => Ok(Command::Batch9Suite(out)),
                "activation-injection-controls" => Ok(Command::ActivationInjectionControls(out)),
                other => Err(format!("unknown command `{other}`")),
            }
        }
        _ => Err("expected `<command> --out NEW_DIRECTORY` or `--help`".into()),
    }
}

fn write_new_file(path: &Path, content: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
    file.write_all(content)
        .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
    Ok(())
}

fn publish(path: &Path, inputs: &str, results: &str, summary: &str) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for (name, content) in [
        ("inputs.json", inputs),
        ("results.json", results),
        ("summary.md", summary),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_with_manifest(
    path: &Path,
    inputs: &str,
    results: &str,
    summary: &str,
    manifest: &str,
) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for (name, content) in [
        ("inputs.json", inputs),
        ("results.json", results),
        ("summary.md", summary),
        ("manifest.json", manifest),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch2_suite(path: &Path, bundle: &batch2::Batch2Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch3_suite(path: &Path, bundle: &batch3::Batch3Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch4_suite(path: &Path, bundle: &batch4::Batch4Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch5_suite(path: &Path, bundle: &batch5::Batch5Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch6_suite(path: &Path, bundle: &batch6::Batch6Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch7_suite(path: &Path, bundle: &batch7::Batch7Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch8_suite(path: &Path, bundle: &batch8::Batch8Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch9_suite(path: &Path, bundle: &batch9::Batch9Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn run(command: Command) -> Result<String, String> {
    match command {
        Command::Help => Ok(HELP.into()),
        Command::Benchmark(path) => {
            let scenario = seed::default_scenario();
            let report = seed::evaluate(&scenario)?;
            let inputs = seed::inputs_json(&scenario);
            let results = seed::results_json(&report);
            let summary = seed::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Diffusion(path) => {
            let report = diffusion::run_default_harness()?;
            let inputs = diffusion::inputs_json(&report);
            let results = diffusion::results_json(&report);
            let summary = diffusion::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Trace(path) => {
            let report = trace::run_default_harness();
            let inputs = trace::inputs_json();
            let results = trace::results_json(&report);
            let summary = trace::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Stage(path) => {
            let report = stage::run_default_harness()?;
            let inputs = stage::inputs_json(&report);
            let results = stage::results_json(&report);
            let summary = stage::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Power(path) => {
            let report = power::run_default_harness();
            let inputs = power::inputs_json(&report);
            let results = power::results_json(&report);
            let summary = power::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Tokenizer(path) => {
            let report = tokenizer::run_default_harness();
            let inputs = tokenizer::inputs_json(&report);
            let results = tokenizer::results_json(&report);
            let summary = tokenizer::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::RelatedWork(path) => {
            let report = related_work::run_default_harness();
            let inputs = related_work::inputs_json(&report);
            let results = related_work::results_json(&report);
            let summary = related_work::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ProofCompletion(path) => {
            let report = proof_completion::run_default_harness();
            let inputs = proof_completion::inputs_json(&report);
            let results = proof_completion::results_json(&report);
            let summary = proof_completion::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Drafting(path) => {
            let report = drafting::run_default_harness()?;
            let inputs = drafting::inputs_json(&report);
            let results = drafting::results_json(&report);
            let summary = drafting::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KernelOpt(path) => {
            let report = kernel_opt::run_default_harness()?;
            let inputs = kernel_opt::inputs_json(&report);
            let results = kernel_opt::results_json(&report);
            let summary = kernel_opt::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KvRouting(path) => {
            let report = kv_routing::run_default_harness();
            let inputs = kv_routing::inputs_json(&report);
            let results = kv_routing::results_json(&report);
            let summary = kv_routing::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::AdaptiveSpec(path) => {
            let report = adaptive_spec::run_default_harness()?;
            let inputs = adaptive_spec::inputs_json(&report);
            let results = adaptive_spec::results_json(&report);
            let summary = adaptive_spec::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ServingBench(path) => {
            let report = serving_bench::run_default_harness();
            let inputs = serving_bench::inputs_json(&report);
            let results = serving_bench::results_json(&report);
            let summary = serving_bench::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DeepSeekSpark(path) => {
            let report = deepseek_spark::run_default_harness();
            let inputs = deepseek_spark::inputs_json(&report);
            let results = deepseek_spark::results_json(&report);
            let summary = deepseek_spark::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LearnedCompiler(path) => {
            let report = learned_compiler::run_default_harness();
            let inputs = learned_compiler::inputs_json(&report);
            let results = learned_compiler::results_json(&report);
            let summary = learned_compiler::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::StructuredRead(path) => {
            let report = structured_read::run_default_harness();
            let inputs = structured_read::inputs_json(&report);
            let results = structured_read::results_json(&report);
            let summary = structured_read::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Lifecycle(path) => {
            let report = lifecycle::run_default_harness();
            let inputs = lifecycle::inputs_json(&report);
            let results = lifecycle::results_json(&report);
            let summary = lifecycle::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch2Suite(path) => {
            let bundle = batch2::run_all()?;
            publish_batch2_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::RsiBench(path) => {
            let report = rsi_bench::run_default_harness()?;
            let inputs = rsi_bench::inputs_json(&report);
            let results = rsi_bench::results_json(&report);
            let summary = rsi_bench::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DjevProtocol(path) => {
            let report = djev_protocol::run_default_harness()?;
            let inputs = djev_protocol::inputs_json(&report);
            let results = djev_protocol::results_json(&report);
            let summary = djev_protocol::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::RwkvLeaf(path) => {
            let report = rwkv_leaf::run_default_harness()?;
            let inputs = rwkv_leaf::inputs_json(&report);
            let results = rwkv_leaf::results_json(&report);
            let summary = rwkv_leaf::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::GgmlOps(path) => {
            let report = ggml_ops::run_default_harness()?;
            let inputs = ggml_ops::inputs_json(&report);
            let results = ggml_ops::results_json(&report);
            let summary = ggml_ops::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::RwkvGgml(path) => {
            let report = rwkv_ggml::run_default_harness()?;
            let inputs = rwkv_ggml::inputs_json(&report);
            let results = rwkv_ggml::results_json(&report);
            let summary = rwkv_ggml::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::GraphicalSampling(path) => {
            let report = graphical_sampling::run_default_harness()?;
            let inputs = graphical_sampling::inputs_json(&report);
            let results = graphical_sampling::results_json(&report);
            let summary = graphical_sampling::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Qwen38Quant(path) => {
            let report = qwen38_quant::run_default_harness()?;
            let inputs = qwen38_quant::inputs_json(&report);
            let results = qwen38_quant::results_json(&report);
            let summary = qwen38_quant::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DeepSeekRef(path) => {
            let report = deepseek_ref::run_default_harness()?;
            let inputs = deepseek_ref::inputs_json(&report);
            let results = deepseek_ref::results_json(&report);
            let summary = deepseek_ref::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MlaComparison(path) => {
            let report = mla_comparison::run_default_harness()?;
            let inputs = mla_comparison::inputs_json(&report);
            let results = mla_comparison::results_json(&report);
            let summary = mla_comparison::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SparseMoe(path) => {
            let report = sparse_moe::run_default_harness()?;
            let inputs = sparse_moe::inputs_json(&report);
            let results = sparse_moe::results_json(&report);
            let summary = sparse_moe::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch3Suite(path) => {
            let bundle = batch3::run_all()?;
            publish_batch3_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::MtpDecoder(path) => {
            let report = mtp_decoder::run_default_harness()?;
            let inputs = mtp_decoder::inputs_json(&report);
            let results = mtp_decoder::results_json(&report);
            let summary = mtp_decoder::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::FlashTopK(path) => {
            let report = flash_topk::run_default_harness()?;
            let inputs = flash_topk::inputs_json(&report);
            let results = flash_topk::results_json(&report);
            let summary = flash_topk::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MacaqR(path) => {
            let report = macaq_r::run_default_harness()?;
            let inputs = macaq_r::inputs_json(&report);
            let results = macaq_r::results_json(&report);
            let summary = macaq_r::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::HeteroRollout(path) => {
            let report = hetero_rollout::run_default_harness()?;
            let inputs = hetero_rollout::inputs_json(&report);
            let results = hetero_rollout::results_json(&report);
            let summary = hetero_rollout::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Nvfp4Mxfp8(path) => {
            let report = nvfp4_mxfp8::run_default_harness()?;
            let inputs = nvfp4_mxfp8::inputs_json(&report);
            let results = nvfp4_mxfp8::results_json(&report);
            let summary = nvfp4_mxfp8::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SparkEngines(path) => {
            let report = spark_engines::run_default_harness()?;
            let inputs = spark_engines::inputs_json(&report);
            let results = spark_engines::results_json(&report);
            let summary = spark_engines::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KvCompression(path) => {
            let report = kv_compression::run_default_harness()?;
            let inputs = kv_compression::inputs_json(&report);
            let results = kv_compression::results_json(&report);
            let summary = kv_compression::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ExpertPlacement(path) => {
            let report = expert_placement::run_default_harness()?;
            let inputs = expert_placement::inputs_json(&report);
            let results = expert_placement::results_json(&report);
            let summary = expert_placement::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::VisionInfusion(path) => {
            let report = vision_infusion::run_default_harness()?;
            let inputs = vision_infusion::inputs_json(&report);
            let results = vision_infusion::results_json(&report);
            let summary = vision_infusion::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Qwen38Qsa(path) => {
            let report = qwen38_qsa::run_default_harness()?;
            let inputs = qwen38_qsa::inputs_json(&report);
            let results = qwen38_qsa::results_json(&report);
            let summary = qwen38_qsa::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch4Suite(path) => {
            let bundle = batch4::run_all()?;
            publish_batch4_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::QsaStateCoherence(path) => {
            let report = qsa_state_coherence::run_default_harness()?;
            let inputs = qsa_state_coherence::inputs_json(&report);
            let results = qsa_state_coherence::results_json(&report);
            let summary = qsa_state_coherence::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::InklingResiduals(path) => {
            let report = inkling_residuals::run_default_harness()?;
            let inputs = inkling_residuals::inputs_json(&report);
            let results = inkling_residuals::results_json(&report);
            let summary = inkling_residuals::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::PhoenixNano(path) => {
            let report = phoenix_nano::run_default_harness()?;
            let inputs = phoenix_nano::inputs_json(&report);
            let results = phoenix_nano::results_json(&report);
            let summary = phoenix_nano::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::NemotronSpark(path) => {
            let report = nemotron_spark::run_default_harness()?;
            let inputs = nemotron_spark::inputs_json(&report);
            let results = nemotron_spark::results_json(&report);
            let summary = nemotron_spark::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::CompactHybrids(path) => {
            let report = compact_hybrids::run_default_harness()?;
            let inputs = compact_hybrids::inputs_json(&report);
            let results = compact_hybrids::results_json(&report);
            let summary = compact_hybrids::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::NativeRecurrentBench(path) => {
            let report = native_recurrent_bench::run_default_harness()?;
            let inputs = native_recurrent_bench::inputs_json(&report);
            let results = native_recurrent_bench::results_json(&report);
            let summary = native_recurrent_bench::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::TieredEval(path) => {
            let report = tiered_eval::run_default_harness()?;
            let inputs = tiered_eval::inputs_json(&report);
            let results = tiered_eval::results_json(&report);
            let summary = tiered_eval::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::CpuContention(path) => {
            let report = cpu_contention::run_default_harness()?;
            let inputs = cpu_contention::inputs_json(&report);
            let results = cpu_contention::results_json(&report);
            let summary = cpu_contention::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SplitM2P150(path) => {
            let report = split_m2_p150::run_default_harness()?;
            let inputs = split_m2_p150::inputs_json(&report);
            let results = split_m2_p150::results_json(&report);
            let summary = split_m2_p150::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::QuarkEval(path) => {
            let report = quark_eval::run_default_harness()?;
            let inputs = quark_eval::inputs_json(&report);
            let results = quark_eval::results_json(&report);
            let summary = quark_eval::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::InterventionReplay(path) => {
            let report = intervention_replay::run_default_harness()?;
            let inputs = intervention_replay::inputs_json(&report);
            let results = intervention_replay::results_json(&report);
            let summary = intervention_replay::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::BioLlmProtocol(path) => {
            let report = biollm_protocol::run_default_harness()?;
            let inputs = biollm_protocol::inputs_json(&report);
            let results = biollm_protocol::results_json(&report);
            let summary = biollm_protocol::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch5Suite(path) => {
            let bundle = batch5::run_all()?;
            publish_batch5_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::AdvSpecReasoning(path) => {
            let report = adv_spec_reasoning::run_default_harness()?;
            let inputs = adv_spec_reasoning::inputs_json(&report);
            let results = adv_spec_reasoning::results_json(&report);
            let summary = adv_spec_reasoning::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::NvmeCoalescing(path) => {
            let report = nvme_coalescing::run_default_harness()?;
            let inputs = nvme_coalescing::inputs_json(&report);
            let results = nvme_coalescing::results_json(&report);
            let summary = nvme_coalescing::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::HarnessLoraCompare(path) => {
            let report = harness_lora_compare::run_default_harness()?;
            let inputs = harness_lora_compare::inputs_json(&report);
            let results = harness_lora_compare::results_json(&report);
            let summary = harness_lora_compare::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LatentFeedback(path) => {
            let report = latent_feedback::run_default_harness()?;
            let inputs = latent_feedback::inputs_json(&report);
            let results = latent_feedback::results_json(&report);
            let summary = latent_feedback::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SimpleHcMhc(path) => {
            let report = simplehc_mhc::run_default_harness()?;
            let inputs = simplehc_mhc::inputs_json(&report);
            let results = simplehc_mhc::results_json(&report);
            let summary = simplehc_mhc::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ServingHillClimb(path) => {
            let report = serving_hill_climb::run_default_harness()?;
            let inputs = serving_hill_climb::inputs_json(&report);
            let results = serving_hill_climb::results_json(&report);
            let summary = serving_hill_climb::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::TufazipOnlineCodec(path) => {
            let report = tufazip_online_codec::run_default_harness()?;
            let inputs = tufazip_online_codec::inputs_json(&report);
            let results = tufazip_online_codec::results_json(&report);
            let summary = tufazip_online_codec::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ClairicExtraction(path) => {
            let report = clairic_extraction::run_default_harness()?;
            let inputs = clairic_extraction::inputs_json(&report);
            let results = clairic_extraction::results_json(&report);
            let summary = clairic_extraction::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::InklingMiniServe(path) => {
            let report = inkling_mini_serve::run_default_harness()?;
            let inputs = inkling_mini_serve::inputs_json(&report);
            let results = inkling_mini_serve::results_json(&report);
            let summary = inkling_mini_serve::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MayaIk2Serve(path) => {
            let report = maya_ik2_serve::run_default_harness()?;
            let inputs = maya_ik2_serve::inputs_json(&report);
            let results = maya_ik2_serve::results_json(&report);
            let summary = maya_ik2_serve::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DecodeMegakernels(path) => {
            let report = decode_megakernels::run_default_harness()?;
            let inputs = decode_megakernels::inputs_json(&report);
            let results = decode_megakernels::results_json(&report);
            let summary = decode_megakernels::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch6Suite(path) => {
            let bundle = batch6::run_all()?;
            publish_batch6_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::DisaggKvTransfer(path) => {
            let report = disagg_kv_transfer::run_default_harness()?;
            let inputs = disagg_kv_transfer::inputs_json(&report);
            let results = disagg_kv_transfer::results_json(&report);
            let summary = disagg_kv_transfer::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SparkRdmaTp(path) => {
            let report = spark_rdma_tp::run_default_harness()?;
            let inputs = spark_rdma_tp::inputs_json(&report);
            let results = spark_rdma_tp::results_json(&report);
            let summary = spark_rdma_tp::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SparkKvCheckpoint(path) => {
            let report = spark_kv_checkpoint::run_default_harness()?;
            let inputs = spark_kv_checkpoint::inputs_json(&report);
            let results = spark_kv_checkpoint::results_json(&report);
            let summary = spark_kv_checkpoint::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SparkCopyDrafter(path) => {
            let report = spark_copydrafter::run_default_harness()?;
            let inputs = spark_copydrafter::inputs_json(&report);
            let results = spark_copydrafter::results_json(&report);
            let summary = spark_copydrafter::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::HypertuneKvAdapters(path) => {
            let report = hypertune_kv_adapters::run_default_harness()?;
            let inputs = hypertune_kv_adapters::inputs_json(&report);
            let results = hypertune_kv_adapters::results_json(&report);
            let summary = hypertune_kv_adapters::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Qwen38LongbookGdn(path) => {
            let report = qwen38_longbook_gdn::run_default_harness()?;
            let inputs = qwen38_longbook_gdn::inputs_json(&report);
            let results = qwen38_longbook_gdn::results_json(&report);
            let summary = qwen38_longbook_gdn::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LinQuantRecurrent(path) => {
            let report = linquant_recurrent::run_default_harness()?;
            let inputs = linquant_recurrent::inputs_json(&report);
            let results = linquant_recurrent::results_json(&report);
            let summary = linquant_recurrent::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::VarloopBudgetProtocol(path) => {
            let report = varloop_budget_protocol::run_default_harness()?;
            let inputs = varloop_budget_protocol::inputs_json(&report);
            let results = varloop_budget_protocol::results_json(&report);
            let summary = varloop_budget_protocol::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::TtfTernaryQwen(path) => {
            let report = ttf_ternary_qwen::run_default_harness()?;
            let inputs = ttf_ternary_qwen::inputs_json(&report);
            let results = ttf_ternary_qwen::results_json(&report);
            let summary = ttf_ternary_qwen::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DrafterOpdReplay(path) => {
            let report = drafter_opd_replay::run_default_harness()?;
            let inputs = drafter_opd_replay::inputs_json(&report);
            let results = drafter_opd_replay::results_json(&report);
            let summary = drafter_opd_replay::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::K2UnoMova(path) => {
            let report = k2_uno_mova::run_default_harness()?;
            let inputs = k2_uno_mova::inputs_json(&report);
            let results = k2_uno_mova::results_json(&report);
            let summary = k2_uno_mova::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Lfm3D1Decision(path) => {
            let report = lfm3_d1_decision::run_default_harness()?;
            let inputs = lfm3_d1_decision::inputs_json(&report);
            let results = lfm3_d1_decision::results_json(&report);
            let summary = lfm3_d1_decision::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch7Suite(path) => {
            let bundle = batch7::run_all()?;
            publish_batch7_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::PdlBreakableGraphs(path) => {
            let report = pdl_breakable_graphs::run_default_harness()?;
            let inputs = pdl_breakable_graphs::inputs_json(&report);
            let results = pdl_breakable_graphs::results_json(&report);
            let summary = pdl_breakable_graphs::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::TensorLeanKernelVerify(path) => {
            let report = tensorlean_kernel_verify::run_default_harness()?;
            let inputs = tensorlean_kernel_verify::inputs_json(&report);
            let results = tensorlean_kernel_verify::results_json(&report);
            let summary = tensorlean_kernel_verify::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::BlinkCpuFreeServing(path) => {
            let report = blink_cpu_free_serving::run_default_harness()?;
            let inputs = blink_cpu_free_serving::inputs_json(&report);
            let results = blink_cpu_free_serving::results_json(&report);
            let summary = blink_cpu_free_serving::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Qwen35NeuronSteering(path) => {
            let report = qwen35_neuron_steering::run_default_harness()?;
            let inputs = qwen35_neuron_steering::inputs_json(&report);
            let results = qwen35_neuron_steering::results_json(&report);
            let summary = qwen35_neuron_steering::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::HNetByteModeling(path) => {
            let report = hnet_byte_modeling::run_default_harness()?;
            let inputs = hnet_byte_modeling::inputs_json(&report);
            let results = hnet_byte_modeling::results_json(&report);
            let summary = hnet_byte_modeling::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::TkccHillclimbProfile(path) => {
            let report = tkcc_hillclimb_profile::run_default_harness()?;
            let inputs = tkcc_hillclimb_profile::inputs_json(&report);
            let results = tkcc_hillclimb_profile::results_json(&report);
            let summary = tkcc_hillclimb_profile::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Exl3LongCtxQuant(path) => {
            let report = exl3_longctx_quant::run_default_harness()?;
            let inputs = exl3_longctx_quant::inputs_json(&report);
            let results = exl3_longctx_quant::results_json(&report);
            let summary = exl3_longctx_quant::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SaberDFlash2Power(path) => {
            let report = saber_dflash2_power::run_default_harness()?;
            let inputs = saber_dflash2_power::inputs_json(&report);
            let results = saber_dflash2_power::results_json(&report);
            let summary = saber_dflash2_power::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MimoHySparse2Kv(path) => {
            let report = mimo_hysparse2_kv::run_default_harness()?;
            let inputs = mimo_hysparse2_kv::inputs_json(&report);
            let results = mimo_hysparse2_kv::results_json(&report);
            let summary = mimo_hysparse2_kv::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::AppleNvidiaSidecar(path) => {
            let report = apple_nvidia_sidecar::run_default_harness()?;
            let inputs = apple_nvidia_sidecar::inputs_json(&report);
            let results = apple_nvidia_sidecar::results_json(&report);
            let summary = apple_nvidia_sidecar::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::SiliconCoDesignAudit(path) => {
            let report = silicon_codesign_audit::run_default_harness()?;
            let inputs = silicon_codesign_audit::inputs_json(&report);
            let results = silicon_codesign_audit::results_json(&report);
            let summary = silicon_codesign_audit::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch8Suite(path) => {
            let bundle = batch8::run_all()?;
            publish_batch8_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::LinkSparseAttention(path) => {
            let report = link_sparse_attention::run_default_harness()?;
            let inputs = link_sparse_attention::inputs_json(&report);
            let results = link_sparse_attention::results_json(&report);
            let summary = link_sparse_attention::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MitQdaQuant(path) => {
            let report = mit_qda_quant::run_default_harness()?;
            let inputs = mit_qda_quant::inputs_json(&report);
            let results = mit_qda_quant::results_json(&report);
            let summary = mit_qda_quant::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KnowledgatorSys1Extract(path) => {
            let report = knowledgator_sys1_extract::run_default_harness()?;
            let inputs = knowledgator_sys1_extract::inputs_json(&report);
            let results = knowledgator_sys1_extract::results_json(&report);
            let summary = knowledgator_sys1_extract::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::UnlimitedOcrLongdoc(path) => {
            let report = unlimited_ocr_longdoc::run_default_harness()?;
            let inputs = unlimited_ocr_longdoc::inputs_json(&report);
            let results = unlimited_ocr_longdoc::results_json(&report);
            let summary = unlimited_ocr_longdoc::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LoopKvCompression(path) => {
            let report = loop_kv_compression::run_default_harness()?;
            let inputs = loop_kv_compression::inputs_json(&report);
            let results = loop_kv_compression::results_json(&report);
            let summary = loop_kv_compression::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::AloraPrefixCache(path) => {
            let report = alora_prefix_cache::run_default_harness()?;
            let inputs = alora_prefix_cache::inputs_json(&report);
            let results = alora_prefix_cache::results_json(&report);
            let summary = alora_prefix_cache::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::PerseusMpkRdma(path) => {
            let report = perseus_mpk_rdma::run_default_harness()?;
            let inputs = perseus_mpk_rdma::inputs_json(&report);
            let results = perseus_mpk_rdma::results_json(&report);
            let summary = perseus_mpk_rdma::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::MoeSpecRoofline(path) => {
            let report = moe_spec_roofline::run_default_harness()?;
            let inputs = moe_spec_roofline::inputs_json(&report);
            let results = moe_spec_roofline::results_json(&report);
            let summary = moe_spec_roofline::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::AnchorLabelDrafter(path) => {
            let report = anchor_label_drafter::run_default_harness()?;
            let inputs = anchor_label_drafter::inputs_json(&report);
            let results = anchor_label_drafter::results_json(&report);
            let summary = anchor_label_drafter::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ParasiteFeatureControl(path) => {
            let report = parasite_feature_control::run_default_harness()?;
            let inputs = parasite_feature_control::inputs_json(&report);
            let results = parasite_feature_control::results_json(&report);
            let summary = parasite_feature_control::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LingFp8KvAllocCap(path) => {
            let report = ling_fp8kv_alloc_cap::run_default_harness()?;
            let inputs = ling_fp8kv_alloc_cap::inputs_json(&report);
            let results = ling_fp8kv_alloc_cap::results_json(&report);
            let summary = ling_fp8kv_alloc_cap::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch9Suite(path) => {
            let bundle = batch9::run_all()?;
            publish_batch9_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
        Command::ActivationInjectionControls(path) => {
            let report = activation_injection_controls::run_default_harness()?;
            let inputs = activation_injection_controls::inputs_json(&report);
            let results = activation_injection_controls::results_json(&report);
            let summary = activation_injection_controls::summary_markdown(&report);
            let manifest =
                activation_injection_controls::manifest_json(&report, &inputs, &results, &summary);
            publish_with_manifest(&path, &inputs, &results, &summary, &manifest)?;
            Ok(summary)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = match parse(&args) {
        Ok(cmd) => cmd,
        Err(error) => {
            eprintln!("{error}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(command) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("inference-lab: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_all_valid_commands_and_rejects_bad_args() {
        assert_eq!(parse(&["--help".into()]), Ok(Command::Help));
        for cmd in [
            "benchmark",
            "diffusion",
            "trace",
            "stage",
            "power",
            "tokenizer",
            "related-work",
            "proof-completion",
            "drafting",
            "kernel-opt",
            "kv-routing",
            "adaptive-spec",
            "serving-bench",
            "deepseek-spark",
            "learned-compiler",
            "structured-read",
            "lifecycle",
            "batch2-suite",
            "rsi-bench",
            "djev-protocol",
            "rwkv-leaf",
            "ggml-ops",
            "rwkv-ggml",
            "graphical-sampling",
            "qwen38-quant",
            "deepseek-ref",
            "mla-comparison",
            "sparse-moe",
            "batch3-suite",
            "mtp-decoder",
            "flash-topk",
            "macaq-r",
            "hetero-rollout",
            "nvfp4-mxfp8",
            "spark-engines",
            "kv-compression",
            "expert-placement",
            "vision-infusion",
            "qwen38-qsa",
            "batch4-suite",
            "qsa-state-coherence",
            "inkling-residuals",
            "phoenix-nano",
            "nemotron-spark",
            "compact-hybrids",
            "native-recurrent-bench",
            "tiered-eval",
            "cpu-contention",
            "split-m2-p150",
            "quark-eval",
            "intervention-replay",
            "biollm-protocol",
            "batch5-suite",
            "adv-spec-reasoning",
            "nvme-coalescing",
            "harness-lora-compare",
            "latent-feedback",
            "simplehc-mhc",
            "serving-hill-climb",
            "tufazip-online-codec",
            "clairic-extraction",
            "inkling-mini-serve",
            "maya-ik2-serve",
            "decode-megakernels",
            "batch6-suite",
            "disagg-kv-transfer",
            "spark-rdma-tp",
            "spark-kv-checkpoint",
            "spark-copydrafter",
            "hypertune-kv-adapters",
            "qwen38-longbook-gdn",
            "linquant-recurrent",
            "varloop-budget-protocol",
            "ttf-ternary-qwen",
            "drafter-opd-replay",
            "k2-uno-mova",
            "lfm3-d1-decision",
            "batch7-suite",
            "pdl-breakable-graphs",
            "tensorlean-kernel-verify",
            "blink-cpu-free-serving",
            "qwen35-neuron-steering",
            "hnet-byte-modeling",
            "tkcc-hillclimb-profile",
            "exl3-longctx-quant",
            "saber-dflash2-power",
            "mimo-hysparse2-kv",
            "apple-nvidia-sidecar",
            "silicon-codesign-audit",
            "batch8-suite",
            "link-sparse-attention",
            "mit-qda-quant",
            "knowledgator-sys1-extract",
            "unlimited-ocr-longdoc",
            "loop-kv-compression",
            "alora-prefix-cache",
            "perseus-mpk-rdma",
            "moe-spec-roofline",
            "anchor-label-drafter",
            "parasite-feature-control",
            "ling-fp8kv-alloc-cap",
            "batch9-suite",
            "activation-injection-controls",
        ] {
            assert!(parse(&[cmd.into(), "--out".into(), "out-dir".into()]).is_ok());
        }
        for bad in [
            vec!["unknown", "--out", "dir"],
            vec!["benchmark", "--out", "--flag"],
            vec!["benchmark", "--out", "dir", "extra"],
            vec!["benchmark"],
        ] {
            let owned: Vec<String> = bad.into_iter().map(String::from).collect();
            assert!(parse(&owned).is_err());
        }
    }
}
