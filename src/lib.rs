//! Inference Lab pure deterministic core (`unsafe_code = "forbid"`, zero external dependencies).
//!
//! Each module implements bounded contract verification, deterministic reference math,
//! negative controls, and explicit provenance/hardware-gate reporting without performing
//! filesystem, network, clock, process, or accelerator I/O.

pub mod adaptive_spec;
pub mod batch2;
pub mod batch3;
pub mod batch4;
pub mod deepseek_ref;
pub mod deepseek_spark;
pub mod diffusion;
pub mod djev_protocol;
pub mod drafting;
pub mod expert_placement;
pub mod flash_topk;
pub mod ggml_ops;
pub mod graphical_sampling;
pub mod hetero_rollout;
pub mod json;
pub mod kernel_opt;
pub mod kv_compression;
pub mod kv_routing;
pub mod learned_compiler;
pub mod lifecycle;
pub mod macaq_r;
pub mod mla_comparison;
pub mod mtp_decoder;
pub mod nvfp4_mxfp8;
pub mod power;
pub mod proof_completion;
pub mod provenance;
pub mod qwen38_qsa;
pub mod qwen38_quant;
pub mod related_work;
pub mod rsi_bench;
pub mod rwkv_ggml;
pub mod rwkv_leaf;
pub mod seed;
pub mod serving_bench;
pub mod spark_engines;
pub mod sparse_moe;
pub mod stage;
pub mod structured_read;
pub mod tokenizer;
pub mod trace;
pub mod vision_infusion;
