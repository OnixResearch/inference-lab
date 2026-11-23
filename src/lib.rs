//! Inference Lab pure deterministic core (`unsafe_code = "forbid"`, zero external dependencies).
//!
//! Each module implements bounded contract verification, deterministic reference math,
//! negative controls, and explicit provenance/hardware-gate reporting without performing
//! filesystem, network, clock, process, or accelerator I/O.

pub mod adaptive_spec;
pub mod batch2;
pub mod deepseek_spark;
pub mod diffusion;
pub mod drafting;
pub mod json;
pub mod kernel_opt;
pub mod kv_routing;
pub mod learned_compiler;
pub mod lifecycle;
pub mod power;
pub mod proof_completion;
pub mod provenance;
pub mod related_work;
pub mod seed;
pub mod serving_bench;
pub mod stage;
pub mod structured_read;
pub mod tokenizer;
pub mod trace;
