//! Inference Lab pure deterministic core (`unsafe_code = "forbid"`, zero external dependencies).
//!
//! Each module implements bounded contract verification, deterministic reference math,
//! negative controls, and explicit provenance/hardware-gate reporting without performing
//! filesystem, network, clock, process, or accelerator I/O.

pub mod diffusion;
pub mod json;
pub mod power;
pub mod provenance;
pub mod related_work;
pub mod seed;
pub mod stage;
pub mod tokenizer;
pub mod trace;
