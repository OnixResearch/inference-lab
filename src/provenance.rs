//! Shared provenance, digest identity, execution classes, and non-collapsing outcomes.

use crate::json::{quote, sha256_hex};

/// Explicit execution classification separating deterministic contract harnesses from
/// physical accelerator runs and artifact-blocked related work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionClass {
    BoundedContractHarness,
    UnavailableHardwareRuntime,
    BlockedOnArtifacts,
}

impl ExecutionClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BoundedContractHarness => "bounded_contract_harness",
            Self::UnavailableHardwareRuntime => "unavailable_hardware_runtime",
            Self::BlockedOnArtifacts => "blocked_on_artifacts",
        }
    }
}

/// Non-collapsing outcome taxonomy shared across Inference Lab harnesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutcomeStatus {
    Accepted,
    Invalid,
    Unsupported,
    Exhausted,
    Timeout,
    Oom,
    Cancelled,
    InternalFailure,
    UnavailableHardwareRuntime,
    BlockedOnArtifacts,
}

impl OutcomeStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Invalid => "invalid",
            Self::Unsupported => "unsupported",
            Self::Exhausted => "exhausted",
            Self::Timeout => "timeout",
            Self::Oom => "oom",
            Self::Cancelled => "cancelled",
            Self::InternalFailure => "internal_failure",
            Self::UnavailableHardwareRuntime => "unavailable_hardware_runtime",
            Self::BlockedOnArtifacts => "blocked_on_artifacts",
        }
    }
}

/// Algorithm-tagged content digest so external identities are never silently translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigestRef {
    pub algorithm: String,
    pub hex: String,
}

impl DigestRef {
    pub fn sha256(bytes: &[u8]) -> Self {
        Self {
            algorithm: "sha256".into(),
            hex: sha256_hex(bytes),
        }
    }

    pub fn from_parts(algorithm: impl Into<String>, hex: impl Into<String>) -> Self {
        Self {
            algorithm: algorithm.into(),
            hex: hex.into(),
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"algorithm\":{},\"hex\":{}}}",
            quote(&self.algorithm),
            quote(&self.hex)
        )
    }
}

/// Explicit record of physical hardware availability alongside a bounded contract run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareGate {
    pub target_hardware: String,
    pub status: ExecutionClass,
    pub reason: String,
}

impl HardwareGate {
    pub fn unavailable(target_hardware: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            target_hardware: target_hardware.into(),
            status: ExecutionClass::UnavailableHardwareRuntime,
            reason: reason.into(),
        }
    }

    pub fn blocked_on_artifacts(
        target_hardware: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            target_hardware: target_hardware.into(),
            status: ExecutionClass::BlockedOnArtifacts,
            reason: reason.into(),
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"target_hardware\":{},\"status\":{},\"reason\":{}}}",
            quote(&self.target_hardware),
            quote(self.status.as_str()),
            quote(&self.reason)
        )
    }
}
