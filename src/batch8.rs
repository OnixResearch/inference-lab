//! Batch 8 aggregate orchestrator covering Issues #74–#84 (`batch8-suite`).
//!
//! Pure deterministic builder producing all 11 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    apple_nvidia_sidecar, blink_cpu_free_serving, exl3_longctx_quant, hnet_byte_modeling,
    mimo_hysparse2_kv, pdl_breakable_graphs, qwen35_neuron_steering, saber_dflash2_power,
    silicon_codesign_audit, tensorlean_kernel_verify, tkcc_hillclimb_profile,
};

const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[derive(Debug, Clone)]
pub struct SuiteArtifacts {
    pub issue: u32,
    pub command: &'static str,
    pub subdir: &'static str,
    pub execution_class: ExecutionClass,
    pub hardware_or_artifact_gate: ExecutionClass,
    pub inputs_json: String,
    pub results_json: String,
    pub summary_md: String,
}

#[derive(Debug, Clone)]
pub struct Batch8Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch8Bundle, String> {
    let r74 = pdl_breakable_graphs::run_default_harness()?;
    let r75 = tensorlean_kernel_verify::run_default_harness()?;
    let r76 = blink_cpu_free_serving::run_default_harness()?;
    let r77 = qwen35_neuron_steering::run_default_harness()?;
    let r78 = hnet_byte_modeling::run_default_harness()?;
    let r79 = tkcc_hillclimb_profile::run_default_harness()?;
    let r80 = exl3_longctx_quant::run_default_harness()?;
    let r81 = saber_dflash2_power::run_default_harness()?;
    let r82 = mimo_hysparse2_kv::run_default_harness()?;
    let r83 = apple_nvidia_sidecar::run_default_harness()?;
    let r84 = silicon_codesign_audit::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 74,
            command: "pdl-breakable-graphs",
            subdir: "issue-74-pdl-breakable-graphs",
            execution_class: r74.execution_class,
            hardware_or_artifact_gate: r74.hardware_gate.status,
            inputs_json: pdl_breakable_graphs::inputs_json(&r74),
            results_json: pdl_breakable_graphs::results_json(&r74),
            summary_md: pdl_breakable_graphs::summary_markdown(&r74),
        },
        SuiteArtifacts {
            issue: 75,
            command: "tensorlean-kernel-verify",
            subdir: "issue-75-tensorlean-kernel-verify",
            execution_class: r75.execution_class,
            hardware_or_artifact_gate: r75.hardware_gate.status,
            inputs_json: tensorlean_kernel_verify::inputs_json(&r75),
            results_json: tensorlean_kernel_verify::results_json(&r75),
            summary_md: tensorlean_kernel_verify::summary_markdown(&r75),
        },
        SuiteArtifacts {
            issue: 76,
            command: "blink-cpu-free-serving",
            subdir: "issue-76-blink-cpu-free-serving",
            execution_class: r76.execution_class,
            hardware_or_artifact_gate: r76.hardware_gate.status,
            inputs_json: blink_cpu_free_serving::inputs_json(&r76),
            results_json: blink_cpu_free_serving::results_json(&r76),
            summary_md: blink_cpu_free_serving::summary_markdown(&r76),
        },
        SuiteArtifacts {
            issue: 77,
            command: "qwen35-neuron-steering",
            subdir: "issue-77-qwen35-neuron-steering",
            execution_class: r77.execution_class,
            hardware_or_artifact_gate: r77.hardware_gate.status,
            inputs_json: qwen35_neuron_steering::inputs_json(&r77),
            results_json: qwen35_neuron_steering::results_json(&r77),
            summary_md: qwen35_neuron_steering::summary_markdown(&r77),
        },
        SuiteArtifacts {
            issue: 78,
            command: "hnet-byte-modeling",
            subdir: "issue-78-hnet-byte-modeling",
            execution_class: r78.execution_class,
            hardware_or_artifact_gate: r78.hardware_gate.status,
            inputs_json: hnet_byte_modeling::inputs_json(&r78),
            results_json: hnet_byte_modeling::results_json(&r78),
            summary_md: hnet_byte_modeling::summary_markdown(&r78),
        },
        SuiteArtifacts {
            issue: 79,
            command: "tkcc-hillclimb-profile",
            subdir: "issue-79-tkcc-hillclimb-profile",
            execution_class: r79.execution_class,
            hardware_or_artifact_gate: r79.hardware_gate.status,
            inputs_json: tkcc_hillclimb_profile::inputs_json(&r79),
            results_json: tkcc_hillclimb_profile::results_json(&r79),
            summary_md: tkcc_hillclimb_profile::summary_markdown(&r79),
        },
        SuiteArtifacts {
            issue: 80,
            command: "exl3-longctx-quant",
            subdir: "issue-80-exl3-longctx-quant",
            execution_class: r80.execution_class,
            hardware_or_artifact_gate: r80.hardware_gate.status,
            inputs_json: exl3_longctx_quant::inputs_json(&r80),
            results_json: exl3_longctx_quant::results_json(&r80),
            summary_md: exl3_longctx_quant::summary_markdown(&r80),
        },
        SuiteArtifacts {
            issue: 81,
            command: "saber-dflash2-power",
            subdir: "issue-81-saber-dflash2-power",
            execution_class: r81.execution_class,
            hardware_or_artifact_gate: r81.hardware_gate.status,
            inputs_json: saber_dflash2_power::inputs_json(&r81),
            results_json: saber_dflash2_power::results_json(&r81),
            summary_md: saber_dflash2_power::summary_markdown(&r81),
        },
        SuiteArtifacts {
            issue: 82,
            command: "mimo-hysparse2-kv",
            subdir: "issue-82-mimo-hysparse2-kv",
            execution_class: r82.execution_class,
            hardware_or_artifact_gate: r82.hardware_gate.status,
            inputs_json: mimo_hysparse2_kv::inputs_json(&r82),
            results_json: mimo_hysparse2_kv::results_json(&r82),
            summary_md: mimo_hysparse2_kv::summary_markdown(&r82),
        },
        SuiteArtifacts {
            issue: 83,
            command: "apple-nvidia-sidecar",
            subdir: "issue-83-apple-nvidia-sidecar",
            execution_class: r83.execution_class,
            hardware_or_artifact_gate: r83.hardware_gate.status,
            inputs_json: apple_nvidia_sidecar::inputs_json(&r83),
            results_json: apple_nvidia_sidecar::results_json(&r83),
            summary_md: apple_nvidia_sidecar::summary_markdown(&r83),
        },
        SuiteArtifacts {
            issue: 84,
            command: "silicon-codesign-audit",
            subdir: "issue-84-silicon-codesign-audit",
            execution_class: r84.execution_class,
            hardware_or_artifact_gate: r84.hardware_gate.status,
            inputs_json: silicon_codesign_audit::inputs_json(&r84),
            results_json: silicon_codesign_audit::results_json(&r84),
            summary_md: silicon_codesign_audit::summary_markdown(&r84),
        },
    ];

    let bounded_count = suites
        .iter()
        .filter(|s| s.execution_class == ExecutionClass::BoundedContractHarness)
        .count();
    let blocked_count = suites
        .iter()
        .filter(|s| s.execution_class == ExecutionClass::BlockedOnArtifacts)
        .count();

    let suite_meta: Vec<String> = suites
        .iter()
        .map(|s| {
            format!(
                "{{\"issue\":{},\"command\":{},\"subdir\":{},\"execution_class\":{},\"hardware_or_artifact_gate\":{}}}",
                s.issue,
                quote(s.command),
                quote(s.subdir),
                quote(s.execution_class.as_str()),
                quote(s.hardware_or_artifact_gate.as_str()),
            )
        })
        .collect();

    let aggregate_inputs_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch8-suite.inputs.v1\",\n",
            "  \"issues\": [74,75,76,77,78,79,80,81,82,83,84],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch8-suite.results.v1\",\n",
            "  \"issues\": [74,75,76,77,78,79,80,81,82,83,84],\n",
            "  \"total_suites\": {},\n",
            "  \"bounded_contract_harness_suites\": {},\n",
            "  \"blocked_on_artifacts_suites\": {},\n",
            "  \"physical_hardware_executed\": false,\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suites.len(),
        bounded_count,
        blocked_count,
        suite_meta.join(",")
    );

    let mut aggregate_summary_md = String::new();
    writeln!(
        aggregate_summary_md,
        "# Batch 8 Contract & Reference-Math Harness Suite (`#74`–`#84`)"
    )
    .unwrap();
    writeln!(aggregate_summary_md).unwrap();
    writeln!(
        aggregate_summary_md,
        "- Total suites: {} (`bounded_contract_harness`: {}, `blocked_on_artifacts`: {})",
        suites.len(),
        bounded_count,
        blocked_count
    )
    .unwrap();
    writeln!(
        aggregate_summary_md,
        "- Physical Blackwell/DGX Spark / SmartNIC RDMA / Strix Halo + RX 7900 XTX / Dual R9700 / Mac Studio + RTX 5090 eGPU / custom silicon execution: `false`"
    )
    .unwrap();
    writeln!(aggregate_summary_md).unwrap();
    writeln!(
        aggregate_summary_md,
        "| Issue | Subcommand | Subdir | Execution Class | Hardware / Artifact Gate |"
    )
    .unwrap();
    writeln!(aggregate_summary_md, "| ---: | --- | --- | --- | --- |").unwrap();
    for s in &suites {
        writeln!(
            aggregate_summary_md,
            "| `#{}` | `{}` | `{}` | `{}` | `{}` |",
            s.issue,
            s.command,
            s.subdir,
            s.execution_class.as_str(),
            s.hardware_or_artifact_gate.as_str()
        )
        .unwrap();
    }
    writeln!(aggregate_summary_md).unwrap();
    writeln!(
        aggregate_summary_md,
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references (`default_enabled: false`). Physical accelerators, SmartNIC GPUDirect RDMA clusters, eGPU links, live model serves, fabricated silicon, and unlocated/unverified upstream artifacts remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch8-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],"
    )
    .unwrap();
    writeln!(manifest_json, "  \"digest_algorithm\": \"sha256\",").unwrap();
    writeln!(manifest_json, "  \"top_level_artifact_sha256\": {{").unwrap();
    writeln!(
        manifest_json,
        "    \"inputs.json\": {},",
        quote(&sha256_hex(aggregate_inputs_json.as_bytes()))
    )
    .unwrap();
    writeln!(
        manifest_json,
        "    \"results.json\": {},",
        quote(&sha256_hex(aggregate_results_json.as_bytes()))
    )
    .unwrap();
    writeln!(
        manifest_json,
        "    \"summary.md\": {},",
        quote(&sha256_hex(aggregate_summary_md.as_bytes()))
    )
    .unwrap();
    writeln!(manifest_json, "    \"COMPLETE\": {}", quote(EMPTY_SHA256)).unwrap();
    writeln!(manifest_json, "  }},").unwrap();
    writeln!(manifest_json, "  \"suites\": [").unwrap();
    for (idx, s) in suites.iter().enumerate() {
        let comma = if idx + 1 == suites.len() { "" } else { "," };
        writeln!(manifest_json, "    {{").unwrap();
        writeln!(manifest_json, "      \"subdir\": {},", quote(s.subdir)).unwrap();
        writeln!(manifest_json, "      \"issues\": [{}],", s.issue).unwrap();
        writeln!(
            manifest_json,
            "      \"command\": [\"inference-lab\", {}, \"--out\", {}],",
            quote(s.command),
            quote(&format!("runs/validation/{}", s.subdir))
        )
        .unwrap();
        writeln!(
            manifest_json,
            "      \"execution_class\": {},",
            quote(s.execution_class.as_str())
        )
        .unwrap();
        writeln!(
            manifest_json,
            "      \"physical_hardware_or_artifact_gate\": {},",
            quote(s.hardware_or_artifact_gate.as_str())
        )
        .unwrap();
        writeln!(manifest_json, "      \"artifact_sha256\": {{").unwrap();
        writeln!(
            manifest_json,
            "        \"inputs.json\": {},",
            quote(&sha256_hex(s.inputs_json.as_bytes()))
        )
        .unwrap();
        writeln!(
            manifest_json,
            "        \"results.json\": {},",
            quote(&sha256_hex(s.results_json.as_bytes()))
        )
        .unwrap();
        writeln!(
            manifest_json,
            "        \"summary.md\": {},",
            quote(&sha256_hex(s.summary_md.as_bytes()))
        )
        .unwrap();
        writeln!(
            manifest_json,
            "        \"COMPLETE\": {}",
            quote(EMPTY_SHA256)
        )
        .unwrap();
        writeln!(manifest_json, "      }}").unwrap();
        writeln!(manifest_json, "    }}{comma}").unwrap();
    }
    writeln!(manifest_json, "  ]").unwrap();
    writeln!(manifest_json, "}}").unwrap();

    Ok(Batch8Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
