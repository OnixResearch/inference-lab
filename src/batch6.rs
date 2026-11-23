//! Batch 6 aggregate orchestrator covering final Issues #47–#51, #53–#55, #57, #59, #61 (`batch6-suite`).
//!
//! Pure deterministic builder producing all 11 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    adv_spec_reasoning, clairic_extraction, decode_megakernels, harness_lora_compare,
    inkling_mini_serve, latent_feedback, maya_ik2_serve, nvme_coalescing, serving_hill_climb,
    simplehc_mhc, tufazip_online_codec,
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
pub struct Batch6Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch6Bundle, String> {
    let r47 = adv_spec_reasoning::run_default_harness()?;
    let r48 = nvme_coalescing::run_default_harness()?;
    let r49 = harness_lora_compare::run_default_harness()?;
    let r50 = latent_feedback::run_default_harness()?;
    let r51 = simplehc_mhc::run_default_harness()?;
    let r53 = serving_hill_climb::run_default_harness()?;
    let r54 = tufazip_online_codec::run_default_harness()?;
    let r55 = clairic_extraction::run_default_harness()?;
    let r57 = inkling_mini_serve::run_default_harness()?;
    let r59 = maya_ik2_serve::run_default_harness()?;
    let r61 = decode_megakernels::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 47,
            command: "adv-spec-reasoning",
            subdir: "issue-47-adv-spec-reasoning",
            execution_class: r47.execution_class,
            hardware_or_artifact_gate: r47.hardware_gate.status,
            inputs_json: adv_spec_reasoning::inputs_json(&r47),
            results_json: adv_spec_reasoning::results_json(&r47),
            summary_md: adv_spec_reasoning::summary_markdown(&r47),
        },
        SuiteArtifacts {
            issue: 48,
            command: "nvme-coalescing",
            subdir: "issue-48-nvme-coalescing",
            execution_class: r48.execution_class,
            hardware_or_artifact_gate: r48.hardware_gate.status,
            inputs_json: nvme_coalescing::inputs_json(&r48),
            results_json: nvme_coalescing::results_json(&r48),
            summary_md: nvme_coalescing::summary_markdown(&r48),
        },
        SuiteArtifacts {
            issue: 49,
            command: "harness-lora-compare",
            subdir: "issue-49-harness-lora-compare",
            execution_class: r49.execution_class,
            hardware_or_artifact_gate: r49.hardware_gate.status,
            inputs_json: harness_lora_compare::inputs_json(&r49),
            results_json: harness_lora_compare::results_json(&r49),
            summary_md: harness_lora_compare::summary_markdown(&r49),
        },
        SuiteArtifacts {
            issue: 50,
            command: "latent-feedback",
            subdir: "issue-50-latent-feedback",
            execution_class: r50.execution_class,
            hardware_or_artifact_gate: r50.hardware_gate.status,
            inputs_json: latent_feedback::inputs_json(&r50),
            results_json: latent_feedback::results_json(&r50),
            summary_md: latent_feedback::summary_markdown(&r50),
        },
        SuiteArtifacts {
            issue: 51,
            command: "simplehc-mhc",
            subdir: "issue-51-simplehc-mhc",
            execution_class: r51.execution_class,
            hardware_or_artifact_gate: r51.hardware_gate.status,
            inputs_json: simplehc_mhc::inputs_json(&r51),
            results_json: simplehc_mhc::results_json(&r51),
            summary_md: simplehc_mhc::summary_markdown(&r51),
        },
        SuiteArtifacts {
            issue: 53,
            command: "serving-hill-climb",
            subdir: "issue-53-serving-hill-climb",
            execution_class: r53.execution_class,
            hardware_or_artifact_gate: r53.hardware_gate.status,
            inputs_json: serving_hill_climb::inputs_json(&r53),
            results_json: serving_hill_climb::results_json(&r53),
            summary_md: serving_hill_climb::summary_markdown(&r53),
        },
        SuiteArtifacts {
            issue: 54,
            command: "tufazip-online-codec",
            subdir: "issue-54-tufazip-online-codec",
            execution_class: r54.execution_class,
            hardware_or_artifact_gate: r54.hardware_gate.status,
            inputs_json: tufazip_online_codec::inputs_json(&r54),
            results_json: tufazip_online_codec::results_json(&r54),
            summary_md: tufazip_online_codec::summary_markdown(&r54),
        },
        SuiteArtifacts {
            issue: 55,
            command: "clairic-extraction",
            subdir: "issue-55-clairic-extraction",
            execution_class: r55.execution_class,
            hardware_or_artifact_gate: r55.hardware_gate.status,
            inputs_json: clairic_extraction::inputs_json(&r55),
            results_json: clairic_extraction::results_json(&r55),
            summary_md: clairic_extraction::summary_markdown(&r55),
        },
        SuiteArtifacts {
            issue: 57,
            command: "inkling-mini-serve",
            subdir: "issue-57-inkling-mini-serve",
            execution_class: r57.execution_class,
            hardware_or_artifact_gate: r57.hardware_gate.status,
            inputs_json: inkling_mini_serve::inputs_json(&r57),
            results_json: inkling_mini_serve::results_json(&r57),
            summary_md: inkling_mini_serve::summary_markdown(&r57),
        },
        SuiteArtifacts {
            issue: 59,
            command: "maya-ik2-serve",
            subdir: "issue-59-maya-ik2-serve",
            execution_class: r59.execution_class,
            hardware_or_artifact_gate: r59.hardware_gate.status,
            inputs_json: maya_ik2_serve::inputs_json(&r59),
            results_json: maya_ik2_serve::results_json(&r59),
            summary_md: maya_ik2_serve::summary_markdown(&r59),
        },
        SuiteArtifacts {
            issue: 61,
            command: "decode-megakernels",
            subdir: "issue-61-decode-megakernels",
            execution_class: r61.execution_class,
            hardware_or_artifact_gate: r61.hardware_gate.status,
            inputs_json: decode_megakernels::inputs_json(&r61),
            results_json: decode_megakernels::results_json(&r61),
            summary_md: decode_megakernels::summary_markdown(&r61),
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
            "  \"schema\": \"inference-lab.batch6-suite.inputs.v1\",\n",
            "  \"issues\": [47,48,49,50,51,53,54,55,57,59,61],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch6-suite.results.v1\",\n",
            "  \"issues\": [47,48,49,50,51,53,54,55,57,59,61],\n",
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
        "# Batch 6 Contract & Reference-Math Harness Suite (`#47`–`#51`, `#53`–`#55`, `#57`, `#59`, `#61`)"
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
        "- Physical GPU / NVMe / 2xB200 / RTX 3090 sm_86 / DGX Spark / live server execution: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references. Physical accelerators, NVMe drives, live model serves, and missing checkpoints/repos remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch6-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [47, 48, 49, 50, 51, 53, 54, 55, 57, 59, 61],"
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

    Ok(Batch6Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
