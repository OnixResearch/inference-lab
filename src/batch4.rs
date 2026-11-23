//! Batch 4 aggregate orchestrator covering Issues #22–#31 (`batch4-suite`).
//!
//! Pure deterministic builder producing all 10 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    expert_placement, flash_topk, hetero_rollout, kv_compression, macaq_r, mtp_decoder,
    nvfp4_mxfp8, qwen38_qsa, spark_engines, vision_infusion,
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
pub struct Batch4Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch4Bundle, String> {
    let r22 = mtp_decoder::run_default_harness()?;
    let r23 = flash_topk::run_default_harness()?;
    let r24 = macaq_r::run_default_harness()?;
    let r25 = hetero_rollout::run_default_harness()?;
    let r26 = nvfp4_mxfp8::run_default_harness()?;
    let r27 = spark_engines::run_default_harness()?;
    let r28 = kv_compression::run_default_harness()?;
    let r29 = expert_placement::run_default_harness()?;
    let r30 = vision_infusion::run_default_harness()?;
    let r31 = qwen38_qsa::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 22,
            command: "mtp-decoder",
            subdir: "issue-22-mtp-decoder",
            execution_class: r22.execution_class,
            hardware_or_artifact_gate: r22.hardware_gate.status,
            inputs_json: mtp_decoder::inputs_json(&r22),
            results_json: mtp_decoder::results_json(&r22),
            summary_md: mtp_decoder::summary_markdown(&r22),
        },
        SuiteArtifacts {
            issue: 23,
            command: "flash-topk",
            subdir: "issue-23-flash-topk",
            execution_class: r23.execution_class,
            hardware_or_artifact_gate: r23.hardware_gate.status,
            inputs_json: flash_topk::inputs_json(&r23),
            results_json: flash_topk::results_json(&r23),
            summary_md: flash_topk::summary_markdown(&r23),
        },
        SuiteArtifacts {
            issue: 24,
            command: "macaq-r",
            subdir: "issue-24-macaq-r",
            execution_class: r24.execution_class,
            hardware_or_artifact_gate: r24.hardware_gate.status,
            inputs_json: macaq_r::inputs_json(&r24),
            results_json: macaq_r::results_json(&r24),
            summary_md: macaq_r::summary_markdown(&r24),
        },
        SuiteArtifacts {
            issue: 25,
            command: "hetero-rollout",
            subdir: "issue-25-hetero-rollout",
            execution_class: r25.execution_class,
            hardware_or_artifact_gate: r25.hardware_gate.status,
            inputs_json: hetero_rollout::inputs_json(&r25),
            results_json: hetero_rollout::results_json(&r25),
            summary_md: hetero_rollout::summary_markdown(&r25),
        },
        SuiteArtifacts {
            issue: 26,
            command: "nvfp4-mxfp8",
            subdir: "issue-26-nvfp4-mxfp8",
            execution_class: r26.execution_class,
            hardware_or_artifact_gate: r26.hardware_gate.status,
            inputs_json: nvfp4_mxfp8::inputs_json(&r26),
            results_json: nvfp4_mxfp8::results_json(&r26),
            summary_md: nvfp4_mxfp8::summary_markdown(&r26),
        },
        SuiteArtifacts {
            issue: 27,
            command: "spark-engines",
            subdir: "issue-27-spark-engines",
            execution_class: r27.execution_class,
            hardware_or_artifact_gate: r27.hardware_gate.status,
            inputs_json: spark_engines::inputs_json(&r27),
            results_json: spark_engines::results_json(&r27),
            summary_md: spark_engines::summary_markdown(&r27),
        },
        SuiteArtifacts {
            issue: 28,
            command: "kv-compression",
            subdir: "issue-28-kv-compression",
            execution_class: r28.execution_class,
            hardware_or_artifact_gate: r28.hardware_gate.status,
            inputs_json: kv_compression::inputs_json(&r28),
            results_json: kv_compression::results_json(&r28),
            summary_md: kv_compression::summary_markdown(&r28),
        },
        SuiteArtifacts {
            issue: 29,
            command: "expert-placement",
            subdir: "issue-29-expert-placement",
            execution_class: r29.execution_class,
            hardware_or_artifact_gate: r29.hardware_gate.status,
            inputs_json: expert_placement::inputs_json(&r29),
            results_json: expert_placement::results_json(&r29),
            summary_md: expert_placement::summary_markdown(&r29),
        },
        SuiteArtifacts {
            issue: 30,
            command: "vision-infusion",
            subdir: "issue-30-vision-infusion",
            execution_class: r30.execution_class,
            hardware_or_artifact_gate: r30.hardware_gate.status,
            inputs_json: vision_infusion::inputs_json(&r30),
            results_json: vision_infusion::results_json(&r30),
            summary_md: vision_infusion::summary_markdown(&r30),
        },
        SuiteArtifacts {
            issue: 31,
            command: "qwen38-qsa",
            subdir: "issue-31-qwen38-qsa",
            execution_class: r31.execution_class,
            hardware_or_artifact_gate: r31.hardware_gate.status,
            inputs_json: qwen38_qsa::inputs_json(&r31),
            results_json: qwen38_qsa::results_json(&r31),
            summary_md: qwen38_qsa::summary_markdown(&r31),
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
            "  \"schema\": \"inference-lab.batch4-suite.inputs.v1\",\n",
            "  \"issues\": [22,23,24,25,26,27,28,29,30,31],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch4-suite.results.v1\",\n",
            "  \"issues\": [22,23,24,25,26,27,28,29,30,31],\n",
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
        "# Batch 4 Contract & Reference-Math Harness Suite (`#22`–`#31`)"
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
        "- Physical GPU / NVFP4 / TensorRT Edge-LLM / Hyperflare / RustingStrata / Qwen3.8-27B QSA live serve executed: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references. Physical hardware and missing trained checkpoints remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch4-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],"
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

    Ok(Batch4Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
