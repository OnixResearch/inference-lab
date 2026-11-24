//! Batch 7 aggregate orchestrator covering Issues #62–#73 (`batch7-suite`).
//!
//! Pure deterministic builder producing all 12 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    disagg_kv_transfer, drafter_opd_replay, hypertune_kv_adapters, k2_uno_mova, lfm3_d1_decision,
    linquant_recurrent, qwen38_longbook_gdn, spark_copydrafter, spark_kv_checkpoint, spark_rdma_tp,
    ttf_ternary_qwen, varloop_budget_protocol,
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
pub struct Batch7Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch7Bundle, String> {
    let r62 = disagg_kv_transfer::run_default_harness()?;
    let r63 = spark_rdma_tp::run_default_harness()?;
    let r64 = spark_kv_checkpoint::run_default_harness()?;
    let r65 = spark_copydrafter::run_default_harness()?;
    let r66 = hypertune_kv_adapters::run_default_harness()?;
    let r67 = qwen38_longbook_gdn::run_default_harness()?;
    let r68 = linquant_recurrent::run_default_harness()?;
    let r69 = varloop_budget_protocol::run_default_harness()?;
    let r70 = ttf_ternary_qwen::run_default_harness()?;
    let r71 = drafter_opd_replay::run_default_harness()?;
    let r72 = k2_uno_mova::run_default_harness()?;
    let r73 = lfm3_d1_decision::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 62,
            command: "disagg-kv-transfer",
            subdir: "issue-62-disagg-kv-transfer",
            execution_class: r62.execution_class,
            hardware_or_artifact_gate: r62.hardware_gate.status,
            inputs_json: disagg_kv_transfer::inputs_json(&r62),
            results_json: disagg_kv_transfer::results_json(&r62),
            summary_md: disagg_kv_transfer::summary_markdown(&r62),
        },
        SuiteArtifacts {
            issue: 63,
            command: "spark-rdma-tp",
            subdir: "issue-63-spark-rdma-tp",
            execution_class: r63.execution_class,
            hardware_or_artifact_gate: r63.hardware_gate.status,
            inputs_json: spark_rdma_tp::inputs_json(&r63),
            results_json: spark_rdma_tp::results_json(&r63),
            summary_md: spark_rdma_tp::summary_markdown(&r63),
        },
        SuiteArtifacts {
            issue: 64,
            command: "spark-kv-checkpoint",
            subdir: "issue-64-spark-kv-checkpoint",
            execution_class: r64.execution_class,
            hardware_or_artifact_gate: r64.hardware_gate.status,
            inputs_json: spark_kv_checkpoint::inputs_json(&r64),
            results_json: spark_kv_checkpoint::results_json(&r64),
            summary_md: spark_kv_checkpoint::summary_markdown(&r64),
        },
        SuiteArtifacts {
            issue: 65,
            command: "spark-copydrafter",
            subdir: "issue-65-spark-copydrafter",
            execution_class: r65.execution_class,
            hardware_or_artifact_gate: r65.hardware_gate.status,
            inputs_json: spark_copydrafter::inputs_json(&r65),
            results_json: spark_copydrafter::results_json(&r65),
            summary_md: spark_copydrafter::summary_markdown(&r65),
        },
        SuiteArtifacts {
            issue: 66,
            command: "hypertune-kv-adapters",
            subdir: "issue-66-hypertune-kv-adapters",
            execution_class: r66.execution_class,
            hardware_or_artifact_gate: r66.hardware_gate.status,
            inputs_json: hypertune_kv_adapters::inputs_json(&r66),
            results_json: hypertune_kv_adapters::results_json(&r66),
            summary_md: hypertune_kv_adapters::summary_markdown(&r66),
        },
        SuiteArtifacts {
            issue: 67,
            command: "qwen38-longbook-gdn",
            subdir: "issue-67-qwen38-longbook-gdn",
            execution_class: r67.execution_class,
            hardware_or_artifact_gate: r67.hardware_gate.status,
            inputs_json: qwen38_longbook_gdn::inputs_json(&r67),
            results_json: qwen38_longbook_gdn::results_json(&r67),
            summary_md: qwen38_longbook_gdn::summary_markdown(&r67),
        },
        SuiteArtifacts {
            issue: 68,
            command: "linquant-recurrent",
            subdir: "issue-68-linquant-recurrent",
            execution_class: r68.execution_class,
            hardware_or_artifact_gate: r68.hardware_gate.status,
            inputs_json: linquant_recurrent::inputs_json(&r68),
            results_json: linquant_recurrent::results_json(&r68),
            summary_md: linquant_recurrent::summary_markdown(&r68),
        },
        SuiteArtifacts {
            issue: 69,
            command: "varloop-budget-protocol",
            subdir: "issue-69-varloop-budget-protocol",
            execution_class: r69.execution_class,
            hardware_or_artifact_gate: r69.hardware_gate.status,
            inputs_json: varloop_budget_protocol::inputs_json(&r69),
            results_json: varloop_budget_protocol::results_json(&r69),
            summary_md: varloop_budget_protocol::summary_markdown(&r69),
        },
        SuiteArtifacts {
            issue: 70,
            command: "ttf-ternary-qwen",
            subdir: "issue-70-ttf-ternary-qwen",
            execution_class: r70.execution_class,
            hardware_or_artifact_gate: r70.hardware_gate.status,
            inputs_json: ttf_ternary_qwen::inputs_json(&r70),
            results_json: ttf_ternary_qwen::results_json(&r70),
            summary_md: ttf_ternary_qwen::summary_markdown(&r70),
        },
        SuiteArtifacts {
            issue: 71,
            command: "drafter-opd-replay",
            subdir: "issue-71-drafter-opd-replay",
            execution_class: r71.execution_class,
            hardware_or_artifact_gate: r71.hardware_gate.status,
            inputs_json: drafter_opd_replay::inputs_json(&r71),
            results_json: drafter_opd_replay::results_json(&r71),
            summary_md: drafter_opd_replay::summary_markdown(&r71),
        },
        SuiteArtifacts {
            issue: 72,
            command: "k2-uno-mova",
            subdir: "issue-72-k2-uno-mova",
            execution_class: r72.execution_class,
            hardware_or_artifact_gate: r72.hardware_gate.status,
            inputs_json: k2_uno_mova::inputs_json(&r72),
            results_json: k2_uno_mova::results_json(&r72),
            summary_md: k2_uno_mova::summary_markdown(&r72),
        },
        SuiteArtifacts {
            issue: 73,
            command: "lfm3-d1-decision",
            subdir: "issue-73-lfm3-d1-decision",
            execution_class: r73.execution_class,
            hardware_or_artifact_gate: r73.hardware_gate.status,
            inputs_json: lfm3_d1_decision::inputs_json(&r73),
            results_json: lfm3_d1_decision::results_json(&r73),
            summary_md: lfm3_d1_decision::summary_markdown(&r73),
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
            "  \"schema\": \"inference-lab.batch7-suite.inputs.v1\",\n",
            "  \"issues\": [62,63,64,65,66,67,68,69,70,71,72,73],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch7-suite.results.v1\",\n",
            "  \"issues\": [62,63,64,65,66,67,68,69,70,71,72,73],\n",
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
        "# Batch 7 Contract & Reference-Math Harness Suite (`#62`–`#73`)"
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
        "- Physical DGX Spark / Apple Mac Studio / RDMA / H100 CUDA 13.0 / 12 GB VRAM / live server execution: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references (`default_enabled: false`). Physical accelerators, multi-host 10GbE/RDMA clusters, live model serves, and unlocated/unverified upstream artifacts remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch7-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73],"
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

    Ok(Batch7Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
