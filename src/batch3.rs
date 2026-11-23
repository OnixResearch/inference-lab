//! Batch 3 aggregate orchestrator covering Issues #12–#21 (`batch3-suite`).
//!
//! Pure deterministic builder producing all 10 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    deepseek_ref, djev_protocol, ggml_ops, graphical_sampling, mla_comparison, qwen38_quant,
    rsi_bench, rwkv_ggml, rwkv_leaf, sparse_moe,
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
pub struct Batch3Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch3Bundle, String> {
    let r12 = rsi_bench::run_default_harness()?;
    let r13 = djev_protocol::run_default_harness()?;
    let r14 = rwkv_leaf::run_default_harness()?;
    let r15 = ggml_ops::run_default_harness()?;
    let r16 = rwkv_ggml::run_default_harness()?;
    let r17 = graphical_sampling::run_default_harness()?;
    let r18 = qwen38_quant::run_default_harness()?;
    let r19 = deepseek_ref::run_default_harness()?;
    let r20 = mla_comparison::run_default_harness()?;
    let r21 = sparse_moe::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 12,
            command: "rsi-bench",
            subdir: "issue-12-rsi-bench",
            execution_class: r12.execution_class,
            hardware_or_artifact_gate: r12.hardware_gate.status,
            inputs_json: rsi_bench::inputs_json(&r12),
            results_json: rsi_bench::results_json(&r12),
            summary_md: rsi_bench::summary_markdown(&r12),
        },
        SuiteArtifacts {
            issue: 13,
            command: "djev-protocol",
            subdir: "issue-13-djev-protocol",
            execution_class: r13.execution_class,
            hardware_or_artifact_gate: r13.hardware_gate.status,
            inputs_json: djev_protocol::inputs_json(&r13),
            results_json: djev_protocol::results_json(&r13),
            summary_md: djev_protocol::summary_markdown(&r13),
        },
        SuiteArtifacts {
            issue: 14,
            command: "rwkv-leaf",
            subdir: "issue-14-rwkv-leaf",
            execution_class: r14.execution_class,
            hardware_or_artifact_gate: r14.hardware_gate.status,
            inputs_json: rwkv_leaf::inputs_json(&r14),
            results_json: rwkv_leaf::results_json(&r14),
            summary_md: rwkv_leaf::summary_markdown(&r14),
        },
        SuiteArtifacts {
            issue: 15,
            command: "ggml-ops",
            subdir: "issue-15-ggml-ops",
            execution_class: r15.execution_class,
            hardware_or_artifact_gate: r15.hardware_gate.status,
            inputs_json: ggml_ops::inputs_json(&r15),
            results_json: ggml_ops::results_json(&r15),
            summary_md: ggml_ops::summary_markdown(&r15),
        },
        SuiteArtifacts {
            issue: 16,
            command: "rwkv-ggml",
            subdir: "issue-16-rwkv-ggml",
            execution_class: r16.execution_class,
            hardware_or_artifact_gate: r16.hardware_gate.status,
            inputs_json: rwkv_ggml::inputs_json(&r16),
            results_json: rwkv_ggml::results_json(&r16),
            summary_md: rwkv_ggml::summary_markdown(&r16),
        },
        SuiteArtifacts {
            issue: 17,
            command: "graphical-sampling",
            subdir: "issue-17-graphical-sampling",
            execution_class: r17.execution_class,
            hardware_or_artifact_gate: r17.hardware_gate.status,
            inputs_json: graphical_sampling::inputs_json(&r17),
            results_json: graphical_sampling::results_json(&r17),
            summary_md: graphical_sampling::summary_markdown(&r17),
        },
        SuiteArtifacts {
            issue: 18,
            command: "qwen38-quant",
            subdir: "issue-18-qwen38-quant",
            execution_class: r18.execution_class,
            hardware_or_artifact_gate: r18.hardware_gate.status,
            inputs_json: qwen38_quant::inputs_json(&r18),
            results_json: qwen38_quant::results_json(&r18),
            summary_md: qwen38_quant::summary_markdown(&r18),
        },
        SuiteArtifacts {
            issue: 19,
            command: "deepseek-ref",
            subdir: "issue-19-deepseek-ref",
            execution_class: r19.execution_class,
            hardware_or_artifact_gate: r19.hardware_gate.status,
            inputs_json: deepseek_ref::inputs_json(&r19),
            results_json: deepseek_ref::results_json(&r19),
            summary_md: deepseek_ref::summary_markdown(&r19),
        },
        SuiteArtifacts {
            issue: 20,
            command: "mla-comparison",
            subdir: "issue-20-mla-comparison",
            execution_class: r20.execution_class,
            hardware_or_artifact_gate: r20.hardware_gate.status,
            inputs_json: mla_comparison::inputs_json(&r20),
            results_json: mla_comparison::results_json(&r20),
            summary_md: mla_comparison::summary_markdown(&r20),
        },
        SuiteArtifacts {
            issue: 21,
            command: "sparse-moe",
            subdir: "issue-21-sparse-moe",
            execution_class: r21.execution_class,
            hardware_or_artifact_gate: r21.hardware_gate.status,
            inputs_json: sparse_moe::inputs_json(&r21),
            results_json: sparse_moe::results_json(&r21),
            summary_md: sparse_moe::summary_markdown(&r21),
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
            "  \"schema\": \"inference-lab.batch3-suite.inputs.v1\",\n",
            "  \"issues\": [12,13,14,15,16,17,18,19,20,21],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch3-suite.results.v1\",\n",
            "  \"issues\": [12,13,14,15,16,17,18,19,20,21],\n",
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
        "# Batch 3 Contract & Reference-Math Harness Suite (`#12`–`#21`)"
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
        "- Physical GPU / CUDA / ggml binary / Tenstorrent P150x2 / JAX / Qwen3.8-27B live serve executed: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications, reference math, or explicit `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch3-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],"
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

    Ok(Batch3Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
