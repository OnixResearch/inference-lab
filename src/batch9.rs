//! Batch 9 aggregate orchestrator covering Issues #85–#95 (`batch9-suite`).
//!
//! Pure deterministic builder producing all 11 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    alora_prefix_cache, anchor_label_drafter, knowledgator_sys1_extract, ling_fp8kv_alloc_cap,
    link_sparse_attention, loop_kv_compression, mit_qda_quant, moe_spec_roofline,
    parasite_feature_control, perseus_mpk_rdma, unlimited_ocr_longdoc,
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
pub struct Batch9Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch9Bundle, String> {
    let r85 = link_sparse_attention::run_default_harness()?;
    let r86 = mit_qda_quant::run_default_harness()?;
    let r87 = knowledgator_sys1_extract::run_default_harness()?;
    let r88 = unlimited_ocr_longdoc::run_default_harness()?;
    let r89 = loop_kv_compression::run_default_harness()?;
    let r90 = alora_prefix_cache::run_default_harness()?;
    let r91 = perseus_mpk_rdma::run_default_harness()?;
    let r92 = moe_spec_roofline::run_default_harness()?;
    let r93 = anchor_label_drafter::run_default_harness()?;
    let r94 = parasite_feature_control::run_default_harness()?;
    let r95 = ling_fp8kv_alloc_cap::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 85,
            command: "link-sparse-attention",
            subdir: "issue-85-link-sparse-attention",
            execution_class: r85.execution_class,
            hardware_or_artifact_gate: r85.hardware_gate.status,
            inputs_json: link_sparse_attention::inputs_json(&r85),
            results_json: link_sparse_attention::results_json(&r85),
            summary_md: link_sparse_attention::summary_markdown(&r85),
        },
        SuiteArtifacts {
            issue: 86,
            command: "mit-qda-quant",
            subdir: "issue-86-mit-qda-quant",
            execution_class: r86.execution_class,
            hardware_or_artifact_gate: r86.hardware_gate.status,
            inputs_json: mit_qda_quant::inputs_json(&r86),
            results_json: mit_qda_quant::results_json(&r86),
            summary_md: mit_qda_quant::summary_markdown(&r86),
        },
        SuiteArtifacts {
            issue: 87,
            command: "knowledgator-sys1-extract",
            subdir: "issue-87-knowledgator-sys1-extract",
            execution_class: r87.execution_class,
            hardware_or_artifact_gate: r87.hardware_gate.status,
            inputs_json: knowledgator_sys1_extract::inputs_json(&r87),
            results_json: knowledgator_sys1_extract::results_json(&r87),
            summary_md: knowledgator_sys1_extract::summary_markdown(&r87),
        },
        SuiteArtifacts {
            issue: 88,
            command: "unlimited-ocr-longdoc",
            subdir: "issue-88-unlimited-ocr-longdoc",
            execution_class: r88.execution_class,
            hardware_or_artifact_gate: r88.hardware_gate.status,
            inputs_json: unlimited_ocr_longdoc::inputs_json(&r88),
            results_json: unlimited_ocr_longdoc::results_json(&r88),
            summary_md: unlimited_ocr_longdoc::summary_markdown(&r88),
        },
        SuiteArtifacts {
            issue: 89,
            command: "loop-kv-compression",
            subdir: "issue-89-loop-kv-compression",
            execution_class: r89.execution_class,
            hardware_or_artifact_gate: r89.hardware_gate.status,
            inputs_json: loop_kv_compression::inputs_json(&r89),
            results_json: loop_kv_compression::results_json(&r89),
            summary_md: loop_kv_compression::summary_markdown(&r89),
        },
        SuiteArtifacts {
            issue: 90,
            command: "alora-prefix-cache",
            subdir: "issue-90-alora-prefix-cache",
            execution_class: r90.execution_class,
            hardware_or_artifact_gate: r90.hardware_gate.status,
            inputs_json: alora_prefix_cache::inputs_json(&r90),
            results_json: alora_prefix_cache::results_json(&r90),
            summary_md: alora_prefix_cache::summary_markdown(&r90),
        },
        SuiteArtifacts {
            issue: 91,
            command: "perseus-mpk-rdma",
            subdir: "issue-91-perseus-mpk-rdma",
            execution_class: r91.execution_class,
            hardware_or_artifact_gate: r91.hardware_gate.status,
            inputs_json: perseus_mpk_rdma::inputs_json(&r91),
            results_json: perseus_mpk_rdma::results_json(&r91),
            summary_md: perseus_mpk_rdma::summary_markdown(&r91),
        },
        SuiteArtifacts {
            issue: 92,
            command: "moe-spec-roofline",
            subdir: "issue-92-moe-spec-roofline",
            execution_class: r92.execution_class,
            hardware_or_artifact_gate: r92.hardware_gate.status,
            inputs_json: moe_spec_roofline::inputs_json(&r92),
            results_json: moe_spec_roofline::results_json(&r92),
            summary_md: moe_spec_roofline::summary_markdown(&r92),
        },
        SuiteArtifacts {
            issue: 93,
            command: "anchor-label-drafter",
            subdir: "issue-93-anchor-label-drafter",
            execution_class: r93.execution_class,
            hardware_or_artifact_gate: r93.hardware_gate.status,
            inputs_json: anchor_label_drafter::inputs_json(&r93),
            results_json: anchor_label_drafter::results_json(&r93),
            summary_md: anchor_label_drafter::summary_markdown(&r93),
        },
        SuiteArtifacts {
            issue: 94,
            command: "parasite-feature-control",
            subdir: "issue-94-parasite-feature-control",
            execution_class: r94.execution_class,
            hardware_or_artifact_gate: r94.hardware_gate.status,
            inputs_json: parasite_feature_control::inputs_json(&r94),
            results_json: parasite_feature_control::results_json(&r94),
            summary_md: parasite_feature_control::summary_markdown(&r94),
        },
        SuiteArtifacts {
            issue: 95,
            command: "ling-fp8kv-alloc-cap",
            subdir: "issue-95-ling-fp8kv-alloc-cap",
            execution_class: r95.execution_class,
            hardware_or_artifact_gate: r95.hardware_gate.status,
            inputs_json: ling_fp8kv_alloc_cap::inputs_json(&r95),
            results_json: ling_fp8kv_alloc_cap::results_json(&r95),
            summary_md: ling_fp8kv_alloc_cap::summary_markdown(&r95),
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
            "  \"schema\": \"inference-lab.batch9-suite.inputs.v1\",\n",
            "  \"issues\": [85,86,87,88,89,90,91,92,93,94,95],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch9-suite.results.v1\",\n",
            "  \"issues\": [85,86,87,88,89,90,91,92,93,94,95],\n",
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
        "# Batch 9 Contract & Reference-Math Harness Suite (`#85`–`#95`)"
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
        "- Physical GPU / DGX Spark GB10 / 2-node RDMA / vLLM+SGLang OCR / live model execution: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references (`default_enabled: false`). Physical accelerators, DGX Spark GB10 262K serves, 2-node RDMA clusters, live model serves, and unlocated public upstream artifacts remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch9-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],"
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

    Ok(Batch9Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
