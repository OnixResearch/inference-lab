//! Batch 2 aggregate orchestrator covering Issues #2–#11 (`batch2-suite`).
//!
//! Pure deterministic builder producing all 10 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    adaptive_spec, deepseek_spark, drafting, kernel_opt, kv_routing, learned_compiler, lifecycle,
    proof_completion, serving_bench, structured_read,
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
pub struct Batch2Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch2Bundle, String> {
    let r2 = proof_completion::run_default_harness();
    let r3 = drafting::run_default_harness()?;
    let r4 = kernel_opt::run_default_harness()?;
    let r5 = kv_routing::run_default_harness();
    let r6 = adaptive_spec::run_default_harness()?;
    let r7 = serving_bench::run_default_harness();
    let r8 = deepseek_spark::run_default_harness();
    let r9 = learned_compiler::run_default_harness();
    let r10 = structured_read::run_default_harness();
    let r11 = lifecycle::run_default_harness();

    let suites = vec![
        SuiteArtifacts {
            issue: 2,
            command: "proof-completion",
            subdir: "issue-02-proof-completion",
            execution_class: r2.execution_class,
            hardware_or_artifact_gate: r2.hardware_gate.status,
            inputs_json: proof_completion::inputs_json(&r2),
            results_json: proof_completion::results_json(&r2),
            summary_md: proof_completion::summary_markdown(&r2),
        },
        SuiteArtifacts {
            issue: 3,
            command: "drafting",
            subdir: "issue-03-drafting",
            execution_class: r3.execution_class,
            hardware_or_artifact_gate: r3.hardware_gate.status,
            inputs_json: drafting::inputs_json(&r3),
            results_json: drafting::results_json(&r3),
            summary_md: drafting::summary_markdown(&r3),
        },
        SuiteArtifacts {
            issue: 4,
            command: "kernel-opt",
            subdir: "issue-04-kernel-opt",
            execution_class: r4.execution_class,
            hardware_or_artifact_gate: r4.hardware_gate.status,
            inputs_json: kernel_opt::inputs_json(&r4),
            results_json: kernel_opt::results_json(&r4),
            summary_md: kernel_opt::summary_markdown(&r4),
        },
        SuiteArtifacts {
            issue: 5,
            command: "kv-routing",
            subdir: "issue-05-kv-routing",
            execution_class: r5.execution_class,
            hardware_or_artifact_gate: r5.hardware_gate.status,
            inputs_json: kv_routing::inputs_json(&r5),
            results_json: kv_routing::results_json(&r5),
            summary_md: kv_routing::summary_markdown(&r5),
        },
        SuiteArtifacts {
            issue: 6,
            command: "adaptive-spec",
            subdir: "issue-06-adaptive-spec",
            execution_class: r6.execution_class,
            hardware_or_artifact_gate: r6.hardware_gate.status,
            inputs_json: adaptive_spec::inputs_json(&r6),
            results_json: adaptive_spec::results_json(&r6),
            summary_md: adaptive_spec::summary_markdown(&r6),
        },
        SuiteArtifacts {
            issue: 7,
            command: "serving-bench",
            subdir: "issue-07-serving-bench",
            execution_class: r7.execution_class,
            hardware_or_artifact_gate: r7.hardware_gate.status,
            inputs_json: serving_bench::inputs_json(&r7),
            results_json: serving_bench::results_json(&r7),
            summary_md: serving_bench::summary_markdown(&r7),
        },
        SuiteArtifacts {
            issue: 8,
            command: "deepseek-spark",
            subdir: "issue-08-deepseek-spark",
            execution_class: r8.execution_class,
            hardware_or_artifact_gate: r8.hardware_gate.status,
            inputs_json: deepseek_spark::inputs_json(&r8),
            results_json: deepseek_spark::results_json(&r8),
            summary_md: deepseek_spark::summary_markdown(&r8),
        },
        SuiteArtifacts {
            issue: 9,
            command: "learned-compiler",
            subdir: "issue-09-learned-compiler",
            execution_class: r9.execution_class,
            hardware_or_artifact_gate: r9.hardware_gate.status,
            inputs_json: learned_compiler::inputs_json(&r9),
            results_json: learned_compiler::results_json(&r9),
            summary_md: learned_compiler::summary_markdown(&r9),
        },
        SuiteArtifacts {
            issue: 10,
            command: "structured-read",
            subdir: "issue-10-structured-read",
            execution_class: r10.execution_class,
            hardware_or_artifact_gate: r10.hardware_gate.status,
            inputs_json: structured_read::inputs_json(&r10),
            results_json: structured_read::results_json(&r10),
            summary_md: structured_read::summary_markdown(&r10),
        },
        SuiteArtifacts {
            issue: 11,
            command: "lifecycle",
            subdir: "issue-11-lifecycle",
            execution_class: r11.execution_class,
            hardware_or_artifact_gate: r11.hardware_gate.status,
            inputs_json: lifecycle::inputs_json(&r11),
            results_json: lifecycle::results_json(&r11),
            summary_md: lifecycle::summary_markdown(&r11),
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
            "  \"schema\": \"inference-lab.batch2-suite.inputs.v1\",\n",
            "  \"issues\": [2,3,4,5,6,7,8,9,10,11],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch2-suite.results.v1\",\n",
            "  \"issues\": [2,3,4,5,6,7,8,9,10,11],\n",
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
        "# Batch 2 Contract Harness Suite (`#2`–`#11`)"
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
        "- Physical GPU / Spark / Tenstorrent / RDMA / Darwin / live model executed: `false`"
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
        "> All outputs are deterministic CPU contract verifications or explicit `blocked_on_artifacts` inventories. No external claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch2-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],"
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

    Ok(Batch2Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
