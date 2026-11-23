//! Batch 5 aggregate orchestrator covering Issues #32–#39, #42, #44–#46 (`batch5-suite`).
//!
//! Pure deterministic builder producing all 12 per-issue artifact bundles, aggregate top-level
//! `inputs.json`/`results.json`/`summary.md`, and exact SHA-256 `manifest.json`.

use std::fmt::Write as FmtWrite;

use crate::json::{quote, sha256_hex};
use crate::provenance::ExecutionClass;
use crate::{
    biollm_protocol, compact_hybrids, cpu_contention, inkling_residuals, intervention_replay,
    native_recurrent_bench, nemotron_spark, phoenix_nano, qsa_state_coherence, quark_eval,
    split_m2_p150, tiered_eval,
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
pub struct Batch5Bundle {
    pub suites: Vec<SuiteArtifacts>,
    pub aggregate_inputs_json: String,
    pub aggregate_results_json: String,
    pub aggregate_summary_md: String,
    pub manifest_json: String,
}

pub fn run_all() -> Result<Batch5Bundle, String> {
    let r32 = qsa_state_coherence::run_default_harness()?;
    let r33 = inkling_residuals::run_default_harness()?;
    let r34 = phoenix_nano::run_default_harness()?;
    let r35 = nemotron_spark::run_default_harness()?;
    let r36 = compact_hybrids::run_default_harness()?;
    let r37 = native_recurrent_bench::run_default_harness()?;
    let r38 = tiered_eval::run_default_harness()?;
    let r39 = cpu_contention::run_default_harness()?;
    let r42 = split_m2_p150::run_default_harness()?;
    let r44 = quark_eval::run_default_harness()?;
    let r45 = intervention_replay::run_default_harness()?;
    let r46 = biollm_protocol::run_default_harness()?;

    let suites = vec![
        SuiteArtifacts {
            issue: 32,
            command: "qsa-state-coherence",
            subdir: "issue-32-qsa-state-coherence",
            execution_class: r32.execution_class,
            hardware_or_artifact_gate: r32.hardware_gate.status,
            inputs_json: qsa_state_coherence::inputs_json(&r32),
            results_json: qsa_state_coherence::results_json(&r32),
            summary_md: qsa_state_coherence::summary_markdown(&r32),
        },
        SuiteArtifacts {
            issue: 33,
            command: "inkling-residuals",
            subdir: "issue-33-inkling-residuals",
            execution_class: r33.execution_class,
            hardware_or_artifact_gate: r33.hardware_gate.status,
            inputs_json: inkling_residuals::inputs_json(&r33),
            results_json: inkling_residuals::results_json(&r33),
            summary_md: inkling_residuals::summary_markdown(&r33),
        },
        SuiteArtifacts {
            issue: 34,
            command: "phoenix-nano",
            subdir: "issue-34-phoenix-nano",
            execution_class: r34.execution_class,
            hardware_or_artifact_gate: r34.hardware_gate.status,
            inputs_json: phoenix_nano::inputs_json(&r34),
            results_json: phoenix_nano::results_json(&r34),
            summary_md: phoenix_nano::summary_markdown(&r34),
        },
        SuiteArtifacts {
            issue: 35,
            command: "nemotron-spark",
            subdir: "issue-35-nemotron-spark",
            execution_class: r35.execution_class,
            hardware_or_artifact_gate: r35.hardware_gate.status,
            inputs_json: nemotron_spark::inputs_json(&r35),
            results_json: nemotron_spark::results_json(&r35),
            summary_md: nemotron_spark::summary_markdown(&r35),
        },
        SuiteArtifacts {
            issue: 36,
            command: "compact-hybrids",
            subdir: "issue-36-compact-hybrids",
            execution_class: r36.execution_class,
            hardware_or_artifact_gate: r36.hardware_gate.status,
            inputs_json: compact_hybrids::inputs_json(&r36),
            results_json: compact_hybrids::results_json(&r36),
            summary_md: compact_hybrids::summary_markdown(&r36),
        },
        SuiteArtifacts {
            issue: 37,
            command: "native-recurrent-bench",
            subdir: "issue-37-native-recurrent-bench",
            execution_class: r37.execution_class,
            hardware_or_artifact_gate: r37.hardware_gate.status,
            inputs_json: native_recurrent_bench::inputs_json(&r37),
            results_json: native_recurrent_bench::results_json(&r37),
            summary_md: native_recurrent_bench::summary_markdown(&r37),
        },
        SuiteArtifacts {
            issue: 38,
            command: "tiered-eval",
            subdir: "issue-38-tiered-eval",
            execution_class: r38.execution_class,
            hardware_or_artifact_gate: r38.hardware_gate.status,
            inputs_json: tiered_eval::inputs_json(&r38),
            results_json: tiered_eval::results_json(&r38),
            summary_md: tiered_eval::summary_markdown(&r38),
        },
        SuiteArtifacts {
            issue: 39,
            command: "cpu-contention",
            subdir: "issue-39-cpu-contention",
            execution_class: r39.execution_class,
            hardware_or_artifact_gate: r39.hardware_gate.status,
            inputs_json: cpu_contention::inputs_json(&r39),
            results_json: cpu_contention::results_json(&r39),
            summary_md: cpu_contention::summary_markdown(&r39),
        },
        SuiteArtifacts {
            issue: 42,
            command: "split-m2-p150",
            subdir: "issue-42-split-m2-p150",
            execution_class: r42.execution_class,
            hardware_or_artifact_gate: r42.hardware_gate.status,
            inputs_json: split_m2_p150::inputs_json(&r42),
            results_json: split_m2_p150::results_json(&r42),
            summary_md: split_m2_p150::summary_markdown(&r42),
        },
        SuiteArtifacts {
            issue: 44,
            command: "quark-eval",
            subdir: "issue-44-quark-eval",
            execution_class: r44.execution_class,
            hardware_or_artifact_gate: r44.hardware_gate.status,
            inputs_json: quark_eval::inputs_json(&r44),
            results_json: quark_eval::results_json(&r44),
            summary_md: quark_eval::summary_markdown(&r44),
        },
        SuiteArtifacts {
            issue: 45,
            command: "intervention-replay",
            subdir: "issue-45-intervention-replay",
            execution_class: r45.execution_class,
            hardware_or_artifact_gate: r45.hardware_gate.status,
            inputs_json: intervention_replay::inputs_json(&r45),
            results_json: intervention_replay::results_json(&r45),
            summary_md: intervention_replay::summary_markdown(&r45),
        },
        SuiteArtifacts {
            issue: 46,
            command: "biollm-protocol",
            subdir: "issue-46-biollm-protocol",
            execution_class: r46.execution_class,
            hardware_or_artifact_gate: r46.hardware_gate.status,
            inputs_json: biollm_protocol::inputs_json(&r46),
            results_json: biollm_protocol::results_json(&r46),
            summary_md: biollm_protocol::summary_markdown(&r46),
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
            "  \"schema\": \"inference-lab.batch5-suite.inputs.v1\",\n",
            "  \"issues\": [32,33,34,35,36,37,38,39,42,44,45,46],\n",
            "  \"execution_class\": \"bounded_contract_harness\",\n",
            "  \"suites\": [{}]\n",
            "}}\n"
        ),
        suite_meta.join(",")
    );

    let aggregate_results_json = format!(
        concat!(
            "{{\n",
            "  \"schema\": \"inference-lab.batch5-suite.results.v1\",\n",
            "  \"issues\": [32,33,34,35,36,37,38,39,42,44,45,46],\n",
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
        "# Batch 5 Contract & Reference-Math Harness Suite (`#32`–`#39`, `#42`, `#44`–`#46`)"
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
        "- Physical GPU / M2 Ultra / P150 / Nemotron / Phoenix / Quark / BioLLM live execution: `false`"
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
        "> All outputs are deterministic pure-Rust CPU contract verifications and numerical references. Physical accelerators, live model serves, and missing checkpoints/recordings remain explicit `unavailable_hardware_runtime` / `blocked_on_artifacts` gates. No external upstream claims are imported as acceptance gates."
    )
    .unwrap();

    let mut manifest_json = String::new();
    writeln!(manifest_json, "{{").unwrap();
    writeln!(
        manifest_json,
        "  \"schema\": \"inference-lab.batch5-evidence.v1\","
    )
    .unwrap();
    writeln!(
        manifest_json,
        "  \"issues\": [32, 33, 34, 35, 36, 37, 38, 39, 42, 44, 45, 46],"
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

    Ok(Batch5Bundle {
        suites,
        aggregate_inputs_json,
        aggregate_results_json,
        aggregate_summary_md,
        manifest_json,
    })
}
