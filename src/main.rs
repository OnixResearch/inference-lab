//! Imperative shell: CLI parsing, deterministic artifact serialization, fresh-directory
//! publication with empty `COMPLETE` marker written last.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use inference_lab::{
    adaptive_spec, batch2, deepseek_spark, diffusion, drafting, kernel_opt, kv_routing,
    learned_compiler, lifecycle, power, proof_completion, related_work, seed, serving_bench, stage,
    structured_read, tokenizer, trace,
};

const HELP: &str = "\
Inference Lab: bounded inference contract and measurement harnesses

Usage:
  inference-lab <command> --out NEW_DIRECTORY
  inference-lab --help

Bootstrap & Batch 1 commands:
  benchmark         Bootstrap serving/KV-budget contract seed (serving-contract-v1)
  diffusion         Issue #1 bounded diffusion proposal contract, EBM selection & tree interchange
  trace             Issue #40 host & GPU trace correlation qualification + negative controls
  stage             Issue #41 versioned transformer stage contract & 4-layer split reference
  power             Issue #43 power & throttle cadence qualification + protocol controls
  tokenizer         Issue #52 BPE & chat-template conformance + disqualifying parity checker
  related-work      Issues #56, #58, #60 pinned default-off related-work gate receipts

Batch 2 commands (#2-#11):
  proof-completion  Issue #2 masked proof completion through Compiler Lab's frozen gate
  drafting          Issue #3 diffusion drafting conformance & exact rejection-sampling parity
  kernel-opt        Issue #4 reference-assisted vs spec-only kernel optimization gate ordering
  kv-routing        Issue #5 session KV routing, bounded soft affinity & RDMA compatibility
  adaptive-spec     Issue #6 adaptive speculative graph & concurrency characterization
  serving-bench     Issue #7 portable serving benchmark extraction, branch-trace & RLM profiles
  deepseek-spark    Issue #8 DeepSeek/Spark migration, repaired counters & semantic quality checks
  learned-compiler  Issue #9 learned-compiler recovery inventory (blocked_on_artifacts) & controls
  structured-read   Issue #10 conditional shared structured-read gate (blocked_on_artifacts)
  lifecycle         Issue #11 centralized model-lifecycle protocol & Spark adapter controls
  batch2-suite      Run all Batch 2 harnesses (#2-#11) with SHA-256 manifest.json & COMPLETE last

Rules:
  The output directory must not exist; its parent directory must exist.
  Writes inputs.json, results.json, summary.md, and finally an empty COMPLETE marker.
  All harnesses run as deterministic bounded_contract_harness or blocked_on_artifacts checks;
  no physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, FPGA, or live serve runs locally.
";

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Help,
    Benchmark(PathBuf),
    Diffusion(PathBuf),
    Trace(PathBuf),
    Stage(PathBuf),
    Power(PathBuf),
    Tokenizer(PathBuf),
    RelatedWork(PathBuf),
    ProofCompletion(PathBuf),
    Drafting(PathBuf),
    KernelOpt(PathBuf),
    KvRouting(PathBuf),
    AdaptiveSpec(PathBuf),
    ServingBench(PathBuf),
    DeepSeekSpark(PathBuf),
    LearnedCompiler(PathBuf),
    StructuredRead(PathBuf),
    Lifecycle(PathBuf),
    Batch2Suite(PathBuf),
}

fn parse(args: &[String]) -> Result<Command, String> {
    match args {
        [arg] if arg == "--help" || arg == "-h" => Ok(Command::Help),
        [command, flag, path] if flag == "--out" && !path.is_empty() && !path.starts_with('-') => {
            let out = PathBuf::from(path);
            match command.as_str() {
                "benchmark" => Ok(Command::Benchmark(out)),
                "diffusion" => Ok(Command::Diffusion(out)),
                "trace" => Ok(Command::Trace(out)),
                "stage" => Ok(Command::Stage(out)),
                "power" => Ok(Command::Power(out)),
                "tokenizer" => Ok(Command::Tokenizer(out)),
                "related-work" => Ok(Command::RelatedWork(out)),
                "proof-completion" => Ok(Command::ProofCompletion(out)),
                "drafting" => Ok(Command::Drafting(out)),
                "kernel-opt" => Ok(Command::KernelOpt(out)),
                "kv-routing" => Ok(Command::KvRouting(out)),
                "adaptive-spec" => Ok(Command::AdaptiveSpec(out)),
                "serving-bench" => Ok(Command::ServingBench(out)),
                "deepseek-spark" => Ok(Command::DeepSeekSpark(out)),
                "learned-compiler" => Ok(Command::LearnedCompiler(out)),
                "structured-read" => Ok(Command::StructuredRead(out)),
                "lifecycle" => Ok(Command::Lifecycle(out)),
                "batch2-suite" => Ok(Command::Batch2Suite(out)),
                other => Err(format!("unknown command `{other}`")),
            }
        }
        _ => Err("expected `<command> --out NEW_DIRECTORY` or `--help`".into()),
    }
}

fn write_new_file(path: &Path, content: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
    file.write_all(content)
        .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
    Ok(())
}

fn publish(path: &Path, inputs: &str, results: &str, summary: &str) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for (name, content) in [
        ("inputs.json", inputs),
        ("results.json", results),
        ("summary.md", summary),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn publish_batch2_suite(path: &Path, bundle: &batch2::Batch2Bundle) -> Result<(), String> {
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create new output directory {}: {error}",
            path.display()
        )
    })?;
    for suite in &bundle.suites {
        publish(
            &path.join(suite.subdir),
            &suite.inputs_json,
            &suite.results_json,
            &suite.summary_md,
        )?;
    }
    for (name, content) in [
        ("inputs.json", bundle.aggregate_inputs_json.as_str()),
        ("results.json", bundle.aggregate_results_json.as_str()),
        ("summary.md", bundle.aggregate_summary_md.as_str()),
        ("manifest.json", bundle.manifest_json.as_str()),
    ] {
        write_new_file(&path.join(name), content.as_bytes())?;
    }
    write_new_file(&path.join("COMPLETE"), &[])?;
    Ok(())
}

fn run(command: Command) -> Result<String, String> {
    match command {
        Command::Help => Ok(HELP.into()),
        Command::Benchmark(path) => {
            let scenario = seed::default_scenario();
            let report = seed::evaluate(&scenario)?;
            let inputs = seed::inputs_json(&scenario);
            let results = seed::results_json(&report);
            let summary = seed::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Diffusion(path) => {
            let report = diffusion::run_default_harness()?;
            let inputs = diffusion::inputs_json(&report);
            let results = diffusion::results_json(&report);
            let summary = diffusion::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Trace(path) => {
            let report = trace::run_default_harness();
            let inputs = trace::inputs_json();
            let results = trace::results_json(&report);
            let summary = trace::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Stage(path) => {
            let report = stage::run_default_harness()?;
            let inputs = stage::inputs_json(&report);
            let results = stage::results_json(&report);
            let summary = stage::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Power(path) => {
            let report = power::run_default_harness();
            let inputs = power::inputs_json(&report);
            let results = power::results_json(&report);
            let summary = power::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Tokenizer(path) => {
            let report = tokenizer::run_default_harness();
            let inputs = tokenizer::inputs_json(&report);
            let results = tokenizer::results_json(&report);
            let summary = tokenizer::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::RelatedWork(path) => {
            let report = related_work::run_default_harness();
            let inputs = related_work::inputs_json(&report);
            let results = related_work::results_json(&report);
            let summary = related_work::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ProofCompletion(path) => {
            let report = proof_completion::run_default_harness();
            let inputs = proof_completion::inputs_json(&report);
            let results = proof_completion::results_json(&report);
            let summary = proof_completion::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Drafting(path) => {
            let report = drafting::run_default_harness()?;
            let inputs = drafting::inputs_json(&report);
            let results = drafting::results_json(&report);
            let summary = drafting::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KernelOpt(path) => {
            let report = kernel_opt::run_default_harness()?;
            let inputs = kernel_opt::inputs_json(&report);
            let results = kernel_opt::results_json(&report);
            let summary = kernel_opt::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::KvRouting(path) => {
            let report = kv_routing::run_default_harness();
            let inputs = kv_routing::inputs_json(&report);
            let results = kv_routing::results_json(&report);
            let summary = kv_routing::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::AdaptiveSpec(path) => {
            let report = adaptive_spec::run_default_harness()?;
            let inputs = adaptive_spec::inputs_json(&report);
            let results = adaptive_spec::results_json(&report);
            let summary = adaptive_spec::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::ServingBench(path) => {
            let report = serving_bench::run_default_harness();
            let inputs = serving_bench::inputs_json(&report);
            let results = serving_bench::results_json(&report);
            let summary = serving_bench::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::DeepSeekSpark(path) => {
            let report = deepseek_spark::run_default_harness();
            let inputs = deepseek_spark::inputs_json(&report);
            let results = deepseek_spark::results_json(&report);
            let summary = deepseek_spark::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::LearnedCompiler(path) => {
            let report = learned_compiler::run_default_harness();
            let inputs = learned_compiler::inputs_json(&report);
            let results = learned_compiler::results_json(&report);
            let summary = learned_compiler::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::StructuredRead(path) => {
            let report = structured_read::run_default_harness();
            let inputs = structured_read::inputs_json(&report);
            let results = structured_read::results_json(&report);
            let summary = structured_read::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Lifecycle(path) => {
            let report = lifecycle::run_default_harness();
            let inputs = lifecycle::inputs_json(&report);
            let results = lifecycle::results_json(&report);
            let summary = lifecycle::summary_markdown(&report);
            publish(&path, &inputs, &results, &summary)?;
            Ok(summary)
        }
        Command::Batch2Suite(path) => {
            let bundle = batch2::run_all()?;
            publish_batch2_suite(&path, &bundle)?;
            Ok(bundle.aggregate_summary_md)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = match parse(&args) {
        Ok(cmd) => cmd,
        Err(error) => {
            eprintln!("{error}\n\n{HELP}");
            return ExitCode::from(2);
        }
    };
    match run(command) {
        Ok(output) => {
            print!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("inference-lab: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_all_valid_commands_and_rejects_bad_args() {
        assert_eq!(parse(&["--help".into()]), Ok(Command::Help));
        for cmd in [
            "benchmark",
            "diffusion",
            "trace",
            "stage",
            "power",
            "tokenizer",
            "related-work",
            "proof-completion",
            "drafting",
            "kernel-opt",
            "kv-routing",
            "adaptive-spec",
            "serving-bench",
            "deepseek-spark",
            "learned-compiler",
            "structured-read",
            "lifecycle",
            "batch2-suite",
        ] {
            assert!(parse(&[cmd.into(), "--out".into(), "out-dir".into()]).is_ok());
        }
        for bad in [
            vec!["unknown", "--out", "dir"],
            vec!["benchmark", "--out", "--flag"],
            vec!["benchmark", "--out", "dir", "extra"],
            vec!["benchmark"],
        ] {
            let owned: Vec<String> = bad.into_iter().map(String::from).collect();
            assert!(parse(&owned).is_err());
        }
    }
}
