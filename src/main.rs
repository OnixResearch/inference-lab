//! Imperative shell: CLI parsing, deterministic artifact serialization, fresh-directory
//! publication with empty `COMPLETE` marker written last.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use inference_lab::{diffusion, power, related_work, seed, stage, tokenizer, trace};

const HELP: &str = "\
Inference Lab: bounded inference contract and measurement harnesses

Usage:
  inference-lab <command> --out NEW_DIRECTORY
  inference-lab --help

Commands:
  benchmark     Bootstrap serving/KV-budget contract seed (serving-contract-v1)
  diffusion     Issue #1 bounded diffusion proposal contract, EBM selection & tree interchange
  trace         Issue #40 host & GPU trace correlation qualification + negative controls
  stage         Issue #41 versioned transformer stage contract & 4-layer split reference
  power         Issue #43 power & throttle cadence qualification + protocol controls
  tokenizer     Issue #52 BPE & chat-template conformance + disqualifying parity checker
  related-work  Issues #56, #58, #60 pinned default-off related-work gate receipts

Rules:
  The output directory must not exist; its parent directory must exist.
  Writes inputs.json, results.json, summary.md, and finally an empty COMPLETE marker.
  All harnesses run as deterministic bounded_contract_harness or blocked_on_artifacts checks;
  no physical GPU, Apple M2 Ultra, Tenstorrent P150, TPU v7x, FPGA, or 27B serve runs locally.
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
                other => Err(format!("unknown command `{other}`")),
            }
        }
        _ => Err("expected `<command> --out NEW_DIRECTORY` or `--help`".into()),
    }
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
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.join(name))
            .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
        file.write_all(content.as_bytes())
            .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path.join("COMPLETE"))
        .map_err(|error| format!("incomplete output at {}: {error}", path.display()))?;
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
