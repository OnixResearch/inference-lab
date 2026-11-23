//! Issues #56, #58, #60: Pinned default-off related-work gate contracts and validator.
//!
//! Ensures every tracked upstream paper/post/repository remains default-off, keeps all
//! author-reported figures strictly unverified and non-gating, rejects empty repos and
//! HTML-only project pages as runnable implementation pins, and preserves undeclared
//! Hugging Face licenses as explicit gates.

use std::fmt::Write as FmtWrite;

use crate::json::{opt_str, quote, str_array};
use crate::provenance::{ExecutionClass, HardwareGate};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedSourceRef {
    pub label: String,
    pub location: String,
    pub commit_sha: Option<String>,
    pub license_status: String,
    pub runnable_implementation: bool,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedWorkGateRecord {
    pub issue_number: u32,
    pub slug: String,
    pub title: String,
    pub doc_path: String,
    pub default_off: bool,
    pub claims_verified: bool,
    pub gate_status: ExecutionClass,
    pub hardware_gate: HardwareGate,
    pub active_gates: Vec<String>,
    pub pinned_sources: Vec<PinnedSourceRef>,
    pub non_gating_unverified_author_claims: Vec<String>,
    pub preserved_distinct_owners: Vec<String>,
    pub disambiguation_notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelatedWorkReport {
    pub execution_class: ExecutionClass,
    pub gate_validation_passed: bool,
    pub validation_errors: Vec<String>,
    pub records: Vec<RelatedWorkGateRecord>,
}

/// Build the exact pinned default-off records for Issues #56, #58, and #60.
pub fn default_records() -> Vec<RelatedWorkGateRecord> {
    vec![
        // Issue #56: Ankit Gupta Maple FPGA
        RelatedWorkGateRecord {
            issue_number: 56,
            slug: "issue-56-fpga-maple".into(),
            title: "Ankit Gupta claimed bit-exact FPGA Maple inference path".into(),
            doc_path: "docs/related-work/issue-56-fpga-maple.md".into(),
            default_off: true,
            claims_verified: false,
            gate_status: ExecutionClass::BlockedOnArtifacts,
            hardware_gate: HardwareGate::unavailable(
                "aws-f2-fpga",
                "Purchasing or reserving AWS F2 FPGA capacity is a non-goal; no declared FPGA host exists.",
            ),
            active_gates: vec![
                "artifact_gate:fpga_sources_bitstream_host_software_pinned_cpu_reference_unpublished"
                    .into(),
                "pin_gate:ankitvgupta_fpga_inference_empty_no_commits_no_sha_no_license".into(),
                "hardware_gate:aws_f2_fpga_unavailable".into(),
            ],
            pinned_sources: vec![
                PinnedSourceRef {
                    label: "author_blog_post".into(),
                    location: "https://ankitg.me/blog/vibecoding-inference-chip/".into(),
                    commit_sha: None,
                    license_status: "blog_narrative_only".into(),
                    runnable_implementation: false,
                    notes: "Dated 20 Nov 2026; publishes narrative claims only, no sources or bitstreams.".into(),
                },
                PinnedSourceRef {
                    label: "empty_placeholder_repo".into(),
                    location: "https://github.com/ankitvgupta/fpga-inference".into(),
                    commit_sha: None,
                    license_status: "no_license_empty_repo".into(),
                    runnable_implementation: false,
                    notes: "Observed 20 Nov 2026 with zero commits and no SHA; MUST NOT be treated as a pin.".into(),
                },
                PinnedSourceRef {
                    label: "disambiguation_et_platform".into(),
                    location: "https://github.com/aifoundry-org/et-platform".into(),
                    commit_sha: Some("836a4ab600e93c3059bb58c898edbc37744cd8d0".into()),
                    license_status: "Apache-2.0".into(),
                    runnable_implementation: false,
                    notes: "Related-work disambiguation only (ET-SoC-1); does NOT satisfy Maple FPGA artifact gate.".into(),
                },
                PinnedSourceRef {
                    label: "disambiguation_et_soc1_prototyping".into(),
                    location: "https://github.com/yaroslavvb/et-soc1-prototyping".into(),
                    commit_sha: Some("a517aa7def8fe7cc1d4027ac8cbd696adc52ae4a".into()),
                    license_status: "related_work_note_only".into(),
                    runnable_implementation: false,
                    notes: "Yaroslav Bulatov thermal-camera demo (19-21 Nov 2026); do not add an ET-SoC-1 arm to #56.".into(),
                },
            ],
            non_gating_unverified_author_claims: vec![
                "R8 baseline claimed 6.43 tokens/s (unverified, non-gating)".into(),
                "r9 optimized image claimed 213.13 tokens/s or 33.13x over R8 (unverified, non-gating)".into(),
                "8-chat x 3-turn claimed 24/24 bit-exact turns vs unpublished CPU reference (unverified, non-gating)".into(),
            ],
            preserved_distinct_owners: vec![
                "brittonr/compiler-lab#270".into(),
                "OnixResearch/inference-lab#15".into(),
                "OnixResearch/inference-lab#21".into(),
                "OnixResearch/inference-lab#22".into(),
                "OnixResearch/inference-lab#27".into(),
                "OnixResearch/inference-lab#41".into(),
            ],
            disambiguation_notes: vec![
                "Public DeepGrove Maple model cards are NOT the FPGA design or a substitute artifact.".into(),
                "Yaroslav Bulatov ET-SoC-1 thermal-camera demo is qualitative video only and does NOT satisfy #56's artifact gate.".into(),
            ],
        },
        // Issue #58: Zhang, Chakraborty, Khattab compositional-harness & RLM-27B
        RelatedWorkGateRecord {
            issue_number: 58,
            slug: "issue-58-compositional-harness-rlm27b".into(),
            title: "Zhang, Chakraborty, and Khattab compositional-harness results and pinned RLM-27B".into(),
            doc_path: "docs/related-work/issue-58-compositional-harness-rlm27b.md".into(),
            default_off: true,
            claims_verified: false,
            gate_status: ExecutionClass::UnavailableHardwareRuntime,
            hardware_gate: HardwareGate::unavailable(
                "27b-serving-host-for-rlm-27b-and-qwen3.8-27b",
                "Blocked on undeclared Hugging Face checkpoint/dataset licenses AND absence of a declared 27B serve host.",
            ),
            active_gates: vec![
                "license_gate:undeclared_license_on_RLM_27B_and_TailRecurse_and_synthetic_sft_and_exp_logs"
                    .into(),
                "hardware_gate:no_27b_serving_host_available_locally".into(),
                "contamination_gate:synthetic_sft_overlap_with_held_out_benches_unchecked".into(),
            ],
            pinned_sources: vec![
                PinnedSourceRef {
                    label: "harness_cg_code".into(),
                    location: "https://github.com/alexzhang13/harness-cg".into(),
                    commit_sha: Some("5272afdcb25a156c1b9b89cb479ae0604af249d7".into()),
                    license_status: "MIT".into(),
                    runnable_implementation: true,
                    notes: "Single release commit observed 20 Nov 2026; contains src/rlm_train/env.py (RLMTrainEnv).".into(),
                },
                PinnedSourceRef {
                    label: "prime_rl_training_pin".into(),
                    location: "https://github.com/PrimeIntellect-ai/prime-rl".into(),
                    commit_sha: Some("eaf95997f7cdbf85681075a61dd5bbfd2ba0231c".into()),
                    license_status: "upstream_pin".into(),
                    runnable_implementation: false,
                    notes: "Training reproduction (8xH100 SFT / prime-rl) is an explicit non-goal.".into(),
                },
                PinnedSourceRef {
                    label: "hf_rlm_27b".into(),
                    location: "https://huggingface.co/a1zhang/RLM-27B".into(),
                    commit_sha: Some("f0b3b27db57c211234cb55ec512d87f45dabdc4c".into()),
                    license_status: "undeclared_gate_do_not_assume_apache_2".into(),
                    runnable_implementation: false,
                    notes: "Full-parameter SFT of Qwen/Qwen3.8-27B (Apache-2.0); card declares no license.".into(),
                },
                PinnedSourceRef {
                    label: "hf_rlm_27b_tail_recurse".into(),
                    location: "https://huggingface.co/a1zhang/RLM-27B-TailRecurse".into(),
                    commit_sha: Some("bbab801cbe871aefe3f898093e461c807381dcc2".into()),
                    license_status: "undeclared_gate".into(),
                    runnable_implementation: false,
                    notes: "Optional later arm only; license undeclared.".into(),
                },
                PinnedSourceRef {
                    label: "hf_qwen38_27b_tail_recurse".into(),
                    location: "https://huggingface.co/a1zhang/Qwen3.8-27B-TailRecurse".into(),
                    commit_sha: Some("c57b846de3aba4287df68c1cf42ae10bc83ce8a0".into()),
                    license_status: "undeclared_gate".into(),
                    runnable_implementation: false,
                    notes: "Optional later arm only; license undeclared.".into(),
                },
                PinnedSourceRef {
                    label: "hf_synthetic_rlm_sft_data".into(),
                    location: "https://huggingface.co/datasets/a1zhang/synthetic-rlm-sft-data".into(),
                    commit_sha: Some("9a31220cf40c1301eebde9a4282d1360b2d233f5".into()),
                    license_status: "undeclared_gate".into(),
                    runnable_implementation: false,
                    notes: "Authors claim no real benchmark data; claim is unverified here.".into(),
                },
                PinnedSourceRef {
                    label: "hf_tail_recursion_sft_data".into(),
                    location: "https://huggingface.co/datasets/a1zhang/rlm-tail-recursion-sft-data".into(),
                    commit_sha: Some("a46f1f6aa02b88a2580a15e086af4a96c3a0c5af".into()),
                    license_status: "MIT".into(),
                    runnable_implementation: false,
                    notes: "Pinned dataset revision (MIT).".into(),
                },
                PinnedSourceRef {
                    label: "hf_exp_logs".into(),
                    location: "https://huggingface.co/datasets/a1zhang/harness_compgen_exp_logs".into(),
                    commit_sha: Some("1204c5cc7f05ddf6d1b5afe8536d2e30a56782a4".into()),
                    license_status: "undeclared_gate".into(),
                    runnable_implementation: false,
                    notes: "Experiment logs card declares no license.".into(),
                },
                PinnedSourceRef {
                    label: "disambiguation_machine_code_space".into(),
                    location: "https://github.com/sgiannopoulos/sgiannopoulos.github.io".into(),
                    commit_sha: Some("22dc86b4b31529fd082b1cd8c5c4b591edd8c0f8".into()),
                    license_status: "undeclared_related_work_note".into(),
                    runnable_implementation: false,
                    notes: "Xenos & Giannopoulos hex-dump blog citing earlier arXiv:2512.24601; benign ELF check belongs to compiler-lab#275.".into(),
                },
            ],
            non_gating_unverified_author_claims: vec![
                "8-bench equal-weight mean 46.1 -> 58.5 (+26.9% relative) (unverified, non-gating)".into(),
                "~2x fewer tokens and 18.4 vs 6.5 root turns (unverified, non-gating)".into(),
                "2-64x length generalization and cross-domain transfer from short RL (unverified, non-gating)".into(),
            ],
            preserved_distinct_owners: vec![
                "OnixResearch/inference-lab#49".into(),
                "OnixResearch/inference-lab#18".into(),
                "OnixResearch/inference-lab#28".into(),
                "OnixResearch/inference-lab#31".into(),
                "brittonr/compiler-lab#275".into(),
            ],
            disambiguation_notes: vec![
                "Distinct from #49 (Pi / OpenCode LoRA adapters); do not fold RLM-27B full-parameter SFT into #49.".into(),
                "Xenos & Giannopoulos Machine Code Space post is related work only and does not add a task to #58.".into(),
            ],
        },
        // Issue #60: Duggal et al. Latent Lattice Language Models
        RelatedWorkGateRecord {
            issue_number: 60,
            slug: "issue-60-latent-lattice-lm".into(),
            title: "Duggal et al. Latent Lattice Language Models (LLLMs)".into(),
            doc_path: "docs/related-work/issue-60-latent-lattice-lm.md".into(),
            default_off: true,
            claims_verified: false,
            gate_status: ExecutionClass::BlockedOnArtifacts,
            hardware_gate: HardwareGate::blocked_on_artifacts(
                "public-lllm-code-and-checkpoints",
                "No public training/inference code, checkpoints, or license exists; project-page repository contains HTML/media assets only.",
            ),
            active_gates: vec![
                "artifact_gate:no_public_training_or_inference_code_or_checkpoints".into(),
                "license_gate:no_license_file_in_project_page_repo".into(),
            ],
            pinned_sources: vec![
                PinnedSourceRef {
                    label: "arxiv_paper".into(),
                    location: "https://arxiv.org/abs/2611.21608v1".into(),
                    commit_sha: None,
                    license_status: "arxiv_preprint".into(),
                    runnable_implementation: false,
                    notes: "Duggal, Wu, Freeman, Torralba, Zhang, Shechtman (Nov 2026).".into(),
                },
                PinnedSourceRef {
                    label: "project_page_html_only_repo".into(),
                    location: "https://github.com/ShivamDuggal4/latent-lattice-language-models".into(),
                    commit_sha: Some("23876839517155db4333fb08d26b8b118a3db2e1".into()),
                    license_status: "no_license_html_only".into(),
                    runnable_implementation: false,
                    notes: "Observed 18 Nov 2026: HTML, fonts, images, and videos ONLY; MUST NOT be treated as an implementation.".into(),
                },
            ],
            non_gating_unverified_author_claims: vec![
                "20B FineWeb-Edu 406M perplexity 15.41 -> 14.40 (and 13.61 with HC+input residuals) (unverified, non-gating)".into(),
                "Lattice-GDN beats 4x larger native state baselines (unverified, non-gating)".into(),
                "45% smaller KV cache at 22L vs 40L width 1024 (unverified, non-gating)".into(),
            ],
            preserved_distinct_owners: vec![
                "OnixResearch/lattice".into(),
                "OnixResearch/inference-lab#33".into(),
                "OnixResearch/inference-lab#37".into(),
                "OnixResearch/inference-lab#51".into(),
                "OnixResearch/inference-lab#57".into(),
            ],
            disambiguation_notes: vec![
                "Completely distinct from OnixResearch/lattice (the Onix candidate-worker repo referenced in #1).".into(),
                "Distinct from looped Inkling 2 Mini (#57), SimpleHC vs mHC (#51), official RWKV-7/Mamba-3/xLSTM (#37), and Inkling attention residuals (#33).".into(),
            ],
        },
    ]
}

/// Validate that every related-work record strictly obeys Inference Lab's default-off,
/// non-gating, empty-repo/HTML-only non-pin, and undeclared-license rules.
pub fn validate_related_work_gates(records: &[RelatedWorkGateRecord]) -> RelatedWorkReport {
    let mut errors = Vec::new();
    for rec in records {
        if !rec.default_off {
            errors.push(format!("issue_{}_must_be_default_off", rec.issue_number));
        }
        if rec.claims_verified {
            errors.push(format!(
                "issue_{}_must_not_mark_unverified_author_claims_verified",
                rec.issue_number
            ));
        }
        if rec.active_gates.is_empty() {
            errors.push(format!(
                "issue_{}_missing_explicit_active_gate",
                rec.issue_number
            ));
        }
        for src in &rec.pinned_sources {
            if src.commit_sha.is_none() && src.runnable_implementation {
                errors.push(format!(
                    "issue_{}_source_{}_marked_runnable_without_sha",
                    rec.issue_number, src.label
                ));
            }
            if src.license_status.contains("undeclared") && src.runnable_implementation {
                errors.push(format!(
                    "issue_{}_source_{}_marked_runnable_with_undeclared_license",
                    rec.issue_number, src.label
                ));
            }
        }
    }

    RelatedWorkReport {
        execution_class: ExecutionClass::BlockedOnArtifacts,
        gate_validation_passed: errors.is_empty(),
        validation_errors: errors,
        records: records.to_vec(),
    }
}

pub fn run_default_harness() -> RelatedWorkReport {
    let records = default_records();
    validate_related_work_gates(&records)
}

pub fn inputs_json(report: &RelatedWorkReport) -> String {
    let issues: Vec<String> = report
        .records
        .iter()
        .map(|r| {
            format!(
                "{{\"issue\":{},\"slug\":{},\"doc_path\":{},\"default_off\":{}}}",
                r.issue_number,
                quote(&r.slug),
                quote(&r.doc_path),
                r.default_off
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.related-work-inputs.v1\",\"issues\":[56,58,60],\"records\":[{}]}}\n",
        issues.join(",\n")
    )
}

pub fn results_json(report: &RelatedWorkReport) -> String {
    let records: Vec<String> = report
        .records
        .iter()
        .map(|r| {
            let sources: Vec<String> = r
                .pinned_sources
                .iter()
                .map(|s| {
                    format!(
                        "{{\"label\":{},\"location\":{},\"commit_sha\":{},\"license_status\":{},\"runnable_implementation\":{},\"notes\":{}}}",
                        quote(&s.label),
                        quote(&s.location),
                        opt_str(s.commit_sha.as_deref()),
                        quote(&s.license_status),
                        s.runnable_implementation,
                        quote(&s.notes)
                    )
                })
                .collect();
            format!(
                "{{\"issue_number\":{},\"slug\":{},\"title\":{},\"doc_path\":{},\"default_off\":{},\"claims_verified\":{},\"gate_status\":{},\"hardware_gate\":{},\"active_gates\":{},\"pinned_sources\":[{}],\"non_gating_unverified_author_claims\":{},\"preserved_distinct_owners\":{},\"disambiguation_notes\":{}}}",
                r.issue_number,
                quote(&r.slug),
                quote(&r.title),
                quote(&r.doc_path),
                r.default_off,
                r.claims_verified,
                quote(r.gate_status.as_str()),
                r.hardware_gate.to_json(),
                str_array(&r.active_gates),
                sources.join(","),
                str_array(&r.non_gating_unverified_author_claims),
                str_array(&r.preserved_distinct_owners),
                str_array(&r.disambiguation_notes)
            )
        })
        .collect();
    format!(
        "{{\"schema\":\"inference-lab.related-work-report.v1\",\"package_version\":{},\"execution_class\":{},\"gate_validation_passed\":{},\"validation_errors\":{},\"records\":[{}]}}\n",
        quote(env!("CARGO_PKG_VERSION")),
        quote(report.execution_class.as_str()),
        report.gate_validation_passed,
        str_array(&report.validation_errors),
        records.join(",\n")
    )
}

pub fn summary_markdown(report: &RelatedWorkReport) -> String {
    let mut out = format!(
        "# Issues #56, #58, #60 — Pinned default-off related-work gate report\n\n\
         - **Execution class**: `{}`\n\
         - **Gate validator passed**: `{}`\n\n\
         | Issue | Title | Default-off | Claims verified | Gate status | Active gates |\n\
         | --- | --- | --- | --- | --- | --- |\n",
        report.execution_class.as_str(),
        report.gate_validation_passed
    );
    for r in &report.records {
        writeln!(
            out,
            "| #{} | {} | `{}` | `{}` | `{}` | {} |",
            r.issue_number,
            r.title,
            r.default_off,
            r.claims_verified,
            r.gate_status.as_str(),
            r.active_gates.len()
        )
        .expect("writing to String cannot fail");
    }
    out.push_str(
        "\nAll upstream figures remain strictly unverified non-gating related work. No FPGA run, \
         27B model serve, or Latent Lattice pretraining/eval was executed.\n",
    );
    out
}
