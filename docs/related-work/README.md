# Pinned related-work notes

This directory records pinned external papers, posts, and repositories tracked by Inference Lab issues where upstream code, checkpoints, licenses, or specialized hardware are unavailable or gated.

## Rules

1. Every note is **default-off**: it never alters default lab serving, training, routing, or acceptance rules.
2. Every upstream number is logged strictly as an **unverified author claim**, never as a lab result or acceptance gate.
3. Empty repositories or HTML-only project pages are never treated as runnable implementation pins.
4. Undeclared Hugging Face or GitHub licenses are recorded as explicit gates rather than assumed to inherit a base model's license.
5. Each note has a matching machine-readable gate contract emitted by `inference-lab related-work --out <dir>` and retained in [evidence/batch1/issue-56-58-60-related-work/](../../evidence/batch1/issue-56-58-60-related-work/summary.md).

## Index

- [Issue #56 — Ankit Gupta's claimed bit-exact FPGA Maple inference path](issue-56-fpga-maple.md) (`blocked_on_artifacts`, `aws_f2_fpga_unavailable`)
- [Issue #58 — Zhang, Chakraborty, and Khattab's compositional-harness results and pinned RLM-27B](issue-58-compositional-harness-rlm27b.md) (`blocked_on_license_and_hardware_gate`)
- [Issue #60 — Duggal et al. Latent Lattice Language Models (LLLMs)](issue-60-latent-lattice-lm.md) (`blocked_on_artifacts`)
