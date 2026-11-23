# Issue #56 — Ankit Gupta's claimed bit-exact FPGA Maple inference path

- **Tracked issue**: [OnixResearch/inference-lab#56](https://github.com/OnixResearch/inference-lab/issues/56)
- **Status**: `blocked_on_artifacts` (and `unavailable_hardware_runtime`: no declared AWS F2 FPGA capacity)
- **Activation**: `default_off = true` (explicit opt-in only; does not alter default lab serving, routing, or backend selection)
- **Verification status**: **Unverified author claims only.** No FPGA image, CPU reference, or throughput run was executed here.

## Primary sources and observed pin state (20 Nov 2026)

| Artifact | Location | Observed state | Valid runnable pin? |
| --- | --- | --- | --- |
| Author write-up | `https://ankitg.me/blog/vibecoding-inference-chip/` (dated 20 Nov 2026) | Public blog post | Narrative only |
| Author placeholder repo | `https://github.com/ankitvgupta/fpga-inference` | Empty repository: **no commits, no default-branch tree, no license, no SHA** | **No — must not be treated as a pin** |
| FPGA RTL / C++ sources, bitstream (`r8`/`r9`), host runtime, pinned CPU reference, 8-chat × 3-turn workload | Not published | Unavailable | **No — active artifact gate** |

Public DeepGrove Maple model cards are **not** the FPGA design and are not a substitute artifact.

## Unverified author claims (non-gating)

The following figures come exclusively from the 20 Nov 2026 blog post and **must not** be imported as lab acceptance gates or targets:

- Initial agent-authored FPGA image (`R8`): claimed `6.43 tokens/s` on AWS F2 FPGA.
- 14-step optimization sequence to `r9`: claimed `213.13 tokens/s` (`33.13x` over `R8`), using a host-CPU cut-down Maple draft model with FPGA exact verification.
- Claimed 15 Nov 2026 8-chat × 3-turn AWS run: `24/24` turns bit-exact against an unpublished pinned CPU reference (while GPU-offload and T4 CUDA context forks did not match bit-exactly).
- Claimed 8,192-token long-context run: `r9` faster at short context, roughly tied with `r8` across the full window; author does not recommend the design for production.

## Gate conditions before any future run

1. Public release under an explicit license of FPGA sources, bitstream/image identity, host software, pinned CPU reference, and the 8-chat × 3-turn (or declared substitute) workload — pinned by exact repository SHAs and file SHA-256 digests.
2. Declared lab access to AWS F2 or equivalent FPGA hardware (purchasing F2 capacity or hardware is an explicit non-goal).
3. Any future executed arm must compare every turn/token (including mismatches) against the pinned CPU reference and keep short-context and long-context throughput separate with exact board/instance/clock provenance.

## Disambiguation and nearby owners

- **Companion hardware-compiler issue**: `brittonr/compiler-lab#270` owns any future default-off RTL/Verilog/C++ admission check if sources become public; inference serving parity stays here.
- **Distinct Inference Lab issues preserved**: `#15` (`ggml` backend-ops), `#21` (sparse MoE dispatch), `#22` (MTP draft-and-verify), `#27` (TensorRT Edge-LLM), `#41` (Apple–Tenstorrent stage contract).
- **Nearby open-source chip demo (related-work note only — does NOT satisfy this gate)**:
  - Yaroslav Bulatov (`@yaroslavvb`) posted a qualitative thermal-camera video demo of AI Foundry / AINekko's ET-SoC-1 (`https://x.com/yaroslavvb/status/2123564715259678871`, 19 Nov 2026).
  - Pinned references for disambiguation only: `aifoundry-org/et-platform@836a4ab600e93c3059bb58c898edbc37744cd8d0` (Apache-2.0) and `yaroslavvb/et-soc1-prototyping@a517aa7def8fe7cc1d4027ac8cbd696adc52ae4a` (21 Nov 2026).
  - Do **not** add an ET-SoC-1 or thermal-camera arm to `#56`; this note only prevents mistaking that demo for Ankit Gupta's unpublished Maple FPGA artifacts.
