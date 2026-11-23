# Issue #59: Default-Off `maya-ik2` Serve of Inkling 2 Mini on 24 GB `sm_86`

- Execution class: `bounded_contract_harness` (`synthetic_contract_fixture: true`, `default_enabled: false`, `distinct_from_issue_57: true`, `installs_gpu_unlock: false`)
- Hardware gate: `unavailable_hardware_runtime` (Linux host with 24 GB sm_86 GPU (RTX 3090 class), CUDA 12.8, and pinned Unsloth UD-Q4_K_XL GGUF)
- Exact 4-bit register unpack vs scalar reference: `max_abs_err = 5.239e-16` (`accepted`)
- FP16 `sm_86` register-unpack emulation: `max_abs_err = 1.130e-3`, `mean_full_vocab_kl = 3.806e-8`, `top1_agreement = Some(1.0)`

## Separated Unverified Non-Gating Author Notes
- X post (`https://x.com/superalesha/status/2124102269138346409`): `781 launches/tok`, `128.7 plain tok/s` — 129.5 HTTP tok/s vs 57.0 llama.cpp (7350 launches); 62.3 tok/s at 340K / 23.57 GiB; 74/96 GSM8K vs 73/96 scalar; 5.7e-5 KL (unverified author claim)
- Pinned README (`alesha-pro/maya-inkling-mini@c520fd14de3b21d67ddc6b7e7348b761aad26d52/README.md`): `757 launches/tok`, `128 plain tok/s` — 128 plain tok/s, 150-210 tok/s with Thinking Machines DSpark drafter; 74 plain / 77 drafter GSM8K (unverified author claim)

## Negative Controls (8 / 8 rejected)
- `nc1_windows_os_unsupported` -> `unsupported` (maya-ik2 requires Linux (host win-rtx3090 reports OS "windows"))
- `nc2_non_sm86_arch_unsupported` -> `unsupported` (maya-ik2 requires sm_86 GPU (host linux-rtx4090-sm89 reports "sm_89"))
- `nc3_vram_below_24gb_unsupported` -> `unsupported` (maya-ik2 requires >=24 GB VRAM (host linux-rtx3080-12gb has 12 GB))
- `nc4_cuda_not_12_8_unsupported` -> `unsupported` (maya-ik2 requires CUDA 12.8 (host linux-rtx3090-cuda124 has 12.4))
- `nc5_non_ud_q4_k_xl_gguf_unsupported` -> `unsupported` (maya-ik2 reads only Unsloth UD-Q4_K_XL GGUF (got "Q8_0"))
- `nc6_swapped_nibble_register_unpack_caught` -> `invalid` (swapped nibble order caused 1.102e0 logit divergence)
- `nc7_corrupted_block_scale_kl_spike` -> `invalid` (corrupted block scale raised full-vocab KL to 9.879e-1)
- `nc8_non_finite_activation_rejected` -> `invalid` (NaN activation cleanly rejected in FP16 register-unpack path)
