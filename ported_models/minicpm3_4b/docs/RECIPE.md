# MiniCPM3-4B Porting Recipe

## Overview

Adds `openbmb/MiniCPM3-4B` (4B-parameter causal LM) to the `llama.cpp-et`
framework. Confirmed via local GGUF metadata inspection:
`general.architecture = minicpm3` -- distinct from `minicpm` (already claimed
elsewhere on this board, MiniCPM5-1B in this campaign's earlier batch),
llama.cpp registers MiniCPM3 as its own execution family (different
attention mechanism: MiniCPM3 uses a compressed KV / latent-attention
scheme related to DeepSeek-V2's MLA, unlike MiniCPM/MiniCPM5's plain GQA).

## Model Reference

- **Source**: `openbmb/MiniCPM3-4B` (Hugging Face), revision
  `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`
- **License**: Apache-2.0
- **GGUF source**: `QuantFactory/MiniCPM3-4B-GGUF` (official repo ships only
  fp16/q4_k_m; used this third-party Q8_0 mirror instead), file
  `MiniCPM3-4B.Q8_0.gguf`
- **Quantization**: Q8_0,
  `sha256=a540961f131d6ab9616b6cb1c266be86735afe84907a445ad68e1e3e9422d122`
  (verified locally against the downloaded file)
- **Architecture**: `arch = minicpm3` per GGUF metadata, 748 tensors (high
  tensor count consistent with MLA's extra compression/decompression
  projection matrices per layer).

## Verification performed this round

GGUF-metadata-level only (downloaded exact file, verified sha256, confirmed
`general.architecture` + tensor count) -- not a full ET sysemu load/offload
test this round.

## Why this port's ET-SoC1 kernel support is an open question

MiniCPM3's MLA-style attention needs low-rank KV compression (down-projecting
K/V into a small latent space, then up-projecting per-head at attention time)
-- this is still expressible as `GGML_OP_MUL_MAT` chains (no new op type
needed in principle), but it has NOT been checked against the real ET sysemu
backend this round. Flagging as unconfirmed rather than assuming it works
like a plain-GQA model would.

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/minicpm3_4b.json`
  is the model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- MLA attention path not live-verified against ET sysemu.
