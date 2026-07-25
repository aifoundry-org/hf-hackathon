# BitNet-b1.58-2B-4T Porting Recipe

## Overview

Adds `microsoft/bitnet-b1.58-2B-4T` (2B-parameter, native 1.58-bit ternary
weight causal LM -- Microsoft's official BitNet release) to the
`llama.cpp-et` framework. Confirmed via a minimal custom GGUF KV parser
(the installed `gguf` PyPI package's `GGMLQuantizationType` enum doesn't
yet recognize this file's tensor dtype -- raw type id 36 -- so its
`GGUFReader` crashes before finishing construction; wrote a small
struct-based reader that only walks the KV section and never touches
tensor dtype, since that's all we need here): `general.architecture =
bitnet-b1.58`. A completely distinct execution family from any other
decoder on this board -- the ONLY ternary-weight model attempted in this
campaign.

## Model Reference

- **Source**: `microsoft/bitnet-b1.58-2B-4T` (Hugging Face), revision
  `04c3b9ad9361b824064a1f25ea60a8be9599b127`
- **License**: MIT
- **GGUF source**: `microsoft/bitnet-b1.58-2B-4T-gguf` (official), file
  `ggml-model-i2_s.gguf`
- **Quantization**: native `i2_s` (2-bit packed ternary -- NOT a
  post-training quantization of a higher-precision model; this is the
  model's actual trained weight representation),
  `sha256=4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162`
  (verified locally against the downloaded file)
- **Architecture**: `arch = bitnet-b1.58` per GGUF metadata, 332 tensors.

## Verification performed this round

GGUF-metadata-level only, and even that needed a workaround (see the custom
parser note above) -- no full ET sysemu load/offload test this round.

## Why this port's ET-SoC1 kernel support is a genuinely open, not-yet-answered question

This is the one model in this whole campaign where "no new kernel work
needed" is NOT a safe assumption. Ternary weight matmul (`i2_s`) is a
fundamentally different compute pattern from every quantization scheme
already proven on the ET backend (Q8_0, Q4_K, etc. all dequantize blocks
of *scaled* values; BitNet's ternary weights are `{-1, 0, +1}` with a
single per-tensor scale, enabling addition/subtraction-only matmul with no
multiplication at all in the ideal case). Whether `ggml-et.cpp`'s
`GGML_OP_MUL_MAT` implementation handles the `i2_s` block format at all is
UNKNOWN -- not checked this round. This is flagged as the least-certain
port in the entire campaign; treat the claim as "GGUF loads and declares
this architecture," not "runs on the board."

## Open items for maintainer review

- Not board-registered; `ported_models/submissions/model_ports/bitnet_2b.json`
  is the model-ports track claim, pending identity approval.
- No changes to any protected file or the vendored submodule.
- **Real, unresolved risk**: `i2_s` ternary matmul may not be supported by
  the ET backend's `MUL_MAT` implementation at all. This needs an actual
  sysemu/board load to know either way -- not assumed positive or negative.
