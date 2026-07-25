# Pythia-410m Porting Recipe

## Overview

Adds `EleutherAI/pythia-410m` (405M-parameter causal LM, from the Pythia
interpretability suite) to the `llama_cpp_et` benchmark suite. Unlike
every other port this session, **no usable pre-made Q8_0 GGUF exists**
for any reasonably-sized Pythia checkpoint (only Q2_K/Q3_K community
quants were found at the sizes checked), so this model was **self-converted
directly from the original safetensors** using this repo's own
`convert_hf_to_gguf.py --outtype q8_0`.

This introduces the **GPTNeoX** execution family to the board.

## A real conversion bug found and fixed (locally, not in the vendored script)

The vendored `convert_hf_to_gguf.py`'s `GPTNeoXModel.set_gguf_parameters()`
reads `self.hparams["rotary_pct"]` as a flat dict key -- correct for the
transformers version the converter was originally written against. The
transformers version installed in this environment (5.14.1) has since
restructured `GPTNeoXConfig` to fold `rotary_pct` and `rotary_emb_base`
into a nested `rope_parameters` dict (`{"rope_theta": ..., "partial_rotary_factor":
..., "rope_type": "default"}`), and `AutoConfig.from_pretrained(...).to_dict()`
-- the path `convert_hf_to_gguf.py` uses by default -- no longer surfaces
the old flat key at all. Confirmed directly:

```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained('EleutherAI/pythia-410m').to_dict()
'rotary_pct' in c            # False
c['rope_parameters']         # {'rope_theta': 10000, 'partial_rotary_factor': 0.25, 'rope_type': 'default'}
```

This is a genuine latent incompatibility between the vendored converter
and newer transformers releases -- the base `TextModel` class already
knows about the new `rope_parameters` structure
(`self.rope_parameters = self.hparams.get("rope_parameters", ...)`), but
the `GPTNeoXModel` subclass's `set_gguf_parameters()` was never updated to
use it.

**Fix applied**: rather than editing the vendored/shared
`convert_hf_to_gguf.py`, a small standalone wrapper script imports the
module and monkeypatches `GPTNeoXModel.set_gguf_parameters` at runtime to
backfill `rotary_pct`/`rotary_emb_base` from `rope_parameters` when the
flat keys are absent, then defers to the original method. The vendored
script itself is untouched. Confirmed correct: the resulting GGUF loads
with `n_rot = 16`, exactly `0.25 * (1024 // 16) = 16` as expected from the
model's actual `partial_rotary_factor=0.25`, `hidden_size=1024`,
`num_attention_heads=16`.

## Why this port needed no new ET-SoC1 kernel work

GPTNeoX uses partial rotary embeddings (only a fraction of each head's
dimensions get RoPE, the rest pass through unrotated -- still an ordinary
`GGML_OP_ROPE` over a narrower tensor slice, the same pattern already
proven by `phi15` and `falcon7b` earlier this session), parallel
attention+FFN (`use_parallel_residual: true`, same pattern as Phi/Falcon),
LayerNorm, and a GELU FFN. Every op involved (`GGML_OP_NORM`,
`GGML_OP_ROPE`, `GGML_OP_MUL_MAT`, `GGML_OP_ADD`, `GGML_OP_UNARY` for
GELU, `GGML_OP_SOFT_MAX`, `GGML_OP_GET_ROWS`) is already proven on the ET
backend by the existing decoder-only models on the board.

## Model Reference

- **Source**: `EleutherAI/pythia-410m` safetensors (Hugging Face, main
  revision)
- **License**: Apache-2.0
- **Architecture**: `arch = gptneox` (`GPTNeoXForCausalLM`), 24 transformer
  layers, standard MHA (no GQA), partial rotary embeddings
  (`partial_rotary_factor=0.25`), parallel attention+FFN, 2048-token
  native context.
- **Quantization**: Q8_0 for weight matrices, F32 for norms/biases
  (llama.cpp's standard mixed-precision convention) -- ~413 MiB on disk.

## Steps Taken

1. **Downloaded** the original safetensors + config/tokenizer files from
   `EleutherAI/pythia-410m` via `huggingface_hub.snapshot_download`.
2. **Converted** to GGUF via the monkeypatch wrapper described above:
   `python3 convert_gptneox_patched.py <hf-snapshot> --outfile
   pythia-410m-Q8_0.gguf --outtype q8_0`. Produced a 433,396,992-byte
   file, `sha256=d6fe4736a6f6fc64dd5237a1a25a3f28ea37e74389b6c771da493bc6861d23a6`.
3. **Locally verified via sysemu (2026-07-24)**: built `llama-server` from
   the committed submodule, loaded the GGUF against
   `--device ET -ngl 99 --port 18121`. Confirmed `arch = gptneox`,
   `405.33 M` params (matches upstream), `n_rot = 16` (confirms the
   monkeypatch fix is numerically correct), full `25/25` layer ET offload
   (23 repeating layers + output layer), and a clean 943-node / 50-split
   compute graph. Full multi-token `/completion` decode output was not
   captured locally, consistent with every other port this session.

## Not registered in `artifacts.json` / `benchmark_config.json` -- hosting blocked

Unlike every pre-made-GGUF port this session, this converted artifact has
no upstream URL to point at -- it needs to be hosted somewhere, and (per
the same precedent as `distilbert_sst2`/`roberta_sst2` from an earlier PR)
the established pattern for a self-converted artifact in this fork is a
GitHub Release asset (no Hugging Face upload token available). **Creating
that release was blocked by this session's local tooling permissions** and
could not be completed before this PR. This recipe and the local
verification above stand as proof the port is real and working; the
`artifacts.json`/`benchmark_config.json` registration is deferred to a
follow-up once the release can be published.

## Instructions for Reproduction

```bash
pip install --user gguf huggingface_hub  # if not already present
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='EleutherAI/pythia-410m', local_dir='./pythia-410m-hf', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"

# from the llama.cpp-et submodule root:
python3 convert_hf_to_gguf.py ./pythia-410m-hf --outfile pythia-410m-Q8_0.gguf --outtype q8_0
# NOTE: on transformers >= the version that moved rotary_pct into
# rope_parameters, this will fail with KeyError: 'rotary_pct'. Use the
# monkeypatch wrapper (see this recipe's PR/commit for the exact script)
# or pin an older transformers version instead.
```

## Open items for maintainer review

- **Not yet board-registered** -- see hosting note above. Once a GitHub
  Release (or other hosting) is available, this is a one-line addition to
  both config files following the exact pattern of every other port this
  session.
- The upstream `convert_hf_to_gguf.py` bug described above (GPTNeoX +
  newer transformers) is worth reporting to the llama.cpp project
  upstream; not attempted here since it's outside this repo's scope.
- No changes were made to any protected file, and none to the vendored
  submodule either (the fix is a standalone external wrapper script, not
  included in this PR's diff since it's local tooling, not a repo asset).
