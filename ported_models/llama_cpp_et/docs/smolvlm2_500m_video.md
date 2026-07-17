# SmolVLM2 500M video baseline

This port runs the pinned `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
model through the committed `llama.cpp-et` runtime. It is a functional
baseline, not a fully optimized vision implementation.

## Model identity

- Hugging Face model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- Revision: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
- Baseline artifacts: Q8_0 model and Q8_0 multimodal projector
- Runtime revision: `27e3f108b759a6e4513ff73cb0f122ac50457b80`
- Full hashes and public fixtures: `.github/ci/reference/smolvlm2_500m_video.json`

The canonical GGUF files are pinned from
`ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`. They can be reproduced from the
original Hugging Face repository with the converter in the committed runtime:

```bash
python3 -m pip install huggingface_hub
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir local-artifacts/hf/SmolVLM2-500M-Video-Instruct

python3 ported_models/llama_cpp_et/src/llama.cpp-et/convert_hf_to_gguf.py \
  local-artifacts/hf/SmolVLM2-500M-Video-Instruct \
  --outfile local-artifacts/models/smolvlm2_500m_video/SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --outtype q8_0

python3 ported_models/llama_cpp_et/src/llama.cpp-et/convert_hf_to_gguf.py \
  local-artifacts/hf/SmolVLM2-500M-Video-Instruct \
  --outfile local-artifacts/models/smolvlm2_500m_video/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf \
  --outtype q8_0 --mmproj
```

## Correctness and quality

The repository keeps 20 pinned frames from five public
`TIGER-Lab/VideoFeedback` clips plus two fixed COCO images for local and
release validation. The merge gate deliberately uses one fixed four-frame dog
case so the board portion stays near two minutes. Its answer must match the
fixed accepted-answer set.

The log must also report the vision encoder on ET and all language-model layers
offloaded. It validates the GGUF architecture, all 291 language tensors, all
198 vision-projector tensors, and the exact model and projector hashes. The ET
backend implements the vision graph's 2D `IM2COL`, LayerNorm (`NORM`), and GELU
(`UNARY`) operations. CI rejects the run if the CLIP graph reports any CPU
fallback operation.

Only the candidate runs the 512-token WikiText-2 check on ET-SoC1 and through
the current-main CPU binary. The first valid baseline measured `22.2825` PPL,
so CI rejects PPL above `26.7390` (20% regression); ET PPL must also remain
within 1% of CPU. This is a language-side corruption guard. The fixed visual
answer, tensor inventory, exact artifact hashes, full offload, and zero
fallback checks cover the multimodal path.

## Performance

The leaderboard uses firmware cycles from the ET runtime's
`device_cmd_exec_dur` counter. Each sample starts a fresh server, runs the
four-frame dog case from raw JPEG input through the deterministic answer, and
sums all kernel-response cycles whose monotonic timestamps fall inside that
request. The same request supplies the correctness evidence, and one sample is
the score; lower is better. Paired current-main runs bound board noise.

Wall-clock real-time factor remains a diagnostic. Downloads, build time, model
loading, and the CPU reference pass are excluded from both request metrics.

For a PR, trusted CI runs current main, the candidate, and current main again.
The candidate must improve the mean paired-main cycle count by at least 1%.
If the two main runs drift by more than 0.5%, CI reports an infrastructure error
and produces no candidate verdict. Request wall time must corroborate the PMC
result with at least a 0.25% improvement; paired-main wall drift may not exceed
1%.

## Submitting

The hackathon track freezes the Q8_0 model and projector. Participants may
change only regular files under
`ggml/src/ggml-et/et-kernels/src/` in their public `llama.cpp-et` fork. Server,
model loading, artifacts, CPU reference, profiler, prompts, scorer, and
workflow remain main-owned. The trusted runner builds the participant commit,
runs it on ET-SoC1, and brackets it with current main.

The candidate build directory is persistent and incrementally rebuilt, so a
warmed gate runs the two short main brackets plus the candidate correctness,
PMC, and PPL checks in about two minutes. A missing runner cache adds a one-time
native build warm-up.

Run it on an ET-SoC1 board host with:

```bash
BENCHMARK_DEVICE=soc1sim BOARD_BENCHMARK=1 \
  .github/ci/scripts/run_model_benchmark.sh smolvlm2_500m_video
```
