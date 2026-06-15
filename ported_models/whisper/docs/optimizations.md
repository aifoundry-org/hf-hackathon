# Whisper Optimization Notes

This guide summarizes the Whisper Tiny EN optimization and fidelity work done so far. It covers two related but different tracks:

- Silicon performance: making the resident ET-SoC1 encoder/decoder faster while preserving exact token match against the current host raw-INT8 audit.
- Model fidelity: debugging why the resident/raw-INT8 model worked on JFK but collapsed to EOS on a harder true-30s Hawking clip.

All silicon validation referenced here was done on `<board-host>` with remote artifacts under `$REMOTE_ROOT`. Large local artifacts are under `<artifact-archive>`.

## Current Best Silicon Run

Best useful full-silicon run:

`<artifact-archive>/whisper-real/phase1/full_silicon/run_20260506-154659_new23_ah16`

Result:

- PASS
- Scope: log-mel features -> encoder cross-cache -> decoder token IDs on ET-SoC1
- Host ONNXRuntime used as audit only
- Token IDs match exactly
- Encoder wait: 46.7682 s
- Decoder wait: 8.9941 s
- Total kernel wait: 55.7623 s
- Total generated token/s: 0.412465
- Decoder-only generated token/s: 2.55723
- Cross K max abs vs host FP16 oracle: 0.0009765625
- Cross V max abs vs host FP16 oracle: 0.001953125

Best true-30s WAV run that still produced useful text:

`<artifact-archive>/whisper-real/phase1/full_silicon/run_20260506-161031_new64_ah16`

That audio is 30.0 s JFK speech followed by silence. It is a true 30 s WAV, but not continuous speech.

- PASS
- Encoder wait: 46.77 s
- Decoder wait: 8.99572 s
- Total kernel wait: 55.76572 s
- Total generated token/s: 0.41244
- Effective audio throughput: about 1.86 seconds of compute per second of audio

The continuous 30 s Hawking clip is not a useful silicon result yet: silicon matched the resident raw-INT8 host audit, but that raw-INT8 audit emitted EOS immediately. That proves implementation consistency, not Whisper transcription quality.

## Silicon Optimization Progression

Baseline full resident run:

`local-artifacts/erbium_amp_probe/whisper-real/phase1/full_silicon/run_20260506-090836_new23_ah16`

- Encoder: 148.186 s
- Decoder: 79.7042 s
- Total: 227.8902 s
- Total token/s: 0.100926

Current best:

- Encoder: 46.7682 s
- Decoder: 8.9941 s
- Total: 55.7623 s
- Total token/s: 0.412465

Net improvement:

- Encoder: 3.17x faster
- Decoder: 8.86x faster
- End to end: 4.09x faster

| Optimization | Result | Good/bad | Why |
| --- | ---: | --- | --- |
| Parallel vocab projection / argmax | Decoder 79.7042 s -> 39.6631 s | Very good | The final vocab scan is large and naturally splits across harts. |
| Parallel decoder cross-attention heads | Decoder 39.6631 s -> 28.2317 s | Good | Split head work gave clear speedup. |
| Parallel decoder MLP | Decoder 28.2317 s -> 17.8129 s | Very good | FC1/GELU/FC2 output blocks distribute cleanly. |
| Encoder row-major INT8 matvec | Encoder 149.25 s -> 116.334 s | Good | Better dense-weight access pattern. |
| Encoder attention V locality | Encoder 116.334 s -> 63.875 s | Very good | Contiguous per-token V reads fixed a major locality issue. |
| Encoder cross-cache locality | Encoder 63.875 s -> 60.812 s | Small good | Cross-cache K/V accumulation became more cache-friendly. |
| Position-major cross-K layout | Decoder 17.8159 s -> 17.019 s | Small good | Improved decoder cross-attention locality. |
| Encoder attention batch-2 | Encoder 60.3043 s -> 52.1997 s | Good | Reuses K/V row loads for two rows. |
| Encoder dense batch-2 | Encoder 52.1997 s -> 46.7745 s | Good | Reuses INT8 weight rows for two output rows. |
| Decoder argmax block-32 | Decoder 17.0159 s -> 16.2045 s | Small good | Better contiguous vocab block scan. |
| Parallel decoder self-attention | Decoder 16.2045 s -> 10.9756 s | Very good | Split QKV, heads, and output projection. |
| Parallel decoder cross query/output projections | Decoder 10.9756 s -> 8.9941 s | Good | Current best decoder path. |

## Rejected Silicon Optimizations

| Optimization | Result | Verdict |
| --- | --- | --- |
| 8 active harts instead of 16 | Encoder 127.638 s for one-token audit; decoder 21.4249 s for 23-token decoder-only | Bad. 16 harts are clearly better for the current kernels. |
| 12 active harts decoder-only | Decoder 18.9416 s | Bad. Better than 8, worse than 16. |
| Encoder dense batch-4 | Encoder regressed to 68.5173 s for one-token audit | Bad. More reuse did not offset register/cache pressure. |
| Encoder conv1 time-major | Encoder 46.9623 s versus about 46.78 s baseline | Neutral/bad. Correct but slightly slower. |
| Decoder argmax block-64 | Decoder 8.99539 s versus 8.99471 s for block-32 | Neutral/bad. Correct but not faster. |

## Correctness Bugs Fixed During Optimization

| Symptom | Cause | Fix |
| --- | --- | --- |
| One-token decoder predicted EOS instead of token 843 | Shared `context` was stale across harts | Evict/invalidate shared context before cross-attention harts write it. |
| Longer decode diverged around token 7 | Dirty cache-line sharing in per-hart score buffers | Align per-hart score banks to 1536 floats. |

These bugs matter because many performance changes looked plausible but only counted if the final token stream still matched the host audit exactly.

## PMC Takeaways

The PMCs were useful for ranking hot regions, but wall-clock kernel wait is still the authoritative metric. A few HPM deltas can wrap or include runtime noise.

Final encoder stage summary from the best run:

| Stage | Records | Notes |
| --- | ---: | --- |
| MLP | 4 | Still one of the main compute-heavy regions. |
| Attention | 4 | Similar scale to MLP; locality fixes helped a lot. |
| QKV | 4 | Still significant; HPM6 can be noisy/wrapped depending on run. |
| Cross-cache | 1 | Important for memory traffic and decoder handoff. |
| Conv1/Conv2 | 1 each | Not the main wall-time bottleneck after optimizations, but fidelity-sensitive in quantization. |

Final decoder stage summary:

| Stage | Records | Notes |
| --- | ---: | --- |
| Cross-attention | 104 | Largest decoder remaining stage. |
| Final argmax | 26 | Still expensive, but much better after parallelization/blocking. |
| MLP | 104 | Much smaller after parallel output blocking. |
| Self-attention | 104 | Much smaller after parallel QKV/head/output work. |
| Embed | 26 | Tiny in runtime, but very important for quantization fidelity. |

## Raw-INT8 Fidelity Problem

The resident raw-INT8 model behaved differently depending on input:

| Clip | FP32 ONNX | Dynamic-INT8 ONNX | Raw-INT8 resident-style ONNX | Silicon |
| --- | --- | --- | --- | --- |
| JFK 11 s padded / 30 s silence variant | Good JFK transcript | Exact match | Exact match | Exact match |
| Hawking continuous 30 s | Good transcript | Good transcript with minor wording changes | Immediate EOS / empty text | Matched EOS exactly |

The Hawking failure means the silicon executor was internally consistent, but the resident raw-INT8 model approximation was not faithful enough.

Raw-INT8 host executor comparison artifacts:

- Hawking fail: `<artifact-archive>/whisper-real/phase1/raw_int8_host_executor_compare/hawking/run_20260506-164322`
- JFK pass: `<artifact-archive>/whisper-real/phase1/raw_int8_host_executor_compare/jfk_30s/run_20260506-164339`

## ONNX Debug Matrix

Mixed encoder/decoder host ONNX runs on the Hawking clip:

| Encoder | Decoder | Result | Meaning |
| --- | --- | --- | --- |
| FP32 | FP32 | Good transcript | Baseline model is fine. |
| Dynamic INT8 | Dynamic INT8 | Good transcript | ORT-style dynamic quantization is viable. |
| Raw INT8 | Raw INT8 | EOS immediately | Current resident quantization is too lossy. |
| FP32 | Raw INT8 | Only "The last might not be difficult." | Raw decoder is a major problem. |
| Raw INT8 | FP32 | EOS immediately | Raw encoder can also push logits over the edge. |
| Dynamic INT8 | Raw INT8 | Only "The last might not be difficult." | Raw decoder remains brittle even with better encoder. |
| Raw INT8 | Dynamic INT8 | Mostly good transcript | Dynamic decoder is much more robust to raw encoder errors. |

Main conclusion: raw decoder quantization is the primary collapse trigger, but encoder quantization also degrades wording.

## Decoder Quantization Findings

The first broad sweep showed that quantizing every floating initializer is bad. It quantizes things ORT dynamic-INT8 does not quantize: token embeddings, positional embeddings, mask/constants, LayerNorm weights/biases, and every bias.

| Variant | Hawking result | Good/bad | Finding |
| --- | --- | --- | --- |
| Raw all float initializers | EOS immediately | Bad | Too aggressive. |
| Raw linear only | Non-empty transcript | Good-ish | Avoiding non-linear/embedding params prevents EOS. |
| Raw linear + bias | Non-empty transcript | Good-ish | Similar to linear only; still wording errors. |
| Keep LayerNorm FP32 only | EOS immediately | Bad | LayerNorm params were not the root collapse. |
| Keep all Gather initializers FP32 | Non-empty transcript | Good | Gather path is critical. |
| Keep token embedding FP32 only | Non-empty transcript | Very informative but not final | Token embedding is the critical Gather initializer. |
| Keep positional embedding FP32 only | Only tail phrase | Bad | Not sufficient. |
| Keep mask FP32 only | EOS immediately | Bad | Not sufficient. |

Token embedding quantization sweep:

| Token embedding scheme | Hawking result | Good/bad | Notes |
| --- | --- | --- | --- |
| Global per-tensor INT8 | EOS immediately | Bad | Original failure mode. |
| Per-token/per-row scale | Only tail phrase | Bad | Lower reconstruction error did not preserve decoding. |
| FP32 token embedding | Non-empty transcript | Quality good but memory bad | `token_embedding.weight` is about 79.7 MiB FP32, too large for the current 64 MiB resident region. |
| Per-dimension/per-column scale | Non-empty transcript | Best portable finding | Keeps INT8-sized token table plus only 384 scales. This is the best next resident implementation target. |

The per-dimension token embedding result is important: it gives most of the FP32-token benefit without blowing up resident memory.

## Encoder Quantization Findings

With decoder stabilized by per-dimension token embedding, Hawking still had wording errors. The main visible one was:

- FP32/dynamic expected: "far in advance"
- raw path often produced: "for an advance"

Encoder sensitivity results:

| Encoder variant | Hawking result | Good/bad | Finding |
| --- | --- | --- | --- |
| Raw all | Non-empty but wording error | Usable but degraded | No EOS, but quality loss remains. |
| MatMul only | Preserved "far in advance" better | Good | Transformer matmuls alone are less damaging than convs/biases. |
| MatMul + bias | Reintroduced "for an advance" | Bad-ish | Bias quantization is surprisingly sensitive. |
| Cross-cache projections only | Pretty good | Good | Cross-cache raw projections are tolerable. |
| QKV only / MLP only / attention-out only | Pretty good | Good | These individual transformer pieces are not the main phrase error. |
| Conv only | Produced "for an advance" | Bad | Front-end conv quantization is a fidelity-sensitive source. |
| Conv per-output-channel scale | Did not recover quality | Bad/neutral | Still worse than keeping conv FP32. |
| Conv FP32 with raw rest | Best tested host quality | Good but not silicon-ported | Costs extra memory but likely feasible compared with FP32 token embedding. |

The conv finding is counterintuitive: convs are not the biggest runtime bottleneck, but they are a quality-sensitive front-end. The first practical fidelity plan should keep convs higher precision or use a better conv-specific quantization scheme.

## What Looks Worth Porting Next

Priority 1: per-dimension token embedding scales.

- Keeps `token_embedding.weight` as INT8.
- Adds 384 FP32 scales.
- Fixes the EOS collapse on Hawking in host ONNX.
- Should be straightforward to port in the decoder embed step:
  `token_embedding[token, i] * token_embedding_dim_scale[i]`.

Priority 2: better encoder conv handling.

Options:

- Keep conv weights/biases FP32 in the resident region.
- Use a better calibrated conv quantization scheme.
- Try per-input-channel or mixed per-output/per-input scaling if we want to keep convs INT8.

Keeping convs FP32 is probably the fastest path to a more honest demo. The memory cost is much smaller than keeping token embedding FP32.

Priority 3: do not quantize small sensitive non-MatMul tensors blindly.

The raw-all approach was bad because it quantized everything. Dynamic-INT8 avoids this by mostly targeting MatMul weights. A resident model should have explicit per-tensor policy:

- Token embedding: INT8 with per-dimension scales.
- Positional embedding: current raw path seems tolerable, but keeping FP32 is cheap enough to consider.
- Mask/constants: keep exact.
- LayerNorm weight/bias: raw path was not the collapse trigger, but keeping FP32 is cheap.
- MatMul weights: INT8 remains broadly viable.
- Biases: treat carefully; encoder bias quantization affected wording.

## What Not To Claim Yet

Do not claim robust full Whisper Tiny EN equivalence yet.

What is fair to claim:

- The resident silicon executor can run log-mel -> token IDs on ET-SoC1.
- It exactly matches the current host audit for JFK.
- The performance optimizations improved full-run wall time from 227.89 s to 55.76 s.
- A harder 30 s Hawking clip exposed a real quantization-fidelity bug.
- Host ONNX debugging found the main EOS trigger: token embedding quantization.
- A portable-looking host fix exists: per-dimension token embedding scales.

What is not fair yet:

- Full robust audio-to-text Whisper on arbitrary 30 s clips.
- That the current raw-INT8 resident weights faithfully match FP32/dynamic-INT8 Whisper.
- That Hawking is solved on silicon. It is solved only far enough on host ONNX to identify the next implementation direction.

## Recommended Next Sequence

1. Update the resident weight manifest format to support per-axis scales for selected tensors.
2. Implement decoder token embedding with per-dimension INT8 scales.
3. Regenerate the raw-INT8 host ONNX audit from the same manifest and require Hawking to produce non-empty useful text.
4. Port the same token embedding scale logic to `whisper_resident_decoder_token_argbuf.c`.
5. Run JFK and Hawking on silicon.
6. If Hawking still has wording issues, fix encoder conv fidelity next, likely by keeping conv weights/biases FP32 or using a better calibrated conv quantization.
7. Only after Hawking is useful on silicon, resume performance work.

## Artifact Index

Core silicon ledger:

- `local-artifacts/erbium_amp_probe/whisper-real/phase1/WHISPER_OPTIMIZATION_RUN_LEDGER.md`
- `local-artifacts/erbium_amp_probe/whisper-real/phase1/whisper_optimization_runs.tsv`

Best silicon reports:

- `<artifact-archive>/whisper-real/phase1/full_silicon/run_20260506-154659_new23_ah16/FULL_SILICON_REPORT.md`
- `<artifact-archive>/whisper-real/phase1/full_silicon/run_20260506-161031_new64_ah16/FULL_SILICON_REPORT.md`

Raw-INT8 host comparison:

- `<artifact-archive>/whisper-real/phase1/raw_int8_host_executor_compare/hawking/run_20260506-164322/RAW_INT8_HOST_EXECUTOR_REPORT.md`
- `<artifact-archive>/whisper-real/phase1/raw_int8_host_executor_compare/jfk_30s/run_20260506-164339/RAW_INT8_HOST_EXECUTOR_REPORT.md`

ONNX sensitivity sweeps:

- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/run_20260506-165259/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/gather_20260506-165437/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/token_perrow_20260506-165602/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/improve_20260506-165732/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/encoder_20260506-165945/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/encoder_split_20260506-170047/summary.json`
- `<artifact-archive>/whisper-real/phase1/raw_int8_debug_onnx/conv_perchan_20260506-170142/summary.json`
