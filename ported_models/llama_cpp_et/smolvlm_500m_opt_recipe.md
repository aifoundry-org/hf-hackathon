# SmolVLM-500M Q8_0 — optimization recipe

## Architecture notes

SmolVLM-500M is a multimodal Vision-Language Model (VLM):
- **Idefics3** vision-text architecture + **SigLIP** vision encoder (~93M)
- Multimodal projection layer + SmolLM2 language backbone
- Exercises real vision: loads mmproj, runs COCO image fixtures, visual-answer gate

## Config changes vs default

| Setting | Value | Rationale |
|---------|-------|-----------|
| `ctx_size` | 256 | Small KV cache |
| `flash_attn` | true | Hardware-accelerated flash attention |
| `gpu_layers` | 99 | Full ET offload |
| `extra_args` | `-nkvo` | No KV offload — tiny cache fits on-device |

## Status

Enrolled as active benchmark default for ET-SoC1 board VLM evaluation.
