"""Torch/transformers host oracle for distilbert-base-uncased-finetuned-sst-2-english --
proposed as a NEW model-port identity (encoder-only text classification, architecturally
distinct from every causal-LM family already registered: llama/qwen2/qwen3/smollm2/
gemma3n/rwkv7/lfm2 are all decoder-only generative models; DistilBERT is a bidirectional
encoder producing a single classification head output, not next-token generation).

Provenance: Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`, file
`model.safetensors`. License: apache-2.0 (per the HF repo card). Architecture:
DistilBertForSequenceClassification, 6 transformer layers, 768 hidden, 12 heads,
2-way sentiment classification (NEGATIVE/POSITIVE).

Emits (mirrors the DnCNN oracle pattern):
  distilbert_sst2_reference_logits.npy  float32[2]
  distilbert_sst2_reference_top1.json   {label, prob}
  distilbert_sst2_input_ids.npy         int64[seq_len]  the exact tokenized gated input

Run: python3 gen_distilbert_sst2_oracle.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"

# Fixed, reproducible board-gate input -- same role as DnCNN's synthetic image and
# MobileNetV2's synthetic RGB tensor: a small, deterministic, self-contained input,
# not a scraped external asset.
GATE_PROMPT = "This board runs faster than anything the competition has shipped."


def build_oracle(revision: str):
    tok = AutoTokenizer.from_pretrained(REPO, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(REPO, revision=revision)
    model.eval()
    return tok, model


def oracle_forward(tok, model, text: str):
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0].numpy().astype(np.float32)
    top1_id = int(logits.argmax())
    label = model.config.id2label[top1_id]
    return logits, top1_id, label, inputs["input_ids"][0].numpy()


def main(revision: str):
    REFS.mkdir(parents=True, exist_ok=True)
    tok, model = build_oracle(revision)
    logits, top1_id, label, input_ids = oracle_forward(tok, model, GATE_PROMPT)

    np.save(REFS / "distilbert_sst2_reference_logits.npy", logits)
    np.save(REFS / "distilbert_sst2_input_ids.npy", input_ids.astype(np.int64))
    (REFS / "distilbert_sst2_reference_top1.json").write_text(json.dumps({
        "label": label,
        "logit": float(logits[top1_id]),
        "prob": float(torch.softmax(torch.from_numpy(logits), dim=0)[top1_id]),
        "prompt": GATE_PROMPT,
        "input_ids": input_ids.tolist(),
    }, indent=2) + "\n")

    print(f"revision={revision}")
    print(f"top1: label={label!r} logit={logits[top1_id]:.4f}")
    print(f"seq_len={len(input_ids)} refs -> {REFS}")


if __name__ == "__main__":
    main(REVISION)
