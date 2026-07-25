"""Torch/transformers host oracle for JeremiahZ/roberta-base-sst2 -- a genuinely
binary (negative/positive) classification model, RoBERTa architecture (distinct
pretraining lineage from DistilBERT and ModernBERT, even though all three route
through llama.cpp's shared LLM_ARCH_BERT graph builder).

Provenance: Hugging Face `JeremiahZ/roberta-base-sst2`, file `pytorch_model.bin`
or `model.safetensors`. License: MIT. Architecture:
RobertaForSequenceClassification, id2label={0: negative, 1: positive}.

Run: python3 gen_roberta_sst2_oracle.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "JeremiahZ/roberta-base-sst2"
REVISION = "0daa4dc5532d79114a5e76e277d2d921c1bfd9ed"
HERE = Path(__file__).resolve().parent
REFS = HERE / "refs"

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

    np.save(REFS / "roberta_sst2_reference_logits.npy", logits)
    np.save(REFS / "roberta_sst2_input_ids.npy", input_ids.astype(np.int64))
    (REFS / "roberta_sst2_reference_top1.json").write_text(json.dumps({
        "label": label,
        "logit": float(logits[top1_id]),
        "logits_all": logits.tolist(),
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "prompt": GATE_PROMPT,
        "input_ids": input_ids.tolist(),
    }, indent=2) + "\n")

    print(f"revision={revision}")
    print(f"top1: label={label!r} logit={logits[top1_id]:.4f}")
    print(f"id2label={model.config.id2label}")
    print(f"seq_len={len(input_ids)} refs -> {REFS}")


if __name__ == "__main__":
    main(REVISION)
