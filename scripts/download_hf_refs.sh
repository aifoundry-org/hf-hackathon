#!/usr/bin/env bash
set -euo pipefail

out="${1:-${HF_REFS_OUT:-local-artifacts/hf_refs}}"

if ! command -v hf >/dev/null 2>&1; then
  echo "error: hf CLI is required. Install huggingface_hub or run: pip install -U huggingface_hub" >&2
  exit 2
fi

download_model() {
  local label="$1"
  local repo="$2"
  local revision="$3"
  local model_dir="$4"
  shift 4
  local download_args=(--revision "$revision" --local-dir "$out/$model_dir")

  if [[ "${HF_DOWNLOAD_DRY_RUN:-0}" == "1" ]]; then
    download_args+=(--dry-run)
  fi

  mkdir -p "$out/$model_dir"
  echo "Downloading $label from Hugging Face: $repo@$revision"
  hf download "$repo" "$@" "${download_args[@]}"
}

download_model \
  "yolov10n" \
  "kadirnar/yolov10n" \
  "9fa42234fbcdb13b78fa57ebaac6c50e6dd2eb21" \
  "yolo" \
  "yolov10n.pt"

echo "Hugging Face references ready under $out"
