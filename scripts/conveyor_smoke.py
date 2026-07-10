import os
import json
import urllib.request
import sys

MODELS = {
    # Tier 1: Micro-Models Q8_0
    "gemma3_270m_q8_gguf": {
        "model": "gemma-3-270m-it",
        "repo": "unsloth/gemma-3-270m-it-GGUF",
        "filename": "gemma-3-270m-it-Q8_0.gguf"
    },
    "lfm25_230m_q8_gguf": {
        "model": "LFM2.5-230M",
        "repo": "LiquidAI/LFM2.5-230M-GGUF",
        "filename": "LFM2.5-230M-Q8_0.gguf"
    },
    "qwen25_coder_05b_q8_gguf": {
        "model": "Qwen2.5-Coder-0.5B-Instruct",
        "repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-coder-0.5b-instruct-q8_0.gguf"
    },
    # Tier 2: Existing Fleet in Q4_K_M
    "llama32_1b_q4_gguf": {
        "model": "Llama-3.2-1B-Instruct",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    },
    "qwen25_05b_q4_gguf": {
        "model": "Qwen2.5-0.5B-Instruct",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    "smollm2_360m_q4_gguf": {
        "model": "SmolLM2-360M-Instruct",
        "repo": "bartowski/SmolLM2-360M-Instruct-GGUF",
        "filename": "SmolLM2-360M-Instruct-Q4_K_M.gguf"
    }
}

ARTIFACTS_FILE = "ported_models/llama_cpp_et/artifacts.json"

def get_hf_metadata(repo, filename):
    url = f'https://huggingface.co/api/models/{repo}/tree/main'
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        files = json.loads(response.read().decode())
    for f in files:
        if f['path'] == filename:
            if 'lfs' in f:
                return f['lfs']['oid'], f['lfs']['size']
            else:
                raise Exception(f"No LFS data for {filename}")
    raise Exception(f"File {filename} not found in repo {repo}")

def main():
    with open(ARTIFACTS_FILE, "r") as f:
        data = json.load(f)

    for key, info in MODELS.items():
        print(f"Fetching metadata for {info['filename']}...")
        sha256_hash, size_bytes = get_hf_metadata(info['repo'], info['filename'])
        url = f"https://huggingface.co/{info['repo']}/resolve/main/{info['filename']}"

        artifact_entry = {
            "kind": "model_weights",
            "model": info["model"],
            "format": "gguf",
            "filename": info["filename"],
            "sha256": sha256_hash,
            "size_bytes": size_bytes,
            "source": {
                "type": "huggingface",
                "repo": info["repo"],
                "filename": info["filename"],
                "url": url
            },
            "env": f"{key.upper()}_PATH",
            "local_cache": f"local-artifacts/models/{info['filename']}",
            "board_path": f"/data/models/{info['filename']}"
        }

        data["artifacts"][key] = artifact_entry
        print(f"Added {key} to artifacts.json with hash {sha256_hash}")

    with open(ARTIFACTS_FILE, "w", newline='\n') as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print("conveyor_smoke.py completed successfully.")

if __name__ == "__main__":
    main()
