import json
import os
import copy

MODELS = {
    # Tier 1
    "gemma3_270m_q8_gguf": {
        "artifact": "gemma3_270m_q8_gguf",
        "variant": "gemma-3-270m-it-Q8_0",
        "filename": "gemma3_270m.json"
    },
    "lfm25_230m_q8_gguf": {
        "artifact": "lfm25_230m_q8_gguf",
        "variant": "LFM2.5-230M-Q8_0",
        "filename": "lfm25_230m.json"
    },
    "qwen25_coder_05b_q8_gguf": {
        "artifact": "qwen25_coder_05b_q8_gguf",
        "variant": "Qwen2.5-Coder-0.5B-Instruct-Q8_0",
        "filename": "qwen25_coder_05b.json"
    },
    # Tier 2
    "llama32_1b_q4_gguf": {
        "artifact": "llama32_1b_q4_gguf",
        "variant": "Llama-3.2-1B-Instruct-Q4_K_M",
        "filename": "llama32_1b_q4.json"
    },
    "qwen25_05b_q4_gguf": {
        "artifact": "qwen25_05b_q4_gguf",
        "variant": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "filename": "qwen25_05b_q4.json"
    },
    "smollm2_360m_q4_gguf": {
        "artifact": "smollm2_360m_q4_gguf",
        "variant": "SmolLM2-360M-Instruct-Q4_K_M",
        "filename": "smollm2_360m_q4.json"
    }
}

TEMPLATE_FILE = "ported_models/llama_cpp_et/benchmarks/qwen25_05b.json"

def main():
    with open(TEMPLATE_FILE, "r") as f:
        template = json.load(f)

    for key, info in MODELS.items():
        config = copy.deepcopy(template)
        config["canonical_variant"] = info["variant"]
        config["llama_server"]["model_artifact"] = info["artifact"]
        config["benchmark_default"] = True

        out_path = os.path.join("ported_models", "llama_cpp_et", "benchmarks", info["filename"])
        with open(out_path, "w", newline='\n') as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(f"Created {out_path}")

if __name__ == "__main__":
    main()
