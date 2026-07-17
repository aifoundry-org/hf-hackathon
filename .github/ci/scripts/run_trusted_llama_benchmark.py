#!/usr/bin/env python3
"""Run the bounded, contract-owned Llama quality and performance check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark_config_helpers import load_config
from prepare_trusted_llama32_candidate import apply_candidate_manifest
from run_llama_server_benchmark import (
    artifact_config,
    materialize_artifact,
    parse_perplexity_log,
    resolve_artifact_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_RUNNER = REPO_ROOT / ".github" / "ci" / "scripts" / "run_llama_server_benchmark.py"
MAX_SAFE_ET_GENERATION_TOKENS = 24


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def effective_config(config_path: Path, contract: dict[str, Any], output: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    model = str(contract["model"])
    mcfg = cfg["models"][model]
    artifact_id = str(mcfg["llama_server"]["model_artifact"])
    artifact = mcfg["artifacts"][artifact_id]
    manifest_path = REPO_ROOT / str(contract["candidate_manifest"])
    if not artifact.get("submission_manifest_sha256") and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        recipe = REPO_ROOT / str(manifest.get("recipe") or "")
        if not recipe.is_file():
            raise RuntimeError("committed candidate manifest recipe does not exist")
        apply_candidate_manifest(mcfg, manifest, contract)
    lcfg = mcfg["llama_server"]
    runtime = contract["runtime"]
    generation = contract["generation_validation"]
    quality = contract["quality"]
    max_tokens = int(generation["max_tokens"])
    if max_tokens > MAX_SAFE_ET_GENERATION_TOKENS:
        raise RuntimeError(
            f"generation contract requests {max_tokens} tokens; "
            f"safe ET limit is {MAX_SAFE_ET_GENERATION_TOKENS}"
        )
    if contract["performance"].get("tool") != "llama-server":
        raise RuntimeError("trusted Llama performance must use the bounded llama-server request")
    lcfg.update(
        {
            "device": runtime["required_device"],
            "gpu_layers": runtime["required_gpu_layers"],
            "require_full_offload": runtime["require_full_offload"],
            "api": "completion",
            "prompt": generation["prompt"],
            "max_tokens": max_tokens,
            "temperature": generation["temperature"],
            "ignore_eos": generation["ignore_eos"],
            "min_completion_tokens": generation["min_completion_tokens"],
        }
    )
    lcfg["perplexity"] = {
        "enabled": False,
        "perplexity_artifact": "llama_perplexity",
        "corpus_artifact": quality["corpus_artifact"],
        "ctx_size": quality["context_size"],
        "batch_size": quality["batch_size"],
        "ubatch_size": quality["ubatch_size"],
        "timeout_s": int(lcfg.get("perplexity", {}).get("timeout_s", 300)),
        "chunks": quality["chunks"],
        "device": "CPU",
        "gpu_layers": 0,
        "require_full_offload": False,
    }
    mcfg["reference_contract"] = str(
        Path(".github") / "ci" / "reference" / "llama32_1b.json"
    )
    write_json(output, cfg)
    return cfg


def _read_exact(handle: Any, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise RuntimeError("truncated GGUF metadata")
    return value


def _read_u32(handle: Any) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: Any) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _skip_gguf_string(handle: Any) -> None:
    length = _read_u64(handle)
    if length > 1 << 30:
        raise RuntimeError("invalid GGUF string length")
    handle.seek(length, os.SEEK_CUR)


def _skip_gguf_value(handle: Any, value_type: int) -> None:
    fixed_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    if value_type in fixed_sizes:
        handle.seek(fixed_sizes[value_type], os.SEEK_CUR)
    elif value_type == 8:
        _skip_gguf_string(handle)
    elif value_type == 9:
        element_type = _read_u32(handle)
        count = _read_u64(handle)
        if count > 1 << 30:
            raise RuntimeError("invalid GGUF array length")
        for _ in range(count):
            _skip_gguf_value(handle, element_type)
    else:
        raise RuntimeError(f"unsupported GGUF metadata type {value_type}")


def gguf_parameter_count(path: Path) -> int:
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise RuntimeError(f"candidate is not a GGUF file: {path}")
        version = _read_u32(handle)
        if version not in (2, 3):
            raise RuntimeError(f"unsupported GGUF version {version}")
        tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        if tensor_count > 1_000_000 or metadata_count > 1_000_000:
            raise RuntimeError("unreasonable GGUF table size")
        for _ in range(metadata_count):
            _skip_gguf_string(handle)
            _skip_gguf_value(handle, _read_u32(handle))
        parameters = 0
        for _ in range(tensor_count):
            _skip_gguf_string(handle)
            dimensions = _read_u32(handle)
            if dimensions > 8:
                raise RuntimeError(f"invalid GGUF tensor rank {dimensions}")
            shape = [_read_u64(handle) for _ in range(dimensions)]
            _read_u32(handle)  # ggml tensor type
            _read_u64(handle)  # data offset
            parameters += math.prod(shape)
    return parameters


def validate_server_model_identity(
    log_path: Path, model_path: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    log = log_path.read_text(errors="replace")
    base = contract["base_model"]
    architecture = str(base["architecture"])
    model_type = str(base["model_type"])
    if not re.search(
        rf"print_info:\s+arch\s+=\s+{re.escape(architecture)}\s*$",
        log,
        re.MULTILINE,
    ):
        raise RuntimeError(f"server did not report architecture {architecture}")
    if not re.search(
        rf"print_info:\s+model type\s+=\s+{re.escape(model_type)}\s*$",
        log,
        re.MULTILINE,
    ):
        raise RuntimeError(f"server did not report model type {model_type}")
    parameter_count = gguf_parameter_count(model_path)
    expected = int(base["parameter_count"])
    if parameter_count != expected:
        raise RuntimeError(
            f"candidate parameter count {parameter_count} != contracted {expected}"
        )
    return {
        "model_type": f"{architecture} {model_type}",
        "model_size": model_path.stat().st_size,
        "model_n_params": parameter_count,
    }


def run_cpu_perplexity(
    *,
    ppl_bin: Path,
    model_path: Path,
    corpus_path: Path,
    contract: dict[str, Any],
    run_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    quality = contract["quality"]
    command = [
        str(ppl_bin),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "-dev",
        "CPU",
        "-ngl",
        "0",
        "-c",
        str(quality["context_size"]),
        "-b",
        str(quality["batch_size"]),
        "-ub",
        str(quality["ubatch_size"]),
        "--chunks",
        str(quality["chunks"]),
        "--no-warmup",
    ]
    write_json(run_dir / "cpu-perplexity-command.json", command)
    cpu_env = env.copy()
    cpu_env["LD_LIBRARY_PATH"] = f"{ppl_bin.parent}:{cpu_env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        command,
        cwd=str(ppl_bin.parent),
        env=cpu_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(quality.get("cpu_timeout_s", 600)),
        check=False,
    )
    (run_dir / "cpu-perplexity.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"CPU llama-perplexity exited rc={proc.returncode}")
    metrics = parse_perplexity_log(proc.stdout)
    if not isinstance(metrics.get("perplexity"), float):
        raise RuntimeError("CPU llama-perplexity did not emit a final estimate")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama32_1b")
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-reference-bin", default="")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    contract_path = Path(args.contract).resolve()
    run_dir = Path(args.results_dir).resolve()
    output = Path(args.output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(contract_path.read_text())
    if args.model != contract.get("model"):
        raise SystemExit("contract model does not match --model")

    trusted_config_path = run_dir / "trusted-config.json"
    cfg = effective_config(config_path, contract, trusted_config_path)
    mcfg = cfg["models"][args.model]
    lcfg = mcfg["llama_server"]

    base_score_path = run_dir / "base-score.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(LEGACY_RUNNER),
            "--model",
            args.model,
            "--config",
            str(trusted_config_path),
            "--results-dir",
            str(run_dir / "server"),
            "--output",
            str(base_score_path),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if not base_score_path.is_file():
        raise SystemExit(f"llama-server runner exited rc={proc.returncode} without a score")
    score = json.loads(base_score_path.read_text())
    failures: list[str] = []
    if not score.get("passed"):
        failures.append(str(score.get("valid_note") or score.get("note") or "llama-server validation failed"))

    try:
        model_path = materialize_artifact(mcfg, str(lcfg["model_artifact"]))
        server_bin = resolve_artifact_path(mcfg, str(lcfg["server_artifact"]))
        ppl_bin = resolve_artifact_path(mcfg, str(lcfg["perplexity"]["perplexity_artifact"]))
        corpus_id = str(contract["quality"]["corpus_artifact"])
        corpus_path = materialize_artifact(mcfg, corpus_id)
        expected_corpus_sha = artifact_config(mcfg, corpus_id).get("sha256")
        if expected_corpus_sha:
            actual = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            if actual != expected_corpus_sha:
                raise RuntimeError(f"PPL corpus sha256 {actual} != {expected_corpus_sha}")

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{server_bin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        cpu_bin = Path(args.cpu_reference_bin).resolve() if args.cpu_reference_bin else ppl_bin
        cpu_ppl = run_cpu_perplexity(
            ppl_bin=cpu_bin,
            model_path=model_path,
            corpus_path=corpus_path,
            contract=contract,
            run_dir=run_dir,
            env=env,
        )
        identity = validate_server_model_identity(
            run_dir / "server" / "server.log", model_path, contract
        )
        decode_speed = score.get("tokens_per_second")
        prompt_speed = score.get("prompt_tokens_per_second")
        if not isinstance(decode_speed, (int, float)) or float(decode_speed) <= 0:
            raise RuntimeError("llama-server did not report positive decode throughput")
        if not isinstance(prompt_speed, (int, float)) or float(prompt_speed) <= 0:
            raise RuntimeError("llama-server did not report positive prompt throughput")

        score.update(
            {
                "performance_samples": {
                    "tool": "llama-server",
                    "decode_tokens_per_second": [float(decode_speed)],
                    "prompt_tokens_per_second": [float(prompt_speed)],
                },
                "perplexity": cpu_ppl["perplexity"],
                "perplexity_error": cpu_ppl.get("perplexity_error"),
                "perplexity_tokens": cpu_ppl.get("perplexity_tokens"),
                "perplexity_prompt_tokens_per_second": cpu_ppl.get(
                    "perplexity_prompt_tokens_per_second"
                ),
                "perplexity_device": "CPU",
                "et_process_count": 1,
                **identity,
            }
        )
    except Exception as exc:
        failures.append(str(exc))

    score["validation_contract_sha256"] = contract_sha256(contract_path)
    score["passed"] = not failures
    score["status"] = "pass" if not failures else "fail"
    score["valid_note"] = (
        (
            "trusted Llama single-process ET generation and CPU PPL passed"
        )
        if not failures
        else "; ".join(failures)
    )
    score["note"] = score["valid_note"]
    write_json(output, score)
    print(json.dumps(score, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
