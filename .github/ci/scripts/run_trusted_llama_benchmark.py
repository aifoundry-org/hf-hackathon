#!/usr/bin/env python3
"""Run a contract-owned LLM quality check and llama-bench performance test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
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
    lcfg.update(
        {
            "device": runtime["required_device"],
            "gpu_layers": runtime["required_gpu_layers"],
            "require_full_offload": runtime["require_full_offload"],
            "api": "completion",
            "prompt": generation["prompt"],
            "max_tokens": generation["max_tokens"],
            "temperature": generation["temperature"],
            "ignore_eos": generation["ignore_eos"],
            "min_completion_tokens": generation["min_completion_tokens"],
        }
    )
    lcfg["perplexity"] = {
        "enabled": True,
        "perplexity_artifact": "llama_perplexity",
        "corpus_artifact": quality["corpus_artifact"],
        "ctx_size": quality["context_size"],
        "batch_size": quality["batch_size"],
        "ubatch_size": quality["ubatch_size"],
        "timeout_s": int(lcfg.get("perplexity", {}).get("timeout_s", 300)),
        "min_ppl": 1.0,
        "max_ppl": 1000.0,
        "chunks": quality["chunks"],
    }
    mcfg["reference_contract"] = str(
        Path(".github") / "ci" / "reference" / "llama32_1b.json"
    )
    write_json(output, cfg)
    return cfg


def parse_json_array(text: str) -> list[dict[str, Any]]:
    starts = [match.start() for match in re.finditer(r"(?m)^\s*\[\s*$", text)]
    for start in reversed(starts):
        try:
            value = json.loads(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value
    raise RuntimeError("llama-bench did not emit a JSON result array")


def find_bench_row(rows: list[dict[str, Any]], *, prompt: int, generation: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("n_prompt", -1)) == prompt and int(row.get("n_gen", -1)) == generation:
            return row
    raise RuntimeError(f"missing llama-bench row pp={prompt} tg={generation}")


def samples(row: dict[str, Any], repetitions: int) -> list[float]:
    values = row.get("samples_ts")
    if not isinstance(values, list) or len(values) != repetitions:
        raise RuntimeError(
            f"llama-bench returned {len(values) if isinstance(values, list) else 0} samples, "
            f"expected {repetitions}"
        )
    out = [float(value) for value in values]
    if any(value <= 0 for value in out):
        raise RuntimeError("llama-bench returned a non-positive throughput sample")
    return out


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if len(values) > 1 else 0.0


def run_bench(
    *,
    bench_bin: Path,
    model_path: Path,
    lcfg: dict[str, Any],
    contract: dict[str, Any],
    run_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    performance = contract["performance"]
    command = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-dev",
        str(contract["runtime"]["required_device"]),
        "-ngl",
        str(contract["runtime"]["required_gpu_layers"]),
        "-p",
        str(performance["prompt_tokens"]),
        "-n",
        str(performance["generation_tokens"]),
        "-b",
        str(lcfg.get("batch_size", performance["batch_size"])),
        "-ub",
        str(lcfg.get("ubatch_size", performance["ubatch_size"])),
        "-fa",
        "1" if lcfg.get("flash_attn", False) else "0",
        "-r",
        str(performance["repetitions"]),
        "-o",
        "json",
    ]
    write_json(run_dir / "llama-bench-command.json", command)
    proc = subprocess.run(
        command,
        cwd=str(bench_bin.parent),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(performance.get("timeout_s", 600)),
        check=False,
    )
    (run_dir / "llama-bench.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench exited rc={proc.returncode}")
    rows = parse_json_array(proc.stdout)
    prompt_row = find_bench_row(rows, prompt=int(performance["prompt_tokens"]), generation=0)
    decode_row = find_bench_row(rows, prompt=0, generation=int(performance["generation_tokens"]))
    expected_params = int(contract["base_model"]["parameter_count"])
    prefix = str(contract["base_model"]["architecture_prefix"])
    for row in (prompt_row, decode_row):
        if int(row.get("model_n_params", 0)) != expected_params:
            raise RuntimeError(
                f"candidate parameter count {row.get('model_n_params')} != contracted {expected_params}"
            )
        if not str(row.get("model_type", "")).startswith(prefix):
            raise RuntimeError(f"candidate model type {row.get('model_type')!r} does not match {prefix!r}")
        if "ET" not in str(row.get("backends", "")) or "ET" not in str(row.get("gpu_info", "")):
            raise RuntimeError("llama-bench did not report the ET backend and ET device")

    repetitions = int(performance["repetitions"])
    prompt_samples = samples(prompt_row, repetitions)
    decode_samples = samples(decode_row, repetitions)
    return {
        "prompt_samples": prompt_samples,
        "decode_samples": decode_samples,
        "prompt_median": statistics.median(prompt_samples),
        "decode_median": statistics.median(decode_samples),
        "prompt_cv": coefficient_of_variation(prompt_samples),
        "decode_cv": coefficient_of_variation(decode_samples),
        "model_type": decode_row.get("model_type"),
        "model_size": decode_row.get("model_size"),
        "model_n_params": decode_row.get("model_n_params"),
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
        workdir = resolve_artifact_path(mcfg, str(lcfg["workdir_artifact"]))
        server_bin = resolve_artifact_path(mcfg, str(lcfg["server_artifact"]))
        ppl_bin = resolve_artifact_path(mcfg, str(lcfg["perplexity"]["perplexity_artifact"]))
        bench_bin = server_bin.parent / "llama-bench"
        corpus_id = str(contract["quality"]["corpus_artifact"])
        corpus_path = materialize_artifact(mcfg, corpus_id)
        expected_corpus_sha = artifact_config(mcfg, corpus_id).get("sha256")
        if expected_corpus_sha:
            actual = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
            if actual != expected_corpus_sha:
                raise RuntimeError(f"PPL corpus sha256 {actual} != {expected_corpus_sha}")
        if not bench_bin.is_file():
            raise RuntimeError(f"missing llama-bench: {bench_bin}")

        env = os.environ.copy()
        runtime_library_path = os.environ.get("LLAMA_CPP_ET_LD_LIBRARY_PATH", "").strip()
        if not runtime_library_path:
            et_platform = os.environ.get("ET_PLATFORM", "").strip()
            if et_platform and (Path(et_platform) / "lib").is_dir():
                runtime_library_path = str(Path(et_platform) / "lib")
        # Do not inherit the standalone launcher's host bundle here: it may have
        # been built on a newer distro and is not part of the scored llama runtime.
        env["LD_LIBRARY_PATH"] = ":".join(
            value for value in (str(server_bin.parent), runtime_library_path) if value
        )
        bench = run_bench(
            bench_bin=bench_bin,
            model_path=model_path,
            lcfg=lcfg,
            contract=contract,
            run_dir=run_dir,
            env=env,
        )
        cpu_bin = Path(args.cpu_reference_bin).resolve() if args.cpu_reference_bin else ppl_bin
        cpu_ppl = run_cpu_perplexity(
            ppl_bin=cpu_bin,
            model_path=model_path,
            corpus_path=corpus_path,
            contract=contract,
            run_dir=run_dir,
            env=env,
        )
        et_ppl = score.get("perplexity")
        if not isinstance(et_ppl, (int, float)):
            raise RuntimeError("ET score has no PPL")
        relative_ppl_difference = abs(float(et_ppl) - float(cpu_ppl["perplexity"])) / float(
            cpu_ppl["perplexity"]
        )
        max_relative = float(contract["quality"]["max_et_cpu_ppl_relative_difference"])
        if relative_ppl_difference > max_relative:
            failures.append(
                f"ET PPL differs from trusted CPU PPL by {relative_ppl_difference:.2%}; "
                f"maximum is {max_relative:.2%}"
            )
        max_cv = float(contract["performance"]["max_sample_cv"])
        if bench["decode_cv"] > max_cv:
            failures.append(
                f"decode throughput CV {bench['decode_cv']:.2%} exceeds {max_cv:.2%}"
            )
        if bench["prompt_cv"] > max_cv:
            failures.append(
                f"prompt throughput CV {bench['prompt_cv']:.2%} exceeds {max_cv:.2%}"
            )

        score.update(
            {
                "tokens_per_second": bench["decode_median"],
                "prompt_tokens_per_second": bench["prompt_median"],
                "performance_samples": {
                    "decode_tokens_per_second": bench["decode_samples"],
                    "prompt_tokens_per_second": bench["prompt_samples"],
                    "decode_cv": bench["decode_cv"],
                    "prompt_cv": bench["prompt_cv"],
                },
                "cpu_perplexity": cpu_ppl["perplexity"],
                "cpu_perplexity_error": cpu_ppl.get("perplexity_error"),
                "et_cpu_ppl_relative_difference": relative_ppl_difference,
                "model_type": bench["model_type"],
                "model_size": bench["model_size"],
                "model_n_params": bench["model_n_params"],
            }
        )
    except Exception as exc:
        failures.append(str(exc))

    score["validation_contract_sha256"] = contract_sha256(contract_path)
    score["passed"] = not failures
    score["status"] = "pass" if not failures else "fail"
    score["valid_note"] = (
        "trusted Llama ET/CPU quality and PP256/TG128 performance passed"
        if not failures
        else "; ".join(failures)
    )
    score["note"] = score["valid_note"]
    write_json(output, score)
    print(json.dumps(score, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
