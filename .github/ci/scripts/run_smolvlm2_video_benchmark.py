#!/usr/bin/env python3
"""Validate and benchmark the pinned SmolVLM2 workload on ET-SoC1."""

from __future__ import annotations

import argparse
import base64
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from board_lock import open_board_lock
from benchmark_config_helpers import load_config
from run_llama_server_benchmark import (
    artifact_config,
    ensure_llama_cpp_build,
    is_dir,
    is_file,
    materialize_artifact,
    post_completion,
    parse_perplexity_log,
    run_perplexity,
    score_common,
    sha256_file,
    terminate,
    wait_ready,
    write_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
REQUEST_PATH = "/completion"
KERNEL_RESPONSE_TYPE = 2


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def unsupported_vision_ops(log: str) -> set[str]:
    return set(re.findall(r"^warmup:\s+([A-Z][A-Z0-9_]+):\s+type\s*=", log, re.MULTILINE))


def log_failures(log: str, *, mode: str, request_count: int, allowed_ops: set[str]) -> list[str]:
    failures: list[str] = []
    observed_requests = log.count(f"done request: POST {REQUEST_PATH}")
    if observed_requests != request_count:
        failures.append(f"{mode}: observed {observed_requests}/{request_count} completed requests")

    if mode == "host":
        if "CLIP using CPU backend" not in log:
            failures.append("host: CPU vision backend not observed")
        return failures

    if "using device ET" not in log and "ET device 0" not in log:
        failures.append("board: ET device use not observed")
    if "CLIP using ET backend" not in log:
        failures.append("board: ET vision backend not observed")
    match = re.search(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU", log)
    if not match:
        failures.append("board: LLM layer offload summary not observed")
    elif match.group(1) != match.group(2):
        failures.append(f"board: expected full LLM offload, got {match.group(1)}/{match.group(2)}")

    observed_ops = unsupported_vision_ops(log)
    if not allowed_ops and "CLIP graph uses unsupported operators" in log:
        failures.append("board: vision graph contains CPU fallback operations")
    unexpected = observed_ops - allowed_ops
    if unexpected:
        failures.append("board: unexpected vision fallback ops: " + ", ".join(sorted(unexpected)))
    return failures


def model_identity_failures(log: str, contract: dict[str, Any]) -> list[str]:
    language = contract["architecture"]["language"]
    vision = contract["architecture"]["vision"]
    checks = {
        "language architecture": rf"general\.architecture\s+str\s+=\s+{re.escape(str(language['general_architecture']))}",
        "model name": rf"general\.name\s+str\s+=\s+{re.escape(str(language['model_name']))}",
        "language parameter count": rf"model params\s+=\s+{float(language['parameter_count_millions']):.2f}\s+M",
        "language tensor count": rf"loaded meta data with\s+[0-9]+\s+key-value pairs and\s+{int(language['tensor_count'])}\s+tensors",
        "language block count": rf"llama\.block_count\s+u32\s+=\s+{int(language['block_count'])}",
        "language embedding length": rf"llama\.embedding_length\s+u32\s+=\s+{int(language['embedding_length'])}",
        "language feed-forward length": rf"llama\.feed_forward_length\s+u32\s+=\s+{int(language['feed_forward_length'])}",
        "language attention heads": rf"llama\.attention\.head_count\s+u32\s+=\s+{int(language['attention_heads'])}",
        "language KV heads": rf"llama\.attention\.head_count_kv\s+u32\s+=\s+{int(language['attention_kv_heads'])}",
        "language vocabulary": rf"llama\.vocab_size\s+u32\s+=\s+{int(language['vocabulary_size'])}",
        "vision model name": rf"clip_model_loader: model name:\s+{re.escape(str(language['model_name']))}",
        "vision tensor count": rf"clip_model_loader: n_tensors:\s+{int(vision['tensor_count'])}",
        "vision projector": rf"load_hparams: projector:\s+{re.escape(str(vision['projector']))}",
        "vision embedding length": rf"load_hparams: n_embd:\s+{int(vision['embedding_length'])}",
        "vision attention heads": rf"load_hparams: n_head:\s+{int(vision['attention_heads'])}",
        "vision feed-forward length": rf"load_hparams: n_ff:\s+{int(vision['feed_forward_length'])}",
        "vision block count": rf"load_hparams: n_layer:\s+{int(vision['block_count'])}",
        "vision projection dimension": rf"load_hparams: projection_dim:\s+{int(vision['projection_dimension'])}",
        "vision image size": rf"load_hparams: image_size:\s+{int(vision['image_size'])}",
        "vision patch size": rf"load_hparams: patch_size:\s+{int(vision['patch_size'])}",
    }
    return [
        f"loader log did not confirm {name}"
        for name, pattern in checks.items()
        if not re.search(pattern, log)
    ]


def build_command(
    server_bin: Path,
    model_path: Path,
    mmproj_path: Path,
    cfg: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--host",
        str(cfg.get("host", "127.0.0.1")),
        "--port",
        str(cfg.get("port", 18107)),
        "-c",
        str(cfg.get("ctx_size", 2048)),
        "-b",
        str(cfg.get("batch_size", 256)),
        "-ub",
        str(cfg.get("ubatch_size", 128)),
        "-np",
        str(cfg.get("parallel", 1)),
        "--cache-ram",
        str(cfg.get("cache_ram_mib", 0)),
        "--no-warmup",
    ]
    if mode == "host":
        cmd.extend(["--no-mmproj-offload", "-dev", "none", "-ngl", "0"])
    else:
        cmd.extend(
            [
                "--mmproj-offload",
                "-dev",
                str(cfg.get("device", "ET")),
                "-ngl",
                str(cfg.get("gpu_layers", 99)),
            ]
        )
    cmd.extend(str(value) for value in cfg.get("extra_args", []))
    return cmd


def make_request(
    case: dict[str, Any],
    media: list[Path],
    cfg: dict[str, Any],
    *,
    max_tokens: int | None = None,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    markers = "".join("<__media__>" for _ in media)
    prompt = str(cfg["prompt_template"]).format(
        media_markers=markers,
        question=str(case["question"]),
    )
    request = {
        "prompt": {
            "prompt_string": prompt,
            "multimodal_data": [base64.b64encode(path.read_bytes()).decode("ascii") for path in media],
        },
        "n_predict": int(max_tokens if max_tokens is not None else cfg.get("max_tokens", 12)),
        "temperature": cfg.get("temperature", 0),
        "top_k": int(cfg.get("top_k", 1)),
    }
    if ignore_eos:
        request["ignore_eos"] = True
    return request


def profile_extras(event: dict[str, Any]) -> dict[str, tuple[int | None, Any]]:
    extras: dict[str, tuple[int | None, Any]] = {}
    for item in event.get("extra", []):
        value = item.get("value") if isinstance(item, dict) else None
        key = item.get("key") if isinstance(item, dict) else None
        if isinstance(key, str) and isinstance(value, dict):
            extras[key] = (value.get("index"), value.get("data"))
    return extras


def profile_timestamp_ns(event: dict[str, Any]) -> int:
    value = event.get("timeStamp", {}).get("time_since_epoch", {}).get("count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("runtime profile event has an invalid monotonic timestamp")
    return value


def _positive_int(extra: dict[str, tuple[int | None, Any]], key: str, index: int) -> int:
    value = extra.get(key)
    if (
        not value
        or value[0] != index
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] <= 0
    ):
        raise ValueError(f"runtime profile has invalid {key}")
    return value[1]


def read_firmware_cycles(
    profile_path: Path,
    kernel_map_path: Path,
    windows: list[dict[str, Any]],
    performance: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(profile_path.read_text())
    if not isinstance(payload, dict) or not payload:
        raise ValueError("runtime profile root is empty or invalid")
    events = list(payload.values())
    if not all(isinstance(event, dict) for event in events):
        raise ValueError("runtime profile contains a non-object event")

    starts = [event for event in events if event.get("class") == "StartProfiling"]
    ends = [event for event in events if event.get("class") == "EndProfiling"]
    if len(starts) != 1 or len(ends) != 1 or events[0] is not starts[0] or events[-1] is not ends[0]:
        raise ValueError("runtime profile does not contain one complete start/end window")
    version = profile_extras(starts[0]).get("version")
    expected_version = int(performance["profile_schema_version"])
    if version != (8, expected_version):
        raise ValueError(f"runtime profile version {version!r} != (8, {expected_version})")

    ordered = sorted(windows, key=lambda item: int(item["started_ns"]))
    for previous, current in zip(ordered, ordered[1:]):
        if int(previous["ended_ns"]) >= int(current["started_ns"]):
            raise ValueError("request profiling windows overlap")

    raw_kernel_map = json.loads(kernel_map_path.read_text())
    if not isinstance(raw_kernel_map, dict) or not raw_kernel_map:
        raise ValueError("runtime kernel map is empty or invalid")
    kernel_names = {int(value): str(name) for name, value in raw_kernel_map.items()}

    launches: dict[int, int] = {}
    for event in events:
        if event.get("class") != "KernelLaunch":
            continue
        extras = profile_extras(event)
        event_id = _positive_int(extras, "event", 1)
        kernel_id = extras.get("kernel_id")
        if (
            not kernel_id
            or kernel_id[0] != 4
            or isinstance(kernel_id[1], bool)
            or not isinstance(kernel_id[1], int)
            or kernel_id[1] < 0
        ):
            raise ValueError("runtime profile has an invalid kernel launch identifier")
        launches[event_id] = kernel_id[1]

    measurements = {
        str(window["request_id"]): {
            "cycles": 0,
            "kernel_launches": 0,
            "kernels": set(),
            "first_profile_timestamp_ns": None,
            "last_profile_timestamp_ns": None,
        }
        for window in windows
    }
    matched_responses = 0
    for event in events:
        if event.get("class") != str(performance["profile_class"]):
            continue
        extras = profile_extras(event)
        if extras.get("rsp_type") != (5, KERNEL_RESPONSE_TYPE):
            continue
        timestamp = profile_timestamp_ns(event)
        window = next(
            (
                item
                for item in ordered
                if int(item["started_ns"]) <= timestamp <= int(item["ended_ns"])
            ),
            None,
        )
        if window is None:
            continue

        cycles = _positive_int(extras, str(performance["primary_counter"]), 0)
        _positive_int(extras, "device_cmd_start_ts", 0)
        wait = extras.get("device_cmd_wait_dur")
        if (
            not wait
            or wait[0] != 0
            or isinstance(wait[1], bool)
            or not isinstance(wait[1], int)
            or wait[1] < 0
        ):
            raise ValueError("runtime profile has invalid device_cmd_wait_dur")
        event_id = _positive_int(extras, "event", 1)
        if event_id not in launches:
            raise ValueError(f"kernel response event {event_id} has no matching launch")
        kernel_id = launches[event_id]
        if kernel_id not in kernel_names:
            raise ValueError(f"kernel response uses unknown kernel id {kernel_id}")

        item = measurements[str(window["request_id"])]
        item["cycles"] += cycles
        item["kernel_launches"] += 1
        item["kernels"].add(kernel_names[kernel_id])
        first = item["first_profile_timestamp_ns"]
        item["first_profile_timestamp_ns"] = timestamp if first is None else min(first, timestamp)
        last = item["last_profile_timestamp_ns"]
        item["last_profile_timestamp_ns"] = timestamp if last is None else max(last, timestamp)
        matched_responses += 1

    minimum_launches = int(performance["minimum_kernel_launches_per_request"])
    for request_id, item in measurements.items():
        if int(item["kernel_launches"]) < minimum_launches:
            raise ValueError(
                f"request {request_id} has {item['kernel_launches']} profiled kernels; "
                f"expected at least {minimum_launches}"
            )
        item["kernels"] = sorted(item["kernels"])

    summary = {
        "profile_schema_version": expected_version,
        "counter": str(performance["primary_counter"]),
        "request_count": len(windows),
        "matched_kernel_responses": matched_responses,
        "kernel_map": {name: int(value) for name, value in raw_kernel_map.items()},
    }
    return measurements, summary


def runtime_env(server_bin: Path, server_cfg: dict[str, Any], *, mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TMPDIR"] = env.get("TMPDIR", "/dev/shm")
    configured_library_path = os.environ.get("LLAMA_CPP_ET_LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        value for value in (str(server_bin.parent), configured_library_path) if value
    )
    if server_cfg.get("flash_attn") is False:
        env["LLAMA_ARG_FLASH_ATTN"] = "off"
    if mode == "board":
        env["MTMD_BACKEND_DEVICE"] = str(server_cfg.get("device", "ET"))
    else:
        env.pop("MTMD_BACKEND_DEVICE", None)
    return env


def correctness_specs(
    cases: list[tuple[dict[str, Any], list[Path]]],
) -> list[dict[str, Any]]:
    return [
        {"request_id": f"correctness-{case['name']}", "kind": "correctness", "case": case, "media": media}
        for case, media in cases
    ]


def selected_correctness_cases(
    cases: list[tuple[dict[str, Any], list[Path]]],
    correctness: dict[str, Any],
) -> list[tuple[dict[str, Any], list[Path]]]:
    selected_names = [str(value) for value in correctness.get("ci_cases", [])]
    if not selected_names:
        return cases
    by_name = {str(case["name"]): (case, media) for case, media in cases}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError("CI correctness cases are not in the input contract: " + ", ".join(missing))
    return [by_name[name] for name in selected_names]


def performance_spec(
    cases: list[tuple[dict[str, Any], list[Path]]],
    performance: dict[str, Any],
    *,
    kind: str,
    repetition: int,
) -> dict[str, Any]:
    performance_case = str(performance["performance_case"])
    selected = next(((case, media) for case, media in cases if case["name"] == performance_case), None)
    if selected is None:
        raise ValueError(f"performance case {performance_case!r} is not in the correctness suite")
    case, media = selected
    return {
        "request_id": f"{kind.replace('_', '-')}-{repetition}",
        "kind": kind,
        "case": case,
        "media": media,
        "repetition": repetition,
    }


def run_mode(
    *,
    mode: str,
    cmd: list[str],
    server_cfg: dict[str, Any],
    multimodal_cfg: dict[str, Any],
    cases: list[tuple[dict[str, Any], list[Path]]],
    performance: dict[str, Any],
    workdir: Path,
    run_dir: Path,
    server_bin: Path,
    specs: list[dict[str, Any]],
    run_label: str,
    profile_board: bool = False,
) -> tuple[list[dict[str, Any]], list[str], set[str], dict[str, Any]]:
    log_path = run_dir / f"server-{run_label}.log"
    command_path = run_dir / f"command-{run_label}.json"
    response_path = run_dir / f"responses-{run_label}.json"
    command_path.write_text(json.dumps(cmd, indent=2) + "\n")
    env = runtime_env(server_bin, server_cfg, mode=mode)
    profile_dir = run_dir / f"et-profile-{run_label}"
    if mode == "board" and profile_board:
        shutil.rmtree(profile_dir, ignore_errors=True)
        profile_dir.mkdir(parents=True)
        env["GGML_ET_PROFILE"] = str(profile_dir)

    proc: subprocess.Popen[bytes] | None = None
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    profile_summary: dict[str, Any] = {}
    try:
        with log_path.open("wb") as log:
            log.write(("$ " + " ".join(cmd) + "\n\n").encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir, env=env)
            ready, note = wait_ready(proc, log_path, int(server_cfg.get("ready_timeout_s", 180)))
            if not ready:
                return [], [f"{mode}: {note}"], set(), {}

            url = f"http://{server_cfg.get('host', '127.0.0.1')}:{server_cfg.get('port', 18107)}{REQUEST_PATH}"
            for spec in specs:
                performance_request = spec["kind"] != "correctness"
                max_tokens = (
                    int(performance["generation_tokens"])
                    if performance_request
                    else int(server_cfg.get("max_tokens", 12))
                )
                payload = make_request(
                    spec["case"],
                    spec["media"],
                    {
                        **multimodal_cfg,
                        "max_tokens": server_cfg.get("max_tokens", 12),
                        "temperature": server_cfg.get("temperature", 0),
                        "top_k": server_cfg.get("top_k", 1),
                    },
                    max_tokens=max_tokens,
                )
                started_ns = time.monotonic_ns()
                status, response = post_completion(url, payload, int(server_cfg.get("request_timeout_s", 300)))
                ended_ns = time.monotonic_ns()
                content = str(response.get("content") or "")
                result = {
                    "request_id": spec["request_id"],
                    "kind": spec["kind"],
                    "case": str(spec["case"]["name"]),
                    "repetition": spec.get("repetition"),
                    "status": status,
                    "content": content,
                    "normalized_answer": normalize_answer(content),
                    "elapsed_s": (ended_ns - started_ns) / 1e9,
                    "started_ns": started_ns,
                    "ended_ns": ended_ns,
                    "frame_count": len(spec["media"]),
                    "tokens_predicted": response.get("tokens_predicted"),
                    "tokens_evaluated": response.get("tokens_evaluated"),
                    "timings": response.get("timings", {}),
                }
                results.append(result)
                if status != 200:
                    failures.append(
                        f"{mode}/{spec['request_id']}: HTTP {status}: {response.get('error', '')}"
                    )
                if not result["normalized_answer"]:
                    failures.append(f"{mode}/{spec['request_id']}: empty answer")
                if performance_request and result["tokens_predicted"] != max_tokens:
                    failures.append(
                        f"{mode}/{spec['request_id']}: generated {result['tokens_predicted']!r}/"
                        f"{max_tokens} fixed performance tokens"
                    )
    except Exception as exc:
        failures.append(f"{mode}: server benchmark error: {exc}")
    finally:
        terminate(proc)

    if mode == "board" and profile_board and results and not failures:
        profile_path = profile_dir / "et_runtime_trace.json"
        kernel_map_path = profile_dir / "kernel_id.json"
        try:
            measurements, profile_summary = read_firmware_cycles(
                profile_path,
                kernel_map_path,
                [
                    {
                        "request_id": item["request_id"],
                        "started_ns": item["started_ns"],
                        "ended_ns": item["ended_ns"],
                    }
                    for item in results
                ],
                performance,
            )
            for result in results:
                result["firmware"] = measurements[result["request_id"]]
            compressed = profile_path.with_suffix(".json.gz")
            with profile_path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            profile_path.unlink()
        except Exception as exc:
            failures.append(f"board: invalid firmware profile: {exc}")

    response_path.write_text(json.dumps(results, indent=2) + "\n")
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    allowed_ops = {str(value) for value in server_cfg.get("allowed_vision_fallback_ops", [])}
    failures.extend(log_failures(log, mode=mode, request_count=len(specs), allowed_ops=allowed_ops))
    return results, failures, unsupported_vision_ops(log), profile_summary


def correctness_failures(
    cases: list[tuple[dict[str, Any], list[Path]]],
    host_results: list[dict[str, Any]],
    board_results: list[dict[str, Any]],
    order_pair: list[str],
) -> list[str]:
    failures: list[str] = []
    host = {str(item["case"]): item for item in host_results if item["kind"] == "correctness"}
    board = {str(item["case"]): item for item in board_results if item["kind"] == "correctness"}
    for case, _ in cases:
        name = str(case["name"])
        accepted = {normalize_answer(str(value)) for value in case["accepted_answers"]}
        if name not in host or name not in board:
            failures.append(f"{name}: missing host or board correctness result")
            continue
        host_answer = str(host[name]["normalized_answer"])
        board_answer = str(board[name]["normalized_answer"])
        if host_answer not in accepted:
            failures.append(f"{name}: host answer {host_answer!r} is not accepted")
        if board_answer not in accepted:
            failures.append(f"{name}: board answer {board_answer!r} is not accepted")
        if board_answer != host_answer:
            failures.append(f"{name}: board answer {board_answer!r} != host reference {host_answer!r}")

    if len(order_pair) == 2 and all(name in board for name in order_pair):
        first = str(board[order_pair[0]]["normalized_answer"])
        second = str(board[order_pair[1]]["normalized_answer"])
        if first == second:
            failures.append(f"image-order check failed: both cases returned {first!r}")
    return failures


def run_cpu_perplexity(
    *,
    ppl_bin: Path,
    mcfg: dict[str, Any],
    server_cfg: dict[str, Any],
    model_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    ppl_cfg = server_cfg["perplexity"]
    corpus_id = str(ppl_cfg["corpus_artifact"])
    corpus_path = materialize_artifact(mcfg, corpus_id)
    expected_sha = artifact_config(mcfg, corpus_id).get("sha256")
    failures: list[str] = []
    if expected_sha and sha256_file(corpus_path) != expected_sha:
        return {}, ["trusted CPU perplexity corpus hash mismatch"]
    command = [
        str(ppl_bin),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "-ngl",
        "0",
        "-c",
        str(ppl_cfg["ctx_size"]),
        "-b",
        str(ppl_cfg["batch_size"]),
        "-ub",
        str(ppl_cfg["ubatch_size"]),
        "--chunks",
        str(ppl_cfg["chunks"]),
        "--no-warmup",
    ]
    (run_dir / "cpu-perplexity-command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        value
        for value in (str(ppl_bin.parent), os.environ.get("LLAMA_CPP_ET_LD_LIBRARY_PATH", ""))
        if value
    )
    try:
        proc = subprocess.run(
            command,
            cwd=ppl_bin.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(ppl_cfg.get("timeout_s", 300)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {}, ["trusted CPU llama-perplexity timed out"]
    (run_dir / "cpu-perplexity.log").write_text(proc.stdout)
    metrics = parse_perplexity_log(proc.stdout)
    if proc.returncode != 0:
        failures.append(f"trusted CPU llama-perplexity exited rc={proc.returncode}")
    if not isinstance(metrics.get("perplexity"), float):
        failures.append("trusted CPU llama-perplexity did not emit a final estimate")
    return metrics, failures


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_perplexity(perplexity_contract: dict[str, Any]) -> float:
    calculated = float(perplexity_contract["first_run_perplexity"]) * (
        1.0 + float(perplexity_contract["max_relative_regression"])
    )
    configured = float(perplexity_contract["maximum_perplexity"])
    if not math.isclose(calculated, configured, rel_tol=0, abs_tol=1e-9):
        raise ValueError("protected first-run PPL ceiling is internally inconsistent")
    return configured


def artifact_policy_error(
    actual_path: Path,
    baseline: dict[str, Any],
    *,
    kind: str,
) -> str | None:
    actual_sha = sha256_file(actual_path)
    if actual_sha == baseline["sha256"] and actual_path.stat().st_size == int(baseline["size_bytes"]):
        return None
    return f"candidate {kind} does not match the frozen baseline artifact"


def validate_contract(
    contract: dict[str, Any],
    mcfg: dict[str, Any],
    model_path: Path,
    mmproj_path: Path,
    fixture_paths: dict[str, list[Path]],
) -> None:
    if contract.get("schema_version") != 2:
        raise ValueError("unsupported SmolVLM2 validation contract")
    multimodal = mcfg["multimodal"]
    input_contract = contract["input_contract"]
    performance = contract["performance"]
    score_cfg = mcfg.get("score", {})
    if score_cfg.get("metric") != performance["metric"] or score_cfg.get("higher_is_better") is not False:
        raise ValueError("leaderboard metric differs from the protected firmware-cycle contract")
    for key in ("frames_per_second", "prompt_template", "cases", "order_pair"):
        if multimodal.get(key) != input_contract.get(key):
            raise ValueError(f"multimodal {key} differs from the protected contract")

    server_cfg = mcfg["llama_server"]
    correctness = contract["correctness"]
    if server_cfg.get("require_full_offload") is not True or correctness["require_full_offload"] is not True:
        raise ValueError("full ET offload requirement is not enabled")
    if (
        server_cfg.get("require_zero_vision_fallbacks") is not True
        or correctness["require_zero_vision_fallbacks"] is not True
    ):
        raise ValueError("zero vision fallback requirement is not enabled")
    allowed = [str(value) for value in server_cfg.get("allowed_vision_fallback_ops", [])]
    if allowed != correctness["allowed_vision_fallback_ops"]:
        raise ValueError("allowed vision fallback ops differ from the protected contract")

    if contract.get("candidate_policy") != {"artifacts_must_match_baseline": True}:
        raise ValueError("SmolVLM2 candidate artifacts must be frozen")
    baseline = contract["baseline_artifacts"]
    for error in (
        artifact_policy_error(model_path, baseline["model"], kind="model"),
        artifact_policy_error(
            mmproj_path,
            baseline["projector"],
            kind="projector",
        ),
    ):
        if error:
            raise ValueError(error)

    fixture_contract = input_contract["fixtures"]
    for fixture_id, paths in fixture_paths.items():
        fixture = fixture_contract[fixture_id]
        expected_frames = fixture.get("frames") or [fixture]
        if len(paths) != len(expected_frames):
            raise ValueError(f"fixture {fixture_id} has the wrong number of frames")
        for path, expected in zip(paths, expected_frames):
            if sha256_file(path) != expected["sha256"]:
                raise ValueError(f"fixture hash mismatch: {path}")

    ppl_contract = contract["quality"]["perplexity"]
    ppl_cfg = server_cfg.get("perplexity", {})
    maximum_ppl = maximum_perplexity(ppl_contract)
    expected_ppl_config = {
        "enabled": True,
        "corpus_artifact": ppl_contract["corpus_artifact"],
        "ctx_size": ppl_contract["context_size"],
        "batch_size": ppl_contract["batch_size"],
        "ubatch_size": ppl_contract["ubatch_size"],
        "chunks": ppl_contract["chunks"],
        "max_ppl": maximum_ppl,
    }
    for key, value in expected_ppl_config.items():
        if ppl_cfg.get(key) != value:
            raise ValueError(f"perplexity {key} differs from the protected first-run contract")


def resolve_fixtures(
    contract: dict[str, Any],
    mcfg: dict[str, Any],
) -> tuple[dict[str, list[Path]], list[tuple[dict[str, Any], list[Path]]]]:
    fixture_paths: dict[str, list[Path]] = {}
    for fixture_id, fixture in contract["input_contract"]["fixtures"].items():
        if fixture.get("artifact"):
            fixture_paths[fixture_id] = [materialize_artifact(mcfg, str(fixture["artifact"]))]
        else:
            fixture_paths[fixture_id] = [REPO_ROOT / str(frame["path"]) for frame in fixture["frames"]]
    cases = [
        (
            case,
            [path for fixture_id in case["fixtures"] for path in fixture_paths[str(fixture_id)]],
        )
        for case in contract["input_contract"]["cases"]
    ]
    return fixture_paths, cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["models"][args.model]
    server_cfg = mcfg["llama_server"]
    multimodal_cfg = mcfg["multimodal"]
    run_dir = Path(args.results_dir)
    score_path = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    score = score_common(args.model, str(mcfg["canonical_variant"]))

    try:
        contract_path = REPO_ROOT / str(mcfg["reference_contract"])
        contract = json.loads(contract_path.read_text())
        performance = contract["performance"]
        server_bin = materialize_artifact(mcfg, str(server_cfg["server_artifact"]))
        model_path = materialize_artifact(mcfg, str(server_cfg["model_artifact"]))
        mmproj_path = materialize_artifact(mcfg, str(server_cfg["mmproj_artifact"]))
        workdir = materialize_artifact(mcfg, str(server_cfg["workdir_artifact"]))
        fixture_paths, all_cases = resolve_fixtures(contract, mcfg)
        cases = selected_correctness_cases(all_cases, contract["correctness"])
        validate_contract(contract, mcfg, model_path, mmproj_path, fixture_paths)
        ensure_llama_cpp_build(mcfg, server_cfg, server_bin, None, workdir)
        host_server_bin = Path(os.environ.get("TRUSTED_SMOLVLM2_CPU_SERVER", str(server_bin)))
        cpu_ppl_bin = Path(
            os.environ.get(
                "TRUSTED_SMOLVLM2_CPU_PERPLEXITY",
                str(materialize_artifact(mcfg, str(server_cfg["perplexity"]["perplexity_artifact"]))),
            )
        )
    except Exception as exc:
        note = f"artifact setup failed: {exc}"
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0

    missing = [
        str(path)
        for path in [server_bin, host_server_bin, cpu_ppl_bin, model_path, mmproj_path]
        if not is_file(path)
    ]
    missing.extend(str(path) for paths in fixture_paths.values() for path in paths if not is_file(path))
    if not is_dir(workdir):
        missing.append(str(workdir))
    if missing:
        note = "missing required files: " + ", ".join(missing)
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0

    board_lock = Path(os.environ.get("BOARD_LOCK", "/var/lock/etsoc-shire0.lock"))
    board_lock.parent.mkdir(parents=True, exist_ok=True)
    ppl_metrics: dict[str, Any] = {}
    cpu_ppl_metrics: dict[str, Any] = {}
    ppl_failures: list[str] = []
    skip_host_reference = os.environ.get("SMOLVLM2_SKIP_HOST_REFERENCE") == "1"
    with open_board_lock(board_lock) as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if skip_host_reference:
            host_results: list[dict[str, Any]] = []
            host_failures: list[str] = []
        else:
            host_results, host_failures, _, _ = run_mode(
                mode="host",
                cmd=build_command(host_server_bin, model_path, mmproj_path, server_cfg, mode="host"),
                server_cfg=server_cfg,
                multimodal_cfg=multimodal_cfg,
                cases=cases,
                performance=performance,
                workdir=workdir,
                run_dir=run_dir,
                server_bin=host_server_bin,
                specs=correctness_specs(cases),
                run_label="host-correctness",
            )
            host_failures.extend(
                model_identity_failures(
                    (run_dir / "server-host-correctness.log").read_text(errors="replace"),
                    contract,
                )
            )
        reuse_measured = bool(performance.get("reuse_measured_request_for_correctness"))
        if reuse_measured:
            board_results: list[dict[str, Any]] = []
            board_failures: list[str] = []
            fallback_ops: set[str] = set()
        else:
            board_results, board_failures, fallback_ops, _ = run_mode(
                mode="board",
                cmd=build_command(server_bin, model_path, mmproj_path, server_cfg, mode="board"),
                server_cfg=server_cfg,
                multimodal_cfg=multimodal_cfg,
                cases=cases,
                performance=performance,
                workdir=workdir,
                run_dir=run_dir,
                server_bin=server_bin,
                specs=correctness_specs(cases),
                run_label="board-correctness",
            )
        profile_runs: list[dict[str, Any]] = []
        for kind, repetitions in (
            ("performance_warmup", int(performance["warmup_repetitions"])),
            ("performance_measured", int(performance["measured_repetitions"])),
        ):
            for repetition in range(repetitions):
                label = f"{kind.replace('_', '-')}-{repetition}"
                results, run_failures, run_fallbacks, profile = run_mode(
                    mode="board",
                    cmd=build_command(server_bin, model_path, mmproj_path, server_cfg, mode="board"),
                    server_cfg=server_cfg,
                    multimodal_cfg=multimodal_cfg,
                    cases=cases,
                    performance=performance,
                    workdir=workdir,
                    run_dir=run_dir,
                    server_bin=server_bin,
                    specs=[
                        performance_spec(
                            cases,
                            performance,
                            kind=kind,
                            repetition=repetition,
                        )
                    ],
                    run_label=label,
                    profile_board=kind == "performance_measured",
                )
                board_results.extend(results)
                board_failures.extend(run_failures)
                fallback_ops.update(run_fallbacks)
                if profile:
                    profile_runs.append({"run": label, **profile})
        profile_summary = {
            "counter": str(performance["primary_counter"]),
            "independent_measured_runs": len(profile_runs),
            "runs": profile_runs,
        }
        skip_ppl = os.environ.get("SMOLVLM2_SKIP_PPL") == "1"
        if not skip_ppl:
            ppl_env = runtime_env(server_bin, server_cfg, mode="board")
            ppl_env.pop("GGML_ET_PROFILE", None)
            ppl_metrics, ppl_failures = run_perplexity(
                mcfg=mcfg,
                lcfg=server_cfg,
                server_bin=server_bin,
                model_path=model_path,
                workdir=workdir,
                run_dir=run_dir,
                env=ppl_env,
            )
            cpu_ppl_metrics, cpu_ppl_failures = run_cpu_perplexity(
                ppl_bin=cpu_ppl_bin,
                mcfg=mcfg,
                server_cfg=server_cfg,
                model_path=model_path,
                run_dir=run_dir,
            )
            ppl_failures.extend(cpu_ppl_failures)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    failures = host_failures + board_failures + ppl_failures
    measured_results = [item for item in board_results if item["kind"] == "performance_measured"]
    if reuse_measured:
        board_correctness_results = [
            {**item, "kind": "correctness"}
            for item in measured_results
        ]
    else:
        board_correctness_results = [item for item in board_results if item["kind"] == "correctness"]
    if skip_host_reference:
        host_results = [{**item, "kind": "correctness"} for item in board_correctness_results]
        measured_log = run_dir / "server-performance-measured-0.log"
        failures.extend(model_identity_failures(measured_log.read_text(errors="replace"), contract))
    failures.extend(
        correctness_failures(
            cases,
            host_results,
            board_correctness_results,
            [
                str(value)
                for value in multimodal_cfg.get("order_pair", [])
                if str(value) in {str(case["name"]) for case, _ in cases}
            ],
        )
    )
    correctness_results = board_correctness_results
    measured_cycles = [int(item.get("firmware", {}).get("cycles", 0)) for item in measured_results]
    if len(measured_cycles) != int(performance["measured_repetitions"]) or any(
        value <= 0 for value in measured_cycles
    ):
        failures.append("board: missing complete measured firmware-cycle samples")
    pmc_cycles = statistics.median(measured_cycles) if measured_cycles else None
    measured_elapsed = [float(item["elapsed_s"]) for item in measured_results]
    median_elapsed = statistics.median(measured_elapsed) if measured_elapsed else None
    performance_frames = int(measured_results[0]["frame_count"]) if measured_results else 0
    fps = float(multimodal_cfg["frames_per_second"])
    represented_s = performance_frames / fps if performance_frames and fps > 0 else None
    real_time_factor = median_elapsed / represented_s if median_elapsed and represented_s else None

    accuracy = (
        sum(
            str(item["normalized_answer"])
            in {normalize_answer(str(value)) for value in case["accepted_answers"]}
            for (case, _), item in zip(cases, correctness_results)
        )
        / len(cases)
        if len(correctness_results) == len(cases) and cases
        else 0.0
    )
    host_correctness = [item for item in host_results if item["kind"] == "correctness"]
    host_agreement = None if skip_host_reference else (
        sum(
            host["normalized_answer"] == board["normalized_answer"]
            for host, board in zip(host_correctness, correctness_results)
        )
        / len(cases)
        if len(host_correctness) == len(correctness_results) == len(cases) and cases
        else 0.0
    )
    if accuracy < float(contract["correctness"]["minimum_task_accuracy"]):
        failures.append(f"task accuracy {accuracy:.4f} is below the protected minimum")
    if host_agreement is not None and host_agreement < float(
        contract["correctness"]["minimum_host_agreement"]
    ):
        failures.append(f"host agreement {host_agreement:.4f} is below the protected minimum")
    performance_case = str(performance["performance_case"])
    performance_definition = next(case for case, _ in cases if case["name"] == performance_case)
    accepted_performance_answers = {
        normalize_answer(str(value)) for value in performance_definition["accepted_answers"]
    }
    host_performance_answer = next(
        (
            str(item["normalized_answer"])
            for item in host_correctness
            if item["case"] == performance_case
        ),
        "",
    )
    for item in board_results:
        if item["kind"] not in {"performance_warmup", "performance_measured"}:
            continue
        answer = str(item["normalized_answer"])
        if answer not in accepted_performance_answers or answer != host_performance_answer:
            failures.append(
                f"{item['request_id']}: profiled answer {answer!r} does not match trusted "
                f"host answer {host_performance_answer!r}"
            )

    predicted_tokens = sum(int(item.get("tokens_predicted") or 0) for item in measured_results)
    predicted_ms = sum(float(item.get("timings", {}).get("predicted_ms") or 0.0) for item in measured_results)
    decode_tps = predicted_tokens / (predicted_ms / 1000.0) if predicted_ms > 0 else None
    ppl = ppl_metrics.get("perplexity")
    cpu_ppl = cpu_ppl_metrics.get("perplexity")
    first_run_ppl = float(contract["quality"]["perplexity"]["first_run_perplexity"])
    ppl_ratio = float(ppl) / first_run_ppl if isinstance(ppl, float) else None
    cpu_ppl_ratio = float(cpu_ppl) / first_run_ppl if isinstance(cpu_ppl, float) else None
    et_cpu_ppl_difference = (
        abs(float(ppl) - float(cpu_ppl)) / float(cpu_ppl)
        if isinstance(ppl, float) and isinstance(cpu_ppl, float) and cpu_ppl > 0
        else None
    )
    maximum_difference = float(
        contract["quality"]["perplexity"]["maximum_et_cpu_relative_difference"]
    )
    if not skip_ppl and (
        et_cpu_ppl_difference is None or et_cpu_ppl_difference > maximum_difference
    ):
        failures.append(
            "ET PPL does not agree with trusted CPU PPL within "
            f"{maximum_difference:.2%}"
        )
    maximum_ppl = float(contract["quality"]["perplexity"]["maximum_perplexity"])
    if not skip_ppl and (not isinstance(cpu_ppl, float) or cpu_ppl > maximum_ppl):
        failures.append(f"trusted CPU PPL must be <= {maximum_ppl:.4f}")
    quality_note = (
        "trusted paired-main PPL check skipped"
        if skip_ppl
        else f"PPL {float(ppl):.4f} <= {maximum_ppl:.4f}"
    )
    note = "; ".join(failures) if failures else (
        f"{len(cases)} public video/order cases passed; {quality_note}; "
        f"median firmware cycles {int(pmc_cycles):,}; vision fallback ops: none"
    )
    score.update(
        {
            "status": "fail" if failures else "pass",
            "passed": not failures,
            "pmc_cycles": pmc_cycles,
            "firmware_cycles_samples": measured_cycles,
            "firmware_profile": profile_summary,
            "real_time_factor": real_time_factor,
            "median_end_to_end_s": median_elapsed,
            "represented_duration_s": represented_s,
            "performance_frames": performance_frames,
            "task_accuracy": accuracy,
            "host_agreement": host_agreement,
            "tokens_per_second": decode_tps,
            "perplexity": ppl,
            "perplexity_error": ppl_metrics.get("perplexity_error"),
            "perplexity_tokens": ppl_metrics.get("perplexity_tokens"),
            "perplexity_first_run": first_run_ppl,
            "perplexity_ratio_to_first_run": ppl_ratio,
            "cpu_perplexity": cpu_ppl,
            "cpu_perplexity_error": cpu_ppl_metrics.get("perplexity_error"),
            "cpu_perplexity_ratio_to_first_run": cpu_ppl_ratio,
            "et_cpu_perplexity_relative_difference": et_cpu_ppl_difference,
            "perplexity_maximum": contract["quality"]["perplexity"]["maximum_perplexity"],
            "trusted_cpu_reference": bool(os.environ.get("TRUSTED_SMOLVLM2_CPU_SERVER")),
            "cpu_reference_executed": not skip_host_reference,
            "cpu_perplexity_reference_executed": not skip_ppl,
            "vision_fallback_ops": sorted(fallback_ops),
            "validation_contract_sha256": contract_sha256(contract_path),
            "valid_dump": not failures,
            "valid_note": note,
            "note": note,
        }
    )
    write_score(score_path, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
