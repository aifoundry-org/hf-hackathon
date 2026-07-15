#!/usr/bin/env python3
"""Run a board-resident llama-server benchmark and write a leaderboard score."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_config_helpers import load_config as load_benchmark_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    return load_benchmark_config(path or CONFIG_PATH)


def env_value(name: str, default: Any) -> Any:
    return os.environ.get(name, default)


def model_local_path(mcfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    model_dir = mcfg.get("_model_dir")
    if model_dir:
        return REPO_ROOT / str(model_dir) / path
    return REPO_ROOT / path


def config_text(mcfg: dict[str, Any], cfg: dict[str, Any], key: str) -> str:
    file_key = f"{key}_file"
    if file_key in cfg:
        return model_local_path(mcfg, str(cfg[file_key])).read_text().strip()
    return str(cfg[key])


def artifact_config(mcfg: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    artifacts = mcfg.get("artifacts", {})
    artifact = artifacts.get(artifact_id)
    if not isinstance(artifact, dict):
        raise KeyError(f"unknown artifact {artifact_id!r}")
    return artifact


def artifact_env_names(artifact: dict[str, Any], fallback_env: str | None = None) -> list[str]:
    names: list[str] = []
    for value in (artifact.get("env"), fallback_env):
        if value:
            names.append(str(value))
    aliases = artifact.get("env_aliases", [])
    if isinstance(aliases, str):
        names.append(aliases)
    else:
        names.extend(str(alias) for alias in aliases)
    return list(dict.fromkeys(names))


def resolve_artifact_path(mcfg: dict[str, Any], artifact_id: str, fallback_env: str | None = None) -> Path:
    artifact = artifact_config(mcfg, artifact_id)
    if artifact.get("kind") == "framework_source":
        submodule_rel = artifact.get("submodule_path")
        if submodule_rel:
            committed = REPO_ROOT / str(submodule_rel)
            if (committed / ".git").exists() or (committed.is_dir() and any(committed.iterdir())):
                return committed
            raise RuntimeError(
                f"{artifact_id}: declared submodule {submodule_rel!r} is not initialized. "
                f"Run: git submodule update --init --recursive {submodule_rel}"
            )
    for env_name in artifact_env_names(artifact, fallback_env):
        if os.environ.get(env_name):
            return Path(os.environ[env_name])
    relative_to = artifact.get("relative_to")
    relative_path = artifact.get("relative_path")
    if relative_to and relative_path:
        return resolve_artifact_path(mcfg, str(relative_to)) / str(relative_path)
    for key in ("path", "default_path", "board_path"):
        value = artifact.get(key)
        if value:
            return Path(str(value))
    local_cache = artifact.get("local_cache")
    if local_cache:
        return REPO_ROOT / str(local_cache)
    raise KeyError(f"artifact {artifact_id!r} has no path/default_path")


def artifact_source_url(artifact: dict[str, Any]) -> str | None:
    if artifact.get("source_url"):
        return str(artifact["source_url"])
    source = artifact.get("source")
    if isinstance(source, dict) and source.get("url"):
        return str(source["url"])
    return None


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        part.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "hf-hackathon-board-ci"})
    print(f"Downloading {url} -> {dest}")
    written = 0
    last_report = 0
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if written - last_report >= 256 * 1024 * 1024:
                last_report = written
                print(f"  downloaded {written / (1024 ** 3):.2f} GiB")
    if expected_size is not None and written != expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download size mismatch for {dest}: {written} != {expected_size}")
    part.replace(dest)


def clone_git_artifact(artifact: dict[str, Any], dest: Path) -> None:
    source = artifact.get("source")
    if not isinstance(source, dict) or source.get("type") != "git" or not source.get("url"):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = source.get("branch")
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", str(branch)])
    cmd.extend([str(source["url"]), str(dest)])
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    revision = source.get("revision")
    if revision:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", str(revision)], check=False)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", str(revision)], check=True)


def materialize_artifact(mcfg: dict[str, Any], artifact_id: str, fallback_env: str | None = None) -> Path:
    artifact = artifact_config(mcfg, artifact_id)
    path = resolve_artifact_path(mcfg, artifact_id, fallback_env=fallback_env)
    expected_sha = artifact.get("sha256")
    env_override = any(os.environ.get(name) for name in artifact_env_names(artifact, fallback_env))
    if is_file(path):
        if expected_sha:
            actual_sha = sha256_file(path)
            if actual_sha == expected_sha:
                return path
            if env_override:
                raise RuntimeError(f"{artifact_id} sha256 mismatch: {actual_sha} != {expected_sha}")
            print(f"{artifact_id} sha256 mismatch in cache; redownloading {path}")
            path.unlink()
        else:
            return path
    if is_dir(path):
        return path
    if env_override:
        return path

    if artifact.get("kind") == "framework_source":
        raise RuntimeError(
            f"{artifact_id}: framework source must come from the committed submodule "
            f"declared via 'submodule_path'. URL-based clones are not permitted; "
            f"submitters commit their framework source as part of the PR."
        )
    else:
        url = artifact_source_url(artifact)
        if url:
            expected_size = artifact.get("size_bytes")
            download_file(url, path, int(expected_size) if expected_size is not None else None)

    if expected_sha and is_file(path):
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{artifact_id} sha256 mismatch: {actual_sha} != {expected_sha}")
    return path


def cmake_cached_source(workdir: Path) -> Path | None:
    cache = workdir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
            return Path(line.split("=", 1)[1])
    return None


def reset_stale_cmake_build(workdir: Path, source_dir: Path) -> None:
    cached_source = cmake_cached_source(workdir)
    if cached_source is None:
        return
    if cached_source.resolve(strict=False) == source_dir.resolve(strict=False):
        return
    print(
        "Removing stale llama.cpp CMake build cache: "
        f"{workdir} was configured from {cached_source}, current source is {source_dir}"
    )
    shutil.rmtree(workdir)


def artifact_has_env_override(mcfg: dict[str, Any], artifact_id: str, fallback_env: str | None = None) -> bool:
    artifact = artifact_config(mcfg, artifact_id)
    return any(os.environ.get(name) for name in artifact_env_names(artifact, fallback_env))


def ensure_llama_cpp_build(mcfg: dict[str, Any], lcfg: dict[str, Any], server_bin: Path, ppl_bin: Path | None, workdir: Path) -> None:
    source_artifact = lcfg.get("source_artifact") or mcfg.get("framework", {}).get("source_artifact")
    if not source_artifact:
        if is_file(server_bin) and (ppl_bin is None or is_file(ppl_bin)):
            return
        return
    source_dir = materialize_artifact(mcfg, str(source_artifact))
    if not is_dir(source_dir):
        return
    source_cfg = artifact_config(mcfg, str(source_artifact))
    source_revision = os.environ.get("LLAMA_CPP_ET_SOURCE_REVISION") or str(
        source_cfg.get("upstream", {}).get("revision") or ""
    )
    revision_stamp = workdir / ".hf-source-revision"
    workdir_artifact = str(lcfg.get("workdir_artifact", "llama_cpp_build"))
    workdir_override = artifact_has_env_override(mcfg, workdir_artifact, "LFM25_LLAMA_WORKDIR")
    binaries_ready = is_file(server_bin) and (ppl_bin is None or is_file(ppl_bin))
    stamped_revision = revision_stamp.read_text().strip() if revision_stamp.is_file() else ""
    reuse_trusted_build = os.environ.get("TRUSTED_LLAMA_REUSE_BUILD") == "1"
    if binaries_ready and source_revision and stamped_revision == source_revision:
        return
    if binaries_ready and source_revision and stamped_revision != source_revision:
        revision_keyed_workdir = workdir.name == f"build-{source_revision}"
        if (
            workdir_override
            and not os.environ.get("TRUSTED_LLAMA_BUILD_KEY")
            and not revision_keyed_workdir
        ):
            raise RuntimeError(
                f"operator-provided llama.cpp workdir {workdir} was built from "
                f"{stamped_revision or 'an unknown revision'}, not {source_revision}"
            )
        if reuse_trusted_build:
            print(
                f"Incrementally rebuilding trusted llama.cpp workdir from "
                f"{stamped_revision or 'unknown'} to {source_revision}"
            )
        else:
            print(
                f"Removing llama.cpp build for stale source revision "
                f"{stamped_revision or 'unknown'}; current revision is {source_revision}"
            )
            shutil.rmtree(workdir)
        binaries_ready = False
    if workdir_override:
        if binaries_ready:
            return
        cached_source = cmake_cached_source(workdir)
        if cached_source is not None and cached_source.resolve(strict=False) != source_dir.resolve(strict=False):
            raise RuntimeError(
                f"operator-provided llama.cpp workdir {workdir} is configured from {cached_source}, "
                f"not committed source {source_dir}; set LLAMA_CPP_ET_SERVER/LLAMA_CPP_ET_PERPLEXITY "
                "to existing binaries or choose a clean LLAMA_CPP_ET_WORKDIR"
            )
    else:
        reset_stale_cmake_build(workdir, source_dir)
    if binaries_ready:
        return
    build_cfg = artifact_config(mcfg, str(lcfg.get("workdir_artifact", "llama_cpp_build"))).get("build", {})
    cmake = str(build_cfg.get("cmake", "cmake"))
    jobs = str(build_cfg.get("jobs", os.environ.get("LLAMA_CPP_ET_BUILD_JOBS", os.cpu_count() or 4)))
    configure_args = [str(arg) for arg in build_cfg.get("configure_args", ["-DGGML_ET=ON", "-DCMAKE_BUILD_TYPE=Release"])]
    build_args = [str(arg) for arg in build_cfg.get("build_args", ["--config", "Release"])]
    build_targets = [str(target) for target in build_cfg.get("targets", [])]
    workdir.parent.mkdir(parents=True, exist_ok=True)
    configure = [cmake, "-S", str(source_dir), "-B", str(workdir), *configure_args]
    build = [cmake, "--build", str(workdir), *build_args]
    if build_targets:
        build.extend(["--target", *build_targets])
    build.extend(["-j", jobs])
    print("$ " + " ".join(configure))
    subprocess.run(configure, check=True)
    print("$ " + " ".join(build))
    subprocess.run(build, check=True)
    if source_revision:
        revision_stamp.write_text(source_revision + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_url() -> str:
    if os.environ.get("BENCHMARK_RUN_URL"):
        return os.environ["BENCHMARK_RUN_URL"]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "local")
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    return f"{server}/{repo}/actions/runs/{run_id}"


def score_common(model: str, variant: str, note: str = "") -> dict[str, Any]:
    return {
        "model": model,
        "variant": variant,
        "status": "fail",
        "passed": False,
        "kernel_wait_s": None,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": True,
        "valid_note": note,
        "emu_cycle_last": None,
        "elapsed_s": None,
        "note": note,
        "sha": os.environ.get("BENCHMARK_SHA") or os.environ.get("GITHUB_SHA", "local"),
        "ref": os.environ.get("BENCHMARK_REF") or os.environ.get("GITHUB_REF", "local"),
        "team": os.environ.get("LEADERBOARD_TEAM") or os.environ.get("GITHUB_ACTOR", "local"),
        "benchmark_device": os.environ.get("BENCHMARK_DEVICE", "unknown"),
        "run_url": run_url(),
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def write_score(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=4)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def wait_ready(proc: subprocess.Popen[bytes], log_path: Path, timeout_s: int) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_s
    last_size = 0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            text = log_path.read_text(errors="replace") if log_path.is_file() else ""
            return False, "llama-server exited before model loaded\n" + text[-4000:]
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if "main: model loaded" in text:
            return True, ""
        if len(text) != last_size:
            last_size = len(text)
        time.sleep(0.5)
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    return False, f"llama-server did not report model loaded within {timeout_s}s\n{text[-4000:]}"


def post_completion(url: str, payload: dict[str, Any], timeout_s: int) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": body}
        return exc.code, parsed


def validate_log(log: str, request_path: str, require_full_offload: bool = True) -> list[str]:
    failures: list[str] = []
    if "using device ET" not in log and "ET device 0" not in log:
        failures.append("ET device use not observed in server log")
    if require_full_offload:
        match = re.search(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU", log)
        if not match:
            failures.append("GPU layer offload summary not observed in server log")
        elif match.group(1) != match.group(2):
            failures.append(f"expected full GPU offload, got {match.group(1)}/{match.group(2)}")
    if f"done request: POST {request_path}" not in log:
        failures.append("completion request not observed in server log")
    return failures


def parse_perplexity_log(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)", text)
    if match:
        out["perplexity"] = float(match.group(1))
        out["perplexity_error"] = float(match.group(2))
    token_match = re.search(r"prompt eval time\s*=.*?/\s*([0-9]+)\s+tokens.*?([0-9.]+)\s+tokens per second", text)
    if token_match:
        out["perplexity_tokens"] = int(token_match.group(1))
        out["perplexity_prompt_tokens_per_second"] = float(token_match.group(2))
    return out


def run_perplexity(
    *,
    mcfg: dict[str, Any],
    lcfg: dict[str, Any],
    server_bin: Path,
    model_path: Path,
    workdir: Path,
    run_dir: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    pcfg = lcfg.get("perplexity", {})
    if not pcfg.get("enabled", False):
        return {}, []

    if pcfg.get("perplexity_artifact"):
        ppl_bin = materialize_artifact(
            mcfg,
            str(pcfg["perplexity_artifact"]),
            fallback_env="LFM25_LLAMA_PERPLEXITY",
        )
    else:
        ppl_bin = Path(str(pcfg.get("perplexity_bin") or (server_bin.parent / "llama-perplexity")))
    log_path = run_dir / "perplexity.log"
    corpus_path = run_dir / "perplexity-corpus.txt"
    command_path = run_dir / "perplexity-command.json"
    failures: list[str] = []
    if not is_file(ppl_bin):
        return {}, [f"missing llama-perplexity: {ppl_bin}"]

    if pcfg.get("corpus_artifact"):
        artifact_id = str(pcfg["corpus_artifact"])
        source = materialize_artifact(mcfg, artifact_id, fallback_env="WIKITEXT_RAW_PATH")
        if not is_file(source):
            return {}, [f"missing perplexity corpus artifact: {source}"]
        expected_sha = artifact_config(mcfg, artifact_id).get("sha256")
        if expected_sha and pcfg.get("verify_corpus_sha256", True):
            actual_sha = sha256_file(source)
            if actual_sha != expected_sha:
                return {}, [f"perplexity corpus sha256 mismatch: {actual_sha} != {expected_sha}"]
        corpus_path = source
    else:
        corpus = config_text(mcfg, pcfg, "corpus")
        repeat = int(pcfg.get("repeat", 1))
        corpus_path.write_text(((corpus + " ") * repeat).strip() + "\n")

    cmd = [
        str(ppl_bin),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "-dev",
        str(lcfg.get("device", "ET")),
        "-ngl",
        str(lcfg.get("gpu_layers", 99)),
        "-c",
        str(pcfg.get("ctx_size", 128)),
        "-b",
        str(pcfg.get("batch_size", 128)),
        "-ub",
        str(pcfg.get("ubatch_size", 128)),
        "--no-warmup",
    ]
    if pcfg.get("chunks") is not None:
        cmd.extend(["--chunks", str(pcfg["chunks"])])
    if pcfg.get("ppl_stride") is not None:
        cmd.extend(["--ppl-stride", str(pcfg["ppl_stride"])])
    for extra in pcfg.get("extra_args", []):
        cmd.append(str(extra))
    command_path.write_text(json.dumps(cmd, indent=2) + "\n")

    with log_path.open("wb") as log:
        log.write(("$ " + " ".join(cmd) + "\n\n").encode())
        log.flush()
        try:
            rc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(workdir),
                env=env,
                timeout=int(pcfg.get("timeout_s", 240)),
            ).returncode
        except subprocess.TimeoutExpired:
            return {}, [f"llama-perplexity timed out after {pcfg.get('timeout_s', 240)}s"]

    text = log_path.read_text(errors="replace")
    metrics = parse_perplexity_log(text)
    if rc != 0:
        failures.append(f"llama-perplexity exited rc={rc}")
    if "using device ET" not in text and "ET device 0" not in text:
        failures.append("ET device use not observed in perplexity log")
    if lcfg.get("require_full_offload", True):
        match = re.search(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU", text)
        if not match:
            failures.append("GPU layer offload summary not observed in perplexity log")
        elif match.group(1) != match.group(2):
            failures.append(f"expected full GPU offload in perplexity, got {match.group(1)}/{match.group(2)}")
    ppl = metrics.get("perplexity")
    if not isinstance(ppl, float):
        failures.append("missing final PPL estimate")
    else:
        min_ppl = float(pcfg.get("min_ppl", 0.0))
        max_ppl = float(pcfg.get("max_ppl", 1e9))
        if not (min_ppl <= ppl <= max_ppl):
            failures.append(f"PPL {ppl:.4f} outside [{min_ppl:.4f}, {max_ppl:.4f}]")
    return metrics, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    mcfg = cfg["models"][args.model]
    lcfg = mcfg["llama_server"]
    variant = mcfg["canonical_variant"]

    score_path = Path(args.output)
    run_dir = Path(args.results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    response_path = run_dir / "response.json"
    command_path = run_dir / "command.json"

    score = score_common(args.model, variant)

    try:
        if lcfg.get("server_artifact"):
            server_bin = materialize_artifact(mcfg, str(lcfg["server_artifact"]), fallback_env="LFM25_LLAMA_SERVER")
        else:
            server_bin = Path(str(env_value("LFM25_LLAMA_SERVER", lcfg["server_bin"])))
        if lcfg.get("model_artifact"):
            model_path = materialize_artifact(mcfg, str(lcfg["model_artifact"]))
        else:
            model_path = Path(str(env_value("LFM25_MODEL_PATH", lcfg["model_path"])))
        if lcfg.get("workdir_artifact"):
            workdir = materialize_artifact(mcfg, str(lcfg["workdir_artifact"]), fallback_env="LFM25_LLAMA_WORKDIR")
        else:
            workdir = Path(str(env_value("LFM25_LLAMA_WORKDIR", lcfg.get("workdir", server_bin.parent))))
    except Exception as exc:
        note = f"artifact setup failed: {exc}"
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0
    host = str(env_value("LFM25_HOST", lcfg.get("host", "127.0.0.1")))
    port = int(env_value("LFM25_PORT", lcfg.get("port", 18080)))
    device = str(env_value("LFM25_DEVICE", lcfg.get("device", "ET")))
    board_lock = Path(os.environ.get("BOARD_LOCK", "/var/lock/etsoc-shire0.lock"))
    ready_timeout_s = int(lcfg.get("ready_timeout_s", 120))
    request_timeout_s = int(lcfg.get("request_timeout_s", 240))
    min_completion_tokens = int(lcfg.get("min_completion_tokens", 1))

    ppl_bin = None
    pcfg = lcfg.get("perplexity", {})
    if pcfg.get("enabled", False):
        if pcfg.get("perplexity_artifact"):
            ppl_bin = resolve_artifact_path(mcfg, str(pcfg["perplexity_artifact"]), fallback_env="LFM25_LLAMA_PERPLEXITY")
        else:
            ppl_bin = Path(str(pcfg.get("perplexity_bin") or (server_bin.parent / "llama-perplexity")))
    try:
        ensure_llama_cpp_build(mcfg, lcfg, server_bin, ppl_bin, workdir)
    except Exception as exc:
        note = f"artifact setup failed: {exc}"
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0

    missing = []
    if not is_file(server_bin):
        missing.append(f"missing llama-server: {server_bin}")
    if not is_file(model_path):
        missing.append(f"missing model: {model_path}")
    if not is_dir(workdir):
        missing.append(f"missing workdir: {workdir}")
    if missing:
        score.update({"status": "skipped", "note": "; ".join(missing), "valid_note": "; ".join(missing)})
        write_score(score_path, score)
        return 0

    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "-dev",
        device,
        "-ngl",
        str(lcfg.get("gpu_layers", 99)),
        "-c",
        str(lcfg.get("ctx_size", 4096)),
        "-b",
        str(lcfg.get("batch_size", 512)),
        "-ub",
        str(lcfg.get("ubatch_size", 128)),
        "-np",
        str(lcfg.get("parallel", 1)),
        "--cache-ram",
        str(lcfg.get("cache_ram_mib", 0)),
    ]
    for extra in lcfg.get("extra_args", []):
        cmd.append(str(extra))

    command_path.write_text(json.dumps(cmd, indent=2) + "\n")

    api_mode = str(lcfg.get("api", "chat"))
    if api_mode == "completion":
        request_path = "/completion"
        request_payload = {
            "prompt": config_text(mcfg, lcfg, "prompt"),
            "n_predict": int(lcfg.get("max_tokens", 128)),
            "temperature": lcfg.get("temperature", 0),
        }
    else:
        request_path = "/v1/chat/completions"
        request_payload = {
            "model": model_path.name,
            "messages": [{"role": "user", "content": config_text(mcfg, lcfg, "prompt")}],
            "max_tokens": int(lcfg.get("max_tokens", 128)),
            "temperature": lcfg.get("temperature", 0),
        }
    if lcfg.get("ignore_eos") is not None:
        request_payload["ignore_eos"] = bool(lcfg.get("ignore_eos"))

    proc: subprocess.Popen[bytes] | None = None
    start = time.monotonic()
    lock_file = None
    ppl_metrics: dict[str, Any] = {}
    ppl_failures: list[str] = []
    try:
        board_lock.parent.mkdir(parents=True, exist_ok=True)
        lock_file = board_lock.open("a")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        env = os.environ.copy()
        env["TMPDIR"] = env.get("TMPDIR", "/dev/shm")
        lib_dir = str(server_bin.parent)
        env["LD_LIBRARY_PATH"] = str(env_value("LLAMA_CPP_ET_LD_LIBRARY_PATH", env_value("LFM25_LD_LIBRARY_PATH", lcfg.get("library_path", lib_dir))))
        if lcfg.get("flash_attn") is False:
            env["LLAMA_ARG_FLASH_ATTN"] = "off"

        with log_path.open("wb") as log:
            log.write(("$ " + " ".join(cmd) + "\n\n").encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(workdir), env=env)
            ready, ready_note = wait_ready(proc, log_path, ready_timeout_s)
            if not ready:
                score.update({"status": "fail", "note": ready_note, "valid_note": ready_note})
                write_score(score_path, score)
                return 0

            status, response = post_completion(
                f"http://{host}:{port}{request_path}",
                request_payload,
                request_timeout_s,
            )
            response_path.write_text(json.dumps(response, indent=2) + "\n")
            terminate(proc)
            proc = None
            ppl_metrics, ppl_failures = run_perplexity(
                mcfg=mcfg,
                lcfg=lcfg,
                server_bin=server_bin,
                model_path=model_path,
                workdir=workdir,
                run_dir=run_dir,
                env=env,
            )
    except Exception as exc:
        note = f"llama-server benchmark error: {exc}"
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0
    finally:
        terminate(proc)
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    elapsed = time.monotonic() - start
    log_text = log_path.read_text(errors="replace")
    response = json.loads(response_path.read_text())
    timings = response.get("timings", {})
    usage = response.get("usage", {})
    if api_mode == "completion":
        content = str(response.get("content") or "")
        completion_tokens_value = response.get("tokens_predicted")
        prompt_tokens_value = response.get("tokens_evaluated")
        usage = {
            "completion_tokens": completion_tokens_value,
            "prompt_tokens": prompt_tokens_value,
            "total_tokens": (
                completion_tokens_value + prompt_tokens_value
                if isinstance(completion_tokens_value, int) and isinstance(prompt_tokens_value, int)
                else None
            ),
        }
    else:
        choice = (response.get("choices") or [{}])[0]
        content = choice.get("message", {}).get("content", "")

    failures = []
    if status != 200:
        failures.append(f"HTTP {status}")
    if not content:
        failures.append("empty response content")
    success_substring = lcfg.get("success_substring")
    if success_substring and success_substring not in content:
        failures.append(f"response missing {success_substring!r}")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < min_completion_tokens:
        failures.append(f"completion_tokens {completion_tokens!r} < {min_completion_tokens}")
    tokens_per_second = timings.get("predicted_per_second")
    if not isinstance(tokens_per_second, (int, float)) or tokens_per_second <= 0:
        failures.append("missing positive predicted_per_second")
    failures.extend(validate_log(log_text, request_path, require_full_offload=bool(lcfg.get("require_full_offload", True))))
    failures.extend(ppl_failures)

    validation_contract_sha256 = None
    validation_contract = mcfg.get("reference_contract")
    if validation_contract:
        validation_contract_path = Path(str(validation_contract))
        if not validation_contract_path.is_absolute():
            validation_contract_path = REPO_ROOT / validation_contract_path
        if validation_contract_path.is_file():
            validation_contract_sha256 = sha256_file(validation_contract_path)
        else:
            failures.append(f"missing validation contract: {validation_contract_path}")

    passed = not failures
    ppl = ppl_metrics.get("perplexity")
    pass_note = "llama-server ET completion valid"
    if isinstance(ppl, float):
        pass_note += f"; PPL {ppl:.2f}"
    note = pass_note if passed else "; ".join(failures)
    score.update(
        {
            "status": "pass" if passed else "fail",
            "passed": passed,
            "tokens_per_second": float(tokens_per_second) if isinstance(tokens_per_second, (int, float)) else None,
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "perplexity": ppl_metrics.get("perplexity"),
            "perplexity_error": ppl_metrics.get("perplexity_error"),
            "perplexity_tokens": ppl_metrics.get("perplexity_tokens"),
            "perplexity_prompt_tokens_per_second": ppl_metrics.get("perplexity_prompt_tokens_per_second"),
            "elapsed_s": elapsed,
            "content_prefix": content[:200],
            "validation_contract_sha256": validation_contract_sha256,
            "valid_note": note,
            "note": note,
        }
    )
    write_score(score_path, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
