#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / ".github" / "ci" / "scripts"))

from benchmark_config_helpers import load_config
from run_llama_server_benchmark import (
    ensure_llama_cpp_build,
    materialize_artifact,
    score_common,
    write_score,
    is_file,
    is_dir,
    artifact_has_env_override,
    cmake_cached_source,
    reset_stale_cmake_build
)

def ensure_ggonnx_build(mcfg: dict[str, Any], lcfg: dict[str, Any], ggonnx_runner_bin: Path, workdir: Path, llama_cpp_root: Path, llama_cpp_build: Path) -> None:
    source_artifact = mcfg.get("framework", {}).get("source_artifact", "ggonnx_source")
    source_dir = materialize_artifact(mcfg, str(source_artifact))
    if not is_dir(source_dir):
        return

    workdir_artifact = "ggonnx_build"
    workdir_override = artifact_has_env_override(mcfg, workdir_artifact, "GGONNX_WORKDIR")
    if workdir_override:
        if is_file(ggonnx_runner_bin):
            return
        cached_source = cmake_cached_source(workdir)
        if cached_source is not None and cached_source.resolve(strict=False) != source_dir.resolve(strict=False):
            raise RuntimeError(f"operator-provided ggonnx workdir {workdir} is configured from {cached_source}")
    else:
        reset_stale_cmake_build(workdir, source_dir)

    if is_file(ggonnx_runner_bin):
        return

    build_cfg = mcfg.get("artifacts", {}).get("ggonnx_build", {}).get("build", {})
    cmake = str(build_cfg.get("cmake", "cmake"))
    jobs = str(build_cfg.get("jobs", os.environ.get("GGONNX_BUILD_JOBS", os.cpu_count() or 4)))
    configure_args = [str(arg) for arg in build_cfg.get("configure_args", ["-DGGONNX_BUILD_TOOLS=ON", "-DCMAKE_BUILD_TYPE=Release"])]
    build_args = [str(arg) for arg in build_cfg.get("build_args", ["--config", "Release"])]
    
    # Inject llama.cpp paths
    configure_args.append(f"-DLLAMA_CPP_ROOT={llama_cpp_root}")
    configure_args.append(f"-DLLAMA_CPP_BUILD_DIR={llama_cpp_build}")

    workdir.parent.mkdir(parents=True, exist_ok=True)
    configure = [cmake, "-S", str(source_dir), "-B", str(workdir), *configure_args]
    build = [cmake, "--build", str(workdir), *build_args, "-j", jobs]
    
    print("$ " + " ".join(configure))
    subprocess.run(configure, check=True)
    print("$ " + " ".join(build))
    subprocess.run(build, check=True)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / ".github" / "ci" / "benchmark_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model not in cfg["models"]:
        print(f"error: model {args.model} not found in config", file=sys.stderr)
        return 1

    mcfg = cfg["models"][args.model]
    variant = mcfg.get("canonical_variant", "base")
    score = score_common(args.model, variant)

    try:
        cmake_dir = subprocess.run(
            ["bash", str(REPO_ROOT / ".github/ci/scripts/ensure_cmake.sh")],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        os.environ["PATH"] = f"{cmake_dir}:{os.environ.get('PATH', '')}"

        if "ET_PLATFORM" not in os.environ and "ET_INSTALL" in os.environ:
            os.environ["ET_PLATFORM"] = os.environ["ET_INSTALL"]

        llama_lcfg = {"workdir_artifact": "llama_cpp_build", "source_artifact": "llama_cpp_source"}
        llama_workdir = materialize_artifact(mcfg, "llama_cpp_build", fallback_env="LLAMA_CPP_ET_WORKDIR")
        llama_server_bin = materialize_artifact(mcfg, "llama_server", fallback_env="LLAMA_CPP_ET_SERVER")
        ensure_llama_cpp_build(mcfg, llama_lcfg, llama_server_bin, None, llama_workdir)
        
        llama_cpp_root = materialize_artifact(mcfg, "llama_cpp_source")

        ggonnx_workdir = materialize_artifact(mcfg, "ggonnx_build", fallback_env="GGONNX_WORKDIR")
        ggonnx_runner_bin = materialize_artifact(mcfg, "ggonnx_runner", fallback_env="GGONNX_RUNNER")
        ensure_ggonnx_build(mcfg, {}, ggonnx_runner_bin, ggonnx_workdir, llama_cpp_root, llama_workdir)

        model_path = materialize_artifact(mcfg, args.model)

        args.results_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.results_dir / "runner.log"
        
        print(f"$ {ggonnx_runner_bin} {model_path}")
        start_t = time.monotonic()
        with log_path.open("w") as f:
            proc = subprocess.run(
                [str(ggonnx_runner_bin), str(model_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "GGONNX_EP_PATH": str(ggonnx_workdir / "libggonnx_ep.so")}
            )
        elapsed = time.monotonic() - start_t
        score["elapsed_s"] = elapsed
        
        log_text = log_path.read_text(errors="replace")
        print(log_text)
        
        if proc.returncode == 0 and "Session created for model" in log_text:
            score["status"] = "pass"
            score["passed"] = True
            score["kernel_wait_s"] = elapsed
        else:
            score["status"] = "fail"
            score["note"] = f"Runner failed with code {proc.returncode}"

        write_score(args.output, score)
        return 0

    except Exception as exc:
        score["status"] = "error"
        score["note"] = str(exc)
        write_score(args.output, score)
        print(f"error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
