from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .db import job_log_dir

ET_KERNEL_ERROR_RE = re.compile(
    r"ET [0-9a-fA-F:.]+: Error Event Detected|OPS Kernel Launch|CM Runtime|"
    r"MM2CMLaunch|KernelLaunch Failed|Execution error|illegal instruction|"
    r"Couldn.t dispatch event:"
)
ET_RUNTIME_ERROR_RE = re.compile(
    r"Stream error \(event|Kernel aborted \(event|Error on kernel launch:|"
    r"FATAL SIGNAL RECEIVED|Couldn.t dispatch event:|OPS Kernel Launch|"
    r"CM Runtime|MM2CMLaunch|KernelLaunch Failed|Execution error|illegal instruction"
)
SMOKE_DUMP_SIZE = 8192


def _log_path(job_id: str, stage: str) -> Path:
    return job_log_dir(job_id) / f"{stage}.log"


def parse_kernel_wait(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="ignore")
    m = re.search(r"Kernel wait seconds:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def _mark_board_quarantined(reason: str) -> None:
    path = Path(config.BOARD_QUARANTINE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_id_path.read_text().strip() if boot_id_path.is_file() else "unknown"
    path.write_text(
        "\n".join(
            [
                f"time_utc={datetime.now(timezone.utc).isoformat()}",
                f"host={os.uname().nodename}",
                f"boot_id={boot_id}",
                f"reason={reason}",
                "",
            ]
        )
    )


def _kernel_error_lines() -> list[str]:
    proc = subprocess.run(
        ["dmesg"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cannot inspect kernel log: {proc.stderr.strip()}")
    return [line for line in proc.stdout.splitlines() if ET_KERNEL_ERROR_RE.search(line)]


def _require_clean_board_state() -> None:
    quarantine = Path(config.BOARD_QUARANTINE_FILE)
    if quarantine.is_file():
        raise RuntimeError(
            "ET-SoC1 is quarantined; a maintainer must verify an external power cycle "
            f"and clear {quarantine}"
        )
    try:
        errors = _kernel_error_lines()
    except Exception as exc:
        _mark_board_quarantined(str(exc))
        raise
    if errors:
        reason = "kernel log already contains ET runtime or firmware errors"
        _mark_board_quarantined(reason)
        raise RuntimeError(f"{reason}: {errors[-1]}")


def _reject_board_errors(stage: str, log_path: Path | None = None) -> None:
    failures: list[str] = []
    if log_path is not None and log_path.is_file():
        text = log_path.read_text(errors="replace")
        match = ET_RUNTIME_ERROR_RE.search(text)
        if match:
            failures.append(f"runtime marker {match.group(0)!r}")
    try:
        kernel_errors = _kernel_error_lines()
    except Exception as exc:
        failures.append(str(exc))
        kernel_errors = []
    if kernel_errors:
        failures.append(kernel_errors[-1])
    if failures:
        reason = f"{stage} produced an ET runtime or firmware error: {'; '.join(failures)}"
        _mark_board_quarantined(reason)
        raise RuntimeError(reason)


def run_cmd(cmd: list[str], log_file: Path, env: dict[str, str] | None = None) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if config.ET_INSTALL:
        merged.setdefault("ET_PLATFORM", config.ET_INSTALL)
    if env:
        merged.update(env)
    with log_file.open("w") as lf:
        lf.write(f"$ {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=merged,
            cwd=str(config.REPO_ROOT) if config.REPO_ROOT.is_dir() else None,
        )
    return proc.returncode


def resolve_launcher() -> str:
    if config.LAUNCHER and Path(config.LAUNCHER).is_file():
        return config.LAUNCHER
    candidates = [
        config.REPO_ROOT / ".ci-work/build/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem",
        config.REPO_ROOT / "build/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        "LAUNCHER not set and erbium_soc1sim_argbuf_dynmem not built; "
        "set LAUNCHER or build .github/ci/launcher"
    )


def resolve_zero_bin() -> str:
    if config.ZERO_BIN and Path(config.ZERO_BIN).is_file():
        return config.ZERO_BIN
    for p in [
        config.MODEL_PORT_ARTIFACTS / "zero2m.bin",
        config.MODEL_PORT_ARTIFACTS / "common/zero2m.bin",
    ]:
        if p.is_file():
            return str(p)
    # minimal zeros file for smoke
    tmp = config.ARTIFACTS_DIR / "zero2m.bin"
    if not tmp.is_file():
        tmp.write_bytes(b"\0" * min(config.MEM_SIZE, 2 * 1024 * 1024))
    return str(tmp)


def run_dry_run(*, job_id: str, device: str, elf: str, stage: str) -> dict:
    log_file = _log_path(job_id, stage)
    log_file.write_text(
        f"ET_JOBS_DRY_RUN=1\n"
        f"device={device}\n"
        f"elf={elf}\n"
        f"Kernel wait seconds: 0.042\n",
        encoding="utf-8",
    )
    return {
        "returncode": 0,
        "kernel_wait_s": 0.042,
        "device": device,
        "elf": elf,
        "log": str(log_file),
        "dry_run": True,
    }


def run_kernel(
    *,
    job_id: str,
    device: str,
    elf: str,
    stage: str,
    extra_file_loads: list[str] | None = None,
    verify_zero_dump: bool = False,
) -> dict:
    if config.DRY_RUN:
        return run_dry_run(job_id=job_id, device=device, elf=elf, stage=stage)
    log_file = _log_path(job_id, stage)
    launcher = resolve_launcher()
    zero = resolve_zero_bin()
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    for lib_dir in [config.ET_LIB_PATH, str(Path(config.ET_INSTALL) / "lib")]:
        if lib_dir and lib_dir not in ld.split(":"):
            ld = f"{lib_dir}:{ld}" if ld else lib_dir

    cmd = [
        launcher,
        f"--device={device}",
        f"--elf-load={elf}",
        f"--shire={config.DEFAULT_SHIRE}",
        f"--file_load=0x0,{zero}",
        f"--timeout={config.KERNEL_TIMEOUT}",
        f"--mem_size={config.MEM_SIZE}",
    ]
    if extra_file_loads:
        for fl in extra_file_loads:
            cmd.append(f"--file_load={fl}")
    dump_path = log_file.with_suffix(".dump.bin")
    if verify_zero_dump:
        dump_path.unlink(missing_ok=True)
        cmd.extend(
            [
                f"--dump_after={dump_path}",
                f"--dump_size={SMOKE_DUMP_SIZE}",
            ]
        )

    lock = config.BOARD_LOCK
    if device == "soc1sim" and Path(lock).parent.is_dir():
        shell_cmd = f"flock -x -w 600 {shlex.quote(lock)} {shlex.join(cmd)}"
        rc = run_cmd(["bash", "-lc", shell_cmd], log_file, {"LD_LIBRARY_PATH": ld})
    else:
        rc = run_cmd(cmd, log_file, {"LD_LIBRARY_PATH": ld})

    if rc == 0 and verify_zero_dump:
        if not dump_path.is_file():
            with log_file.open("a") as log:
                log.write("error: deterministic smoke dump was not produced\n")
            rc = 1
        elif dump_path.read_bytes() != b"\0" * SMOKE_DUMP_SIZE:
            with log_file.open("a") as log:
                log.write("error: deterministic smoke dump does not match zero input\n")
            rc = 1

    wait_s = parse_kernel_wait(log_file)
    return {
        "returncode": rc,
        "kernel_wait_s": wait_s,
        "device": device,
        "elf": elf,
        "log": str(log_file),
    }


def run_sim_benchmark(job_id: str, model: str) -> dict:
    log_file = _log_path(job_id, "sim")
    script = config.REPO_ROOT / ".github/ci/scripts/run_model_benchmark.sh"
    if not script.is_file():
        return run_kernel(
            job_id=job_id,
            device="sys_emu",
            elf=config.SMOKE_ELF,
            stage="sim",
        )
    env = {
        "BENCHMARK_OUTPUT": str(config.ARTIFACTS_DIR / job_id / "benchmark"),
        "WORK_ROOT": str(config.ARTIFACTS_DIR / job_id / "work"),
    }
    rc = run_cmd(["bash", str(script), model], log_file, env)
    score_path = Path(env["BENCHMARK_OUTPUT"]) / f"score-{model}.json"
    score = {}
    if score_path.is_file():
        score = json.loads(score_path.read_text())
    return {"returncode": rc, "score": score, "log": str(log_file)}


def run_sim_smoke(job_id: str) -> dict:
    elf = config.SMOKE_ELF if config.DRY_RUN else _resolve_smoke_elf()
    return run_kernel(job_id=job_id, device="sys_emu", elf=elf, stage="sim")


def _resolve_smoke_elf() -> str:
    elf = config.SMOKE_ELF
    path = Path(elf)
    if not path.is_file():
        raise FileNotFoundError(f"SMOKE_ELF missing: {config.SMOKE_ELF}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != config.SMOKE_ELF_SHA256:
        raise RuntimeError(
            f"SMOKE_ELF sha256 {actual} != pinned {config.SMOKE_ELF_SHA256}"
        )
    return elf


def run_board_smoke(job_id: str) -> dict:
    if not config.DRY_RUN:
        _require_clean_board_state()
    elf = config.SMOKE_ELF if config.DRY_RUN else _resolve_smoke_elf()
    result = run_kernel(
        job_id=job_id,
        device="soc1sim",
        elf=elf,
        stage="board-smoke",
        verify_zero_dump=True,
    )
    if not config.DRY_RUN:
        log_path = Path(result["log"])
        if result["returncode"] != 0:
            _mark_board_quarantined("deterministic board smoke launcher failed")
        _reject_board_errors("deterministic board smoke", log_path)
    return result


def run_board_benchmark(job_id: str, model: str) -> dict:
    """Board path: build ELF + run via .github/ci/scripts/run_model_benchmark.sh on real silicon."""
    log_file = _log_path(job_id, "board")
    script = config.REPO_ROOT / ".github/ci/scripts/run_model_benchmark.sh"
    if not script.is_file():
        return run_board_benchmark_legacy(job_id, model)
    smoke = run_board_smoke(job_id)
    if smoke.get("returncode") != 0:
        return {"returncode": 2, "preflight": smoke, "score": {}, "log": smoke["log"]}
    env = {
        "BENCHMARK_DEVICE": "soc1sim",
        "BOARD_BENCHMARK": "1",
        "BENCHMARK_OUTPUT": str(config.ARTIFACTS_DIR / job_id / "benchmark"),
        "WORK_ROOT": str(config.ARTIFACTS_DIR / job_id / "work"),
        "BOARD_LOCK": config.BOARD_LOCK,
        "LAUNCHER": config.LAUNCHER,
        "ET_INSTALL": config.ET_INSTALL,
        "ET_PLATFORM": config.ET_PLATFORM or config.ET_INSTALL,
        "LD_LIBRARY_PATH": config.ET_LIB_PATH,
    }
    if config.ET_PLATFORM_SRC:
        env["ET_PLATFORM_SRC"] = config.ET_PLATFORM_SRC
    rc = run_cmd(["bash", str(script), model], log_file, env)
    score_path = Path(env["BENCHMARK_OUTPUT"]) / f"score-{model}.json"
    score: dict = {}
    if score_path.is_file():
        score = json.loads(score_path.read_text())
    if not config.DRY_RUN:
        _reject_board_errors(model, log_file)
    if not score.get("passed") and rc == 0:
        rc = 1
    return {"returncode": rc, "score": score, "log": str(log_file)}


def run_board_benchmark_legacy(job_id: str, model: str) -> dict:
    """Fallback: single-kernel run when CI scripts are missing."""
    smoke = run_board_smoke(job_id)
    if smoke.get("returncode") != 0:
        return {"returncode": 2, "preflight": smoke, "log": smoke["log"]}
    sys.path.insert(0, str(config.REPO_ROOT / ".github/ci/scripts"))
    from benchmark_config_helpers import load_config

    cfg = load_config(config.REPO_ROOT / ".github/ci/benchmark_config.json")
    variant = cfg["models"][model]["canonical_variant"]
    bench_dir = cfg["models"][model]["bench_dir"]
    elf_name = f"{variant}.elf"
    elf = config.MODEL_PORT_ARTIFACTS / bench_dir / elf_name
    if not elf.is_file():
        raise FileNotFoundError(f"ELF not found for board run: {elf_name}")
    result = run_kernel(
        job_id=job_id,
        device="soc1sim",
        elf=str(elf),
        stage="board",
    )
    if not config.DRY_RUN:
        _reject_board_errors(model, Path(result["log"]))
    return result


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--stage", choices=["sim", "board"], required=True)
    p.add_argument("--job-type", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()
    try:
        if args.stage == "sim":
            if args.job_type == "smoke_histogram":
                result = run_sim_smoke(args.job_id)
            else:
                result = run_sim_benchmark(args.job_id, args.model)
        else:
            if args.job_type == "smoke_histogram":
                result = run_board_smoke(args.job_id)
            else:
                result = run_board_benchmark(args.job_id, args.model)
        print(json.dumps(result, indent=2))
        score = result.get("score")
        if isinstance(score, dict) and score:
            return 0 if result.get("returncode") == 0 and score.get("passed") else 1
        return 0 if result.get("returncode") == 0 and result.get("kernel_wait_s") is not None else 1
    except Exception as exc:
        log = _log_path(args.job_id, args.stage)
        log.write_text(f"runner error: {exc}\n", encoding="utf-8")
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
