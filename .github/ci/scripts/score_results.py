#!/usr/bin/env python3
"""Score sys-emu benchmark output for one model."""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config_helpers import load_config as load_benchmark_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"

DNCNN_MAGIC = 0xD3C11003
YOLO_MAGIC = 0x10500001
SUMMARY = struct.Struct("<16I")


def load_config() -> dict:
    return load_benchmark_config(CONFIG_PATH)


def wait_seconds(log_path: Path) -> float | None:
    text = log_path.read_text(errors="ignore")
    match = re.search(r"Kernel wait seconds:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def dump_summary(path: Path) -> dict:
    data = path.read_bytes()[: 0x1000 + SUMMARY.size]
    fields = SUMMARY.unpack_from(data, 0x1000)
    return {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
    }


def validate_dump(model: str, dump_path: Path | None, magic_cfg: str | None) -> tuple[bool, str]:
    if magic_cfg is None:
        return True, "no dump magic configured"
    if dump_path is None or not dump_path.is_file():
        return False, "missing dump.bin"
    magic_expected = int(magic_cfg, 0)
    s = dump_summary(dump_path)
    ok = (
        s["magic"] == magic_expected
        and s["done_count"] == s["active_harts"]
        and s["output_sum"] == s["slot_sum"]
    )
    if not ok:
        return False, (
            f"dump check failed magic=0x{s['magic']:x} "
            f"done={s['done_count']} harts={s['active_harts']} "
            f"out={s['output_sum']} slot={s['slot_sum']}"
        )
    return True, "dump valid"


def find_job_dir(results_dir: Path, model: str, variant: str) -> Path | None:
    jobs = results_dir / "jobs"
    if not jobs.is_dir():
        return None
    for job in sorted(jobs.iterdir()):
        if variant in job.name and model in job.name:
            return job
    return None


def score_from_results(
    model: str,
    results_dir: Path,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
) -> dict:
    cfg = load_config()
    mcfg = cfg["models"][model]
    variant = mcfg["canonical_variant"]
    magic_cfg = mcfg.get("dump_magic")

    results_tsv = results_dir / "results.tsv"
    if not results_tsv.is_file():
        return fail_payload(model, variant, sha, ref, actor, run_url, "missing results.tsv")

    row = None
    with results_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            if r.get("model") == model and r.get("variant") == variant:
                row = r
                break

    if row is None:
        return fail_payload(model, variant, sha, ref, actor, run_url, "canonical variant not in results.tsv")

    status = row.get("status", "")
    job_dir = find_job_dir(results_dir, model, variant)
    log_path_early = (results_dir / row["log"]) if row.get("log") else None
    if job_dir and (job_dir / "run.log").is_file():
        log_path_early = job_dir / "run.log"
    if status == "fail" and log_path_early and log_path_early.is_file():
        log_text = log_path_early.read_text(errors="ignore")
        if "kernel execution timed out" in log_text or "timed out" in log_text.lower():
            status = "timeout"
    log_path = (results_dir / row["log"]) if row.get("log") else None
    if job_dir and (job_dir / "run.log").is_file():
        log_path = job_dir / "run.log"
    dump_path = (results_dir / row["dump"]) if row.get("dump") else None
    if job_dir and (job_dir / "dump.bin").is_file():
        dump_path = job_dir / "dump.bin"

    kernel_wait = row.get("kernel_wait_s") or ""
    if not kernel_wait and log_path and log_path.is_file():
        w = wait_seconds(log_path)
        kernel_wait = f"{w:.6f}" if w is not None else ""

    valid_dump, valid_note = validate_dump(model, dump_path, magic_cfg)
    passed = status == "pass" and bool(kernel_wait) and valid_dump

    return {
        "model": model,
        "variant": variant,
        "status": "pass" if passed else status or "fail",
        "passed": passed,
        "kernel_wait_s": float(kernel_wait) if kernel_wait else None,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": valid_dump,
        "valid_note": valid_note,
        "emu_cycle_last": row.get("emu_cycle_last") or None,
        "elapsed_s": float(row["elapsed_s"]) if row.get("elapsed_s") else None,
        "note": row.get("note") or "",
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def fail_payload(
    model: str,
    variant: str,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
    note: str,
    status: str = "fail",
) -> dict:
    return {
        "model": model,
        "variant": variant,
        "status": status,
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
        "valid_dump": False,
        "valid_note": note,
        "emu_cycle_last": None,
        "elapsed_s": None,
        "note": note,
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status")
    parser.add_argument("--note", default="")
    parser.add_argument("--sha", default="local")
    parser.add_argument("--ref", default="local")
    parser.add_argument("--actor", default="local")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()

    if args.status:
        cfg = load_config()
        variant = cfg["models"][args.model]["canonical_variant"]
        payload = fail_payload(
            args.model,
            variant,
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
            args.note,
            status=args.status,
        )
    else:
        if not args.results_dir:
            print("error: --results-dir required unless --status is set", file=sys.stderr)
            return 2
        payload = score_from_results(
            args.model,
            Path(args.results_dir),
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
