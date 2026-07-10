#!/usr/bin/env bash
# Local preflight for the parts of CI that run on GitHub-hosted runners:
# selector, PR-comment formatter, config/workflow validity. It does NOT run the
# board hardware benchmark (that needs the ET-SoC1 board), only the logic that
# can break a GitHub-hosted job. Wired as a git pre-push hook; also runnable by
# hand: .github/ci/scripts/ci_preflight.sh
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
fail=0
note() { printf '  %s\n' "$*"; }
step() { printf '==> %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }

step "Python syntax (.github/ci/scripts/*.py)"
if ! python3 -m py_compile .github/ci/scripts/*.py; then
  bad "python compile errors"
fi

step "Bash syntax (CI shell scripts)"
while IFS= read -r -d '' sh; do
  bash -n "$sh" || bad "bash -n: $sh"
done < <(find .github/ci -name '*.sh' -print0)

step "Workflow YAML parses"
for yml in .github/workflows/*.yml; do
  python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$yml" || bad "invalid YAML: $yml"
done

step "JSON config validity (benchmark_config + per-model + artifacts)"
python3 - <<'PY' || fail=1
import json, sys
from pathlib import Path
paths = [Path(".github/ci/benchmark_config.json")]
cfg = json.load(open(paths[0]))
for m in cfg.get("models", {}).values():
    if "config" in m:
        paths.append(Path(m["config"]))
paths += list(Path("ported_models").glob("*/artifacts.json"))
bad = False
for p in paths:
    try:
        json.load(open(p))
    except Exception as exc:
        print(f"FAIL: invalid JSON {p}: {exc}", file=sys.stderr); bad = True
sys.exit(1 if bad else 0)
PY

step "Config loads/expands for board + sysemu targets"
python3 .github/ci/scripts/benchmark_config_helpers.py --target board --models all --format space >/dev/null \
  || bad "board model list failed to resolve"
python3 .github/ci/scripts/benchmark_config_helpers.py --target sysemu --models all --format space >/dev/null \
  || bad "sysemu model list failed to resolve"

step "PR-comment formatter renders"
tmp="$(mktemp -d)"
python3 .github/ci/scripts/format_pr_comment.py --scores-dir "$tmp" --output "$tmp/c.md" \
  --target board --models "" --unregistered "smoketest" \
  --uncovered "ported_models/example/src/not_benchmarked.c" --sha x --ref y >/dev/null \
  || bad "format_pr_comment.py failed"

step "Score results accuracy gates"
python3 - "$tmp" <<'PY' || bad "score_results.py accuracy gates failed"
import json
import os
import struct
import subprocess
import sys
import hashlib
from pathlib import Path

root = Path.cwd()
tmp = Path(sys.argv[1])
assets = tmp / "assets"
scores = tmp / "scores"
test_config_path = tmp / "benchmark_config.json"
SUMMARY = struct.Struct("<16I")


def write_results(model: str, variant: str, dump: bytes) -> Path:
    run_dir = tmp / f"results-{model}"
    job = run_dir / "jobs" / f"0001_smoke_{model}_{variant}"
    job.mkdir(parents=True, exist_ok=True)
    (job / "run.log").write_text("Kernel wait seconds: 0.010000\n")
    (job / "dump.bin").write_bytes(dump)
    (run_dir / "results.tsv").write_text(
        "index\tsuite\tmodel\tvariant\tstatus\trc\telapsed_s\tkernel_wait_s\temu_cycle_last\tlog\tdump\tnote\n"
        f"1\tsmoke\t{model}\t{variant}\tpass\t0\t1.000\t0.010000\t\t"
        f"{(job / 'run.log').relative_to(run_dir)}\t{(job / 'dump.bin').relative_to(run_dir)}\t\n"
    )
    return run_dir


def write_case_results(model: str, variant: str, case_dumps: list[tuple[str, bytes]]) -> Path:
    run_dir = tmp / f"results-{model}-cases"
    rows = [
        "index\tsuite\tmodel\tcase\tvariant\tstatus\trc\telapsed_s\tkernel_wait_s\temu_cycle_last\tlog\tdump\tnote"
    ]
    for idx, (case, dump) in enumerate(case_dumps, start=1):
        job = run_dir / "jobs" / f"{idx:04d}_smoke_{model}_{case}_{variant}"
        job.mkdir(parents=True, exist_ok=True)
        (job / "run.log").write_text("Kernel wait seconds: 0.010000\n")
        (job / "dump.bin").write_bytes(dump)
        rows.append(
            f"{idx}\tsmoke\t{model}\t{case}\t{variant}\tpass\t0\t1.000\t0.010000\t\t"
            f"{(job / 'run.log').relative_to(run_dir)}\t{(job / 'dump.bin').relative_to(run_dir)}\t"
        )
    (run_dir / "results.tsv").write_text("\n".join(rows) + "\n")
    return run_dir


def dump_with_summary(size: int, fields: list[int], writes: list[tuple[int, bytes]]) -> bytes:
    data = bytearray(size)
    data[0x1000:0x1000 + SUMMARY.size] = SUMMARY.pack(*fields)
    for offset, payload in writes:
        data[offset:offset + len(payload)] = payload
    return bytes(data)


def run_score(model: str, run_dir: Path, name: str) -> dict:
    out = scores / f"{name}.json"
    env = os.environ.copy()
    env["BENCHMARK_ARTIFACT_ROOT"] = str(assets)
    env["BENCHMARK_CONFIG"] = str(test_config_path)
    subprocess.run(
        [
            sys.executable,
            str(root / ".github/ci/scripts/score_results.py"),
            "--model",
            model,
            "--results-dir",
            str(run_dir),
            "--output",
            str(out),
        ],
        cwd=root,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return json.loads(out.read_text())


dncnn_ref = bytes([7]) * (64 * 64)
test_config = json.loads((root / ".github/ci/benchmark_config.json").read_text())
test_config["models"]["dncnn"]["accuracy"]["expected_sha256"] = hashlib.sha256(dncnn_ref).hexdigest()
test_config_path.write_text(json.dumps(test_config) + "\n")
dncnn_fields = [
    0xD3C11003, 1, 1, 64, 64, 16, 5, 1,
    1, sum(dncnn_ref), sum(dncnn_ref), 1, 0, 0, 0, 0,
]
dncnn_good = dump_with_summary(0x12000, dncnn_fields, [(0x10000, dncnn_ref)])
dncnn_bad = bytearray(dncnn_good)
dncnn_bad[0x10000] = 9
dncnn_good_score = run_score("dncnn", write_results("dncnn", "int8_tfma_8hart", dncnn_good), "dncnn-good")
dncnn_bad_score = run_score("dncnn", write_results("dncnn", "int8_tfma_8hart", bytes(dncnn_bad)), "dncnn-bad")
assert dncnn_good_score["passed"] and dncnn_good_score["valid_accuracy"]
assert not dncnn_bad_score["passed"] and not dncnn_bad_score["valid_accuracy"]
assert hashlib.sha256(dncnn_ref).hexdigest()[:12] in dncnn_good_score["valid_note"]

yolo_cases = test_config["models"]["yolo"]["benchmark_cases"]
yolo_det_offset = int(yolo_cases[0]["accuracy"]["offset"], 0)
yolo_fields = [
    0x10500001, 1, 1, 480, 640, 3, 1, 1,
    1, 0, 0, 1, 0, 0, 0, 0,
]


def yolo_detection_payload(detections):
    payload = struct.pack("<I", len(detections))
    for det in detections:
        payload += struct.pack("<I5f", *det)
    return payload


def yolo_case_dump(case, *, fail_first=False):
    detections = []
    for idx, exp in enumerate(case["accuracy"]["expected"]):
        score = float(exp.get("min_score", 0.5)) + 0.10
        class_id = int(exp["class_id"])
        if fail_first and idx == 0:
            score = max(0.0, float(exp.get("min_score", 0.5)) - 0.20)
        detections.append((class_id, score, *[float(v) for v in exp["box"]]))
    return dump_with_summary(
        yolo_det_offset + 4096,
        yolo_fields,
        [(yolo_det_offset, yolo_detection_payload(detections))],
    )


yolo_det_good_cases = [
    (case["name"], yolo_case_dump(case))
    for case in yolo_cases
]
yolo_det_bad_cases = [
    (case["name"], yolo_case_dump(case, fail_first=(idx == 1)))
    for idx, case in enumerate(yolo_cases)
]
yolo_det_good_score = run_score(
    "yolo",
    write_case_results("yolo", "yolo_m30_vpus2", yolo_det_good_cases),
    "yolo-good",
)
yolo_det_bad_score = run_score(
    "yolo",
    write_case_results("yolo", "yolo_m30_vpus2", yolo_det_bad_cases),
    "yolo-bad",
)
assert yolo_det_good_score["passed"] and yolo_det_good_score["valid_accuracy"]
assert not yolo_det_bad_score["passed"] and not yolo_det_bad_score["valid_accuracy"]
assert "5/5 YOLO image cases valid" in yolo_det_good_score["valid_note"]
assert "coco_cat_524280" in yolo_det_bad_score["valid_note"]

PY

step "Leaderboard gate renders"
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-pass.md" \
  --target board --models "" --unregistered "" --uncovered "" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py no-op render failed"
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-fail.md" \
  --target board --models "" --unregistered "smoketest" --base-ref HEAD >/dev/null; then
  bad "leaderboard_gate.py should fail unregistered ports"
fi
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-uncovered-fail.md" \
  --target board --models "" --uncovered "ported_models/example/src/not_benchmarked.c" --base-ref HEAD >/dev/null; then
  bad "leaderboard_gate.py should fail uncovered runtime source changes"
fi
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

tmp = Path(sys.argv[1])
baseline = json.loads(Path("data/llama32_1b.json").read_text())["entries"][0]
score = {
    "model": "llama32_1b",
    "variant": baseline["variant"],
    "status": "pass",
    "passed": True,
    "tokens_per_second": baseline["tokens_per_second"] + 1.0,
    "perplexity": baseline["perplexity"] * 1.1,
}
(tmp / "score-llama32_1b.json").write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ppl-pass.md" \
  --target board --models "llama32_1b" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py should allow PPL within 20% of best seen"
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

tmp = Path(sys.argv[1])
path = tmp / "score-llama32_1b.json"
score = json.loads(path.read_text())
baseline = json.loads(Path("data/llama32_1b.json").read_text())["entries"][0]
score["perplexity"] = baseline["perplexity"] * 1.21
path.write_text(json.dumps(score) + "\n")
PY
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ppl-fail.md" \
  --target board --models "llama32_1b" --base-ref HEAD >/dev/null; then
  bad "leaderboard_gate.py should fail PPL more than 20% worse than best seen"
fi
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

tmp = Path(sys.argv[1])
baseline = json.loads(Path("data/dncnn.json").read_text())["entries"][0]
score = {
    "model": "dncnn",
    "variant": baseline["variant"],
    "status": "pass",
    "passed": True,
    "kernel_wait_s": baseline["kernel_wait_s"] + 1.0,
    "valid_dump": True,
    "valid_accuracy": True,
    "valid_note": "dump valid; accuracy valid",
}
(tmp / "score-dncnn.json").write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ci-only-pass.md" \
  --target board --models "dncnn" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py should allow non-submission CI/scoring-only changes without runtime improvement"
rm -rf "$tmp"

step "Leaderboard team resolver"
expected_author="$(git show -s --format=%an HEAD)"
resolved_push="$(GITHUB_EVENT_NAME=push GITHUB_REF=refs/heads/main GITHUB_ACTOR=ci-actor GH_TOKEN= GITHUB_TOKEN= \
  .github/ci/scripts/resolve_leaderboard_team.sh HEAD)"
if [[ "$resolved_push" != "$expected_author" ]]; then
  bad "resolve_leaderboard_team.sh push fallback returned '$resolved_push', expected '$expected_author'"
fi
resolved_pr="$(GITHUB_EVENT_NAME=pull_request GITHUB_REF=refs/pull/1/merge GITHUB_ACTOR=ci-actor GH_TOKEN= GITHUB_TOKEN= \
  .github/ci/scripts/resolve_leaderboard_team.sh HEAD)"
if [[ "$resolved_pr" != "ci-actor" ]]; then
  bad "resolve_leaderboard_team.sh PR fallback returned '$resolved_pr'"
fi

step "Selector runs against merge-base (if available)"
base="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base main HEAD 2>/dev/null || true)"
if [[ -n "$base" ]]; then
  python3 .github/ci/scripts/changed_benchmark_models.py --target board --base "$base" --head HEAD \
    --format space --unregistered-out "$(mktemp)" --uncovered-out "$(mktemp)" >/dev/null \
    || bad "changed_benchmark_models.py failed"
else
  note "no merge-base found; skipping selector run"
fi

step "Selector detects uncovered runtime code"
covered_out="$(mktemp)"
python3 .github/ci/scripts/changed_benchmark_models.py --target board \
  --changed-file ported_models/yolo/src/yolo_m30_argbuf.c \
  --format space --unregistered-out "$(mktemp)" --uncovered-out "$covered_out" >/dev/null \
  || bad "changed_benchmark_models.py covered YOLO source case failed"
if [[ -s "$covered_out" ]]; then
  bad "changed_benchmark_models.py marked configured YOLO benchmark source as uncovered"
fi

yolo_unvalidated_cfg="$(mktemp)"
python3 - "$yolo_unvalidated_cfg" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
cfg = json.loads(Path(".github/ci/benchmark_config.json").read_text())
cfg["models"]["yolo"].pop("validation", None)
cfg["models"]["yolo"]["accuracy"] = {
    "kind": "constant_u8",
    "offset": "0x160000",
    "count": 102400,
    "expected_value": 128,
    "max_abs": 0,
}
out.write_text(json.dumps(cfg))
PY
yolo_unvalidated_out="$(mktemp)"
python3 .github/ci/scripts/changed_benchmark_models.py --target board \
  --config "$yolo_unvalidated_cfg" \
  --changed-file ported_models/yolo/src/yolo_m30_argbuf.c \
  --format space --unregistered-out "$(mktemp)" --uncovered-out "$yolo_unvalidated_out" >/dev/null \
  || bad "changed_benchmark_models.py under-validated YOLO case failed"
if ! grep -qx 'ported_models/yolo/src/yolo_m30_argbuf.c' "$yolo_unvalidated_out"; then
  bad "changed_benchmark_models.py allowed YOLO source without real-image detection validation"
fi

yolo_cfg="$(mktemp)"
python3 - "$yolo_cfg" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
cfg = json.loads(Path(".github/ci/benchmark_config.json").read_text())
cfg["models"]["yolo"]["validation"] = {
    "kind": "yolo_real_image_detections",
    "image": "web_car",
    "source_shape": [480, 640, 3],
}
cfg["models"]["yolo"]["accuracy"] = {
    "kind": "yolo_detections",
    "offset": "0x01D00000",
    "max_detections": 64,
    "expected": [
        {
            "class_id": 2,
            "label": "car",
            "min_score": 0.55,
            "box": [4.6, 56.0, 505.5, 273.6],
            "min_iou": 0.70,
        }
    ],
}
out.write_text(json.dumps(cfg))
PY
yolo_validated_out="$(mktemp)"
python3 .github/ci/scripts/changed_benchmark_models.py --target board \
  --config "$yolo_cfg" \
  --changed-file ported_models/yolo/src/yolo_m30_argbuf.c \
  --format space --unregistered-out "$(mktemp)" --uncovered-out "$yolo_validated_out" >/dev/null \
  || bad "changed_benchmark_models.py real-image YOLO validation case failed"
if [[ -s "$yolo_validated_out" ]]; then
  bad "changed_benchmark_models.py rejected YOLO with real-image detection validation"
fi
rm -f "$covered_out" "$yolo_unvalidated_cfg" "$yolo_unvalidated_out" "$yolo_cfg" "$yolo_validated_out"

if [[ "$fail" -ne 0 ]]; then
  printf '\npreflight FAILED — fix the above before pushing to CI.\n' >&2
  exit 1
fi
printf '\npreflight OK\n'
