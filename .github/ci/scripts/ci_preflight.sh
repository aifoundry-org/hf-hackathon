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
  --target board --models "" --unregistered "smoketest" --sha x --ref y >/dev/null \
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

yolo_payload = bytes([128]) * 102400
yolo_fields = [
    0x10500001, 1, 1, 80, 80, 16, 4, 1,
    1, sum(yolo_payload), sum(yolo_payload), 1, 0, 16, 0, 0,
]
yolo_good = dump_with_summary(0x180000, yolo_fields, [(0x160000, yolo_payload)])
yolo_bad = bytearray(yolo_good)
yolo_bad[0x160000 + 17] = 127
yolo_good_score = run_score("yolo", write_results("yolo", "y10_00_base", yolo_good), "yolo-good")
yolo_bad_score = run_score("yolo", write_results("yolo", "y10_00_base", bytes(yolo_bad)), "yolo-bad")
assert yolo_good_score["passed"] and yolo_good_score["valid_accuracy"]
assert not yolo_bad_score["passed"] and not yolo_bad_score["valid_accuracy"]

whisper_expected = 2097152
whisper_fields = [
    0x57485350, 1, 1, 256, 64, 256, 1, 1,
    whisper_expected, whisper_expected, 1, 0, 0, 0, 0, 0,
]
whisper_bad_fields = list(whisper_fields)
whisper_bad_fields[8] = whisper_bad_fields[9] = whisper_expected + 1
whisper_good = dump_with_summary(0x2000, whisper_fields, [])
whisper_bad = dump_with_summary(0x2000, whisper_bad_fields, [])
whisper_good_score = run_score("whisper", write_results("whisper", "w10_00_base", whisper_good), "whisper-good")
whisper_bad_score = run_score("whisper", write_results("whisper", "w10_00_base", whisper_bad), "whisper-bad")
assert whisper_good_score["passed"] and whisper_good_score["valid_accuracy"]
assert not whisper_bad_score["passed"] and not whisper_bad_score["valid_accuracy"]
PY

step "Leaderboard gate renders"
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-pass.md" \
  --target board --models "" --unregistered "" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py no-op render failed"
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-fail.md" \
  --target board --models "" --unregistered "smoketest" --base-ref HEAD >/dev/null; then
  bad "leaderboard_gate.py should fail unregistered ports"
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
baseline = json.loads(Path("data/yolo.json").read_text())["entries"][0]
score = {
    "model": "yolo",
    "variant": baseline["variant"],
    "status": "pass",
    "passed": True,
    "kernel_wait_s": baseline["kernel_wait_s"] + 1.0,
    "valid_dump": True,
    "valid_accuracy": True,
    "valid_note": "dump valid; accuracy valid",
}
(tmp / "score-yolo.json").write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ci-only-pass.md" \
  --target board --models "yolo" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py should allow non-submission CI/scoring-only changes without runtime improvement"

step "Discord leaderboard announcement renders"
python3 - "$tmp" <<'PY' || bad "Discord leaderboard announcement render failed"
import json
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
tmp = Path(sys.argv[1])
sys.path.insert(0, str(root / ".github/ci/scripts"))

from merge_leaderboard import announcement_for_change

yolo_before = [{"team": "oldteam", "variant": "base", "kernel_wait_s": 0.200000, "sha": "old"}]
yolo_after = [{"team": "newteam", "variant": "fast", "kernel_wait_s": 0.100000, "sha": "new"}]
yolo_msg = announcement_for_change("yolo", yolo_before, yolo_after)
assert yolo_msg == (
    "yolo has been beaten by newteam: Kernel wait 0.100000s "
    "(was 0.200000s by oldteam)."
)
assert announcement_for_change("yolo", yolo_before, [dict(yolo_before[0])]) is None

llama_before = [{
    "team": "oldteam",
    "variant": "base",
    "tokens_per_second": 10.0,
    "perplexity": 12.0,
    "perplexity_error": 1.0,
    "sha": "old",
}]
llama_after = [{
    "team": "newteam",
    "variant": "fast",
    "tokens_per_second": 12.345,
    "perplexity": 11.0,
    "perplexity_error": 2.0,
    "sha": "new",
}]
llama_msg = announcement_for_change("llama32_1b", llama_before, llama_after)
assert "llama32_1b has been beaten by newteam" in llama_msg
assert "Decode tokens/s 12.35, PPL 11.00 (+/- 2.00)" in llama_msg

messages = tmp / "discord-messages.json"
messages.write_text(json.dumps([yolo_msg, llama_msg]) + "\n")
proc = subprocess.run(
    [
        sys.executable,
        str(root / ".github/ci/scripts/post_discord_webhook.py"),
        "--messages-file",
        str(messages),
        "--dry-run",
    ],
    cwd=root,
    check=True,
    text=True,
    stdout=subprocess.PIPE,
)
assert yolo_msg in proc.stdout
assert llama_msg in proc.stdout
PY
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
    --format space --unregistered-out "$(mktemp)" >/dev/null \
    || bad "changed_benchmark_models.py failed"
else
  note "no merge-base found; skipping selector run"
fi

if [[ "$fail" -ne 0 ]]; then
  printf '\npreflight FAILED — fix the above before pushing to CI.\n' >&2
  exit 1
fi
printf '\npreflight OK\n'
