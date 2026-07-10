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
python3 -m py_compile ported_models/yolo/tools/host_reference.py \
  || bad "YOLO host-reference compile errors"

step "Bash syntax (CI shell scripts)"
while IFS= read -r -d '' sh; do
  bash -n "$sh" || bad "bash -n: $sh"
done < <(find .github/ci -name '*.sh' -print0)

step "Workflow YAML parses"
for yml in .github/workflows/*.yml; do
  python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$yml" || bad "invalid YAML: $yml"
done
if ! grep -qF 'uses: aifoundry-org/hf-hackathon/.github/workflows/benchmark-board.yml@main' \
  .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO PR caller is not pinned to the main-owned reusable workflow"
fi
if ! grep -qF 'pull_request_target:' .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO caller must be loaded from the default branch"
fi
if ! grep -qF 'run-name: "Trusted YOLO PR #' .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO run name must retain the PR number and head SHA"
fi
if ! grep -qF 'context=trusted-yolo/main-gate' \
  .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO caller does not publish its merge status on the participant commit"
fi
if grep -qE '^[[:space:]]+paths:' .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO final check must run on every PR so it can be required"
fi
if ! grep -qF 'Trusted YOLO leaderboard gate' \
  .github/workflows/benchmark-board.yml; then
  bad "trusted YOLO reusable workflow has no stable final check name"
fi

step "Trusted YOLO tree applies only implementation paths"
python3 - <<'PY' || bad "trusted YOLO overlay isolation failed"
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path.cwd()
helper = root / ".github/ci/scripts/prepare_trusted_yolo_tree.py"


def run(repo, *args, check=True):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    return subprocess.run(
        args,
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def initialize(repo):
    run(repo.parent, "git", "init", "-q", str(repo))
    run(repo, "git", "config", "user.name", "CI")
    run(repo, "git", "config", "user.email", "ci@example.com")
    (repo / ".gitattributes").write_text("*.bin binary\n")
    (repo / ".github/ci/reference").mkdir(parents=True)
    (repo / "ported_models/yolo/src").mkdir(parents=True)
    (repo / "ported_models/yolo/assets/yolo").mkdir(parents=True)
    (repo / ".github/ci/reference/yolo.json").write_text('{"trusted": true}\n')
    (repo / "ported_models/yolo/src/kernel.c").write_text("int kernel(void) { return 1; }\n")
    (repo / "ported_models/yolo/assets/yolo/weights_region.bin").write_bytes(b"main\0weights")
    run(repo, "git", "add", ".")
    run(repo, "git", "commit", "-q", "-m", "main")
    return run(repo, "git", "rev-parse", "HEAD").stdout.strip()


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    repo = tmp / "repo"
    base = initialize(repo)
    run(repo, "git", "switch", "-q", "-c", "participant")
    # Formatting is participant-owned and must not block an otherwise valid
    # implementation overlay.
    (repo / "ported_models/yolo/src/kernel.c").write_text("int kernel(void) { return 2; } \n")
    (repo / "ported_models/yolo/src/fused.inc").write_text("static const int fused = 1;\n")
    (repo / "ported_models/yolo/src/build.sh").write_text("exit 99\n")
    (repo / "ported_models/yolo/assets/yolo/weights_region.bin").write_bytes(b"fused\0weights")
    (repo / ".github/ci/reference/yolo.json").write_text('{"trusted": false}\n')
    run(repo, "git", "add", ".")
    run(repo, "git", "commit", "-q", "-m", "submission")
    head = run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    run(repo, "git", "switch", "-q", "--detach", base)
    (repo / "ported_models/yolo/src/main_only.h").write_text("#define MAIN_ONLY 1\n")
    run(repo, "git", "add", ".")
    run(repo, "git", "commit", "-q", "-m", "main advanced")
    latest_main = run(repo, "git", "rev-parse", "HEAD").stdout.strip()

    metadata = tmp / "metadata.json"
    result = run(
        repo,
        sys.executable,
        str(helper),
        "--repo",
        str(repo),
        "--base",
        latest_main,
        "--head",
        head,
        "--main",
        latest_main,
        "--metadata",
        str(metadata),
    )
    payload = json.loads(metadata.read_text())
    assert payload["applied_paths"] == [
        "ported_models/yolo/assets/yolo/weights_region.bin",
        "ported_models/yolo/src/fused.inc",
        "ported_models/yolo/src/kernel.c",
    ]
    assert ".github/ci/reference/yolo.json" in payload["ignored_paths"]
    assert "ported_models/yolo/src/build.sh" in payload["ignored_paths"]
    assert payload["participant_merge_base_sha"] == base
    assert json.loads((repo / ".github/ci/reference/yolo.json").read_text())["trusted"]
    assert not (repo / "ported_models/yolo/src/build.sh").exists()
    assert (repo / "ported_models/yolo/src/main_only.h").is_file()
    assert (repo / "ported_models/yolo/assets/yolo/weights_region.bin").read_bytes() == b"fused\0weights"

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    repo = tmp / "repo"
    base = initialize(repo)
    run(repo, "git", "switch", "-q", "-c", "participant")
    os.symlink("/etc/passwd", repo / "ported_models/yolo/src/linked.h")
    run(repo, "git", "add", ".")
    run(repo, "git", "commit", "-q", "-m", "symlink")
    head = run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    run(repo, "git", "switch", "-q", "--detach", base)
    result = run(
        repo,
        sys.executable,
        str(helper),
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--main",
        base,
        check=False,
    )
    assert result.returncode != 0
    assert "regular non-executable" in result.stderr
PY

step "JSON config validity (benchmark_config + per-model + artifacts)"
python3 - <<'PY' || fail=1
import json, sys
from pathlib import Path
paths = [
    Path(".github/ci/benchmark_config.json"),
    Path(".github/ci/reference/yolo.json"),
]
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

step "YOLO reference contract and public COCO fixtures"
python3 - <<'PY' || bad "YOLO reference contract does not match benchmark fixtures"
import hashlib
import json
from pathlib import Path

contract = json.loads(Path(".github/ci/reference/yolo.json").read_text())
config = json.loads(Path(".github/ci/benchmark_config.json").read_text())
model = config["models"]["yolo"]
assert model["reference_contract"] == ".github/ci/reference/yolo.json"
fixtures = {item["name"]: item for item in contract["fixtures"]["cases"]}
cases = {item["name"]: item for item in model["benchmark_cases"]}
assert len(fixtures) == len(cases) >= 5
assert fixtures.keys() == cases.keys()
for name, fixture in fixtures.items():
    assert fixture["source_url"].startswith("http://images.cocodataset.org/val2017/")
    asset = Path(fixture["asset"])
    assert asset.is_file(), asset
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == fixture["asset_sha256"]
    expected_load = str(asset).removeprefix("ported_models/yolo/assets/")
    loads = cases[name]["file_loads"]
    assert any(expected_load in load.get("paths", []) for load in loads)
    accuracy = cases[name]["accuracy"]
    assert accuracy["kind"] == "yolo_reference_detections"
    assert accuracy["reference_case"] == name
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
import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
tmp = Path(sys.argv[1])
assets = tmp / "assets"
scores = tmp / "scores"
test_config_path = tmp / "benchmark_config.json"
yolo_reference_path = tmp / "yolo-host-reference.json"
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
    env["YOLO_HOST_REFERENCE_JSON"] = str(yolo_reference_path)
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


test_config = json.loads((root / ".github/ci/benchmark_config.json").read_text())
test_config_path.write_text(json.dumps(test_config) + "\n")

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


contract_path = root / test_config["models"]["yolo"]["reference_contract"]
contract = json.loads(contract_path.read_text())
fixture_by_name = {case["name"]: case for case in contract["fixtures"]["cases"]}
reference_cases = {}
for idx, case in enumerate(yolo_cases):
    name = case["name"]
    reference_cases[name] = {
        "image_id": fixture_by_name[name]["image_id"],
        "input_sha256": fixture_by_name[name]["asset_sha256"],
        "detections": [
            {
                "class_id": idx,
                "label": f"class_{idx}",
                "score": 0.9,
                "box": [10.0 + idx, 20.0, 110.0 + idx, 120.0],
            }
        ],
    }
yolo_reference_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "reference": {
                "checkpoint_sha256": contract["model"]["source"]["sha256"],
            },
            "agreement": contract["agreement"],
            "cases": reference_cases,
        }
    )
    + "\n"
)


def yolo_case_dump(case, *, fail=False, add_extra=False):
    expected = reference_cases[case["name"]]["detections"]
    detections = [] if fail else [
        (item["class_id"], item["score"], *item["box"])
        for item in expected
    ]
    if add_extra:
        detections.append((79, 0.9, 200.0, 200.0, 250.0, 250.0))
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
    (case["name"], yolo_case_dump(case, fail=(idx == 1)))
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
yolo_det_extra_score = run_score(
    "yolo",
    write_case_results(
        "yolo",
        "yolo_m30",
        [
            (case["name"], yolo_case_dump(case, add_extra=(idx == 2)))
            for idx, case in enumerate(yolo_cases)
        ],
    ),
    "yolo-extra",
)
assert yolo_det_good_score["passed"] and yolo_det_good_score["valid_accuracy"]
assert not yolo_det_bad_score["passed"] and not yolo_det_bad_score["valid_accuracy"]
assert not yolo_det_extra_score["passed"] and not yolo_det_extra_score["valid_accuracy"]
assert yolo_det_good_score["accuracy_precision"] == 1.0
assert yolo_det_good_score["accuracy_recall"] == 1.0
assert "5/5 YOLO COCO cases valid" in yolo_det_good_score["valid_note"]
assert "coco_cat_524280" in yolo_det_bad_score["valid_note"]

test_config["models"]["yolo"]["benchmark_cases"] = []
test_config_path.write_text(json.dumps(test_config) + "\n")
yolo_no_cases_score = run_score(
    "yolo",
    write_case_results("yolo", "yolo_m30", yolo_det_good_cases),
    "yolo-no-cases",
)
assert not yolo_no_cases_score["passed"]
assert "fixed COCO reference contract" in yolo_no_cases_score["valid_note"]

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
import hashlib
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
    "validation_contract_sha256": hashlib.sha256(
        Path(".github/ci/reference/yolo.json").read_bytes()
    ).hexdigest(),
}
(tmp / "score-yolo.json").write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ci-only-pass.md" \
  --target board --models "yolo" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py should allow non-submission CI/scoring-only changes without runtime improvement"
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-forced-submission-fail.md" \
  --target board --models "yolo" --base-ref HEAD --require-baseline \
  --force-submission-models yolo >/dev/null; then
  bad "leaderboard_gate.py should require a trusted YOLO submission to improve runtime"
fi
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "score-yolo.json"
score = json.loads(path.read_text())
score["validation_contract_sha256"] = "wrong-contract"
path.write_text(json.dumps(score) + "\n")
PY
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-contract-fail.md" \
  --target board --models "yolo" --base-ref HEAD >/dev/null; then
  bad "leaderboard_gate.py should reject a score from another validation contract"
fi
python3 - <<'PY' || bad "leaderboard_gate.py YOLO validation-only classification failed"
import copy
import importlib.util
import json
import sys
from pathlib import Path

path = Path(".github/ci/scripts/leaderboard_gate.py")
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("leaderboard_gate", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

base = json.loads(Path(".github/ci/benchmark_config.json").read_text())["models"]["yolo"]
validation_change = copy.deepcopy(base)
validation_change["benchmark_cases"] = []
validation_change["validation"]["min_image_count"] = 99
validation_change["file_loads"][-1]["paths"] = ["yolo/another_coco_input.bin"]
assert module.strip_validation_only_keys(base, "yolo") == module.strip_validation_only_keys(
    validation_change, "yolo"
)

implementation_change = copy.deepcopy(base)
implementation_change["file_loads"][1]["paths"] = ["yolo/fused_weights.bin"]
assert module.strip_validation_only_keys(base, "yolo") != module.strip_validation_only_keys(
    implementation_change, "yolo"
)
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
resolver_tmp="$(mktemp -d)"
cat > "$resolver_tmp/gh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  */pulls*) printf '%s\n' "${FAKE_PR_LOGIN:-}" ;;
  *) printf '%s\n' "${FAKE_COMMIT_LOGIN:-}" ;;
esac
SH
chmod +x "$resolver_tmp/gh"
resolved_merge="$(PATH="$resolver_tmp:$PATH" FAKE_PR_LOGIN=participant FAKE_COMMIT_LOGIN=merger \
  GITHUB_EVENT_NAME=push GITHUB_REF=refs/heads/main GITHUB_ACTOR=ci-actor \
  GITHUB_REPOSITORY=example/repo GH_TOKEN=test-token GITHUB_TOKEN= \
  .github/ci/scripts/resolve_leaderboard_team.sh HEAD)"
if [[ "$resolved_merge" != "participant" ]]; then
  bad "resolve_leaderboard_team.sh merge returned '$resolved_merge', expected 'participant'"
fi
resolved_direct="$(PATH="$resolver_tmp:$PATH" FAKE_PR_LOGIN= FAKE_COMMIT_LOGIN=direct-author \
  GITHUB_EVENT_NAME=push GITHUB_REF=refs/heads/main GITHUB_ACTOR=ci-actor \
  GITHUB_REPOSITORY=example/repo GH_TOKEN=test-token GITHUB_TOKEN= \
  .github/ci/scripts/resolve_leaderboard_team.sh HEAD)"
if [[ "$resolved_direct" != "direct-author" ]]; then
  bad "resolve_leaderboard_team.sh direct push returned '$resolved_direct', expected 'direct-author'"
fi
rm -rf "$resolver_tmp"

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
  bad "changed_benchmark_models.py allowed YOLO source without COCO host-reference validation"
fi

yolo_cfg="$(mktemp)"
python3 - "$yolo_cfg" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
cfg = json.loads(Path(".github/ci/benchmark_config.json").read_text())
out.write_text(json.dumps(cfg))
PY
yolo_validated_out="$(mktemp)"
python3 .github/ci/scripts/changed_benchmark_models.py --target board \
  --config "$yolo_cfg" \
  --changed-file ported_models/yolo/src/yolo_m30_argbuf.c \
  --format space --unregistered-out "$(mktemp)" --uncovered-out "$yolo_validated_out" >/dev/null \
  || bad "changed_benchmark_models.py COCO host-reference YOLO validation case failed"
if [[ -s "$yolo_validated_out" ]]; then
  bad "changed_benchmark_models.py rejected YOLO with COCO host-reference validation"
fi
rm -f "$covered_out" "$yolo_unvalidated_cfg" "$yolo_unvalidated_out" "$yolo_cfg" "$yolo_validated_out"

step "Selector ignores zero-blob fallback path cleanup"
python3 - <<'PY' || bad "changed_benchmark_models.py zero-blob fallback normalization failed"
import importlib.util
import sys
from pathlib import Path

path = Path(".github/ci/scripts/changed_benchmark_models.py")
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("changed_benchmark_models", path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

old = {
    "file_loads": [
        {"address": "0x0", "paths": ["zero2m.bin", "legacy/zero2m.bin"], "required": True},
        {"address": "0x1000", "paths": ["weights.bin", "fallback/weights.bin"], "required": True},
    ],
    "requires_removed_inputs": False,
}
new = {
    "file_loads": [
        {"address": "0x0", "paths": ["zero2m.bin", "common/zero2m.bin"], "required": True},
        {"address": "0x1000", "paths": ["weights.bin", "different/weights.bin"], "required": True},
    ],
}
normalized_old = mod.normalize_zero_blob_fallbacks(old)
normalized_new = mod.normalize_zero_blob_fallbacks(new)
assert normalized_old["file_loads"][0] == normalized_new["file_loads"][0]
assert normalized_old["file_loads"][1] != normalized_new["file_loads"][1]
assert "requires_removed_inputs" not in normalized_old
PY

if [[ "$fail" -ne 0 ]]; then
  printf '\npreflight FAILED — fix the above before pushing to CI.\n' >&2
  exit 1
fi
printf '\npreflight OK\n'
