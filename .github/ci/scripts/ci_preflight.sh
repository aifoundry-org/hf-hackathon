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
# Git exports repository-local GIT_* variables to hooks. Drop them after
# resolving this worktree so temporary repositories created by the tests do
# not accidentally commit into or move the caller's branch.
while IFS= read -r variable; do
  unset "$variable"
done < <(git rev-parse --local-env-vars)
fail=0
note() { printf '  %s\n' "$*"; }
step() { printf '==> %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; fail=1; }

step "Python syntax (.github/ci/scripts/*.py)"
if ! python3 -m py_compile .github/ci/scripts/*.py; then
  bad "python compile errors"
fi
python3 -m unittest discover -s .github/ci/scripts -p 'test_track_labels.py' \
  || bad "track label classifier tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_smolvlm2_video_benchmark.py' \
  || bad "SmolVLM2 video benchmark tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_prepare_trusted_smolvlm2_candidate.py' \
  || bad "trusted SmolVLM2 candidate scope tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_trusted_smolvlm2_gate.py' \
  || bad "trusted SmolVLM2 gate tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_trusted_llama32_policy.py' \
  || bad "trusted Llama track policy tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_model_port_track.py' \
  || bad "trusted model-port track tests failed"
python3 -m unittest discover -s .github/ci/scripts -p 'test_board_lock.py' \
  || bad "shared board lock tests failed"
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
python3 - <<'PY' || bad "ET-SoC1 workflows are not pinned exclusively to aifoundry2"
from pathlib import Path

import yaml

targets = {
    "benchmark-board.yml": "board-benchmark",
    "trusted-llama32-pr.yml": "board",
    "trusted-model-port-pr.yml": "board",
    "trusted-smolvlm2-pr.yml": "board",
}
for filename, job_name in targets.items():
    workflow = yaml.safe_load((Path(".github/workflows") / filename).read_text())
    labels = workflow["jobs"][job_name]["runs-on"]
    assert "aifoundry2" in labels, f"{filename} is not pinned to aifoundry2"
    assert "aifoundry3" not in labels, f"{filename} still selects aifoundry3"
PY
if ! grep -qF 'uses: aifoundry-org/hf-hackathon/.github/workflows/benchmark-board.yml@main' \
  .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO PR caller is not pinned to the main-owned reusable workflow"
fi
if ! grep -qF 'pull_request_target:' .github/workflows/trusted-yolo-pr.yml; then
  bad "trusted YOLO caller must be loaded from the default branch"
fi
if ! grep -qF 'pull_request_target:' .github/workflows/trusted-smolvlm2-pr.yml; then
  bad "trusted SmolVLM2 workflow must be loaded from the default branch"
fi
if ! grep -qF 'context=trusted-model/smolvlm2_500m_video' \
  .github/workflows/trusted-smolvlm2-pr.yml; then
  bad "trusted SmolVLM2 workflow does not publish its merge status on the participant commit"
fi
if grep -qE '^[[:space:]]+paths:' .github/workflows/trusted-smolvlm2-pr.yml; then
  bad "trusted SmolVLM2 final check must run on every PR so it can be required"
fi
if ! grep -qF 'pull_request_target:' .github/workflows/trusted-llama32-pr.yml; then
  bad "trusted Llama workflow must be loaded from the default branch"
fi
if ! grep -qF 'context=trusted-model/llama32_1b' \
  .github/workflows/trusted-llama32-pr.yml; then
  bad "trusted Llama workflow does not publish its merge status on the participant commit"
fi
if grep -qE '^[[:space:]]+paths:' .github/workflows/trusted-llama32-pr.yml; then
  bad "trusted Llama final check must run on every PR so it can be required"
fi
if ! grep -qF 'pull_request_target:' .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port workflow must be loaded from the default branch"
fi
if ! grep -qF 'context=trusted-track/model-port-credit' \
  .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port workflow does not publish its status on the participant commit"
fi
if grep -qE '^[[:space:]]+paths:' .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port final check must run on every PR so it can be required"
fi
if ! grep -qF 'persist-credentials: false' .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port board checkout must not retain repository credentials"
fi
if ! grep -qF 'approve_board_execution' .github/workflows/trusted-model-port-pr.yml; then
  bad "external model-port code must require explicit per-head maintainer approval"
fi
if ! grep -qF 'timeout-minutes: 10' .github/workflows/trusted-model-port-pr.yml \
  || ! grep -qF 'TRUSTED_BOARD_OUTER_TIMEOUT_CAP: "120"' \
    .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port smoke must retain its bounded execution budget"
fi
if ! grep -qF 'Remove participant workspace' .github/workflows/trusted-model-port-pr.yml; then
  bad "trusted model-port workflow must clean the persistent board workspace"
fi
if grep -qE '^ {6}SOC3_DEST: \$\{\{ runner\.temp \}\}/trusted-model-port-board$' \
  .github/workflows/trusted-model-port-pr.yml; then
  bad "runner.temp is unavailable in job-level env; scope SOC3_DEST to board steps"
fi
if [[ "$(grep -cF 'SOC3_DEST: ${{ runner.temp }}/trusted-model-port-board' \
  .github/workflows/trusted-model-port-pr.yml)" -ne 2 ]]; then
  bad "trusted model-port board and cleanup steps must share the runner-temp workspace"
fi
if ! grep -qF 'et_platform_src_complete' .github/ci/platform/deploy/soc3-benchmark.sh \
  || ! grep -qF '_launcher_lib_dir' .github/ci/platform/deploy/soc3-benchmark.sh; then
  bad "board deployment must bind a complete platform tree and matching launcher libraries"
fi
if grep -qF '.removeprefix(' .github/ci/scripts/score_results.py; then
  bad "board scorer must remain compatible with the Python 3.8 board host"
fi
if ! grep -qF -- '--expected-claim-paths' .github/workflows/benchmark-board.yml \
  || ! grep -qF 'git switch --detach origin/main' .github/workflows/benchmark-board.yml; then
  bad "merge-time model-port credit must bind to PR files and current ledger state"
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
if grep -qF "github.event_name == 'pull_request' || inputs.trusted_yolo" \
  .github/workflows/benchmark-board.yml; then
  bad "trusted YOLO board failures must not be marked continue-on-error"
fi
for token in BENCHMARK_SHA BENCHMARK_REF BENCHMARK_RUN_URL; do
  if ! grep -qF "$token" .github/workflows/benchmark-board.yml \
    || ! grep -qF "$token" .github/ci/platform/deploy/soc3-benchmark.sh; then
    bad "trusted board score provenance is missing $token"
  fi
done

step "Failed prebuild cannot reuse a stale board score"
stale_tmp="$(mktemp -d)"
mkdir -p \
  "$stale_tmp/board/benchmark-output" \
  "$stale_tmp/et/bin" \
  "$stale_tmp/platform/gp-sdk/device/sdk/lib/erbium-soc1sim"
printf '{"passed": true, "kernel_wait_s": 0.001}\n' \
  > "$stale_tmp/board/benchmark-output/score-yolo.json"
cat > "$stale_tmp/et/bin/riscv64-unknown-elf-gcc" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "$stale_tmp/et/bin/riscv64-unknown-elf-gcc"
touch "$stale_tmp/platform/gp-sdk/device/sdk/lib/erbium-soc1sim/erbium.ld"
if MODELS=yolo \
  SOC3_LOCAL=1 \
  SOC3_HOST=local \
  SOC3_DEST="$stale_tmp/board" \
  SOC3_BUILD_ET="$stale_tmp/et" \
  ET_PLATFORM_SRC="$stale_tmp/platform" \
  BENCHMARK_ARTIFACT_ROOT="$stale_tmp/artifacts" \
  .github/ci/platform/deploy/soc3-benchmark.sh \
    >"$stale_tmp/run.log" 2>&1; then
  bad "forced YOLO prebuild failure unexpectedly succeeded"
fi
if [[ -e "$stale_tmp/board/benchmark-output/score-yolo.json" ]]; then
  bad "failed prebuild left a stale board score available for collection"
fi
rm -rf "$stale_tmp"

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
    Path(".github/ci/reference/llama32_1b.json"),
    Path(".github/ci/reference/llama32_1b_track.json"),
    Path(".github/ci/reference/model_ports_track.json"),
    Path(".github/ci/reference/rwkv7_15b.json"),
    Path(".github/ci/reference/yolo.json"),
]
cfg = json.load(open(paths[0]))
for m in cfg.get("models", {}).values():
    if "config" in m:
        paths.append(Path(m["config"]))
paths += list(Path("ported_models").glob("*/artifacts.json"))
paths += [
    Path("data/model-port-identities.json"),
    Path("data/model-port-credits.json"),
    Path("data/model-port-standings.json"),
]
bad = False
for p in paths:
    try:
        json.load(open(p))
    except Exception as exc:
        print(f"FAIL: invalid JSON {p}: {exc}", file=sys.stderr); bad = True
sys.exit(1 if bad else 0)
PY

step "Trusted Llama contract covers the shared leaderboard runtime"
python3 - <<'PY' || bad "trusted Llama contract or build definition is incomplete"
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".github/ci/scripts")
from benchmark_config_helpers import load_config, model_runner

contract = json.loads(Path(".github/ci/reference/llama32_1b.json").read_text())
cfg = load_config(Path(".github/ci/benchmark_config.json"))
target = cfg["models"][contract["model"]]
source_artifact = target["framework"]["source_artifact"]
expected = set()
for model, model_cfg in cfg["models"].items():
    if model == contract["model"]:
        continue
    if model_cfg.get("framework", {}).get("source_artifact") != source_artifact:
        continue
    if model_runner(cfg, model) != "llama_server":
        continue
    data = Path("data") / f"{model}.json"
    if not data.is_file():
        continue
    entries = json.loads(data.read_text()).get("entries", [])
    if any(isinstance(entry.get("tokens_per_second"), (int, float)) for entry in entries):
        expected.add(model)

actual = set(contract["runtime"]["regression_models"])
assert actual == expected, (sorted(actual), sorted(expected))
build = target["artifacts"]["llama_cpp_build"]["build"]
assert build["targets"] == ["llama-server", "llama-perplexity", "llama-bench"]

rwkv = cfg["models"]["rwkv7_15b"]
assert rwkv["reference_contract"] == ".github/ci/reference/rwkv7_15b.json"
assert rwkv["llama_server"]["gpu_layers"] == 99
assert rwkv["llama_server"].get("require_full_offload", True)
assert "-nkvo" in rwkv["llama_server"]["extra_args"]
assert "-nkvo" in rwkv["llama_server"]["perplexity"]["extra_args"]
PY

step "Trusted model-port policy, identity registry, ledger, and standings"
python3 - <<'PY' || bad "trusted model-port policy data is inconsistent"
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".github/ci/scripts")
from model_port_claim import active_credits, validate_policy, validate_registry
from render_model_port_standings import readme_section, standings

policy = json.loads(Path(".github/ci/reference/model_ports_track.json").read_text())
registry = json.loads(Path("data/model-port-identities.json").read_text())
ledger = json.loads(Path("data/model-port-credits.json").read_text())
rendered = json.loads(Path("data/model-port-standings.json").read_text())
validate_policy(policy)
identities = validate_registry(registry, policy)
assert identities
baseline_roots = subprocess.check_output(
    ["git", "ls-tree", "-d", "--name-only", f"{policy['baseline_sha']}:ported_models"],
    text=True,
).splitlines()
assert sorted(registry["baseline_port_roots"]) == sorted(
    f"ported_models/{root}" for root in baseline_roots
)
credits = active_credits(ledger, policy)
assert standings(policy, ledger) == rendered
assert readme_section(rendered, credits) in Path("README.md").read_text()
assert "AFOliveira" in policy["excluded_logins"]
for script in (
    ".github/ci/scripts/run_llama_server_benchmark.py",
    ".github/ci/scripts/score_results.py",
):
    assert "benchmark_device" in Path(script).read_text()
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
    write_case_results("yolo", "yolo_m30", yolo_det_good_cases),
    "yolo-good",
)
yolo_det_bad_score = run_score(
    "yolo",
    write_case_results("yolo", "yolo_m30", yolo_det_bad_cases),
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
runtime_error_dir = write_case_results(
    "yolo", "yolo_m30", yolo_det_good_cases
)
runtime_error_log = next(runtime_error_dir.glob("jobs/*/run.log"))
runtime_error_log.write_text(
    "Kernel wait seconds: 0.002000\n"
    "Stream error (event 33): code 7\n"
    "Kernel completed successfully\n"
)
runtime_error_score = run_score("yolo", runtime_error_dir, "yolo-runtime-error")
assert yolo_det_good_score["passed"] and yolo_det_good_score["valid_accuracy"]
assert not yolo_det_bad_score["passed"] and not yolo_det_bad_score["valid_accuracy"]
assert not yolo_det_extra_score["passed"] and not yolo_det_extra_score["valid_accuracy"]
assert not runtime_error_score["passed"]
assert not runtime_error_score["valid_dump"]
assert not runtime_error_score["valid_accuracy"]
assert "runtime log rejected: Stream error" in runtime_error_score["valid_note"]
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
import hashlib
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
    "validation_contract_sha256": hashlib.sha256(
        Path(".github/ci/reference/llama32_1b.json").read_bytes()
    ).hexdigest(),
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
    "sha": "expected-sha",
    "ref": "refs/pull/123/head",
    "run_url": "https://github.com/aifoundry-org/hf-hackathon/actions/runs/456",
}
(tmp / "score-yolo.json").write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-ci-only-pass.md" \
  --target board --models "yolo" --base-ref HEAD >/dev/null \
  || bad "leaderboard_gate.py should allow non-submission CI/scoring-only changes without runtime improvement"
python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-provenance-pass.md" \
  --target board --models "yolo" --base-ref HEAD \
  --expected-sha expected-sha \
  --expected-ref refs/pull/123/head \
  --expected-run-url https://github.com/aifoundry-org/hf-hackathon/actions/runs/456 >/dev/null \
  || bad "leaderboard_gate.py should accept matching score provenance"
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "score-yolo.json"
score = json.loads(path.read_text())
score["run_url"] = "https://github.com/aifoundry-org/hf-hackathon/actions/runs/stale"
path.write_text(json.dumps(score) + "\n")
PY
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-provenance-fail.md" \
  --target board --models "yolo" --base-ref HEAD \
  --expected-sha expected-sha \
  --expected-ref refs/pull/123/head \
  --expected-run-url https://github.com/aifoundry-org/hf-hackathon/actions/runs/456 >/dev/null; then
  bad "leaderboard_gate.py should reject a stale score from another run"
fi
python3 - "$tmp" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "score-yolo.json"
score = json.loads(path.read_text())
score["run_url"] = "https://github.com/aifoundry-org/hf-hackathon/actions/runs/456"
path.write_text(json.dumps(score) + "\n")
PY
if python3 .github/ci/scripts/leaderboard_gate.py --scores-dir "$tmp" --output "$tmp/gate-forced-submission-fail.md" \
  --target board --models "yolo" --base-ref HEAD --require-baseline \
  --force-submission-models yolo >/dev/null; then
  bad "leaderboard_gate.py should require a trusted YOLO submission to improve runtime"
fi
grep -F "Correctness passed on all 5 images, but" "$tmp/gate-forced-submission-fail.md" >/dev/null \
  || bad "leaderboard_gate.py should explain a correct but slower YOLO result"
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

step "Trusted Llama merge gate fixtures"
trusted_scores="$tmp/trusted-llama-scores"
mkdir -p "$trusted_scores"
python3 - "$tmp/trusted-llama-baseline.json" "$trusted_scores" <<'PY' \
  || bad "could not create trusted Llama gate fixtures"
import hashlib
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
scores_dir = Path(sys.argv[2])
contract_path = Path(".github/ci/reference/llama32_1b.json")
contract = json.loads(contract_path.read_text())
contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()


def best(model, key, *, higher):
    entries = json.loads((Path("data") / f"{model}.json").read_text())["entries"]
    values = [float(entry[key]) for entry in entries if isinstance(entry.get(key), (int, float))]
    return (max if higher else min)(values)


model = contract["model"]
target_speed = best(model, "tokens_per_second", higher=True)
target_ppl = best(model, "perplexity", higher=False)
baseline_path.write_text(json.dumps({
    "model": model,
    "passed": True,
    "tokens_per_second": target_speed,
    "perplexity": target_ppl,
    "validation_contract_sha256": contract_sha,
}) + "\n")
(scores_dir / f"score-{model}.json").write_text(json.dumps({
    "model": model,
    "passed": True,
    "team": "ci-fixture",
    "sha": "f" * 40,
    "run_url": "https://github.com/aifoundry-org/hf-hackathon/actions/runs/1",
    "tokens_per_second": target_speed * 1.02,
    "perplexity": target_ppl * 1.10,
    "validation_contract_sha256": contract_sha,
}) + "\n")
for regression in contract["runtime"]["regression_models"]:
    speed = best(regression, "tokens_per_second", higher=True)
    ppl = best(regression, "perplexity", higher=False)
    (scores_dir / f"score-{regression}.json").write_text(json.dumps({
        "model": regression,
        "passed": True,
        "tokens_per_second": speed * 1.01,
        "perplexity": ppl * 1.10,
    }) + "\n")
PY
trusted_regressions="$(jq -r '.runtime.regression_models | join(" ")' .github/ci/reference/llama32_1b.json)"
python3 .github/ci/scripts/trusted_llama32_gate.py \
  --contract .github/ci/reference/llama32_1b.json \
  --track-policy .github/ci/reference/llama32_1b_track.json \
  --mode competition \
  --baseline-score "$tmp/trusted-llama-baseline.json" \
  --candidate-scores-dir "$trusted_scores" \
  --regression-models "$trusted_regressions" \
  --participant ci-fixture \
  --head-sha ffffffffffffffffffffffffffffffffffffffff \
  --output "$tmp/trusted-llama-pass.md" \
  --result-output "$tmp/trusted-llama-result.json" >/dev/null \
  || bad "trusted Llama gate rejected a valid improvement fixture"
python3 - "$tmp/trusted-llama-result.json" <<'PY' \
  || bad "trusted Llama result did not preserve canonical standing metadata"
import json
import sys

result = json.load(open(sys.argv[1]))
assert result["eligible_for_standings"] is True
assert result["participant_login"] == "ci-fixture"
assert result["participant_head_sha"] == "f" * 40
PY
python3 - "$trusted_scores/score-tinyllama11b.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
score = json.loads(path.read_text())
entries = json.loads(Path("data/tinyllama11b.json").read_text())["entries"]
baseline = max(float(entry["tokens_per_second"]) for entry in entries)
score["tokens_per_second"] = baseline * 0.98
path.write_text(json.dumps(score) + "\n")
PY
if python3 .github/ci/scripts/trusted_llama32_gate.py \
  --contract .github/ci/reference/llama32_1b.json \
  --track-policy .github/ci/reference/llama32_1b_track.json \
  --mode competition \
  --baseline-score "$tmp/trusted-llama-baseline.json" \
  --candidate-scores-dir "$trusted_scores" \
  --regression-models "$trusted_regressions" \
  --participant ci-fixture \
  --head-sha ffffffffffffffffffffffffffffffffffffffff \
  --output "$tmp/trusted-llama-fail.md" >/dev/null; then
  bad "trusted Llama gate accepted a shared-runtime regression"
fi
python3 - "$tmp/trusted-llama-baseline.json" "$trusted_scores" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.load(open(sys.argv[1]))
scores = Path(sys.argv[2])
target = scores / "score-llama32_1b.json"
score = json.loads(target.read_text())
score["tokens_per_second"] = baseline["tokens_per_second"] * 0.995
target.write_text(json.dumps(score) + "\n")
tiny = scores / "score-tinyllama11b.json"
score = json.loads(tiny.read_text())
entries = json.loads(Path("data/tinyllama11b.json").read_text())["entries"]
score["tokens_per_second"] = max(float(e["tokens_per_second"]) for e in entries) * 0.995
tiny.write_text(json.dumps(score) + "\n")
PY
python3 .github/ci/scripts/trusted_llama32_gate.py \
  --contract .github/ci/reference/llama32_1b.json \
  --track-policy .github/ci/reference/llama32_1b_track.json \
  --mode regression \
  --baseline-score "$tmp/trusted-llama-baseline.json" \
  --candidate-scores-dir "$trusted_scores" \
  --regression-models "$trusted_regressions" \
  --participant ci-fixture \
  --head-sha ffffffffffffffffffffffffffffffffffffffff \
  --output "$tmp/trusted-llama-regression-pass.md" >/dev/null \
  || bad "trusted Llama gate rejected throughput inside the 1% regression tolerance"
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
fallback_sha=HEAD
read -r -a fallback_parents <<< "$(git show -s --format=%P HEAD)"
if (( ${#fallback_parents[@]} >= 2 )); then
  fallback_sha="${fallback_parents[1]}"
fi
expected_author="$(git show -s --format=%an "$fallback_sha")"
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
resolver_mock="$(mktemp -d)"
cat > "$resolver_mock/gh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' canonical-pr-login
EOF
chmod +x "$resolver_mock/gh"
resolved_api="$(PATH="$resolver_mock:$PATH" GITHUB_EVENT_NAME=push GITHUB_REF=refs/heads/main \
  GITHUB_ACTOR=ci-actor GH_TOKEN=test-token GITHUB_REPOSITORY=org/repo \
  .github/ci/scripts/resolve_leaderboard_team.sh HEAD)"
rm -rf "$resolver_mock"
if [[ "$resolved_api" != "canonical-pr-login" ]]; then
  bad "resolve_leaderboard_team.sh API lookup returned '$resolved_api'"
fi
merge_sha="$(git rev-list --merges --max-count=1 HEAD)"
if [[ -n "$merge_sha" ]]; then
  expected_merge_author="$(git show -s --format=%an "${merge_sha}^2")"
  resolved_merge="$(GITHUB_EVENT_NAME=push GITHUB_REF=refs/heads/main GITHUB_ACTOR=ci-actor \
    .github/ci/scripts/resolve_leaderboard_team.sh "$merge_sha")"
  if [[ "$resolved_merge" != "$expected_merge_author" ]]; then
    bad "resolve_leaderboard_team.sh merge returned '$resolved_merge', expected '$expected_merge_author'"
  fi
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

step "Selector scopes Llama track declarations to Llama 3.2 1B"
llama_submission_selection="$(python3 .github/ci/scripts/changed_benchmark_models.py \
  --target board \
  --changed-file ported_models/llama_cpp_et/submissions/llama32_1b.json \
  --changed-file ported_models/llama_cpp_et/submissions/llama32_1b.track.json \
  --format space 2>/dev/null)"
if [[ "$llama_submission_selection" != "llama32_1b" ]]; then
  bad "Llama track declarations selected '$llama_submission_selection', expected only llama32_1b"
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
