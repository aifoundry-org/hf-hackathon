#!/usr/bin/env bash
set -uo pipefail

# Run local ET-SoC1 model-port artifacts on sys-emu.
# Local only: no ssh, scp, rsync, or board access.

CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="${BENCHMARK_ARTIFACT_ROOT:-${AMP_ROOT:-$CHECKOUT/local-artifacts/model-port-benchmarks}}"
DEFAULT_BUILD_ROOT="${BUILD_ROOT:-$CHECKOUT/.ci-work/build}"
DEFAULT_LAUNCHER="${LAUNCHER:-$DEFAULT_BUILD_ROOT/erbium_soc1sim_argbuf/erbium_soc1sim_argbuf_dynmem}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-$CHECKOUT/.github/ci/benchmark_config.json}"
if ! PORTED_MODELS="$(python3 "$CHECKOUT/.github/ci/scripts/benchmark_config_helpers.py" --config "$BENCHMARK_CONFIG" --target sysemu --models all --format csv)"; then
  exit 2
fi

suites=()
model_filters=()
variant_filters=()
limit_per_model=0
launcher="$DEFAULT_LAUNCHER"
device="${BENCHMARK_DEVICE:-sys_emu}"
output_dir=""
timeout_s=300
launcher_timeout=240
shire=0
board_lock="${BOARD_LOCK:-/var/lock/etsoc-shire0.lock}"
no_dump=0
dry_run=0
list_only=0
keep_going=1

usage() {
  cat <<'EOF'
Usage: scripts/run_sysemu_model_ports.sh [options]

Options:
  --suite NAME             smoke, full, focused20, focused20b, all. Repeatable.
  --model LIST             Model filter, comma-separated or repeatable.
  --variant LIST           Variant filter, comma-separated or repeatable.
  --limit-per-model N      Limit jobs per suite/model.
  --launcher PATH          Path to erbium_soc1sim_argbuf.
  --output-dir DIR         Output directory. Default: /tmp/hf_sysemu_model_ports_<stamp>.
  --timeout N              Outer per-job timeout in seconds. Default: 300.
  --launcher-timeout N     Timeout passed to the launcher. Default: 240.
  --device NAME            sys_emu (default) or soc1sim (real PCIe board).
  --shire N                Shire index. Default: 0.
  --no-dump                Do not request dump_after.
  --dry-run                Write manifest and print the planned run.
  --list                   List selected jobs and missing inputs.
  --keep-going             Continue after failures. Default.
  --no-keep-going          Stop after the first missing/failing/timed-out job.
  -h, --help               Show this help.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

need_value() {
  local opt="$1"
  local value="${2:-}"
  [[ -n "$value" ]] || die "$opt requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) need_value "$1" "${2:-}"; suites+=("$2"); shift 2 ;;
    --model) need_value "$1" "${2:-}"; model_filters+=("$2"); shift 2 ;;
    --variant) need_value "$1" "${2:-}"; variant_filters+=("$2"); shift 2 ;;
    --limit-per-model) need_value "$1" "${2:-}"; limit_per_model="$2"; shift 2 ;;
    --launcher) need_value "$1" "${2:-}"; launcher="$2"; shift 2 ;;
    --output-dir) need_value "$1" "${2:-}"; output_dir="$2"; shift 2 ;;
    --timeout) need_value "$1" "${2:-}"; timeout_s="$2"; shift 2 ;;
    --launcher-timeout) need_value "$1" "${2:-}"; launcher_timeout="$2"; shift 2 ;;
    --device) need_value "$1" "${2:-}"; device="$2"; shift 2 ;;
    --shire) need_value "$1" "${2:-}"; shire="$2"; shift 2 ;;
    --no-dump) no_dump=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --list) list_only=1; shift ;;
    --keep-going) keep_going=1; shift ;;
    --no-keep-going) keep_going=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$limit_per_model" in ''|*[!0-9]*) die "--limit-per-model must be an integer" ;; esac
case "$timeout_s" in ''|*[!0-9]*) die "--timeout must be an integer" ;; esac
case "$launcher_timeout" in ''|*[!0-9]*) die "--launcher-timeout must be an integer" ;; esac
case "$shire" in ''|*[!0-9]*) die "--shire must be an integer" ;; esac

split_filters() {
  local value item
  for value in "$@"; do
    IFS=',' read -ra parts <<<"$value"
    for item in "${parts[@]}"; do
      item="${item//[[:space:]]/}"
      [[ -n "$item" ]] && printf '%s\n' "$item"
    done
  done
}

contains_line() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

mapfile -t selected_models < <(split_filters "${model_filters[@]}")
mapfile -t selected_variants < <(split_filters "${variant_filters[@]}")
if [[ ${#selected_models[@]} -gt 0 ]]; then
  selected_arg="$(IFS=,; echo "${selected_models[*]}")"
  if ! python3 "$CHECKOUT/.github/ci/scripts/benchmark_config_helpers.py" \
    --config "$BENCHMARK_CONFIG" \
    --target sysemu \
    --models "$selected_arg" \
    --format space >/dev/null; then
    exit 2
  fi
fi

job_suite=()
job_model=()
job_case=()
job_variant=()
job_elf=()
job_loads=()
job_mem=()
job_dump=()

common_loads() {
  local model="$1"
  local case_name="${2:-}"
  python3 - "$CHECKOUT" "$BENCHMARK_CONFIG" "$model" "$ARTIFACTS" "$case_name" <<'PY'
import sys

repo, cfg_path, model, amp, case_name = sys.argv[1:6]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config, resolve_file_loads

loads, _ = resolve_file_loads(load_config(cfg_path), model, amp, case_name or None)
print("|".join(f"{item['address']},{item['path']}" for item in loads))
PY
}

model_cases() {
  local model="$1"
  python3 - "$CHECKOUT" "$BENCHMARK_CONFIG" "$model" <<'PY'
import sys

repo, cfg_path, model = sys.argv[1:4]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config, model_benchmark_cases

for case in model_benchmark_cases(load_config(cfg_path), model):
    print(case["name"])
PY
}

model_dump_size() {
  local model="$1"
  python3 - "$CHECKOUT" "$BENCHMARK_CONFIG" "$model" <<'PY'
import sys

repo, cfg_path, model = sys.argv[1:4]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config

value = load_config(cfg_path)["models"][model].get("dump_size", 8192)
print(int(value, 0) if isinstance(value, str) else int(value))
PY
}

model_mem_size() {
  local model="$1"
  python3 - "$CHECKOUT" "$BENCHMARK_CONFIG" "$model" <<'PY'
import sys

repo, cfg_path, model = sys.argv[1:4]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config

value = load_config(cfg_path)["models"][model].get("mem_size", 16 * 1024 * 1024)
print(int(value, 0) if isinstance(value, str) else int(value))
PY
}

add_job() {
  local suite="$1" model="$2" variant="$3" elf="$4" case_name="${5:-}"
  if [[ ${#selected_models[@]} -gt 0 ]] && ! contains_line "$model" "${selected_models[@]}"; then
    return
  fi
  if [[ ${#selected_variants[@]} -gt 0 ]] && ! contains_line "$variant" "${selected_variants[@]}"; then
    return
  fi
  job_suite+=("$suite")
  job_model+=("$model")
  job_case+=("$case_name")
  job_variant+=("$variant")
  job_elf+=("$elf")
  job_loads+=("$(common_loads "$model" "$case_name")")
  job_mem+=("$(model_mem_size "$model")")
  job_dump+=("$(model_dump_size "$model")")
}

read_variants() {
  local suite="$1" model="$2" workdir="$3" manifest="$4"
  local path="$workdir/$manifest"
  local variant
  [[ -f "$path" ]] || return
  while IFS= read -r variant || [[ -n "$variant" ]]; do
    variant="${variant%%#*}"
    variant="${variant//[[:space:]]/}"
    [[ -n "$variant" ]] || continue
    mapfile -t cases < <(model_cases "$model")
    if [[ ${#cases[@]} -gt 0 ]]; then
      for case_name in "${cases[@]}"; do
        add_job "$suite" "$model" "$variant" "$workdir/$variant.elf" "$case_name"
      done
    else
      add_job "$suite" "$model" "$variant" "$workdir/$variant.elf"
    fi
  done < "$path"
}

build_bench_suite() {
  local suite="$1"
  local model bench_dir manifest
  [[ "$suite" == "smoke" || "$suite" == "full" ]] || die "unsupported bench suite: $suite"
  while IFS=$'\t' read -r model bench_dir manifest || [[ -n "${model:-}" ]]; do
    [[ -n "${model:-}" && -n "${bench_dir:-}" && -n "${manifest:-}" ]] || continue
    read_variants "$suite" "$model" "$ARTIFACTS/$bench_dir" "$manifest"
  done < <(
    python3 "$CHECKOUT/.github/ci/scripts/benchmark_config_helpers.py" \
      --config "$BENCHMARK_CONFIG" \
      --target sysemu \
      --suite "$suite" \
      --models "$PORTED_MODELS"
  )
}

build_focused_suite() {
  local suite="$1"
  local root="$ARTIFACTS/optimization-kb/$suite"
  local manifest="$root/focused20_variants.tsv"
  local model variant
  [[ -f "$manifest" ]] || return
  while IFS=$'\t' read -r model variant _ || [[ -n "${model:-}" ]]; do
    [[ -n "${model:-}" && -n "${variant:-}" ]] || continue
    add_job "$suite" "$model" "$variant" "$root/$variant.elf"
  done < "$manifest"
}

if [[ ${#suites[@]} -eq 0 ]]; then
  suites=(smoke)
fi

expanded_suites=()
for suite in "${suites[@]}"; do
  case "$suite" in
    all) expanded_suites+=(smoke focused20 focused20b full) ;;
    smoke|full|focused20|focused20b) expanded_suites+=("$suite") ;;
    *) die "unknown suite: $suite" ;;
  esac
done

seen_suites=()
for suite in "${expanded_suites[@]}"; do
  contains_line "$suite" "${seen_suites[@]}" && continue
  seen_suites+=("$suite")
  case "$suite" in
    smoke|full) build_bench_suite "$suite" ;;
    focused20|focused20b) build_focused_suite "$suite" ;;
  esac
done

if [[ "$limit_per_model" -gt 0 ]]; then
  declare -A counts=()
  kept_suite=()
  kept_model=()
  kept_case=()
  kept_variant=()
  kept_elf=()
  kept_loads=()
  kept_mem=()
  kept_dump=()
  for i in "${!job_suite[@]}"; do
    key="${job_suite[$i]}:${job_model[$i]}"
    count="${counts[$key]:-0}"
    [[ "$count" -ge "$limit_per_model" ]] && continue
    counts[$key]=$((count + 1))
    kept_suite+=("${job_suite[$i]}")
    kept_model+=("${job_model[$i]}")
    kept_case+=("${job_case[$i]}")
    kept_variant+=("${job_variant[$i]}")
    kept_elf+=("${job_elf[$i]}")
    kept_loads+=("${job_loads[$i]}")
    kept_mem+=("${job_mem[$i]}")
    kept_dump+=("${job_dump[$i]}")
  done
  job_suite=("${kept_suite[@]}")
  job_model=("${kept_model[@]}")
  job_case=("${kept_case[@]}")
  job_variant=("${kept_variant[@]}")
  job_elf=("${kept_elf[@]}")
  job_loads=("${kept_loads[@]}")
  job_mem=("${kept_mem[@]}")
  job_dump=("${kept_dump[@]}")
fi

missing_reasons() {
  local i="$1"
  local reasons=()
  [[ -x "$launcher" ]] || reasons+=("missing launcher: $launcher")
  [[ -f "${job_elf[$i]}" ]] || reasons+=("missing elf: ${job_elf[$i]}")
  while IFS= read -r reason; do
    [[ -n "$reason" ]] && reasons+=("$reason")
  done < <(
    python3 - "$CHECKOUT" "$BENCHMARK_CONFIG" "${job_model[$i]}" "$ARTIFACTS" "${job_case[$i]}" <<'PY'
import sys

repo, cfg_path, model, amp = sys.argv[1:5]
sys.path.insert(0, f"{repo}/.github/ci/scripts")
from benchmark_config_helpers import load_config, resolve_file_loads

case_name = sys.argv[5] if len(sys.argv) > 5 else ""
_, missing = resolve_file_loads(load_config(cfg_path), model, amp, case_name or None)
print("\n".join(missing))
PY
  )
  [[ ${#reasons[@]} -gt 0 ]] && printf '%s\n' "${reasons[@]}"
}

if [[ "$list_only" -eq 1 ]]; then
  echo -e "checkout\t$CHECKOUT"
  echo -e "ported_models\t$PORTED_MODELS"
  echo -e "launcher\t$launcher\t$([[ -x "$launcher" ]] && echo OK || echo MISSING)"
  echo
  for i in "${!job_suite[@]}"; do
    mapfile -t reasons < <(missing_reasons "$i")
    state="OK"
    [[ ${#reasons[@]} -gt 0 ]] && state="MISSING"
    echo -e "$((i + 1))\t$state\t${job_suite[$i]}\t${job_model[$i]}\t${job_case[$i]}\t${job_variant[$i]}\t${job_elf[$i]}"
    for reason in "${reasons[@]}"; do
      echo -e "\t$reason"
    done
  done
  exit 0
fi

if [[ ${#job_suite[@]} -eq 0 ]]; then
  echo "No jobs selected." >&2
  exit 2
fi

stamp="$(date -u +%Y%m%d-%H%M%SZ)"
if [[ -z "$output_dir" ]]; then
  output_dir="/tmp/hf_sysemu_model_ports_$stamp"
fi
mkdir -p "$output_dir/jobs"

manifest="$output_dir/manifest.tsv"
{
  echo -e "index\tsuite\tmodel\tcase\tvariant\telf\tfile_loads"
  for i in "${!job_suite[@]}"; do
    echo -e "$((i + 1))\t${job_suite[$i]}\t${job_model[$i]}\t${job_case[$i]}\t${job_variant[$i]}\t${job_elf[$i]}\t${job_loads[$i]}"
  done
} > "$manifest"

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run: ${#job_suite[@]} jobs"
  echo "Output dir would be: $output_dir"
  echo "Manifest: $manifest"
  exit 0
fi

results="$output_dir/results.tsv"
echo -e "index\tsuite\tmodel\tcase\tvariant\tstatus\trc\telapsed_s\tkernel_wait_s\temu_cycle_last\tlog\tdump\tnote" > "$results"

echo "Output dir: $output_dir"
echo "Jobs: ${#job_suite[@]}"

if [[ "$device" == "soc1sim" && -n "$board_lock" ]]; then
  bash "$CHECKOUT/.github/ci/scripts/prepare_board_lock.sh" "$board_lock"
fi

fail_count=0
for i in "${!job_suite[@]}"; do
  index=$((i + 1))
  safe_variant="$(printf '%s' "${job_variant[$i]}" | tr -c 'A-Za-z0-9_.-' '_')"
  safe_case="$(printf '%s' "${job_case[$i]:-default}" | tr -c 'A-Za-z0-9_.-' '_')"
  job_dir="$output_dir/jobs/$(printf '%04d_%s_%s_%s_%s' "$index" "${job_suite[$i]}" "${job_model[$i]}" "$safe_case" "$safe_variant")"
  mkdir -p "$job_dir"
  log="$job_dir/run.log"
  dump="$job_dir/dump.bin"

  label="${job_suite[$i]}/${job_model[$i]}/${job_variant[$i]}"
  [[ -n "${job_case[$i]}" ]] && label="${job_suite[$i]}/${job_model[$i]}/${job_case[$i]}/${job_variant[$i]}"
  echo "[$index/${#job_suite[@]}] $label"
  mapfile -t reasons < <(missing_reasons "$i")
  if [[ ${#reasons[@]} -gt 0 ]]; then
    printf '%s\n' "${reasons[@]}" > "$log"
    note="$(IFS='; '; echo "${reasons[*]}")"
    echo -e "$index\t${job_suite[$i]}\t${job_model[$i]}\t${job_case[$i]}\t${job_variant[$i]}\tmissing\t\t0.000\t\t\t${log#$output_dir/}\t${dump#$output_dir/}\t$note" >> "$results"
    echo "  missing log=$log"
    fail_count=$((fail_count + 1))
    [[ "$keep_going" -eq 1 ]] || break
    continue
  fi

  cmd=(
    "$launcher"
    --device "$device"
    --elf-load "${job_elf[$i]}"
    --shire "$shire"
  )
  IFS='|' read -ra loads <<<"${job_loads[$i]}"
  for load in "${loads[@]}"; do
    [[ -n "$load" ]] && cmd+=(--file_load "$load")
  done
  if [[ "$no_dump" -eq 0 ]]; then
    cmd+=(--dump_after "$dump")
  fi
  cmd+=(--timeout "$launcher_timeout" --mem_size "${job_mem[$i]}" --dump_size "${job_dump[$i]}")

  printf '%q ' "${cmd[@]}" > "$job_dir/command.txt"
  echo >> "$job_dir/command.txt"

  start=$(date +%s)
  if [[ "$device" == "soc1sim" && -n "$board_lock" && -d "$(dirname "$board_lock")" ]]; then
    python3 "$CHECKOUT/.github/ci/scripts/board_lock.py" \
      --lock "$board_lock" \
      --timeout 600 \
      -- \
      timeout --kill-after=10s "$timeout_s" "${cmd[@]}" > "$log" 2>&1
    rc=$?
  else
    timeout --kill-after=10s "$timeout_s" "${cmd[@]}" > "$log" 2>&1
    rc=$?
  fi
  end=$(date +%s)
  elapsed=$((end - start))

  if [[ "$rc" -eq 124 ]]; then
    status="timeout"
  elif [[ "$rc" -ne 0 ]]; then
    status="fail"
  else
    status="pass"
  fi
  [[ "$status" == "pass" ]] || fail_count=$((fail_count + 1))

  wait_value="$(grep -E 'Kernel wait seconds:' "$log" | tail -1 | sed -E 's/.*Kernel wait seconds:[[:space:]]*([0-9.]+).*/\1/')"
  cycle_value="$(grep -Eo 'DBG emu_cycle=[0-9]+' "$log" | tail -1 | sed 's/DBG emu_cycle=//')"
  dump_field=""
  [[ "$no_dump" -eq 0 ]] && dump_field="${dump#$output_dir/}"
  echo -e "$index\t${job_suite[$i]}\t${job_model[$i]}\t${job_case[$i]}\t${job_variant[$i]}\t$status\t$rc\t$elapsed.000\t$wait_value\t$cycle_value\t${log#$output_dir/}\t$dump_field\t" >> "$results"
  echo "  $status rc=$rc elapsed=${elapsed}s wait=${wait_value:-NA} log=$log"

  if [[ "$status" != "pass" && "$keep_going" -ne 1 ]]; then
    break
  fi
done

summary="$output_dir/SUMMARY.md"
{
  echo "# Sys-Emu Model Port Run"
  echo
  echo "- Jobs attempted: $(($(wc -l < "$results") - 1))"
  echo "- Failures: $fail_count"
  echo
  echo "See \`results.tsv\` for per-variant logs and dumps."
} > "$summary"

echo "Results: $results"
echo "Summary: $summary"
[[ "$fail_count" -eq 0 ]]
