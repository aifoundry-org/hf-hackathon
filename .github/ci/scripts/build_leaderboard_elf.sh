#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
source "$(dirname "$0")/resolve_et_platform_paths.sh"

model="${1:?model required}"

if [[ ! -x "${ET_INSTALL}/bin/riscv64-unknown-elf-gcc" ]]; then
  echo "error: toolchain missing" >&2
  exit 1
fi

resolve_et_platform_paths

eval "$(python3 - <<'PY' "$model" "$BENCHMARK_CONFIG" "$REPO_ROOT" "$ET_INSTALL" "$AMP_ROOT"
import json, shlex, sys, subprocess, os

model, cfg_path, repo, et_install, amp = sys.argv[1:6]
sys.path.insert(0, os.path.join(repo, ".github", "ci", "scripts"))
from benchmark_config_helpers import build_defines, ci_smoke_enabled, load_config, model_names, model_runner
cfg = load_config(cfg_path)
if model not in cfg["models"]:
    raise SystemExit(f"error: unknown benchmark model {model!r}; configured models: {', '.join(model_names(cfg))}")
m = cfg["models"][model]
if model_runner(cfg, model) != "elf":
    raise SystemExit(f"error: benchmark model {model!r} does not build an ELF")
defines = build_defines(cfg, model)
default_hart_defines = []
if not any(d.startswith("-DNUM_HARTS=") for d in defines):
    default_hart_defines.append("-DNUM_HARTS=16")
if not any(d.startswith("-DACTIVE_HARTS=") for d in defines):
    default_hart_defines.append("-DACTIVE_HARTS=16")
hart_defines = default_hart_defines + defines
link_defines = []
for define in hart_defines:
    if define.startswith("-DNUM_HARTS="):
        value = define.split("=", 1)[1]
        if value.isdigit():
            link_defines.append(f"-Wl,--defsym=NUM_HARTS={value}")
variant = m["canonical_variant"]
src = os.path.join(repo, m["source"])
bench_dir = os.path.join(amp, m["bench_dir"])
os.makedirs(bench_dir, exist_ok=True)
out_elf = os.path.join(bench_dir, variant + ".elf")
manifest_src = os.path.join(repo, "ported_models", model, "manifests")
if os.path.isdir(manifest_src):
    for name in os.listdir(manifest_src):
        subprocess.run(["cp", "-a", os.path.join(manifest_src, name), bench_dir], check=False)

gcc = os.path.join(et_install, "bin", "riscv64-unknown-elf-gcc")
crt = os.path.join(repo, ".github/ci/support/hart_report_crt.S")
ld = os.environ["ERBIUM_LD"]
layout = os.environ["ERBIUM_LAYOUT"]
inc_flags = os.environ.get("ERBIUM_GCC_INCLUDE_FLAGS", "").split()

if ci_smoke_enabled():
    print("echo", shlex.join(["CI smoke build (reduced passes):", " ".join(defines)]))

cmd = [
    gcc, m["build"]["opt"],
    "-march=rv64imfc", "-mabi=lp64f", "-mcmodel=medany",
    "-nostdlib", "-fno-zero-initialized-in-bss",
    "-ffunction-sections", "-fdata-sections",
    *inc_flags,
    "-Wl,--gc-sections", "-Wl,--no-warn-rwx-segments", "-Wl,--emit-relocs",
    *link_defines,
    f"-T{ld}",
    *default_hart_defines,
    *defines,
    "-o", out_elf, src, crt, layout,
]
print("echo", shlex.join(["Building", out_elf]))
print(shlex.join(cmd))
PY
)"
