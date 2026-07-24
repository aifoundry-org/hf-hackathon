#!/usr/bin/env bash
# Bind an ET ELF to the exact full manifest and scalar runner source hashes.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
full_dir="${1:?usage: verify_full_elf.sh FULL_DIR ELF [BUILD_RECORD]}"
elf="${2:?usage: verify_full_elf.sh FULL_DIR ELF [BUILD_RECORD]}"
record="${3:-$elf.build.json}"

python3 - \
  "$full_dir" "$elf" "$record" "$port_root" <<'PY'
import hashlib
import json
from pathlib import Path
import struct
import sys

full_dir = Path(sys.argv[1]).resolve()
elf = Path(sys.argv[2]).resolve()
record_path = Path(sys.argv[3]).resolve()
port_root = Path(sys.argv[4]).resolve()


def fail(message):
    raise SystemExit("error: full-ELF contract failed: " + message)


def require(condition, message):
    if not condition:
        fail(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


require(elf.is_file(), "ELF is missing")
require(record_path.is_file(), "ELF build record is missing")
record = json.loads(record_path.read_text(encoding="utf-8"))
require(record.get("schema_version") == 1, "unknown build-record schema")
elf_record = record.get("elf")
require(isinstance(elf_record, dict), "build record lacks ELF identity")
require(
    elf.stat().st_size == elf_record.get("bytes")
    and digest(elf) == elf_record.get("sha256"),
    "ELF identity differs from build record",
)
prefix = elf.read_bytes()[:20]
require(
    len(prefix) == 20
    and prefix[:4] == b"\x7fELF"
    and prefix[4] == 2
    and prefix[5] == 1
    and struct.unpack_from("<H", prefix, 18)[0] == 243,
    "file is not a little-endian ELF64 RISC-V executable",
)

compiler = record.get("compiler")
require(isinstance(compiler, dict), "build record lacks compiler")
require(
    "riscv64-unknown-elf-gcc" in str(compiler.get("path", ""))
    or "et_gcc_docker_wrapper.sh" in str(compiler.get("path", "")),
    "compiler path is not the repository-supported ET GCC path",
)
require(
    "riscv64-unknown-elf-gcc" in str(compiler.get("version", "")),
    "compiler version is not ET RISC-V GCC",
)

inputs = record.get("inputs")
require(isinstance(inputs, dict), "build record lacks input identities")
expected = {
    "slice_manifest_header": full_dir / "slice_manifest.h",
    "runtime_c": port_root / "src" / "ref_runtime.c",
    "runtime_h": port_root / "src" / "ref_runtime.h",
    "pmc_h": port_root / "src" / "ref_pmc.h",
    "et_runner_c": port_root / "src" / "et_slice_runner.c",
}
for name, path in expected.items():
    require(path.is_file(), "{} source is missing".format(name))
    stored = inputs.get(name)
    require(isinstance(stored, dict), "build record lacks " + name)
    require(
        path.stat().st_size == stored.get("bytes")
        and digest(path) == stored.get("sha256"),
        name + " differs from the compiled identity",
    )
for name in ("linker_script", "layout", "crt"):
    stored = inputs.get(name)
    require(isinstance(stored, dict), "build record lacks " + name)
    path = Path(stored.get("path", ""))
    require(path.is_file(), "{} build input is missing".format(name))
    require(
        path.stat().st_size == stored.get("bytes")
        and digest(path) == stored.get("sha256"),
        name + " differs from the compiled identity",
    )

command = record.get("command")
require(isinstance(command, list), "build command is not a list")
require("-DYR_PMC" in command, "build did not enable PMC support")
require(
    "-fno-fast-math" in command
    and "-ffp-contract=off" in command
    and "-fno-tree-vectorize" in command,
    "correctness-first floating-point flags are incomplete",
)
print(
    "FULL_ELF PASS bytes={} sha256={} compiler={!r}".format(
        elf.stat().st_size,
        digest(elf),
        compiler.get("version"),
    )
)
PY
