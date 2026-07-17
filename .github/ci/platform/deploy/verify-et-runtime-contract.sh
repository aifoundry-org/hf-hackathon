#!/usr/bin/env bash
# Refuse board work unless the host runtime is the exact, audited build whose
# EventId allocator safely handles uint16_t wraparound.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
contract="${ET_RUNTIME_CONTRACT:-$ROOT/.github/ci/reference/et_runtime.json}"
manifest="${ET_RUNTIME_MANIFEST:-/var/lib/et-soc1-ci/runtime.json}"

if [[ ! -r "$contract" ]]; then
  echo "error: ET runtime contract is missing: $contract" >&2
  exit 1
fi
if [[ ! -r "$manifest" ]]; then
  echo "error: ET runtime install manifest is missing: $manifest" >&2
  exit 1
fi

readarray -t expected < <(python3 - "$contract" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("source_revision", "required_marker"):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"invalid ET runtime contract field: {key}")
    print(value)
libraries = data.get("libraries")
if not isinstance(libraries, list) or not libraries:
    raise SystemExit("invalid ET runtime contract field: libraries")
for library in libraries:
    if not isinstance(library, dict):
        raise SystemExit("invalid ET runtime library entry")
    values = [library.get(key) for key in ("role", "path", "sha256")]
    if not all(isinstance(value, str) and value for value in values):
        raise SystemExit("invalid ET runtime library entry")
    print("\t".join(values))
PY
)
expected_revision="${expected[0]}"
required_marker="${expected[1]}"
expected_libraries=("${expected[@]:2}")

readarray -t installed < <(python3 - "$manifest" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
revision = data.get("source_revision")
if not isinstance(revision, str) or not revision:
    raise SystemExit("invalid ET runtime manifest field: source_revision")
print(revision)
libraries = data.get("libraries")
if not isinstance(libraries, list) or not libraries:
    raise SystemExit("invalid ET runtime manifest field: libraries")
for library in libraries:
    values = [library.get(key) for key in ("role", "path", "sha256")]
    if not all(isinstance(value, str) and value for value in values):
        raise SystemExit("invalid ET runtime manifest library entry")
    print("\t".join(values))
PY
)

if [[ "${installed[0]}" != "$expected_revision" ]]; then
  echo "error: installed ET runtime manifest does not match the audited contract" >&2
  exit 1
fi
installed_libraries=("${installed[@]:1}")
if [[ "${installed_libraries[*]}" != "${expected_libraries[*]}" ]]; then
  echo "error: installed ET runtime libraries do not match the audited contract" >&2
  exit 1
fi

for entry in "${expected_libraries[@]}"; do
  IFS=$'\t' read -r role library expected_sha <<<"$entry"
  if [[ ! -r "$library" ]]; then
    echo "error: audited ET runtime library is missing: $library" >&2
    exit 1
  fi
  actual_sha="$(sha256sum "$library" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: ET runtime $role sha256 $actual_sha != audited $expected_sha" >&2
    exit 1
  fi
  if ! grep -Fqx "$required_marker" < <(strings "$library"); then
    echo "error: ET runtime $role lacks the audited EventId exhaustion guard" >&2
    exit 1
  fi
done

echo "ET runtime contract OK: source=$expected_revision libraries=${#expected_libraries[@]}"
