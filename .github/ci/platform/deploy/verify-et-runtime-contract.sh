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
for key in ("source_revision", "library_path", "sha256", "required_marker"):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"invalid ET runtime contract field: {key}")
    print(value)
PY
)
expected_revision="${expected[0]}"
library="${expected[1]}"
expected_sha="${expected[2]}"
required_marker="${expected[3]}"

readarray -t installed < <(python3 - "$manifest" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("source_revision", "library_path", "sha256"):
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise SystemExit(f"invalid ET runtime manifest field: {key}")
    print(value)
PY
)

if [[ "${installed[0]}" != "$expected_revision" \
   || "${installed[1]}" != "$library" \
   || "${installed[2]}" != "$expected_sha" ]]; then
  echo "error: installed ET runtime manifest does not match the audited contract" >&2
  exit 1
fi
if [[ ! -r "$library" ]]; then
  echo "error: audited ET runtime library is missing: $library" >&2
  exit 1
fi
actual_sha="$(sha256sum "$library" | awk '{print $1}')"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "error: ET runtime sha256 $actual_sha != audited $expected_sha" >&2
  exit 1
fi
if ! grep -Fqx "$required_marker" < <(strings "$library"); then
  echo "error: ET runtime lacks the audited EventId exhaustion guard" >&2
  exit 1
fi

echo "ET runtime contract OK: source=$expected_revision sha256=$expected_sha"
