#!/usr/bin/env bash
# Install a prebuilt host runtime only when it exactly matches the repository
# contract. This script does not touch, reset, or power-cycle the ET card and
# does not start the Actions runner.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
contract="${ET_RUNTIME_CONTRACT:-$ROOT/.github/ci/reference/et_runtime.json}"
manifest="${ET_RUNTIME_MANIFEST:-/var/lib/et-soc1-ci/runtime.json}"
staged_dir="${1:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: run this installer as root" >&2
  exit 1
fi
if [[ "$#" -ne 1 || ! -d "$staged_dir" ]]; then
  echo "usage: $0 /path/to/audited/runtime-library-directory" >&2
  exit 2
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
    values = [library.get(key) for key in ("role", "path", "sha256")]
    if not all(isinstance(value, str) and value for value in values):
        raise SystemExit("invalid ET runtime library entry")
    print("\t".join(values))
PY
)
source_revision="${expected[0]}"
required_marker="${expected[1]}"
libraries=("${expected[@]:2}")

for entry in "${libraries[@]}"; do
  IFS=$'\t' read -r role library expected_sha <<<"$entry"
  staged_library="$staged_dir/$(basename "$library")"
  if [[ ! -f "$staged_library" ]]; then
    echo "error: staged ET runtime $role is missing: $staged_library" >&2
    exit 1
  fi
  actual_sha="$(sha256sum "$staged_library" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "error: staged ET runtime $role sha256 $actual_sha != audited $expected_sha" >&2
    exit 1
  fi
  if ! grep -Fqx "$required_marker" < <(strings "$staged_library"); then
    echo "error: staged ET runtime $role lacks the audited EventId exhaustion guard" >&2
    exit 1
  fi
done
if pgrep -f 'Runner\.(Listener|Worker)|llama-(bench|server|perplexity)|erbium_soc1sim' >/dev/null; then
  echo "error: a runner or board workload is active; refusing runtime replacement" >&2
  exit 1
fi

install -d -m 0770 "$(dirname "$manifest")"
install_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for entry in "${libraries[@]}"; do
  IFS=$'\t' read -r role library expected_sha <<<"$entry"
  staged_library="$staged_dir/$(basename "$library")"
  install -d -m 0755 "$(dirname "$library")"
  if [[ -e "$library" ]]; then
    backup="${library}.pre-event-id-guard.$install_stamp"
    cp -a "$library" "$backup"
    echo "Previous $role archived at: $backup"
  fi
  temporary="${library}.tmp.$$"
  install -m 0644 "$staged_library" "$temporary"
  mv -f "$temporary" "$library"
  sync -f "$library"
done

manifest_tmp="${manifest}.tmp.$$"
python3 - "$manifest_tmp" "$source_revision" "$contract" <<'PY'
import datetime
import json
import os
import sys

path, revision, contract_path = sys.argv[1:]
contract = json.load(open(contract_path, encoding="utf-8"))
data = {
    "source_revision": revision,
    "libraries": contract["libraries"],
    "installed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": os.uname().nodename,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 0640 "$manifest_tmp"
mv -f "$manifest_tmp" "$manifest"
sync -f "$manifest"

"$ROOT/.github/ci/platform/deploy/verify-et-runtime-contract.sh"
echo "The runner was not started and the board was not accessed."
