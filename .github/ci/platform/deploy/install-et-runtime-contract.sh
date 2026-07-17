#!/usr/bin/env bash
# Install a prebuilt host runtime only when it exactly matches the repository
# contract. This script does not touch, reset, or power-cycle the ET card and
# does not start the Actions runner.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
contract="${ET_RUNTIME_CONTRACT:-$ROOT/.github/ci/reference/et_runtime.json}"
manifest="${ET_RUNTIME_MANIFEST:-/var/lib/et-soc1-ci/runtime.json}"
staged_library="${1:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: run this installer as root" >&2
  exit 1
fi
if [[ "$#" -ne 1 || ! -f "$staged_library" ]]; then
  echo "usage: $0 /path/to/audited/libetrt.so" >&2
  exit 2
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
source_revision="${expected[0]}"
library="${expected[1]}"
expected_sha="${expected[2]}"
required_marker="${expected[3]}"
actual_sha="$(sha256sum "$staged_library" | awk '{print $1}')"

if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "error: staged ET runtime sha256 $actual_sha != audited $expected_sha" >&2
  exit 1
fi
if ! strings "$staged_library" | grep -Fqx "$required_marker"; then
  echo "error: staged ET runtime lacks the audited EventId exhaustion guard" >&2
  exit 1
fi
if pgrep -f 'Runner\.(Listener|Worker)|llama-(bench|server|perplexity)|erbium_soc1sim' >/dev/null; then
  echo "error: a runner or board workload is active; refusing runtime replacement" >&2
  exit 1
fi

install -d -m 0770 "$(dirname "$manifest")"
install -d -m 0755 "$(dirname "$library")"
if [[ -e "$library" ]]; then
  backup="${library}.pre-event-id-guard.$(date -u +%Y%m%dT%H%M%SZ)"
  cp -a "$library" "$backup"
  echo "Previous runtime archived at: $backup"
fi
temporary="${library}.tmp.$$"
install -m 0644 "$staged_library" "$temporary"
mv -f "$temporary" "$library"

manifest_tmp="${manifest}.tmp.$$"
python3 - "$manifest_tmp" "$source_revision" "$library" "$expected_sha" <<'PY'
import datetime
import json
import os
import sys

path, revision, library, sha256 = sys.argv[1:]
data = {
    "source_revision": revision,
    "library_path": library,
    "sha256": sha256,
    "installed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": os.uname().nodename,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
chmod 0640 "$manifest_tmp"
mv -f "$manifest_tmp" "$manifest"

"$ROOT/.github/ci/platform/deploy/verify-et-runtime-contract.sh"
echo "The runner was not started and the board was not accessed."
