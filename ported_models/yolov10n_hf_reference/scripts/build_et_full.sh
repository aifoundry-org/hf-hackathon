#!/usr/bin/env bash
# Compile only a checksum-checked schema-v2 N000:N307 package.
set -euo pipefail

port_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$port_root/../.." && pwd)"
full_dir="${1:-$repo_root/local-artifacts/yolov10n_hf_reference/full_graph/deterministic}"
output="${2:-$full_dir/yolov10n_hf_full.elf}"

"$port_root/scripts/verify_full_package.sh" "$full_dir"
"$port_root/scripts/build_et_slice.sh" "$full_dir" "$output"

echo "ET_FULL_BUILD PASS selector=N000:N307 output=$output record=$output.build.json"
