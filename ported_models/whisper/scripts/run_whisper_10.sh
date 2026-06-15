#!/usr/bin/env bash
set -euo pipefail

BASE=$REMOTE_ROOT
PARENT="${BASE}/erbium-amp-probe"
WORK="${PARENT}/whisper-bench"

cd "${WORK}"
export LD_LIBRARY_PATH="${BASE}:${PARENT}:${LD_LIBRARY_PATH:-}"

LAUNCH="../erbium_soc1sim_argbuf"
STAMP="$(date +%Y%m%d-%H%M%S)"

while read -r variant; do
	[ -n "${variant}" ] || continue
	echo "=== ${variant} ==="
	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--dump_after "dump_${variant}_${STAMP}.bin" \
		--timeout 240 \
		> "run_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_${variant}_${STAMP}.log" || true
done < whisper_10_variants.txt
