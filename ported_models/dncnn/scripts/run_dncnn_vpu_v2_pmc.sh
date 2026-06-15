#!/usr/bin/env bash
set -euo pipefail

BASE=$REMOTE_ROOT
PARENT="${BASE}/erbium-amp-probe"
WORK="${PARENT}/dncnn3-pmc"

cd "${WORK}"
export LD_LIBRARY_PATH="${BASE}:${PARENT}:${LD_LIBRARY_PATH:-}"

LAUNCH="../erbium_soc1sim_argbuf"
SNAP="./pmc_snapshot_16h.elf"
STAMP="$(date +%Y%m%d-%H%M%S)"

variants=(
	vpuv2_00_baseline_sharedw
	vpuv2_01_acc3x3_sharedw
	vpuv2_06_oc2_sharedw
)

for variant in "${variants[@]}"; do
	echo "=== ${variant} ==="
	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--dump_after "pmc_before_${variant}.bin" \
		--timeout 60 \
		> "run_pmc_before_${variant}_${STAMP}.log" 2>&1

	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--file_load 0x2000,../dncnn3_input.bin \
		--file_load 0x4000,../dncnn3_weights.bin \
		--dump_after "dump_pmc_dncnn_${variant}.bin" \
		--timeout 240 \
		> "run_pmc_dncnn_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_pmc_dncnn_${variant}_${STAMP}.log" || true

	"${LAUNCH}" \
		--elf-load "${SNAP}" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--dump_after "pmc_after_${variant}.bin" \
		--timeout 60 \
		> "run_pmc_after_${variant}_${STAMP}.log" 2>&1
done
