#!/usr/bin/env bash
set -euo pipefail

BASE=$REMOTE_ROOT
PARENT="${BASE}/erbium-amp-probe"
WORK="${PARENT}/dncnn3-pmc"

cd "${WORK}"
export LD_LIBRARY_PATH="${BASE}:${PARENT}:${LD_LIBRARY_PATH:-}"

LAUNCH="../erbium_soc1sim_argbuf"
STAMP="$(date +%Y%m%d-%H%M%S)"

variants=(
	vpuv2_00_baseline_sharedw
	vpuv2_01_acc3x3_sharedw
	vpuv2_02_acc3x3_sharedw_lastonly
	vpuv2_03_acc3x3_noshared
	vpuv2_04_acc3x3_p1_sharedw
	vpuv2_05_acc3x3_private_sharedw
	vpuv2_06_oc2_sharedw
	vpuv2_07_oc2_sharedw_lastonly
	vpuv2_08_oc2_accfinal_sharedw
	vpuv2_09_oc2_accfinal_lastonly
	vpuv2_scale_acc3x3_sharedw_1h
	vpuv2_scale_acc3x3_sharedw_2h
	vpuv2_scale_acc3x3_sharedw_4h
	vpuv2_scale_acc3x3_sharedw_8h
	vpuv2_scale_acc3x3_sharedw_16h
	vpuv2_scale_oc2_sharedw_1h
	vpuv2_scale_oc2_sharedw_2h
	vpuv2_scale_oc2_sharedw_4h
	vpuv2_scale_oc2_sharedw_8h
	vpuv2_scale_oc2_sharedw_16h
	vpuv2_scale_oc2_accfinal_1h
	vpuv2_scale_oc2_accfinal_2h
	vpuv2_scale_oc2_accfinal_4h
	vpuv2_scale_oc2_accfinal_8h
	vpuv2_scale_oc2_accfinal_16h
)

for variant in "${variants[@]}"; do
	echo "=== ${variant} ==="
	"${LAUNCH}" \
		--elf-load "./${variant}.elf" \
		--shire 0 \
		--file_load 0x0,../zero2m.bin \
		--file_load 0x2000,../dncnn3_input.bin \
		--file_load 0x4000,../dncnn3_weights.bin \
		--dump_after "dump_${variant}_${STAMP}.bin" \
		--timeout 240 \
		> "run_${variant}_${STAMP}.log" 2>&1
	grep "Kernel wait seconds" "run_${variant}_${STAMP}.log" || true
done
