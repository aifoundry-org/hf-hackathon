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
	adv20_01_scalar_control
	adv20_02_scalar_lastonly
	adv20_03_scalar_unroll
	adv20_04_scalar_unroll_lastonly
	adv20_05_scalar_boundary
	adv20_06_scalar_boundary_unroll
	adv20_07_scalar_prefetch
	adv20_08_scalar_prefetch_unroll
	adv20_09_scalar_private_fused
	adv20_10_scalar_private_fused_unroll
	adv20_11_scalar_private_fused_barrier
	adv20_12_scalar_private_fused_barrier_unroll
	adv20_13_scalar_ch8
	adv20_14_scalar_ch4
	adv20_15_vpu_fp_p1
	adv20_16_vpu_fp_p8
	adv20_17_vpu_fp_p8_lastonly
	adv20_18_vpu_fp_p8_sharedw
	adv20_19_vpu_fp_p8_private
	adv20_20_vpu_fp_p8_private_sharedw
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
