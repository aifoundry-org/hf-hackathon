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
	dncnn3_bench_unroll
	dncnn3_bench_skip_finalbar
	dncnn3_bench_nolayerbar
	dncnn3_bench_fused_halo
	dncnn3_bench_fused_halo_unroll
	dncnn3_bench_ch8
	dncnn3_bench_ch4
)

for prefix in "${variants[@]}"; do
	for harts in 1 2 4 8 16; do
		tag="${prefix}_${harts}h"
		echo "=== ${tag} ==="

		"${LAUNCH}" \
			--elf-load "${SNAP}" \
			--shire 0 \
			--file_load 0x0,../zero2m.bin \
			--dump_after "pmc_before_${tag}.bin" \
			--timeout 60 \
			> "run_pmc_before_${tag}_${STAMP}.log" 2>&1

		"${LAUNCH}" \
			--elf-load "./${tag}.elf" \
			--shire 0 \
			--file_load 0x0,../zero2m.bin \
			--file_load 0x2000,../dncnn3_input.bin \
			--file_load 0x4000,../dncnn3_weights.bin \
			--dump_after "dump_pmc_dncnn_${tag}.bin" \
			--timeout 180 \
			> "run_pmc_dncnn_${tag}_${STAMP}.log" 2>&1

		grep "Kernel wait seconds" "run_pmc_dncnn_${tag}_${STAMP}.log" || true

		"${LAUNCH}" \
			--elf-load "${SNAP}" \
			--shire 0 \
			--file_load 0x0,../zero2m.bin \
			--dump_after "pmc_after_${tag}.bin" \
			--timeout 60 \
			> "run_pmc_after_${tag}_${STAMP}.log" 2>&1
	done
done
