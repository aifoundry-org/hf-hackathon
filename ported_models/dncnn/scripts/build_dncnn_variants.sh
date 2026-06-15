#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/erbium_amp_probe/dncnn3-bench/dncnn3_bench_argbuf.c"
OUT="${ROOT}/erbium_amp_probe/dncnn3-pmc"
CRT="${ROOT}/erbium_amp_probe/hart-report/hart_report_crt.S"

GCC="${GCC:-$ET_INSTALL/bin/riscv64-unknown-elf-gcc}"
LAYOUT="${LAYOUT:-$ET_PLATFORM_SRC/erbium-examples/runtime/erbium-soc1sim/layout.c}"
LD_SCRIPT="${LD_SCRIPT:-/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld}"

COMMON=(
	-O3
	-march=rv64imfc
	-mabi=lp64f
	-mcmodel=medany
	-nostdlib
	-fno-zero-initialized-in-bss
	-ffunction-sections
	-fdata-sections
	-I/tmp/et-vidas-install/erbium-soc1sim-umode/include
	-I/tmp/et-vidas-install/include/esperanto-fw/erbium_hal
	-I$ET_PLATFORM_SRC/hal/platform/etsoc/include
	-I$ET_PLATFORM_SRC/et-common-libs/include
	-Wl,--gc-sections
	-Wl,--no-warn-rwx-segments
	-Wl,--emit-relocs
	-T "${LD_SCRIPT}"
)

build_one() {
	local prefix="$1"
	local harts="$2"
	shift 2

	"${GCC}" "${COMMON[@]}" \
		-DNUM_HARTS="${harts}" \
		-DACTIVE_HARTS="${harts}" \
		"$@" \
		-o "${OUT}/${prefix}_${harts}h.elf" \
		"${SRC}" "${CRT}" "${LAYOUT}"
}

for harts in 1 2 4 8 16; do
	build_one dncnn3_bench_unroll "${harts}" -DDNCNN_UNROLL_HIDDEN=1
	build_one dncnn3_bench_skip_finalbar "${harts}" -DDNCNN_SKIP_FINAL_BARRIER=1
	build_one dncnn3_bench_nolayerbar "${harts}" -DDNCNN_SKIP_LAYER_BARRIERS=1 -DDNCNN_SKIP_FINAL_BARRIER=1
	build_one dncnn3_bench_fused_halo "${harts}" -DDNCNN_FUSED_HALO=1
	build_one dncnn3_bench_fused_halo_unroll "${harts}" -DDNCNN_FUSED_HALO=1 -DDNCNN_UNROLL_HIDDEN=1
	build_one dncnn3_bench_ch8 "${harts}" -DDNCNN_CH=8u
	build_one dncnn3_bench_ch4 "${harts}" -DDNCNN_CH=4u
done
