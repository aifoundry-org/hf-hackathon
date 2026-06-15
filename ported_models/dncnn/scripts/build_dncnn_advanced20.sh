#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCALAR_SRC="${ROOT}/erbium_amp_probe/dncnn3-bench/dncnn3_bench_argbuf.c"
VPU_SRC="${ROOT}/erbium_amp_probe/dncnn3-pmc/dncnn3_vpu_fp_argbuf.c"
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

build_scalar() {
	local name="$1"
	shift
	"${GCC}" "${COMMON[@]}" -DNUM_HARTS=16 -DACTIVE_HARTS=16 "$@" \
		-o "${OUT}/${name}.elf" "${SCALAR_SRC}" "${CRT}" "${LAYOUT}"
}

build_vpu() {
	local name="$1"
	shift
	"${GCC}" "${COMMON[@]}" -DNUM_HARTS=16 -DACTIVE_HARTS=16 "$@" \
		-o "${OUT}/${name}.elf" "${VPU_SRC}" "${CRT}" "${LAYOUT}"
}

build_scalar adv20_01_scalar_control
build_scalar adv20_02_scalar_lastonly -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1
build_scalar adv20_03_scalar_unroll -DDNCNN_UNROLL_HIDDEN=1
build_scalar adv20_04_scalar_unroll_lastonly -DDNCNN_UNROLL_HIDDEN=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1
build_scalar adv20_05_scalar_boundary -DDNCNN_BOUNDARY_ONLY_EVICT=1
build_scalar adv20_06_scalar_boundary_unroll -DDNCNN_BOUNDARY_ONLY_EVICT=1 -DDNCNN_UNROLL_HIDDEN=1
build_scalar adv20_07_scalar_prefetch -DDNCNN_PREFETCH_READ_WINDOW=1
build_scalar adv20_08_scalar_prefetch_unroll -DDNCNN_PREFETCH_READ_WINDOW=1 -DDNCNN_UNROLL_HIDDEN=1
build_scalar adv20_09_scalar_private_fused -DDNCNN_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1
build_scalar adv20_10_scalar_private_fused_unroll -DDNCNN_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_UNROLL_HIDDEN=1
build_scalar adv20_11_scalar_private_fused_barrier -DDNCNN_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_PRIVATE_FUSED_PASS_BARRIER=1
build_scalar adv20_12_scalar_private_fused_barrier_unroll -DDNCNN_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_PRIVATE_FUSED_PASS_BARRIER=1 -DDNCNN_UNROLL_HIDDEN=1
build_scalar adv20_13_scalar_ch8 -DDNCNN_CH=8u
build_scalar adv20_14_scalar_ch4 -DDNCNN_CH=4u

build_vpu adv20_15_vpu_fp_p1 -DDNCNN_PASSES=1
build_vpu adv20_16_vpu_fp_p8 -DDNCNN_PASSES=8
build_vpu adv20_17_vpu_fp_p8_lastonly -DDNCNN_PASSES=8 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1
build_vpu adv20_18_vpu_fp_p8_sharedw -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1
build_vpu adv20_19_vpu_fp_p8_private -DDNCNN_PASSES=8 -DDNCNN_VPU_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1
build_vpu adv20_20_vpu_fp_p8_private_sharedw -DDNCNN_PASSES=8 -DDNCNN_VPU_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_VPU_SHARED_WPACK=1
