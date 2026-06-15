#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/erbium_amp_probe/dncnn3-pmc/dncnn3_vpu_fp_argbuf.c"
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

build_vpu() {
	local name="$1"
	local harts="$2"
	shift 2
	"${GCC}" "${COMMON[@]}" -DNUM_HARTS="${harts}" -DACTIVE_HARTS="${harts}" "$@" \
		-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
}

build_vpu vpuv2_00_baseline_sharedw 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1
build_vpu vpuv2_01_acc3x3_sharedw 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_ACCUM_3X3=1
build_vpu vpuv2_02_acc3x3_sharedw_lastonly 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_ACCUM_3X3=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1
build_vpu vpuv2_03_acc3x3_noshared 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_ACCUM_3X3=1
build_vpu vpuv2_04_acc3x3_p1_sharedw 16 -DDNCNN_PASSES=1 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_ACCUM_3X3=1
build_vpu vpuv2_05_acc3x3_private_sharedw 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_PRIVATE_FUSED=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_ACCUM_3X3=1
build_vpu vpuv2_06_oc2_sharedw 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1
build_vpu vpuv2_07_oc2_sharedw_lastonly 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1
build_vpu vpuv2_08_oc2_accfinal_sharedw 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1 -DDNCNN_VPU_ACCUM_3X3=1
build_vpu vpuv2_09_oc2_accfinal_lastonly 16 -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1 -DDNCNN_VPU_ACCUM_3X3=1 -DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1

for harts in 1 2 4 8 16; do
	build_vpu "vpuv2_scale_acc3x3_sharedw_${harts}h" "${harts}" -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_ACCUM_3X3=1
	build_vpu "vpuv2_scale_oc2_sharedw_${harts}h" "${harts}" -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1
	build_vpu "vpuv2_scale_oc2_accfinal_${harts}h" "${harts}" -DDNCNN_PASSES=8 -DDNCNN_VPU_SHARED_WPACK=1 -DDNCNN_VPU_OC2=1 -DDNCNN_VPU_ACCUM_3X3=1
done
