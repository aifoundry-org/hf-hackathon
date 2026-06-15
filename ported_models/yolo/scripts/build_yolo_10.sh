#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/erbium_amp_probe/yolo-bench/yolo_vpu_argbuf.c"
OUT="${ROOT}/erbium_amp_probe/yolo-bench"
CRT="${ROOT}/erbium_amp_probe/hart-report/hart_report_crt.S"

GCC="${GCC:-$ET_INSTALL/bin/riscv64-unknown-elf-gcc}"
LAYOUT="${LAYOUT:-$ET_PLATFORM_SRC/erbium-examples/runtime/erbium-soc1sim/layout.c}"
LD_SCRIPT="${LD_SCRIPT:-/tmp/et-vidas-install/erbium-soc1sim-umode/share/erbium.ld}"

COMMON=(
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
	-DNUM_HARTS=16
	-DACTIVE_HARTS=16
	-DYOLO_PASSES=4
)

names=(
	y10_00_base
	y10_01_oc2
	y10_02_oc2_prefw
	y10_03_oc2_boundary
	y10_04_oc2_skipread
	y10_05_oc2_lastout
	y10_06_oc2_skipfinal
	y10_07_oc2_blocks3
	y10_08_oc2_blocks5
	y10_09_oc2_ofast
)

opts=(
	"-O3"
	"-O3 -DYOLO_VPU_OC2=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_PREFETCH_WEIGHTS=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_BOUNDARY_ONLY_EVICT=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_SKIP_READ_EVICT=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_EVICT_OUTPUT_LAST_ONLY=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_SKIP_FINAL_BARRIER=1"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_BLOCKS=3"
	"-O3 -DYOLO_VPU_OC2=1 -DYOLO_BLOCKS=5"
	"-Ofast -DYOLO_VPU_OC2=1 -DYOLO_PREFETCH_WEIGHTS=1"
)

: > "${OUT}/yolo_10_variants.txt"

for i in "${!names[@]}"; do
	name="${names[$i]}"
	read -r -a opt_array <<< "${opts[$i]}"
	echo "${name}" >> "${OUT}/yolo_10_variants.txt"
	"${GCC}" "${opt_array[@]}" "${COMMON[@]}" \
		-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
done
