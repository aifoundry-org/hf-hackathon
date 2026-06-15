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
	-DYOLO_VPU_OC2=1
)

opt_names=(
	o00_o3
	o01_o3_nosched
	o02_o3_unroll
	o03_o3_nounroll
	o04_o3_align32
	o05_o3_align64
	o06_o3_noipa
	o07_o3_notreecopy
	o08_ofast
	o09_o2
)

opt_flags=(
	"-O3"
	"-O3 -fno-schedule-insns -fno-schedule-insns2"
	"-O3 -funroll-loops"
	"-O3 -fno-unroll-loops"
	"-O3 -falign-functions=32 -falign-loops=32"
	"-O3 -falign-functions=64 -falign-loops=64"
	"-O3 -fno-ipa-cp -fno-ipa-sra"
	"-O3 -fno-tree-loop-distribute-patterns -fno-tree-loop-vectorize"
	"-Ofast"
	"-O2"
)

macro_names=(
	m00_base
	m01_prefw
	m02_boundary
	m03_skipread
	m04_skipfinal
	m05_lastout
	m06_boundary_skipfinal
	m07_boundary_prefw
	m08_skipread_skipfinal
	m09_lastout_skipfinal
)

macro_flags=(
	""
	"-DYOLO_PREFETCH_WEIGHTS=1"
	"-DYOLO_BOUNDARY_ONLY_EVICT=1"
	"-DYOLO_SKIP_READ_EVICT=1"
	"-DYOLO_SKIP_FINAL_BARRIER=1"
	"-DYOLO_EVICT_OUTPUT_LAST_ONLY=1"
	"-DYOLO_BOUNDARY_ONLY_EVICT=1 -DYOLO_SKIP_FINAL_BARRIER=1"
	"-DYOLO_BOUNDARY_ONLY_EVICT=1 -DYOLO_PREFETCH_WEIGHTS=1"
	"-DYOLO_SKIP_READ_EVICT=1 -DYOLO_SKIP_FINAL_BARRIER=1"
	"-DYOLO_EVICT_OUTPUT_LAST_ONLY=1 -DYOLO_SKIP_FINAL_BARRIER=1"
)

: > "${OUT}/yolo_100_variants.txt"

for oi in "${!opt_names[@]}"; do
	for mi in "${!macro_names[@]}"; do
		name="y100_${opt_names[$oi]}_${macro_names[$mi]}"
		read -r -a opt_array <<< "${opt_flags[$oi]}"
		read -r -a macro_array <<< "${macro_flags[$mi]}"
		echo "${name}" >> "${OUT}/yolo_100_variants.txt"
		"${GCC}" "${opt_array[@]}" "${COMMON[@]}" "${macro_array[@]}" \
			-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
	done
done
