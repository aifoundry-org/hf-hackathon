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
	-DDNCNN_PASSES=8
	-DDNCNN_VPU_SHARED_WPACK=1
	-DDNCNN_VPU_OC2=1
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
	m02_skipw
	m03_skipw_prefw
	m04_skipfinal
	m05_skipfinal_prefw
	m06_skipfinal_skipw
	m07_lastonly_prefw
	m08_accfinal
	m09_accfinal_prefw
)

macro_flags=(
	""
	"-DDNCNN_VPU_PREFETCH_WPACK=1"
	"-DDNCNN_VPU_SKIP_WPACK_ALLHART_EVICT=1"
	"-DDNCNN_VPU_SKIP_WPACK_ALLHART_EVICT=1 -DDNCNN_VPU_PREFETCH_WPACK=1"
	"-DDNCNN_SKIP_FINAL_BARRIER=1"
	"-DDNCNN_SKIP_FINAL_BARRIER=1 -DDNCNN_VPU_PREFETCH_WPACK=1"
	"-DDNCNN_SKIP_FINAL_BARRIER=1 -DDNCNN_VPU_SKIP_WPACK_ALLHART_EVICT=1"
	"-DDNCNN_EVICT_OUTPUT_LAST_ONLY=1 -DDNCNN_SKIP_NONLAST_FINAL_BARRIER=1 -DDNCNN_VPU_PREFETCH_WPACK=1"
	"-DDNCNN_VPU_ACCUM_3X3=1"
	"-DDNCNN_VPU_ACCUM_3X3=1 -DDNCNN_VPU_PREFETCH_WPACK=1"
)

: > "${OUT}/v3_100_variants.txt"

for oi in "${!opt_names[@]}"; do
	for mi in "${!macro_names[@]}"; do
		name="v3100_${opt_names[$oi]}_${macro_names[$mi]}"
		read -r -a opt_array <<< "${opt_flags[$oi]}"
		read -r -a macro_array <<< "${macro_flags[$mi]}"
		echo "${name}" >> "${OUT}/v3_100_variants.txt"
		"${GCC}" "${opt_array[@]}" "${COMMON[@]}" "${macro_array[@]}" \
			-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
	done
done
