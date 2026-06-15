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

build_v3() {
	local name="$1"
	local opt="$2"
	shift 2
	"${GCC}" "${opt}" "${COMMON[@]}" "$@" \
		-o "${OUT}/${name}.elf" "${SRC}" "${CRT}" "${LAYOUT}"
}

cat > "${OUT}/v3x_variants.txt" <<'EOF'
v3x_01_oc2_base
v3x_02_oc2_boundary
v3x_03_oc2_boundary_skipfinal
v3x_04_oc2_skipfinal
v3x_05_oc2_skip_wpack_evict
v3x_06_oc2_boundary_skip_wpack
v3x_07_oc2_prefetch_read
v3x_08_oc2_boundary_prefetch
v3x_09_oc2_prefetch_wpack
v3x_10_oc2_ofast
EOF

build_v3 v3x_01_oc2_base -O3
build_v3 v3x_02_oc2_boundary -O3 -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1
build_v3 v3x_03_oc2_boundary_skipfinal -O3 -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1 -DDNCNN_SKIP_FINAL_BARRIER=1
build_v3 v3x_04_oc2_skipfinal -O3 -DDNCNN_SKIP_FINAL_BARRIER=1
build_v3 v3x_05_oc2_skip_wpack_evict -O3 -DDNCNN_VPU_SKIP_WPACK_ALLHART_EVICT=1
build_v3 v3x_06_oc2_boundary_skip_wpack -O3 -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1 -DDNCNN_VPU_SKIP_WPACK_ALLHART_EVICT=1
build_v3 v3x_07_oc2_prefetch_read -O3 -DDNCNN_VPU_PREFETCH_READ_WINDOW=1
build_v3 v3x_08_oc2_boundary_prefetch -O3 -DDNCNN_VPU_BOUNDARY_ONLY_EVICT=1 -DDNCNN_VPU_PREFETCH_READ_WINDOW=1
build_v3 v3x_09_oc2_prefetch_wpack -O3 -DDNCNN_VPU_PREFETCH_WPACK=1
build_v3 v3x_10_oc2_ofast -Ofast
