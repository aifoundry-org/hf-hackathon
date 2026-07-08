#!/usr/bin/env bash
set -euo pipefail

# Build script for Whisper Tiny EN with fidelity improvements
# This variant uses FP32 conv weights and per-dimension token embedding scales

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/src/whisper_resident_encoder_argbuf.c"
OUT="${ROOT}/weights"
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
	-I"${ROOT}/src"
	-Wl,--gc-sections
	-Wl,--no-warn-rwx-segments
	-Wl,--emit-relocs
	-T "${LD_SCRIPT}"
	-DNUM_HARTS=16
	-DACTIVE_HARTS=16
	-DENCODER_CONV_FP32=1
)

echo "Building Whisper encoder with fidelity improvements..."
echo "  - ENCODER_CONV_FP32=1 (FP32 conv weights)"
echo "  - Per-dimension token embedding scales in decoder"

mkdir -p "${OUT}"

"${GCC}" -O3 "${COMMON[@]}" \
	-o "${OUT}/whisper_encoder_fidelity.elf" "${SRC}" "${CRT}" "${LAYOUT}"

echo "Build complete: ${OUT}/whisper_encoder_fidelity.elf"
echo ""
echo "To build the decoder with per-dimension scales, use:"
echo "  ${ROOT}/src/whisper_resident_decoder_token_argbuf.c"
echo ""
echo "Note: Weight binaries are stored in Modal and downloaded on demand."
echo "Run: modal volume get whisper-packed-weights <filename>"
