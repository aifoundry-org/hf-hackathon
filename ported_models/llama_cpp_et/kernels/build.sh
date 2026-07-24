#!/usr/bin/env bash
# Build the fused FFN uberkernel for Llama 3.2 1B on ET-SoC1.
#
# Usage:
#   ./build.sh                    # builds the fused FFN kernel (.o)
#   ./build.sh link               # builds and links a standalone .elf
#
# Output: build/llama32_fused_ffn.o (or .elf with 'link')
#
# Prerequisites:
#   - ET RISC-V toolchain on PATH (riscv64-unknown-elf-gcc, /opt/et/bin)
#   - ET SDK headers (installed at /opt/et)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/llama32_fused_ffn.c"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

ET_SDK="${ET_SDK:-/opt/et}"
CC="${ET_SDK}/bin/riscv64-unknown-elf-gcc"
OBJCOPY="${ET_SDK}/bin/riscv64-unknown-elf-objcopy"

COMMON_FLAGS=(
    -march=rv64imfc
    -mabi=lp64f
    -O2
    -ffast-math
    -fno-tree-loop-distribute-patterns
    -I"${ET_SDK}/cm-umode/include"
    -I"${ET_SDK}/include/esperanto"
    -I"${ET_SDK}/include"
    -I"${ET_SDK}/minion-bl/include"
    -I"${ET_SDK}/sp-bl1/include"
)

"${CC}" "${COMMON_FLAGS[@]}" \
    -c "$KERNEL_SRC" \
    -o "$BUILD_DIR/llama32_fused_ffn.o"

echo "✅ Compiled: $BUILD_DIR/llama32_fused_ffn.o"
"${ET_SDK}/bin/riscv64-unknown-elf-size" "$BUILD_DIR/llama32_fused_ffn.o"

if [ "${1:-}" = "link" ]; then
    # For a standalone .elf, link against the ET runtime stub.
    # The resulting .elf is loadable by the launcher via /dev/et0_ops.
    # This requires the ET linker script and runtime library.
    LD="${ET_SDK}/bin/riscv64-unknown-elf-ld"
    LINKER_SCRIPT="${ET_SDK}/cm-umode/lib/kernel.ld"  # adjust if different

    if [ -f "$LINKER_SCRIPT" ]; then
        "${LD}" -T "$LINKER_SCRIPT" \
            "$BUILD_DIR/llama32_fused_ffn.o" \
            -o "$BUILD_DIR/llama32_fused_ffn.elf"
        echo "✅ Linked: $BUILD_DIR/llama32_fused_ffn.elf"
    else
        echo "Note: linker script not found at $LINKER_SCRIPT"
        echo "  The .o file can still be embedded in the llama.cpp-et build."
    fi
fi
