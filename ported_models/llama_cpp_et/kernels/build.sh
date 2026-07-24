#!/usr/bin/env bash
# Build the fused FFN uberkernel for Llama 3.2 1B on ET-SoC1.
#
# Prerequisites:
#   - ET toolchain on PATH (riscv64-unknown-elf-gcc)
#   - ET platform headers at /opt/et-platform
#   - llama.cpp-et source at the paths below (adjust if different)
#
# Usage:
#   ./build.sh                    # builds the fused FFN kernel
#
# Output: build/llama32_fused_ffn.elf
#
# Integration:
#   Option A: Add to uberkernel build
#     - Copy to ggml/src/ggml-et/et-kernels/src/
#     - Add "llama32_fused_ffn" to KERNELS list in CMakeLists.txt
#
#   Option B: Runtime load
#     - Place .elf where GGML_ET_KERNELS_PATH points
#     - Modify llama.cpp-et FFN code to call this kernel
#       instead of dispatching 5 separate ops

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KERNEL_SRC="$SCRIPT_DIR/llama32_fused_ffn.c"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# Paths to llama.cpp-et source (adjust for your workspace)
LLAMA_CPP_ET="${LLAMA_CPP_ET:-$HOME/workspace/llama.cpp-et}"
GGML_SRC="${LLAMA_CPP_ET}/ggml/src"
GGML_INC="${LLAMA_CPP_ET}/ggml/include"

# ET platform SDK (from toolchain install)
ET_PLATFORM="${ET_PLATFORM:-/opt/et-platform}"

set -x
riscv64-unknown-elf-gcc \
    -march=rv64imf -mabi=lp64f \
    -O3 -ffreestanding -nostdlib -ffat-lto-objects -fno-zero-initialized-in-bss \
    -std=gnu99 -ffunction-sections -fdata-sections -g0 \
    -I"$SCRIPT_DIR" \
    -I"$GGML_SRC" \
    -I"$GGML_INC" \
    -I"${ET_PLATFORM}/et-common-libs/include" \
    -I"${ET_PLATFORM}/erbium-hal/include" \
    -I"${ET_PLATFORM}/et-trace/include" \
    -c "$KERNEL_SRC" \
    -o "$BUILD_DIR/llama32_fused_ffn.elf"
set +x

echo ""
echo "✅ Built: $BUILD_DIR/llama32_fused_ffn.elf"
riscv64-unknown-elf-size "$BUILD_DIR/llama32_fused_ffn.elf"
