/*
 * Whisper-Tiny-shaped transformer benchmark.
 *
 * Reuses the transformer kernel implementation with a longer token sequence
 * and wider MLP than the Depth/ViT run.
 */

#define DEPTH_MAGIC       0x57485350u
#define TOK               256u
#define DIM               64u
#define HIDDEN            256u

#define X_OFFSET          0x4000u
#define XT_OFFSET         0x18000u
#define W1_OFFSET         0x28000u
#define W2_OFFSET         0x38000u
#define S_OFFSET          0x50000u
#define O_OFFSET          0x90000u
#define H_OFFSET          0xA0000u
#define Y_OFFSET          0xE0000u

#include "whisper_vpu_bench_common.c"
