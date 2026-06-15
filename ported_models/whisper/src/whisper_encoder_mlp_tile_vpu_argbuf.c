/*
 * Whisper Tiny En encoder-MLP-shaped VPU FP32 benchmark.
 *
 * This is not a full Whisper runner. It reuses the transformer microkernel
 * harness with dimensions matching a row tile of the real encoder MLP:
 *   M=128 rows, K=384 model dimension, N=1536 hidden dimension.
 */

#define DEPTH_MAGIC       0x57484D31u
#define TOK               128u
#define DIM               384u
#define HIDDEN            1536u

#define X_OFFSET          0x004000u
#define XT_OFFSET         0x034000u
#define W1_OFFSET         0x064000u
#define W2_OFFSET         0x2A4000u
#define S_OFFSET          0x4E4000u
#define O_OFFSET          0x4F4000u
#define H_OFFSET          0x524000u
#define Y_OFFSET          0x5E4000u

#include "whisper_vpu_bench_common.c"
