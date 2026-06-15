/*
 * Whisper Tiny En decoder-vocab-projection-shaped VPU FP32 benchmark.
 *
 * The real decoder logits projection is M=1, K=384, N=51864 per token. The
 * current shared transformer harness partitions by rows, so M=1 would use only
 * one hart. This proxy batches 32 rows and uses N=4096 as a column tile to
 * measure the wide-projection access pattern while keeping the 16 MiB arg
 * buffer intact.
 */

#define DEPTH_MAGIC       0x57485631u
#define TOK               32u
#define DIM               384u
#define HIDDEN            4096u

#define X_OFFSET          0x004000u
#define XT_OFFSET         0x010000u
#define W1_OFFSET         0x01C000u
#define W2_OFFSET         0x61C000u
#define S_OFFSET          0xC1C000u
#define O_OFFSET          0xC1D000u
#define H_OFFSET          0xC29000u
#define Y_OFFSET          0xCA9000u

#include "whisper_vpu_bench_common.c"
