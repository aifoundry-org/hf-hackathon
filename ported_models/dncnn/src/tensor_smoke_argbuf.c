/*
 * Minimal tensor FMA smoke test for the Erbium-on-SoC1 launcher.
 *
 * Hart 0 initializes A as 14x16 FP32, B as 16x16 identity, runs one tensor
 * FMA, stores C, and writes a small summary into the arg buffer.  This is not a
 * DnCNN kernel; it is a bring-up check that tensor load/FMA/store is usable
 * from the same U-mode launch path before attempting a tensorized convolution.
 */

#include <stdint.h>

#define ERBIUM_TENSOR_ASSERT(cond) ((void)(cond))
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/tensors.h"

extern char heap0_end[];

#define TENSOR_SMOKE_MAGIC 0x54534D4Bu

#define SUMMARY_OFFSET     0x1000u
#define A_OFFSET           0x2000u
#define B_OFFSET           0x4000u
#define C_OFFSET           0x8000u

#define A_ROWS             14u
#define A_COLS             16u
#define B_ROWS             16u
#define B_COLS             16u
#define C_ROWS             14u
#define C_COLS             16u
#define ROW_STRIDE_BYTES   64u

struct tensor_smoke_summary {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t tensor_error;
	uint32_t checksum;
	uint32_t expected;
	uint32_t done;
	uint32_t reserved[8];
};

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}

	return ptr;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id != 0u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	float *const a = (float *)(base + A_OFFSET);
	float *const b = (float *)(base + B_OFFSET);
	float *const c = (float *)(base + C_OFFSET);
	volatile struct tensor_smoke_summary *const summary =
		(volatile struct tensor_smoke_summary *)(base + SUMMARY_OFFSET);

	for (uint32_t r = 0; r < A_ROWS; r++) {
		for (uint32_t col = 0; col < A_COLS; col++) {
			a[r * A_COLS + col] = (float)(r * A_COLS + col + 1u);
		}
	}

	for (uint32_t r = 0; r < B_ROWS; r++) {
		for (uint32_t col = 0; col < B_COLS; col++) {
			b[r * B_COLS + col] = r == col ? 1.0f : 0.0f;
		}
	}

	for (uint32_t r = 0; r < C_ROWS; r++) {
		for (uint32_t col = 0; col < C_COLS; col++) {
			c[r * C_COLS + col] = 0.0f;
		}
	}

	FENCE;
	evict(a, A_ROWS * ROW_STRIDE_BYTES);
	evict(b, B_ROWS * ROW_STRIDE_BYTES);
	evict(c, C_ROWS * ROW_STRIDE_BYTES);
	WAIT_CACHEOPS;

	tensor_load(false, false, 0, 0, 0, (uint64_t)a, 0,
		    A_ROWS - 1u, ROW_STRIDE_BYTES, 0);
	tensor_load(false, false, 32, 0, 1, (uint64_t)b, 0,
		    B_ROWS - 1u, ROW_STRIDE_BYTES, 1);
	tensor_wait(TENSOR_LOAD_WAIT_0);

	tensor_fma(false, (B_COLS / 4u) - 1u, A_ROWS - 1u, A_COLS - 1u,
		   0, false, false, false, true, 0, 0, 0, true);
	tensor_store(0, 0, ((C_COLS * sizeof(float)) / 4u) - 1u, C_ROWS - 1u,
		     (uint64_t)c, 0, ROW_STRIDE_BYTES);
	tensor_wait(TENSOR_STORE_WAIT);

	FENCE;
	evict(c, C_ROWS * ROW_STRIDE_BYTES);
	WAIT_CACHEOPS;

	uint32_t checksum = 0;

	for (uint32_t i = 0; i < C_ROWS * C_COLS; i++) {
		checksum += (uint32_t)c[i];
	}

	summary->magic = TENSOR_SMOKE_MAGIC;
	summary->hart_id = hart_id;
	summary->minion_id = get_minion_id();
	summary->thread_id = get_thread_id();
	summary->tensor_error = (uint32_t)get_tensor_error();
	summary->checksum = checksum;
	summary->expected = (C_ROWS * C_COLS * (C_ROWS * C_COLS + 1u)) / 2u;
	summary->done = 1u;

	FENCE;
	evict((void *)summary, sizeof(*summary));
	WAIT_CACHEOPS;

	return 0;
}
