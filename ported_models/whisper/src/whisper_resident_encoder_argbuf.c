/*
 * Resident raw-INT8 Whisper encoder executor.
 *
 * Input:  log-mel features at AUDIO_OFFSET in the 16 MiB runtime arena.
 * Output: decoder cross-attention K/V caches as FP16 at the same offsets used
 * by whisper_resident_decoder_token_argbuf.c.
 *
 * The packed raw-INT8 weight region starts at WEIGHT_REGION_OFFSET.  Temporary
 * hidden/qkv tensors live in the unused tail of that 64 MiB region, at the
 * generated ENCODER_SCRATCH_OFFSET.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "whisper_resident_encoder_weights_auto.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef MAGIC
#define MAGIC 0x5752454eu
#endif
#ifndef RUNTIME_REGION_BYTES
#define RUNTIME_REGION_BYTES (16u * 1024u * 1024u)
#endif
#ifndef WEIGHT_REGION_OFFSET
#define WEIGHT_REGION_OFFSET (16u * 1024u * 1024u)
#endif
#ifndef ENCODER_ROW_MAJOR_MATVEC
#define ENCODER_ROW_MAJOR_MATVEC 1
#endif
#ifndef ENCODER_FAST_CONV_SCALE
#define ENCODER_FAST_CONV_SCALE 1
#endif
#ifndef ENCODER_FAST_CROSS_CACHE_SCALE
#define ENCODER_FAST_CROSS_CACHE_SCALE 1
#endif
#ifndef ENCODER_CONV1_TIME_MAJOR
#define ENCODER_CONV1_TIME_MAJOR 0
#endif
#ifndef CROSS_K_POS_MAJOR
#define CROSS_K_POS_MAJOR 1
#endif
#ifndef ENCODER_ATTENTION_BATCH2
#define ENCODER_ATTENTION_BATCH2 1
#endif
#ifndef ENCODER_MATVEC_BATCH2
#define ENCODER_MATVEC_BATCH2 1
#endif
#ifndef ENCODER_MATVEC_BATCH4
#define ENCODER_MATVEC_BATCH4 0
#endif
#ifndef ENCODER_ATTENTION_VPU_DOT
#define ENCODER_ATTENTION_VPU_DOT 0
#endif
#ifndef ENCODER_ATTENTION_VPU_AXPY
#define ENCODER_ATTENTION_VPU_AXPY 0
#endif
#ifndef ENCODER_CROSS_CACHE_VPU_I8TMP
#define ENCODER_CROSS_CACHE_VPU_I8TMP 0
#endif
#ifndef FRONTEND_T0_ONLY
#define FRONTEND_T0_ONLY 1
#endif
#ifndef FRONTEND_FRAME_LIMIT
#define FRONTEND_FRAME_LIMIT AUDIO_LEN
#endif
#ifndef FRONTEND_ONLY
#define FRONTEND_ONLY 0
#endif
#ifndef FRONTEND_USE_VPU
#define FRONTEND_USE_VPU 1
#endif
#ifndef FRONTEND_SKIP_DFT
#define FRONTEND_SKIP_DFT 0
#endif
#ifndef FRONTEND_SKIP_MEL
#define FRONTEND_SKIP_MEL 0
#endif
#ifndef FRONTEND_FAKE_MEL_ACC
#define FRONTEND_FAKE_MEL_ACC 0
#endif
#ifndef FRONTEND_SKIP_LOG
#define FRONTEND_SKIP_LOG 0
#endif
#ifndef FRONTEND_MEL_K_LIMIT
#define FRONTEND_MEL_K_LIMIT 201
#endif
#ifndef FRONTEND_MEL_PROBE_ONLY
#define FRONTEND_MEL_PROBE_ONLY 0
#endif
#ifndef FRONTEND_MEL_PROBE_ROW
#define FRONTEND_MEL_PROBE_ROW 0
#endif
#ifndef FRONTEND_FAKE_POWER
#define FRONTEND_FAKE_POWER 0
#endif
#ifndef FRONTEND_FAKE_POWER_VALUE
#define FRONTEND_FAKE_POWER_VALUE 0.0f
#endif
#ifndef FRONTEND_MEL_ADD_ONLY
#define FRONTEND_MEL_ADD_ONLY 0
#endif

#define HEADS 6u
#define HEAD_DIM 64u
#define DIM 384u
#define MLP_DIM 1536u
#define MEL_BINS 80u
#define AUDIO_LEN 3000u
#define SRC_LEN 1500u
#define QKV_DIM 1152u
#define LAYERS 4u

#define BENCH_FLB 2u
#define BENCH_FCC FCC_0
#define FRONTEND_FLB 3u
#define FRONTEND_FCC FCC_1

#define SUMMARY_OFFSET 0x1000u
#define STAGE_RECORD_OFFSET 0x1800u
#define PARAM_OFFSET   0x2000u
#define FRONTEND_MAX_OFFSET 0x3000u
#define FRONTEND_MAX_STRIDE_FLOATS 16u
#define AUDIO_OFFSET   0x4000u
#define CROSS_K_OFFSET 0x100000u
#define CROSS_V_OFFSET 0x580000u
#define WAV_OFFSET     0x100000u
#define CONV1_OFFSET   0x100000u
#define SCRATCH_OFFSET 0xa00000u

#if ENCODER_MATVEC_BATCH4
#define ROW_LN_FLOATS      (4u * DIM)
#define ROW_TMP_FLOATS     (4u * DIM)
#define ROW_WIDE_FLOATS    (4u * MLP_DIM)
#else
#define ROW_LN_FLOATS      DIM
#define ROW_TMP_FLOATS     DIM
#define ROW_WIDE_FLOATS    MLP_DIM
#endif
#define ROW_CONTEXT_FLOATS DIM
#define ROW_SCORES_FLOATS  1536u
#define ROW_SLOT_FLOATS    (ROW_LN_FLOATS + ROW_CONTEXT_FLOATS + \
			    ROW_TMP_FLOATS + ROW_WIDE_FLOATS + \
			    ROW_SCORES_FLOATS)

struct run_params {
	uint32_t magic;
	uint32_t max_new_tokens;
	uint32_t suppress_eos_until;
	uint32_t input_format;
	uint32_t input_bytes;
	uint32_t reserved[11];
};

struct summary {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t active_harts;
	uint32_t src_len;
	uint32_t scratch_offset_lo;
	uint32_t scratch_offset_hi;
	uint32_t hpm3_lo;
	uint32_t hpm3_hi;
	uint32_t hpm4_lo;
	uint32_t hpm4_hi;
	uint32_t hpm5_lo;
	uint32_t hpm5_hi;
	uint32_t hpm6_lo;
	uint32_t hpm6_hi;
	uint32_t hpm7_lo;
	uint32_t hpm7_hi;
	uint32_t hpm8_lo;
	uint32_t hpm8_hi;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t done;
	uint32_t reserved[11];
};

struct pmc_snapshot {
	uint64_t hpm3;
	uint64_t hpm4;
	uint64_t hpm5;
	uint64_t hpm6;
	uint64_t hpm7;
	uint64_t hpm8;
};

struct stage_record {
	uint32_t stage_id;
	uint32_t layer;
	uint32_t step;
	uint32_t flags;
	uint32_t hpm3_lo;
	uint32_t hpm3_hi;
	uint32_t hpm4_lo;
	uint32_t hpm4_hi;
	uint32_t hpm5_lo;
	uint32_t hpm5_hi;
	uint32_t hpm6_lo;
	uint32_t hpm6_hi;
	uint32_t hpm7_lo;
	uint32_t hpm7_hi;
	uint32_t hpm8_lo;
	uint32_t hpm8_hi;
};

enum encoder_stage_id {
	ENC_STAGE_FRONTEND = 0,
	ENC_STAGE_CONV1 = 1,
	ENC_STAGE_CONV2 = 2,
	ENC_STAGE_QKV = 3,
	ENC_STAGE_ATTENTION = 4,
	ENC_STAGE_MLP = 5,
	ENC_STAGE_CROSS_CACHE = 6,
};

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
	}

	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;

	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - RUNTIME_REGION_BYTES;
	}

	return ptr;
}

static inline uint32_t active_mask_t0(void)
{
	uint32_t mask = 0u;

	for (uint32_t h = 0u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
	}

	return mask;
}

static inline uint32_t active_mask_t1(void)
{
	uint32_t mask = 0u;

	for (uint32_t h = 1u; h < ACTIVE_HARTS; h += 2u) {
		mask |= 1u << (h >> 1u);
	}

	return mask;
}

static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
	}
}

static inline void frontend_barrier(uint32_t workers)
{
	(void)workers;
	bench_barrier();
}

static inline uint64_t read_hpm3(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter3" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm4(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter4" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm5(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter5" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm6(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter6" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm7(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter7" : "=r"(v));
	return v;
}

static inline uint64_t read_hpm8(void)
{
	uint64_t v;

	__asm__ volatile("csrr %0, hpmcounter8" : "=r"(v));
	return v;
}

static inline struct pmc_snapshot read_pmc_snapshot(void)
{
	struct pmc_snapshot p;

	p.hpm3 = read_hpm3();
	p.hpm4 = read_hpm4();
	p.hpm5 = read_hpm5();
	p.hpm6 = read_hpm6();
	p.hpm7 = read_hpm7();
	p.hpm8 = read_hpm8();
	return p;
}

static inline void vpu_mask_all(void)
{
	__asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : : : "memory");
}

static inline float vpu_dot16_f32_tmp(const float *a, const float *b,
				      float *tmp8)
{
	const uint64_t zero = 0u;

	__asm__ __volatile__(
		"fbcx.ps  f0, %[zero]\n"
		"flq2     f1, 0(%[a0])\n"
		"flq2     f2, 0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 32(%[a0])\n"
		"flq2     f2, 32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2     f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [b0] "r"(b),
		  [tmp] "r"(tmp8)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0u; i < 8u; i++) {
		sum += tmp8[i];
	}
	return sum;
}

static inline float vpu_dot64_f32_tmp(const float *a, const float *b,
				      float *tmp8)
{
	const uint64_t zero = 0u;

	__asm__ __volatile__(
		"fbcx.ps  f0, %[zero]\n"
		"flq2     f1, 0(%[a0])\n"
		"flq2     f2, 0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 32(%[a0])\n"
		"flq2     f2, 32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 64(%[a0])\n"
		"flq2     f2, 64(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 96(%[a0])\n"
		"flq2     f2, 96(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 128(%[a0])\n"
		"flq2     f2, 128(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 160(%[a0])\n"
		"flq2     f2, 160(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 192(%[a0])\n"
		"flq2     f2, 192(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 224(%[a0])\n"
		"flq2     f2, 224(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2     f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [b0] "r"(b),
		  [tmp] "r"(tmp8)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0u; i < 8u; i++) {
		sum += tmp8[i];
	}
	return sum;
}

static inline void vpu_dot64_pair_f32_tmp(const float *a0, const float *a1,
					  const float *b, float *tmp0,
					  float *tmp1, float *out0,
					  float *out1)
{
	const uint64_t zero = 0u;

	__asm__ __volatile__(
		"fbcx.ps  f0, %[zero]\n"
		"fbcx.ps  f3, %[zero]\n"
		"flq2     f1, 0(%[b0])\n"
		"flq2     f2, 0(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 0(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 32(%[b0])\n"
		"flq2     f2, 32(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 32(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 64(%[b0])\n"
		"flq2     f2, 64(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 64(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 96(%[b0])\n"
		"flq2     f2, 96(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 96(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 128(%[b0])\n"
		"flq2     f2, 128(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 128(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 160(%[b0])\n"
		"flq2     f2, 160(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 160(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 192(%[b0])\n"
		"flq2     f2, 192(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 192(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"flq2     f1, 224(%[b0])\n"
		"flq2     f2, 224(%[a0])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"flq2     f4, 224(%[a1])\n"
		"fmadd.ps f3, f4, f1, f3\n"
		"fsq2     f0, 0(%[tmp0])\n"
		"fsq2     f3, 0(%[tmp1])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a0), [a1] "r"(a1),
		  [b0] "r"(b), [tmp0] "r"(tmp0), [tmp1] "r"(tmp1)
		: "memory", "f0", "f1", "f2", "f3", "f4");

	float s0 = 0.0f;
	float s1 = 0.0f;

	for (uint32_t i = 0u; i < 8u; i++) {
		s0 += tmp0[i];
		s1 += tmp1[i];
	}
	*out0 = s0;
	*out1 = s1;
}

static inline void vpu_axpy64_f32(float *acc, float scale,
				  const float *x)
{
	__asm__ __volatile__(
		"fbcx.ps  f2, %[scale]\n"
		"flq2     f0, 0(%[acc])\n"
		"flq2     f1, 0(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 0(%[acc])\n"
		"flq2     f0, 32(%[acc])\n"
		"flq2     f1, 32(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 32(%[acc])\n"
		"flq2     f0, 64(%[acc])\n"
		"flq2     f1, 64(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 64(%[acc])\n"
		"flq2     f0, 96(%[acc])\n"
		"flq2     f1, 96(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 96(%[acc])\n"
		"flq2     f0, 128(%[acc])\n"
		"flq2     f1, 128(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 128(%[acc])\n"
		"flq2     f0, 160(%[acc])\n"
		"flq2     f1, 160(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 160(%[acc])\n"
		"flq2     f0, 192(%[acc])\n"
		"flq2     f1, 192(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 192(%[acc])\n"
		"flq2     f0, 224(%[acc])\n"
		"flq2     f1, 224(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 224(%[acc])\n"
		:
		: [scale] "r"(scale), [acc] "r"(acc), [x] "r"(x)
		: "memory", "f0", "f1", "f2");
}

static inline float vpu_dot400_f32(const float *a, const float *b,
				   float *tmp8)
{
	const uint64_t zero = 0u;

	__asm__ __volatile__(
		"fbcx.ps  f0, %[zero]\n"
		"flq2     f1, 0(%[a0])\n"
		"flq2     f2, 0(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 32(%[a0])\n"
		"flq2     f2, 32(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 64(%[a0])\n"
		"flq2     f2, 64(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 96(%[a0])\n"
		"flq2     f2, 96(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 128(%[a0])\n"
		"flq2     f2, 128(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 160(%[a0])\n"
		"flq2     f2, 160(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 192(%[a0])\n"
		"flq2     f2, 192(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 224(%[a0])\n"
		"flq2     f2, 224(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 256(%[a0])\n"
		"flq2     f2, 256(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 288(%[a0])\n"
		"flq2     f2, 288(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 320(%[a0])\n"
		"flq2     f2, 320(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 352(%[a0])\n"
		"flq2     f2, 352(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 384(%[a0])\n"
		"flq2     f2, 384(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 416(%[a0])\n"
		"flq2     f2, 416(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 448(%[a0])\n"
		"flq2     f2, 448(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 480(%[a0])\n"
		"flq2     f2, 480(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 512(%[a0])\n"
		"flq2     f2, 512(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 544(%[a0])\n"
		"flq2     f2, 544(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 576(%[a0])\n"
		"flq2     f2, 576(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 608(%[a0])\n"
		"flq2     f2, 608(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 640(%[a0])\n"
		"flq2     f2, 640(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 672(%[a0])\n"
		"flq2     f2, 672(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 704(%[a0])\n"
		"flq2     f2, 704(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 736(%[a0])\n"
		"flq2     f2, 736(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 768(%[a0])\n"
		"flq2     f2, 768(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 800(%[a0])\n"
		"flq2     f2, 800(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 832(%[a0])\n"
		"flq2     f2, 832(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 864(%[a0])\n"
		"flq2     f2, 864(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 896(%[a0])\n"
		"flq2     f2, 896(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 928(%[a0])\n"
		"flq2     f2, 928(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 960(%[a0])\n"
		"flq2     f2, 960(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 992(%[a0])\n"
		"flq2     f2, 992(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1024(%[a0])\n"
		"flq2     f2, 1024(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1056(%[a0])\n"
		"flq2     f2, 1056(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1088(%[a0])\n"
		"flq2     f2, 1088(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1120(%[a0])\n"
		"flq2     f2, 1120(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1152(%[a0])\n"
		"flq2     f2, 1152(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1184(%[a0])\n"
		"flq2     f2, 1184(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1216(%[a0])\n"
		"flq2     f2, 1216(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1248(%[a0])\n"
		"flq2     f2, 1248(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1280(%[a0])\n"
		"flq2     f2, 1280(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1312(%[a0])\n"
		"flq2     f2, 1312(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1344(%[a0])\n"
		"flq2     f2, 1344(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1376(%[a0])\n"
		"flq2     f2, 1376(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1408(%[a0])\n"
		"flq2     f2, 1408(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1440(%[a0])\n"
		"flq2     f2, 1440(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1472(%[a0])\n"
		"flq2     f2, 1472(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1504(%[a0])\n"
		"flq2     f2, 1504(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1536(%[a0])\n"
		"flq2     f2, 1536(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"flq2     f1, 1568(%[a0])\n"
		"flq2     f2, 1568(%[b0])\n"
		"fmadd.ps f0, f1, f2, f0\n"
		"fsq2     f0, 0(%[tmp])\n"
		:
		: [zero] "r"(zero), [a0] "r"(a), [b0] "r"(b),
		  [tmp] "r"(tmp8)
		: "memory", "f0", "f1", "f2");

	float sum = 0.0f;

	for (uint32_t i = 0u; i < 8u; i++) {
		sum += tmp8[i];
	}
	return sum;
}

static inline float scalar_dot400_f32(const float *a, const float *b)
{
	float sum0 = 0.0f;
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	float sum3 = 0.0f;

	for (uint32_t i = 0u; i < 400u; i += 4u) {
		sum0 += a[i + 0u] * b[i + 0u];
		sum1 += a[i + 1u] * b[i + 1u];
		sum2 += a[i + 2u] * b[i + 2u];
		sum3 += a[i + 3u] * b[i + 3u];
	}

	return (sum0 + sum1) + (sum2 + sum3);
}

static inline float frontend_dot400_f32(const float *a, const float *b,
					float *tmp8)
{
#if FRONTEND_USE_VPU
	return vpu_dot400_f32(a, b, tmp8);
#else
	(void)tmp8;
	return scalar_dot400_f32(a, b);
#endif
}

static void record_stage(volatile struct stage_record *records,
			 uint32_t idx, uint32_t stage_id, uint32_t layer,
			 struct pmc_snapshot start, struct pmc_snapshot end)
{
	volatile struct stage_record *const r = &records[idx];

	r->stage_id = stage_id;
	r->layer = layer;
	r->step = 0u;
	r->flags = 0u;
	r->hpm3_lo = (uint32_t)(end.hpm3 - start.hpm3);
	r->hpm3_hi = (uint32_t)((end.hpm3 - start.hpm3) >> 32);
	r->hpm4_lo = (uint32_t)(end.hpm4 - start.hpm4);
	r->hpm4_hi = (uint32_t)((end.hpm4 - start.hpm4) >> 32);
	r->hpm5_lo = (uint32_t)(end.hpm5 - start.hpm5);
	r->hpm5_hi = (uint32_t)((end.hpm5 - start.hpm5) >> 32);
	r->hpm6_lo = (uint32_t)(end.hpm6 - start.hpm6);
	r->hpm6_hi = (uint32_t)((end.hpm6 - start.hpm6) >> 32);
	r->hpm7_lo = (uint32_t)(end.hpm7 - start.hpm7);
	r->hpm7_hi = (uint32_t)((end.hpm7 - start.hpm7) >> 32);
	r->hpm8_lo = (uint32_t)(end.hpm8 - start.hpm8);
	r->hpm8_hi = (uint32_t)((end.hpm8 - start.hpm8) >> 32);
}

static inline float abs_f32(float v)
{
	return v < 0.0f ? -v : v;
}

static inline float make_pow2(int32_t n)
{
	union {
		uint32_t u;
		float f;
	} bits;

	if (n < -126) {
		return 0.0f;
	}
	if (n > 127) {
		bits.u = 0x7f800000u;
		return bits.f;
	}

	bits.u = (uint32_t)(n + 127) << 23;
	return bits.f;
}

static float fast_recipf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	if (x == 0.0f) {
		return 3.4028234663852886e38f;
	}

	v.f = x;
	v.u = 0x7EF311C3u - v.u;
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	v.f = v.f * (2.0f - x * v.f);
	return v.f;
}

static float fast_expf(float x)
{
	if (x <= -87.33654475f) {
		return 0.0f;
	}
	if (x >= 88.72283905f) {
		return 3.4028234663852886e38f;
	}

	const float inv_ln2 = 1.44269504088896341f;
	const float ln2_hi = 0.693359375f;
	const float ln2_lo = -0.00021219444005469057f;
	const float z = x * inv_ln2;
	const int32_t n = (int32_t)(z + (z >= 0.0f ? 0.5f : -0.5f));
	const float r = (x - ((float)n * ln2_hi)) - ((float)n * ln2_lo);
	const float r2 = r * r;
	const float r3 = r2 * r;
	const float r4 = r2 * r2;
	const float r5 = r4 * r;
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3) +
			(0.0416666679f * r4) + (0.0083333310f * r5);

	return p * make_pow2(n);
}

static float fast_erff(float x)
{
	const float ax = abs_f32(x);
	const float t = fast_recipf(1.0f + 0.3275911f * ax);
	const float a1 = 0.254829592f;
	const float a2 = -0.284496736f;
	const float a3 = 1.421413741f;
	const float a4 = -1.453152027f;
	const float a5 = 1.061405429f;
	const float poly =
		(((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
	const float y = 1.0f - poly * fast_expf(-(ax * ax));

	return x < 0.0f ? -y : y;
}

static float fast_inv_sqrtf(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	v.f = x;
	v.u = 0x5f3759dfu - (v.u >> 1u);
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	v.f = v.f * (1.5f - (0.5f * x * v.f * v.f));
	return v.f;
}

static float fast_log10f(float x)
{
	union {
		float f;
		uint32_t u;
	} v;

	if (x <= 0.0f) {
		return -10.0f;
	}

	v.f = x;
	const int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127;
	v.u = (v.u & 0x007fffffu) | 0x3f800000u;
	const float m = v.f;
	const float z = (m - 1.0f) * fast_recipf(m + 1.0f);
	const float z2 = z * z;
	const float z3 = z * z2;
	const float z5 = z3 * z2;
	const float z7 = z5 * z2;
	const float z9 = z7 * z2;
	const float ln_m = 2.0f * (z + (z3 * 0.3333333333f) +
				   (z5 * 0.2f) + (z7 * 0.1428571429f) +
				   (z9 * 0.1111111111f));
	const float ln_x = ((float)exp * 0.6931471805599453f) + ln_m;

	return ln_x * 0.4342944819032518f;
}

static uint16_t float_to_half(float f)
{
	union {
		float f;
		uint32_t u;
	} v;
	uint32_t sign;
	int32_t exp;
	uint32_t mant;

	v.f = f;
	sign = (v.u >> 16) & 0x8000u;
	exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
	mant = v.u & 0x7fffffu;

	if (exp <= 0) {
		if (exp < -10) {
			return (uint16_t)sign;
		}
		mant |= 0x800000u;
		const uint32_t shift = (uint32_t)(14 - exp);
		uint32_t hmant = mant >> shift;
		if ((mant >> (shift - 1u)) & 1u) {
			hmant++;
		}
		return (uint16_t)(sign | hmant);
	}
	if (exp >= 31) {
		return (uint16_t)(sign | 0x7c00u);
	}

	mant += 0x1000u;
	if (mant & 0x800000u) {
		mant = 0u;
		exp++;
	}
	if (exp >= 31) {
		return (uint16_t)(sign | 0x7c00u);
	}
	return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static inline const int8_t *wptr(uint8_t *base, uint32_t off)
{
	return (const int8_t *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline const float *fptr(uint8_t *base, uint32_t off)
{
	return (const float *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline const float *rptr(uint8_t *base, uint32_t off)
{
	return (const float *)(base + off);
}

static inline float wval(uint8_t *base, struct weight_desc w, uint32_t idx)
{
	return (float)wptr(base, w.offset)[idx] * w.scale;
}

static uint32_t rd32(const uint8_t *p)
{
	return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const uint8_t *p)
{
	return (uint16_t)(((uint32_t)p[0]) | ((uint32_t)p[1] << 8));
}

static int16_t pcm16_at(const uint8_t *p, uint32_t sample)
{
	return (int16_t)rd16(p + sample * 2u);
}

static const uint8_t *wav_pcm16_data(uint8_t *base, uint32_t bytes,
				     uint32_t *sample_count,
				     uint32_t *status)
{
	const uint8_t *const wav = base + WAV_OFFSET;
	uint32_t off = 12u;

	*sample_count = 0u;
	*status = 0u;
	if (bytes < 44u || rd32(wav) != 0x46464952u ||
	    rd32(wav + 8u) != 0x45564157u) {
		*status = 1u;
		return (const uint8_t *)0;
	}

	while (off + 8u <= bytes) {
		const uint32_t tag = rd32(wav + off);
		const uint32_t len = rd32(wav + off + 4u);
		const uint32_t data = off + 8u;

		if (data + len > bytes) {
			*status = 2u;
			return (const uint8_t *)0;
		}
		if (tag == 0x20746d66u) {
			if (len < 16u) {
				*status = 3u;
				return (const uint8_t *)0;
			}
			const uint16_t audio_format = rd16(wav + data);
			const uint16_t channels = rd16(wav + data + 2u);
			const uint32_t sample_rate = rd32(wav + data + 4u);
			const uint16_t bits = rd16(wav + data + 14u);

			if (audio_format != 1u || channels != 1u ||
			    sample_rate != 16000u || bits != 16u) {
				*status = 4u;
				return (const uint8_t *)0;
			}
		} else if (tag == 0x61746164u) {
			*sample_count = len / 2u;
			*status = 100u;
			return wav + data;
		}
		off = data + len + (len & 1u);
	}

	*status = 5u;
	return (const uint8_t *)0;
}

static float wav_sample_reflect(const uint8_t *pcm, uint32_t samples,
				int32_t idx)
{
	if (samples == 0u) {
		return 0.0f;
	}
	if (idx < 0) {
		idx = -idx;
	}
	if (idx >= 480000) {
		idx = (int32_t)(960000u) - idx - 2;
	}
	if (idx < 0 || idx >= 480000 || idx >= (int32_t)samples) {
		return 0.0f;
	}

	return (float)pcm16_at(pcm, (uint32_t)idx) * (1.0f / 32767.0f);
}

static void wav_frontend_stage(uint8_t *base, float *features,
			       uint32_t input_bytes, uint32_t hart_id,
			       float *scratch,
			       volatile struct summary *summary)
{
	uint32_t samples = 0u;
	uint32_t status = 0u;
	const uint8_t *const pcm =
		wav_pcm16_data(base, input_bytes, &samples, &status);
	const float *const hann = rptr(base, frontend_hann_offset);
	const float *const mel = rptr(base, frontend_mel_filters_offset);
	const float *const dct_cos = rptr(base, frontend_dft_cos_offset);
	const float *const dct_sin = rptr(base, frontend_dft_sin_offset);
	volatile float *const max_slots =
		(volatile float *)(base + FRONTEND_MAX_OFFSET);
	float *const frame = scratch;
	float *const power = frame + 400u;
	float *const dot_tmp = power + 208u;
	float local_max = -10.0f;
#if FRONTEND_T0_ONLY
	const uint32_t worker_id = hart_id >> 1u;
	const uint32_t workers = (ACTIVE_HARTS + 1u) >> 1u;
	const uint32_t is_frontend_worker = ((hart_id & 1u) == 0u);
#else
	const uint32_t worker_id = hart_id;
	const uint32_t workers = ACTIVE_HARTS;
	const uint32_t is_frontend_worker = 1u;
#endif
	const uint32_t frontend_frames =
		FRONTEND_FRAME_LIMIT < AUDIO_LEN ? FRONTEND_FRAME_LIMIT : AUDIO_LEN;
	const uint32_t chunk = (frontend_frames + workers - 1u) / workers;
	const uint32_t t0_raw = worker_id * chunk;
	const uint32_t t1_raw = t0_raw + chunk;
	const uint32_t t0 = t0_raw < frontend_frames ? t0_raw : frontend_frames;
	const uint32_t t1 = t1_raw < frontend_frames ? t1_raw : frontend_frames;

	if (hart_id == 0u) {
		summary->reserved[2] = status;
		summary->reserved[3] = samples;
		summary->reserved[4] = input_bytes;
		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}
	if (samples > 480000u) {
		samples = 480000u;
	}
	if (pcm == (const uint8_t *)0) {
		if (is_frontend_worker) {
			for (uint32_t t = t0; t < t1; t++) {
				for (uint32_t m = 0u; m < MEL_BINS; m++) {
					features[m * AUDIO_LEN + t] = -1.5f;
				}
			}
		}
		return;
	}

	if (hart_id == 0u) {
		features[0] = 0.125f;
		summary->reserved[7] = 10u;
		FENCE;
		evict(features, sizeof(float));
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	if (is_frontend_worker) {
	for (uint32_t t = t0; t < t1; t++) {
		const int32_t frame0 = (int32_t)(t * 160u) - 200;

		for (uint32_t n = 0u; n < 400u; n++) {
			frame[n] = wav_sample_reflect(
				pcm, samples, frame0 + (int32_t)n) * hann[n];
		}
		for (uint32_t k = 0u; k < 201u; k++) {
			const float *const cos_row = dct_cos + k * 400u;
			const float *const sin_row = dct_sin + k * 400u;

			if (hart_id == 0u && t == t0 && k == 0u) {
				summary->reserved[7] = 11u;
				FENCE;
				evict((void *)summary, sizeof(*summary));
				WAIT_CACHEOPS;
			}
			float re = 0.0f;
			float im = 0.0f;
#if !FRONTEND_SKIP_DFT
			re = frontend_dot400_f32(frame, cos_row, dot_tmp);
			im = -frontend_dot400_f32(frame, sin_row, dot_tmp);
#else
			(void)cos_row;
			(void)sin_row;
			(void)dot_tmp;
#endif

			if (hart_id == 0u && t == t0 && k == 0u) {
				summary->reserved[7] = 12u;
				FENCE;
				evict((void *)summary, sizeof(*summary));
				WAIT_CACHEOPS;
			}
			power[k] = (re * re) + (im * im);
			if (hart_id == 0u && t == t0 && k == 0u) {
				features[1] = power[k];
				summary->reserved[7] = 13u;
				FENCE;
				evict(features, 2u * sizeof(float));
				evict((void *)summary, sizeof(*summary));
				WAIT_CACHEOPS;
			}
		}

#if FRONTEND_MEL_PROBE_ONLY
		if (hart_id == 0u && t == t0) {
			features[2] = mel[0];
			summary->reserved[7] = 16u;
			FENCE;
			evict(features, 3u * sizeof(float));
			evict((void *)summary, sizeof(*summary));
			WAIT_CACHEOPS;
		}
#elif FRONTEND_MEL_PROBE_ROW
		if (hart_id == 0u && t == t0) {
			float probe = 0.0f;

			for (uint32_t m = 0u; m < MEL_BINS; m++) {
				probe += mel[m];
			}
			features[2] = probe;
			summary->reserved[7] = 17u;
			FENCE;
			evict(features, 3u * sizeof(float));
			evict((void *)summary, sizeof(*summary));
			WAIT_CACHEOPS;
		}
#else
#if !FRONTEND_SKIP_MEL
		for (uint32_t m = 0u; m < MEL_BINS; m++) {
			float acc = 0.0f;

#if !FRONTEND_FAKE_MEL_ACC
			for (uint32_t k = 0u; k < FRONTEND_MEL_K_LIMIT; k++) {
#if FRONTEND_FAKE_POWER
				const float pwr = FRONTEND_FAKE_POWER_VALUE;
#else
				const float pwr = power[k];
#endif
#if FRONTEND_MEL_ADD_ONLY
				(void)pwr;
				acc += mel[k * MEL_BINS + m];
#else
				acc += mel[k * MEL_BINS + m] * pwr;
#endif
			}
#else
			(void)mel;
			acc = 1.0e-10f;
#endif
			if (acc < 1.0e-10f) {
				acc = 1.0e-10f;
			}
#if FRONTEND_SKIP_LOG
			const float logv = -10.0f;
#else
			const float logv = fast_log10f(acc);
#endif
			features[m * AUDIO_LEN + t] = logv;
			if (hart_id == 0u && t == t0 && m == 0u) {
				summary->reserved[7] = 14u;
				FENCE;
				evict(features, sizeof(float));
				evict((void *)summary, sizeof(*summary));
				WAIT_CACHEOPS;
			}
			if (logv > local_max) {
				local_max = logv;
			}
		}
#endif
#endif
		if (hart_id == 0u && t == t0) {
			summary->reserved[7] = 15u;
			FENCE;
			evict((void *)summary, sizeof(*summary));
			WAIT_CACHEOPS;
		}
	}
	}

	if (is_frontend_worker) {
		max_slots[worker_id * FRONTEND_MAX_STRIDE_FLOATS] = local_max;
		FENCE;
		evict((void *)&max_slots[worker_id * FRONTEND_MAX_STRIDE_FLOATS],
		      64u);
		WAIT_CACHEOPS;
	}
	frontend_barrier(workers);

	if (hart_id == 0u) {
		float max_log = -10.0f;

		evict((void *)max_slots,
		      workers * FRONTEND_MAX_STRIDE_FLOATS * sizeof(float));
		WAIT_CACHEOPS;
		for (uint32_t h = 0u; h < workers; h++) {
			const float v = max_slots[h * FRONTEND_MAX_STRIDE_FLOATS];

			if (v > max_log) {
				max_log = v;
			}
		}
		max_slots[0] = max_log;
		FENCE;
		evict((void *)max_slots, sizeof(float));
		WAIT_CACHEOPS;
	}
	frontend_barrier(workers);
	evict((void *)max_slots, sizeof(float));
	WAIT_CACHEOPS;

	const float floor = max_slots[0] - 8.0f;
	if (is_frontend_worker) {
	for (uint32_t t = t0; t < t1; t++) {
		for (uint32_t m = 0u; m < MEL_BINS; m++) {
			float v = features[m * AUDIO_LEN + t];

			if (v < floor) {
				v = floor;
			}
			features[m * AUDIO_LEN + t] = (v + 4.0f) * 0.25f;
		}
	}
	}
}

static void evict_frontend_feature_slice(float *features, uint32_t hart_id)
{
#if FRONTEND_T0_ONLY
	const uint32_t worker_id = hart_id >> 1u;
	const uint32_t workers = (ACTIVE_HARTS + 1u) >> 1u;
	const uint32_t is_frontend_worker = ((hart_id & 1u) == 0u);
#else
	const uint32_t worker_id = hart_id;
	const uint32_t workers = ACTIVE_HARTS;
	const uint32_t is_frontend_worker = 1u;
#endif
	const uint32_t frontend_frames =
		FRONTEND_FRAME_LIMIT < AUDIO_LEN ? FRONTEND_FRAME_LIMIT : AUDIO_LEN;
	const uint32_t chunk = (frontend_frames + workers - 1u) / workers;
	const uint32_t t0_raw = worker_id * chunk;
	const uint32_t t1_raw = t0_raw + chunk;
	const uint32_t t0 = t0_raw < frontend_frames ? t0_raw : frontend_frames;
	const uint32_t t1 = t1_raw < frontend_frames ? t1_raw : frontend_frames;

	if (!is_frontend_worker || t1 <= t0) {
		return;
	}

	FENCE;
	for (uint32_t m = 0u; m < MEL_BINS; m++) {
		evict(features + (m * AUDIO_LEN) + t0,
		      (t1 - t0) * sizeof(float));
	}
	WAIT_CACHEOPS;
}

static inline float gelu(float x)
{
	return 0.5f * x * (1.0f + fast_erff(x * 0.70710678118654757f));
}

static void layernorm(uint8_t *base, float *out, const float *in,
		      struct weight_desc gamma, struct weight_desc beta)
{
	float sum = 0.0f;
	float sumsq = 0.0f;

	for (uint32_t i = 0u; i < DIM; i++) {
		const float v = in[i];
		sum += v;
		sumsq += v * v;
	}

	const float mean = sum * (1.0f / (float)DIM);
	const float var = (sumsq * (1.0f / (float)DIM)) - (mean * mean);
	const float inv = fast_inv_sqrtf(var + 0.00001f);

	for (uint32_t i = 0u; i < DIM; i++) {
		out[i] = ((in[i] - mean) * inv) * wval(base, gamma, i) +
			 wval(base, beta, i);
	}
}

static void matvec_i8(uint8_t *base, float *out, const float *in,
		      uint32_t in_dim, uint32_t out_dim,
		      struct weight_desc weight, struct weight_desc bias)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);

#if ENCODER_ROW_MAJOR_MATVEC
	for (uint32_t n = 0u; n < out_dim; n++) {
		out[n] = (float)b[n] * bias.scale;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled_in = in[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			out[n] += scaled_in * (float)wrow[n];
		}
	}
#else
	for (uint32_t n = 0u; n < out_dim; n++) {
		float acc = (float)b[n] * bias.scale;

		for (uint32_t k = 0u; k < in_dim; k++) {
			acc += in[k] * ((float)wt[k * out_dim + n] * weight.scale);
		}
		out[n] = acc;
	}
#endif
}

static void matvec_i8_nobias(uint8_t *base, float *out, const float *in,
			     uint32_t in_dim, uint32_t out_dim,
			     struct weight_desc weight)
{
	const int8_t *const wt = wptr(base, weight.offset);

#if ENCODER_ROW_MAJOR_MATVEC
	for (uint32_t n = 0u; n < out_dim; n++) {
		out[n] = 0.0f;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled_in = in[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			out[n] += scaled_in * (float)wrow[n];
		}
	}
#else
	for (uint32_t n = 0u; n < out_dim; n++) {
		float acc = 0.0f;

		for (uint32_t k = 0u; k < in_dim; k++) {
			acc += in[k] * ((float)wt[k * out_dim + n] * weight.scale);
		}
		out[n] = acc;
	}
#endif
}

#if ENCODER_MATVEC_BATCH4
static void matvec_i8_four(uint8_t *base,
			   float *out0, float *out1, float *out2, float *out3,
			   const float *in0, const float *in1,
			   const float *in2, const float *in3,
			   uint32_t in_dim, uint32_t out_dim,
			   struct weight_desc weight, struct weight_desc bias)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		const float bias_v = (float)b[n] * bias.scale;

		out0[n] = bias_v;
		out1[n] = bias_v;
		out2[n] = bias_v;
		out3[n] = bias_v;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled0 = in0[k] * weight.scale;
		const float scaled1 = in1[k] * weight.scale;
		const float scaled2 = in2[k] * weight.scale;
		const float scaled3 = in3[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			const float w = (float)wrow[n];

			out0[n] += scaled0 * w;
			out1[n] += scaled1 * w;
			out2[n] += scaled2 * w;
			out3[n] += scaled3 * w;
		}
	}
}

static void matvec_i8_nobias_four(uint8_t *base,
				  float *out0, float *out1,
				  float *out2, float *out3,
				  const float *in0, const float *in1,
				  const float *in2, const float *in3,
				  uint32_t in_dim, uint32_t out_dim,
				  struct weight_desc weight)
{
	const int8_t *const wt = wptr(base, weight.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		out0[n] = 0.0f;
		out1[n] = 0.0f;
		out2[n] = 0.0f;
		out3[n] = 0.0f;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled0 = in0[k] * weight.scale;
		const float scaled1 = in1[k] * weight.scale;
		const float scaled2 = in2[k] * weight.scale;
		const float scaled3 = in3[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			const float w = (float)wrow[n];

			out0[n] += scaled0 * w;
			out1[n] += scaled1 * w;
			out2[n] += scaled2 * w;
			out3[n] += scaled3 * w;
		}
	}
}
#endif

#if ENCODER_MATVEC_BATCH2
static void matvec_i8_pair(uint8_t *base, float *out0, float *out1,
			   const float *in0, const float *in1,
			   uint32_t in_dim, uint32_t out_dim,
			   struct weight_desc weight, struct weight_desc bias)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		const float bias_v = (float)b[n] * bias.scale;

		out0[n] = bias_v;
		out1[n] = bias_v;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled0 = in0[k] * weight.scale;
		const float scaled1 = in1[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			const float w = (float)wrow[n];

			out0[n] += scaled0 * w;
			out1[n] += scaled1 * w;
		}
	}
}

static void matvec_i8_nobias_pair(uint8_t *base, float *out0, float *out1,
				  const float *in0, const float *in1,
				  uint32_t in_dim, uint32_t out_dim,
				  struct weight_desc weight)
{
	const int8_t *const wt = wptr(base, weight.offset);

	for (uint32_t n = 0u; n < out_dim; n++) {
		out0[n] = 0.0f;
		out1[n] = 0.0f;
	}

	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float scaled0 = in0[k] * weight.scale;
		const float scaled1 = in1[k] * weight.scale;

		for (uint32_t n = 0u; n < out_dim; n++) {
			const float w = (float)wrow[n];

			out0[n] += scaled0 * w;
			out1[n] += scaled1 * w;
		}
	}
}
#endif

static void softmax(float *scores, uint32_t count)
{
	float maxv = -3.4028234663852886e38f;
	float sum = 0.0f;

	for (uint32_t i = 0u; i < count; i++) {
		if (scores[i] > maxv) {
			maxv = scores[i];
		}
	}
	for (uint32_t i = 0u; i < count; i++) {
		const float e = fast_expf(scores[i] - maxv);
		scores[i] = e;
		sum += e;
	}
	const float inv = fast_recipf(sum);
	for (uint32_t i = 0u; i < count; i++) {
		scores[i] *= inv;
	}
}

static inline uint32_t part0(uint32_t n, uint32_t hart_id)
{
	return (n * hart_id) / ACTIVE_HARTS;
}

static inline uint32_t part1(uint32_t n, uint32_t hart_id)
{
	return (n * (hart_id + 1u)) / ACTIVE_HARTS;
}

static inline uint32_t cross_k_index(uint32_t layer, uint32_t head,
				     uint32_t dim, uint32_t pos)
{
#if CROSS_K_POS_MAJOR
	return (((layer * HEADS + head) * SRC_LEN + pos) * HEAD_DIM) + dim;
#else
	return (((layer * HEADS + head) * HEAD_DIM + dim) * SRC_LEN) + pos;
#endif
}

static inline uint32_t cross_v_index(uint32_t layer, uint32_t head,
				     uint32_t pos, uint32_t dim)
{
	return (((layer * HEADS + head) * SRC_LEN + pos) * HEAD_DIM) + dim;
}

static void conv1_stage(uint8_t *base, uint32_t hart_id, float *audio,
			float *conv1)
{
	const int8_t *const wt = wptr(base, enc_conv1_weight.offset);
	const float *const b = fptr(base, enc_conv1_bias_f32_offset);
	const uint32_t total = DIM * AUDIO_LEN;
	const uint32_t i0 = part0(total, hart_id);
	const uint32_t i1 = part1(total, hart_id);

	for (uint32_t idx = i0; idx < i1; idx++) {
#if ENCODER_CONV1_TIME_MAJOR
		const uint32_t t = idx / DIM;
		const uint32_t oc = idx - t * DIM;
#else
		const uint32_t oc = idx / AUDIO_LEN;
		const uint32_t t = idx - oc * AUDIO_LEN;
#endif
		float acc = b[oc];
#if ENCODER_FAST_CONV_SCALE
		float raw = 0.0f;
#endif

		for (uint32_t ic = 0u; ic < MEL_BINS; ic++) {
			for (uint32_t k = 0u; k < 3u; k++) {
				const int32_t src_t = (int32_t)t + (int32_t)k - 1;

				if (src_t >= 0 && src_t < (int32_t)AUDIO_LEN) {
					const uint32_t widx =
						((oc * MEL_BINS + ic) * 3u) + k;
#if ENCODER_FAST_CONV_SCALE
					raw += audio[ic * AUDIO_LEN + (uint32_t)src_t] *
					       (float)wt[widx];
#else
					acc += audio[ic * AUDIO_LEN + (uint32_t)src_t] *
					       wval(base, enc_conv1_weight, widx);
#endif
				}
			}
		}
#if ENCODER_FAST_CONV_SCALE
		acc += raw * enc_conv1_weight.scale;
#endif
#if ENCODER_CONV1_TIME_MAJOR
		conv1[t * DIM + oc] = gelu(acc);
#else
		conv1[idx] = gelu(acc);
#endif
	}
}

static void conv2_stage(uint8_t *base, uint32_t hart_id, float *conv1,
			float *hidden)
{
	const int8_t *const wt = wptr(base, enc_conv2_weight.offset);
	const float *const b = fptr(base, enc_conv2_bias_f32_offset);
	const uint32_t total = SRC_LEN * DIM;
	const uint32_t i0 = part0(total, hart_id);
	const uint32_t i1 = part1(total, hart_id);

	for (uint32_t idx = i0; idx < i1; idx++) {
		const uint32_t t = idx / DIM;
		const uint32_t oc = idx - t * DIM;
		float acc = b[oc];
#if ENCODER_FAST_CONV_SCALE
		float raw = 0.0f;
#endif

		for (uint32_t ic = 0u; ic < DIM; ic++) {
			for (uint32_t k = 0u; k < 3u; k++) {
				const int32_t src_t =
					(int32_t)(t * 2u) + (int32_t)k - 1;

				if (src_t >= 0 && src_t < (int32_t)AUDIO_LEN) {
					const uint32_t widx =
						((oc * DIM + ic) * 3u) + k;
#if ENCODER_FAST_CONV_SCALE
					raw +=
#if ENCODER_CONV1_TIME_MAJOR
					       conv1[(uint32_t)src_t * DIM + ic] *
#else
					       conv1[ic * AUDIO_LEN + (uint32_t)src_t] *
#endif
					       (float)wt[widx];
#else
					acc +=
#if ENCODER_CONV1_TIME_MAJOR
					       conv1[(uint32_t)src_t * DIM + ic] *
#else
					       conv1[ic * AUDIO_LEN + (uint32_t)src_t] *
#endif
					       wval(base, enc_conv2_weight, widx);
#endif
				}
			}
		}
#if ENCODER_FAST_CONV_SCALE
		acc += raw * enc_conv2_weight.scale;
#endif
		hidden[t * DIM + oc] = gelu(acc) +
				       wval(base, enc_positional_embedding, idx);
	}
}

static void qkv_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
		      float *hidden, float *qkv, float *ln)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);
	float *const ln1 = ln + DIM;
	float *const ln2 = ln1 + DIM;
	float *const ln3 = ln2 + DIM;

	for (uint32_t row = r0; row < r1;) {
		float *const dst = qkv + row * QKV_DIM;

#if ENCODER_MATVEC_BATCH4
		if (row + 3u < r1) {
			float *const dst1 = dst + QKV_DIM;
			float *const dst2 = dst1 + QKV_DIM;
			float *const dst3 = dst2 + QKV_DIM;

			layernorm(base, ln, hidden + row * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			layernorm(base, ln1, hidden + (row + 1u) * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			layernorm(base, ln2, hidden + (row + 2u) * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			layernorm(base, ln3, hidden + (row + 3u) * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			matvec_i8_nobias_four(base, dst, dst1, dst2, dst3,
					      ln, ln1, ln2, ln3, DIM,
					      QKV_DIM, d.self_qkv_weight);
			for (uint32_t i = 0u; i < DIM; i++) {
				const float qb = wval(base, d.self_q_bias, i);
				const float vb = wval(base, d.self_v_bias, i);

				dst[i] += qb;
				dst[768u + i] += vb;
				dst1[i] += qb;
				dst1[768u + i] += vb;
				dst2[i] += qb;
				dst2[768u + i] += vb;
				dst3[i] += qb;
				dst3[768u + i] += vb;
			}
			row += 4u;
			continue;
		}
#endif
#if ENCODER_MATVEC_BATCH2
		if (row + 1u < r1) {
			float *const dst1 = dst + QKV_DIM;

			layernorm(base, ln, hidden + row * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			layernorm(base, ln1, hidden + (row + 1u) * DIM,
				  d.attn_ln_weight, d.attn_ln_bias);
			matvec_i8_nobias_pair(base, dst, dst1, ln, ln1, DIM,
					      QKV_DIM, d.self_qkv_weight);
			for (uint32_t i = 0u; i < DIM; i++) {
				const float qb = wval(base, d.self_q_bias, i);
				const float vb = wval(base, d.self_v_bias, i);

				dst[i] += qb;
				dst[768u + i] += vb;
				dst1[i] += qb;
				dst1[768u + i] += vb;
			}
			row += 2u;
			continue;
		}
#endif
		layernorm(base, ln, hidden + row * DIM, d.attn_ln_weight,
			  d.attn_ln_bias);
		matvec_i8_nobias(base, dst, ln, DIM, QKV_DIM, d.self_qkv_weight);
		for (uint32_t i = 0u; i < DIM; i++) {
			dst[i] += wval(base, d.self_q_bias, i);
			dst[768u + i] += wval(base, d.self_v_bias, i);
		}
		row++;
	}
}

static void attention_stage_one(uint8_t *base, uint32_t layer, uint32_t row,
				float *hidden, float *qkv, float *context,
				float *scores, float *tmp)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const float scale = wval(base, enc_attn_scale, 0u);
	const float score_scale = scale * scale;

	for (uint32_t i = 0u; i < DIM; i++) {
		context[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q = qkv + row * QKV_DIM + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			const float *const k =
				qkv + t * QKV_DIM + DIM + h * HEAD_DIM;
			float s = 0.0f;

#if ENCODER_ATTENTION_VPU_DOT
			s = vpu_dot64_f32_tmp(q, k, tmp);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * k[i];
			}
#endif
			scores[t] = s * score_scale;
		}
		softmax(scores, SRC_LEN);
		float *const head_context = context + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			const float score = scores[t];
			const float *const vrow =
				qkv + t * QKV_DIM + 768u + h * HEAD_DIM;

#if ENCODER_ATTENTION_VPU_AXPY
			vpu_axpy64_f32(head_context, score, vrow);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				head_context[i] += score * vrow[i];
			}
#endif
		}
	}

	matvec_i8(base, tmp, context, DIM, DIM, d.self_out_weight,
		  d.self_out_bias);
	for (uint32_t i = 0u; i < DIM; i++) {
		hidden[row * DIM + i] += tmp[i];
	}
}

#if ENCODER_ATTENTION_BATCH2
static void attention_stage_two(uint8_t *base, uint32_t layer, uint32_t row,
				float *hidden, float *qkv, float *context0,
				float *context1, float *scores0,
				float *scores1, float *tmp)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const float scale = wval(base, enc_attn_scale, 0u);
	const float score_scale = scale * scale;

	for (uint32_t i = 0u; i < DIM; i++) {
		context0[i] = 0.0f;
		context1[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q0 = qkv + row * QKV_DIM + h * HEAD_DIM;
		const float *const q1 = qkv + (row + 1u) * QKV_DIM + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			const float *const k =
				qkv + t * QKV_DIM + DIM + h * HEAD_DIM;
			float s0 = 0.0f;
			float s1 = 0.0f;

#if ENCODER_ATTENTION_VPU_DOT
			vpu_dot64_pair_f32_tmp(q0, q1, k, tmp, tmp + 8u,
					       &s0, &s1);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				const float kv = k[i];

				s0 += q0[i] * kv;
				s1 += q1[i] * kv;
			}
#endif
			scores0[t] = s0 * score_scale;
			scores1[t] = s1 * score_scale;
		}
		softmax(scores0, SRC_LEN);
		softmax(scores1, SRC_LEN);
		float *const head_context0 = context0 + h * HEAD_DIM;
		float *const head_context1 = context1 + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			const float score0 = scores0[t];
			const float score1 = scores1[t];
			const float *const vrow =
				qkv + t * QKV_DIM + 768u + h * HEAD_DIM;

#if ENCODER_ATTENTION_VPU_AXPY
			vpu_axpy64_f32(head_context0, score0, vrow);
			vpu_axpy64_f32(head_context1, score1, vrow);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				const float vv = vrow[i];

				head_context0[i] += score0 * vv;
				head_context1[i] += score1 * vv;
			}
#endif
		}
	}

	matvec_i8(base, tmp, context0, DIM, DIM, d.self_out_weight,
		  d.self_out_bias);
	for (uint32_t i = 0u; i < DIM; i++) {
		hidden[row * DIM + i] += tmp[i];
	}
	matvec_i8(base, tmp, context1, DIM, DIM, d.self_out_weight,
		  d.self_out_bias);
	for (uint32_t i = 0u; i < DIM; i++) {
		hidden[(row + 1u) * DIM + i] += tmp[i];
	}
}
#endif

static void attention_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
			    float *hidden, float *qkv, float *context,
			    float *context1, float *scores, float *scores1,
			    float *tmp)
{
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);

	for (uint32_t row = r0; row < r1;) {
#if ENCODER_ATTENTION_BATCH2
		if (row + 1u < r1) {
			attention_stage_two(base, layer, row, hidden, qkv,
					    context, context1, scores, scores1,
					    tmp);
			row += 2u;
			continue;
		}
#endif
		attention_stage_one(base, layer, row, hidden, qkv, context,
				    scores, tmp);
		row++;
	}
}

static void mlp_stage(uint8_t *base, uint32_t hart_id, uint32_t layer,
		      float *hidden, float *ln, float *ln1, float *wide,
		      float *wide1, float *tmp)
{
	const struct encoder_layer_desc d = encoder_layers[layer];
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);
	float *const ln2 = ln + (2u * DIM);
	float *const ln3 = ln + (3u * DIM);

	for (uint32_t row = r0; row < r1;) {
#if ENCODER_MATVEC_BATCH4
		if (row + 3u < r1) {
			float *const ln1b = ln + DIM;
			float *const wide1b = wide + MLP_DIM;
			float *const wide2 = wide + (2u * MLP_DIM);
			float *const wide3 = wide + (3u * MLP_DIM);
			float *const tmp1 = tmp + DIM;
			float *const tmp2 = tmp + (2u * DIM);
			float *const tmp3 = tmp + (3u * DIM);

			layernorm(base, ln, hidden + row * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			layernorm(base, ln1b, hidden + (row + 1u) * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			layernorm(base, ln2, hidden + (row + 2u) * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			layernorm(base, ln3, hidden + (row + 3u) * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			matvec_i8_four(base, wide, wide1b, wide2, wide3,
				       ln, ln1b, ln2, ln3, DIM, MLP_DIM,
				       d.mlp_fc1_weight, d.mlp_fc1_bias);
			for (uint32_t i = 0u; i < MLP_DIM; i++) {
				wide[i] = gelu(wide[i]);
				wide1b[i] = gelu(wide1b[i]);
				wide2[i] = gelu(wide2[i]);
				wide3[i] = gelu(wide3[i]);
			}
			matvec_i8_four(base, tmp, tmp1, tmp2, tmp3,
				       wide, wide1b, wide2, wide3, MLP_DIM, DIM,
				       d.mlp_fc2_weight, d.mlp_fc2_bias);
			for (uint32_t i = 0u; i < DIM; i++) {
				hidden[row * DIM + i] += tmp[i];
				hidden[(row + 1u) * DIM + i] += tmp1[i];
				hidden[(row + 2u) * DIM + i] += tmp2[i];
				hidden[(row + 3u) * DIM + i] += tmp3[i];
			}
			row += 4u;
			continue;
		}
#endif
#if ENCODER_MATVEC_BATCH2
		if (row + 1u < r1) {
			layernorm(base, ln, hidden + row * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			layernorm(base, ln1, hidden + (row + 1u) * DIM,
				  d.mlp_ln_weight, d.mlp_ln_bias);
			matvec_i8_pair(base, wide, wide1, ln, ln1, DIM,
				       MLP_DIM, d.mlp_fc1_weight,
				       d.mlp_fc1_bias);
			for (uint32_t i = 0u; i < MLP_DIM; i++) {
				wide[i] = gelu(wide[i]);
				wide1[i] = gelu(wide1[i]);
			}
			matvec_i8_pair(base, tmp, ln1, wide, wide1, MLP_DIM,
				       DIM, d.mlp_fc2_weight,
				       d.mlp_fc2_bias);
			for (uint32_t i = 0u; i < DIM; i++) {
				hidden[row * DIM + i] += tmp[i];
				hidden[(row + 1u) * DIM + i] += ln1[i];
			}
			row += 2u;
			continue;
		}
#endif
		layernorm(base, ln, hidden + row * DIM, d.mlp_ln_weight,
			  d.mlp_ln_bias);
		matvec_i8(base, wide, ln, DIM, MLP_DIM, d.mlp_fc1_weight,
			  d.mlp_fc1_bias);
		for (uint32_t i = 0u; i < MLP_DIM; i++) {
			wide[i] = gelu(wide[i]);
		}
		matvec_i8(base, tmp, wide, MLP_DIM, DIM, d.mlp_fc2_weight,
			  d.mlp_fc2_bias);
		for (uint32_t i = 0u; i < DIM; i++) {
			hidden[row * DIM + i] += tmp[i];
		}
		row++;
	}
}

static void cross_cache_stage(uint8_t *base, uint32_t hart_id, float *hidden,
			      float *ln, float *kbuf, float *vbuf)
{
	uint16_t *const cross_k = (uint16_t *)(base + CROSS_K_OFFSET);
	uint16_t *const cross_v = (uint16_t *)(base + CROSS_V_OFFSET);
	const uint32_t r0 = part0(SRC_LEN, hart_id);
	const uint32_t r1 = part1(SRC_LEN, hart_id);

	for (uint32_t row = r0; row < r1; row++) {
		layernorm(base, ln, hidden + row * DIM, enc_ln_post_weight,
			  enc_ln_post_bias);

		for (uint32_t layer = 0u; layer < LAYERS; layer++) {
			for (uint32_t h = 0u; h < HEADS; h++) {
				const struct weight_desc kw =
					enc_cross_k_weight[layer][h];
				const struct weight_desc vw =
					enc_cross_v_weight[layer][h];
				const struct weight_desc vb =
					enc_cross_v_bias[layer][h];
				const int8_t *const kwt = wptr(base, kw.offset);
				const int8_t *const vwt = wptr(base, vw.offset);
				const int8_t *const vbt = wptr(base, vb.offset);

#if ENCODER_FAST_CROSS_CACHE_SCALE
				for (uint32_t d = 0u; d < HEAD_DIM; d++) {
					kbuf[d] = 0.0f;
					vbuf[d] = 0.0f;
				}

				for (uint32_t i = 0u; i < DIM; i++) {
					const float x = ln[i];
					const int8_t *const krow = kwt + i * HEAD_DIM;
					const int8_t *const vrow = vwt + i * HEAD_DIM;

#if ENCODER_CROSS_CACHE_VPU_I8TMP
					float *const ktmp = kbuf + HEAD_DIM;
					float *const vtmp = vbuf + HEAD_DIM;

					for (uint32_t d = 0u; d < HEAD_DIM; d++) {
						ktmp[d] = (float)krow[d];
						vtmp[d] = (float)vrow[d];
					}
					vpu_axpy64_f32(kbuf, x, ktmp);
					vpu_axpy64_f32(vbuf, x, vtmp);
#else
					for (uint32_t d = 0u; d < HEAD_DIM; d++) {
						kbuf[d] += x * (float)krow[d];
						vbuf[d] += x * (float)vrow[d];
					}
#endif
				}

				for (uint32_t d = 0u; d < HEAD_DIM; d++) {
					cross_k[cross_k_index(layer, h, d, row)] =
						float_to_half(kbuf[d] * kw.scale);
					cross_v[cross_v_index(layer, h, row, d)] =
						float_to_half(((float)vbt[d] * vb.scale) +
							      (vbuf[d] * vw.scale));
				}
#else
				for (uint32_t d = 0u; d < HEAD_DIM; d++) {
					float kacc = 0.0f;
					float vacc = (float)vbt[d] * vb.scale;

					for (uint32_t i = 0u; i < DIM; i++) {
						const uint32_t widx = i * HEAD_DIM + d;
						kacc += ln[i] * wval(base, kw, widx);
						vacc += ln[i] * wval(base, vw, widx);
					}
					cross_k[cross_k_index(layer, h, d, row)] =
						float_to_half(kacc);
					cross_v[cross_v_index(layer, h, row, d)] =
						float_to_half(vacc);
				}
#endif
			}
		}
	}
}

static void flush_all_encoder_state(float *hidden, float *qkv)
{
	FENCE;
	evict(hidden, SRC_LEN * DIM * sizeof(float));
	evict(qkv, SRC_LEN * QKV_DIM * sizeof(float));
	WAIT_CACHEOPS;
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	volatile struct run_params *const params =
		(volatile struct run_params *)(base + PARAM_OFFSET);
	volatile struct summary *const s =
		(volatile struct summary *)(base + SUMMARY_OFFSET);
	volatile struct stage_record *const stage_records =
		(volatile struct stage_record *)(base + STAGE_RECORD_OFFSET);
	float *const audio = (float *)(base + AUDIO_OFFSET);
	float *const conv1 = (float *)(base + CONV1_OFFSET);
	float *const hidden = (float *)(base + WEIGHT_REGION_OFFSET +
					ENCODER_SCRATCH_OFFSET);
	float *const qkv = hidden + (SRC_LEN * DIM);
	float *const slot = (float *)(base + SCRATCH_OFFSET +
				      hart_id * ROW_SLOT_FLOATS * sizeof(float));
	float *const ln = slot;
	float *const context = ln + ROW_LN_FLOATS;
	float *const tmp = context + ROW_CONTEXT_FLOATS;
	float *const wide = tmp + ROW_TMP_FLOATS;
	float *const scores = wide + ROW_WIDE_FLOATS;

	const uint64_t hpm3_0 = read_hpm3();
	const uint64_t hpm4_0 = read_hpm4();
	const uint64_t hpm5_0 = read_hpm5();
	const uint64_t hpm6_0 = read_hpm6();
	const uint64_t hpm7_0 = read_hpm7();
	const uint64_t hpm8_0 = read_hpm8();
	uint32_t stage_count = 0u;
	struct pmc_snapshot stage_start;

	if (params->input_format == 1u) {
#if FRONTEND_USE_VPU
		if (!FRONTEND_T0_ONLY || ((hart_id & 1u) == 0u)) {
			vpu_mask_all();
		}
#endif
		if (hart_id == 0u) {
			s->magic = MAGIC;
			s->hart_id = hart_id;
			s->active_harts = ACTIVE_HARTS;
			s->src_len = SRC_LEN;
			s->reserved[5] = params->input_format;
			s->reserved[6] = params->input_bytes;
			FENCE;
			evict((void *)s, sizeof(*s));
			WAIT_CACHEOPS;
		}
		bench_barrier();
		if (hart_id == 0u) {
			stage_start = read_pmc_snapshot();
		}
		wav_frontend_stage(base, audio, params->input_bytes, hart_id,
				   scores, s);
		evict_frontend_feature_slice(audio, hart_id);
		bench_barrier();
		if (hart_id == 0u) {
			record_stage(stage_records, stage_count++,
				     ENC_STAGE_FRONTEND, 0u, stage_start,
				     read_pmc_snapshot());
		}
#if FRONTEND_ONLY
		if (hart_id == 0u) {
			s->magic = MAGIC;
			s->hart_id = hart_id;
			s->active_harts = ACTIVE_HARTS;
			s->src_len = SRC_LEN;
			s->done = 1u;
			s->reserved[0] = stage_count;
			s->reserved[1] = STAGE_RECORD_OFFSET;
			FENCE;
			evict((void *)stage_records,
			      stage_count * sizeof(struct stage_record));
			evict((void *)s, sizeof(*s));
			WAIT_CACHEOPS;
		}
		bench_barrier();
		return 0;
#endif
	} else {
#if ENCODER_ATTENTION_VPU_DOT || ENCODER_ATTENTION_VPU_AXPY || \
	ENCODER_CROSS_CACHE_VPU_I8TMP
		vpu_mask_all();
#endif
		if (hart_id == 0u) {
			FENCE;
			evict(audio, MEL_BINS * AUDIO_LEN * sizeof(float));
			WAIT_CACHEOPS;
		}
		bench_barrier();
	}

	stage_start = read_pmc_snapshot();
	conv1_stage(base, hart_id, audio, conv1);
	FENCE;
	evict(conv1, DIM * AUDIO_LEN * sizeof(float));
	WAIT_CACHEOPS;
	bench_barrier();
	if (hart_id == 0u) {
		record_stage(stage_records, stage_count++, ENC_STAGE_CONV1, 0u,
			     stage_start, read_pmc_snapshot());
	}

	stage_start = read_pmc_snapshot();
	conv2_stage(base, hart_id, conv1, hidden);
	FENCE;
	evict(hidden, SRC_LEN * DIM * sizeof(float));
	WAIT_CACHEOPS;
	bench_barrier();
	if (hart_id == 0u) {
		record_stage(stage_records, stage_count++, ENC_STAGE_CONV2, 0u,
			     stage_start, read_pmc_snapshot());
	}

	for (uint32_t layer = 0u; layer < LAYERS; layer++) {
		stage_start = read_pmc_snapshot();
		qkv_stage(base, hart_id, layer, hidden, qkv, ln);
		FENCE;
		evict(qkv, SRC_LEN * QKV_DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();
		if (hart_id == 0u) {
			record_stage(stage_records, stage_count++, ENC_STAGE_QKV,
				     layer, stage_start, read_pmc_snapshot());
		}

		stage_start = read_pmc_snapshot();
		attention_stage(base, hart_id, layer, hidden, qkv, context,
				ln, scores, wide, tmp);
		FENCE;
		evict(hidden, SRC_LEN * DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();
		if (hart_id == 0u) {
			record_stage(stage_records, stage_count++,
				     ENC_STAGE_ATTENTION, layer, stage_start,
				     read_pmc_snapshot());
		}

		stage_start = read_pmc_snapshot();
		mlp_stage(base, hart_id, layer, hidden, ln, context, wide,
			  scores, tmp);
		FENCE;
		evict(hidden, SRC_LEN * DIM * sizeof(float));
		WAIT_CACHEOPS;
		bench_barrier();
		if (hart_id == 0u) {
			record_stage(stage_records, stage_count++, ENC_STAGE_MLP,
				     layer, stage_start, read_pmc_snapshot());
		}
	}

	stage_start = read_pmc_snapshot();
	cross_cache_stage(base, hart_id, hidden, ln, context, tmp);
	FENCE;
	evict((void *)(base + CROSS_K_OFFSET),
	      LAYERS * HEADS * HEAD_DIM * SRC_LEN * sizeof(uint16_t));
	evict((void *)(base + CROSS_V_OFFSET),
	      LAYERS * HEADS * SRC_LEN * HEAD_DIM * sizeof(uint16_t));
	WAIT_CACHEOPS;
	bench_barrier();
	if (hart_id == 0u) {
		record_stage(stage_records, stage_count++, ENC_STAGE_CROSS_CACHE,
			     0u, stage_start, read_pmc_snapshot());
	}

	const uint64_t hpm3_1 = read_hpm3();
	const uint64_t hpm4_1 = read_hpm4();
	const uint64_t hpm5_1 = read_hpm5();
	const uint64_t hpm6_1 = read_hpm6();
	const uint64_t hpm7_1 = read_hpm7();
	const uint64_t hpm8_1 = read_hpm8();

	if (hart_id == 0u) {
		const uint64_t conv_ops =
			((uint64_t)DIM * AUDIO_LEN * MEL_BINS * 3u * 2u) +
			((uint64_t)DIM * SRC_LEN * DIM * 3u * 2u);
		const uint64_t block_ops = (uint64_t)LAYERS *
			(((uint64_t)SRC_LEN * DIM * QKV_DIM * 2u) +
			 ((uint64_t)SRC_LEN * HEADS * SRC_LEN * HEAD_DIM * 2u) +
			 ((uint64_t)SRC_LEN * HEADS * SRC_LEN * HEAD_DIM * 2u) +
			 ((uint64_t)SRC_LEN * DIM * DIM * 2u) +
			 ((uint64_t)SRC_LEN * DIM * MLP_DIM * 2u) +
			 ((uint64_t)SRC_LEN * MLP_DIM * DIM * 2u));
		const uint64_t cross_ops =
			(uint64_t)LAYERS * HEADS * SRC_LEN * DIM * HEAD_DIM * 2u * 2u;
		const uint64_t ops = conv_ops + block_ops + cross_ops;

		s->magic = MAGIC;
		s->hart_id = hart_id;
		s->active_harts = ACTIVE_HARTS;
		s->src_len = SRC_LEN;
		s->scratch_offset_lo = (uint32_t)ENCODER_SCRATCH_OFFSET;
		s->scratch_offset_hi = (uint32_t)(ENCODER_SCRATCH_OFFSET >> 32);
		s->hpm3_lo = (uint32_t)(hpm3_1 - hpm3_0);
		s->hpm3_hi = (uint32_t)((hpm3_1 - hpm3_0) >> 32);
		s->hpm4_lo = (uint32_t)(hpm4_1 - hpm4_0);
		s->hpm4_hi = (uint32_t)((hpm4_1 - hpm4_0) >> 32);
		s->hpm5_lo = (uint32_t)(hpm5_1 - hpm5_0);
		s->hpm5_hi = (uint32_t)((hpm5_1 - hpm5_0) >> 32);
		s->hpm6_lo = (uint32_t)(hpm6_1 - hpm6_0);
		s->hpm6_hi = (uint32_t)((hpm6_1 - hpm6_0) >> 32);
		s->hpm7_lo = (uint32_t)(hpm7_1 - hpm7_0);
		s->hpm7_hi = (uint32_t)((hpm7_1 - hpm7_0) >> 32);
		s->hpm8_lo = (uint32_t)(hpm8_1 - hpm8_0);
		s->hpm8_hi = (uint32_t)((hpm8_1 - hpm8_0) >> 32);
		s->ops_lo = (uint32_t)ops;
		s->ops_hi = (uint32_t)(ops >> 32);
		s->done = 1u;
		s->reserved[0] = stage_count;
		s->reserved[1] = STAGE_RECORD_OFFSET;
		FENCE;
		evict((void *)stage_records,
		      stage_count * sizeof(struct stage_record));
		evict((void *)s, sizeof(*s));
		WAIT_CACHEOPS;
	}

	bench_barrier();
	(void)flush_all_encoder_state;
	return 0;
}
