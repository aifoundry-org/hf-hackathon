/*
 * Resident raw-INT8 Whisper decoder-token executor.
 *
 * This is the first full decoder-side graph executor: token embedding,
 * positional embedding, all four decoder blocks, final LayerNorm, full vocab
 * projection, and greedy token selection run inside one ET-SoC1 kernel.
 *
 * Scope: decoder only.  The host supplies encoder cross-attention K/V caches
 * as FP16 in the 16 MiB runtime arena.  The decoder weights are read from the
 * 64 MiB resident raw-INT8 weight region.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "whisper_resident_decoder_weights_auto.h"

extern char heap0_end[];

#ifndef MAGIC
#define MAGIC 0x57524443u
#endif
#ifndef RUNTIME_REGION_BYTES
#define RUNTIME_REGION_BYTES (16u * 1024u * 1024u)
#endif
#ifndef WEIGHT_REGION_OFFSET
#define WEIGHT_REGION_OFFSET (16u * 1024u * 1024u)
#endif
#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef CROSS_K_POS_MAJOR
#define CROSS_K_POS_MAJOR 1
#endif
#ifndef DECODER_ARGMAX_BLOCK
#define DECODER_ARGMAX_BLOCK 8u
#endif
#ifndef DECODER_PARALLEL_SELF_ATTN
#define DECODER_PARALLEL_SELF_ATTN 1
#endif
#ifndef DECODER_PARALLEL_CROSS_ATTN
#define DECODER_PARALLEL_CROSS_ATTN 1
#endif
#ifndef DECODER_SELF_VPU_DOT
#define DECODER_SELF_VPU_DOT 0
#endif
#ifndef DECODER_CROSS_VPU_DOT
#define DECODER_CROSS_VPU_DOT 0
#endif
#ifndef DECODER_CROSS_VPU_AXPY
#define DECODER_CROSS_VPU_AXPY 0
#endif
#ifndef DECODER_ARGMAX_VPU_I8TMP
#define DECODER_ARGMAX_VPU_I8TMP 0
#endif
#ifndef DECODER_ROW_MAJOR_MATVEC
#define DECODER_ROW_MAJOR_MATVEC 0
#endif

#define LAYERS 4u
#define HEADS 6u
#define HEAD_DIM 64u
#define DIM 384u
#define MLP_DIM 1536u
#define VOCAB 51864u
#define SRC_LEN 1500u
#define MAX_TOKENS 224u
#define SCORES_STRIDE 1536u
#define CACHELINE_FLOATS 16u

#define SUMMARY_OFFSET 0x1000u
#define STAGE_RECORD_OFFSET 0x40000u
#define PARAM_OFFSET   0x2000u
#define TOKENS_OFFSET  0x3000u
#define HIDDEN0_OFFSET 0x4000u
#define HIDDEN1_OFFSET 0x5000u
#define QKV_OFFSET     0x7000u
#define TMP384_OFFSET  0xc000u
#define TMP1536_OFFSET 0x12000u
#define SCORES_OFFSET  0x18000u
#define CONTEXT_OFFSET 0x31000u
#define ARGMAX_SLOTS_OFFSET 0x38000u
#define DECODER_CTRL_OFFSET 0x39000u
#define TEXT_OFFSET    0x60000u
#define TEXT_BYTES     0x20000u
#define CROSS_K_OFFSET 0x100000u
#define CROSS_V_OFFSET 0x580000u
#define SELF_K_OFFSET  0xa00000u
#define SELF_V_OFFSET  0xb80000u

#define BENCH_FLB 2u
#define BENCH_FCC FCC_0

struct run_params {
	uint32_t magic;
	uint32_t max_new_tokens;
	uint32_t suppress_eos_until;
	uint32_t reserved[13];
};

struct summary {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t max_new_tokens;
	uint32_t total_steps;
	uint32_t generated_tokens;
	uint32_t eos_seen;
	uint32_t final_token;
	uint32_t final_argmax;
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
	uint32_t reserved[9];
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

struct argmax_slot {
	float best;
	uint32_t argmax;
	uint32_t done;
	uint32_t reserved[13];
};

struct decoder_control {
	uint32_t step;
	uint32_t stop;
	uint32_t final_argmax;
	uint32_t reserved[13];
};

enum decoder_stage_id {
	DEC_STAGE_EMBED = 1,
	DEC_STAGE_SELF_ATTN = 2,
	DEC_STAGE_CROSS_ATTN = 3,
	DEC_STAGE_MLP = 4,
	DEC_STAGE_FINAL_ARGMAX = 5,
};

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

static inline void vpu_axpy8_f32(float *acc, float scale, const float *x)
{
	__asm__ __volatile__(
		"fbcx.ps  f2, %[scale]\n"
		"flq2     f0, 0(%[acc])\n"
		"flq2     f1, 0(%[x])\n"
		"fmadd.ps f0, f2, f1, f0\n"
		"fsq2     f0, 0(%[acc])\n"
		:
		: [scale] "r"(scale), [acc] "r"(acc), [x] "r"(x)
		: "memory", "f0", "f1", "f2");
}

static void record_stage(volatile struct stage_record *records,
			 uint32_t idx, uint32_t stage_id, uint32_t layer,
			 uint32_t step, struct pmc_snapshot start,
			 struct pmc_snapshot end)
{
	volatile struct stage_record *const r = &records[idx];

	r->stage_id = stage_id;
	r->layer = layer;
	r->step = step;
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

static float half_to_float(uint16_t h)
{
	const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
	const uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
	const uint32_t mant = (uint32_t)h & 0x3ffu;
	union {
		uint32_t u;
		float f;
	} out;

	if (exp == 0u) {
		if (mant == 0u) {
			out.u = sign;
			return out.f;
		}
		float v = (float)mant * (1.0f / 16777216.0f);
		return sign ? -v : v;
	}
	if (exp == 31u) {
		out.u = sign | 0x7f800000u | (mant << 13);
		return out.f;
	}
	out.u = sign | ((exp + 112u) << 23) | (mant << 13);
	return out.f;
}

static inline const int8_t *wptr(uint8_t *base, uint32_t off)
{
	return (const int8_t *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline const float *fptr(uint8_t *base, uint32_t off)
{
	return (const float *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline const uint32_t *u32ptr(uint8_t *base, uint32_t off)
{
	return (const uint32_t *)(base + WEIGHT_REGION_OFFSET + off);
}

static inline float wval(uint8_t *base, struct weight_desc w, uint32_t idx)
{
	return (float)wptr(base, w.offset)[idx] * w.scale;
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

#if DECODER_ROW_MAJOR_MATVEC
	for (uint32_t n = 0u; n < out_dim; n++) {
		out[n] = (float)b[n] * bias.scale;
	}
	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float x = in[k];

		for (uint32_t n = 0u; n < out_dim; n++) {
			out[n] += x * ((float)wrow[n] * weight.scale);
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

static void matvec_i8_per_out_scale(uint8_t *base, float *out,
				    const float *in, uint32_t in_dim,
				    uint32_t out_dim,
				    struct weight_desc weight,
				    struct weight_desc bias,
				    uint32_t scales_offset)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);
	const float *const scales = fptr(base, scales_offset);

#if DECODER_ROW_MAJOR_MATVEC
	for (uint32_t n = 0u; n < out_dim; n++) {
		out[n] = (float)b[n] * bias.scale;
	}
	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float x = in[k];

		for (uint32_t n = 0u; n < out_dim; n++) {
			out[n] += x * ((float)wrow[n] * scales[n]);
		}
	}
#else
	for (uint32_t n = 0u; n < out_dim; n++) {
		float acc = (float)b[n] * bias.scale;
		const float scale = scales[n];

		for (uint32_t k = 0u; k < in_dim; k++) {
			acc += in[k] * ((float)wt[k * out_dim + n] * scale);
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

#if DECODER_ROW_MAJOR_MATVEC
	for (uint32_t n = 0u; n < out_dim; n++) {
		out[n] = 0.0f;
	}
	for (uint32_t k = 0u; k < in_dim; k++) {
		const int8_t *const wrow = wt + k * out_dim;
		const float x = in[k];

		for (uint32_t n = 0u; n < out_dim; n++) {
			out[n] += x * ((float)wrow[n] * weight.scale);
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

static void matvec_i8_part_blocks_per_out_scale(uint8_t *base, float *out,
						const float *in,
						uint32_t in_dim,
						uint32_t out_dim,
						struct weight_desc weight,
						struct weight_desc bias,
						uint32_t scales_offset,
						uint32_t hart_id)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);
	const float *const scales = fptr(base, scales_offset);
	const uint32_t blocks =
		(out_dim + CACHELINE_FLOATS - 1u) / CACHELINE_FLOATS;
	float acc[CACHELINE_FLOATS];

	evict((void *)in, in_dim * sizeof(float));
	WAIT_CACHEOPS;

	for (uint32_t block = hart_id; block < blocks; block += ACTIVE_HARTS) {
		const uint32_t n0 = block * CACHELINE_FLOATS;
		uint32_t n1 = n0 + CACHELINE_FLOATS;

		if (n1 > out_dim) {
			n1 = out_dim;
		}
#if DECODER_ROW_MAJOR_MATVEC
		const uint32_t count = n1 - n0;

		for (uint32_t j = 0u; j < count; j++) {
			acc[j] = (float)b[n0 + j] * bias.scale;
		}
		for (uint32_t k = 0u; k < in_dim; k++) {
			const int8_t *const wrow = wt + k * out_dim + n0;
			const float x = in[k];

			for (uint32_t j = 0u; j < count; j++) {
				const uint32_t n = n0 + j;

				acc[j] += x * ((float)wrow[j] * scales[n]);
			}
		}
		for (uint32_t j = 0u; j < count; j++) {
			out[n0 + j] = acc[j];
		}
#else
		for (uint32_t n = n0; n < n1; n++) {
			float acc = (float)b[n] * bias.scale;
			const float scale = scales[n];

			for (uint32_t k = 0u; k < in_dim; k++) {
				acc += in[k] *
				       ((float)wt[k * out_dim + n] * scale);
			}
			out[n] = acc;
		}
#endif
		FENCE;
		evict(out + n0, (n1 - n0) * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void matvec_i8_part_blocks(uint8_t *base, float *out, const float *in,
				  uint32_t in_dim, uint32_t out_dim,
				  struct weight_desc weight,
				  struct weight_desc bias,
				  uint32_t hart_id)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const int8_t *const b = wptr(base, bias.offset);
	const uint32_t blocks =
		(out_dim + CACHELINE_FLOATS - 1u) / CACHELINE_FLOATS;
	float acc[CACHELINE_FLOATS];

	evict((void *)in, in_dim * sizeof(float));
	WAIT_CACHEOPS;

	for (uint32_t block = hart_id; block < blocks; block += ACTIVE_HARTS) {
		const uint32_t n0 = block * CACHELINE_FLOATS;
		uint32_t n1 = n0 + CACHELINE_FLOATS;

		if (n1 > out_dim) {
			n1 = out_dim;
		}
#if DECODER_ROW_MAJOR_MATVEC
		const uint32_t count = n1 - n0;

		for (uint32_t j = 0u; j < count; j++) {
			acc[j] = (float)b[n0 + j] * bias.scale;
		}
		for (uint32_t k = 0u; k < in_dim; k++) {
			const int8_t *const wrow = wt + k * out_dim + n0;
			const float x = in[k];

			for (uint32_t j = 0u; j < count; j++) {
				acc[j] += x * ((float)wrow[j] * weight.scale);
			}
		}
		for (uint32_t j = 0u; j < count; j++) {
			out[n0 + j] = acc[j];
		}
#else
		for (uint32_t n = n0; n < n1; n++) {
			float acc = (float)b[n] * bias.scale;

			for (uint32_t k = 0u; k < in_dim; k++) {
				acc += in[k] *
				       ((float)wt[k * out_dim + n] * weight.scale);
			}
			out[n] = acc;
		}
#endif
		FENCE;
		evict(out + n0, (n1 - n0) * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void matvec_i8_nobias_part_blocks(uint8_t *base, float *out,
					 const float *in, uint32_t in_dim,
					 uint32_t out_dim,
					 struct weight_desc weight,
					 uint32_t hart_id)
{
	const int8_t *const wt = wptr(base, weight.offset);
	const uint32_t blocks =
		(out_dim + CACHELINE_FLOATS - 1u) / CACHELINE_FLOATS;
	float acc[CACHELINE_FLOATS];

	evict((void *)in, in_dim * sizeof(float));
	WAIT_CACHEOPS;

	for (uint32_t block = hart_id; block < blocks; block += ACTIVE_HARTS) {
		const uint32_t n0 = block * CACHELINE_FLOATS;
		uint32_t n1 = n0 + CACHELINE_FLOATS;

		if (n1 > out_dim) {
			n1 = out_dim;
		}
#if DECODER_ROW_MAJOR_MATVEC
		const uint32_t count = n1 - n0;

		for (uint32_t j = 0u; j < count; j++) {
			acc[j] = 0.0f;
		}
		for (uint32_t k = 0u; k < in_dim; k++) {
			const int8_t *const wrow = wt + k * out_dim + n0;
			const float x = in[k];

			for (uint32_t j = 0u; j < count; j++) {
				acc[j] += x * ((float)wrow[j] * weight.scale);
			}
		}
		for (uint32_t j = 0u; j < count; j++) {
			out[n0 + j] = acc[j];
		}
#else
		for (uint32_t n = n0; n < n1; n++) {
			float acc = 0.0f;

			for (uint32_t k = 0u; k < in_dim; k++) {
				acc += in[k] *
				       ((float)wt[k * out_dim + n] * weight.scale);
			}
			out[n] = acc;
		}
#endif
		FENCE;
		evict(out + n0, (n1 - n0) * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void add_inplace(float *dst, const float *src)
{
	for (uint32_t i = 0u; i < DIM; i++) {
		dst[i] += src[i];
	}
}

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

static inline uint32_t self_k_index(uint32_t layer, uint32_t head,
				    uint32_t dim, uint32_t step)
{
	return (((layer * HEADS + head) * HEAD_DIM + dim) * MAX_TOKENS) + step;
}

static inline uint32_t self_v_index(uint32_t layer, uint32_t head,
				    uint32_t step, uint32_t dim)
{
	return (((layer * HEADS + head) * MAX_TOKENS + step) * HEAD_DIM) + dim;
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

static inline uint32_t part0(uint32_t n, uint32_t hart_id)
{
	return (n * hart_id) / ACTIVE_HARTS;
}

static inline uint32_t part1(uint32_t n, uint32_t hart_id)
{
	return (n * (hart_id + 1u)) / ACTIVE_HARTS;
}

static void self_attention(uint8_t *base, uint32_t layer, uint32_t step,
			   float *hidden, float *ln, float *qkv,
			   float *context, float *scores, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];
	float *const self_k = (float *)(base + SELF_K_OFFSET);
	float *const self_v = (float *)(base + SELF_V_OFFSET);

	layernorm(base, ln, hidden, d.attn_ln_weight, d.attn_ln_bias);
	matvec_i8_nobias(base, qkv, ln, DIM, 1152u, d.self_qkv_weight);

	for (uint32_t i = 0u; i < DIM; i++) {
		qkv[i] += wval(base, d.self_q_bias, i);
		qkv[768u + i] += wval(base, d.self_v_bias, i);
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			self_k[self_k_index(layer, h, i, step)] =
				qkv[DIM + h * HEAD_DIM + i];
			self_v[self_v_index(layer, h, step, i)] =
				qkv[768u + h * HEAD_DIM + i];
		}
	}

	for (uint32_t i = 0u; i < DIM; i++) {
		context[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q = qkv + h * HEAD_DIM;

		for (uint32_t t = 0u; t <= step; t++) {
			float s = 0.0f;

			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * self_k[self_k_index(layer, h, i, t)];
			}
			scores[t] = s;
		}
		softmax(scores, step + 1u);
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t <= step; t++) {
				v += scores[t] * self_v[self_v_index(layer, h, t, i)];
			}
			context[h * HEAD_DIM + i] = v;
		}
	}

	matvec_i8(base, tmp, context, DIM, DIM, d.self_out_weight, d.self_out_bias);
	add_inplace(hidden, tmp);
}

#if DECODER_PARALLEL_SELF_ATTN
static void evict_self_step(float *self_k, float *self_v, uint32_t layer,
			    uint32_t step)
{
	for (uint32_t h = 0u; h < HEADS; h++) {
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			evict(&self_k[self_k_index(layer, h, i, step)],
			      sizeof(float));
		}
		evict(&self_v[self_v_index(layer, h, step, 0u)],
		      HEAD_DIM * sizeof(float));
	}
}

static void self_attention_heads_part(uint8_t *base, uint32_t layer,
				      uint32_t step, const float *qkv,
				      float *context, float *scores,
				      uint32_t hart_id)
{
	const float *const self_k = (const float *)(base + SELF_K_OFFSET);
	const float *const self_v = (const float *)(base + SELF_V_OFFSET);
#if DECODER_SELF_VPU_DOT
	__attribute__((aligned(32))) float ktmp[HEAD_DIM];
	__attribute__((aligned(32))) float vtmp8[8u];
#endif

	evict((void *)qkv, DIM * sizeof(float));
	WAIT_CACHEOPS;

	for (uint32_t h = hart_id; h < HEADS; h += ACTIVE_HARTS) {
		const float *const q = qkv + h * HEAD_DIM;

		for (uint32_t t = 0u; t <= step; t++) {
			float s = 0.0f;

#if DECODER_SELF_VPU_DOT
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				ktmp[i] = self_k[self_k_index(layer, h, i, t)];
			}
			s = vpu_dot64_f32_tmp(q, ktmp, vtmp8);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * self_k[self_k_index(layer, h, i, t)];
			}
#endif
			scores[t] = s;
		}
		softmax(scores, step + 1u);
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t <= step; t++) {
				v += scores[t] * self_v[self_v_index(layer, h, t, i)];
			}
			context[h * HEAD_DIM + i] = v;
		}
		FENCE;
		evict(context + h * HEAD_DIM, HEAD_DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void self_attention_parallel(uint8_t *base, uint32_t layer,
				    uint32_t step, float *hidden, float *ln,
				    float *qkv, float *context,
				    float *scores, float *tmp,
				    uint32_t hart_id)
{
	const struct layer_desc d = decoder_layers[layer];
	float *const self_k = (float *)(base + SELF_K_OFFSET);
	float *const self_v = (float *)(base + SELF_V_OFFSET);

	if (hart_id == 0u) {
		layernorm(base, ln, hidden, d.attn_ln_weight, d.attn_ln_bias);
		FENCE;
		evict(ln, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	matvec_i8_nobias_part_blocks(base, qkv, ln, DIM, 1152u,
				     d.self_qkv_weight, hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		for (uint32_t i = 0u; i < DIM; i++) {
			qkv[i] += wval(base, d.self_q_bias, i);
			qkv[768u + i] += wval(base, d.self_v_bias, i);
		}
		for (uint32_t h = 0u; h < HEADS; h++) {
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				self_k[self_k_index(layer, h, i, step)] =
					qkv[DIM + h * HEAD_DIM + i];
				self_v[self_v_index(layer, h, step, i)] =
					qkv[768u + h * HEAD_DIM + i];
			}
		}
		for (uint32_t i = 0u; i < DIM; i++) {
			context[i] = 0.0f;
		}
		FENCE;
		evict(qkv, 1152u * sizeof(float));
		evict(context, DIM * sizeof(float));
		evict_self_step(self_k, self_v, layer, step);
		WAIT_CACHEOPS;
	}
	bench_barrier();
	self_attention_heads_part(base, layer, step, qkv, context, scores,
				  hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		evict(context, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	matvec_i8_part_blocks(base, tmp, context, DIM, DIM,
			      d.self_out_weight, d.self_out_bias, hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		evict(tmp, DIM * sizeof(float));
		WAIT_CACHEOPS;
		add_inplace(hidden, tmp);
	}
	bench_barrier();
}
#endif

static void cross_attention(uint8_t *base, uint32_t layer, float *hidden,
			    float *ln, float *query, float *context,
			    float *scores, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];
	const uint16_t *const cross_k = (const uint16_t *)(base + CROSS_K_OFFSET);
	const uint16_t *const cross_v = (const uint16_t *)(base + CROSS_V_OFFSET);

	layernorm(base, ln, hidden, d.cross_ln_weight, d.cross_ln_bias);
	matvec_i8(base, query, ln, DIM, DIM, d.cross_q_weight, d.cross_q_bias);

	for (uint32_t i = 0u; i < DIM; i++) {
		context[i] = 0.0f;
	}

	for (uint32_t h = 0u; h < HEADS; h++) {
		const float *const q = query + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			float s = 0.0f;

			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * half_to_float(
					cross_k[cross_k_index(layer, h, i, t)]);
			}
			scores[t] = s;
		}
		softmax(scores, SRC_LEN);
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t < SRC_LEN; t++) {
				v += scores[t] * half_to_float(
					cross_v[cross_v_index(layer, h, t, i)]);
			}
			context[h * HEAD_DIM + i] = v;
		}
	}

	matvec_i8(base, tmp, context, DIM, DIM, d.cross_out_weight,
		  d.cross_out_bias);
	add_inplace(hidden, tmp);
}

static void cross_attention_prepare(uint8_t *base, uint32_t layer,
				    const float *hidden, float *ln,
				    float *query, float *context)
{
	const struct layer_desc d = decoder_layers[layer];

	layernorm(base, ln, hidden, d.cross_ln_weight, d.cross_ln_bias);
	matvec_i8(base, query, ln, DIM, DIM, d.cross_q_weight,
		  d.cross_q_bias);
	FENCE;
	evict(query, DIM * sizeof(float));
	evict(context, DIM * sizeof(float));
	WAIT_CACHEOPS;
}

static void cross_attention_heads_part(uint8_t *base, uint32_t layer,
				       const float *query, float *context,
				       float *scores, uint32_t hart_id)
{
	const uint16_t *const cross_k = (const uint16_t *)(base + CROSS_K_OFFSET);
	const uint16_t *const cross_v = (const uint16_t *)(base + CROSS_V_OFFSET);
#if DECODER_CROSS_VPU_DOT || DECODER_CROSS_VPU_AXPY
	__attribute__((aligned(32))) float ftmp[HEAD_DIM];
	__attribute__((aligned(32))) float vtmp8[8u];
#endif

	evict((void *)query, DIM * sizeof(float));
	WAIT_CACHEOPS;

	for (uint32_t h = hart_id; h < HEADS; h += ACTIVE_HARTS) {
		const float *const q = query + h * HEAD_DIM;

		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			float s = 0.0f;

#if DECODER_CROSS_VPU_DOT
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				ftmp[i] = half_to_float(
					cross_k[cross_k_index(layer, h, i, t)]);
			}
			s = vpu_dot64_f32_tmp(q, ftmp, vtmp8);
#else
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				s += q[i] * half_to_float(
					cross_k[cross_k_index(layer, h, i, t)]);
			}
#endif
			scores[t] = s;
		}
		softmax(scores, SRC_LEN);
#if DECODER_CROSS_VPU_AXPY
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			context[h * HEAD_DIM + i] = 0.0f;
		}
		for (uint32_t t = 0u; t < SRC_LEN; t++) {
			for (uint32_t i = 0u; i < HEAD_DIM; i++) {
				ftmp[i] = half_to_float(
					cross_v[cross_v_index(layer, h, t, i)]);
			}
			vpu_axpy64_f32(context + h * HEAD_DIM, scores[t], ftmp);
		}
#else
		for (uint32_t i = 0u; i < HEAD_DIM; i++) {
			float v = 0.0f;

			for (uint32_t t = 0u; t < SRC_LEN; t++) {
				v += scores[t] * half_to_float(
					cross_v[cross_v_index(layer, h, t, i)]);
			}
			context[h * HEAD_DIM + i] = v;
		}
#endif
		FENCE;
		evict(context + h * HEAD_DIM, HEAD_DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void cross_attention_finish(uint8_t *base, uint32_t layer,
				   float *hidden, float *context, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];

	evict(context, DIM * sizeof(float));
	WAIT_CACHEOPS;
	matvec_i8(base, tmp, context, DIM, DIM, d.cross_out_weight,
		  d.cross_out_bias);
	add_inplace(hidden, tmp);
}

#if DECODER_PARALLEL_CROSS_ATTN
static void cross_attention_parallel(uint8_t *base, uint32_t layer,
				     float *hidden, float *ln, float *query,
				     float *context, float *scores, float *tmp,
				     uint32_t hart_id)
{
	const struct layer_desc d = decoder_layers[layer];

	if (hart_id == 0u) {
		layernorm(base, ln, hidden, d.cross_ln_weight, d.cross_ln_bias);
		for (uint32_t i = 0u; i < DIM; i++) {
			context[i] = 0.0f;
		}
		FENCE;
		evict(ln, DIM * sizeof(float));
		evict(context, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	matvec_i8_part_blocks(base, query, ln, DIM, DIM, d.cross_q_weight,
			      d.cross_q_bias, hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		FENCE;
		evict(query, DIM * sizeof(float));
		evict(context, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	cross_attention_heads_part(base, layer, query, context, scores,
				   hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		evict(context, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	matvec_i8_part_blocks(base, tmp, context, DIM, DIM,
			      d.cross_out_weight, d.cross_out_bias, hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		evict(tmp, DIM * sizeof(float));
		WAIT_CACHEOPS;
		add_inplace(hidden, tmp);
	}
	bench_barrier();
}
#endif

static void mlp(uint8_t *base, uint32_t layer, float *hidden, float *ln,
		float *wide, float *tmp)
{
	const struct layer_desc d = decoder_layers[layer];

	layernorm(base, ln, hidden, d.mlp_ln_weight, d.mlp_ln_bias);
	matvec_i8(base, wide, ln, DIM, MLP_DIM, d.mlp_fc1_weight,
		  d.mlp_fc1_bias);

	for (uint32_t i = 0u; i < MLP_DIM; i++) {
		const float x = wide[i];
		const float y = 0.5f * x *
			(1.0f + fast_erff(x * 0.70710678118654757f));
		wide[i] = y;
	}

	matvec_i8_per_out_scale(
		base, tmp, wide, MLP_DIM, DIM, d.mlp_fc2_weight,
		d.mlp_fc2_bias,
		decoder_mlp_fc2_output_scales_offset + layer * DIM * sizeof(float));
	add_inplace(hidden, tmp);
}

static void gelu_part_blocks(float *wide, uint32_t hart_id)
{
	const uint32_t blocks =
		(MLP_DIM + CACHELINE_FLOATS - 1u) / CACHELINE_FLOATS;

	evict(wide, MLP_DIM * sizeof(float));
	WAIT_CACHEOPS;
	for (uint32_t block = hart_id; block < blocks; block += ACTIVE_HARTS) {
		const uint32_t n0 = block * CACHELINE_FLOATS;
		uint32_t n1 = n0 + CACHELINE_FLOATS;

		if (n1 > MLP_DIM) {
			n1 = MLP_DIM;
		}
		for (uint32_t i = n0; i < n1; i++) {
			const float x = wide[i];
			wide[i] = 0.5f * x *
				  (1.0f + fast_erff(x * 0.70710678118654757f));
		}
		FENCE;
		evict(wide + n0, (n1 - n0) * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void mlp_parallel(uint8_t *base, uint32_t layer, float *hidden,
			 float *ln, float *wide, float *tmp, uint32_t hart_id)
{
	const struct layer_desc d = decoder_layers[layer];

	if (hart_id == 0u) {
		layernorm(base, ln, hidden, d.mlp_ln_weight, d.mlp_ln_bias);
		FENCE;
		evict(ln, DIM * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();
	matvec_i8_part_blocks(base, wide, ln, DIM, MLP_DIM, d.mlp_fc1_weight,
			      d.mlp_fc1_bias, hart_id);
	bench_barrier();
	gelu_part_blocks(wide, hart_id);
	bench_barrier();
	matvec_i8_part_blocks_per_out_scale(
		base, tmp, wide, MLP_DIM, DIM, d.mlp_fc2_weight,
		d.mlp_fc2_bias,
		decoder_mlp_fc2_output_scales_offset + layer * DIM * sizeof(float),
		hart_id);
	bench_barrier();
	if (hart_id == 0u) {
		evict(tmp, DIM * sizeof(float));
		WAIT_CACHEOPS;
		add_inplace(hidden, tmp);
	}
	bench_barrier();
}

static void final_layernorm(uint8_t *base, float *hidden, float *ln)
{
	layernorm(base, ln, hidden, final_ln_weight, final_ln_bias);
}

static void final_argmax_part(uint8_t *base, const float *ln, uint32_t hart_id,
			      uint32_t suppress_eos,
			      volatile struct argmax_slot *slots)
{
	const int8_t *const wt = wptr(base, vocab_proj_weight.offset);
	const float *const scales = fptr(base, vocab_proj_output_scales_offset);
	float best = -3.4028234663852886e38f;
	uint32_t argmax = part0(VOCAB, hart_id);
	const uint32_t n0 = part0(VOCAB, hart_id);
	const uint32_t n1 = part1(VOCAB, hart_id);
	float acc[DECODER_ARGMAX_BLOCK];
#if DECODER_ARGMAX_VPU_I8TMP
	__attribute__((aligned(32))) float wtmp[8u];
#endif

	for (uint32_t n = n0; n < n1; n += DECODER_ARGMAX_BLOCK) {
		uint32_t count = n1 - n;

		if (count > DECODER_ARGMAX_BLOCK) {
			count = DECODER_ARGMAX_BLOCK;
		}
		for (uint32_t j = 0u; j < count; j++) {
			acc[j] = 0.0f;
		}
		for (uint32_t k = 0u; k < DIM; k++) {
			const int8_t *const wrow = wt + k * VOCAB + n;
			const float x = ln[k];

#if DECODER_ARGMAX_VPU_I8TMP
			if (count == 8u) {
				for (uint32_t j = 0u; j < 8u; j++) {
					wtmp[j] = (float)wrow[j];
				}
				vpu_axpy8_f32(acc, x, wtmp);
				continue;
			}
#endif
			for (uint32_t j = 0u; j < count; j++) {
				acc[j] += x * (float)wrow[j];
			}
		}
		for (uint32_t j = 0u; j < count; j++) {
			const uint32_t token = n + j;
			const float v = acc[j] * scales[token];

			if (suppress_eos && token == 50256u) {
				continue;
			}
			if (v > best || (v == best && token < argmax)) {
				best = v;
				argmax = token;
			}
		}
	}

	slots[hart_id].best = best;
	slots[hart_id].argmax = argmax;
	slots[hart_id].done = 1u;
	FENCE;
	evict((void *)&slots[hart_id], sizeof(slots[hart_id]));
	WAIT_CACHEOPS;
}

static void run_step(uint8_t *base, uint32_t token, uint32_t step,
		     float *hidden, float *ln, float *qkv, float *tmp,
		     float *wide, float *scores, float *context,
		     volatile struct stage_record *stage_records,
		     uint32_t *stage_count, uint32_t hart_id)
{
	struct pmc_snapshot stage_start = {0};

	if (hart_id == 0u) {
		stage_start = read_pmc_snapshot();
		const int8_t *const token_w = wptr(base, token_embedding.offset);
		const float *const token_scales =
			fptr(base, token_embedding_dim_scales_offset);
		for (uint32_t i = 0u; i < DIM; i++) {
			hidden[i] = (float)token_w[token * DIM + i] *
				    token_scales[i] +
				    wval(base, positional_embedding, step * DIM + i);
		}
		record_stage(stage_records, (*stage_count)++, DEC_STAGE_EMBED, 0u,
			     step, stage_start, read_pmc_snapshot());
	}

	for (uint32_t layer = 0u; layer < LAYERS; layer++) {
#if DECODER_PARALLEL_SELF_ATTN
		if (hart_id == 0u) {
			stage_start = read_pmc_snapshot();
		}
		self_attention_parallel(base, layer, step, hidden, ln, qkv,
					context, scores, tmp, hart_id);
		if (hart_id == 0u) {
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_SELF_ATTN, layer, step, stage_start,
				     read_pmc_snapshot());
			stage_start = read_pmc_snapshot();
		}
#if DECODER_PARALLEL_CROSS_ATTN
		cross_attention_parallel(base, layer, hidden, ln, tmp, context,
					 scores, qkv, hart_id);
		if (hart_id == 0u) {
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_CROSS_ATTN, layer, step,
				     stage_start, read_pmc_snapshot());
			stage_start = read_pmc_snapshot();
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_MLP, layer, step, stage_start,
				     read_pmc_snapshot());
		} else {
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
		}
		continue;
#else
		if (hart_id == 0u) {
			cross_attention_prepare(base, layer, hidden, ln, tmp,
						context);
		}
#endif
#else
		if (hart_id == 0u) {
			stage_start = read_pmc_snapshot();
			self_attention(base, layer, step, hidden, ln, qkv, context,
				       scores, tmp);
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_SELF_ATTN, layer, step, stage_start,
				     read_pmc_snapshot());
			stage_start = read_pmc_snapshot();
#if DECODER_PARALLEL_CROSS_ATTN
		}
		cross_attention_parallel(base, layer, hidden, ln, tmp, context,
					 scores, qkv, hart_id);
		if (hart_id == 0u) {
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_CROSS_ATTN, layer, step,
				     stage_start, read_pmc_snapshot());
			stage_start = read_pmc_snapshot();
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_MLP, layer, step, stage_start,
				     read_pmc_snapshot());
		} else {
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
		}
		continue;
#else
			cross_attention_prepare(base, layer, hidden, ln, tmp,
						context);
		}
#endif
#endif
		bench_barrier();
		cross_attention_heads_part(base, layer, tmp, context, scores,
					   hart_id);
		bench_barrier();
		if (hart_id == 0u) {
			cross_attention_finish(base, layer, hidden, context, qkv);
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_CROSS_ATTN, layer, step,
				     stage_start, read_pmc_snapshot());
			stage_start = read_pmc_snapshot();
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
			record_stage(stage_records, (*stage_count)++,
				     DEC_STAGE_MLP, layer, step, stage_start,
				     read_pmc_snapshot());
		} else {
			mlp_parallel(base, layer, hidden, ln, wide, tmp, hart_id);
		}
	}
}

static uint32_t detokenize_to_text(uint8_t *base, const uint32_t *tokens,
				   uint32_t token_count)
{
	const uint32_t *const offsets = u32ptr(base, token_text_offsets_offset);
	const uint8_t *const bytes = base + WEIGHT_REGION_OFFSET +
				     token_text_bytes_offset;
	uint8_t *const out = base + TEXT_OFFSET;
	uint32_t pos = 0u;

	for (uint32_t i = 0u; i < token_count && pos + 1u < TEXT_BYTES; i++) {
		const uint32_t token = tokens[i];

		if (token >= VOCAB) {
			continue;
		}
		uint32_t start = offsets[token];
		uint32_t end = offsets[token + 1u];

		if (start > end || end > token_text_bytes_size) {
			continue;
		}
		for (uint32_t j = start; j < end && pos + 1u < TEXT_BYTES; j++) {
			out[pos++] = bytes[j];
		}
	}
	out[pos] = 0u;
	return pos;
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
	volatile struct argmax_slot *const argmax_slots =
		(volatile struct argmax_slot *)(base + ARGMAX_SLOTS_OFFSET);
	volatile struct decoder_control *const ctrl =
		(volatile struct decoder_control *)(base + DECODER_CTRL_OFFSET);
	uint32_t *const tokens = (uint32_t *)(base + TOKENS_OFFSET);
	float *const hidden = (float *)(base + HIDDEN0_OFFSET);
	float *const ln = (float *)(base + HIDDEN1_OFFSET);
	float *const qkv = (float *)(base + QKV_OFFSET);
	float *const tmp = (float *)(base + TMP384_OFFSET);
	float *const wide = (float *)(base + TMP1536_OFFSET);
	float *const scores = (float *)(base + SCORES_OFFSET + hart_id * SCORES_STRIDE * sizeof(float));
	float *const context = (float *)(base + CONTEXT_OFFSET);
	uint32_t max_new = params->max_new_tokens;
	uint32_t suppress_eos_until = params->suppress_eos_until;

#if DECODER_SELF_VPU_DOT || DECODER_CROSS_VPU_DOT || \
	DECODER_CROSS_VPU_AXPY || DECODER_ARGMAX_VPU_I8TMP
	vpu_mask_all();
#endif
	if (max_new > 128u) {
		max_new = 128u;
	}
	if (suppress_eos_until > max_new) {
		suppress_eos_until = max_new;
	}

	if (hart_id == 0u) {
		tokens[0] = 50257u;
		ctrl->step = 0u;
		ctrl->stop = 0u;
		ctrl->final_argmax = 0u;
		FENCE;
		evict((void *)ctrl, sizeof(*ctrl));
		evict((void *)tokens, sizeof(uint32_t));
		WAIT_CACHEOPS;
	}

	const uint64_t hpm3_0 = read_hpm3();
	const uint64_t hpm4_0 = read_hpm4();
	const uint64_t hpm5_0 = read_hpm5();
	const uint64_t hpm6_0 = read_hpm6();
	const uint64_t hpm7_0 = read_hpm7();
	const uint64_t hpm8_0 = read_hpm8();

	uint32_t eos_seen = 0u;
	uint32_t final_arg = 0u;
	uint32_t step;
	uint32_t stage_count = 0u;
	const uint32_t total_steps = max_new + 3u;

	for (step = 0u; step < total_steps && step < (MAX_TOKENS - 1u); step++) {
		struct pmc_snapshot argmax_stage_start = {0};
		const uint32_t suppress_eos =
			(step >= 3u && (step - 3u) < suppress_eos_until);
		const uint32_t decoder_step = step >= 3u ? step - 3u : 0u;

		run_step(base, tokens[step], decoder_step, hidden, ln, qkv, tmp,
			 wide, scores, context, stage_records, &stage_count,
			 hart_id);
		if (hart_id == 0u) {
			final_layernorm(base, hidden, ln);
			for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
				argmax_slots[h].done = 0u;
			}
			FENCE;
			evict(ln, DIM * sizeof(float));
			evict((void *)argmax_slots,
			      ACTIVE_HARTS * sizeof(struct argmax_slot));
			WAIT_CACHEOPS;
			argmax_stage_start = read_pmc_snapshot();
		}
		bench_barrier();

		final_argmax_part(base, ln, hart_id, suppress_eos, argmax_slots);
		bench_barrier();

		if (hart_id == 0u) {
			float best = -3.4028234663852886e38f;
			uint32_t argmax = 0u;

			evict((void *)argmax_slots,
			      ACTIVE_HARTS * sizeof(struct argmax_slot));
			WAIT_CACHEOPS;
			for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
				const float v = argmax_slots[h].best;
				const uint32_t n = argmax_slots[h].argmax;

				if (v > best || (v == best && n < argmax)) {
					best = v;
					argmax = n;
				}
			}
			final_arg = argmax;
			record_stage(stage_records, stage_count++,
				     DEC_STAGE_FINAL_ARGMAX, 0u, step,
				     argmax_stage_start, read_pmc_snapshot());

			if (step == 0u) {
				tokens[step + 1u] = 50258u;
			} else if (step == 1u) {
				tokens[step + 1u] = 50358u;
			} else if (step == 2u) {
				tokens[step + 1u] = 50362u;
			} else {
				tokens[step + 1u] = final_arg;
				if (final_arg == 50256u && !suppress_eos) {
					eos_seen = 1u;
					ctrl->stop = 1u;
				}
			}
			if (step + 1u >= total_steps || step + 1u >= (MAX_TOKENS - 1u)) {
				ctrl->stop = 1u;
			}
			ctrl->step = step;
			ctrl->final_argmax = final_arg;
			FENCE;
			evict((void *)tokens, (step + 2u) * sizeof(uint32_t));
			evict((void *)ctrl, sizeof(*ctrl));
			WAIT_CACHEOPS;
		}
		bench_barrier();
		evict((void *)ctrl, sizeof(*ctrl));
		WAIT_CACHEOPS;
		if (ctrl->stop) {
			step++;
			break;
		}
	}

	const uint64_t hpm3_1 = read_hpm3();
	const uint64_t hpm4_1 = read_hpm4();
	const uint64_t hpm5_1 = read_hpm5();
	const uint64_t hpm6_1 = read_hpm6();
	const uint64_t hpm7_1 = read_hpm7();
	const uint64_t hpm8_1 = read_hpm8();

	const uint64_t ops_per_step =
		(uint64_t)LAYERS *
		((uint64_t)DIM * 1152u * 2u +
		 (uint64_t)DIM * DIM * 2u * 4u +
		 (uint64_t)DIM * MLP_DIM * 2u * 2u +
		 (uint64_t)HEADS * SRC_LEN * HEAD_DIM * 2u * 2u) +
		(uint64_t)DIM * VOCAB * 2u;
	const uint64_t ops = ops_per_step * step;

	if (hart_id == 0u) {
		const uint32_t text_bytes = detokenize_to_text(base, tokens,
							      step + 1u);

		s->magic = MAGIC;
		s->hart_id = hart_id;
		s->max_new_tokens = max_new;
		s->total_steps = step;
		s->generated_tokens = step > 3u ? step - 3u : 0u;
		s->eos_seen = eos_seen;
		s->final_token = tokens[step];
		s->final_argmax = final_arg;
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
		s->reserved[2] = suppress_eos_until;
		s->reserved[3] = TEXT_OFFSET;
		s->reserved[4] = text_bytes;
		FENCE;
		evict((void *)tokens, (step + 1u) * sizeof(uint32_t));
		evict((void *)(base + TEXT_OFFSET), text_bytes + 1u);
		evict((void *)stage_records,
		      stage_count * sizeof(struct stage_record));
		evict((void *)s, sizeof(*s));
		WAIT_CACHEOPS;
	}

	return 0;
}
