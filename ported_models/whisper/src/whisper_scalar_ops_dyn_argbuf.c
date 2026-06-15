/*
 * Runtime-parameterized Whisper scalar-node audit kernel.
 *
 * The older scalar audit kernel bakes ELEM_COUNT/ROWS/COLS/OFFSETS into the
 * ELF.  This variant reads those values from a small parameter block so one ELF
 * can audit every encoder/decoder scalar node through the dynamic-memory
 * launcher.
 */

#include <stdint.h>

#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"

extern char heap0_end[];

#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 16u
#endif
#ifndef MAGIC
#define MAGIC 0x57534459u
#endif
#ifndef FAST_EXP_ORDER
#define FAST_EXP_ORDER 5
#endif
#ifndef SOFTMAX_FAST_EXP_ORDER
#define SOFTMAX_FAST_EXP_ORDER 3
#endif

#define OP_ADD     1u
#define OP_MUL     2u
#define OP_DIV     3u
#define OP_ERF     4u
#define OP_SOFTMAX 5u
#define OP_COPY    6u

#define SLOT_BYTES        64u
#define SLOTS_OFFSET      0x0000u
#define SUMMARY_OFFSET    0x1000u
#define PARAM_OFFSET      0x2000u

#define BENCH_FLB         2u
#define BENCH_FCC         FCC_0

struct audit_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t op_mode;
	uint32_t i0;
	uint32_t i1;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[7];
};

struct audit_summary {
	uint32_t magic;
	uint32_t op_mode;
	uint32_t elem_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t active_harts;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_hash;
	uint32_t reference_hash;
	uint32_t max_abs_scaled;
	uint32_t mean_abs_scaled;
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
	uint32_t reserved[8];
};

struct scalar_params {
	uint32_t magic;
	uint32_t op_mode;
	uint32_t elem_count;
	uint32_t rows;
	uint32_t cols;
	uint32_t out_offset;
	uint32_t a_offset;
	uint32_t b_offset;
	uint32_t ref_offset;
	uint32_t reserved[7];
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

static inline float abs_f32(float v)
{
	return v < 0.0f ? -v : v;
}

static inline uint8_t clamp_u8_from_f32(float v)
{
	if (v < 0.0f) {
		return 0u;
	}
	if (v > 255.0f) {
		return 255u;
	}
	return (uint8_t)(v + 0.5f);
}

static uint32_t hash_f32(const float *x, uint32_t i0, uint32_t i1)
{
	uint32_t sum = 0u;

	for (uint32_t i = i0; i < i1; i++) {
		sum += clamp_u8_from_f32(128.0f + x[i] * 0.25f);
	}

	return sum;
}

static inline uint32_t line_count_for(uint32_t elems)
{
	return (elems + 15u) / 16u;
}

static inline uint32_t part_i0(uint32_t elems, uint32_t hart_id)
{
	const uint32_t lines = line_count_for(elems);
	const uint32_t line0 = (lines * hart_id) / ACTIVE_HARTS;

	return line0 * 16u;
}

static inline uint32_t part_i1(uint32_t elems, uint32_t hart_id)
{
	const uint32_t lines = line_count_for(elems);
	const uint32_t line1 = (lines * (hart_id + 1u)) / ACTIVE_HARTS;
	const uint32_t i1 = line1 * 16u;

	return i1 > elems ? elems : i1;
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
#if FAST_EXP_ORDER <= 3
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3);
#elif FAST_EXP_ORDER == 4
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3) +
			(0.0416666679f * r4);
#else
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3) +
			(0.0416666679f * r4) + (0.0083333310f * r5);
#endif

	return p * make_pow2(n);
}

static float fast_expf_softmax(float x)
{
#if SOFTMAX_FAST_EXP_ORDER <= 3
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
	const float p = 1.0f + r + (0.5f * r2) + (0.1666666716f * r3);

	return p * make_pow2(n);
#else
	return fast_expf(x);
#endif
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

static float soft_divf(float a, float b)
{
	union {
		float f;
		uint32_t u;
	} au, bu, ru;
	uint32_t sign;
	uint32_t ea;
	uint32_t eb;
	uint32_t ma;
	uint32_t mb;
	int32_t exp;
	uint64_t num;
	uint64_t q;
	uint64_t rem;
	uint32_t mant;
	uint32_t round_bits;
	uint32_t sticky;

	au.f = a;
	bu.f = b;
	sign = (au.u ^ bu.u) & 0x80000000u;
	ea = (au.u >> 23) & 0xffu;
	eb = (bu.u >> 23) & 0xffu;
	ma = au.u & 0x7fffffu;
	mb = bu.u & 0x7fffffu;

	if ((au.u & 0x7fffffffu) == 0u) {
		ru.u = sign;
		return ru.f;
	}
	if ((bu.u & 0x7fffffffu) == 0u) {
		ru.u = sign | 0x7f800000u;
		return ru.f;
	}

	ma |= 0x800000u;
	mb |= 0x800000u;
	exp = (int32_t)ea - (int32_t)eb;
	if (ma < mb) {
		ma <<= 1u;
		exp--;
	}

	num = (uint64_t)ma << 26u;
	q = num / (uint64_t)mb;
	rem = num - q * (uint64_t)mb;
	mant = (uint32_t)(q >> 3u);
	round_bits = (uint32_t)(q & 7u);
	sticky = rem != 0u;

	if (round_bits > 4u ||
	    (round_bits == 4u && (sticky || ((mant & 1u) != 0u)))) {
		mant++;
		if (mant == 0x1000000u) {
			mant >>= 1u;
			exp++;
		}
	}

	exp += 127;
	if (exp <= 0) {
		ru.u = sign;
		return ru.f;
	}
	if (exp >= 255) {
		ru.u = sign | 0x7f800000u;
		return ru.f;
	}

	ru.u = sign | ((uint32_t)exp << 23u) | (mant & 0x7fffffu);
	return ru.f;
}

static inline float compute_elem(uint32_t op_mode, float a, float b)
{
	switch (op_mode) {
	case OP_ADD:
		return a + b;
	case OP_MUL:
		return a * b;
	case OP_DIV:
		return soft_divf(a, b);
	case OP_ERF:
		return fast_erff(a);
	case OP_COPY:
		(void)b;
		return a;
	default:
		(void)b;
		return a;
	}
}

static void run_elementwise(uint32_t hart_id, const struct scalar_params *p,
			    float *out, const float *a, const float *b)
{
	const uint32_t i0 = part_i0(p->elem_count, hart_id);
	const uint32_t i1 = part_i1(p->elem_count, hart_id);

	for (uint32_t i = i0; i < i1; i++) {
		out[i] = compute_elem(p->op_mode, a[i], b[i]);
	}

	FENCE;
	if (i1 > i0) {
		evict(out + i0, (i1 - i0) * sizeof(float));
		WAIT_CACHEOPS;
	}
}

static void run_softmax(uint32_t hart_id, const struct scalar_params *p,
			float *out, const float *a)
{
	const uint32_t row0 = (p->rows * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (p->rows * (hart_id + 1u)) / ACTIVE_HARTS;

	for (uint32_t row = row0; row < row1; row++) {
		const uint32_t base_idx = row * p->cols;
		float maxv = -3.4028234663852886e38f;

		for (uint32_t c = 0u; c < p->cols; c++) {
			const float v = a[base_idx + c];

			if (v > maxv) {
				maxv = v;
			}
		}

		float sum = 0.0f;

		for (uint32_t c = 0u; c < p->cols; c++) {
			const float e = fast_expf_softmax(a[base_idx + c] - maxv);

			out[base_idx + c] = e;
			sum += e;
		}

		const float inv_sum = fast_recipf(sum);

		for (uint32_t c = 0u; c < p->cols; c++) {
			out[base_idx + c] *= inv_sum;
		}
	}

	FENCE;
	if (row1 > row0) {
		evict(out + row0 * p->cols,
		      (row1 - row0) * p->cols * sizeof(float));
		WAIT_CACHEOPS;
	}
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = get_hart_id();

	if (hart_id >= ACTIVE_HARTS || hart_id >= 16u) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	const volatile struct scalar_params *const vp =
		(const volatile struct scalar_params *)(base + PARAM_OFFSET);
	struct scalar_params p;

	p.magic = vp->magic;
	p.op_mode = vp->op_mode;
	p.elem_count = vp->elem_count;
	p.rows = vp->rows;
	p.cols = vp->cols;
	p.out_offset = vp->out_offset;
	p.a_offset = vp->a_offset;
	p.b_offset = vp->b_offset;
	p.ref_offset = vp->ref_offset;

	float *const out = (float *)(base + p.out_offset);
	const float *const a = (const float *)(base + p.a_offset);
	const float *const b = (const float *)(base + p.b_offset);
	const float *const ref = (const float *)(base + p.ref_offset);
	volatile struct audit_slot *const slots =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET);
	volatile struct audit_summary *const summary =
		(volatile struct audit_summary *)(base + SUMMARY_OFFSET);

	if (p.magic != MAGIC || p.elem_count == 0u || p.rows == 0u ||
	    p.cols == 0u || (p.rows * p.cols) != p.elem_count) {
		if (hart_id == 0u) {
			summary->magic = 0xBAD00001u;
			FENCE;
			evict((void *)summary, sizeof(*summary));
			WAIT_CACHEOPS;
		}
		return 0;
	}

	if (hart_id == 0u) {
		FENCE;
		evict((void *)a, p.elem_count * sizeof(float));
		evict((void *)b, p.elem_count * sizeof(float));
		evict((void *)ref, p.elem_count * sizeof(float));
		WAIT_CACHEOPS;
	}
	bench_barrier();

	const uint64_t hpm3_start = read_hpm3();
	const uint64_t hpm4_start = read_hpm4();
	const uint64_t hpm5_start = read_hpm5();
	const uint64_t hpm6_start = read_hpm6();
	const uint64_t hpm7_start = read_hpm7();
	const uint64_t hpm8_start = read_hpm8();

	if (p.op_mode == OP_SOFTMAX) {
		run_softmax(hart_id, &p, out, a);
	} else {
		run_elementwise(hart_id, &p, out, a, b);
	}
	bench_barrier();

	const uint64_t hpm3_end = read_hpm3();
	const uint64_t hpm4_end = read_hpm4();
	const uint64_t hpm5_end = read_hpm5();
	const uint64_t hpm6_end = read_hpm6();
	const uint64_t hpm7_end = read_hpm7();
	const uint64_t hpm8_end = read_hpm8();

	const uint32_t i0 = part_i0(p.elem_count, hart_id);
	const uint32_t i1 = part_i1(p.elem_count, hart_id);
	volatile struct audit_slot *const slot =
		(volatile struct audit_slot *)(base + SLOTS_OFFSET +
					      hart_id * SLOT_BYTES);

	slot->magic = MAGIC;
	slot->hart_id = hart_id;
	slot->minion_id = get_minion_id();
	slot->thread_id = get_thread_id();
	slot->op_mode = p.op_mode;
	slot->i0 = i0;
	slot->i1 = i1;
	slot->checksum = hash_f32(out, i0, i1);
	slot->done = 1u;
	FENCE;
	evict((void *)slot, sizeof(*slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t done_count = 0u;
		uint32_t active_mask = 0u;
		uint32_t output_hash = 0u;
		float max_abs = 0.0f;
		float sum_abs = 0.0f;

		for (uint32_t h = 0u; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				output_hash += slots[h].checksum;
			}
		}

		for (uint32_t i = 0u; i < p.elem_count; i++) {
			const float d = abs_f32(out[i] - ref[i]);

			if (d > max_abs) {
				max_abs = d;
			}
			sum_abs += d;
		}

		const uint64_t dhpm3 = hpm3_end - hpm3_start;
		const uint64_t dhpm4 = hpm4_end - hpm4_start;
		const uint64_t dhpm5 = hpm5_end - hpm5_start;
		const uint64_t dhpm6 = hpm6_end - hpm6_start;
		const uint64_t dhpm7 = hpm7_end - hpm7_start;
		const uint64_t dhpm8 = hpm8_end - hpm8_start;

		summary->magic = MAGIC;
		summary->op_mode = p.op_mode;
		summary->elem_count = p.elem_count;
		summary->rows = p.rows;
		summary->cols = p.cols;
		summary->active_harts = ACTIVE_HARTS;
		summary->active_mask = active_mask;
		summary->done_count = done_count;
		summary->output_hash = output_hash;
		summary->reference_hash = hash_f32(ref, 0u, p.elem_count);
		summary->max_abs_scaled = (uint32_t)(max_abs * 1000000.0f);
		summary->mean_abs_scaled =
			(uint32_t)((sum_abs * fast_recipf((float)p.elem_count)) *
				   1000000.0f);
		summary->hpm3_lo = (uint32_t)dhpm3;
		summary->hpm3_hi = (uint32_t)(dhpm3 >> 32);
		summary->hpm4_lo = (uint32_t)dhpm4;
		summary->hpm4_hi = (uint32_t)(dhpm4 >> 32);
		summary->hpm5_lo = (uint32_t)dhpm5;
		summary->hpm5_hi = (uint32_t)(dhpm5 >> 32);
		summary->hpm6_lo = (uint32_t)dhpm6;
		summary->hpm6_hi = (uint32_t)(dhpm6 >> 32);
		summary->hpm7_lo = (uint32_t)dhpm7;
		summary->hpm7_hi = (uint32_t)(dhpm7 >> 32);
		summary->hpm8_lo = (uint32_t)dhpm8;
		summary->hpm8_hi = (uint32_t)(dhpm8 >> 32);
		FENCE;
		evict((void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
