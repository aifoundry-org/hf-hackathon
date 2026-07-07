/*
 * DnCNN int8 TFMA kernel — full 5-layer mixed-precision network.
 *
 *   conv_first+quantize (FP32) -> hidden1/2/3 (int8 TFMA) -> dequantize+conv_final (FP32)
 *
 * conv_first/conv_final stay FP32 (they are <1% of the MACs but the most
 * quantization-sensitive) and fold the quantize/dequantize into the same pass; the 3
 * hidden layers run int8 on the tensor engine (conv_tile). Reads the harness input
 * @0x2000 and int8 weight blob @0x4000, writes the image @0x10000. Scales come from the
 * generated dncnn_int8_scales.h (gen_dncnn_int8.py).
 *
 * Partitioned across ACTIVE_HARTS harts by row band; layers exchange band boundaries
 * through DRAM (layer_publish / invalidate_read). Bit-exact (max_abs=0 vs
 * dncnn_reference_int8.npy) at 1 hart sysemu and 8 harts on board.
 */

#include <stdint.h>

#define ERBIUM_TENSOR_ASSERT(cond) ((void)(cond))
#include "erbium/isa/atomic.h"
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/tensors.h"
#include "dncnn_int8_scales.h"     /* generated: DNCNN_QUANT0, DNCNN_REQUANT[3], DNCNN_DEQUANT3 */
#include "pmc_probe.h"             /* -DDNCNN_PMC: hardware perf-counter probe (else no-op) */

extern char heap0_end[];

#define DNCNN_MAGIC 0xD3C11003u    /* CI leaderboard dump magic (score_results.py gate) */

/* ---- multi-hart config (canonical board: NUM_HARTS=16 ACTIVE_HARTS=8 BENCH_THREAD0_ONLY=1) ---- */
#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 1u
#endif
#define BENCH_FLB 2u
#define BENCH_FCC FCC_0

/* ---- network / tile shape ---- */
#define IMG_W   64u
#define IMG_H   64u
#define CH      16u                /* IC = OC = 16 hidden channels        */
#define K       3u
#define TAPS    9u                 /* 3x3                                 */
#define P       16u                /* spatial tile width (= FMA bcols)    */
#define LAYERS  5u                 /* conv_first + 3 hidden + conv_final  */
#define HIDDEN  (LAYERS - 2u)      /* 3 int8 hidden layers                */
#define NTX     (IMG_W / P)        /* tiles per row                       */
/* Padded row STRIDE in pixels: the halo-padded width (IMG_W+2 = 66) rounded up to 68
 * so a row spans a whole number of 64B cache lines (68*16 = 1088 = 17 lines). This
 * keeps every hart's row band on cache-line boundaries: each band, and the halo, own
 * their lines outright, so no line is shared between two harts writing it back. Cols
 * 66,67 are stride padding — never addressed (PAD_AT only reaches col 65). */
#define PADW    68u
#define PADH    (IMG_H + 2u)
_Static_assert((PADW * CH) % 64u == 0u,
	"row stride must be a whole number of 64B cache lines so no line spans two bands");
_Static_assert(PADW >= IMG_W + 2u, "PADW must cover the halo-padded width");
#define ROW_STRIDE_BYTES 64u       /* one L1 scratchpad cache line               */
#define QUARTET          4u        /* int8 IC packing unit (4 IC per FMA step)   */

/* L1 scratchpad line allocation for one int8 tile (per hart/minion) */
#define SCP_A_LINE      0u         /* weights A: 16 lines (one OC per line)      */
#define SCP_B_LINE      16u        /* activations B: CH/QUARTET quartet lines    */
#define SCP_SCALE_LINE  20u        /* requant per-OC scale vector: 1 line        */
#define OPCODE_INT8     3u         /* tensor_fma type = int8                     */
#define HART_SCRATCH    256u       /* per-hart bpk / temp stride (bytes)         */

/* Baked-in shape assumptions; assert so a wrong constant fails the build. */
_Static_assert(CH == 16u,        "CH must be 16 (4 IC-quartets, 32 FREGs)");
_Static_assert(P  == 16u,        "P must be 16 (FMA bcols / one store block)");
_Static_assert(IMG_W % P == 0u,  "IMG_W must be a multiple of P");

/* Per-hart scratch (bpk, temp) must hold each buffer AND be line-aligned, so that a
 * hart's own evict of its slot never shares a 64B line with the neighbouring hart's
 * slot (same false-sharing hazard as the band seams). */
_Static_assert((CH / QUARTET) * ROW_STRIDE_BYTES <= HART_SCRATCH, "bpk exceeds HART_SCRATCH");
_Static_assert(CH * P <= HART_SCRATCH, "temp exceeds HART_SCRATCH");
_Static_assert(HART_SCRATCH % 64u == 0u, "HART_SCRATCH must be a whole number of 64B lines");

/* Hart count must fit the barrier mask (1u<<ACTIVE_HARTS) and leave every band non-empty. */
_Static_assert(ACTIVE_HARTS >= 1u && ACTIVE_HARTS <= 32u, "ACTIVE_HARTS must be in [1,32]");
_Static_assert(ACTIVE_HARTS <= IMG_H, "ACTIVE_HARTS must not exceed image rows (else empty bands)");

/* FP32 boundary scales (match the numpy reference). Compile-time constants ->
 * these are multiplies, never fdiv (fdiv hangs in U-mode on this platform). */
#define FIRST_SCALE (1.0f / 128.0f)
#define FINAL_SCALE (1.0f / 16.0f)

/* ---- weight blob sub-sizes (int8, as gen_dncnn_int8.py emits) ---- */
#define W0_BYTES  (CH * K * K)              /* 144  conv_first  W0[oc][ky][kx]      */
#define WH_BYTES  (HIDDEN * CH * CH * K * K)/* 6912 hidden      WH[l][oc][ic][k]     */
#define WF_BYTES  (CH * K * K)              /* 144  conv_final  WF[ic][ky][kx]      */

/* ---- working-buffer sizes ---- */
#define IMG_BYTES     (IMG_W * IMG_H)               /* 1-channel uint8 image        */
#define ACT_BYTES     (IMG_W * IMG_H * CH)          /* NHWC uint8 activation        */
#define PAD_BYTES     (PADW * PADH * CH)            /* padded NHWC uint8            */
#define AW_TAP_BYTES  (TAPS * CH * ROW_STRIDE_BYTES)/* tap-major A operand          */

/* ---- memory map (byte offsets from base; explicit + asserted non-overlapping) ---- */
#define SLOTS_OFFSET   0x0000u     /* per-hart attestation slots (ACTIVE_HARTS x 64B) */
#define SLOT_BYTES     64u         /* one slot = one cache line (no cross-hart sharing) */
#define SUMMARY_OFFSET 0x1000u
#define BARRIER_OFFSET 0x1800u     /* cross-hart barrier state                 */
#define INPUT_OFFSET   0x2000u     /* harness: uint8[64][64] image             */
#define WEIGHTS_OFFSET 0x4000u     /* harness: int8 blob W0|WH|WF              */
#define OUTPUT_OFFSET  0x10000u    /* harness: uint8[64][64] result            */
#define QPADA_OFFSET   0x60000u    /* hidden ping-pong A (padded uint8) — 0x11000..0x60000 unused */
#define QPADB_OFFSET   0x72000u    /* hidden ping-pong B (padded uint8)        */
#define AW_TAP_OFFSET  0x84000u    /* hidden weights, re-arranged per layer    */
#define MVEC_OFFSET    0x87000u    /* per-OC requant scale line                */
#define BPACK_OFFSET   0x88000u    /* quartet B tile (4 lines), per hart       */
#define TEMP_OFFSET    0x89000u    /* quant output [OC][P], per hart           */
#define TEMP_END       0x8A000u    /* ceiling of the per-hart temp slots; 0x8A000..0xD0000 unused */
#define QDUMP_OFFSET   0xD0000u    /* 4x copies of q0..q3 interiors (debug)    */
#define DUMP_END       (QDUMP_OFFSET + 4u * ACT_BYTES)
#ifdef DNCNN_PMC
#define PMC_OFFSET     0xC0000u    /* perf-counter dump region (in 0x8A000..0xD0000 gap) */
_Static_assert(PMC_OFFSET >= TEMP_END, "PMC region collides with per-hart temp");
_Static_assert(PMC_OFFSET + sizeof(struct pmc_region) <= QDUMP_OFFSET, "PMC region overruns into QDUMP");
#endif

/* All tensor-touched buffers must be 64-byte aligned (tensor_load rounds addr). */
_Static_assert(WEIGHTS_OFFSET % 64u == 0u, "weights misaligned");
_Static_assert(AW_TAP_OFFSET  % 64u == 0u, "aw_tap misaligned");
_Static_assert(MVEC_OFFSET    % 64u == 0u, "mvec misaligned");
_Static_assert(BPACK_OFFSET   % 64u == 0u, "bpk misaligned");
/* Regions must not overlap their successor. */
_Static_assert(WEIGHTS_OFFSET + W0_BYTES + WH_BYTES + WF_BYTES <= OUTPUT_OFFSET, "weights overlap output");
_Static_assert(OUTPUT_OFFSET  + IMG_BYTES    <= QPADA_OFFSET,  "output overlaps qpadA");
_Static_assert(QPADA_OFFSET   + PAD_BYTES    <= QPADB_OFFSET,  "qpadA overlaps qpadB");
_Static_assert(QPADB_OFFSET   + PAD_BYTES    <= AW_TAP_OFFSET, "qpadB overlaps aw_tap");
_Static_assert(AW_TAP_OFFSET  + AW_TAP_BYTES <= MVEC_OFFSET,   "aw_tap overlaps mvec");
_Static_assert(DUMP_END <= 16u * 1024u * 1024u, "buffers exceed the 16MB launch region");
_Static_assert(ACTIVE_HARTS * HART_SCRATCH <= (TEMP_OFFSET - BPACK_OFFSET), "per-hart bpk region overflow");
_Static_assert(ACTIVE_HARTS * HART_SCRATCH <= (TEMP_END - TEMP_OFFSET),     "per-hart temp region overflow");
_Static_assert(ACTIVE_HARTS * SLOT_BYTES <= SUMMARY_OFFSET, "attestation slots overflow into summary");

/* pointer to padded pixel (y,x)'s CH channels; y,x are REAL coords (-1..IMG ok) */
#define PAD_AT(pad, y, x) ((pad) + ((uint32_t)((y) + 1) * PADW + ((x) + 1)) * CH)

/* tensor_fma(tenc_loc=1) overwrites f0..f31, but GCC can't see it through the CSR
 * asm, so it may keep live FP values there — force a clobber to spill them first. */
#define FREG_CLOBBER_BARRIER() __asm__ __volatile__("" ::: "memory", \
	"f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15", \
	"f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31")

/* Each hart attests its own output band in a private 64B slot; hart 0 folds them into
 * the summary. Field order in both structs is fixed by the CI scorer (score_results.py
 * reads a "<16I" summary at SUMMARY_OFFSET and gates on magic, done_count==active_harts,
 * output_sum==slot_checksum_sum). Keep these layouts in sync with the FP32 kernel. */
struct dncnn_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t row0;
	uint32_t row1;
	uint32_t active_harts;
	uint32_t checksum;      /* sum of this hart's output-band bytes */
	uint32_t done;
	uint32_t reserved[7];
};
_Static_assert(sizeof(struct dncnn_slot) == SLOT_BYTES, "slot must be exactly one cache line");

struct dncnn_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t passes;
	uint32_t width;
	uint32_t height;
	uint32_t channels;
	uint32_t layers;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t slot_checksum_sum;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t tensor_error;  /* diagnostics; ignored by the scorer's reserved tail */
	uint32_t reserved[2];
};
_Static_assert(sizeof(struct dncnn_summary) == 16u * 4u, "summary must be 16 x uint32 for the CI scorer");

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

/* ================= multi-hart barrier + hart-id (from the FP32 bench kernel) ======== */

struct bench_barrier_state {
	uint32_t count;
	uint32_t epoch;
	uint32_t reserved[14];
};
static volatile struct bench_barrier_state *g_barrier;

static inline uint32_t active_mask_t0(void)
{
#ifdef BENCH_THREAD0_ONLY
	if (ACTIVE_HARTS >= 32u) return 0xffffffffu;
	return (1u << ACTIVE_HARTS) - 1u;
#else
	uint32_t mask = 0;
	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) mask |= 1u << (h >> 1);
	return mask;
#endif
}
static inline uint32_t active_mask_t1(void)
{
#ifdef BENCH_THREAD0_ONLY
	return 0u;
#else
	uint32_t mask = 0;
	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) mask |= 1u << (h >> 1);
	return mask;
#endif
}
static inline uint32_t bench_hart_id(void)
{
#ifdef BENCH_THREAD0_ONLY
	return get_minion_id();
#else
	return get_hart_id() & 0x3fu;
#endif
}
static inline int bench_hart_enabled(uint32_t hart_id)
{
#ifdef BENCH_THREAD0_ONLY
	return get_thread_id() == 0u && hart_id < ACTIVE_HARTS && hart_id < 32u;
#else
	return hart_id < ACTIVE_HARTS && hart_id < 16u;
#endif
}
static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
#ifdef BENCH_THREAD0_ONLY
		volatile struct bench_barrier_state *const barrier = g_barrier;
		const uint32_t epoch = atomic_load_local_32(&barrier->epoch);
		const uint32_t prior = atomic_add_local_32(&barrier->count, 1u);
		if (prior + 1u == ACTIVE_HARTS) {
			atomic_store_local_32(&barrier->count, 0u);
			FENCE;
			atomic_add_local_32(&barrier->epoch, 1u);
		} else {
			while (atomic_load_local_32(&barrier->epoch) == epoch) FENCE;
		}
		FENCE;
#else
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
#endif
	}
}

/* ================= tile helpers (each tensor CSR encoding lives here once) ========= */

/* Gather one tap's activation window into the quartet-interleaved B layout (4 lines). */
static inline void pack_b_tile(int8_t *restrict bpk, const uint8_t *restrict pad,
			       uint32_t y, uint32_t x0, int dy, int dx)
{
	for (uint32_t q = 0; q < CH / QUARTET; q++)
		for (uint32_t j = 0; j < P; j++)
			for (uint32_t x = 0; x < QUARTET; x++)
				bpk[q * ROW_STRIDE_BYTES + j * QUARTET + x] =
					(int8_t)PAD_AT(pad, (int)y + dy, (int)(x0 + j) + dx)[q * QUARTET + x];
}

/* One int8 tap MAC (16 OC x 16 IC x 16 spatial). A is signed / B unsigned — note the
 * tensors.h argument names for the two operands are swapped. first: reset the TENC
 * accumulator; last: copy TENC into the float registers for the store. */
static inline void fma_tap(int first, int last)
{
	tensor_fma(false, (P / QUARTET) - 1u, CH - 1u, (CH / QUARTET) - 1u, 0,
		   (bool)last, false, true, false, SCP_B_LINE, SCP_A_LINE, OPCODE_INT8, (bool)first);
}

/* Fused int32-accumulator -> uint8 requant, in FREGs (dequant/scale/round/relu+clamp/pack). */
static inline void requant_u8(uint32_t scp_scale)
{
	tensor_quant(0, (P / QUARTET) - 1u, CH - 1u, scp_scale,
		     QUANT_LAST_TRANS, QUANT_LAST_TRANS, QUANT_LAST_TRANS,
		     QUANT_LAST_TRANS, QUANT_LAST_TRANS,
		     QUANT_PACK_128B, QUANT_SATUINT8, QUANT_FP32_TO_INT32,
		     QUANT_FP32_MUL_COL, QUANT_INT32_TO_FP32);
}

/* Store packed uint8 [OC][P] (rows in FREG 0,2,4,...; one 16B block/row; P stride). */
static inline void store_u8_tile(uint8_t *dst)
{
	tensor_store(1, 0, 0, CH - 1u, (uint64_t)dst, 0, P);
}

/* One hidden-layer output tile: 9-tap int8 accumulate -> requant -> uint8 [OC][P],
 * scattered into a PADDED destination's interior (so the next layer's pack_B has neighbors). */
static void conv_tile(const uint8_t *restrict pad, const int8_t *restrict aw,
		      int8_t *restrict bpk, const float *restrict mvec,
		      uint8_t *restrict temp, uint8_t *restrict padout,
		      uint32_t y, uint32_t x0)
{
	for (uint32_t t = 0; t < TAPS; t++) {
		pack_b_tile(bpk, pad, y, x0, (int)(t / K) - 1, (int)(t % K) - 1);
		FENCE; evict(bpk, (CH / QUARTET) * ROW_STRIDE_BYTES); WAIT_CACHEOPS;

		tensor_load(false, false, SCP_A_LINE, 0, 0,
			    (uint64_t)(aw + t * CH * ROW_STRIDE_BYTES), 0,
			    CH - 1u, ROW_STRIDE_BYTES, 0);              /* A_t: 16 weight lines */
		tensor_load(false, false, SCP_B_LINE, 0, 0, (uint64_t)bpk, 0,
			    (CH / QUARTET) - 1u, ROW_STRIDE_BYTES, 0);  /* B_t: 4 quartet lines */
		tensor_wait(TENSOR_LOAD_WAIT_0);

		fma_tap(/*first*/ t == 0u, /*last*/ t == TAPS - 1u);
		tensor_wait(TENSOR_FMA_WAIT);
	}

	tensor_load(false, false, SCP_SCALE_LINE, 0, 0, (uint64_t)mvec, 0,
		    0u, ROW_STRIDE_BYTES, 0);                           /* per-OC requant scale */
	tensor_wait(TENSOR_LOAD_WAIT_0);
	requant_u8(SCP_SCALE_LINE);  tensor_wait(TENSOR_QUANT_WAIT);
	store_u8_tile(temp);  tensor_wait(TENSOR_STORE_WAIT);
	FREG_CLOBBER_BARRIER();
	FENCE; evict(temp, CH * P); WAIT_CACHEOPS;

	for (uint32_t oc = 0; oc < CH; oc++)
		for (uint32_t j = 0; j < P; j++)
			PAD_AT(padout, y, x0 + j)[oc] = temp[oc * P + j];
}

/* Replicate-fill the 1-pixel halo around rows [row0,row1): left/right columns for every
 * row, the top halo row when row0==0, the bottom halo row when row1==IMG_H. layer_publish
 * calls it once over the whole image (0,IMG_H) on hart 0, which owns the shared halo. */
static void fill_halo_band(uint8_t *pad, uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++)              /* left/right columns of my rows */
		for (uint32_t ic = 0; ic < CH; ic++) {
			PAD_AT(pad, (int)y, -1)[ic]         = PAD_AT(pad, (int)y, 0)[ic];
			PAD_AT(pad, (int)y, (int)IMG_W)[ic] = PAD_AT(pad, (int)y, (int)IMG_W - 1)[ic];
		}
	if (row0 == 0u)                                     /* top halo row + corners */
		for (int x = -1; x <= (int)IMG_W; x++)
			for (uint32_t ic = 0; ic < CH; ic++)
				PAD_AT(pad, -1, x)[ic] = PAD_AT(pad, 0, x)[ic];
	if (row1 == IMG_H)                                  /* bottom halo row + corners */
		for (int x = -1; x <= (int)IMG_W; x++)
			for (uint32_t ic = 0; ic < CH; ic++)
				PAD_AT(pad, (int)IMG_H, x)[ic] = PAD_AT(pad, (int)IMG_H - 1, x)[ic];
}

/* Publish this hart's just-written band so all bands + the halo are globally visible
 * before the next layer reads across band boundaries:
 *   evict my band -> barrier -> hart 0 fills the whole halo + evicts whole buffer -> barrier.
 * Barrier 1 puts every band in DRAM before hart 0 reads them to build the halo;
 * barrier 2 puts the halo in DRAM before anyone reads it. The read-side invalidate is
 * a separate step (invalidate_read) issued right before each read, so a neighbour that
 * has already published cannot leave a stale copy cached in this hart's L1. */
static void layer_publish(uint8_t *pad, uint32_t hart_id, uint32_t row0, uint32_t row1)
{
	FENCE;
	evict(pad + (uint64_t)(row0 + 1u) * PADW * CH, (uint64_t)(row1 - row0) * PADW * CH);
	WAIT_CACHEOPS;
	bench_barrier();
	if (hart_id == 0u) {
		fill_halo_band(pad, 0u, IMG_H);   /* whole-image halo */
		FENCE;
		evict(pad, PAD_BYTES);
		WAIT_CACHEOPS;
	}
	bench_barrier();
}

/* Invalidate this hart's read window (band +/- 1 row) so it reads neighbours' fresh
 * data from DRAM. Call RIGHT BEFORE the read (no barrier between invalidate and read). */
static void invalidate_read(const uint8_t *pad, uint32_t row0, uint32_t row1)
{
	const uint32_t rp1 = (row1 + 2u <= PADH) ? (row1 + 2u) : PADH;
	evict(pad + (uint64_t)row0 * PADW * CH, (uint64_t)(rp1 - row0) * PADW * CH);
	WAIT_CACHEOPS;
}

/* ================= FP32 boundary + quant glue (scalar; matches numpy exactly) ===== */

/* edge-replicate clamp of a coordinate into [0, lim) */
static inline int clampi(int v, uint32_t lim)
{
	if (v < 0) return 0;
	if (v >= (int)lim) return (int)lim - 1;
	return v;
}

/* round-to-nearest-even (matches numpy np.rint); no libm, no fdiv */
static inline int32_t rint_rne(float v)
{
	int32_t r;
	__asm__("fcvt.w.s %0, %1, rne" : "=r"(r) : "f"(v));
	return r;
}

/* round-half-up = floor(v + 0.5): matches conv_final's np.floor(val+0.5). rdn = round down. */
static inline int32_t round_half_up(float v)
{
	int32_t r; const float vv = v + 0.5f;
	__asm__("fcvt.w.s %0, %1, rdn" : "=r"(r) : "f"(vv));
	return r;
}

/* conv_first with the quantize folded in: uint8 image (1ch) -> uint8 padded activation,
 * written straight into qpad's interior (no intermediate FP32 buffer). Per output:
 * x=img-128, 3x3, xFIRST_SCALE, ReLU, then xQUANT0, round-nearest-even, clamp[0,255].
 * Accumulates in FP32 in the reference's tap order so the result matches numpy exactly. */
static void conv_first_quant(const uint8_t *restrict img, const int8_t *restrict w0,
			     uint8_t *restrict qpad, float qscale, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		for (uint32_t y = row0; y < row1; y++) {
			for (uint32_t x = 0; x < IMG_W; x++) {
				float acc = 0.0f;
				for (int ky = -1; ky <= 1; ky++) {
					const int yy = clampi((int)y + ky, IMG_H);
					for (int kx = -1; kx <= 1; kx++) {
						const int xx = clampi((int)x + kx, IMG_W);
						const float xv =
							(float)img[(uint32_t)yy * IMG_W + (uint32_t)xx] - 128.0f;
						acc += xv * (float)w0[oc * K * K +
								     (uint32_t)(ky + 1) * K + (uint32_t)(kx + 1)];
					}
				}
				const float o = acc * FIRST_SCALE;
				int32_t q = rint_rne((o > 0.0f ? o : 0.0f) * qscale);
				if (q < 0) q = 0; else if (q > 255) q = 255;
				PAD_AT(qpad, (int)y, (int)x)[oc] = (uint8_t)q;
			}
		}
	}
}

/* conv_final with the dequant folded in: reads the halo-filled padded uint8 activation
 * directly and produces the uint8 image. Per output: dequantize each input (xDEQUANT3),
 * 3x3, xFINAL_SCALE, +128, round-half-up, clamp[0,255]. Neighbours come from the halo,
 * so no coordinate clamping is needed. Accumulates in FP32 to match numpy exactly. */
static void conv_final_dequant(const uint8_t *restrict cur, float dequant,
			       const int8_t *restrict wf, uint8_t *restrict out,
			       uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++) {
		for (uint32_t x = 0; x < IMG_W; x++) {
			float acc = 0.0f;
			for (int ky = -1; ky <= 1; ky++) {
				for (int kx = -1; kx <= 1; kx++) {
					const uint8_t *const px = PAD_AT(cur, (int)y + ky, (int)x + kx);
					for (uint32_t ic = 0; ic < CH; ic++)
						acc += ((float)px[ic] * dequant) *
						       (float)wf[ic * K * K +
								 (uint32_t)(ky + 1) * K + (uint32_t)(kx + 1)];
				}
			}
			int32_t o = round_half_up(128.0f + acc * FINAL_SCALE);
			if (o < 0) o = 0; else if (o > 255) o = 255;
			out[y * IMG_W + x] = (uint8_t)o;
		}
	}
}

/* Sum of the output bytes in rows [row0,row1). Each hart checksums its own band; the
 * bands tile the image, so the band sums add up to the whole-image sum (the CI self-check). */
static uint32_t stripe_checksum(const uint8_t *out, uint32_t row0, uint32_t row1)
{
	uint32_t sum = 0;
	for (uint32_t y = row0; y < row1; y++)
		for (uint32_t x = 0; x < IMG_W; x++)
			sum += out[y * IMG_W + x];
	return sum;
}

/* re-arrange one hidden layer's weights from the blob's WH[l][oc][ic][k] into the
 * FMA's tap-major A layout aw_tap[t][oc][ic] (tap t == kernel index k). */
static void rearrange_hidden_weights(const int8_t *restrict wh, uint32_t layer,
				     int8_t *restrict aw_tap)
{
	const int8_t *const W = wh + layer * CH * CH * (K * K);   /* WH[layer] */
	for (uint32_t oc = 0; oc < CH; oc++)
		for (uint32_t ic = 0; ic < CH; ic++)
			for (uint32_t t = 0; t < TAPS; t++)
				aw_tap[t * CH * ROW_STRIDE_BYTES + oc * ROW_STRIDE_BYTES + ic] =
					W[oc * CH * (K * K) + ic * (K * K) + t];
}

#ifdef DNCNN_DUMP
/* copy this hart's band of a padded buffer into a flat dump slot + evict it (debug only) */
static void dump_band(const uint8_t *restrict pad, uint8_t *restrict dst,
		      uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++)
		for (uint32_t x = 0; x < IMG_W; x++)
			for (uint32_t oc = 0; oc < CH; oc++)
				dst[(y * IMG_W + x) * CH + oc] = PAD_AT(pad, (int)y, (int)x)[oc];
	FENCE;
	evict(dst + (uint64_t)row0 * IMG_W * CH, (uint64_t)(row1 - row0) * IMG_W * CH);
	WAIT_CACHEOPS;
}
#endif

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = bench_hart_id();
	if (!bench_hart_enabled(hart_id)) {
		return 0;
	}

	uint8_t *const base = (uint8_t *)buffer_base_from_args(arg_area);
	const uint8_t *const input   = base + INPUT_OFFSET;
	const int8_t  *const weights = (const int8_t *)(base + WEIGHTS_OFFSET);
	const int8_t  *const w0      = weights;
	const int8_t  *const wh      = weights + W0_BYTES;
	const int8_t  *const wf      = weights + W0_BYTES + WH_BYTES;
	uint8_t *const output = base + OUTPUT_OFFSET;
	uint8_t *const qpadA  = base + QPADA_OFFSET;
	uint8_t *const qpadB  = base + QPADB_OFFSET;
	int8_t  *const aw_tap = (int8_t *)(base + AW_TAP_OFFSET);   /* shared (hart 0 builds) */
	float   *const mvec   = (float *)(base + MVEC_OFFSET);      /* shared (hart 0 builds) */
	int8_t  *const bpk    = (int8_t *)(base + BPACK_OFFSET + hart_id * HART_SCRATCH);  /* per-hart */
	uint8_t *const temp   = base + TEMP_OFFSET + hart_id * HART_SCRATCH;               /* per-hart */
#ifdef DNCNN_DUMP
	uint8_t *const qdump  = base + QDUMP_OFFSET;
#endif
	volatile struct dncnn_slot *const slots =
		(volatile struct dncnn_slot *)(base + SLOTS_OFFSET);
	volatile struct dncnn_summary *const summary =
		(volatile struct dncnn_summary *)(base + SUMMARY_OFFSET);
	g_barrier = (volatile struct bench_barrier_state *)(base + BARRIER_OFFSET);

	/* this hart's row band [row0, row1) */
	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

#ifdef DNCNN_PMC
	pmc_probe_begin(base + PMC_OFFSET, hart_id, ACTIVE_HARTS);
#endif

	/* ---- Stage 1: conv_first + quantize -> qpadA band, then publish it across harts ---- */
	conv_first_quant(input, w0, qpadA, DNCNN_QUANT0, row0, row1);
#ifdef DNCNN_DUMP
	dump_band(qpadA, qdump + 0u * ACT_BYTES, row0, row1);   /* q0 -> dump slot 0 */
#endif
	layer_publish(qpadA, hart_id, row0, row1);

	/* ---- Stage 2: the 3 int8 hidden layers, banded, with a publish/sync between layers ---- */
	uint8_t *cur = qpadA;   /* holds q0 (synced) */
	uint8_t *nxt = qpadB;
	for (uint32_t l = 0; l < HIDDEN; l++) {
		if (hart_id == 0u) {                          /* build the shared weights + scale */
			rearrange_hidden_weights(wh, l, aw_tap);
			for (uint32_t oc = 0; oc < CH; oc++)
				mvec[oc] = DNCNN_REQUANT[l];
			FENCE;
			evict(aw_tap, AW_TAP_BYTES);
			evict(mvec, 64u);
			WAIT_CACHEOPS;
		}
		bench_barrier();                              /* all harts wait for aw_tap/mvec in DRAM */

		invalidate_read(cur, row0, row1);             /* fresh neighbours, right before reading */
		for (uint32_t ty = row0; ty < row1; ty++)     /* my row band only */
			for (uint32_t tx = 0; tx < NTX; tx++)
				conv_tile(cur, aw_tap, bpk, mvec, temp, nxt, ty, tx * P);
#ifdef DNCNN_DUMP
		dump_band(nxt, qdump + (l + 1u) * ACT_BYTES, row0, row1);
#endif
		layer_publish(nxt, hart_id, row0, row1);

		uint8_t *const t = cur; cur = nxt; nxt = t;       /* ping-pong; cur ends holding q3 */
	}

	/* ---- Stage 3: conv_final + dequant, reads the synced activation directly -> output ---- */
	invalidate_read(cur, row0, row1);                 /* fresh neighbours before conv_final reads */
	conv_final_dequant(cur, DNCNN_DEQUANT3, wf, output, row0, row1);

	FENCE;
	evict(output + (uint64_t)row0 * IMG_W, (uint64_t)(row1 - row0) * IMG_W);   /* my band */
	WAIT_CACHEOPS;
	bench_barrier();   /* whole output image in DRAM before anyone checksums it */

#ifdef DNCNN_PMC
	pmc_probe_end(base + PMC_OFFSET, hart_id);
#endif

	/* Attest this hart's band: publish a slot with the band checksum, so hart 0 can
	 * fold them and cross-check against a fresh whole-image sum (the CI correctness gate). */
	volatile struct dncnn_slot *const my_slot = &slots[hart_id];
	my_slot->magic        = DNCNN_MAGIC;
	my_slot->hart_id      = hart_id;
	my_slot->minion_id    = get_minion_id();
	my_slot->thread_id    = get_thread_id();
	my_slot->row0         = row0;
	my_slot->row1         = row1;
	my_slot->active_harts = ACTIVE_HARTS;
	my_slot->checksum     = stripe_checksum(output, row0, row1);
	my_slot->done         = 1u;
	FENCE;
	evict((const void *)my_slot, sizeof(*my_slot));
	WAIT_CACHEOPS;
	bench_barrier();   /* all slots in DRAM before hart 0 reads them */

	if (hart_id == 0u) {
		uint32_t active_mask = 0u, done_count = 0u, slot_checksum_sum = 0u, output_sum = 0u;
		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == DNCNN_MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}
		for (uint32_t i = 0; i < IMG_BYTES; i++)
			output_sum += output[i];

		const uint64_t macs = (uint64_t)IMG_W * IMG_H *
			((uint64_t)CH * K * K + (uint64_t)HIDDEN * CH * CH * K * K + (uint64_t)CH * K * K);
		const uint64_t ops = macs * 2u;   /* one pass; multiply + add */

		summary->magic             = DNCNN_MAGIC;
		summary->active_harts      = ACTIVE_HARTS;
		summary->passes            = 1u;
		summary->width             = IMG_W;
		summary->height            = IMG_H;
		summary->channels          = CH;
		summary->layers            = LAYERS;
		summary->active_mask       = active_mask;
		summary->done_count        = done_count;
		summary->output_sum        = output_sum;
		summary->slot_checksum_sum = slot_checksum_sum;
		summary->ops_lo            = (uint32_t)ops;
		summary->ops_hi            = (uint32_t)(ops >> 32);
		summary->tensor_error      = (uint32_t)get_tensor_error();
		FENCE;
		evict((const void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}
