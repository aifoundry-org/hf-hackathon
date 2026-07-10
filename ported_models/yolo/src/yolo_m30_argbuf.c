/*
 * YOLOv10n M30 - end-to-end detector path for fixed 480x640 RGB samples.
 *
 * Runs on-chip raw-RGB preprocessing, YOLOv10n backbone/neck/head, DFL decode,
 * class sigmoid, thresholding, and class-aware NMS. The host reads the compact
 * detection list at DETECTIONS_OFFSET.
 *
 * Key layout:
 *   0x04A00000  raw RGB input [480,640,3]
 *   0x02000000  external weights region
 *   0x01C00000  final [1,84,3024] decoded head tensor
 *   0x01D00000  compact detections {N, class_id, score, box}
 *
 * Scratch base starts at 0x3000000. The total buffer is 80 MB.
 */
#include "yolo_common.h"
#include "yolo_weight_offsets.h"

#define INPUT_OFFSET        0x00010000u
#define RAW_INPUT_OFFSET    0x04A00000u   /* uint8 RGB [SH, SW, 3], HWC, host-loaded */
#define DETECTIONS_OFFSET   0x01D00000u   /* host reads small detection list here */
#define SRC_H               480u           /* hard-coded for 640x480 RGB inputs */
#define SRC_W               640u
#define DST_H               288u
#define DST_W               512u
#define MAX_DETECTIONS      64u            /* output cap after class-aware NMS */
#define CONV0_OUT_OFFSET    0x00300000u
#define CONV1_OUT_OFFSET    0x00600000u
#define C2F_M2_OUT_OFFSET   0x00800000u
#define CONV3_OUT_OFFSET    0x00A00000u
#define C2F_M4_OUT_OFFSET   0x00B00000u
#define CONV5_OUT_OFFSET    0x00C00000u
#define C2F_M6_OUT_OFFSET   0x00E00000u
#define CONV7_OUT_OFFSET    0x00F00000u
#define C2F_M8_OUT_OFFSET   0x01100000u
#define SPPF_M9_OUT_OFFSET  0x01200000u
#define PSA_M10_OUT_OFFSET  0x01300000u
#define HEAD_P3_IN_OFFSET   0x01400000u   /* m.16 cv2 out  [1, 64,36,64] */
#define HEAD_P4_IN_OFFSET   0x01500000u   /* m.19 cv2 out  [1,128,18,32] */
#define HEAD_P5_IN_OFFSET   0x01600000u   /* m.22 cv2 out  [1,256, 9,16] */
#define REG_LOGITS_0_OFFSET 0x01700000u   /* [64,36,64] = 576 KB */
#define CLS_LOGITS_0_OFFSET 0x01800000u   /* [80,36,64] = 720 KB */
#define REG_LOGITS_1_OFFSET 0x01A00000u   /* [64,18,32] = 144 KB */
#define CLS_LOGITS_1_OFFSET 0x01A80000u   /* [80,18,32] = 180 KB */
#define REG_LOGITS_2_OFFSET 0x01B00000u   /* [64, 9,16] =  36 KB */
#define CLS_LOGITS_2_OFFSET 0x01B40000u   /* [80, 9,16] =  45 KB */
#define SCR_HEAD_A          0x04980000u   /* 1 MB ping */
#define SCR_HEAD_B          0x04A80000u   /* 1 MB pong */
#define SCR_HEAD_C          0x04B80000u   /* 1 MB cls dw */
#define SCR_HEAD_D          0x04C80000u   /* 1 MB cls more */
#define FINAL_OUT_OFFSET    0x01C00000u   /* [1, 84, 3024] = 992 KB <- M9 tap */

/* Scratch slots (after weights at 0x2000000+12MB -> use 0x3000000+).
 * Sizes are 64-byte multiples; every block is followed by a slack region
 * before the next slot to make overlap impossible.
 *
 *   tensor                    bytes      (channels x HW x 4)
 *   m2_concat   [48,72,128]   1,769,472  = 0x1B0000
 *   m2_m0_cv1   [16,72,128]     589,824  = 0x090000
 *   m4_concat   [128,36,64]   1,179,648  = 0x120000
 *   m4_mX_cvY   [32,36,64]      294,912  = 0x048000
 *   m5_cv2      [128,18,32]     294,912  = 0x048000
 *   m6_concat   [256,18,32]     589,824  = 0x090000
 *   m6_mX_cvY   [64,18,32]      147,456  = 0x024000
 *   m7_cv2      [256,9,16]      147,456  = 0x024000
 *   m8_concat   [384,9,16]      221,184  = 0x036000
 *   m8_m0_cvY   [128,9,16]       73,728  = 0x012000
 *   m9_*        [128,9,16]       73,728  = 0x012000
 *   m9_concat   [512,9,16]      294,912  = 0x048000
 */
#define SCR_M2_CONCAT       0x03000000u   /* + 0x1B0000 */
#define SCR_M2_M0_CV1       0x031C0000u   /* + 0x090000 */
#define SCR_M2_M0_CV2       0x03260000u   /* + 0x090000 */
#define SCR_M4_CONCAT       0x03300000u   /* + 0x120000 */
#define SCR_M4_M0_CV1       0x03430000u   /* + 0x048000 */
#define SCR_M4_M0_CV2       0x03480000u   /* + 0x048000 */
#define SCR_M4_M1_CV1       0x034D0000u   /* + 0x048000 */
#define SCR_M4_M1_CV2       0x03520000u   /* + 0x048000 */
#define SCR_M5_CV2_OUT      0x03570000u   /* + 0x048000 */
#define SCR_M6_CONCAT       0x035C0000u   /* + 0x090000 */
#define SCR_M6_M0_CV1       0x03660000u   /* + 0x024000 */
#define SCR_M6_M0_CV2       0x03690000u   /* + 0x024000 */
#define SCR_M6_M1_CV1       0x036C0000u   /* + 0x024000 */
#define SCR_M6_M1_CV2       0x036F0000u   /* + 0x024000 */
#define SCR_M7_CV2_OUT      0x03720000u   /* + 0x024000 */
#define SCR_M8_CONCAT       0x03750000u   /* + 0x036000 */
#define SCR_M8_M0_CV1       0x03790000u   /* + 0x012000 */
#define SCR_M8_M0_CV2       0x037B0000u   /* + 0x012000 */
#define SCR_M9_CV1_OUT      0x037D0000u   /* + 0x012000 */
#define SCR_M9_MP1          0x037F0000u   /* + 0x012000 */
#define SCR_M9_MP2          0x03810000u   /* + 0x012000 */
#define SCR_M9_MP3          0x03830000u   /* + 0x012000 */
#define SCR_M9_CONCAT       0x03850000u   /* + 0x048000 */

/* PSA scratch: well above all M5 scratch (which ends at 0x3898000). */
#define SCR_PSA_CV1_OUT     0x03A00000u   /* [256, 9,16] = 144 KB */
#define SCR_PSA_Y1_BUF      0x03A30000u   /* [128, 9,16] =  72 KB (mutable copy) */
#define SCR_PSA_QKV         0x03A50000u   /* [256, 9,16] = 144 KB */
#define SCR_PSA_Q           0x03A80000u   /* [2, 32,144] =  36 KB */
#define SCR_PSA_K           0x03A90000u
#define SCR_PSA_V           0x03AA0000u   /* [2, 64,144] =  72 KB */
#define SCR_PSA_QT          0x03AC0000u   /* [2,144, 32] =  36 KB */
#define SCR_PSA_LOGITS      0x03AD0000u   /* [2,144,144] = 162 KB */
#define SCR_PSA_SM_T        0x03B10000u
#define SCR_PSA_V_RESH      0x03B50000u   /* [128,9,16]  =  72 KB */
#define SCR_PSA_PE          0x03B70000u
#define SCR_PSA_ATTN_OUT    0x03B90000u
#define SCR_PSA_PROJ        0x03BB0000u
#define SCR_PSA_FFN0        0x03BD0000u   /* [256,9,16]  = 144 KB */
#define SCR_PSA_FFN1        0x03C00000u

/* FPN scratch.  Sizes carefully recomputed (256-ch tensors are 0x90000,
 * not 0x48000).  Earlier bug: m11_up at 0x4000000 overlapped m12_cat
 * because I'd reserved 0x48000 instead of 0x90000 - bad channels 142+. */
#define SCR_M11_UP          0x04000000u   /* [256,18,32] 0x90000 */
#define SCR_M12_CONCAT      0x04090000u   /* [384,18,32] 0xE0000 */
#define SCR_M13_CV1         0x04170000u   /* [128,18,32] 0x50000 */
#define SCR_M13_M0_CV1      0x041C0000u   /* [64,18,32]  0x30000 */
#define SCR_M13_M0_CV2      0x041F0000u
#define SCR_M13_CV2_OUT     0x04220000u   /* [128,18,32] 0x50000 */
#define SCR_M14_UP          0x04270000u   /* [128,36,64] 0x130000 */
#define SCR_M15_CONCAT      0x043A0000u   /* [192,36,64] 0x1C0000 */
#define SCR_M16_CV1         0x04560000u   /* [64,36,64]  0x90000 */
#define SCR_M16_M0_CV1      0x045F0000u   /* [32,36,64]  0x50000 */
#define SCR_M16_M0_CV2      0x04640000u
#define SCR_M17_DOWN        0x04690000u   /* [64,18,32]  0x30000 */
#define SCR_M18_CONCAT      0x046C0000u   /* [192,18,32] 0x70000 */
#define SCR_M19_CV1         0x04730000u   /* [128,18,32] 0x50000 */
#define SCR_M19_M0_CV1      0x04780000u   /* [64,18,32]  0x30000 */
#define SCR_M19_M0_CV2      0x047B0000u
#define SCR_M20_CV1         0x047E0000u   /* [128,18,32] 0x50000 */
#define SCR_M20_DOWN        0x04830000u   /* [128, 9,16] 0x20000 */
#define SCR_M21_CONCAT      0x04850000u   /* [384, 9,16] 0x40000 */
#define SCR_M22_CV1         0x04890000u   /* [256, 9,16] 0x30000 */
#define SCR_M22_T0          0x048C0000u   /* [128, 9,16] 0x20000 */
#define SCR_M22_T1          0x048E0000u   /* [256, 9,16] 0x30000 */
#define SCR_M22_T2          0x04910000u   /* [256, 9,16] 0x30000 */
#define SCR_M22_T3          0x04940000u   /* [128, 9,16] 0x20000 */
#define SCR_M22_T4          0x04960000u   /* [128, 9,16] 0x20000 */
/* total scratch ends at 0x4980000 = 73.5 MB; 80 MB buffer available. */

#define WP(off) ((const float *)(base + WEIGHT_REGION_OFFSET + (off)))

int main(uintptr_t arg_area)
{
    const uint32_t hid = get_hart_id();
    if (!mh_is_active_hart(hid)) return 0;
    const int is_h0 = mh_is_leader(hid);

    uint8_t *base = (uint8_t *)buffer_base_from_args(arg_area);
    mh_init_barrier(base);

    /* === STAGE 0: PREPROCESS on silicon ===
     * Read raw uint8 RGB image at RAW_INPUT_OFFSET (HWC, [SRC_H, SRC_W, 3]).
     * Bilinear resize -> divide by 255 -> transpose HWC->CHW.
     * Write FP32 [1, 3, DST_H, DST_W] at INPUT_OFFSET (where the model expects).
     *
     * Multi-hart by output row (DST_H = 288 split across 8 T0 harts -> 36 rows each). */
    if (yolo_is_compute(hid)) {
        const uint32_t cidx = yolo_compute_idx(hid);
        uint32_t oh_lo, oh_hi;
        yolo_range(DST_H, cidx, &oh_lo, &oh_hi);

        const uint8_t *raw = (const uint8_t *)(base + RAW_INPUT_OFFSET);
        float       *fp   = (float       *)(base + INPUT_OFFSET);

        const float scale_h = (float)SRC_H / (float)DST_H;
        const float scale_w = (float)SRC_W / (float)DST_W;
        const float inv255  = 1.0f / 255.0f;

        for (uint32_t oh = oh_lo; oh < oh_hi; oh++) {
            const float src_h = ((float)oh + 0.5f) * scale_h - 0.5f;
            int32_t h0 = (int32_t)src_h;
            if (h0 < 0) h0 = 0;
            int32_t h1 = h0 + 1;
            if (h1 >= (int32_t)SRC_H) h1 = SRC_H - 1;
            const float dh = src_h - (float)h0;
            const float wh0 = 1.0f - dh, wh1 = dh;

            for (uint32_t ow = 0; ow < DST_W; ow++) {
                const float src_w = ((float)ow + 0.5f) * scale_w - 0.5f;
                int32_t w0 = (int32_t)src_w;
                if (w0 < 0) w0 = 0;
                int32_t w1 = w0 + 1;
                if (w1 >= (int32_t)SRC_W) w1 = SRC_W - 1;
                const float dw = src_w - (float)w0;
                const float ww0 = 1.0f - dw, ww1 = dw;

                /* For each channel c in 0..3: bilinear blend of 4 source pixels. */
                for (uint32_t c = 0; c < 3u; c++) {
                    const float p00 = (float)raw[((uint32_t)h0 * SRC_W + (uint32_t)w0) * 3u + c];
                    const float p01 = (float)raw[((uint32_t)h0 * SRC_W + (uint32_t)w1) * 3u + c];
                    const float p10 = (float)raw[((uint32_t)h1 * SRC_W + (uint32_t)w0) * 3u + c];
                    const float p11 = (float)raw[((uint32_t)h1 * SRC_W + (uint32_t)w1) * 3u + c];
                    const float v = (wh0 * (ww0 * p00 + ww1 * p01) +
                                     wh1 * (ww0 * p10 + ww1 * p11)) * inv255;
                    fp[(c * DST_H + oh) * DST_W + ow] = v;
                }
            }
        }
        /* Evict our slice across all 3 channels.  channel-stride = DST_H*DST_W. */
        if (oh_hi > oh_lo) {
            for (uint32_t c = 0; c < 3u; c++) {
                evict((const void *)(fp + c * DST_H * DST_W + oh_lo * DST_W),
                      (oh_hi - oh_lo) * DST_W * sizeof(float));
            }
        }
    }
    MH_BARRIER();

    const float *img    = (const float *)(base + INPUT_OFFSET);
    float       *c0     = (float *)(base + CONV0_OUT_OFFSET);
    float       *c1     = (float *)(base + CONV1_OUT_OFFSET);
    float       *c2f_m2 = (float *)(base + C2F_M2_OUT_OFFSET);
    float       *c3     = (float *)(base + CONV3_OUT_OFFSET);
    float       *c2f_m4 = (float *)(base + C2F_M4_OUT_OFFSET);
    float       *c5_post= (float *)(base + CONV5_OUT_OFFSET);
    float       *c2f_m6 = (float *)(base + C2F_M6_OUT_OFFSET);
    float       *c7_post= (float *)(base + CONV7_OUT_OFFSET);
    float       *c2f_m8 = (float *)(base + C2F_M8_OUT_OFFSET);
    float       *sppf   = (float *)(base + SPPF_M9_OUT_OFFSET);

    /* === M2 Layers (kept from M3/M4) === */

    /* conv0 */
    CONV_3x3_S2_P1_VPU(img, c0, WP(WR_model_0_conv_Conv_W), WP(WR_model_0_conv_Conv_B),
                       3u, 288u, 512u, 16u, 144u, 256u, 1u);
    /* conv1 */
    CONV_3x3_S2_P1_VPU(c0, c1, WP(WR_model_1_conv_Conv_W), WP(WR_model_1_conv_Conv_B),
                       16u, 144u, 256u, 32u, 72u, 128u, 1u);

    /* C2f model.2 (1 bottleneck) */
    {
        float *concat = (float *)(base + SCR_M2_CONCAT);
        float *y1     = concat + 16u * 72u * 128u;
        float *m0_out = concat + 32u * 72u * 128u;
        float *m0_cv1 = (float *)(base + SCR_M2_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M2_M0_CV2);
        CONV_1x1(c1, concat, WP(WR_model_2_cv1_conv_Conv_W), WP(WR_model_2_cv1_conv_Conv_B), 32u, 72u, 128u, 32u, 1u);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_2_m_0_cv1_conv_Conv_W), WP(WR_model_2_m_0_cv1_conv_Conv_B), 16u, 72u, 128u, 16u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_2_m_0_cv2_conv_Conv_W), WP(WR_model_2_m_0_cv2_conv_Conv_B), 16u, 72u, 128u, 16u, 1u);
        MH_ADD(m0_out, y1, m0_cv2, 16u * 72u * 128u);
        CONV_1x1(concat, c2f_m2, WP(WR_model_2_cv2_conv_Conv_W), WP(WR_model_2_cv2_conv_Conv_B), 48u, 72u, 128u, 32u, 1u);
    }

    /* conv3: 32 -> 64, 3x3 s=2 */
    CONV_3x3_S2_P1_VPU(c2f_m2, c3, WP(WR_model_3_conv_Conv_W), WP(WR_model_3_conv_Conv_B),
                       32u, 72u, 128u, 64u, 36u, 64u, 1u);

    /* C2f model.4 (2 bottlenecks) */
    {
        const uint32_t HW = 36u * 64u;
        float *concat = (float *)(base + SCR_M4_CONCAT);
        float *y1     = concat + 32u * HW;
        float *m0_out = concat + 64u * HW;
        float *m1_out = concat + 96u * HW;
        float *m0_cv1 = (float *)(base + SCR_M4_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M4_M0_CV2);
        float *m1_cv1 = (float *)(base + SCR_M4_M1_CV1);
        float *m1_cv2 = (float *)(base + SCR_M4_M1_CV2);
        CONV_1x1(c3, concat, WP(WR_model_4_cv1_conv_Conv_W), WP(WR_model_4_cv1_conv_Conv_B), 64u, 36u, 64u, 64u, 1u);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_4_m_0_cv1_conv_Conv_W), WP(WR_model_4_m_0_cv1_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_4_m_0_cv2_conv_Conv_W), WP(WR_model_4_m_0_cv2_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);
        MH_ADD(m0_out, y1, m0_cv2, 32u * HW);
        CONV_3x3_P1_VPU(m0_out, m1_cv1, WP(WR_model_4_m_1_cv1_conv_Conv_W), WP(WR_model_4_m_1_cv1_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);
        CONV_3x3_P1_VPU(m1_cv1, m1_cv2, WP(WR_model_4_m_1_cv2_conv_Conv_W), WP(WR_model_4_m_1_cv2_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);
        MH_ADD(m1_out, m0_out, m1_cv2, 32u * HW);
        CONV_1x1(concat, c2f_m4, WP(WR_model_4_cv2_conv_Conv_W), WP(WR_model_4_cv2_conv_Conv_B), 128u, 36u, 64u, 64u, 1u);
    }

    /* === New for M5 === */

    /* SCDown m.5: cv1 64->128 (1x1+SiLU), cv2 128->128 (depthwise 3x3 s=2, no act) */
    CONV_1x1(c2f_m4, c5_post, WP(WR_model_5_cv1_conv_Conv_W), WP(WR_model_5_cv1_conv_Conv_B), 64u, 36u, 64u, 128u, 1u);
    {
        float *m5_cv2 = (float *)(base + SCR_M5_CV2_OUT);
        CONV_DW_MH(c5_post, m5_cv2, WP(WR_model_5_cv2_conv_Conv_W), WP(WR_model_5_cv2_conv_Conv_B),
                       128u, 36u, 64u, 18u, 32u, 3u, 3u, 2u, 2u, 1u, 1u, 0u);

        /* C2f model.6 (2 bottlenecks).  Input = m5_cv2 [128,18,32] */
        const uint32_t HW = 18u * 32u;
        float *concat = (float *)(base + SCR_M6_CONCAT);
        float *y1     = concat + 64u * HW;
        float *m0_out = concat + 128u * HW;
        float *m1_out = concat + 192u * HW;
        float *m0_cv1 = (float *)(base + SCR_M6_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M6_M0_CV2);
        float *m1_cv1 = (float *)(base + SCR_M6_M1_CV1);
        float *m1_cv2 = (float *)(base + SCR_M6_M1_CV2);
        CONV_1x1(m5_cv2, concat, WP(WR_model_6_cv1_conv_Conv_W), WP(WR_model_6_cv1_conv_Conv_B), 128u, 18u, 32u, 128u, 1u);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_6_m_0_cv1_conv_Conv_W), WP(WR_model_6_m_0_cv1_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_6_m_0_cv2_conv_Conv_W), WP(WR_model_6_m_0_cv2_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        MH_ADD(m0_out, y1, m0_cv2, 64u * HW);
        CONV_3x3_P1_VPU(m0_out, m1_cv1, WP(WR_model_6_m_1_cv1_conv_Conv_W), WP(WR_model_6_m_1_cv1_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        CONV_3x3_P1_VPU(m1_cv1, m1_cv2, WP(WR_model_6_m_1_cv2_conv_Conv_W), WP(WR_model_6_m_1_cv2_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        MH_ADD(m1_out, m0_out, m1_cv2, 64u * HW);
        CONV_1x1(concat, c2f_m6, WP(WR_model_6_cv2_conv_Conv_W), WP(WR_model_6_cv2_conv_Conv_B), 256u, 18u, 32u, 128u, 1u);
    }

    /* SCDown m.7: cv1 128->256, cv2 256->256 depthwise s=2 */
    CONV_1x1(c2f_m6, c7_post, WP(WR_model_7_cv1_conv_Conv_W), WP(WR_model_7_cv1_conv_Conv_B), 128u, 18u, 32u, 256u, 1u);
    {
        float *m7_cv2 = (float *)(base + SCR_M7_CV2_OUT);
        CONV_DW_MH(c7_post, m7_cv2, WP(WR_model_7_cv2_conv_Conv_W), WP(WR_model_7_cv2_conv_Conv_B),
                       256u, 18u, 32u, 9u, 16u, 3u, 3u, 2u, 2u, 1u, 1u, 0u);

        /* C2f model.8 (1 bottleneck).  Input = m7_cv2 [256, 9, 16] */
        const uint32_t HW = 9u * 16u;
        float *concat = (float *)(base + SCR_M8_CONCAT);
        float *y1     = concat + 128u * HW;
        float *m0_out = concat + 256u * HW;
        float *m0_cv1 = (float *)(base + SCR_M8_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M8_M0_CV2);
        CONV_1x1(m7_cv2, concat, WP(WR_model_8_cv1_conv_Conv_W), WP(WR_model_8_cv1_conv_Conv_B), 256u, 9u, 16u, 256u, 1u);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_8_m_0_cv1_conv_Conv_W), WP(WR_model_8_m_0_cv1_conv_Conv_B), 128u, 9u, 16u, 128u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_8_m_0_cv2_conv_Conv_W), WP(WR_model_8_m_0_cv2_conv_Conv_B), 128u, 9u, 16u, 128u, 1u);
        MH_ADD(m0_out, y1, m0_cv2, 128u * HW);
        CONV_1x1(concat, c2f_m8, WP(WR_model_8_cv2_conv_Conv_W), WP(WR_model_8_cv2_conv_Conv_B), 384u, 9u, 16u, 256u, 1u);
    }

    /* SPPF m.9: cv1 256->128, then 3 sequential MaxPool 5x5 s=1 p=2, concat 4 -> cv2 512->256 */
    {
        float *m9_cv1 = (float *)(base + SCR_M9_CV1_OUT);
        float *m9_mp1 = (float *)(base + SCR_M9_MP1);
        float *m9_mp2 = (float *)(base + SCR_M9_MP2);
        float *m9_mp3 = (float *)(base + SCR_M9_MP3);
        float *concat = (float *)(base + SCR_M9_CONCAT);
        const uint32_t HW = 9u * 16u;

        CONV_1x1(c2f_m8, m9_cv1, WP(WR_model_9_cv1_conv_Conv_W), WP(WR_model_9_cv1_conv_Conv_B), 256u, 9u, 16u, 128u, 1u);
        H0_RUN(maxpool_fp32(m9_cv1,m9_mp1,128u,9u,16u,9u,16u,5u,5u,1u,1u,2u,2u), m9_mp1, (128u)*(9u)*(16u)*sizeof(float));
        H0_RUN(maxpool_fp32(m9_mp1,m9_mp2,128u,9u,16u,9u,16u,5u,5u,1u,1u,2u,2u), m9_mp2, (128u)*(9u)*(16u)*sizeof(float));
        H0_RUN(maxpool_fp32(m9_mp2,m9_mp3,128u,9u,16u,9u,16u,5u,5u,1u,1u,2u,2u), m9_mp3, (128u)*(9u)*(16u)*sizeof(float));

        /* concat [m9_cv1, mp1, mp2, mp3] = 512 channels at 9x16 */
        MH_CONCAT4(concat, m9_cv1, m9_mp1, m9_mp2, m9_mp3, 128u * HW);

        CONV_1x1(concat, sppf, WP(WR_model_9_cv2_conv_Conv_W), WP(WR_model_9_cv2_conv_Conv_B), 512u, 9u, 16u, 256u, 1u);
    }

    /* === PSA model.10 ===
     * Input: sppf (256, 9, 16).  Output: psa_m10_out (256, 9, 16).
     */
    float *psa_out = (float *)(base + PSA_M10_OUT_OFFSET);
    {
        const uint32_t H = 9u, W = 16u, HW = H * W;          /* 144 */
        const uint32_t NHEAD = 2u, KEY_DIM = 32u, HEAD_DIM = 64u;
        const float SCALE = 0.1767766922712326f;             /* 1/sqrt(32) */

        float *cv1_out = (float *)(base + SCR_PSA_CV1_OUT);   /* [256, 9,16] */
        float *y0      = cv1_out;                              /* alias chs 0..128 */
        float *y1_src  = cv1_out + 128u * HW;                  /* alias chs 128..256 */
        float *y1      = (float *)(base + SCR_PSA_Y1_BUF);     /* mutable copy */

        float *qkv     = (float *)(base + SCR_PSA_QKV);        /* [256, 9,16] */
        float *Q       = (float *)(base + SCR_PSA_Q);          /* [2, 32, 144] */
        float *K       = (float *)(base + SCR_PSA_K);
        float *V       = (float *)(base + SCR_PSA_V);          /* [2, 64, 144] */
        float *QT      = (float *)(base + SCR_PSA_QT);         /* [2, 144, 32] */
        float *logits  = (float *)(base + SCR_PSA_LOGITS);     /* [2, 144, 144] */
        float *sm_T    = (float *)(base + SCR_PSA_SM_T);       /* [2, 144, 144] */
        float *V_resh  = (float *)(base + SCR_PSA_V_RESH);     /* [128, 9,16] */
        float *pe      = (float *)(base + SCR_PSA_PE);
        float *attn_o  = (float *)(base + SCR_PSA_ATTN_OUT);   /* [128, 9,16] */
        float *proj_o  = (float *)(base + SCR_PSA_PROJ);
        float *ffn0    = (float *)(base + SCR_PSA_FFN0);       /* [256, 9,16] */
        float *ffn1    = (float *)(base + SCR_PSA_FFN1);

        /* cv1: 256 -> 256, 1x1 + SiLU */
        CONV_1x1(sppf, cv1_out, WP(WR_model_10_cv1_conv_Conv_W), WP(WR_model_10_cv1_conv_Conv_B), 256u, 9u, 16u, 256u, 1u);

        /* Copy y1 into mutable buffer (will receive residuals). */
        if (is_h0) {
            for (uint32_t i = 0; i < 128u * HW; i++) y1[i] = y1_src[i];
            evict((const void *)y1, 128u * HW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        /* qkv: 128 -> 256, 1x1 (no act).  Input = y1 (mutable copy). */
        CONV_1x1(y1, qkv, WP(WR_model_10_attn_qkv_conv_Conv_W), WP(WR_model_10_attn_qkv_conv_Conv_B), 128u, 9u, 16u, 256u, 0u);

        /* Reshape qkv -> Q/K/V (single hart). */
        if (is_h0) {
            for (uint32_t h = 0; h < NHEAD; h++) {
                const uint32_t base_c = h * 128u;
                for (uint32_t c = 0; c < KEY_DIM; c++) {
                    const float *src = qkv + (base_c + c) * HW;
                    float *dst = Q + (h * KEY_DIM + c) * HW;
                    for (uint32_t s = 0; s < HW; s++) dst[s] = src[s];
                }
                for (uint32_t c = 0; c < KEY_DIM; c++) {
                    const float *src = qkv + (base_c + KEY_DIM + c) * HW;
                    float *dst = K + (h * KEY_DIM + c) * HW;
                    for (uint32_t s = 0; s < HW; s++) dst[s] = src[s];
                }
                for (uint32_t c = 0; c < HEAD_DIM; c++) {
                    const float *src = qkv + (base_c + 2u*KEY_DIM + c) * HW;
                    float *dst = V + (h * HEAD_DIM + c) * HW;
                    for (uint32_t s = 0; s < HW; s++) dst[s] = src[s];
                }
            }
            for (uint32_t h = 0; h < NHEAD; h++) {
                for (uint32_t c = 0; c < HEAD_DIM; c++) {
                    const float *src = V + (h * HEAD_DIM + c) * HW;
                    float *dst = V_resh + (h * HEAD_DIM + c) * HW;
                    for (uint32_t s = 0; s < HW; s++) dst[s] = src[s];
                }
            }
            evict((const void *)Q,      NHEAD * KEY_DIM * HW * sizeof(float));
            evict((const void *)K,      NHEAD * KEY_DIM * HW * sizeof(float));
            evict((const void *)V,      NHEAD * HEAD_DIM * HW * sizeof(float));
            evict((const void *)V_resh, 128u * HW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        /* pe = depthwise Conv3x3 pad1 on V_resh (no activation). */
        CONV_DW3x3_S1_P1_VPU(V_resh, pe, WP(WR_model_10_attn_pe_conv_Conv_W), WP(WR_model_10_attn_pe_conv_Conv_B), 128u, 9u, 16u, 0u);

        /* Attention scoring + softmax + value matmul + pe-add (single hart). */
        if (is_h0) {
            for (uint32_t h = 0; h < NHEAD; h++) {
                transpose_2d(Q + h * KEY_DIM * HW, QT + h * HW * KEY_DIM, KEY_DIM, HW);
            }
            for (uint32_t h = 0; h < NHEAD; h++) {
                matmul_2d_fp32(QT + h * HW * KEY_DIM, K + h * KEY_DIM * HW,
                               logits + h * HW * HW, HW, KEY_DIM, HW);
            }
            for (uint32_t i = 0; i < NHEAD * HW * HW; i++) logits[i] *= SCALE;
            softmax_rows(logits, NHEAD * HW, HW);
            for (uint32_t h = 0; h < NHEAD; h++) {
                transpose_2d(logits + h * HW * HW, sm_T + h * HW * HW, HW, HW);
            }
            for (uint32_t h = 0; h < NHEAD; h++) {
                matmul_2d_fp32(V + h * HEAD_DIM * HW, sm_T + h * HW * HW,
                               attn_o + h * HEAD_DIM * HW, HEAD_DIM, HW, HW);
            }
            for (uint32_t i = 0; i < 128u * HW; i++) attn_o[i] += pe[i];
            evict((const void *)attn_o, 128u * HW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        /* proj: 128 -> 128, 1x1 (no activation). */
        CONV_1x1(attn_o, proj_o, WP(WR_model_10_attn_proj_conv_Conv_W), WP(WR_model_10_attn_proj_conv_Conv_B), 128u, 9u, 16u, 128u, 0u);

        /* y1 += proj_o (residual) */
        if (is_h0) {
            for (uint32_t i = 0; i < 128u * HW; i++) y1[i] += proj_o[i];
            evict((const void *)y1, 128u * HW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        /* ffn0: 128 -> 256, 1x1 + SiLU */
        CONV_1x1(y1, ffn0, WP(WR_model_10_ffn_ffn_0_conv_Conv_W), WP(WR_model_10_ffn_ffn_0_conv_Conv_B), 128u, 9u, 16u, 256u, 1u);
        /* ffn1: 256 -> 128, 1x1 (no act) */
        CONV_1x1(ffn0, ffn1, WP(WR_model_10_ffn_ffn_1_conv_Conv_W), WP(WR_model_10_ffn_ffn_1_conv_Conv_B), 256u, 9u, 16u, 128u, 0u);

        /* y1 += ffn1; Concat [y0, y1] into cv1_out. */
        if (is_h0) {
            for (uint32_t i = 0; i < 128u * HW; i++) y1[i] += ffn1[i];
            for (uint32_t i = 0; i < 128u * HW; i++) y1_src[i] = y1[i];
            evict((const void *)y1_src, 128u * HW * sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        /* cv2: 256 -> 256, 1x1 + SiLU.  Output: psa_out @ 0x1300000. */
        CONV_1x1(cv1_out, psa_out, WP(WR_model_10_cv2_conv_Conv_W), WP(WR_model_10_cv2_conv_Conv_B), 256u, 9u, 16u, 256u, 1u);
    }

    /* === FPN going up (m.11..m.16) === */
    float *p3_out = (float *)(base + HEAD_P3_IN_OFFSET);
    float *p4_out = (float *)(base + HEAD_P4_IN_OFFSET);
    float *p5_out = (float *)(base + HEAD_P5_IN_OFFSET);
    float *m13_cv2_out = (float *)(base + SCR_M13_CV2_OUT);

    /* m.11: nearest-2x upsample of psa_out [256,9,16] -> [256,18,32]. */
    {
        float *up = (float *)(base + SCR_M11_UP);
        H0_RUN(upsample_nearest_2x(psa_out,up,256u,9u,16u), up, (256u)*(9u*2u)*(16u*2u)*sizeof(float));

        /* m.12: concat [up, c2f_m6] axis=1 -> [384,18,32] */
        float *cat = (float *)(base + SCR_M12_CONCAT);
        H0_RUN(concat_c_chw(up,256u,c2f_m6,128u,cat,18u,32u), cat, ((256u)+(128u))*(18u)*(32u)*sizeof(float));

        /* m.13: C2f without shortcut.
         *   cv1: 384 -> 128, 1x1+SiLU; split -> y0(64)+y1(64)
         *   m.0: y1 -> cv1 (3x3+SiLU) -> cv2 (3x3+SiLU)  [no residual]
         *   concat [y0, y1, m0_cv2_out] = 192 ch
         *   cv2: 192 -> 128, 1x1+SiLU
         */
        float *cv1 = (float *)(base + SCR_M13_CV1);
        float *y1  = cv1 + 64u * 18u * 32u;
        float *m0_out_slot = cv1 + 128u * 18u * 32u;   /* 3rd 64-ch slot? Need bigger */
        (void)m0_out_slot;

        CONV_1x1(cat, cv1, WP(WR_model_13_cv1_conv_Conv_W), WP(WR_model_13_cv1_conv_Conv_B), 384u, 18u, 32u, 128u, 1u);

        float *m0_cv1 = (float *)(base + SCR_M13_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M13_M0_CV2);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_13_m_0_cv1_conv_Conv_W), WP(WR_model_13_m_0_cv1_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_13_m_0_cv2_conv_Conv_W), WP(WR_model_13_m_0_cv2_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);

        /* concat [y0, y1, m0_cv2] - 192 channels.  Use a fresh tmp buf
         * (overwrite SCR_M12_CONCAT, no longer needed). */
        float *cat192 = (float *)(base + SCR_M12_CONCAT);   /* 384*18*32*4 = 0xD8000 -> 192 ch fits */
        const uint32_t HW = 18u*32u;
        MH_CONCAT3(cat192, cv1, y1, m0_cv2, 64u*HW);

        CONV_1x1(cat192, m13_cv2_out, WP(WR_model_13_cv2_conv_Conv_W), WP(WR_model_13_cv2_conv_Conv_B), 192u, 18u, 32u, 128u, 1u);
    }

    /* m.14: nearest-2x upsample of m13 -> [128,36,64]; m.15: concat with c2f_m4 [64,36,64] = [192,36,64]. */
    {
        float *up = (float *)(base + SCR_M14_UP);
        H0_RUN(upsample_nearest_2x(m13_cv2_out,up,128u,18u,32u), up, (128u)*(18u*2u)*(32u*2u)*sizeof(float));
        float *cat = (float *)(base + SCR_M15_CONCAT);
        H0_RUN(concat_c_chw(up,128u,c2f_m4,64u,cat,36u,64u), cat, ((128u)+(64u))*(36u)*(64u)*sizeof(float));

        /* m.16: C2f without shortcut.  cv1: 192 -> 64; split into y0(32)+y1(32);
         * m.0: 32 -> 32, 32 -> 32; concat [y0,y1,m0_cv2] = 96; cv2: 96 -> 64. */
        float *cv1 = (float *)(base + SCR_M16_CV1);
        float *y1 = cv1 + 32u * 36u * 64u;

        CONV_1x1(cat, cv1, WP(WR_model_16_cv1_conv_Conv_W), WP(WR_model_16_cv1_conv_Conv_B), 192u, 36u, 64u, 64u, 1u);

        float *m0_cv1 = (float *)(base + SCR_M16_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M16_M0_CV2);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_16_m_0_cv1_conv_Conv_W), WP(WR_model_16_m_0_cv1_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_16_m_0_cv2_conv_Conv_W), WP(WR_model_16_m_0_cv2_conv_Conv_B), 32u, 36u, 64u, 32u, 1u);

        /* concat [y0, y1, m0_cv2] = 96 channels at 36x64.  Reuse SCR_M15_CONCAT (192 ch buffer). */
        float *cat96 = (float *)(base + SCR_M15_CONCAT);
        const uint32_t HW = 36u*64u;
        MH_CONCAT3(cat96, cv1, y1, m0_cv2, 32u*HW);

        CONV_1x1(cat96, p3_out, WP(WR_model_16_cv2_conv_Conv_W), WP(WR_model_16_cv2_conv_Conv_B), 96u, 36u, 64u, 64u, 1u);
    }

    /* m.17: down conv 64->64 3x3 s=2 + SiLU on p3_out -> [64,18,32]; m.18: concat with m13_cv2_out = [192,18,32]. */
    {
        float *down = (float *)(base + SCR_M17_DOWN);
        CONV_3x3_S2_P1_VPU(p3_out, down, WP(WR_model_17_conv_Conv_W), WP(WR_model_17_conv_Conv_B),
                           64u, 36u, 64u, 64u, 18u, 32u, 1u);
        float *cat = (float *)(base + SCR_M18_CONCAT);
        H0_RUN(concat_c_chw(down,64u,m13_cv2_out,128u,cat,18u,32u), cat, ((64u)+(128u))*(18u)*(32u)*sizeof(float));

        /* m.19: C2f w/o shortcut.  cv1 192->128; split 64+64; m.0 64->64, 64->64; concat 192->cv2 128. */
        float *cv1 = (float *)(base + SCR_M19_CV1);
        float *y1 = cv1 + 64u * 18u * 32u;

        CONV_1x1(cat, cv1, WP(WR_model_19_cv1_conv_Conv_W), WP(WR_model_19_cv1_conv_Conv_B), 192u, 18u, 32u, 128u, 1u);

        float *m0_cv1 = (float *)(base + SCR_M19_M0_CV1);
        float *m0_cv2 = (float *)(base + SCR_M19_M0_CV2);
        CONV_3x3_P1_VPU(y1, m0_cv1, WP(WR_model_19_m_0_cv1_conv_Conv_W), WP(WR_model_19_m_0_cv1_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
        CONV_3x3_P1_VPU(m0_cv1, m0_cv2, WP(WR_model_19_m_0_cv2_conv_Conv_W), WP(WR_model_19_m_0_cv2_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);

        const uint32_t HW = 18u*32u;
        float *cat192 = (float *)(base + SCR_M18_CONCAT);   /* 192 ch buffer */
        MH_CONCAT3(cat192, cv1, y1, m0_cv2, 64u*HW);

        CONV_1x1(cat192, p4_out, WP(WR_model_19_cv2_conv_Conv_W), WP(WR_model_19_cv2_conv_Conv_B), 192u, 18u, 32u, 128u, 1u);
    }

    /* m.20: SCDown 128->128.  cv1 1x1+SiLU, cv2 DW 3x3 s=2 no act. */
    /* m.21: concat with psa_out = [384,9,16].  m.22: C2fCIB. */
    {
        float *cv1 = (float *)(base + SCR_M20_CV1);
        float *down = (float *)(base + SCR_M20_DOWN);
        CONV_1x1(p4_out, cv1, WP(WR_model_20_cv1_conv_Conv_W), WP(WR_model_20_cv1_conv_Conv_B), 128u, 18u, 32u, 128u, 1u);
        CONV_DW_MH(cv1, down, WP(WR_model_20_cv2_conv_Conv_W), WP(WR_model_20_cv2_conv_Conv_B),
                       128u, 18u, 32u, 9u, 16u, 3u, 3u, 2u, 2u, 1u, 1u, 0u);

        float *cat = (float *)(base + SCR_M21_CONCAT);
        H0_RUN(concat_c_chw(down,128u,psa_out,256u,cat,9u,16u), cat, ((128u)+(256u))*(9u)*(16u)*sizeof(float));

        /* m.22: C2fCIB block.
         *   cv1 384->256 (1x1+SiLU); split 128+128 (y0, y1)
         *   m.0 (CIB): y1 -> DW3x3 -> 1x1 -> DW7x7 -> 1x1 -> DW3x3 -> +y1 (residual).  All convs have SiLU.
         *   concat [y0, y1, m.0_out] = 384 ch
         *   cv2 384->256 (1x1+SiLU)
         */
        float *m22_cv1 = (float *)(base + SCR_M22_CV1);
        const uint32_t HW = 9u*16u;
        CONV_1x1(cat, m22_cv1, WP(WR_model_22_cv1_conv_Conv_W), WP(WR_model_22_cv1_conv_Conv_B), 384u, 9u, 16u, 256u, 1u);
        float *y0 = m22_cv1;
        float *y1 = m22_cv1 + 128u * HW;

        /* CIB pipeline */
        float *t0 = (float *)(base + SCR_M22_T0);   /* DW3x3 of y1 (128 ch, no spatial change) */
        CONV_DW3x3_S1_P1_VPU(y1, t0, WP(WR_model_22_m_0_cv1_cv1_0_conv_Conv_W), WP(WR_model_22_m_0_cv1_cv1_0_conv_Conv_B), 128u, 9u, 16u, 1u);
        float *t1 = (float *)(base + SCR_M22_T1);
        CONV_1x1(t0, t1, WP(WR_model_22_m_0_cv1_cv1_1_conv_Conv_W), WP(WR_model_22_m_0_cv1_cv1_1_conv_Conv_B), 128u, 9u, 16u, 256u, 1u);
        float *t2 = (float *)(base + SCR_M22_T2);
        CONV_DW_MH(t1, t2, WP(WR_model_22_m_0_cv1_cv1_2_conv_Conv_W), WP(WR_model_22_m_0_cv1_cv1_2_conv_Conv_B),
                       256u, 9u, 16u, 9u, 16u, 7u, 7u, 1u, 1u, 3u, 3u, 1u);
        float *t3 = (float *)(base + SCR_M22_T3);
        CONV_1x1(t2, t3, WP(WR_model_22_m_0_cv1_cv1_3_conv_Conv_W), WP(WR_model_22_m_0_cv1_cv1_3_conv_Conv_B), 256u, 9u, 16u, 128u, 1u);
        float *t4 = (float *)(base + SCR_M22_T4);
        CONV_DW3x3_S1_P1_VPU(t3, t4, WP(WR_model_22_m_0_cv1_cv1_4_conv_Conv_W), WP(WR_model_22_m_0_cv1_cv1_4_conv_Conv_B), 128u, 9u, 16u, 1u);

        /* Residual + concat (single hart). */
        float *cat384 = (float *)(base + SCR_M21_CONCAT);
        if (is_h0) {
            for (uint32_t i = 0; i < 128u*HW; i++) t4[i] = y1[i] + t4[i];
            for (uint32_t i = 0; i < 128u*HW; i++) cat384[0*128u*HW + i] = y0[i];
            for (uint32_t i = 0; i < 128u*HW; i++) cat384[1*128u*HW + i] = y1[i];
            for (uint32_t i = 0; i < 128u*HW; i++) cat384[2*128u*HW + i] = t4[i];
            evict((const void *)cat384, 384u*HW*sizeof(float));
            WAIT_CACHEOPS; FENCE;
        }
        MH_BARRIER();

        CONV_1x1(cat384, p5_out, WP(WR_model_22_cv2_conv_Conv_W), WP(WR_model_22_cv2_conv_Conv_B), 384u, 9u, 16u, 256u, 1u);
    }

    /* Evict all dump taps */
    EVICT_AND_FENCE(c0,      16u * 144u * 256u * sizeof(float));
    EVICT_AND_FENCE(c1,      32u *  72u * 128u * sizeof(float));
    EVICT_AND_FENCE(c2f_m2,  32u *  72u * 128u * sizeof(float));
    EVICT_AND_FENCE(c3,      64u *  36u *  64u * sizeof(float));
    EVICT_AND_FENCE(c2f_m4,  64u *  36u *  64u * sizeof(float));
    EVICT_AND_FENCE(c5_post, 128u * 36u *  64u * sizeof(float));
    EVICT_AND_FENCE(c2f_m6,  128u * 18u *  32u * sizeof(float));
    EVICT_AND_FENCE(c7_post, 256u * 18u *  32u * sizeof(float));
    EVICT_AND_FENCE(c2f_m8,  256u *  9u *  16u * sizeof(float));
    EVICT_AND_FENCE(sppf,    256u *  9u *  16u * sizeof(float));
    EVICT_AND_FENCE(psa_out, 256u *  9u *  16u * sizeof(float));
    EVICT_AND_FENCE(p3_out,   64u * 36u *  64u * sizeof(float));
    EVICT_AND_FENCE(p4_out,  128u * 18u *  32u * sizeof(float));
    EVICT_AND_FENCE(p5_out,  256u *  9u *  16u * sizeof(float));

    /* === Detection heads (model.23) ===
     * For each scale k in {0=P3, 1=P4, 2=P5}:
     *   reg branch (cv2.k.0 -> cv2.k.1 -> cv2.k.2) - 64 ch DFL logits, no act on .2
     *   cls branch (cv3.k.0.0 DW3x3 -> cv3.k.0.1 1x1 -> cv3.k.1.0 DW3x3 -> cv3.k.1.1 1x1 -> cv3.k.2 1x1) - 80 ch
     */
    float *reg0 = (float *)(base + REG_LOGITS_0_OFFSET);
    float *cls0 = (float *)(base + CLS_LOGITS_0_OFFSET);
    float *reg1 = (float *)(base + REG_LOGITS_1_OFFSET);
    float *cls1 = (float *)(base + CLS_LOGITS_1_OFFSET);
    float *reg2 = (float *)(base + REG_LOGITS_2_OFFSET);
    float *cls2 = (float *)(base + CLS_LOGITS_2_OFFSET);

    float *ta = (float *)(base + SCR_HEAD_A);
    float *tb = (float *)(base + SCR_HEAD_B);
    float *tc = (float *)(base + SCR_HEAD_C);
    float *td = (float *)(base + SCR_HEAD_D);

    /* === Scale 0 (P3, 36x64, IN_C=64) === */
    /* reg.0: 3x3 64->64 + SiLU -> ta */
    CONV_3x3_P1_VPU(p3_out, ta, WP(WR_model_23_cv2_0_cv2_0_0_conv_Conv_W), WP(WR_model_23_cv2_0_cv2_0_0_conv_Conv_B), 64u, 36u, 64u, 64u, 1u);
    /* reg.1: 3x3 64->64 + SiLU -> tb */
    CONV_3x3_P1_VPU(ta, tb, WP(WR_model_23_cv2_0_cv2_0_1_conv_Conv_W), WP(WR_model_23_cv2_0_cv2_0_1_conv_Conv_B), 64u, 36u, 64u, 64u, 1u);
    /* reg.2: 1x1 64->64, no act -> reg0 */
    CONV_1x1(tb, reg0, WP(WR_model_23_cv2_0_cv2_0_2_Conv_W), WP(WR_model_23_cv2_0_cv2_0_2_Conv_B), 64u, 36u, 64u, 64u, 0u);

    /* cls.0.0: DW3x3 64->64 + SiLU -> ta */
    CONV_DW3x3_S1_P1_VPU(p3_out, ta, WP(WR_model_23_cv3_0_cv3_0_0_cv3_0_0_0_conv_Conv_W), WP(WR_model_23_cv3_0_cv3_0_0_cv3_0_0_0_conv_Conv_B), 64u, 36u, 64u, 1u);
    /* cls.0.1: 1x1 64->80 + SiLU -> tb */
    CONV_1x1(ta, tb, WP(WR_model_23_cv3_0_cv3_0_0_cv3_0_0_1_conv_Conv_W), WP(WR_model_23_cv3_0_cv3_0_0_cv3_0_0_1_conv_Conv_B), 64u, 36u, 64u, 80u, 1u);
    /* cls.1.0: DW3x3 80->80 + SiLU -> tc */
    CONV_DW3x3_S1_P1_VPU(tb, tc, WP(WR_model_23_cv3_0_cv3_0_1_cv3_0_1_0_conv_Conv_W), WP(WR_model_23_cv3_0_cv3_0_1_cv3_0_1_0_conv_Conv_B), 80u, 36u, 64u, 1u);
    /* cls.1.1: 1x1 80->80 + SiLU -> td */
    CONV_1x1(tc, td, WP(WR_model_23_cv3_0_cv3_0_1_cv3_0_1_1_conv_Conv_W), WP(WR_model_23_cv3_0_cv3_0_1_cv3_0_1_1_conv_Conv_B), 80u, 36u, 64u, 80u, 1u);
    /* cls.2: 1x1 80->80, no act -> cls0 */
    CONV_1x1(td, cls0, WP(WR_model_23_cv3_0_cv3_0_2_Conv_W), WP(WR_model_23_cv3_0_cv3_0_2_Conv_B), 80u, 36u, 64u, 80u, 0u);

    /* === Scale 1 (P4, 18x32, IN_C=128) === */
    CONV_3x3_P1_VPU(p4_out, ta, WP(WR_model_23_cv2_1_cv2_1_0_conv_Conv_W), WP(WR_model_23_cv2_1_cv2_1_0_conv_Conv_B), 128u, 18u, 32u, 64u, 1u);
    CONV_3x3_P1_VPU(ta, tb, WP(WR_model_23_cv2_1_cv2_1_1_conv_Conv_W), WP(WR_model_23_cv2_1_cv2_1_1_conv_Conv_B), 64u, 18u, 32u, 64u, 1u);
    CONV_1x1(tb, reg1, WP(WR_model_23_cv2_1_cv2_1_2_Conv_W), WP(WR_model_23_cv2_1_cv2_1_2_Conv_B), 64u, 18u, 32u, 64u, 0u);

    CONV_DW3x3_S1_P1_VPU(p4_out, ta, WP(WR_model_23_cv3_1_cv3_1_0_cv3_1_0_0_conv_Conv_W), WP(WR_model_23_cv3_1_cv3_1_0_cv3_1_0_0_conv_Conv_B), 128u, 18u, 32u, 1u);
    CONV_1x1(ta, tb, WP(WR_model_23_cv3_1_cv3_1_0_cv3_1_0_1_conv_Conv_W), WP(WR_model_23_cv3_1_cv3_1_0_cv3_1_0_1_conv_Conv_B), 128u, 18u, 32u, 80u, 1u);
    CONV_DW3x3_S1_P1_VPU(tb, tc, WP(WR_model_23_cv3_1_cv3_1_1_cv3_1_1_0_conv_Conv_W), WP(WR_model_23_cv3_1_cv3_1_1_cv3_1_1_0_conv_Conv_B), 80u, 18u, 32u, 1u);
    CONV_1x1(tc, td, WP(WR_model_23_cv3_1_cv3_1_1_cv3_1_1_1_conv_Conv_W), WP(WR_model_23_cv3_1_cv3_1_1_cv3_1_1_1_conv_Conv_B), 80u, 18u, 32u, 80u, 1u);
    CONV_1x1(td, cls1, WP(WR_model_23_cv3_1_cv3_1_2_Conv_W), WP(WR_model_23_cv3_1_cv3_1_2_Conv_B), 80u, 18u, 32u, 80u, 0u);

    /* === Scale 2 (P5, 9x16, IN_C=256) === */
    CONV_3x3_P1_VPU(p5_out, ta, WP(WR_model_23_cv2_2_cv2_2_0_conv_Conv_W), WP(WR_model_23_cv2_2_cv2_2_0_conv_Conv_B), 256u, 9u, 16u, 64u, 1u);
    CONV_3x3_P1_VPU(ta, tb, WP(WR_model_23_cv2_2_cv2_2_1_conv_Conv_W), WP(WR_model_23_cv2_2_cv2_2_1_conv_Conv_B), 64u, 9u, 16u, 64u, 1u);
    CONV_1x1(tb, reg2, WP(WR_model_23_cv2_2_cv2_2_2_Conv_W), WP(WR_model_23_cv2_2_cv2_2_2_Conv_B), 64u, 9u, 16u, 64u, 0u);

    CONV_DW3x3_S1_P1_VPU(p5_out, ta, WP(WR_model_23_cv3_2_cv3_2_0_cv3_2_0_0_conv_Conv_W), WP(WR_model_23_cv3_2_cv3_2_0_cv3_2_0_0_conv_Conv_B), 256u, 9u, 16u, 1u);
    CONV_1x1(ta, tb, WP(WR_model_23_cv3_2_cv3_2_0_cv3_2_0_1_conv_Conv_W), WP(WR_model_23_cv3_2_cv3_2_0_cv3_2_0_1_conv_Conv_B), 256u, 9u, 16u, 80u, 1u);
    CONV_DW3x3_S1_P1_VPU(tb, tc, WP(WR_model_23_cv3_2_cv3_2_1_cv3_2_1_0_conv_Conv_W), WP(WR_model_23_cv3_2_cv3_2_1_cv3_2_1_0_conv_Conv_B), 80u, 9u, 16u, 1u);
    CONV_1x1(tc, td, WP(WR_model_23_cv3_2_cv3_2_1_cv3_2_1_1_conv_Conv_W), WP(WR_model_23_cv3_2_cv3_2_1_cv3_2_1_1_conv_Conv_B), 80u, 9u, 16u, 80u, 1u);
    CONV_1x1(td, cls2, WP(WR_model_23_cv3_2_cv3_2_2_Conv_W), WP(WR_model_23_cv3_2_cv3_2_2_Conv_B), 80u, 9u, 16u, 80u, 0u);

    EVICT_AND_FENCE(reg0, 64u * 36u * 64u * sizeof(float));
    EVICT_AND_FENCE(cls0, 80u * 36u * 64u * sizeof(float));
    EVICT_AND_FENCE(reg1, 64u * 18u * 32u * sizeof(float));
    EVICT_AND_FENCE(cls1, 80u * 18u * 32u * sizeof(float));
    EVICT_AND_FENCE(reg2, 64u *  9u * 16u * sizeof(float));
    EVICT_AND_FENCE(cls2, 80u *  9u * 16u * sizeof(float));

    /* === DFL decode + box decode + class sigmoid -> final [1,84,3024] === */
    float *final_out = (float *)(base + FINAL_OUT_OFFSET);
    const uint32_t HW0 = 36u * 64u;     /* 2304 */
    const uint32_t HW1 = 18u * 32u;     /* 576  */

    /* DFL + box decode + class sigmoid (single-hart hart 0).
     * The naive parallel-by-anchor version had a non-coherent-L1D race
     * around the scattered writes to final_out (84 disjoint regions per
     * hart), even with whole-buffer evicts.  Decode is only ~50 ms anyway. */
    if (is_h0) {
        for (uint32_t k = 0; k < 3u; k++) {
            const float *reg_in;
            const float *cls_in;
            uint32_t H, W;
            float stride;
            uint32_t anchor_off;
            if (k == 0u)      { reg_in = reg0; cls_in = cls0; H = 36u; W = 64u; stride = 8.0f;  anchor_off = 0u; }
            else if (k == 1u) { reg_in = reg1; cls_in = cls1; H = 18u; W = 32u; stride = 16.0f; anchor_off = HW0; }
            else              { reg_in = reg2; cls_in = cls2; H =  9u; W = 16u; stride = 32.0f; anchor_off = HW0 + HW1; }
            const uint32_t HW = H * W;

            for (uint32_t e = 0; e < 4u; e++) {
                for (uint32_t s = 0; s < HW; s++) {
                    float row[16];
                    float m = -3.4e38f;
                    for (uint32_t b = 0; b < 16u; b++) {
                        row[b] = reg_in[(e*16u + b) * HW + s];
                        if (row[b] > m) m = row[b];
                    }
                    float sumexp = 0.0f;
                    for (uint32_t b = 0; b < 16u; b++) { row[b] = my_expf(row[b] - m); sumexp += row[b]; }
                    const float inv = fast_recip(sumexp);
                    float ev = 0.0f;
                    for (uint32_t b = 0; b < 16u; b++) ev += row[b] * inv * (float)b;
                    tb[e * HW + s] = ev;
                }
            }

            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    const uint32_t s = h * W + w;
                    const float a_cx = (float)w + 0.5f;
                    const float a_cy = (float)h + 0.5f;
                    const float lt_x = tb[0u*HW + s];
                    const float lt_y = tb[1u*HW + s];
                    const float rb_x = tb[2u*HW + s];
                    const float rb_y = tb[3u*HW + s];
                    const float left   = (a_cx - lt_x) * stride;
                    const float top    = (a_cy - lt_y) * stride;
                    const float right  = (a_cx + rb_x) * stride;
                    const float bottom = (a_cy + rb_y) * stride;
                    const uint32_t a = anchor_off + s;
                    final_out[0u * 3024u + a] = (left + right) * 0.5f;
                    final_out[1u * 3024u + a] = (top + bottom) * 0.5f;
                    final_out[2u * 3024u + a] = right - left;
                    final_out[3u * 3024u + a] = bottom - top;
                }
            }

            for (uint32_t c = 0; c < 80u; c++) {
                for (uint32_t s = 0; s < HW; s++) {
                    const float v = cls_in[c * HW + s];
                    final_out[(4u + c) * 3024u + (anchor_off + s)] = fast_recip(1.0f + my_expf(-v));
                }
            }
        }
        evict((const void *)final_out, 84u * 3024u * sizeof(float));
        WAIT_CACHEOPS; FENCE;
    }
    MH_BARRIER();

    /* === STAGE 2: POSTPROCESS on silicon (threshold + class-aware NMS) ===
     * Read final[1, 84, 3024], output a small detection list at DETECTIONS_OFFSET:
     *   uint32 count N
     *   then N x { uint32 class_id; float score; float x1,y1,x2,y2; }
     */
    if (is_h0) {
        const float CONF_THRESH = 0.25f;
        const float IOU_THRESH  = 0.5f;
        /* Step 1: scan anchors, keep those with max-class-prob >= CONF_THRESH.
         * Build candidate list in scratch (use `tb`, plenty of space). */
        struct __attribute__((packed)) Cand {
            uint32_t class_id;
            float    score;
            float    x1, y1, x2, y2;
            uint8_t  alive;
            uint8_t  pad[3];
        };
        struct Cand *cands = (struct Cand *)tb;   /* up to 3024 candidates */
        uint32_t n_cands = 0;
        for (uint32_t a = 0; a < 3024u; a++) {
            float best_score = 0.0f;
            uint32_t best_cls = 0;
            for (uint32_t c = 0; c < 80u; c++) {
                const float p = final_out[(4u + c) * 3024u + a];
                if (p > best_score) { best_score = p; best_cls = c; }
            }
            if (best_score < CONF_THRESH) continue;
            const float cx = final_out[0u * 3024u + a];
            const float cy = final_out[1u * 3024u + a];
            const float bw = final_out[2u * 3024u + a];
            const float bh = final_out[3u * 3024u + a];
            cands[n_cands].class_id = best_cls;
            cands[n_cands].score    = best_score;
            cands[n_cands].x1 = cx - 0.5f * bw;
            cands[n_cands].y1 = cy - 0.5f * bh;
            cands[n_cands].x2 = cx + 0.5f * bw;
            cands[n_cands].y2 = cy + 0.5f * bh;
            cands[n_cands].alive = 1u;
            n_cands++;
        }

        /* Step 2: simple class-aware NMS (O(n^2), n ~ 100 at most). */
        for (uint32_t i = 0; i < n_cands; i++) {
            if (!cands[i].alive) continue;
            for (uint32_t j = i + 1; j < n_cands; j++) {
                if (!cands[j].alive) continue;
                if (cands[j].class_id != cands[i].class_id) continue;
                /* Pick the one with higher score; suppress the lower one if IoU > thresh. */
                const float xx1 = cands[i].x1 > cands[j].x1 ? cands[i].x1 : cands[j].x1;
                const float yy1 = cands[i].y1 > cands[j].y1 ? cands[i].y1 : cands[j].y1;
                const float xx2 = cands[i].x2 < cands[j].x2 ? cands[i].x2 : cands[j].x2;
                const float yy2 = cands[i].y2 < cands[j].y2 ? cands[i].y2 : cands[j].y2;
                const float iw = xx2 > xx1 ? xx2 - xx1 : 0.0f;
                const float ih = yy2 > yy1 ? yy2 - yy1 : 0.0f;
                const float inter = iw * ih;
                const float ai = (cands[i].x2 - cands[i].x1) * (cands[i].y2 - cands[i].y1);
                const float aj = (cands[j].x2 - cands[j].x1) * (cands[j].y2 - cands[j].y1);
                const float uni = ai + aj - inter + 1e-9f;
                const float iou = inter * fast_recip(uni);
                if (iou > IOU_THRESH) {
                    if (cands[i].score >= cands[j].score) cands[j].alive = 0u;
                    else                                  cands[i].alive = 0u;
                }
            }
        }

        /* Step 3: write detection list at DETECTIONS_OFFSET.
         *   header: uint32 N
         *   N x { uint32 class_id; float score; float x1,y1,x2,y2; } = 24 bytes */
        uint32_t *out_count = (uint32_t *)(base + DETECTIONS_OFFSET);
        struct __attribute__((packed)) DetOut {
            uint32_t class_id; float score; float x1, y1, x2, y2;
        } *out_dets = (struct DetOut *)(base + DETECTIONS_OFFSET + sizeof(uint32_t));

        uint32_t n_out = 0;
        /* Write in score order: simple selection of remaining survivors (no global sort). */
        while (n_out < MAX_DETECTIONS) {
            float best_score = 0.0f;
            int32_t best_idx = -1;
            for (uint32_t i = 0; i < n_cands; i++) {
                if (!cands[i].alive) continue;
                if (cands[i].score > best_score) {
                    best_score = cands[i].score;
                    best_idx   = (int32_t)i;
                }
            }
            if (best_idx < 0) break;
            out_dets[n_out].class_id = cands[best_idx].class_id;
            out_dets[n_out].score    = cands[best_idx].score;
            out_dets[n_out].x1       = cands[best_idx].x1;
            out_dets[n_out].y1       = cands[best_idx].y1;
            out_dets[n_out].x2       = cands[best_idx].x2;
            out_dets[n_out].y2       = cands[best_idx].y2;
            cands[best_idx].alive = 0u;
            n_out++;
        }
        *out_count = n_out;

        evict((const void *)out_count, sizeof(uint32_t) + n_out * sizeof(*out_dets));
        WAIT_CACHEOPS; FENCE;
    }
    MH_BARRIER();
    return 0;
}
