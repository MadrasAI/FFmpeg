/*
 * Copyright (c) 2010 Mans Rullgard
 * Copyright (c) 2014 James Yu <james.yu@linaro.org>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <arm_neon.h>

#include "config.h"

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#if   ARCH_AARCH64
#   include "libavutil/aarch64/cpu.h"
#elif ARCH_ARM
#   include "libavutil/arm/cpu.h"
#endif

#include "libavcodec/mpegvideo.h"
#include "libavcodec/mpegvideo_unquantize.h"
#include "libavcodec/mpegvideodata.h"

static void inline ff_dct_unquantize_h263_neon(int qscale, int qadd, int nCoeffs,
                                               int16_t *block)
{
    int16x8_t q0s16, q2s16, q3s16, q8s16, q10s16, q11s16, q13s16;
    int16x8_t q14s16, q15s16, qzs16;
    uint16x8_t q1u16, q9u16;

    qzs16 = vdupq_n_s16(0);

    q15s16 = vdupq_n_s16(qscale << 1);
    q14s16 = vdupq_n_s16(qadd);
    q13s16 = vnegq_s16(q14s16);

    if (nCoeffs > 4) {
        for (; nCoeffs > 8; nCoeffs -= 16, block += 16) {
            q0s16 = vld1q_s16(block);
            q3s16 = vreinterpretq_s16_u16(vcltq_s16(q0s16, qzs16));
            q8s16 = vld1q_s16(block + 8);
            q1u16 = vceqq_s16(q0s16, qzs16);
            q2s16 = vmulq_s16(q0s16, q15s16);
            q11s16 = vreinterpretq_s16_u16(vcltq_s16(q8s16, qzs16));
            q10s16 = vmulq_s16(q8s16, q15s16);
            q3s16 = vbslq_s16(vreinterpretq_u16_s16(q3s16), q13s16, q14s16);
            q11s16 = vbslq_s16(vreinterpretq_u16_s16(q11s16), q13s16, q14s16);
            q2s16 = vaddq_s16(q2s16, q3s16);
            q9u16 = vceqq_s16(q8s16, qzs16);
            q10s16 = vaddq_s16(q10s16, q11s16);
            q0s16 = vbslq_s16(q1u16, q0s16, q2s16);
            q8s16 = vbslq_s16(q9u16, q8s16, q10s16);
            vst1q_s16(block, q0s16);
            vst1q_s16(block + 8, q8s16);
        }
    }
    if (nCoeffs <= 0)
        return;

    q0s16 = vld1q_s16(block);
    q3s16 = vreinterpretq_s16_u16(vcltq_s16(q0s16, qzs16));
    q1u16 = vceqq_s16(q0s16, qzs16);
    q2s16 = vmulq_s16(q0s16, q15s16);
    q3s16 = vbslq_s16(vreinterpretq_u16_s16(q3s16), q13s16, q14s16);
    q2s16 = vaddq_s16(q2s16, q3s16);
    q0s16 = vbslq_s16(q1u16, q0s16, q2s16);
    vst1q_s16(block, q0s16);
}

static void dct_unquantize_h263_inter_neon(const MPVContext *s, int16_t *block,
                                           int n, int qscale)
{
    int nCoeffs = s->inter_scantable.raster_end[s->block_last_index[n]];
    int qadd    = (qscale - 1) | 1;

    ff_dct_unquantize_h263_neon(qscale, qadd, nCoeffs + 1, block);
}

static void dct_unquantize_h263_intra_neon(const MPVContext *s, int16_t *block,
                                           int n, int qscale)
{
    int qadd;
    int nCoeffs, blk0;

    if (!s->h263_aic) {
        if (n < 4)
            block[0] *= s->y_dc_scale;
        else
            block[0] *= s->c_dc_scale;
        qadd = (qscale - 1) | 1;
    } else {
        qadd = 0;
    }

    if (s->ac_pred) {
        nCoeffs = 63;
    } else {
        nCoeffs = s->intra_scantable.raster_end[s->block_last_index[n]];
        if (nCoeffs <= 0)
            return;
    }

    blk0 = block[0];

    ff_dct_unquantize_h263_neon(qscale, qadd, nCoeffs + 1, block);

    block[0] = blk0;
}

/*
 * MPEG-1/2 dequantization — sequential raster access matching x86 SSSE3/SSE2
 *
 * All four functions use raster_end[] for sequential block[] access instead of
 * following the permuted scan table.  This is valid because raster_end[k] is
 * defined as max(permutated[0..k]), so block[0..raster_end[last_index]] covers
 * every position that could be nonzero; positions not in the scan are zero and
 * stay zero through the SIMD path.
 *
 * vmulq_s16 truncates to 16 low bits, matching x86 pmullw.  The checkasm test
 * constrains inputs so the product fits in int16 (no overflow).
 *
 * Loop bound: n8 = (nCoeffs+8)&~7 rounds nCoeffs+1 up to the next multiple of
 * 8 and always stays ≤ 64, so vld1q_s16(&block[i]) with i≤56 is in-bounds for
 * a 64-element block.
 */

/* psignw equivalent: negate result where orig < 0, zero where orig == 0 */
static inline int16x8_t psignw_neon(int16x8_t result, int16x8_t orig)
{
    int16x8_t neg_m = vshrq_n_s16(orig, 15);           /* 0xFFFF if orig<0, else 0 */
    result = veorq_s16(vaddq_s16(result, neg_m), neg_m); /* conditional negate      */
    return vbicq_s16(result,
                     vreinterpretq_s16_u16(vceqq_s16(orig, vdupq_n_s16(0))));
}

static void dct_unquantize_mpeg1_intra_neon(const MPVContext *s,
                                             int16_t *block, int n, int qscale)
{
    int nCoeffs = s->intra_scantable.raster_end[s->block_last_index[n]];
    int block0  = block[0] * (n < 4 ? s->y_dc_scale : s->c_dc_scale);
    const uint16_t *qm = s->intra_matrix;
    int16x8_t q_s   = vdupq_n_s16((int16_t)qscale);
    int16x8_t ones  = vdupq_n_s16(1);
    int n8 = (nCoeffs + 8) & ~7;

    for (int i = 0; i < n8; i += 8) {
        int16x8_t orig   = vld1q_s16(&block[i]);
        int16x8_t q      = vmulq_s16(q_s, vreinterpretq_s16_u16(vld1q_u16(&qm[i])));
        int16x8_t abs_v  = vabsq_s16(orig);
        int16x8_t result = vshrq_n_s16(vmulq_s16(abs_v, q), 3);
        result = vorrq_s16(vsubq_s16(result, ones), ones);
        result = psignw_neon(result, orig);
        vst1q_s16(&block[i], result);
    }
    block[0] = block0;
}

static void dct_unquantize_mpeg1_inter_neon(const MPVContext *s,
                                             int16_t *block, int n, int qscale)
{
    int nCoeffs = s->intra_scantable.raster_end[s->block_last_index[n]];
    const uint16_t *qm = s->inter_matrix;
    int16x8_t q_s   = vdupq_n_s16((int16_t)qscale);
    int16x8_t ones  = vdupq_n_s16(1);
    int n8 = (nCoeffs + 8) & ~7;

    for (int i = 0; i < n8; i += 8) {
        int16x8_t orig   = vld1q_s16(&block[i]);
        int16x8_t q      = vmulq_s16(q_s, vreinterpretq_s16_u16(vld1q_u16(&qm[i])));
        int16x8_t abs_v  = vabsq_s16(orig);
        int16x8_t abs2p1 = vaddq_s16(vaddq_s16(abs_v, abs_v), ones);
        int16x8_t result = vshrq_n_s16(vmulq_s16(abs2p1, q), 4);
        result = vorrq_s16(vsubq_s16(result, ones), ones);
        result = psignw_neon(result, orig);
        vst1q_s16(&block[i], result);
    }
}

/*
 * MPEG-2 intra — bitexact version (with mismatch-control sum tracking).
 * Uses the signed-multiply + psrlw-12 bias trick from x86 SSE2 for the
 * per-coefficient computation.  The loop includes block[0] (DC), which is
 * saved before and restored after; the sum is corrected accordingly.
 */
static void dct_unquantize_mpeg2_intra_neon(const MPVContext *s,
                                             int16_t *block, int n, int qscale)
{
    if (s->q_scale_type) qscale = ff_mpeg2_non_linear_qscale[qscale];
    else                 qscale <<= 1;

    int nCoeffs = s->intra_scantable.raster_end[s->block_last_index[n]];
    int block0  = block[0] * (n < 4 ? s->y_dc_scale : s->c_dc_scale);
    const uint16_t *qm = s->intra_matrix;
    int16x8_t q_s   = vdupq_n_s16((int16_t)qscale);
    int16x8_t sum_v = vdupq_n_s16(0);
    int n8 = (nCoeffs + 8) & ~7;

    for (int i = 0; i < n8; i += 8) {
        int16x8_t orig   = vld1q_s16(&block[i]);
        int16x8_t q      = vmulq_s16(q_s, vreinterpretq_s16_u16(vld1q_u16(&qm[i])));
        /* bias: for orig in -2048..-1, (uint16)orig >> 12 == 15; for >=0, 0.
         * Adding before >>4 rounds towards zero (matching x86 psrlw 12 + psraw 4). */
        int16x8_t bias   = vreinterpretq_s16_u16(
                               vshrq_n_u16(vreinterpretq_u16_s16(orig), 12));
        int16x8_t result = vshrq_n_s16(vaddq_s16(vmulq_s16(orig, q), bias), 4);
        sum_v = vaddq_s16(sum_v, result);
        vst1q_s16(&block[i], result);
    }
    /* The loop computed mpeg2-formula(original block[0]) for DC.
     * Correct the sum: subtract that wrong DC and add the true scaled DC. */
    int dc_wrong = (int)(int16_t)block[0];
    block[0] = block0;
    /* sum = -1 + block0 + sum_of_AC_levels; only bit 0 matters for XOR */
    block[63] ^= ((unsigned)vaddvq_s16(sum_v) - (unsigned)dc_wrong +
                  (unsigned)block0 - 1u) & 1u;
}

/*
 * MPEG-2 inter — always bitexact (mismatch control is mandatory).
 * Follows x86 SSSE3: abs*2*q + q = (abs*2+1)*q, then unsigned >>5, psignw.
 */
static void dct_unquantize_mpeg2_inter_neon(const MPVContext *s,
                                             int16_t *block, int n, int qscale)
{
    if (s->q_scale_type) qscale = ff_mpeg2_non_linear_qscale[qscale];
    else                 qscale <<= 1;

    int nCoeffs = s->intra_scantable.raster_end[s->block_last_index[n]];
    const uint16_t *qm = s->inter_matrix;
    int16x8_t q_s   = vdupq_n_s16((int16_t)qscale);
    int16x8_t sum_v = vdupq_n_s16(0);
    int n8 = (nCoeffs + 8) & ~7;

    for (int i = 0; i < n8; i += 8) {
        int16x8_t orig   = vld1q_s16(&block[i]);
        int16x8_t q      = vmulq_s16(q_s, vreinterpretq_s16_u16(vld1q_u16(&qm[i])));
        int16x8_t abs_v  = vabsq_s16(orig);
        int16x8_t abs2   = vaddq_s16(abs_v, abs_v);
        /* (abs*2)*q + q == (abs*2+1)*q, matches x86 paddw xmm4 pattern */
        int16x8_t prod   = vaddq_s16(vmulq_s16(abs2, q), q);
        /* unsigned >>5: product is non-negative; psrlw $5 equivalent */
        int16x8_t result = vreinterpretq_s16_u16(
                               vshrq_n_u16(vreinterpretq_u16_s16(prod), 5));
        result = psignw_neon(result, orig);
        sum_v = vaddq_s16(sum_v, result);
        vst1q_s16(&block[i], result);
    }
    /* sum starts at -1; only bit 0 used for mismatch XOR */
    block[63] ^= ((unsigned)vaddvq_s16(sum_v) - 1u) & 1u;
}


av_cold void ff_mpv_unquantize_init_neon(MPVUnquantDSPContext *s, int bitexact)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        s->dct_unquantize_h263_intra  = dct_unquantize_h263_intra_neon;
        s->dct_unquantize_h263_inter  = dct_unquantize_h263_inter_neon;
        s->dct_unquantize_mpeg1_intra = dct_unquantize_mpeg1_intra_neon;
        s->dct_unquantize_mpeg1_inter = dct_unquantize_mpeg1_inter_neon;
        /* mpeg2_intra bitexact version includes mismatch-control sum tracking;
         * the non-bitexact C path does not XOR block[63], so only register when
         * bitexact to avoid changing non-bitexact decoder behaviour. */
        if (bitexact)
            s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_neon;
        s->dct_unquantize_mpeg2_inter = dct_unquantize_mpeg2_inter_neon;
    }
}
