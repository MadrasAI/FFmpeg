/*
 * ARM NEON optimised H.264 chroma functions
 * Copyright (c) 2008 Mans Rullgard <mans@mansr.com>
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
#include <stdint.h>

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/h264chroma.h"

#include "config.h"

void ff_put_h264_chroma_mc8_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);
void ff_put_h264_chroma_mc4_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);
void ff_put_h264_chroma_mc2_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);

void ff_avg_h264_chroma_mc8_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);
void ff_avg_h264_chroma_mc4_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);
void ff_avg_h264_chroma_mc2_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride,
                                 int h, int x, int y);

/*
 * H.264 chroma bilinear MC for 9-bit and 10-bit depth (NEON intrinsics).
 *
 * Algorithm (from h264chroma_template.c):
 *   A=(8-x)*(8-y), B=x*(8-y), C=(8-x)*y, D=x*y  (A+B+C+D=64)
 *   put: dst[i] = (A*src[i] + B*src[i+1] + C*src[s+i] + D*src[s+i+1] + 32) >> 6
 *   avg: dst[i] = (dst[i] + put_result[i] + 1) >> 1
 *
 * Overflow safety for 10-bit pixels (max 1023):
 *   max weighted sum = 64 * 1023 = 65472; +32 = 65504 < 65535 (uint16 max) ✓
 * For 9-bit pixels (max 511): 64 * 511 + 32 = 32736, trivially safe ✓
 * This implementation must NOT be used for bit_depth > 10 (12-bit: 64*4095+32
 * overflows uint16).
 *
 * mc2 for 9/10-bit is not implemented here (left as C fallback).
 */

static void put_h264_chroma_mc8_10_neon(uint8_t *_dst, const uint8_t *_src,
                                         ptrdiff_t stride, int h, int x, int y)
{
    uint16_t       *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const ptrdiff_t ps  = stride >> 1;        /* pixel stride */
    const int A = (8 - x) * (8 - y);
    const int B = x * (8 - y);
    const int C = (8 - x) * y;
    const int D = x * y;

    if (D) {
        for (int i = 0; i < h; i++) {
            uint16x8_t r0  = vld1q_u16(src);
            uint16x8_t r1  = vld1q_u16(src + 1);
            uint16x8_t s0  = vld1q_u16(src + ps);
            uint16x8_t s1  = vld1q_u16(src + ps + 1);
            uint16x8_t sum = vmulq_n_u16(r0, (uint16_t)A);
            sum = vmlaq_n_u16(sum, r1, (uint16_t)B);
            sum = vmlaq_n_u16(sum, s0, (uint16_t)C);
            sum = vmlaq_n_u16(sum, s1, (uint16_t)D);
            vst1q_u16(dst, vshrq_n_u16(vaddq_u16(sum, vdupq_n_u16(32)), 6));
            dst += ps;
            src += ps;
        }
    } else if (B + C) {
        const int E             = B + C;
        const ptrdiff_t step    = C ? ps : 1;
        for (int i = 0; i < h; i++) {
            uint16x8_t r0  = vld1q_u16(src);
            uint16x8_t r1  = vld1q_u16(src + step);
            uint16x8_t sum = vmulq_n_u16(r0, (uint16_t)A);
            sum = vmlaq_n_u16(sum, r1, (uint16_t)E);
            vst1q_u16(dst, vshrq_n_u16(vaddq_u16(sum, vdupq_n_u16(32)), 6));
            dst += ps;
            src += ps;
        }
    } else {
        /* A=64, x=0, y=0: (64*val+32)>>6 = val — simple copy */
        for (int i = 0; i < h; i++) {
            vst1q_u16(dst, vld1q_u16(src));
            dst += ps;
            src += ps;
        }
    }
}

static void avg_h264_chroma_mc8_10_neon(uint8_t *_dst, const uint8_t *_src,
                                         ptrdiff_t stride, int h, int x, int y)
{
    uint16_t       *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const ptrdiff_t ps  = stride >> 1;
    const int A = (8 - x) * (8 - y);
    const int B = x * (8 - y);
    const int C = (8 - x) * y;
    const int D = x * y;

    if (D) {
        for (int i = 0; i < h; i++) {
            uint16x8_t d   = vld1q_u16(dst);
            uint16x8_t r0  = vld1q_u16(src);
            uint16x8_t r1  = vld1q_u16(src + 1);
            uint16x8_t s0  = vld1q_u16(src + ps);
            uint16x8_t s1  = vld1q_u16(src + ps + 1);
            uint16x8_t sum = vmulq_n_u16(r0, (uint16_t)A);
            sum = vmlaq_n_u16(sum, r1, (uint16_t)B);
            sum = vmlaq_n_u16(sum, s0, (uint16_t)C);
            sum = vmlaq_n_u16(sum, s1, (uint16_t)D);
            uint16x8_t interp = vshrq_n_u16(vaddq_u16(sum, vdupq_n_u16(32)), 6);
            vst1q_u16(dst, vshrq_n_u16(
                vaddq_u16(vaddq_u16(d, interp), vdupq_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    } else if (B + C) {
        const int E             = B + C;
        const ptrdiff_t step    = C ? ps : 1;
        for (int i = 0; i < h; i++) {
            uint16x8_t d   = vld1q_u16(dst);
            uint16x8_t r0  = vld1q_u16(src);
            uint16x8_t r1  = vld1q_u16(src + step);
            uint16x8_t sum = vmulq_n_u16(r0, (uint16_t)A);
            sum = vmlaq_n_u16(sum, r1, (uint16_t)E);
            uint16x8_t interp = vshrq_n_u16(vaddq_u16(sum, vdupq_n_u16(32)), 6);
            vst1q_u16(dst, vshrq_n_u16(
                vaddq_u16(vaddq_u16(d, interp), vdupq_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    } else {
        /* A=64: interp = src; avg = (dst + src + 1) >> 1 */
        for (int i = 0; i < h; i++) {
            uint16x8_t d = vld1q_u16(dst);
            uint16x8_t s = vld1q_u16(src);
            vst1q_u16(dst, vshrq_n_u16(
                vaddq_u16(vaddq_u16(d, s), vdupq_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    }
}

/* mc4: 4 pixels per row — use 64-bit NEON registers */
static void put_h264_chroma_mc4_10_neon(uint8_t *_dst, const uint8_t *_src,
                                         ptrdiff_t stride, int h, int x, int y)
{
    uint16_t       *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const ptrdiff_t ps  = stride >> 1;
    const int A = (8 - x) * (8 - y);
    const int B = x * (8 - y);
    const int C = (8 - x) * y;
    const int D = x * y;

    if (D) {
        for (int i = 0; i < h; i++) {
            uint16x4_t r0  = vld1_u16(src);
            uint16x4_t r1  = vld1_u16(src + 1);
            uint16x4_t s0  = vld1_u16(src + ps);
            uint16x4_t s1  = vld1_u16(src + ps + 1);
            uint16x4_t sum = vmul_n_u16(r0, (uint16_t)A);
            sum = vmla_n_u16(sum, r1, (uint16_t)B);
            sum = vmla_n_u16(sum, s0, (uint16_t)C);
            sum = vmla_n_u16(sum, s1, (uint16_t)D);
            vst1_u16(dst, vshr_n_u16(vadd_u16(sum, vdup_n_u16(32)), 6));
            dst += ps;
            src += ps;
        }
    } else if (B + C) {
        const int E             = B + C;
        const ptrdiff_t step    = C ? ps : 1;
        for (int i = 0; i < h; i++) {
            uint16x4_t r0  = vld1_u16(src);
            uint16x4_t r1  = vld1_u16(src + step);
            uint16x4_t sum = vmul_n_u16(r0, (uint16_t)A);
            sum = vmla_n_u16(sum, r1, (uint16_t)E);
            vst1_u16(dst, vshr_n_u16(vadd_u16(sum, vdup_n_u16(32)), 6));
            dst += ps;
            src += ps;
        }
    } else {
        for (int i = 0; i < h; i++) {
            vst1_u16(dst, vld1_u16(src));
            dst += ps;
            src += ps;
        }
    }
}

static void avg_h264_chroma_mc4_10_neon(uint8_t *_dst, const uint8_t *_src,
                                         ptrdiff_t stride, int h, int x, int y)
{
    uint16_t       *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const ptrdiff_t ps  = stride >> 1;
    const int A = (8 - x) * (8 - y);
    const int B = x * (8 - y);
    const int C = (8 - x) * y;
    const int D = x * y;

    if (D) {
        for (int i = 0; i < h; i++) {
            uint16x4_t d   = vld1_u16(dst);
            uint16x4_t r0  = vld1_u16(src);
            uint16x4_t r1  = vld1_u16(src + 1);
            uint16x4_t s0  = vld1_u16(src + ps);
            uint16x4_t s1  = vld1_u16(src + ps + 1);
            uint16x4_t sum = vmul_n_u16(r0, (uint16_t)A);
            sum = vmla_n_u16(sum, r1, (uint16_t)B);
            sum = vmla_n_u16(sum, s0, (uint16_t)C);
            sum = vmla_n_u16(sum, s1, (uint16_t)D);
            uint16x4_t interp = vshr_n_u16(vadd_u16(sum, vdup_n_u16(32)), 6);
            vst1_u16(dst, vshr_n_u16(
                vadd_u16(vadd_u16(d, interp), vdup_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    } else if (B + C) {
        const int E             = B + C;
        const ptrdiff_t step    = C ? ps : 1;
        for (int i = 0; i < h; i++) {
            uint16x4_t d   = vld1_u16(dst);
            uint16x4_t r0  = vld1_u16(src);
            uint16x4_t r1  = vld1_u16(src + step);
            uint16x4_t sum = vmul_n_u16(r0, (uint16_t)A);
            sum = vmla_n_u16(sum, r1, (uint16_t)E);
            uint16x4_t interp = vshr_n_u16(vadd_u16(sum, vdup_n_u16(32)), 6);
            vst1_u16(dst, vshr_n_u16(
                vadd_u16(vadd_u16(d, interp), vdup_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    } else {
        for (int i = 0; i < h; i++) {
            uint16x4_t d = vld1_u16(dst);
            uint16x4_t s = vld1_u16(src);
            vst1_u16(dst, vshr_n_u16(vadd_u16(vadd_u16(d, s), vdup_n_u16(1)), 1));
            dst += ps;
            src += ps;
        }
    }
}

av_cold void ff_h264chroma_init_aarch64(H264ChromaContext *c, int bit_depth)
{
    const int high_bit_depth = bit_depth > 8;
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags) && !high_bit_depth) {
        c->put_h264_chroma_pixels_tab[0] = ff_put_h264_chroma_mc8_neon;
        c->put_h264_chroma_pixels_tab[1] = ff_put_h264_chroma_mc4_neon;
        c->put_h264_chroma_pixels_tab[2] = ff_put_h264_chroma_mc2_neon;

        c->avg_h264_chroma_pixels_tab[0] = ff_avg_h264_chroma_mc8_neon;
        c->avg_h264_chroma_pixels_tab[1] = ff_avg_h264_chroma_mc4_neon;
        c->avg_h264_chroma_pixels_tab[2] = ff_avg_h264_chroma_mc2_neon;
    }

    /* 9-bit and 10-bit high-depth MC.  The uint16 arithmetic requires
     * max_weighted_sum = 64 * pixel_max + 32 <= 65535; this holds for
     * bit_depth <= 10 (64*1023+32=65504) but NOT for 12-bit (overflows).
     * mc2 for high bit-depth is left as C fallback. */
    if (have_neon(cpu_flags) && bit_depth > 8 && bit_depth <= 10) {
        c->put_h264_chroma_pixels_tab[0] = put_h264_chroma_mc8_10_neon;
        c->put_h264_chroma_pixels_tab[1] = put_h264_chroma_mc4_10_neon;

        c->avg_h264_chroma_pixels_tab[0] = avg_h264_chroma_mc8_10_neon;
        c->avg_h264_chroma_pixels_tab[1] = avg_h264_chroma_mc4_10_neon;
    }
}
