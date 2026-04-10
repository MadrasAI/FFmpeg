/*
 * Copyright (c) 2026 Ramaseshan M S
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

#include <stdint.h>

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/cavsdsp.h"

/* 8-pixel-wide NEON kernels */
void ff_put_cavs_qpel8_mc20_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);
void ff_put_cavs_qpel8_mc02_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);
void ff_put_cavs_qpel8_mc03_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);
void ff_avg_cavs_qpel8_mc20_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);
void ff_avg_cavs_qpel8_mc02_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);
void ff_avg_cavs_qpel8_mc03_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride);

/* Variable-height NEON kernels (used for 16×16 wrappers) */
void ff_put_cavs_qpel8_h_neon (uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);
void ff_put_cavs_qpel8_v2_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);
void ff_put_cavs_qpel8_v3_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);
void ff_avg_cavs_qpel8_h_neon (uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);
void ff_avg_cavs_qpel8_v2_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);
void ff_avg_cavs_qpel8_v3_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h);

/*
 * 16×16 wrappers: call the 8-pixel-wide kernel twice (left and right 8-column strips).
 * mc01 (V qpel_l) is computed from mc03 (V qpel_r) by time-reversal:
 *   ff_put_cavs_qpel8_mc03_neon(dst + 7*stride, src + 8*stride, -stride)
 * reverses the filter direction, yielding exactly the qpel_l output.
 */

#define DEF_QPEL16(OPNAME) \
static void OPNAME ## _cavs_qpel16_mc20_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride) \
{ \
    ff_ ## OPNAME ## _cavs_qpel8_h_neon(dst,     src,     stride, 16); \
    ff_ ## OPNAME ## _cavs_qpel8_h_neon(dst + 8, src + 8, stride, 16); \
} \
static void OPNAME ## _cavs_qpel16_mc02_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride) \
{ \
    ff_ ## OPNAME ## _cavs_qpel8_v2_neon(dst,     src,     stride, 16); \
    ff_ ## OPNAME ## _cavs_qpel8_v2_neon(dst + 8, src + 8, stride, 16); \
} \
static void OPNAME ## _cavs_qpel16_mc03_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride) \
{ \
    ff_ ## OPNAME ## _cavs_qpel8_v3_neon(dst,     src,     stride, 16); \
    ff_ ## OPNAME ## _cavs_qpel8_v3_neon(dst + 8, src + 8, stride, 16); \
} \
static void OPNAME ## _cavs_qpel8_mc01_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride) \
{ \
    ff_ ## OPNAME ## _cavs_qpel8_mc03_neon(dst + 7 * stride, src + 8 * stride, -stride); \
} \
static void OPNAME ## _cavs_qpel16_mc01_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride) \
{ \
    OPNAME ## _cavs_qpel16_mc03_neon(dst + 15 * stride, src + 16 * stride, -stride); \
}

DEF_QPEL16(put)
DEF_QPEL16(avg)

av_cold void ff_cavsdsp_init_aarch64(CAVSDSPContext *c)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        /* 8×8 block positions */
        c->put_cavs_qpel_pixels_tab[1][ 2] = ff_put_cavs_qpel8_mc20_neon;
        c->put_cavs_qpel_pixels_tab[1][ 4] = put_cavs_qpel8_mc01_neon;
        c->put_cavs_qpel_pixels_tab[1][ 8] = ff_put_cavs_qpel8_mc02_neon;
        c->put_cavs_qpel_pixels_tab[1][12] = ff_put_cavs_qpel8_mc03_neon;

        c->avg_cavs_qpel_pixels_tab[1][ 2] = ff_avg_cavs_qpel8_mc20_neon;
        c->avg_cavs_qpel_pixels_tab[1][ 4] = avg_cavs_qpel8_mc01_neon;
        c->avg_cavs_qpel_pixels_tab[1][ 8] = ff_avg_cavs_qpel8_mc02_neon;
        c->avg_cavs_qpel_pixels_tab[1][12] = ff_avg_cavs_qpel8_mc03_neon;

        /* 16×16 block positions */
        c->put_cavs_qpel_pixels_tab[0][ 2] = put_cavs_qpel16_mc20_neon;
        c->put_cavs_qpel_pixels_tab[0][ 4] = put_cavs_qpel16_mc01_neon;
        c->put_cavs_qpel_pixels_tab[0][ 8] = put_cavs_qpel16_mc02_neon;
        c->put_cavs_qpel_pixels_tab[0][12] = put_cavs_qpel16_mc03_neon;

        c->avg_cavs_qpel_pixels_tab[0][ 2] = avg_cavs_qpel16_mc20_neon;
        c->avg_cavs_qpel_pixels_tab[0][ 4] = avg_cavs_qpel16_mc01_neon;
        c->avg_cavs_qpel_pixels_tab[0][ 8] = avg_cavs_qpel16_mc02_neon;
        c->avg_cavs_qpel_pixels_tab[0][12] = avg_cavs_qpel16_mc03_neon;
    }
}
