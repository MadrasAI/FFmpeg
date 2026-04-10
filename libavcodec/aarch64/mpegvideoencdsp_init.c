/*
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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "libavutil/attributes.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/mpegvideoencdsp.h"
#include "config.h"

int  ff_pix_sum16_neon(const uint8_t *pix, ptrdiff_t line_size);
int  ff_pix_norm1_neon(const uint8_t *pix, ptrdiff_t line_size);
void ff_denoise_dct_neon(int16_t block[64], int dct_error_sum[64],
                         const uint16_t dct_offset[64]);
void ff_add_8x8basis_neon(int16_t rem[64], const int16_t basis[64], int scale);
void ff_draw_edges_lr_neon(uint8_t *buf, ptrdiff_t wrap, int width,
                           int height, int w);

#if HAVE_DOTPROD
int ff_pix_norm1_neon_dotprod(const uint8_t *pix, ptrdiff_t line_size);
#endif

static void draw_edges_neon(uint8_t *buf, ptrdiff_t wrap, int width, int height,
                            int w, int h, int sides)
{
    uint8_t *last_line;
    int i;

    ff_draw_edges_lr_neon(buf, wrap, width, height, w);

    buf -= w;
    last_line = buf + (height - 1) * wrap;
    if (sides & EDGE_TOP)
        for (i = 0; i < h; i++)
            memcpy(buf - (i + 1) * wrap, buf, width + w + w);
    if (sides & EDGE_BOTTOM)
        for (i = 0; i < h; i++)
            memcpy(last_line + (i + 1) * wrap, last_line, width + w + w);
}

av_cold void ff_mpegvideoencdsp_init_aarch64(MpegvideoEncDSPContext *c,
                                             AVCodecContext *avctx)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->pix_sum    = ff_pix_sum16_neon;
        c->pix_norm1  = ff_pix_norm1_neon;
        c->denoise_dct = ff_denoise_dct_neon;
        c->add_8x8basis = ff_add_8x8basis_neon;
        c->draw_edges = draw_edges_neon;
    }

#if HAVE_DOTPROD
    if (have_dotprod(cpu_flags)) {
        c->pix_norm1 = ff_pix_norm1_neon_dotprod;
    }
#endif
}
