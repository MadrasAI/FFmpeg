/*
 * Copyright (c) 2025
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/huffyuvencdsp.h"

void ff_sub_hfyu_median_pred_int16_neon(uint16_t *dst, const uint16_t *src1,
                                        const uint16_t *src2, unsigned mask,
                                        int w, int *left, int *left_top);

av_cold void ff_huffyuvencdsp_init_aarch64(HuffYUVEncDSPContext *c, int bpp, int width)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->sub_hfyu_median_pred_int16 = ff_sub_hfyu_median_pred_int16_neon;
    }
}
