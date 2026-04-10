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
#include "libavcodec/vp56dsp.h"

void ff_vp6_filter_diag4_neon(uint8_t *dst, const uint8_t *src,
                               ptrdiff_t stride,
                               const int16_t *h_weights,
                               const int16_t *v_weights);

av_cold void ff_vp6dsp_init_aarch64(VP6DSPContext *s)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags))
        s->vp6_filter_diag4 = ff_vp6_filter_diag4_neon;
}
