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

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavfilter/vf_blackdetect.h"

unsigned ff_blackdetect_8_neon(const uint8_t *src, ptrdiff_t stride,
                               ptrdiff_t width, ptrdiff_t height,
                               unsigned threshold);
unsigned ff_blackdetect_16_neon(const uint8_t *src, ptrdiff_t stride,
                                ptrdiff_t width, ptrdiff_t height,
                                unsigned threshold);

ff_blackdetect_fn ff_blackdetect_get_fn_aarch64(int depth)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags))
        return depth == 8 ? ff_blackdetect_8_neon : ff_blackdetect_16_neon;
    return NULL;
}
