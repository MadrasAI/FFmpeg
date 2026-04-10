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
#include "libavfilter/gblur.h"

void ff_postscale_slice_neon(float *buffer, int length,
                             float postscale, float min, float max);
void ff_verti_slice_neon(float *buffer, int width, int height,
                         int slice_start, int slice_end, int steps,
                         float nu, float bscale);

av_cold void ff_gblur_init_aarch64(GBlurContext *s)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        s->postscale_slice = ff_postscale_slice_neon;
        // verti_slice: IIR row dependency means OOO scalar (8-col) beats NEON (4-col)
        // horiz_slice: IIR dependency chain — not vectorized without localbuf
    }
}
