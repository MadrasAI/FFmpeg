/*
 * Copyright (c) 2026 Ramaseshan M S
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
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
#include "libavfilter/vf_eq.h"

void ff_eq_process_neon(const uint8_t *src, uint8_t *dst,
                        short contrast, short brightness, int w);

/*
 * C wrapper matching the signature of EQContext->process.
 * Computes int16 contrast/brightness from the double fields in EQParameters
 * (identical formula to process_one_line_sse2 in x86/vf_eq_init.c),
 * then delegates each row to ff_eq_process_neon.
 */
static void process_neon(EQParameters *param, uint8_t *dst, int dst_stride,
                         const uint8_t *src, int src_stride, int w, int h)
{
    short contrast   = (short)(param->contrast * 256 * 16);
    short brightness = (short)(((short)(100.0 * param->brightness + 100.0) * 511)
                       / 200 - 128 - contrast / 32);

    while (h--) {
        ff_eq_process_neon(src, dst, contrast, brightness, w);
        src += src_stride;
        dst += dst_stride;
    }
}

av_cold void ff_eq_init_aarch64(EQContext *eq)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags))
        eq->process = process_neon;
}
