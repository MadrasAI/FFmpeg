/*
 * AArch64 NEON optimised lossless audio DSP functions
 * Copyright (c) 2025 FFmpeg contributors
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
#include "libavcodec/lossless_audiodsp.h"

int32_t ff_scalarproduct_and_madd_int16_neon(int16_t *v1,
                                             const int16_t *v2,
                                             const int16_t *v3,
                                             int order, int mul);
int32_t ff_scalarproduct_and_madd_int32_neon(int16_t *v1,
                                             const int32_t *v2,
                                             const int16_t *v3,
                                             int order, int mul);

av_cold void ff_llauddsp_init_aarch64(LLAudDSPContext *c)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->scalarproduct_and_madd_int16 = ff_scalarproduct_and_madd_int16_neon;
        c->scalarproduct_and_madd_int32 = ff_scalarproduct_and_madd_int32_neon;
    }
}
