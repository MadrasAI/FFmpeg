/*
 * ARM NEON optimised Fixed DSP functions
 * Copyright (c) 2024
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
#include "libavutil/fixed_dsp.h"
#include "cpu.h"

void ff_vector_fmul_fixed_neon(int *dst, const int *src0, const int *src1,
                                int len);

void ff_vector_fmul_add_fixed_neon(int *dst, const int *src0, const int *src1,
                                    const int *src2, int len);

void ff_vector_fmul_reverse_fixed_neon(int *dst, const int *src0,
                                        const int *src1, int len);

void ff_butterflies_fixed_neon(int *restrict v1, int *restrict v2, int len);

int ff_scalarproduct_fixed_neon(const int *v1, const int *v2, int len);

void ff_vector_fmul_window_fixed_neon(int32_t *dst, const int32_t *src0,
                                       const int32_t *src1, const int32_t *win,
                                       int len);

void ff_vector_fmul_window_scaled_fixed_neon(int16_t *dst, const int32_t *src0,
                                              const int32_t *src1,
                                              const int32_t *win,
                                              int len, uint8_t bits);

av_cold void ff_fixed_dsp_init_aarch64(AVFixedDSPContext *fdsp)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        fdsp->vector_fmul                = ff_vector_fmul_fixed_neon;
        fdsp->vector_fmul_add            = ff_vector_fmul_add_fixed_neon;
        fdsp->vector_fmul_reverse        = ff_vector_fmul_reverse_fixed_neon;
        fdsp->butterflies_fixed          = ff_butterflies_fixed_neon;
        fdsp->scalarproduct_fixed        = ff_scalarproduct_fixed_neon;
        fdsp->vector_fmul_window         = ff_vector_fmul_window_fixed_neon;
        fdsp->vector_fmul_window_scaled  = ff_vector_fmul_window_scaled_fixed_neon;
    }
}
