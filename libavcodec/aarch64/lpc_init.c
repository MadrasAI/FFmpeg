/*
 * AArch64 NEON optimised LPC DSP functions
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

#include <stdint.h>
#include <stddef.h>

#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/lpc.h"

void ff_lpc_apply_welch_window_neon(const int32_t *data, ptrdiff_t len,
                                    double *w_data);
void ff_lpc_compute_autocorr_neon(const double *data, ptrdiff_t len,
                                  int lag, double *autoc);

av_cold void ff_lpc_init_aarch64(LPCContext *s)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        s->lpc_apply_welch_window = ff_lpc_apply_welch_window_neon;
        s->lpc_compute_autocorr   = ff_lpc_compute_autocorr_neon;
    }
}
