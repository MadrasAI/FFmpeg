/*
 * RV30/40 NEON optimisations for AArch64 — init file
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
#include "libavcodec/rv34dsp.h"

void ff_rv34_inv_transform_noround_neon(int16_t *block);
void ff_rv34_inv_transform_noround_dc_neon(int16_t *block);
void ff_rv34_idct_add_neon(uint8_t *dst, ptrdiff_t stride, int16_t *block);
void ff_rv34_idct_dc_add_neon(uint8_t *dst, ptrdiff_t stride, int dc);

av_cold void ff_rv34dsp_init_aarch64(RV34DSPContext *c)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->rv34_inv_transform = ff_rv34_inv_transform_noround_neon;
        c->rv34_idct_add      = ff_rv34_idct_add_neon;
        c->rv34_idct_dc_add   = ff_rv34_idct_dc_add_neon;
        // rv34_inv_transform_dc is NOT wired: the AArch64 scalar C path
        // (8.5 cycles) outperforms the NEON implementation (18.8 cycles)
        // for this tiny DC-fill operation.
    }
}
