/*
 * AArch64 NEON optimised block functions
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

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavcodec/blockdsp.h"

void ff_clear_block_neon(int16_t *block);
void ff_clear_blocks_neon(int16_t *blocks);
void ff_fill_block16_neon(uint8_t *block, uint8_t value, ptrdiff_t line_size, int h);
void ff_fill_block8_neon(uint8_t *block, uint8_t value, ptrdiff_t line_size, int h);

av_cold void ff_blockdsp_init_aarch64(BlockDSPContext *c)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        // clear_block/clear_blocks intentionally left as C (glibc memset with
        // stp-q or DC ZVA is faster than sequential st1 stores for these sizes)
        c->fill_block_tab[0] = ff_fill_block16_neon;
        c->fill_block_tab[1] = ff_fill_block8_neon;
    }
}
