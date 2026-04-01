/*
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
#include "libavcodec/utvideodsp.h"

void ff_restore_rgb_planes_neon(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b,
                                ptrdiff_t linesize_r, ptrdiff_t linesize_g,
                                ptrdiff_t linesize_b, int width, int height);
void ff_restore_rgb_planes10_neon(uint16_t *src_r, uint16_t *src_g, uint16_t *src_b,
                                  ptrdiff_t linesize_r, ptrdiff_t linesize_g,
                                  ptrdiff_t linesize_b, int width, int height);

av_cold void ff_utvideodsp_init_aarch64(UTVideoDSPContext *c)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->restore_rgb_planes   = ff_restore_rgb_planes_neon;
        c->restore_rgb_planes10 = ff_restore_rgb_planes10_neon;
    }
}
