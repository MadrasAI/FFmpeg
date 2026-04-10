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
#include "libavfilter/hflip.h"

void ff_hflip_byte_neon(const uint8_t *src, uint8_t *dst, int w);
void ff_hflip_short_neon(const uint8_t *src, uint8_t *dst, int w);

av_cold void ff_hflip_init_aarch64(FlipContext *s, int step[4], int nb_planes)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        for (int i = 0; i < nb_planes; i++) {
            switch (step[i]) {
            case 1: s->flip_line[i] = ff_hflip_byte_neon;  break;
            case 2: s->flip_line[i] = ff_hflip_short_neon; break;
            }
        }
    }
}
