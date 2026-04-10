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

#include <stdint.h>

#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavfilter/scene_sad.h"

void ff_scene_sad8_neon(const uint8_t *src1, ptrdiff_t stride1,
                        const uint8_t *src2, ptrdiff_t stride2,
                        ptrdiff_t awidth, ptrdiff_t height, uint64_t sad[2]);
void ff_scene_sad16_neon(const uint8_t *src1, ptrdiff_t stride1,
                         const uint8_t *src2, ptrdiff_t stride2,
                         ptrdiff_t awidth, ptrdiff_t height, uint64_t sad[2]);

/* C wrappers: align width to 16-byte boundary, call NEON for bulk,
 * fall back to C scalar for any remaining bytes. */

static void scene_sad8_neon(SCENE_SAD_PARAMS)
{
    uint64_t sad_arr[2] = { 0, 0 };
    ptrdiff_t awidth = width & ~15;
    *sum = 0;
    if (awidth > 0)
        ff_scene_sad8_neon(src1, stride1, src2, stride2, awidth, height, sad_arr);
    *sum = sad_arr[0] + sad_arr[1];
    if (width > awidth) {
        uint64_t tail = 0;
        ff_scene_sad_c(src1 + awidth, stride1,
                       src2 + awidth, stride2,
                       width - awidth, height, &tail);
        *sum += tail;
    }
}

static void scene_sad16_neon(SCENE_SAD_PARAMS)
{
    uint64_t sad_arr[2] = { 0, 0 };
    /* width is in uint16 pixels; align byte count to 16 */
    ptrdiff_t bytes = (width << 1) & ~15;
    *sum = 0;
    if (bytes > 0)
        ff_scene_sad16_neon(src1, stride1, src2, stride2, bytes, height, sad_arr);
    *sum = sad_arr[0] + sad_arr[1];
    ptrdiff_t tail_pixels = width - (bytes >> 1);
    if (tail_pixels > 0) {
        uint64_t tail = 0;
        ff_scene_sad16_c(src1 + bytes, stride1,
                         src2 + bytes, stride2,
                         tail_pixels, height, &tail);
        *sum += tail;
    }
}

ff_scene_sad_fn ff_scene_sad_get_fn_aarch64(int depth)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        if (depth <= 8)
            return scene_sad8_neon;
        else if (depth <= 16)   /* uabd.8h is unsigned: safe for full uint16 range */
            return scene_sad16_neon;
    }
    return NULL;
}
