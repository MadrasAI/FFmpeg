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

#include "libavutil/aarch64/asm.S"

/*
 * void ff_scene_sad8_neon(const uint8_t *src1, ptrdiff_t stride1,
 *                         const uint8_t *src2, ptrdiff_t stride2,
 *                         ptrdiff_t awidth, ptrdiff_t height,
 *                         uint64_t sad[2])
 *
 * 8-bit scene SAD: accumulate sum(abs(src1[x] - src2[x])) across all pixels.
 * awidth must be a multiple of 16. Writes two uint64_t partial sums to sad[].
 *
 * Algorithm per 16-byte chunk:
 *   uabd.16b  → abs(a-b) per byte (uint8)
 *   uaddlp.8h → pairwise widen to uint16
 *   uaddlp.4s → pairwise widen to uint32
 *   uaddlp.2d → pairwise widen to uint64 (two-lane accumulator)
 *   add v_acc.2d += chunk_sum.2d
 *
 * Overflow analysis: max abs diff = 255 per byte, 16 bytes per chunk,
 * up to 65536×65536 / 16 = 268M chunks → max per 2d lane = 134M × 255×8 = 2.7e11 < 2^64.
 */
function ff_scene_sad8_neon, export=1
        // x0=src1  x1=stride1  x2=src2  x3=stride2
        // x4=awidth  x5=height  x6=sad_ptr
        movi            v16.2d,  #0             // running accumulator (two uint64)

1:      // row loop
        mov             x7,  x4                 // column byte counter = awidth

2:      // column loop (16 bytes per iteration)
        ld1             {v0.16b},  [x0], #16
        ld1             {v1.16b},  [x2], #16
        uabd            v2.16b,  v0.16b,  v1.16b
        uaddlp          v2.8h,   v2.16b
        uaddlp          v2.4s,   v2.8h
        uaddlp          v2.2d,   v2.4s
        add             v16.2d,  v16.2d,  v2.2d
        subs            x7,  x7,  #16
        b.gt            2b

        // advance to next row
        sub             x0,  x0,  x4            // back to row start
        add             x0,  x0,  x1            // + stride1
        sub             x2,  x2,  x4            // back to row start
        add             x2,  x2,  x3            // + stride2
        subs            x5,  x5,  #1
        b.gt            1b

        st1             {v16.2d}, [x6]          // sad[0..1] = two partial sums
        ret
endfunc

/*
 * void ff_scene_sad16_neon(const uint8_t *src1, ptrdiff_t stride1,
 *                          const uint8_t *src2, ptrdiff_t stride2,
 *                          ptrdiff_t awidth, ptrdiff_t height,
 *                          uint64_t sad[2])
 *
 * 16-bit scene SAD: same as above but inputs are uint16.
 * awidth is in bytes (multiple of 16 = 8 uint16 values per iteration).
 *
 * uabd.8h computes max(a,b)-min(a,b) per uint16 lane (correct unsigned abs diff).
 */
function ff_scene_sad16_neon, export=1
        // x0=src1  x1=stride1  x2=src2  x3=stride2
        // x4=awidth (bytes, multiple of 16)  x5=height  x6=sad_ptr
        movi            v16.2d,  #0

1:      // row loop
        mov             x7,  x4

2:      // column loop (8 uint16 = 16 bytes per iteration)
        ld1             {v0.8h},   [x0], #16
        ld1             {v1.8h},   [x2], #16
        uabd            v2.8h,   v0.8h,   v1.8h
        uaddlp          v2.4s,   v2.8h
        uaddlp          v2.2d,   v2.4s
        add             v16.2d,  v16.2d,  v2.2d
        subs            x7,  x7,  #16
        b.gt            2b

        sub             x0,  x0,  x4
        add             x0,  x0,  x1
        sub             x2,  x2,  x4
        add             x2,  x2,  x3
        subs            x5,  x5,  #1
        b.gt            1b

        st1             {v16.2d}, [x6]
        ret
endfunc
