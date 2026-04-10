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
 * int ff_ssd_int8_vs_int16_neon(const int8_t *pix1, const int16_t *pix2,
 *                               intptr_t size)
 * x0=pix1, x1=pix2, x2=size
 *
 * Computes sum of (pix1[i] - pix2[i])^2 for i = 0..size-1.
 * size is always a multiple of 8 (guaranteed by caller).
 *
 * Per iteration (8 elements):
 *   v0.8b  : 8 × int8  from pix1  (→ sign-extended to v0.8h)
 *   v1.8h  : 8 × int16 from pix2
 *   v0.8h  : diff = pix1 - pix2  (int16)
 *   v2.4s  : smull(v0.4h, v0.4h) + smlal2(v0.8h, v0.8h) = diff^2 (int32)
 *   v3.4s  : running accumulator (4 partial int32 sums)
 *
 * Register plan (all volatile, no callee-saved registers used):
 *   v0    : pix1 int8 load, then widened diff
 *   v1    : pix2 int16 load
 *   v2    : per-iteration squared diffs (int32)
 *   v3    : accumulator (int32 × 4)
 */
function ff_ssd_int8_vs_int16_neon, export=1
        eor     v3.16b, v3.16b, v3.16b         // accumulator = 0

.Lssd_loop:
        ld1     {v0.8b},  [x0], #8             // load 8 × int8
        ld1     {v1.8h},  [x1], #16            // load 8 × int16
        sxtl    v0.8h, v0.8b                   // sign-extend int8 → int16
        sub     v0.8h, v0.8h, v1.8h            // diff = pix1 - pix2
        smull   v2.4s,  v0.4h, v0.4h           // low 4: diff^2 → int32
        smlal2  v2.4s,  v0.8h, v0.8h           // high 4: diff^2 → int32
        add     v3.4s,  v3.4s, v2.4s           // accumulate
        subs    x2, x2, #8
        b.gt    .Lssd_loop

        addp    v3.4s, v3.4s, v3.4s            // pairwise reduce: [a+b, c+d, a+b, c+d]
        addp    v3.4s, v3.4s, v3.4s            // pairwise reduce: [a+b+c+d, ...]
        fmov    w0, s3                          // extract int32 result
        ret
endfunc
