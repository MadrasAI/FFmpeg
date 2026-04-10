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
 * unsigned ff_blackdetect_8_neon(const uint8_t *src, ptrdiff_t stride,
 *                                ptrdiff_t width, ptrdiff_t height,
 *                                unsigned threshold)
 * x0=src  x1=stride(bytes)  x2=width(uint8 count)  x3=height  w4=threshold
 *
 * Counts pixels where pixel <= threshold across the entire frame.
 * Main loop: 16 pixels/iter — cmhs + neg + uaddlp×2 → 4 uint32 partial sums.
 * Scalar tail: remaining 0-15 pixels via ldrb + cinc.
 */
function ff_blackdetect_8_neon, export=1
        dup     v16.16b, w4        // broadcast threshold (low 8 bits)
        movi    v17.4s, #0         // 32-bit vector accumulator
        mov     x8, #0             // scalar tail counter

.Lbd8_row:
        cbz     x3, .Lbd8_done
        mov     x5, x0             // row pointer
        mov     x6, x2             // columns remaining

.Lbd8_col:
        cmp     x6, #16
        b.lt    .Lbd8_tail
        ld1     {v0.16b}, [x5], #16
        cmhs    v1.16b, v16.16b, v0.16b   // 0xFF where threshold >= pixel
        neg     v1.16b, v1.16b            // 0x01 where pixel <= threshold
        uaddlp  v2.8h, v1.16b
        uaddlp  v3.4s, v2.8h
        add     v17.4s, v17.4s, v3.4s
        sub     x6, x6, #16
        b       .Lbd8_col

.Lbd8_tail:
        cbz     x6, .Lbd8_row_end
.Lbd8_tail_loop:
        ldrb    w7, [x5], #1
        cmp     w7, w4
        cinc    x8, x8, ls         // ls = lower or same (unsigned ≤)
        subs    x6, x6, #1
        b.gt    .Lbd8_tail_loop

.Lbd8_row_end:
        add     x0, x0, x1        // advance to next row (stride in bytes)
        sub     x3, x3, #1
        b       .Lbd8_row

.Lbd8_done:
        addv    s18, v17.4s        // horizontal sum of 4 uint32 accumulators
        fmov    w0, s18
        add     w0, w0, w8         // add scalar tail count
        ret
endfunc

/*
 * unsigned ff_blackdetect_16_neon(const uint8_t *src, ptrdiff_t stride,
 *                                 ptrdiff_t width, ptrdiff_t height,
 *                                 unsigned threshold)
 * x0=src  x1=stride(bytes)  x2=width(uint16 count)  x3=height  w4=threshold
 *
 * Counts uint16 pixels where pixel <= threshold.
 * Main loop: 8 uint16/iter — cmhs.8h + neg + uaddlp → 4 uint32 partial sums.
 * Scalar tail: remaining 0-7 uint16 via ldrh + cinc.
 */
function ff_blackdetect_16_neon, export=1
        dup     v16.8h, w4         // broadcast threshold (low 16 bits)
        movi    v17.4s, #0
        mov     x8, #0

.Lbd16_row:
        cbz     x3, .Lbd16_done
        mov     x5, x0
        mov     x6, x2

.Lbd16_col:
        cmp     x6, #8
        b.lt    .Lbd16_tail
        ld1     {v0.8h}, [x5], #16
        cmhs    v1.8h, v16.8h, v0.8h     // 0xFFFF where threshold >= pixel
        neg     v1.8h, v1.8h             // 0x0001 where pixel <= threshold
        uaddlp  v2.4s, v1.8h
        add     v17.4s, v17.4s, v2.4s
        sub     x6, x6, #8
        b       .Lbd16_col

.Lbd16_tail:
        cbz     x6, .Lbd16_row_end
.Lbd16_tail_loop:
        ldrh    w7, [x5], #2
        cmp     w7, w4
        cinc    x8, x8, ls
        subs    x6, x6, #1
        b.gt    .Lbd16_tail_loop

.Lbd16_row_end:
        add     x0, x0, x1
        sub     x3, x3, #1
        b       .Lbd16_row

.Lbd16_done:
        addv    s18, v17.4s
        fmov    w0, s18
        add     w0, w0, w8
        ret
endfunc
