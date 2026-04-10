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
 * int ff_idet_filter_line_neon(const uint8_t *a, const uint8_t *b,
 *                              const uint8_t *c, int w)
 * x0=a  x1=b  x2=c  w3=w
 *
 * Computes sum(|a[x] + c[x] - 2*b[x]|) for x=0..w-1
 * Main loop: 16 px/iter using widening uaddl/ushll + sub + abs + uaddlp.
 * Guard: check w3 >= 16 BEFORE each ld1 to avoid reading garbage past end.
 * Scalar tail: 0-15 remaining bytes via ldrb.
 */
function ff_idet_filter_line_neon, export=1
        movi    v17.4s, #0         // 32-bit vector accumulator
        mov     x4, #0             // scalar tail sum
        cmp     w3, #16
        b.lt    .Lid8_tail

.Lid8_main:
        ld1     {v0.16b}, [x0], #16
        ld1     {v1.16b}, [x1], #16
        ld1     {v2.16b}, [x2], #16
        uaddl   v3.8h, v0.8b, v2.8b      // a_lo + c_lo → uint16
        uaddl2  v4.8h, v0.16b, v2.16b    // a_hi + c_hi → uint16
        ushll   v5.8h, v1.8b, #1         // 2*b_lo → uint16
        ushll2  v6.8h, v1.16b, #1        // 2*b_hi → uint16
        sub     v3.8h, v3.8h, v5.8h      // a+c - 2b (int16 via wrap; range [-510,510])
        sub     v4.8h, v4.8h, v6.8h
        abs     v3.8h, v3.8h
        abs     v4.8h, v4.8h
        add     v3.8h, v3.8h, v4.8h      // merge 16 abs values into 8
        uaddlp  v5.4s, v3.8h             // pairwise widen to 4 uint32
        add     v17.4s, v17.4s, v5.4s
        sub     w3, w3, #16              // non-flag-setting; check before next load
        cmp     w3, #16
        b.ge    .Lid8_main               // loop only when >= 16 remain

.Lid8_tail:
        cbz     w3, .Lid8_done
.Lid8_tail_loop:
        ldrb    w5, [x0], #1
        ldrb    w6, [x1], #1
        ldrb    w7, [x2], #1
        add     w5, w5, w7
        lsl     w6, w6, #1
        sub     w5, w5, w6
        cmp     w5, #0
        cneg    w5, w5, lt
        add     w4, w4, w5
        subs    w3, w3, #1
        b.gt    .Lid8_tail_loop

.Lid8_done:
        addv    s18, v17.4s
        fmov    w0, s18
        add     w0, w0, w4
        ret
endfunc

/*
 * int ff_idet_filter_line_16bit_neon(const uint8_t *a, const uint8_t *b,
 *                                    const uint8_t *c, int w)
 * x0=a  x1=b  x2=c  w3=w (uint16 element count; pointers stride by 2 bytes/element)
 *
 * Main loop: 8 uint16/iter. Guard checks w3 >= 8 before each load.
 */
function ff_idet_filter_line_16bit_neon, export=1
        movi    v17.4s, #0
        mov     x4, #0
        cmp     w3, #8
        b.lt    .Lid16_tail

.Lid16_main:
        ld1     {v0.8h}, [x0], #16
        ld1     {v1.8h}, [x1], #16
        ld1     {v2.8h}, [x2], #16
        uaddl   v3.4s, v0.4h, v2.4h
        uaddl2  v4.4s, v0.8h, v2.8h
        ushll   v5.4s, v1.4h, #1
        ushll2  v6.4s, v1.8h, #1
        sub     v3.4s, v3.4s, v5.4s
        sub     v4.4s, v4.4s, v6.4s
        abs     v3.4s, v3.4s
        abs     v4.4s, v4.4s
        add     v17.4s, v17.4s, v3.4s
        add     v17.4s, v17.4s, v4.4s
        sub     w3, w3, #8
        cmp     w3, #8
        b.ge    .Lid16_main

.Lid16_tail:
        cbz     w3, .Lid16_done
.Lid16_tail_loop:
        ldrh    w5, [x0], #2
        ldrh    w6, [x1], #2
        ldrh    w7, [x2], #2
        add     w5, w5, w7
        lsl     w6, w6, #1
        sub     w5, w5, w6
        cmp     w5, #0
        cneg    w5, w5, lt
        add     w4, w4, w5
        subs    w3, w3, #1
        b.gt    .Lid16_tail_loop

.Lid16_done:
        addv    s18, v17.4s
        fmov    w0, s18
        add     w0, w0, w4
        ret
endfunc
