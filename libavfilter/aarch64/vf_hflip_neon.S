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
 * void ff_hflip_byte_neon(const uint8_t *src, uint8_t *dst, int w)
 * x0=src (points to LAST element)  x1=dst (forward)  w2=w (byte count)
 *
 * Per call: dst[j] = src[-j] for j=0..w-1
 * Main loop: 16 bytes/iter — load [src-15..src], rev64.16b + ext #8 → full reversal.
 * Scalar tail: 0-15 remaining bytes.
 * w2 is int (32-bit); all loop bounds use wN to avoid AAPCS64 upper-bit garbage.
 */
function ff_hflip_byte_neon, export=1
        cmp     w2, #16
        b.lt    .Lbyte_tail

.Lbyte_main:
        sub     x3, x0, #15
        ld1     {v0.16b}, [x3]
        rev64   v0.16b, v0.16b
        ext     v0.16b, v0.16b, v0.16b, #8
        st1     {v0.16b}, [x1], #16
        sub     x0, x0, #16
        subs    w2, w2, #16
        b.gt    .Lbyte_main

.Lbyte_tail:
        cbz     w2, .Lbyte_done
.Lbyte_tail_loop:
        ldrb    w3, [x0]
        strb    w3, [x1], #1
        sub     x0, x0, #1
        subs    w2, w2, #1
        b.gt    .Lbyte_tail_loop

.Lbyte_done:
        ret
endfunc

/*
 * void ff_hflip_short_neon(const uint8_t *src, uint8_t *dst, int w)
 * x0=src (points to LAST uint16 element, as uint8_t*)  x1=dst  w2=w (uint16 count)
 *
 * Per call: reinterpret as uint16_t*; dst[j] = src[-j] for j=0..w-1
 * Main loop: 8 uint16/iter — load [src-14..src+1], rev64.8h + ext #8 → full reversal.
 * Scalar tail: 0-7 remaining uint16 (1 at a time via ldrh/strh).
 */
function ff_hflip_short_neon, export=1
        cmp     w2, #8
        b.lt    .Lshort_tail

.Lshort_main:
        sub     x3, x0, #14
        ld1     {v0.8h}, [x3]
        rev64   v0.8h, v0.8h
        ext     v0.16b, v0.16b, v0.16b, #8
        st1     {v0.8h}, [x1], #16
        sub     x0, x0, #16
        subs    w2, w2, #8
        b.gt    .Lshort_main

.Lshort_tail:
        cbz     w2, .Lshort_done
.Lshort_tail_loop:
        ldrh    w3, [x0]
        strh    w3, [x1], #2
        sub     x0, x0, #2
        subs    w2, w2, #1
        b.gt    .Lshort_tail_loop

.Lshort_done:
        ret
endfunc
