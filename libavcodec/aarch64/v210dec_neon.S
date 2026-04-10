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
 * TBL byte-shuffle index tables for V210 unpack.
 *
 * Per-iteration source block layout (24 bytes → 2 NEON registers):
 *   v4 bytes 0-7  : v1.4h = A group = [u0, y1, v1_val, y4]  (src & 0x3FF, narrowed)
 *   v4 bytes 8-15 : v2.4h = B group = [y0, u1, y3, v2_val]  ((src>>10)&0x3FF, narrowed)
 *   v5 bytes 0-7  : v3.4h = C group = [v0, y2, u2, y5]       (src>>20 & 0x3FF, narrowed)
 *   v5 bytes 8-15 : unused (no TBL index references bytes 24-31)
 *
 * V210 4-READ_PIXELS pattern (4 × 32-bit → 6Y + 3U + 3V):
 *   word0: u0=bits[9:0],  y0=bits[19:10], v0=bits[29:20]
 *   word1: y1=bits[9:0],  u1=bits[19:10], y2=bits[29:20]
 *   word2: v1=bits[9:0],  y3=bits[19:10], u2=bits[29:20]
 *   word3: y4=bits[9:0],  v2=bits[19:10], y5=bits[29:20]
 *
 * Y = {y0, y1, y2, y3, y4, y5}:
 *   y0 = B[0] = v4[8,9]   → index  8,  9
 *   y1 = A[1] = v4[2,3]   → index  2,  3
 *   y2 = C[1] = v5[2,3]   → index 18, 19
 *   y3 = B[2] = v4[12,13] → index 12, 13
 *   y4 = A[3] = v4[6,7]   → index  6,  7
 *   y5 = C[3] = v5[6,7]   → index 22, 23
 *
 * U = {u0, u1, u2}:
 *   u0 = A[0] = v4[0,1]   → index  0,  1
 *   u1 = B[1] = v4[10,11] → index 10, 11
 *   u2 = C[2] = v5[4,5]   → index 20, 21
 *
 * V = {v0, v1, v2}:
 *   v0 = C[0] = v5[0,1]   → index 16, 17
 *   v1 = A[2] = v4[4,5]   → index  4,  5
 *   v2 = B[3] = v4[14,15] → index 14, 15
 */
const v210_unpack_tbl_y, align=4
        .byte  8,  9,  2,  3, 18, 19, 12, 13,  6,  7, 22, 23, 0xff, 0xff, 0xff, 0xff
endconst

const v210_unpack_tbl_u, align=4
        .byte  0,  1, 10, 11, 20, 21, 0xff, 0xff
endconst

const v210_unpack_tbl_v, align=4
        .byte 16, 17,  4,  5, 14, 15, 0xff, 0xff
endconst

/*
 * void ff_v210_planar_unpack_neon(const uint32_t *src, uint16_t *y,
 *                                 uint16_t *u, uint16_t *v, int width)
 * x0=src, x1=y, x2=u, x3=v, w4=width
 *
 * width is int (32-bit). AAPCS64: caller-passed int has garbage upper 32 bits;
 * use w4 (32-bit) for loop comparisons, never x4.
 *
 * Per iteration: 4 × uint32 (16 bytes src) → 6Y + 3U + 3V uint16 output.
 * Loop processes floor(width/6)*6 samples (matching C ref: i < width-5, i+=6).
 *
 * Register plan (all volatile, no callee-saved registers used):
 *   v0   : loaded 4×uint32
 *   v1   : A = v0 & mask (then xtn → 4h)
 *   v2   : B = (v0>>10) & mask (then xtn → 4h)
 *   v3   : C = (v0>>20) & mask (then xtn → 4h)
 *   v4   : tbl source reg0 (v1.d[0] | v2.d[0])
 *   v5   : tbl source reg1 (v3.d[0], bytes 8-15 unused)
 *   v6   : Y tbl output (16b → bytes 0-11 = y[0..5])
 *   v7   : U tbl output (8b  → bytes 0-5  = u[0..2])
 *   v20  : V tbl output (8b  → bytes 0-5  = v[0..2])
 *   v16  : Y tbl index table (16b)
 *   v17  : U tbl index table (8b)
 *   v18  : V tbl index table (8b)
 *   v19  : 10-bit mask (0x000003FF per 4s lane)
 */
function ff_v210_planar_unpack_neon, export=1
        /* Load TBL index tables */
        adrp    x5, v210_unpack_tbl_y
        add     x5, x5, :lo12:v210_unpack_tbl_y
        ld1     {v16.16b}, [x5]

        adrp    x5, v210_unpack_tbl_u
        add     x5, x5, :lo12:v210_unpack_tbl_u
        ld1     {v17.8b}, [x5]

        adrp    x5, v210_unpack_tbl_v
        add     x5, x5, :lo12:v210_unpack_tbl_v
        ld1     {v18.8b}, [x5]

        /* 10-bit mask: 0x000003FF per 32-bit lane */
        movi    v19.4s, #0xff                   // 0x000000FF
        orr     v19.4s, #3, lsl #8              // 0x000003FF

        cmp     w4, #6
        b.lt    .Lunpack_done

.Lunpack_loop:
        ld1     {v0.4s}, [x0], #16             // load 4 × uint32

        /* Extract A (bits [9:0]), B (bits [19:10]), C (bits [29:20]) */
        and     v1.16b, v0.16b, v19.16b        // A = src & 0x3FF
        ushr    v2.4s, v0.4s, #10
        and     v2.16b, v2.16b, v19.16b        // B = (src>>10) & 0x3FF
        ushr    v3.4s, v0.4s, #20
        and     v3.16b, v3.16b, v19.16b        // C = (src>>20) & 0x3FF (V210 padding bits may be set)

        /* Narrow each group from uint32 to uint16 (10-bit, no saturation) */
        xtn     v1.4h, v1.4s                   // A → 4h
        xtn     v2.4h, v2.4s                   // B → 4h
        xtn     v3.4h, v3.4s                   // C → 4h

        /* Build 2-register TBL source block: {v4, v5} */
        ins     v4.d[0], v1.d[0]               // v4 bytes 0-7  = A = [u0,y1,v1,y4]
        ins     v4.d[1], v2.d[0]               // v4 bytes 8-15 = B = [y0,u1,y3,v2]
        ins     v5.d[0], v3.d[0]               // v5 bytes 0-7  = C = [v0,y2,u2,y5]

        /* TBL scatter: Y (12 bytes), U (6 bytes), V (6 bytes) */
        tbl     v6.16b, {v4.16b, v5.16b}, v16.16b  // Y: y[0..5] in bytes 0-11
        tbl     v7.8b,  {v4.16b, v5.16b}, v17.8b   // U: u[0..2] in bytes 0-5
        tbl     v20.8b, {v4.16b, v5.16b}, v18.8b   // V: v[0..2] in bytes 0-5

        /* Store Y: 12 bytes (6 × uint16) */
        st1     {v6.d}[0], [x1], #8            // y[0..3]
        st1     {v6.s}[2], [x1], #4            // y[4..5]

        /* Store U: 6 bytes (3 × uint16) */
        st1     {v7.s}[0], [x2], #4            // u[0..1]
        st1     {v7.h}[2], [x2], #2            // u[2]

        /* Store V: 6 bytes (3 × uint16) */
        st1     {v20.s}[0], [x3], #4           // v[0..1]
        st1     {v20.h}[2], [x3], #2           // v[2]

        sub     w4, w4, #6
        cmp     w4, #6
        b.ge    .Lunpack_loop

.Lunpack_done:
        ret
endfunc
