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
 * TBL byte-shuffle index tables for V210 packing.
 *
 * Source block layout (24 bytes in two 128-bit registers):
 *
 *  pack_8  (x0=y 8-bit, x1=u 8-bit, x2=v 8-bit):
 *   v0 bytes 0-11  : y[0..11]
 *   v0 bytes 12-15 : u[0..3]
 *   v1 bytes 0-1   : u[4..5]
 *   v1 bytes 2-3   : v[0..1]
 *   v1 bytes 4-7   : v[2..5]
 *
 *  V210 8-word output pattern (8 × WRITE_PIXELS, depth=8):
 *   word0: a=u[0], b=y[0], c=v[0]
 *   word1: a=y[1], b=u[1], c=y[2]
 *   word2: a=v[1], b=y[3], c=u[2]
 *   word3: a=y[4], b=v[2], c=y[5]
 *   word4: a=u[3], b=y[6], c=v[3]
 *   word5: a=y[7], b=u[4], c=y[8]
 *   word6: a=v[4], b=y[9], c=u[5]
 *   word7: a=y[10],b=v[5], c=y[11]
 *
 *  Each word = (a << 2) | (b << 12) | (c << 22)  (depth=8, shift=10-8=2)
 *  Shift chains: A=0+2, B=4+8, C=7+15 (ushll 16->32 max shift is 15)
 *
 *  A indices (bits [9:2]):
 *   u[0]=12, y[1]=1, v[1]=19, y[4]=4, u[3]=15, y[7]=7, v[4]=22, y[10]=10
 *  B indices (bits [19:12]):
 *   y[0]=0,  u[1]=13, y[3]=3,  v[2]=20, y[6]=6,  u[4]=16, y[9]=9,  v[5]=23
 *  C indices (bits [29:22]):
 *   v[0]=18, y[2]=2,  u[2]=14, y[5]=5,  v[3]=21, y[8]=8,  u[5]=17, y[11]=11
 */
const v210_pack8_tbl, align=4
        .byte 12,  1, 19,  4, 15,  7, 22, 10   /* A */
        .byte  0, 13,  3, 20,  6, 16,  9, 23   /* B */
        .byte 18,  2, 14,  5, 21,  8, 17, 11   /* C */
endconst

/*
 *  pack_10 (x0=y 16-bit, x1=u 16-bit, x2=v 16-bit):
 *   v0 bytes 0-11  : y[0..5]  (6 × uint16)
 *   v0 bytes 12-15 : u[0..1]  (2 × uint16)
 *   v1 bytes 0-1   : u[2]
 *   v1 bytes 2-7   : v[0..2]  (3 × uint16)
 *
 *  V210 4-word output pattern (4 × WRITE_PIXELS, depth=10):
 *   word0: a=u[0], b=y[0], c=v[0]
 *   word1: a=y[1], b=u[1], c=y[2]
 *   word2: a=v[1], b=y[3], c=u[2]
 *   word3: a=y[4], b=v[2], c=y[5]
 *
 *  Each word = a | (b << 10) | (c << 20)  (depth=10, no shift for a/b/c)
 *
 *  Byte indices (2 bytes per uint16 value, little-endian):
 *  A: u[0]={12,13}, y[1]={2,3},  v[1]={20,21}, y[4]={8,9}
 *  B: y[0]={0,1},   u[1]={14,15},y[3]={6,7},   v[2]={22,23}
 *  C: v[0]={18,19}, y[2]={4,5},  u[2]={16,17}, y[5]={10,11}
 */
const v210_pack10_tbl, align=4
        .byte 12, 13,  2,  3, 20, 21,  8,  9   /* A */
        .byte  0,  1, 14, 15,  6,  7, 22, 23   /* B */
        .byte 18, 19,  4,  5, 16, 17, 10, 11   /* C */
endconst

/*
 * void ff_v210_planar_pack_8_neon(const uint8_t *y, const uint8_t *u,
 *                                 const uint8_t *v, uint8_t *dst,
 *                                 ptrdiff_t width)
 * x0=y, x1=u, x2=v, x3=dst, x4=width
 *
 * Processes floor(width/12)*12 luma samples per call.
 * Caller has arranged width to be a multiple of 12 (sample_factor_8=2).
 *
 * Per iteration: 12Y + 6U + 6V → 8 × 32-bit V210 words (32 bytes)
 *
 * Inner loop register use:
 *   v0,v1   : source block (2 × 16 bytes = 32 byte tbl source)
 *   v16-v18 : tbl index tables A, B, C (stable across loop)
 *   v19,v20 : clip_min (1), clip_max (254) for 8-bit (stable)
 *   v21-v23 : gathered A, B, C byte values
 *   v0-v6   : ushll intermediates (reused after tbl)
 *   v24,v25 : C_lo, C_hi 32-bit shifted values
 *   v1,v2   : final output words 0-3, 4-7
 *
 * No callee-saved NEON registers (v8-v15) are used.
 */
function ff_v210_planar_pack_8_neon, export=1
        /* Load stable clip constants */
        movi    v19.16b, #1                     // clip_min = 1
        movi    v20.16b, #0xfe                  // clip_max = 254

        /* Load TBL index tables (A=v16, B=v17, C=v18) */
        adrp    x5, v210_pack8_tbl
        add     x5, x5, :lo12:v210_pack8_tbl
        ld1     {v16.8b, v17.8b, v18.8b}, [x5]

        cmp     x4, #12
        b.lt    .Lpack8_done

.Lpack8_loop:
        /* Build source block in v0 (y[0..11] | u[0..3]) */
        ld1     {v0.8b},   [x0], #8             // v0[0..7]  = y[0..7]
        ld1     {v0.s}[2], [x0], #4             // v0[8..11] = y[8..11]
        ld1     {v0.s}[3], [x1], #4             // v0[12..15]= u[0..3]

        /* Build source block in v1 (u[4..5] | v[0..5]) */
        ld1     {v1.h}[0], [x1], #2             // v1[0..1]  = u[4..5]
        ld1     {v1.h}[1], [x2], #2             // v1[2..3]  = v[0..1]
        ld1     {v1.s}[1], [x2], #4             // v1[4..7]  = v[2..5]

        /* Clip to [1, 254] */
        umax    v0.16b, v0.16b, v19.16b
        umin    v0.16b, v0.16b, v20.16b
        umax    v1.16b, v1.16b, v19.16b
        umin    v1.16b, v1.16b, v20.16b

        /* Gather A, B, C byte values via tbl */
        tbl     v21.8b, {v0.16b, v1.16b}, v16.8b  // A: word[i] bits[9:2]
        tbl     v22.8b, {v0.16b, v1.16b}, v17.8b  // B: word[i] bits[19:12]
        tbl     v23.8b, {v0.16b, v1.16b}, v18.8b  // C: word[i] bits[29:22]

        /* Widen and shift A: total shift = 2 (0 + 2) */
        ushll   v0.8h,  v21.8b, #0              // zero-extend A to 16-bit
        ushll   v1.4s,  v0.4h,  #2              // A_lo: words 0-3 → bits[9:2]
        ushll2  v2.4s,  v0.8h,  #2              // A_hi: words 4-7 → bits[9:2]

        /* Widen and shift B: total shift = 12 (4 + 8) */
        ushll   v3.8h,  v22.8b, #4              // shift 4 during 8→16
        ushll   v4.4s,  v3.4h,  #8              // B_lo: shift 8 more → total 12
        ushll2  v5.4s,  v3.8h,  #8              // B_hi

        /* Widen and shift C: total shift = 22 (7 + 15) */
        ushll   v6.8h,  v23.8b, #7              // shift 7 during 8→16
        ushll   v24.4s, v6.4h,  #15             // C_lo: shift 15 more → total 22
        ushll2  v25.4s, v6.8h,  #15             // C_hi

        /* Combine: output = A | B | C */
        orr     v1.16b, v1.16b, v4.16b          // words 0-3: A|B
        orr     v1.16b, v1.16b, v24.16b         // words 0-3: A|B|C
        orr     v2.16b, v2.16b, v5.16b          // words 4-7: A|B
        orr     v2.16b, v2.16b, v25.16b         // words 4-7: A|B|C

        /* Store 32 bytes */
        st1     {v1.4s, v2.4s}, [x3], #32

        sub     x4, x4, #12
        cmp     x4, #12
        b.ge    .Lpack8_loop

.Lpack8_done:
        ret
endfunc

/*
 * void ff_v210_planar_pack_10_neon(const uint16_t *y, const uint16_t *u,
 *                                  const uint16_t *v, uint8_t *dst,
 *                                  ptrdiff_t width)
 * x0=y, x1=u, x2=v, x3=dst, x4=width
 *
 * Processes floor(width/6)*6 luma samples per call.
 * Caller has arranged width to be a multiple of 6 (sample_factor_10=1).
 *
 * Per iteration: 6Y + 3U + 3V → 4 × 32-bit V210 words (16 bytes)
 *
 * No callee-saved NEON registers are used.
 */
function ff_v210_planar_pack_10_neon, export=1
        /* Load stable clip constants for 10-bit [4, 1019] */
        movi    v27.8h, #4                      // clip_min = 4
        movi    v28.8h, #3, lsl #8              // 0x0300
        orr     v28.8h, #0xfb                   // 0x03fb = 1019 = clip_max

        /* Load TBL index tables (A=v29, B=v30, C=v31) */
        adrp    x5, v210_pack10_tbl
        add     x5, x5, :lo12:v210_pack10_tbl
        ld1     {v29.8b, v30.8b, v31.8b}, [x5]

        cmp     x4, #6
        b.lt    .Lpack10_done

.Lpack10_loop:
        /* Build source block in v0 (y[0..5] | u[0..1]) */
        ld1     {v0.4h},   [x0], #8             // v0[0..7] = y[0..3] (4×16b)
        ld1     {v0.h}[4], [x0], #2             // v0[8..9] = y[4]
        ld1     {v0.h}[5], [x0], #2             // v0[10..11] = y[5]
        ld1     {v0.h}[6], [x1], #2             // v0[12..13] = u[0]
        ld1     {v0.h}[7], [x1], #2             // v0[14..15] = u[1]

        /* Build source block in v1 (u[2] | v[0..2]) */
        ld1     {v1.h}[0], [x1], #2             // v1[0..1] = u[2]
        ld1     {v1.h}[1], [x2], #2             // v1[2..3] = v[0]
        ld1     {v1.h}[2], [x2], #2             // v1[4..5] = v[1]
        ld1     {v1.h}[3], [x2], #2             // v1[6..7] = v[2]

        /* Clip to [4, 1019] */
        umax    v0.8h, v0.8h, v27.8h
        umin    v0.8h, v0.8h, v28.8h
        umax    v1.8h, v1.8h, v27.8h
        umin    v1.8h, v1.8h, v28.8h

        /* Gather A, B, C values via tbl (4 × uint16 each = 8 bytes) */
        tbl     v2.8b, {v0.16b, v1.16b}, v29.8b // A: word[i] bits[9:0]
        tbl     v3.8b, {v0.16b, v1.16b}, v30.8b // B: word[i] bits[19:10]
        tbl     v4.8b, {v0.16b, v1.16b}, v31.8b // C: word[i] bits[29:20]

        /* Widen to 32-bit and shift */
        ushll   v2.4s, v2.4h, #0               // A at bits[9:0] (no shift)
        ushll   v3.4s, v3.4h, #10              // B at bits[19:10] (shift 10)
        ushll   v4.4s, v4.4h, #15              // C: first 15 bits of 20-bit shift
        shl     v4.4s, v4.4s, #5               // C: +5 = total 20 → bits[29:20]

        /* Combine and store */
        orr     v2.16b, v2.16b, v3.16b
        orr     v2.16b, v2.16b, v4.16b
        st1     {v2.4s}, [x3], #16

        sub     x4, x4, #6
        cmp     x4, #6
        b.ge    .Lpack10_loop

.Lpack10_done:
        ret
endfunc
