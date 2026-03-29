/*
 * AArch64 NEON optimised VP9 intra prediction functions
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

/*
 * Phase 1: 9 simple intra prediction modes x 4 block sizes = 36 functions
 *
 * Function signature (all modes):
 *   void f(uint8_t *dst, ptrdiff_t stride, const uint8_t *left, const uint8_t *top)
 *   x0=dst  x1=stride  x2=left  x3=top
 *
 * left[N-1] is the pixel adjacent to the top-left corner (top of block).
 * left[0]   is the bottom-most pixel of the left column.
 * top[-1]   is the top-left corner pixel (used by TM mode).
 *
 * Modes implemented:
 *   VERT (v)      — fill each row from the top row
 *   HOR  (h)      — fill each row with the corresponding left pixel
 *   DC            — average of left + top, rounded, broadcast
 *   DC_LEFT       — average of left column only
 *   DC_TOP        — average of top row only
 *   DC_128        — fill with constant 128
 *   DC_127        — fill with constant 127
 *   DC_129        — fill with constant 129
 *   TM            — true-motion: clip_u8(top[x] + left[N-1-y] - top[-1])
 */

#include "libavutil/aarch64/asm.S"

/* ------------------------------------------------------------------ */
/* VERT — copy top row to every output row                             */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_v_4x4_neon, export=1
        ldr     s0,  [x3]
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_v_8x8_neon, export=1
        ldr     d0,  [x3]
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_v_16x16_neon, export=1
        ld1     {v0.16b},  [x3]
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_v_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x3]
        mov     w4,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* HOR — fill each row with left[N-1-y]                                */
/* left[N-1] → row 0, left[N-2] → row 1, ..., left[0] → row N-1      */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_h_4x4_neon, export=1
        add     x2,  x2,  #3           // x2 → left[3]
        ldrb    w4,  [x2],  #-1
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        ldrb    w4,  [x2],  #-1
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        ldrb    w4,  [x2],  #-1
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        ldrb    w4,  [x2]
        dup     v0.8b,  w4
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_h_8x8_neon, export=1
        add     x2,  x2,  #7
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ldrb    w5,  [x2]
        dup     v0.8b,  w5
        st1     {v0.8b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_h_16x16_neon, export=1
        add     x2,  x2,  #15
        mov     w4,  #8                 // 2-row unrolled loop
1:      ldrb    w5,  [x2],  #-1
        dup     v0.16b,  w5
        st1     {v0.16b},  [x0],  x1
        ldrb    w5,  [x2],  #-1
        dup     v0.16b,  w5
        st1     {v0.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_h_32x32_neon, export=1
        add     x2,  x2,  #31
        mov     w4,  #32
1:      ldrb    w5,  [x2],  #-1
        dup     v0.16b,  w5
        mov     v1.16b,  v0.16b
        st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC_128 — fill block with constant 128                               */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_128_4x4_neon, export=1
        movi    v0.8b,  #128
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_128_8x8_neon, export=1
        movi    v0.8b,  #128
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_dc_128_16x16_neon, export=1
        movi    v0.16b,  #128
        mov     w4,  #8                 // 2-row unrolled loop
1:      st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_128_32x32_neon, export=1
        movi    v0.16b,  #128
        mov     v1.16b,  v0.16b
        mov     w4,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC_127 — fill block with constant 127                               */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_127_4x4_neon, export=1
        movi    v0.8b,  #127
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_127_8x8_neon, export=1
        movi    v0.8b,  #127
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_dc_127_16x16_neon, export=1
        movi    v0.16b,  #127
        mov     w4,  #8
1:      st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_127_32x32_neon, export=1
        movi    v0.16b,  #127
        mov     v1.16b,  v0.16b
        mov     w4,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC_129 — fill block with constant 129                               */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_129_4x4_neon, export=1
        movi    v0.8b,  #129
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_129_8x8_neon, export=1
        movi    v0.8b,  #129
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        st1     {v0.8b},  [x0],  x1
        ret
endfunc

function ff_vp9_ipred_dc_129_16x16_neon, export=1
        movi    v0.16b,  #129
        mov     w4,  #8
1:      st1     {v0.16b},  [x0],  x1
        st1     {v0.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_129_32x32_neon, export=1
        movi    v0.16b,  #129
        mov     v1.16b,  v0.16b
        mov     w4,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w4,  w4,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC_LEFT — dc from left column only                                  */
/* dc = (sum(left[0..N-1]) + N/2) >> log2(N)                          */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_left_4x4_neon, export=1
        ldr     s0,  [x2]               // left[0..3], zero-extends upper bytes
        uaddlv  h0,  v0.8b              // sum bytes 0-3 (bytes 4-7 == 0)
        umov    w4,  v0.h[0]
        add     w4,  w4,  #2
        lsr     w4,  w4,  #2
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_left_8x8_neon, export=1
        ldr     d0,  [x2]
        uaddlv  h0,  v0.8b
        umov    w4,  v0.h[0]
        add     w4,  w4,  #4
        lsr     w4,  w4,  #3
        dup     v0.8b,  w4
        mov     w5,  #8
1:      st1     {v0.8b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_left_16x16_neon, export=1
        ld1     {v0.16b},  [x2]
        uaddlv  h0,  v0.16b
        umov    w4,  v0.h[0]
        add     w4,  w4,  #8
        lsr     w4,  w4,  #4
        dup     v0.16b,  w4
        mov     w5,  #16
1:      st1     {v0.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_left_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x2]
        uaddlv  h0,  v0.16b
        uaddlv  h1,  v1.16b
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        add     w4,  w4,  w5
        add     w4,  w4,  #16
        lsr     w4,  w4,  #5
        dup     v0.16b,  w4
        mov     v1.16b,  v0.16b
        mov     w5,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC_TOP — dc from top row only                                       */
/* Identical to DC_LEFT but reads from x3 (top) instead of x2 (left)  */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_top_4x4_neon, export=1
        ldr     s0,  [x3]
        uaddlv  h0,  v0.8b
        umov    w4,  v0.h[0]
        add     w4,  w4,  #2
        lsr     w4,  w4,  #2
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_top_8x8_neon, export=1
        ldr     d0,  [x3]
        uaddlv  h0,  v0.8b
        umov    w4,  v0.h[0]
        add     w4,  w4,  #4
        lsr     w4,  w4,  #3
        dup     v0.8b,  w4
        mov     w5,  #8
1:      st1     {v0.8b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_top_16x16_neon, export=1
        ld1     {v0.16b},  [x3]
        uaddlv  h0,  v0.16b
        umov    w4,  v0.h[0]
        add     w4,  w4,  #8
        lsr     w4,  w4,  #4
        dup     v0.16b,  w4
        mov     w5,  #16
1:      st1     {v0.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_top_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x3]
        uaddlv  h0,  v0.16b
        uaddlv  h1,  v1.16b
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        add     w4,  w4,  w5
        add     w4,  w4,  #16
        lsr     w4,  w4,  #5
        dup     v0.16b,  w4
        mov     v1.16b,  v0.16b
        mov     w5,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* DC — average of left column and top row combined                    */
/* dc = (sum(left[0..N-1]) + sum(top[0..N-1]) + N) >> (log2(N)+1)    */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_dc_4x4_neon, export=1
        ldr     s0,  [x2]               // left[0..3]
        ldr     s1,  [x3]               // top[0..3]
        uaddlv  h0,  v0.8b              // sum(left) — bytes 4-7 are 0
        uaddlv  h1,  v1.8b              // sum(top)  — bytes 4-7 are 0
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        add     w4,  w4,  w5
        add     w4,  w4,  #4
        lsr     w4,  w4,  #3
        dup     v0.8b,  w4
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        add     x0,  x0,  x1
        str     s0,  [x0]
        ret
endfunc

function ff_vp9_ipred_dc_8x8_neon, export=1
        ldr     d0,  [x2]
        ldr     d1,  [x3]
        uaddlv  h0,  v0.8b
        uaddlv  h1,  v1.8b
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        add     w4,  w4,  w5
        add     w4,  w4,  #8
        lsr     w4,  w4,  #4
        dup     v0.8b,  w4
        mov     w5,  #8
1:      st1     {v0.8b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

function ff_vp9_ipred_dc_16x16_neon, export=1
        ld1     {v0.16b},  [x2]
        ld1     {v1.16b},  [x3]
        uaddlv  h0,  v0.16b
        uaddlv  h1,  v1.16b
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        add     w4,  w4,  w5
        add     w4,  w4,  #16
        lsr     w4,  w4,  #5
        dup     v0.16b,  w4
        mov     w5,  #16
1:      st1     {v0.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

// 32x32 DC: four uaddlv+umov groups, then scalar combine.
// C reference: (sum(left[0..31]) + sum(top[0..31]) + 32) >> 6
function ff_vp9_ipred_dc_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x2]    // left[0..31]
        ld1     {v2.16b,  v3.16b},  [x3]    // top[0..31]
        uaddlv  h0,  v0.16b                  // sum(left[0..15])
        uaddlv  h1,  v1.16b                  // sum(left[16..31])
        uaddlv  h2,  v2.16b                  // sum(top[0..15])
        uaddlv  h3,  v3.16b                  // sum(top[16..31])
        umov    w4,  v0.h[0]
        umov    w5,  v1.h[0]
        umov    w6,  v2.h[0]
        umov    w7,  v3.h[0]
        add     w4,  w4,  w5
        add     w6,  w6,  w7
        add     w4,  w4,  w6
        add     w4,  w4,  #32
        lsr     w4,  w4,  #6
        dup     v0.16b,  w4
        mov     v1.16b,  v0.16b
        mov     w5,  #32
1:      st1     {v0.16b,  v1.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

/* ------------------------------------------------------------------ */
/* TM — true-motion: clip_u8(top[x] + left[N-1-y] - top[-1])          */
/* Pre-compute diff[x] = top[x] - tl (s16), then per row add left[y]  */
/* ------------------------------------------------------------------ */

// 4x4 TM: pre-compute in v0.8h (lanes 0-3 valid, 4-7 = 0-tl garbage)
// Per row: dup left[y] as s16, add, sqxtun, store lower 4 bytes
function ff_vp9_ipred_tm_4x4_neon, export=1
        ldrb    w4,  [x3,  #-1]             // tl = top[-1]
        ldr     s0,  [x3]                   // top[0..3], zero-extends v0
        uxtl    v0.8h,  v0.8b               // top[0..3] as u16 in lanes 0-3
        dup     v2.8h,  w4                  // spread tl
        sub     v0.8h,  v0.8h,  v2.8h      // diff[x] = top[x] - tl (s16)
        add     x2,  x2,  #3               // x2 → left[3]
        mov     w5,  #4
1:      ldrb    w6,  [x2],  #-1
        dup     v1.8h,  w6
        add     v1.8h,  v0.8h,  v1.8h      // diff[x] + left[y]
        sqxtun  v1.8b,  v1.8h              // saturating narrow to u8
        st1     {v1.s}[0],  [x0],  x1      // store lower 4 bytes
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

// 8x8 TM: pre-compute in v0.8h (all 8 lanes valid)
function ff_vp9_ipred_tm_8x8_neon, export=1
        ldrb    w4,  [x3,  #-1]
        ldr     d0,  [x3]
        uxtl    v0.8h,  v0.8b
        dup     v2.8h,  w4
        sub     v0.8h,  v0.8h,  v2.8h
        add     x2,  x2,  #7
        mov     w5,  #8
1:      ldrb    w6,  [x2],  #-1
        dup     v1.8h,  w6
        add     v1.8h,  v0.8h,  v1.8h
        sqxtun  v1.8b,  v1.8h
        st1     {v1.8b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

// 16x16 TM: pre-compute in v4.8h (top[0..7]-tl) and v5.8h (top[8..15]-tl)
function ff_vp9_ipred_tm_16x16_neon, export=1
        ldrb    w4,  [x3,  #-1]
        ld1     {v0.16b},  [x3]
        uxtl    v4.8h,  v0.8b              // top[0..7] as u16
        uxtl2   v5.8h,  v0.16b             // top[8..15] as u16
        dup     v2.8h,  w4
        sub     v4.8h,  v4.8h,  v2.8h
        sub     v5.8h,  v5.8h,  v2.8h
        add     x2,  x2,  #15
        mov     w5,  #16
1:      ldrb    w6,  [x2],  #-1
        dup     v1.8h,  w6
        add     v6.8h,  v4.8h,  v1.8h
        add     v7.8h,  v5.8h,  v1.8h
        sqxtun  v6.8b,  v6.8h
        sqxtun2 v6.16b,  v7.8h
        st1     {v6.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc

// 32x32 TM: pre-compute in v16-v19 (caller-saved — no save needed)
// v16.8h = top[0..7]-tl,  v17.8h = top[8..15]-tl
// v18.8h = top[16..23]-tl, v19.8h = top[24..31]-tl
function ff_vp9_ipred_tm_32x32_neon, export=1
        ldrb    w4,  [x3,  #-1]
        ld1     {v0.16b,  v1.16b},  [x3]
        uxtl    v16.8h,  v0.8b             // top[0..7]
        uxtl2   v17.8h,  v0.16b            // top[8..15]
        uxtl    v18.8h,  v1.8b             // top[16..23]
        uxtl2   v19.8h,  v1.16b            // top[24..31]
        dup     v2.8h,  w4
        sub     v16.8h,  v16.8h,  v2.8h
        sub     v17.8h,  v17.8h,  v2.8h
        sub     v18.8h,  v18.8h,  v2.8h
        sub     v19.8h,  v19.8h,  v2.8h
        add     x2,  x2,  #31
        mov     w5,  #32
1:      ldrb    w6,  [x2],  #-1
        dup     v1.8h,  w6
        add     v4.8h,  v16.8h,  v1.8h
        add     v5.8h,  v17.8h,  v1.8h
        sqxtun  v4.8b,  v4.8h
        sqxtun2 v4.16b,  v5.8h             // v4.16b = top[0..15] row
        add     v5.8h,  v18.8h,  v1.8h
        add     v6.8h,  v19.8h,  v1.8h
        sqxtun  v5.8b,  v5.8h
        sqxtun2 v5.16b,  v6.8h             // v5.16b = top[16..31] row
        st1     {v4.16b,  v5.16b},  [x0],  x1
        subs    w5,  w5,  #1
        b.ne    1b
        ret
endfunc
