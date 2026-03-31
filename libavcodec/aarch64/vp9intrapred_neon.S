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
 * Modes implemented (Phase 1):
 *   VERT (v)      — fill each row from the top row
 *   HOR  (h)      — fill each row with the corresponding left pixel
 *   DC            — average of left + top, rounded, broadcast
 *   DC_LEFT       — average of left column only
 *   DC_TOP        — average of top row only
 *   DC_128        — fill with constant 128
 *   DC_127        — fill with constant 127
 *   DC_129        — fill with constant 129
 *   TM            — true-motion: clip_u8(top[x] + left[N-1-y] - top[-1])
 *
 * Phase 2: 6 diagonal intra prediction modes x 4 block sizes = 24 functions
 *
 * left[] ordering for Phase 2 modes:
 *   Standard modes (DIAG_DOWN_LEFT, DIAG_DOWN_RIGHT, VERT_RIGHT, HOR_DOWN,
 *   VERT_LEFT): left[0]=bottommost, left[N-1]=topmost  (same as Phase 1)
 *   HOR_UP: left[0]=topmost, left[N-1]=bottommost  (invert_left=1 in decoder)
 *
 * Modes implemented (Phase 2):
 *   DIAG_DOWN_LEFT  — 3-tap filter along top row, sliding diagonal
 *   VERT_LEFT       — even/odd 2-tap and 3-tap interleaved from top row
 *   HOR_UP          — interleaved 2-tap/3-tap from left column (inverted order)
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

/* ------------------------------------------------------------------ */
/* Phase 2A: DIAG_DOWN_LEFT                                            */
/*                                                                     */
/* dst[y][x] = filter3(top[x+y], top[x+y+1], top[x+y+2])             */
/*           = (top[x+y] + 2*top[x+y+1] + top[x+y+2] + 2) >> 2      */
/* with top[N] = top[N-1] clamped.                                     */
/* Output row y is a sliding window: result[y .. y+N-1].               */
/*                                                                     */
/* 4x4 reads top[0..7] (2N); 8x8/16x16/32x32 read top[0..N-1].       */
/* v_pad = dup(top[N-1]) supplies clamped extension.                   */
/* ------------------------------------------------------------------ */

// 4x4: top[0..7] loaded; filter3 over 8 positions with dup padding;
// ins v_result.b[6] = top[7] to match C reference (DST(3,3) = a7).
function ff_vp9_ipred_diag_downleft_4x4_neon, export=1
        ldr     d0,  [x3]                       // v0 = top[0..7]
        ldrb    w4,  [x3, #7]
        dup     v4.8b,  w4                       // v4 = all top[7]
        ext     v1.8b,  v0.8b,  v4.8b,  #1      // top[1..7, top7]
        ext     v2.8b,  v0.8b,  v4.8b,  #2      // top[2..7, top7, top7]
        movi    v5.8b,  #2
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v0.8b,  v3.8h,  #2              // v0[i] = filter3(top[i..i+2])
        ins     v0.b[6],  w4                     // v0[6] = top[7]  (not filtered)
        st1     {v0.s}[0],  [x0],  x1           // row 0: v0[0..3]
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        st1     {v1.s}[0],  [x0],  x1           // row 1: v0[1..4]
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        st1     {v2.s}[0],  [x0],  x1           // row 2: v0[2..5]
        ext     v1.8b,  v0.8b,  v4.8b,  #3
        st1     {v1.s}[0],  [x0]                // row 3: v0[3..5], top[7]
        ret
endfunc

// 8x8: v0[7] = filter3(top7,top7,top7) = top7, so no ins needed.
function ff_vp9_ipred_diag_downleft_8x8_neon, export=1
        ldr     d0,  [x3]
        ldrb    w4,  [x3, #7]
        dup     v4.8b,  w4
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        movi    v5.8b,  #2
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v0.8b,  v3.8h,  #2
        st1     {v0.8b},  [x0],  x1
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        st1     {v1.8b},  [x0],  x1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        st1     {v2.8b},  [x0],  x1
        ext     v1.8b,  v0.8b,  v4.8b,  #3
        st1     {v1.8b},  [x0],  x1
        ext     v2.8b,  v0.8b,  v4.8b,  #4
        st1     {v2.8b},  [x0],  x1
        ext     v1.8b,  v0.8b,  v4.8b,  #5
        st1     {v1.8b},  [x0],  x1
        ext     v2.8b,  v0.8b,  v4.8b,  #6
        st1     {v2.8b},  [x0],  x1
        st1     {v4.8b},  [x0]                  // row 7 = all top[7]
        ret
endfunc

function ff_vp9_ipred_diag_downleft_16x16_neon, export=1
        ld1     {v0.16b},  [x3]
        ldrb    w4,  [x3, #15]
        dup     v4.16b,  w4
        ext     v1.16b,  v0.16b,  v4.16b,  #1
        ext     v2.16b,  v0.16b,  v4.16b,  #2
        movi    v5.16b,  #2
        uaddl   v6.8h,   v0.8b,   v2.8b
        uaddl2  v7.8h,   v0.16b,  v2.16b
        umlal   v6.8h,   v1.8b,   v5.8b
        umlal2  v7.8h,   v1.16b,  v5.16b
        rshrn   v0.8b,   v6.8h,   #2
        rshrn2  v0.16b,  v7.8h,   #2
        st1     {v0.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #1
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #2
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #3
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #4
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #5
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #6
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #7
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #8
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #9
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #10
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #11
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #12
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #13
        st1     {v1.16b},  [x0],  x1
        ext     v1.16b,  v0.16b,  v4.16b,  #14
        st1     {v1.16b},  [x0],  x1
        st1     {v4.16b},  [x0]                 // row 15 = all top[15]
        ret
endfunc

// 32x32: compute 32-byte filter result on stack with 32 bytes of padding,
// then do a sliding 1-byte-per-row load loop (unaligned loads, AArch64 safe).
function ff_vp9_ipred_diag_downleft_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x3]        // top[0..31]
        ldrb    w4,  [x3, #31]
        dup     v2.16b,  w4                      // padding = all top[31]
        ext     v3.16b,  v0.16b,  v1.16b,  #1
        ext     v4.16b,  v1.16b,  v2.16b,  #1
        ext     v5.16b,  v0.16b,  v1.16b,  #2
        ext     v6.16b,  v1.16b,  v2.16b,  #2
        movi    v7.16b,  #2
        // filter3 for low 16
        uaddl   v16.8h,  v0.8b,   v5.8b
        uaddl2  v17.8h,  v0.16b,  v5.16b
        umlal   v16.8h,  v3.8b,   v7.8b
        umlal2  v17.8h,  v3.16b,  v7.16b
        rshrn   v0.8b,   v16.8h,  #2
        rshrn2  v0.16b,  v17.8h,  #2
        // filter3 for high 16
        uaddl   v16.8h,  v1.8b,   v6.8b
        uaddl2  v17.8h,  v1.16b,  v6.16b
        umlal   v16.8h,  v4.8b,   v7.8b
        umlal2  v17.8h,  v4.16b,  v7.16b
        rshrn   v1.8b,   v16.8h,  #2
        rshrn2  v1.16b,  v17.8h,  #2
        // stack: result[0..31] at sp+0, padding at sp+32..63
        sub     sp,  sp,  #64
        st1     {v0.16b,  v1.16b},  [sp]
        add     x5,  sp,  #32
        st1     {v2.16b},  [x5]
        add     x5,  x5,  #16
        st1     {v2.16b},  [x5]
        mov     x5,  sp
        mov     w6,  #32
1:      ld1     {v3.16b,  v4.16b},  [x5]
        st1     {v3.16b,  v4.16b},  [x0],  x1
        add     x5,  x5,  #1
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #64
        ret
endfunc

/* ------------------------------------------------------------------ */
/* Phase 2A: VERT_LEFT                                                 */
/*                                                                     */
/* Even rows 2j: dst[2j][x] = (top[x+j] + top[x+j+1] + 1) >> 1      */
/* Odd rows 2j+1: dst[2j+1][x] = filter3(top[x+j..x+j+2])            */
/*                                                                     */
/* ve[i] = urhadd(top[i], top[i+1])                                   */
/* vo[i] = (top[i] + 2*top[i+1] + top[i+2] + 2) >> 2                 */
/* Row 2j: ext(ve, pad, j);  Row 2j+1: ext(vo, pad, j)                */
/* ------------------------------------------------------------------ */

// 4x4: 4 rows (j=0,1 → rows 0-3). top[0..7] read for coverage.
function ff_vp9_ipred_vert_left_4x4_neon, export=1
        ldr     d0,  [x3]
        ldrb    w4,  [x3, #7]
        dup     v4.8b,  w4
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        movi    v5.8b,  #2
        urhadd  v6.8b,  v0.8b,  v1.8b           // ve[0..7]
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v7.8b,  v3.8h,  #2              // vo[0..7]
        st1     {v6.s}[0],  [x0],  x1           // row 0: ve[0..3]
        st1     {v7.s}[0],  [x0],  x1           // row 1: vo[0..3]
        ext     v0.8b,  v6.8b,  v4.8b,  #1
        st1     {v0.s}[0],  [x0],  x1           // row 2: ve[1..4]
        ext     v1.8b,  v7.8b,  v4.8b,  #1
        st1     {v1.s}[0],  [x0]                // row 3: vo[1..4]
        ret
endfunc

function ff_vp9_ipred_vert_left_8x8_neon, export=1
        ldr     d0,  [x3]
        ldrb    w4,  [x3, #7]
        dup     v4.8b,  w4
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        movi    v5.8b,  #2
        urhadd  v6.8b,  v0.8b,  v1.8b
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v7.8b,  v3.8h,  #2
        st1     {v6.8b},  [x0],  x1             // row 0 (j=0 even)
        st1     {v7.8b},  [x0],  x1             // row 1 (j=0 odd)
        ext     v0.8b,  v6.8b,  v4.8b,  #1
        ext     v1.8b,  v7.8b,  v4.8b,  #1
        st1     {v0.8b},  [x0],  x1             // row 2 (j=1 even)
        st1     {v1.8b},  [x0],  x1             // row 3 (j=1 odd)
        ext     v0.8b,  v6.8b,  v4.8b,  #2
        ext     v1.8b,  v7.8b,  v4.8b,  #2
        st1     {v0.8b},  [x0],  x1             // row 4
        st1     {v1.8b},  [x0],  x1             // row 5
        ext     v0.8b,  v6.8b,  v4.8b,  #3
        ext     v1.8b,  v7.8b,  v4.8b,  #3
        st1     {v0.8b},  [x0],  x1             // row 6
        st1     {v1.8b},  [x0]                  // row 7
        ret
endfunc

function ff_vp9_ipred_vert_left_16x16_neon, export=1
        ld1     {v0.16b},  [x3]
        ldrb    w4,  [x3, #15]
        dup     v4.16b,  w4
        ext     v1.16b,  v0.16b,  v4.16b,  #1
        ext     v2.16b,  v0.16b,  v4.16b,  #2
        movi    v5.16b,  #2
        urhadd  v6.16b,  v0.16b,  v1.16b
        uaddl   v16.8h,  v0.8b,   v2.8b
        uaddl2  v17.8h,  v0.16b,  v2.16b
        umlal   v16.8h,  v1.8b,   v5.8b
        umlal2  v17.8h,  v1.16b,  v5.16b
        rshrn   v7.8b,   v16.8h,  #2
        rshrn2  v7.16b,  v17.8h,  #2
        st1     {v6.16b},  [x0],  x1
        st1     {v7.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #1
        ext     v1.16b,  v7.16b,  v4.16b,  #1
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #2
        ext     v1.16b,  v7.16b,  v4.16b,  #2
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #3
        ext     v1.16b,  v7.16b,  v4.16b,  #3
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #4
        ext     v1.16b,  v7.16b,  v4.16b,  #4
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #5
        ext     v1.16b,  v7.16b,  v4.16b,  #5
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #6
        ext     v1.16b,  v7.16b,  v4.16b,  #6
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0],  x1
        ext     v0.16b,  v6.16b,  v4.16b,  #7
        ext     v1.16b,  v7.16b,  v4.16b,  #7
        st1     {v0.16b},  [x0],  x1
        st1     {v1.16b},  [x0]
        ret
endfunc

// 32x32: compute ve/vo (32 bytes each) on stack with 16-byte padding each;
// loop reads 32 bytes at sp+j (ve) and sp+64+j (vo) per row pair.
function ff_vp9_ipred_vert_left_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x3]
        ldrb    w4,  [x3, #31]
        dup     v2.16b,  w4
        ext     v3.16b,  v0.16b,  v1.16b,  #1
        ext     v4.16b,  v1.16b,  v2.16b,  #1
        ext     v5.16b,  v0.16b,  v1.16b,  #2
        ext     v6.16b,  v1.16b,  v2.16b,  #2
        movi    v7.16b,  #2
        // ve
        urhadd  v16.16b,  v0.16b,  v3.16b
        urhadd  v17.16b,  v1.16b,  v4.16b
        // vo
        uaddl   v18.8h,  v0.8b,   v5.8b
        uaddl2  v19.8h,  v0.16b,  v5.16b
        umlal   v18.8h,  v3.8b,   v7.8b
        umlal2  v19.8h,  v3.16b,  v7.16b
        rshrn   v20.8b,  v18.8h,  #2
        rshrn2  v20.16b, v19.8h,  #2
        uaddl   v18.8h,  v1.8b,   v6.8b
        uaddl2  v19.8h,  v1.16b,  v6.16b
        umlal   v18.8h,  v4.8b,   v7.8b
        umlal2  v19.8h,  v4.16b,  v7.16b
        rshrn   v21.8b,  v18.8h,  #2
        rshrn2  v21.16b, v19.8h,  #2
        // stack layout (128 bytes):
        //   sp+0..31:  ve[0..31]; sp+32..47:  ve-pad (16 bytes)
        //   sp+64..95: vo[0..31]; sp+96..111: vo-pad (16 bytes)
        sub     sp,  sp,  #128
        st1     {v16.16b,  v17.16b},  [sp]
        add     x5,  sp,  #32
        st1     {v2.16b},  [x5]
        add     x5,  sp,  #64
        st1     {v20.16b,  v21.16b},  [x5]
        add     x5,  sp,  #96
        st1     {v2.16b},  [x5]
        mov     x5,  sp
        add     x6,  sp,  #64
        mov     w7,  #16
1:      ld1     {v0.16b,  v1.16b},  [x5]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        ld1     {v0.16b,  v1.16b},  [x6]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        add     x5,  x5,  #1
        add     x6,  x6,  #1
        subs    w7,  w7,  #1
        b.ne    1b
        add     sp,  sp,  #128
        ret
endfunc

/* ------------------------------------------------------------------ */
/* Phase 2A: HOR_UP                                                    */
/*                                                                     */
/* IMPORTANT: left[] uses invert_left ordering: left[0]=topmost row,  */
/* left[N-1]=bottommost row (opposite of all other modes).             */
/*                                                                     */
/* v[2i]   = ve[i] = (left[i] + left[i+1] + 1) >> 1                  */
/* v[2i+1] = vo[i] = (left[i] + 2*left[i+1] + left[i+2] + 2) >> 2   */
/* with left[N] = left[N-1] clamped.                                   */
/*                                                                     */
/* Interleave: v_lo = zip1(ve, vo); v_hi = zip2(ve, vo)               */
/* Row j < N/2:  ext(v_lo, v_hi, 2j)[0..N-1]                         */
/* Row j >= N/2: ext(v_hi, v_pad, 2j-N)[0..N-1]                      */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_hor_up_4x4_neon, export=1
        // Load only 4 valid bytes; clamp l[4..7] to l[3] for filter padding.
        ldrb    w4,  [x2, #3]
        dup     v4.8b,  w4                       // v4 = all l[3]
        ldr     s0,  [x2]                        // v0[0..3]=l[0..3], v0[4..15]=0
        ins     v0.s[1],  v4.s[1]                // v0[4..7] = l[3] (clamped)
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        movi    v5.8b,  #2
        urhadd  v6.8b,  v0.8b,  v1.8b
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v7.8b,  v3.8h,  #2
        zip1    v0.8b,  v6.8b,  v7.8b           // v_lo = {ve0,vo0,ve1,vo1,...,ve3=l3,vo3=l3}
        st1     {v0.s}[0],  [x0],  x1           // row 0: v_lo[0..3]
        ext     v1.8b,  v0.8b,  v4.8b,  #2
        st1     {v1.s}[0],  [x0],  x1           // row 1: v_lo[2..5]
        ext     v1.8b,  v0.8b,  v4.8b,  #4
        st1     {v1.s}[0],  [x0],  x1           // row 2: v_lo[4..7]={ve2,vo2,l3,l3}
        st1     {v4.s}[0],  [x0]                // row 3: all l3
        ret
endfunc

function ff_vp9_ipred_hor_up_8x8_neon, export=1
        ldr     d0,  [x2]
        ldrb    w4,  [x2, #7]
        dup     v4.8b,  w4
        ext     v1.8b,  v0.8b,  v4.8b,  #1
        ext     v2.8b,  v0.8b,  v4.8b,  #2
        movi    v5.8b,  #2
        urhadd  v6.8b,  v0.8b,  v1.8b
        uaddl   v3.8h,  v0.8b,  v2.8b
        umlal   v3.8h,  v1.8b,  v5.8b
        rshrn   v7.8b,  v3.8h,  #2
        zip1    v0.8b,  v6.8b,  v7.8b           // v_lo = v[0..7]
        zip2    v1.8b,  v6.8b,  v7.8b           // v_hi = v[8..15]; v_hi[6]=l7, v_hi[7]=l7
        st1     {v0.8b},  [x0],  x1             // row 0: v[0..7]
        ext     v2.8b,  v0.8b,  v1.8b,  #2
        st1     {v2.8b},  [x0],  x1             // row 1: v[2..9]
        ext     v2.8b,  v0.8b,  v1.8b,  #4
        st1     {v2.8b},  [x0],  x1             // row 2: v[4..11]
        ext     v2.8b,  v0.8b,  v1.8b,  #6
        st1     {v2.8b},  [x0],  x1             // row 3: v[6..13]
        st1     {v1.8b},  [x0],  x1             // row 4: v[8..15]
        ext     v2.8b,  v1.8b,  v4.8b,  #2
        st1     {v2.8b},  [x0],  x1             // row 5: v[10..17]
        ext     v2.8b,  v1.8b,  v4.8b,  #4
        st1     {v2.8b},  [x0],  x1             // row 6: v[12..19]
        ext     v2.8b,  v1.8b,  v4.8b,  #6
        st1     {v2.8b},  [x0]                  // row 7: all l7
        ret
endfunc

function ff_vp9_ipred_hor_up_16x16_neon, export=1
        ld1     {v0.16b},  [x2]
        ldrb    w4,  [x2, #15]
        dup     v4.16b,  w4
        ext     v1.16b,  v0.16b,  v4.16b,  #1
        ext     v2.16b,  v0.16b,  v4.16b,  #2
        movi    v5.16b,  #2
        urhadd  v6.16b,  v0.16b,  v1.16b
        uaddl   v16.8h,  v0.8b,   v2.8b
        uaddl2  v17.8h,  v0.16b,  v2.16b
        umlal   v16.8h,  v1.8b,   v5.8b
        umlal2  v17.8h,  v1.16b,  v5.16b
        rshrn   v7.8b,   v16.8h,  #2
        rshrn2  v7.16b,  v17.8h,  #2
        zip1    v0.16b,  v6.16b,  v7.16b        // v_lo = v[0..15]
        zip2    v1.16b,  v6.16b,  v7.16b        // v_hi = v[16..31]
        st1     {v0.16b},  [x0],  x1            // row 0
        ext     v2.16b,  v0.16b,  v1.16b,  #2
        st1     {v2.16b},  [x0],  x1            // row 1
        ext     v2.16b,  v0.16b,  v1.16b,  #4
        st1     {v2.16b},  [x0],  x1            // row 2
        ext     v2.16b,  v0.16b,  v1.16b,  #6
        st1     {v2.16b},  [x0],  x1            // row 3
        ext     v2.16b,  v0.16b,  v1.16b,  #8
        st1     {v2.16b},  [x0],  x1            // row 4
        ext     v2.16b,  v0.16b,  v1.16b,  #10
        st1     {v2.16b},  [x0],  x1            // row 5
        ext     v2.16b,  v0.16b,  v1.16b,  #12
        st1     {v2.16b},  [x0],  x1            // row 6
        ext     v2.16b,  v0.16b,  v1.16b,  #14
        st1     {v2.16b},  [x0],  x1            // row 7
        st1     {v1.16b},  [x0],  x1            // row 8
        ext     v2.16b,  v1.16b,  v4.16b,  #2
        st1     {v2.16b},  [x0],  x1            // row 9
        ext     v2.16b,  v1.16b,  v4.16b,  #4
        st1     {v2.16b},  [x0],  x1            // row 10
        ext     v2.16b,  v1.16b,  v4.16b,  #6
        st1     {v2.16b},  [x0],  x1            // row 11
        ext     v2.16b,  v1.16b,  v4.16b,  #8
        st1     {v2.16b},  [x0],  x1            // row 12
        ext     v2.16b,  v1.16b,  v4.16b,  #10
        st1     {v2.16b},  [x0],  x1            // row 13
        ext     v2.16b,  v1.16b,  v4.16b,  #12
        st1     {v2.16b},  [x0],  x1            // row 14
        ext     v2.16b,  v1.16b,  v4.16b,  #14
        st1     {v2.16b},  [x0]                 // row 15: all l15
        ret
endfunc

// 32x32: compute interleaved v[0..63] on stack, advance by 2 per row.
function ff_vp9_ipred_hor_up_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x2]
        ldrb    w4,  [x2, #31]
        dup     v2.16b,  w4
        ext     v3.16b,  v0.16b,  v1.16b,  #1
        ext     v4.16b,  v1.16b,  v2.16b,  #1
        ext     v5.16b,  v0.16b,  v1.16b,  #2
        ext     v6.16b,  v1.16b,  v2.16b,  #2
        movi    v7.16b,  #2
        urhadd  v16.16b,  v0.16b,  v3.16b
        urhadd  v17.16b,  v1.16b,  v4.16b
        uaddl   v18.8h,  v0.8b,   v5.8b
        uaddl2  v19.8h,  v0.16b,  v5.16b
        umlal   v18.8h,  v3.8b,   v7.8b
        umlal2  v19.8h,  v3.16b,  v7.16b
        rshrn   v20.8b,  v18.8h,  #2
        rshrn2  v20.16b, v19.8h,  #2
        uaddl   v18.8h,  v1.8b,   v6.8b
        uaddl2  v19.8h,  v1.16b,  v6.16b
        umlal   v18.8h,  v4.8b,   v7.8b
        umlal2  v19.8h,  v4.16b,  v7.16b
        rshrn   v21.8b,  v18.8h,  #2
        rshrn2  v21.16b, v19.8h,  #2
        // interleave ve/vo into v[0..63]
        zip1    v0.16b,  v16.16b,  v20.16b      // v[0..15]
        zip2    v1.16b,  v16.16b,  v20.16b      // v[16..31]
        zip1    v3.16b,  v17.16b,  v21.16b      // v[32..47]
        zip2    v4.16b,  v17.16b,  v21.16b      // v[48..63]
        // stack: v[0..63] at sp+0..63, padding at sp+64..95
        sub     sp,  sp,  #96
        st1     {v0.16b,  v1.16b},  [sp]
        add     x5,  sp,  #32
        st1     {v3.16b,  v4.16b},  [x5]
        add     x5,  sp,  #64
        st1     {v2.16b},  [x5]
        add     x5,  x5,  #16
        st1     {v2.16b},  [x5]
        mov     x5,  sp
        mov     w6,  #32
1:      ld1     {v0.16b,  v1.16b},  [x5]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        add     x5,  x5,  #2
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #96
        ret
endfunc

/* ------------------------------------------------------------------ */
/* Phase 2B: DIAG_DOWN_RIGHT                                           */
/*                                                                     */
/* seq[i] = left[0..N-1] ++ top[-1] ++ top[0..N-1]  (2N+1 elements)  */
/* v[i]   = filter3(seq[i], seq[i+1], seq[i+2])      (2N-1 outputs)   */
/* Row j  = v[N-1-j .. N-1-j+N-1]   (window slides down by 1/row)    */
/* ------------------------------------------------------------------ */

// 4x4: seq[0..8]; v[0..6] fit in one d-register.
// Build v0=[l0,l1,l2,l3,tl,t0,t1,t2], v1=dup(t3) for ext overflow.
function ff_vp9_ipred_diag_downright_4x4_neon, export=1
        ldr     s0,  [x2]               // v0.b[0..3] = left[0..3]
        sub     x4,  x3,  #1
        ldr     s1,  [x4]               // v1.s[0] = [tl,t0,t1,t2]
        ins     v0.s[1],  v1.s[0]       // v0 = [l0,l1,l2,l3,tl,t0,t1,t2]
        ldrb    w4,  [x3, #3]           // t3
        dup     v1.8b,  w4              // padding = all t3
        ext     v2.8b,  v0.8b,  v1.8b, #1   // seq[1..8]
        ext     v3.8b,  v0.8b,  v1.8b, #2   // seq[2..9]
        movi    v4.8b,  #2
        uaddl   v5.8h,  v0.8b,  v3.8b
        umlal   v5.8h,  v2.8b,  v4.8b
        rshrn   v0.8b,  v5.8h,  #2         // v0[0..6] = v[0..6]
        // row j = ext(v0, v1, 3-j)[0..3]
        ext     v2.8b,  v0.8b,  v1.8b, #3
        st1     {v2.s}[0],  [x0],  x1      // row 0: v[3..6]
        ext     v2.8b,  v0.8b,  v1.8b, #2
        st1     {v2.s}[0],  [x0],  x1      // row 1: v[2..5]
        ext     v2.8b,  v0.8b,  v1.8b, #1
        st1     {v2.s}[0],  [x0],  x1      // row 2: v[1..4]
        st1     {v0.s}[0],  [x0]           // row 3: v[0..3]
        ret
endfunc

// 8x8: v_lo=v[0..7], v_hi=v[8..14+junk]; row j = ext(v_lo,v_hi,7-j)
function ff_vp9_ipred_diag_downright_8x8_neon, export=1
        ldr     d0,  [x2]               // left[0..7]
        sub     x4,  x3,  #1
        ldr     d1,  [x4]               // [tl,t0..t6]
        ldrb    w4,  [x3, #7]           // t7
        dup     v2.8b,  w4              // padding
        movi    v5.8b,  #2
        ext     v3.8b,  v0.8b,  v1.8b, #1
        ext     v4.8b,  v0.8b,  v1.8b, #2
        uaddl   v6.8h,  v0.8b,  v4.8b
        umlal   v6.8h,  v3.8b,  v5.8b
        rshrn   v0.8b,  v6.8h,  #2         // v_lo = v[0..7]
        ext     v3.8b,  v1.8b,  v2.8b, #1
        ext     v4.8b,  v1.8b,  v2.8b, #2
        uaddl   v6.8h,  v1.8b,  v4.8b
        umlal   v6.8h,  v3.8b,  v5.8b
        rshrn   v1.8b,  v6.8h,  #2         // v_hi = v[8..14+junk]
        ext     v3.8b,  v0.8b,  v1.8b, #7
        st1     {v3.8b},  [x0],  x1        // row 0: v[7..14]
        ext     v3.8b,  v0.8b,  v1.8b, #6
        st1     {v3.8b},  [x0],  x1        // row 1: v[6..13]
        ext     v3.8b,  v0.8b,  v1.8b, #5
        st1     {v3.8b},  [x0],  x1        // row 2: v[5..12]
        ext     v3.8b,  v0.8b,  v1.8b, #4
        st1     {v3.8b},  [x0],  x1        // row 3: v[4..11]
        ext     v3.8b,  v0.8b,  v1.8b, #3
        st1     {v3.8b},  [x0],  x1        // row 4: v[3..10]
        ext     v3.8b,  v0.8b,  v1.8b, #2
        st1     {v3.8b},  [x0],  x1        // row 5: v[2..9]
        ext     v3.8b,  v0.8b,  v1.8b, #1
        st1     {v3.8b},  [x0],  x1        // row 6: v[1..8]
        st1     {v0.8b},  [x0]             // row 7: v[0..7]
        ret
endfunc

// 16x16: v_lo=v[0..15], v_hi=v[16..30+junk]; row j = ext(v_lo,v_hi,15-j)
function ff_vp9_ipred_diag_downright_16x16_neon, export=1
        ld1     {v0.16b},  [x2]         // left[0..15]
        sub     x4,  x3,  #1
        ld1     {v1.16b},  [x4]         // [tl,t0..t14]
        ldrb    w4,  [x3, #15]          // t15
        dup     v2.16b,  w4             // padding
        movi    v5.16b,  #2
        ext     v3.16b,  v0.16b,  v1.16b, #1
        ext     v4.16b,  v0.16b,  v1.16b, #2
        uaddl   v6.8h,   v0.8b,   v4.8b
        uaddl2  v7.8h,   v0.16b,  v4.16b
        umlal   v6.8h,   v3.8b,   v5.8b
        umlal2  v7.8h,   v3.16b,  v5.16b
        rshrn   v0.8b,   v6.8h,   #2
        rshrn2  v0.16b,  v7.8h,   #2       // v_lo = v[0..15]
        ext     v3.16b,  v1.16b,  v2.16b, #1
        ext     v4.16b,  v1.16b,  v2.16b, #2
        uaddl   v6.8h,   v1.8b,   v4.8b
        uaddl2  v7.8h,   v1.16b,  v4.16b
        umlal   v6.8h,   v3.8b,   v5.8b
        umlal2  v7.8h,   v3.16b,  v5.16b
        rshrn   v1.8b,   v6.8h,   #2
        rshrn2  v1.16b,  v7.8h,   #2       // v_hi = v[16..30+junk]
        ext     v3.16b,  v0.16b,  v1.16b, #15
        st1     {v3.16b},  [x0],  x1       // row 0: v[15..30]
        ext     v3.16b,  v0.16b,  v1.16b, #14
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #13
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #12
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #11
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #10
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #9
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #8
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #7
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #6
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #5
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #4
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #3
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #2
        st1     {v3.16b},  [x0],  x1
        ext     v3.16b,  v0.16b,  v1.16b, #1
        st1     {v3.16b},  [x0],  x1
        st1     {v0.16b},  [x0]            // row 15: v[0..15]
        ret
endfunc

// 32x32: stack holds v[0..63] (four 16B blocks).
// Row j reads 32B from sp+(31-j); loop: x5=sp+31, decrement by 1.
function ff_vp9_ipred_diag_downright_32x32_neon, export=1
        ld1     {v0.16b,  v1.16b},  [x2]    // left[0..31]
        sub     x4,  x3,  #1
        ld1     {v2.16b},  [x4]             // [tl,t0..t14]
        add     x4,  x3,  #15
        ld1     {v3.16b},  [x4]             // [t15..t30]
        ldrb    w4,  [x3, #31]              // t31
        dup     v4.16b,  w4                 // padding
        movi    v7.16b,  #2
        sub     sp,  sp,  #64
        // v_A = v[0..15]
        ext     v5.16b,  v0.16b,  v1.16b, #1
        ext     v6.16b,  v0.16b,  v1.16b, #2
        uaddl   v16.8h,  v0.8b,   v6.8b
        uaddl2  v17.8h,  v0.16b,  v6.16b
        umlal   v16.8h,  v5.8b,   v7.8b
        umlal2  v17.8h,  v5.16b,  v7.16b
        rshrn   v20.8b,  v16.8h,  #2
        rshrn2  v20.16b, v17.8h,  #2
        st1     {v20.16b},  [sp]
        // v_B = v[16..31]
        ext     v5.16b,  v1.16b,  v2.16b, #1
        ext     v6.16b,  v1.16b,  v2.16b, #2
        uaddl   v16.8h,  v1.8b,   v6.8b
        uaddl2  v17.8h,  v1.16b,  v6.16b
        umlal   v16.8h,  v5.8b,   v7.8b
        umlal2  v17.8h,  v5.16b,  v7.16b
        rshrn   v20.8b,  v16.8h,  #2
        rshrn2  v20.16b, v17.8h,  #2
        add     x5,  sp,  #16
        st1     {v20.16b},  [x5]
        // v_C = v[32..47]
        ext     v5.16b,  v2.16b,  v3.16b, #1
        ext     v6.16b,  v2.16b,  v3.16b, #2
        uaddl   v16.8h,  v2.8b,   v6.8b
        uaddl2  v17.8h,  v2.16b,  v6.16b
        umlal   v16.8h,  v5.8b,   v7.8b
        umlal2  v17.8h,  v5.16b,  v7.16b
        rshrn   v20.8b,  v16.8h,  #2
        rshrn2  v20.16b, v17.8h,  #2
        add     x5,  sp,  #32
        st1     {v20.16b},  [x5]
        // v_D = v[48..62+junk]
        ext     v5.16b,  v3.16b,  v4.16b, #1
        ext     v6.16b,  v3.16b,  v4.16b, #2
        uaddl   v16.8h,  v3.8b,   v6.8b
        uaddl2  v17.8h,  v3.16b,  v6.16b
        umlal   v16.8h,  v5.8b,   v7.8b
        umlal2  v17.8h,  v5.16b,  v7.16b
        rshrn   v20.8b,  v16.8h,  #2
        rshrn2  v20.16b, v17.8h,  #2
        add     x5,  sp,  #48
        st1     {v20.16b},  [x5]
        // loop: row 0 reads sp+31..sp+62, row 31 reads sp+0..sp+31
        add     x5,  sp,  #31
        mov     w6,  #32
1:      ld1     {v0.16b,  v1.16b},  [x5]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        sub     x5,  x5,  #1
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #64
        ret
endfunc

/* Phase 2B: VERT_RIGHT                                                */
/*                                                                     */
/* ve[i]: urhadd pairs from left col then top row                      */
/* vo[i]: filter3 triples from left col then top row                   */
/* Row 2j   = ve[N/2-1-j .. N/2-1-j+N-1]                             */
/* Row 2j+1 = vo[N/2-1-j .. N/2-1-j+N-1]                             */
/* ------------------------------------------------------------------ */

function ff_vp9_ipred_vert_right_4x4_neon, export=1
        ldrb    w4,  [x2, #3]
        sub     x5,  x3,  #1
        ldrb    w5,  [x5]
        ldrb    w6,  [x3]
        ldrb    w7,  [x3, #1]
        ldrb    w8,  [x3, #2]
        ldrb    w9,  [x3, #3]
        ins     v0.b[0], w4
        ins     v0.b[1], w5
        ins     v0.b[2], w6
        ins     v0.b[3], w7
        ins     v0.b[4], w8
        ins     v0.b[5], w9
        ext     v1.8b,  v0.8b,  v0.8b,  #1
        ext     v2.8b,  v0.8b,  v0.8b,  #2
        urhadd  v3.8b,  v1.8b,  v2.8b          // ve[1..4] = urhadd(seq[1..4],seq[2..5])
        movi    v4.8b,  #2
        uaddl   v5.8h,  v0.8b,  v2.8b
        umlal   v5.8h,  v1.8b,  v4.8b
        rshrn   v6.8b,  v5.8h,  #2             // vo[1..4]
        st1     {v3.s}[0],  [x0],  x1          // row 0
        st1     {v6.s}[0],  [x0],  x1          // row 1
        ldrb    w7,  [x2, #2]
        add     w10, w7,  w5
        lsl     w11, w4,  #1
        add     w10, w10, w11
        add     w10, w10, #2
        lsr     w10, w10, #2                    // ve[0]
        ldrb    w11, [x2, #1]
        add     w12, w11, w4
        lsl     w13, w7,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2                    // vo[0]
        ins     v3.b[3],  v3.b[2]
        ins     v3.b[2],  v3.b[1]
        ins     v3.b[1],  v3.b[0]
        ins     v3.b[0],  w10
        st1     {v3.s}[0],  [x0],  x1          // row 2: ve[0..3]
        ins     v6.b[3],  v6.b[2]
        ins     v6.b[2],  v6.b[1]
        ins     v6.b[1],  v6.b[0]
        ins     v6.b[0],  w12
        st1     {v6.s}[0],  [x0]               // row 3: vo[0..3]
        ret
endfunc

function ff_vp9_ipred_vert_right_8x8_neon, export=1
        sub     sp,  sp,  #32
        ldrb    w4,  [x2, #1]
        ldrb    w5,  [x2, #2]
        ldrb    w6,  [x2, #3]
        ldrb    w7,  [x2, #4]
        ldrb    w8,  [x2, #5]
        ldrb    w9,  [x2, #6]
        ldrb    w10, [x2, #7]
        sub     x12, x3,  #1
        ldrb    w11, [x12]
        add     w12, w4,  w6
        lsl     w13, w5,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #16]
        add     w12, w6,  w8
        lsl     w13, w7,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #17]
        add     w12, w8,  w10
        lsl     w13, w9,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #18]
        add     w12, w5,  w7
        lsl     w13, w6,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #0]
        add     w12, w7,  w9
        lsl     w13, w8,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #1]
        add     w12, w9,  w11
        lsl     w13, w10, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [sp, #2]
        // ve[3..10] = urhadd([tl,t0..t6], [t0..t7])
        sub     x12, x3,  #1
        ld1     {v0.8b},  [x12]                 // [tl,t0..t6]
        ld1     {v1.8b},  [x3]                  // [t0..t7]
        urhadd  v2.8b,  v0.8b,  v1.8b           // ve[3..10] ✓
        add     x12, sp,  #3
        st1     {v2.8b},  [x12]
        // vo[3] = f3(l7, tl, t0) scalar
        umov    w12, v1.b[0]                    // t0
        lsl     w13, w11, #1                    // 2*tl
        add     w12, w12, w10                   // t0+l7
        add     w12, w12, w13                   // t0+l7+2*tl
        add     w12, w12, #2
        lsr     w12, w12, #2                    // vo[3]
        add     x13, sp,  #19
        strb    w12, [x13]
        // vo[4..10] = filter3([tl,t0..t6],[t0..t7],[t1..t7,junk])
        // v0=[tl,t0..t6], v1=[t0..t7]
        ext     v3.8b,  v1.8b,  v1.8b,  #1     // c=[t1..t7,t0_wrap]
        movi    v4.8b,  #2
        uaddl   v5.8h,  v0.8b,  v3.8b           // a+c=[tl+t1,t0+t2,...,t6+t0_wrap]
        umlal   v5.8h,  v1.8b,  v4.8b           // +2b=[+2t0,...,+2t7]
        rshrn   v6.8b,  v5.8h,  #2              // vo[4..10] in [0..6], garbage at [7]
        add     x12, sp,  #20
        st1     {v6.8b},  [x12]                 // sp[20..26]=vo[4..10], sp[27]=junk
        add     x4,  sp,  #3
        add     x5,  sp,  #19
        mov     w6,  #4
1:      ld1     {v0.8b},  [x4]
        st1     {v0.8b},  [x0],  x1
        ld1     {v1.8b},  [x5]
        st1     {v1.8b},  [x0],  x1
        sub     x4,  x4,  #1
        sub     x5,  x5,  #1
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #32
        ret
endfunc

// Stack: sp+0..6=ve[0..6], sp+7=ve[7], sp+8..23=ve[8..22+junk]
//        sp+24..30=vo[0..6], sp+31=vo[7], sp+32..47=vo[8..22+junk]
// ve[8..22]=urhadd(top[0..14],top[1..15])  (NOT starting from tl)
// vo[8..22]=filter3([tl,t0..t14],[t0..t15],[t1..t15,junk])
function ff_vp9_ipred_vert_right_16x16_neon, export=1
        sub     sp,  sp,  #48
        add     x4,  x2,  #1
        add     x5,  sp,  #0
        add     x6,  sp,  #24
        mov     w7,  #6
1:      ldrb    w8,  [x4]
        ldrb    w9,  [x4, #1]
        ldrb    w10, [x4, #2]
        add     w11, w8,  w10
        lsl     w12, w9,  #1
        add     w11, w11, w12
        add     w11, w11, #2
        lsr     w11, w11, #2
        strb    w11, [x6], #1
        ldrb    w11, [x4, #3]
        add     w12, w9,  w11
        lsl     w13, w10, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x5], #1
        add     x4,  x4,  #2
        subs    w7,  w7,  #1
        b.ne    1b
        ldrb    w8,  [x2, #13]
        ldrb    w9,  [x2, #14]
        ldrb    w10, [x2, #15]
        sub     x11, x3,  #1
        ldrb    w11, [x11]
        add     w12, w8,  w10
        lsl     w13, w9,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x6]
        add     w12, w9,  w11
        lsl     w13, w10, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x5]
        ldrb    w8,  [x3]
        add     w9,  w11, w8
        add     w9,  w9,  #1
        lsr     w9,  w9,  #1
        add     x12, sp,  #7
        strb    w9,  [x12]
        add     w12, w10, w8
        lsl     w13, w11, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        add     x13, sp,  #31
        strb    w12, [x13]
        // ve[8..22] = urhadd(top[0..14], top[1..15])
        ld1     {v0.16b},  [x3]
        ext     v1.16b,  v0.16b,  v0.16b,  #1   // [t1..t15,t0_junk]
        urhadd  v2.16b,  v0.16b,  v1.16b         // correct: urhadd(t0..t14, t1..t15)
        add     x12, sp,  #8
        st1     {v2.16b},  [x12]
        // vo[8..22] = filter3([tl,t0..t14],[t0..t15],[t1..t15,junk])
        // a=v3=[tl,t0..t14], b=v0=[t0..t15], c=v1=[t1..t15,junk]
        sub     x12, x3,  #1
        ld1     {v3.16b},  [x12]
        movi    v5.16b,  #2
        uaddl   v6.8h,   v3.8b,   v1.8b
        uaddl2  v7.8h,   v3.16b,  v1.16b
        umlal   v6.8h,   v0.8b,   v5.8b
        umlal2  v7.8h,   v0.16b,  v5.16b
        rshrn   v6.8b,   v6.8h,   #2
        rshrn2  v6.16b,  v7.8h,   #2
        add     x12, sp,  #32
        st1     {v6.16b},  [x12]
        add     x4,  sp,  #7
        add     x5,  sp,  #31
        mov     w6,  #8
1:      ld1     {v0.16b},  [x4]
        st1     {v0.16b},  [x0],  x1
        ld1     {v1.16b},  [x5]
        st1     {v1.16b},  [x0],  x1
        sub     x4,  x4,  #1
        sub     x5,  x5,  #1
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #48
        ret
endfunc

// Stack: sp+0..14=ve[0..14], sp+15=ve[15], sp+16..31=ve[16..31], sp+32..47=ve[32..46+junk]
//        sp+48..62=vo[0..14], sp+63=vo[15], sp+64..79=vo[16..31], sp+80..95=vo[32..46+junk]
function ff_vp9_ipred_vert_right_32x32_neon, export=1
        sub     sp,  sp,  #96
        add     x4,  x2,  #1
        add     x5,  sp,  #0
        add     x6,  sp,  #48
        mov     w7,  #14
1:      ldrb    w8,  [x4]
        ldrb    w9,  [x4, #1]
        ldrb    w10, [x4, #2]
        add     w11, w8,  w10
        lsl     w12, w9,  #1
        add     w11, w11, w12
        add     w11, w11, #2
        lsr     w11, w11, #2
        strb    w11, [x6], #1
        ldrb    w11, [x4, #3]
        add     w12, w9,  w11
        lsl     w13, w10, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x5], #1
        add     x4,  x4,  #2
        subs    w7,  w7,  #1
        b.ne    1b
        ldrb    w8,  [x2, #29]
        ldrb    w9,  [x2, #30]
        ldrb    w10, [x2, #31]
        sub     x11, x3,  #1
        ldrb    w11, [x11]
        add     w12, w8,  w10
        lsl     w13, w9,  #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x6]
        add     w12, w9,  w11
        lsl     w13, w10, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        strb    w12, [x5]
        ldrb    w8,  [x3]
        add     w9,  w11, w8
        add     w9,  w9,  #1
        lsr     w9,  w9,  #1
        add     x12, sp,  #15
        strb    w9,  [x12]
        add     w12, w10, w8
        lsl     w13, w11, #1
        add     w12, w12, w13
        add     w12, w12, #2
        lsr     w12, w12, #2
        add     x13, sp,  #63
        strb    w12, [x13]
        // ve chunk1: urhadd(top[0..15],top[1..16])
        ld1     {v0.16b},  [x3]
        add     x12, x3,  #1
        ld1     {v1.16b},  [x12]
        urhadd  v2.16b,  v0.16b,  v1.16b
        add     x12, sp,  #16
        st1     {v2.16b},  [x12]
        // vo chunk1: filter3([tl,t0..t14],[t0..t15],[t1..t16])
        sub     x12, x3,  #1
        ld1     {v3.16b},  [x12]
        movi    v5.16b,  #2
        uaddl   v6.8h,   v3.8b,   v1.8b
        uaddl2  v7.8h,   v3.16b,  v1.16b
        umlal   v6.8h,   v0.8b,   v5.8b
        umlal2  v7.8h,   v0.16b,  v5.16b
        rshrn   v6.8b,   v6.8h,   #2
        rshrn2  v6.16b,  v7.8h,   #2
        add     x12, sp,  #64
        st1     {v6.16b},  [x12]
        // ve chunk2: urhadd(top[16..30],top[17..31])
        add     x12, x3,  #16
        ld1     {v0.16b},  [x12]               // [t16..t31]
        ext     v1.16b,  v0.16b,  v0.16b,  #1  // [t17..t31,t16_junk]
        urhadd  v2.16b,  v0.16b,  v1.16b
        add     x12, sp,  #32
        st1     {v2.16b},  [x12]
        // vo chunk2: filter3([t15..t29],[t16..t30],[t17..t31,junk])
        add     x12, x3,  #15
        ld1     {v3.16b},  [x12]               // [t15..t30]
        uaddl   v6.8h,   v3.8b,   v1.8b
        uaddl2  v7.8h,   v3.16b,  v1.16b
        umlal   v6.8h,   v0.8b,   v5.8b
        umlal2  v7.8h,   v0.16b,  v5.16b
        rshrn   v6.8b,   v6.8h,   #2
        rshrn2  v6.16b,  v7.8h,   #2
        add     x12, sp,  #80
        st1     {v6.16b},  [x12]
        add     x4,  sp,  #15
        add     x5,  sp,  #63
        mov     w6,  #16
1:      ld1     {v0.16b,  v1.16b},  [x4]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        ld1     {v2.16b,  v3.16b},  [x5]
        st1     {v2.16b,  v3.16b},  [x0],  x1
        sub     x4,  x4,  #1
        sub     x5,  x5,  #1
        subs    w6,  w6,  #1
        b.ne    1b
        add     sp,  sp,  #96
        ret
endfunc

/* ------------------------------------------------------------------ */
/* Phase 2B: HOR_DOWN   (v0-v7 only; v16-v20 for 32x32)               */
/* ------------------------------------------------------------------ */

// 4x4: seq=[l0,l1,l2,l3,tl,t0,t1,t2]. v0-v7 only.
function ff_vp9_ipred_hor_down_4x4_neon, export=1
        ldr     s0,  [x2]
        sub     x4,  x3,  #1
        ldrb    w4,  [x4]
        ins     v0.b[4],  w4
        ldrb    w4,  [x3]
        ins     v0.b[5],  w4
        ldrb    w4,  [x3, #1]
        ins     v0.b[6],  w4
        ldrb    w4,  [x3, #2]
        ins     v0.b[7],  w4
        ext     v1.8b,  v0.8b,  v0.8b,  #1
        ext     v2.8b,  v0.8b,  v0.8b,  #2
        urhadd  v3.8b,  v0.8b,  v1.8b          // [v0,v2,v4,v6]
        movi    v4.8b,  #2
        uaddl   v5.8h,  v0.8b,  v2.8b
        umlal   v5.8h,  v1.8b,  v4.8b
        rshrn   v6.8b,  v5.8h,  #2             // [v1,v3,v5,v7,v8,v9,j,j]
        zip1    v7.8b,  v3.8b,  v6.8b          // v[0..7]
        ext     v0.8b,  v6.8b,  v6.8b,  #4     // [v8,v9,j,j,...] reuse v0
        ext     v1.8b,  v7.8b,  v0.8b,  #6     // reuse v1
        st1     {v1.s}[0],  [x0],  x1
        ext     v1.8b,  v7.8b,  v0.8b,  #4
        st1     {v1.s}[0],  [x0],  x1
        ext     v1.8b,  v7.8b,  v0.8b,  #2
        st1     {v1.s}[0],  [x0],  x1
        st1     {v7.s}[0],  [x0]
        ret
endfunc

// 8x8: stack 32 bytes (16-byte aligned). v0-v7 only.
// Lower chunk first (needs seq_lo=v0 and seq_hi=v3), then upper.
function ff_vp9_ipred_hor_down_8x8_neon, export=1
        sub     sp,  sp,  #32
        ld1     {v0.8b},  [x2]          // seq_lo
        ld1     {v1.8b},  [x3]          // [t0..t7] — used only to build seq_hi
        sub     x4,  x3,  #1
        ldrb    w4,  [x4]
        dup     v2.8b,  w4
        ext     v3.8b,  v2.8b,  v1.8b,  #7    // seq_hi=[tl,t0..t6]; v1,v2 now free
        movi    v7.8b,  #2
        // Lower chunk
        ext     v4.8b,  v0.8b,  v3.8b,  #1
        ext     v5.8b,  v0.8b,  v3.8b,  #2
        urhadd  v6.8b,  v0.8b,  v4.8b
        uaddl   v1.8h,  v0.8b,  v5.8b         // reuse v1 as 8H
        umlal   v1.8h,  v4.8b,  v7.8b
        rshrn   v1.8b,  v1.8h,  #2            // f3 in v1
        zip1    v0.8b,  v6.8b,  v1.8b         // [v0..v7]
        zip2    v1.8b,  v6.8b,  v1.8b         // [v8..v15] (zip2 reads old v1=f3)
        st1     {v0.8b},  [sp]
        add     x4,  sp,  #8
        st1     {v1.8b},  [x4]
        // Upper chunk from seq_hi=v3
        ext     v4.8b,  v3.8b,  v3.8b,  #1
        ext     v5.8b,  v3.8b,  v3.8b,  #2
        uaddl   v1.8h,  v3.8b,  v5.8b
        umlal   v1.8h,  v4.8b,  v7.8b
        rshrn   v2.8b,  v1.8h,  #2
        add     x4,  sp,  #16
        st1     {v2.8b},  [x4]
        // Loop: ptr=sp+14, step -2, 8 iters
        add     x4,  sp,  #14
        mov     w5,  #8
1:      ld1     {v0.8b},  [x4]
        st1     {v0.8b},  [x0],  x1
        sub     x4,  x4,  #2
        subs    w5,  w5,  #1
        b.ne    1b
        add     sp,  sp,  #32
        ret
endfunc

// 16x16: stack 48 bytes. v0-v7 only (v1,v2 as 8H temps after building seq_hi).
function ff_vp9_ipred_hor_down_16x16_neon, export=1
        sub     sp,  sp,  #48
        ld1     {v0.16b},  [x2]
        sub     x4,  x3,  #1
        ldrb    w4,  [x4]
        dup     v2.16b,  w4
        ld1     {v1.16b},  [x3]
        ext     v3.16b,  v2.16b,  v1.16b,  #15  // seq_hi; v1,v2 now free
        movi    v7.16b,  #2
        // Lower chunk
        ext     v4.16b,  v0.16b,  v3.16b,  #1
        ext     v5.16b,  v0.16b,  v3.16b,  #2
        urhadd  v6.16b,  v0.16b,  v4.16b
        uaddl   v1.8h,   v0.8b,   v5.8b
        uaddl2  v2.8h,   v0.16b,  v5.16b
        umlal   v1.8h,   v4.8b,   v7.8b
        umlal2  v2.8h,   v4.16b,  v7.16b
        rshrn   v1.8b,   v1.8h,   #2
        rshrn2  v1.16b,  v2.8h,   #2
        zip1    v0.16b,  v6.16b,  v1.16b
        zip2    v1.16b,  v6.16b,  v1.16b
        st1     {v0.16b},  [sp]
        add     x4,  sp,  #16
        st1     {v1.16b},  [x4]
        // Upper chunk from seq_hi=v3
        ext     v4.16b,  v3.16b,  v3.16b,  #1
        ext     v5.16b,  v3.16b,  v3.16b,  #2
        uaddl   v1.8h,   v3.8b,   v5.8b
        uaddl2  v2.8h,   v3.16b,  v5.16b
        umlal   v1.8h,   v4.8b,   v7.8b
        umlal2  v2.8h,   v4.16b,  v7.16b
        rshrn   v1.8b,   v1.8h,   #2
        rshrn2  v1.16b,  v2.8h,   #2
        add     x4,  sp,  #32
        st1     {v1.16b},  [x4]
        // Loop: ptr=sp+30, step -2, 16 iters
        add     x4,  sp,  #30
        mov     w5,  #16
1:      ld1     {v0.16b},  [x4]
        st1     {v0.16b},  [x0],  x1
        sub     x4,  x4,  #2
        subs    w5,  w5,  #1
        b.ne    1b
        add     sp,  sp,  #48
        ret
endfunc

// 32x32: stack 96 bytes. v0-v7 for seqs/exts; v2,v3 as 8H temps; v16-v20 for chunk results.
// (v16-v20 are caller-saved; no callee-saved regs touched)
function ff_vp9_ipred_hor_down_32x32_neon, export=1
        sub     sp,  sp,  #96
        ld1     {v0.16b},  [x2]
        add     x4,  x2,  #16
        ld1     {v1.16b},  [x4]
        sub     x4,  x3,  #1
        ldrb    w4,  [x4]
        dup     v2.16b,  w4
        ld1     {v3.16b},  [x3]
        ext     v4.16b,  v2.16b,  v3.16b,  #15  // seq_c=[tl,t0..t14]; v2,v3 now free as 8H temps
        add     x4,  x3,  #15
        ld1     {v5.16b},  [x4]                  // seq_d=[t15..t30]
        movi    v16.16b, #2
        // Chunk A: seq_a=v0, seq_b=v1
        ext     v6.16b,  v0.16b,  v1.16b,  #1
        ext     v7.16b,  v0.16b,  v1.16b,  #2
        urhadd  v17.16b, v0.16b,  v6.16b
        uaddl   v2.8h,   v0.8b,   v7.8b
        uaddl2  v3.8h,   v0.16b,  v7.16b
        umlal   v2.8h,   v6.8b,   v16.8b
        umlal2  v3.8h,   v6.16b,  v16.16b
        rshrn   v18.8b,  v2.8h,   #2
        rshrn2  v18.16b, v3.8h,   #2
        zip1    v19.16b, v17.16b, v18.16b
        zip2    v20.16b, v17.16b, v18.16b
        st1     {v19.16b},  [sp]
        add     x4,  sp,  #16
        st1     {v20.16b},  [x4]
        // Chunk B: seq_b=v1, seq_c=v4
        ext     v6.16b,  v1.16b,  v4.16b,  #1
        ext     v7.16b,  v1.16b,  v4.16b,  #2
        urhadd  v17.16b, v1.16b,  v6.16b
        uaddl   v2.8h,   v1.8b,   v7.8b
        uaddl2  v3.8h,   v1.16b,  v7.16b
        umlal   v2.8h,   v6.8b,   v16.8b
        umlal2  v3.8h,   v6.16b,  v16.16b
        rshrn   v18.8b,  v2.8h,   #2
        rshrn2  v18.16b, v3.8h,   #2
        zip1    v19.16b, v17.16b, v18.16b
        zip2    v20.16b, v17.16b, v18.16b
        add     x4,  sp,  #32
        st1     {v19.16b},  [x4]
        add     x4,  sp,  #48
        st1     {v20.16b},  [x4]
        // Chunk C: seq_c=v4, seq_d=v5
        ext     v6.16b,  v4.16b,  v5.16b,  #1
        ext     v7.16b,  v4.16b,  v5.16b,  #2
        uaddl   v2.8h,   v4.8b,   v7.8b
        uaddl2  v3.8h,   v4.16b,  v7.16b
        umlal   v2.8h,   v6.8b,   v16.8b
        umlal2  v3.8h,   v6.16b,  v16.16b
        rshrn   v2.8b,   v2.8h,   #2
        rshrn2  v2.16b,  v3.8h,   #2
        add     x4,  sp,  #64
        st1     {v2.16b},  [x4]
        // Chunk D: seq_d=v5 rotations
        ext     v6.16b,  v5.16b,  v5.16b,  #1
        ext     v7.16b,  v5.16b,  v5.16b,  #2
        uaddl   v2.8h,   v5.8b,   v7.8b
        uaddl2  v3.8h,   v5.16b,  v7.16b
        umlal   v2.8h,   v6.8b,   v16.8b
        umlal2  v3.8h,   v6.16b,  v16.16b
        rshrn   v2.8b,   v2.8h,   #2
        rshrn2  v2.16b,  v3.8h,   #2
        add     x4,  sp,  #80
        st1     {v2.16b},  [x4]
        // Loop: ptr=sp+62, step -2, 32 iters
        add     x4,  sp,  #62
        mov     w5,  #32
1:      ld1     {v0.16b,  v1.16b},  [x4]
        st1     {v0.16b,  v1.16b},  [x0],  x1
        sub     x4,  x4,  #2
        subs    w5,  w5,  #1
        b.ne    1b
        add     sp,  sp,  #96
        ret
endfunc
