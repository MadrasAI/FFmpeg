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
