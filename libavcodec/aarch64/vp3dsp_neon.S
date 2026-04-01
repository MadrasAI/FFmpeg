/*
 * Copyright (c) 2009 David Conrad
 * AArch64 NEON port (c) 2025 FFmpeg contributors
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
#include "neon.S"

/* IDCT constants: C1S7 C2S6 C3S5 C4S4 C5S3 C6S2 C7S1 */
const   vp3_idct_constants, align=4
.short 64277, 60547, 54491, 46341, 36410, 25080, 12785
endconst

/*
 * AArch64 register mapping vs ARM32 original (libavcodec/arm/vp3dsp_neon.S):
 *   ARM32 q8-q15  →  AArch64 v16-v23  (all caller-saved, no push/pop needed)
 *   ARM32 q0-q7   →  AArch64 v0-v7    (same)
 *   v0.8h         = IDCT constants (same as d0/d1 in ARM32)
 *   v16..v23      = row/column data
 *   v24, v25      = transpose scratch
 *   x10           = saved lr across subroutine chains
 */

/* ------------------------------------------------------------------ */
/* vp3_loop_filter macro                                               */
/*   Inputs:  v16-v19 = rows [−2, −1, 0, +1] as uint8              */
/*            w2 = filter_limit*2 (from bounding_values[129] low byte) */
/*   Outputs: v0.8b = new row[−1],  v1.8b = new row[0]              */
/* ------------------------------------------------------------------ */
.macro vp3_loop_filter
    usubl       v3.8h,  v18.8b, v17.8b  // row[0]  − row[−1]  (unsigned widen)
    usubl       v2.8h,  v16.8b, v19.8b  // row[−2] − row[+1]
    add         v1.8h,  v3.8h,  v3.8h   // 2*(row[0]−row[−1])
    add         v2.8h,  v2.8h,  v3.8h   // + (row[−2]−row[+1])
    add         v0.8h,  v1.8h,  v2.8h   // 3*(row[0]−row[−1]) + (row[−2]−row[+1])
    srshr       v0.8h,  v0.8h,  #3      // (filter_value + 4) >> 3
    uxtl        v5.8h,  v18.8b          // row[0] widened for final add
    dup         v6.8h,  w2              // filter_limit*2 replicated

    abs         v1.8h,  v0.8h           // |filter_value|
    sshr        v0.8h,  v0.8h,  #15     // sign mask: 0 if ≥0, −1 if <0
    uqsub       v2.8h,  v6.8h,  v1.8h  // max(0, limit − |fv|)
    uqsub       v3.8h,  v2.8h,  v1.8h  // max(0, limit − 2|fv|)
    sub         v1.8h,  v2.8h,  v3.8h  // min(|fv|, limit − |fv|)
    eor         v1.16b, v1.16b, v0.16b // conditionally negate
    sub         v0.8h,  v1.8h,  v0.8h  // restore sign → bounding_values result

    uaddw       v2.8h,  v0.8h,  v17.8b // row[−1] + filter_value
    sub         v3.8h,  v5.8h,  v0.8h  // row[0]  − filter_value
    sqxtun      v0.8b,  v2.8h          // saturate/narrow new row[−1]
    sqxtun      v1.8b,  v3.8h          // saturate/narrow new row[0]
.endm

/* ------------------------------------------------------------------ */
/* ff_vp3_v_loop_filter_neon(uint8_t *src, ptrdiff_t stride, int *bv) */
/* ------------------------------------------------------------------ */
function ff_vp3_v_loop_filter_neon, export=1
    sub         x12, x0,  x1           // x12 = src−stride  (write ptr row[−1])
    sub         x0,  x0,  x1,  lsl #1 // x0  = src−2*stride
    ld1         {v16.8b}, [x0], x1
    ld1         {v17.8b}, [x0], x1
    ld1         {v18.8b}, [x0], x1
    ld1         {v19.8b}, [x0], x1
    ldr         w2,  [x2, #516]        // bv[129] = filter_limit * 0x02020202
    uxtb        w2,  w2                // low byte = filter_limit * 2

    vp3_loop_filter

    st1         {v0.8b}, [x12], x1
    st1         {v1.8b}, [x12], x1
    ret
endfunc

/* ------------------------------------------------------------------ */
/* ff_vp3_h_loop_filter_neon(uint8_t *src, ptrdiff_t stride, int *bv) */
/* ------------------------------------------------------------------ */
function ff_vp3_h_loop_filter_neon, export=1
    sub         x12, x0,  #1           // write ptr: src−1
    sub         x0,  x0,  #2           // read ptr:  src−2

    /* Load 8 rows × 4 bytes into v16−v19 using ld1r (dup 32-bit) then
     * overwrite the upper 32-bit element for rows 4−7.
     *   v16: [row0[0..3], row4[0..3]]
     *   v17: [row1[0..3], row5[0..3]]
     *   v18: [row2[0..3], row6[0..3]]
     *   v19: [row3[0..3], row7[0..3]]   */
    ld1r        {v16.2s}, [x0], x1
    ld1r        {v17.2s}, [x0], x1
    ld1r        {v18.2s}, [x0], x1
    ld1r        {v19.2s}, [x0], x1
    ld1         {v16.s}[1], [x0], x1
    ld1         {v17.s}[1], [x0], x1
    ld1         {v18.s}[1], [x0], x1
    ld1         {v19.s}[1], [x0], x1

    ldr         w2,  [x2, #516]
    uxtb        w2,  w2

    /* 4-column × 8-row transpose, out-of-place using v20-v23 as scratch.
     * Stage 1: byte-level interleave (8 insns, was 12 with in-place mov). */
    trn1        v20.8b, v16.8b, v17.8b
    trn2        v21.8b, v16.8b, v17.8b
    trn1        v22.8b, v18.8b, v19.8b
    trn2        v23.8b, v18.8b, v19.8b
    /* Stage 2: halfword-level interleave into v16-v19 */
    trn1        v16.4h, v20.4h, v22.4h
    trn2        v18.4h, v20.4h, v22.4h
    trn1        v17.4h, v21.4h, v23.4h
    trn2        v19.4h, v21.4h, v23.4h

    vp3_loop_filter

    /* ST2 single-element stores: writes v0[n]→[x12] (src[-1]) and
     * v1[n]→[x12+1] (src[0]) per row, then x12 += stride.
     * Eliminates the post-filter trn1/trn2 transpose entirely.        */
    st2         {v0.b, v1.b}[0], [x12], x1
    st2         {v0.b, v1.b}[1], [x12], x1
    st2         {v0.b, v1.b}[2], [x12], x1
    st2         {v0.b, v1.b}[3], [x12], x1
    st2         {v0.b, v1.b}[4], [x12], x1
    st2         {v0.b, v1.b}[5], [x12], x1
    st2         {v0.b, v1.b}[6], [x12], x1
    st2         {v0.b, v1.b}[7], [x12], x1
    ret
endfunc

/* ================================================================== */
/* IDCT helper functions                                               */
/*                                                                     */
/* Calling convention (shared across all helpers):                    */
/*   x0 = dest ptr   x1 = stride   x2 = block ptr (int16_t[64])      */
/*   x10 = saved x30 of top-level caller                              */
/*   w3  = rounding value passed into end_col (8 or 8+2048)          */
/*                                                                     */
/* Data registers after vp3_idct_start_neon / between calls:         */
/*   v0.8h  = IDCT constants (C1S7..C7S1 in h[0..6])                 */
/*   v1.8h  = ip[0]+ip[4]  (set up at start, updated after transpose) */
/*   v16.8h = ip[0]−ip[4]                                             */
/*   v17.8h = ip[1]  (row 1)                                          */
/*   v18.8h = ip[2]  (row 2)                                          */
/*   v19.8h = ip[3]  (row 3)                                          */
/*   v20.8h = ip[4]  (row 4 — still present after start)             */
/*   v21.8h = ip[5]  (row 5)                                          */
/*   v22.8h = ip[6]  (row 6)                                          */
/*   v23.8h = ip[7]  (row 7)                                          */
/*   v2..v7, v24, v25 = temporaries                                   */
/* ================================================================== */

/* Load block, zero it, set up v1/v16 for the core pass.
 * Falls through into vp3_idct_core_neon.                             */
function vp3_idct_start_neon
    movi        v4.16b, #0
    movi        v5.16b, #0
    movrel      x3,  vp3_idct_constants
    ld1         {v0.2d}, [x3]

    ld1         {v16.2d, v17.2d}, [x2]
    st1         {v4.2d, v5.2d},   [x2], #32
    ld1         {v18.2d, v19.2d}, [x2]
    st1         {v4.2d, v5.2d},   [x2], #32
    ld1         {v20.2d, v21.2d}, [x2]
    st1         {v4.2d, v5.2d},   [x2], #32
    add         v1.8h,  v16.8h, v20.8h  // ip[0]+ip[4]
    sub         v16.8h, v16.8h, v20.8h  // ip[0]−ip[4]
    ld1         {v22.2d, v23.2d}, [x2]
    st1         {v4.2d, v5.2d},   [x2], #32

/* ---- fall-through into vp3_idct_core_neon ---- */
vp3_idct_core_neon:
    /* Block 1: ip[1]*C1,  E=(ip[0]+ip[4])*C4,  F=(ip[0]−ip[4])*C4
     *
     * For each quantity, compute lower-4 and upper-4 results separately
     * then pack with shrn/shrn2 into a single .8h register.
     * This matches the ARM32 vmull.s16/vshrn.s32 pattern exactly.   */
    smull       v2.4s,  v17.4h, v0.h[0]   // ip[1]*C1 (lo 4)
    smull2      v3.4s,  v17.8h, v0.h[0]   // ip[1]*C1 (hi 4)
    smull       v4.4s,  v1.4h,  v0.h[3]   // E (lo 4)
    smull2      v5.4s,  v1.8h,  v0.h[3]   // E (hi 4)
    smull       v6.4s,  v16.4h, v0.h[3]   // F (lo 4)
    smull2      v7.4s,  v16.8h, v0.h[3]   // F (hi 4)
    shrn        v2.4h,  v2.4s,  #16
    shrn2       v2.8h,  v3.4s,  #16       // v2 = ip[1]*C1
    shrn        v3.4h,  v4.4s,  #16
    shrn2       v3.8h,  v5.4s,  #16       // v3 = E
    shrn        v4.4h,  v6.4s,  #16
    shrn2       v4.8h,  v7.4s,  #16       // v4 = F
    add         v20.8h, v1.8h,  v3.8h     // v20 = E  (≡ q12 in ARM32)
    add         v16.8h, v16.8h, v4.8h     // v16 = F  (≡ q8)
    add         v1.8h,  v2.8h,  v17.8h   // v1  = ip[1]*C1 partial  (≡ q1)

    /* Block 2: ip[7]*C1, ip[7]*C7, ip[1]*C7 */
    smull       v2.4s,  v23.4h, v0.h[0]   // ip[7]*C1 lo
    smull2      v3.4s,  v23.8h, v0.h[0]   // ip[7]*C1 hi
    smull       v4.4s,  v23.4h, v0.h[6]   // ip[7]*C7 lo
    smull2      v5.4s,  v23.8h, v0.h[6]   // ip[7]*C7 hi
    smull       v6.4s,  v17.4h, v0.h[6]   // ip[1]*C7 lo
    smull2      v7.4s,  v17.8h, v0.h[6]   // ip[1]*C7 hi
    shrn        v2.4h,  v2.4s,  #16
    shrn2       v2.8h,  v3.4s,  #16       // v2 = ip[7]*C1
    shrn        v3.4h,  v4.4s,  #16
    shrn2       v3.8h,  v5.4s,  #16       // v3 = ip[7]*C7
    shrn        v4.4h,  v6.4s,  #16
    shrn2       v4.8h,  v7.4s,  #16       // v4 = ip[1]*C7
    add         v2.8h,  v2.8h,  v23.8h   // v2  = ip[7]*C1 complete
    add         v17.8h, v1.8h,  v3.8h    // v17 = A = ip[1]*C1 + ip[7]*C7  (≡ q9)
    sub         v23.8h, v4.8h,  v2.8h    // v23 = B = ip[1]*C7 − ip[7]*C1  (≡ q15)

    /* Block 3: ip[3]*C5, ip[3]*C3, ip[5]*C5 */
    smull       v2.4s,  v19.4h, v0.h[4]   // ip[3]*C5 lo
    smull2      v3.4s,  v19.8h, v0.h[4]   // ip[3]*C5 hi
    smull       v4.4s,  v19.4h, v0.h[2]   // ip[3]*C3 lo
    smull2      v5.4s,  v19.8h, v0.h[2]   // ip[3]*C3 hi
    smull       v6.4s,  v21.4h, v0.h[4]   // ip[5]*C5 lo
    smull2      v7.4s,  v21.8h, v0.h[4]   // ip[5]*C5 hi
    shrn        v2.4h,  v2.4s,  #16
    shrn2       v2.8h,  v3.4s,  #16       // v2 = ip[3]*C5
    shrn        v3.4h,  v4.4s,  #16
    shrn2       v3.8h,  v5.4s,  #16       // v3 = ip[3]*C3
    shrn        v4.4h,  v6.4s,  #16
    shrn2       v4.8h,  v7.4s,  #16       // v4 = ip[5]*C5
    add         v3.8h,  v3.8h,  v19.8h   // ip[3]*C3 complete
    add         v4.8h,  v4.8h,  v21.8h   // ip[5]*C5 complete
    add         v1.8h,  v2.8h,  v19.8h   // ip[3]*C5 complete  (≡ q1 ARM32)
    add         v19.8h, v3.8h,  v4.8h    // v19 = C = ip[3]*C3 + ip[5]*C5  (≡ q11)

    /* Block 4: ip[5]*C3, ip[2]*C2, ip[6]*C6 */
    smull       v2.4s,  v21.4h, v0.h[2]   // ip[5]*C3 lo
    smull2      v3.4s,  v21.8h, v0.h[2]   // ip[5]*C3 hi
    smull       v4.4s,  v18.4h, v0.h[1]   // ip[2]*C2 lo
    smull2      v5.4s,  v18.8h, v0.h[1]   // ip[2]*C2 hi
    smull       v6.4s,  v22.4h, v0.h[5]   // ip[6]*C6 lo
    smull2      v7.4s,  v22.8h, v0.h[5]   // ip[6]*C6 hi
    shrn        v2.4h,  v2.4s,  #16
    shrn2       v2.8h,  v3.4s,  #16       // v2 = ip[5]*C3
    shrn        v3.4h,  v4.4s,  #16
    shrn2       v3.8h,  v5.4s,  #16       // v3 = ip[2]*C2
    shrn        v4.4h,  v6.4s,  #16
    shrn2       v4.8h,  v7.4s,  #16       // v4 = ip[6]*C6
    add         v2.8h,  v2.8h,  v21.8h   // ip[5]*C3 complete
    add         v3.8h,  v3.8h,  v18.8h   // ip[2]*C2 complete
    sub         v21.8h, v2.8h,  v1.8h    // v21 = D = ip[5]*C3 − ip[3]*C5  (≡ q13)
    sub         v1.8h,  v17.8h, v19.8h   // v1  = A−C
    add         v19.8h, v17.8h, v19.8h   // v19 = Cd = A+C  (≡ q11)
    sub         v17.8h, v23.8h, v21.8h   // v17 = B−D  (≡ q9)
    add         v21.8h, v23.8h, v21.8h   // v21 = Dd = B+D  (≡ q13)
    add         v23.8h, v3.8h,  v4.8h    // v23 = G = ip[2]*C2 + ip[6]*C6  (≡ q15)

    /* Block 5: (A−C)*C4,  ip[6]*C2,  ip[2]*C6,  (B−D)*C4 */
    smull       v2.4s,  v1.4h,  v0.h[3]   // (A−C)*C4 lo
    smull2      v3.4s,  v1.8h,  v0.h[3]   // (A−C)*C4 hi
    smull       v4.4s,  v22.4h, v0.h[1]   // ip[6]*C2 lo
    smull2      v5.4s,  v22.8h, v0.h[1]   // ip[6]*C2 hi
    smull       v6.4s,  v18.4h, v0.h[5]   // ip[2]*C6 lo
    smull2      v7.4s,  v18.8h, v0.h[5]   // ip[2]*C6 hi
    smull       v24.4s, v17.4h, v0.h[3]   // (B−D)*C4 lo  (scratch v24)
    smull2      v25.4s, v17.8h, v0.h[3]   // (B−D)*C4 hi  (scratch v25)
    shrn        v2.4h,  v2.4s,  #16
    shrn2       v2.8h,  v3.4s,  #16       // v2 = Ad = (A−C)*C4
    shrn        v3.4h,  v4.4s,  #16
    shrn2       v3.8h,  v5.4s,  #16       // v3 = ip[6]*C2
    add         v3.8h,  v3.8h,  v22.8h   // ip[6]*C2 complete
    shrn        v4.4h,  v6.4s,  #16
    shrn2       v4.8h,  v7.4s,  #16       // v4 = ip[2]*C6
    add         v18.8h, v1.8h,  v2.8h    // v18 = Ad complete  (≡ q10)
    sub         v22.8h, v4.8h,  v3.8h    // v22 = H = ip[2]*C6 − ip[6]*C2  (≡ q14)
    ret
endfunc

/* ---- Row-pass ending: complete butterfly + 8×8 transpose.
 *      Called between start_neon and a second call to core_neon.    */
.macro VP3_IDCT_END type
function vp3_idct_end_\type\()_neon
.ifc \type, col
    /* add rounding to E (v20) and F (v16) */
    dup         v0.8h,  w3
    add         v20.8h, v20.8h, v0.8h
    add         v16.8h, v16.8h, v0.8h
.endif

    /* Bd = (B−D)*C4  (results from block 5 still in v24/v25) */
    shrn        v24.4h, v24.4s, #16
    shrn2       v24.8h, v25.4s, #16      // v24 = Bd

    /* Final butterfly outputs → v16..v23 */
    add         v2.8h,  v20.8h, v23.8h  // Gd  = E+G
    add         v17.8h, v1.8h,  v24.8h  // v17 = Bd  (≡ q9)
    sub         v20.8h, v20.8h, v23.8h  // Ed  = E−G
    sub         v3.8h,  v16.8h, v18.8h  // Fd  = F−Ad
    add         v18.8h, v16.8h, v18.8h  // Add = F+Ad  (≡ q10)
    add         v4.8h,  v17.8h, v22.8h  // Hd  = Bd+H
    sub         v22.8h, v17.8h, v22.8h  // Bdd = Bd−H  (≡ q14)
    add         v16.8h, v2.8h,  v19.8h  // [0] = Gd+Cd
    sub         v23.8h, v2.8h,  v19.8h  // [7] = Gd−Cd
    add         v17.8h, v18.8h, v4.8h   // [1] = Add+Hd
    sub         v18.8h, v18.8h, v4.8h   // [2] = Add−Hd
    add         v19.8h, v20.8h, v21.8h  // [3] = Ed+Dd
    sub         v20.8h, v20.8h, v21.8h  // [4] = Ed−Dd
    add         v21.8h, v3.8h,  v22.8h  // [5] = Fd+Bdd
    sub         v22.8h, v3.8h,  v22.8h  // [6] = Fd−Bdd

.ifc \type, row
    /* 8×8 transpose using neon.S macro; v24/v25 are scratch           */
    transpose_8x8H v16, v17, v18, v19, v20, v21, v22, v23, v24, v25

    /* Re-enter core for the column pass */
    add         v1.8h,  v16.8h, v20.8h  // ip[0]+ip[4] (transposed)
    sub         v16.8h, v16.8h, v20.8h  // ip[0]−ip[4] (transposed)
.endif
    ret
endfunc
.endm

VP3_IDCT_END row
VP3_IDCT_END col

/* ------------------------------------------------------------------ */
/* ff_vp3_idct_put_neon(uint8_t *dest, ptrdiff_t stride, int16_t *block) */
/* ------------------------------------------------------------------ */
function ff_vp3_idct_put_neon, export=1
    mov         x10, x30
    bl          vp3_idct_start_neon
    bl          vp3_idct_end_row_neon
    mov         w3,  #2056              // 8 + 2048: rounding + signed→unsigned offset
    bl          vp3_idct_core_neon
    bl          vp3_idct_end_col_neon

    sqshrun     v0.8b,  v16.8h, #4
    sqshrun     v1.8b,  v17.8h, #4
    sqshrun     v2.8b,  v18.8h, #4
    sqshrun     v3.8b,  v19.8h, #4
    st1         {v0.8b}, [x0], x1
    sqshrun     v4.8b,  v20.8h, #4
    st1         {v1.8b}, [x0], x1
    sqshrun     v5.8b,  v21.8h, #4
    st1         {v2.8b}, [x0], x1
    sqshrun     v6.8b,  v22.8h, #4
    st1         {v3.8b}, [x0], x1
    sqshrun     v7.8b,  v23.8h, #4
    st1         {v4.8b}, [x0], x1
    st1         {v5.8b}, [x0], x1
    st1         {v6.8b}, [x0], x1
    st1         {v7.8b}, [x0], x1
    ret         x10
endfunc

/* ------------------------------------------------------------------ */
/* ff_vp3_idct_add_neon(uint8_t *dest, ptrdiff_t stride, int16_t *block) */
/* ------------------------------------------------------------------ */
function ff_vp3_idct_add_neon, export=1
    mov         x10, x30
    bl          vp3_idct_start_neon
    bl          vp3_idct_end_row_neon
    mov         w3,  #8                 // rounding only (add to existing pixels)
    bl          vp3_idct_core_neon
    bl          vp3_idct_end_col_neon

    mov         x2,  x0
    ld1         {v0.8b}, [x0], x1
    sshr        v16.8h, v16.8h, #4
    ld1         {v1.8b}, [x0], x1
    sshr        v17.8h, v17.8h, #4
    ld1         {v2.8b}, [x0], x1
    uaddw       v16.8h, v16.8h, v0.8b
    ld1         {v3.8b}, [x0], x1
    uaddw       v17.8h, v17.8h, v1.8b
    ld1         {v4.8b}, [x0], x1
    sshr        v18.8h, v18.8h, #4
    ld1         {v5.8b}, [x0], x1
    sshr        v19.8h, v19.8h, #4
    ld1         {v6.8b}, [x0], x1
    sqxtun      v0.8b,  v16.8h
    ld1         {v7.8b}, [x0], x1
    sqxtun      v1.8b,  v17.8h
    uaddw       v18.8h, v18.8h, v2.8b
    uaddw       v19.8h, v19.8h, v3.8b
    sshr        v20.8h, v20.8h, #4
    sshr        v21.8h, v21.8h, #4
    sqxtun      v2.8b,  v18.8h
    sqxtun      v3.8b,  v19.8h
    uaddw       v20.8h, v20.8h, v4.8b
    uaddw       v21.8h, v21.8h, v5.8b
    sshr        v22.8h, v22.8h, #4
    sshr        v23.8h, v23.8h, #4
    st1         {v0.8b}, [x2], x1
    sqxtun      v4.8b,  v20.8h
    st1         {v1.8b}, [x2], x1
    sqxtun      v5.8b,  v21.8h
    st1         {v2.8b}, [x2], x1
    uaddw       v22.8h, v22.8h, v6.8b
    st1         {v3.8b}, [x2], x1
    uaddw       v23.8h, v23.8h, v7.8b
    st1         {v4.8b}, [x2], x1
    sqxtun      v6.8b,  v22.8h
    st1         {v5.8b}, [x2], x1
    sqxtun      v7.8b,  v23.8h
    st1         {v6.8b}, [x2], x1
    st1         {v7.8b}, [x2], x1
    ret         x10
endfunc

/* ------------------------------------------------------------------ */
/* ff_vp3_idct_dc_add_neon(uint8_t *dest, ptrdiff_t stride, int16_t *block) */
/* ------------------------------------------------------------------ */
function ff_vp3_idct_dc_add_neon, export=1
    ldrsh       w12, [x2]
    mov         x3,  x0
    add         w12, w12, #15
    dup         v15.8h, w12
    mov         w12, #0
    strh        w12, [x2]
    sshr        v15.8h, v15.8h, #5

    ld1         {v0.8b}, [x0], x1
    ld1         {v1.8b}, [x0], x1
    ld1         {v2.8b}, [x0], x1
    uaddw       v8.8h,  v15.8h, v0.8b
    ld1         {v3.8b}, [x0], x1
    uaddw       v9.8h,  v15.8h, v1.8b
    ld1         {v4.8b}, [x0], x1
    uaddw       v10.8h, v15.8h, v2.8b
    ld1         {v5.8b}, [x0], x1
    uaddw       v11.8h, v15.8h, v3.8b
    ld1         {v6.8b}, [x0], x1
    uaddw       v12.8h, v15.8h, v4.8b
    ld1         {v7.8b}, [x0], x1
    uaddw       v13.8h, v15.8h, v5.8b
    sqxtun      v0.8b,  v8.8h
    uaddw       v14.8h, v15.8h, v6.8b
    sqxtun      v1.8b,  v9.8h
    uaddw       v15.8h, v15.8h, v7.8b
    sqxtun      v2.8b,  v10.8h
    st1         {v0.8b}, [x3], x1
    sqxtun      v3.8b,  v11.8h
    st1         {v1.8b}, [x3], x1
    sqxtun      v4.8b,  v12.8h
    st1         {v2.8b}, [x3], x1
    sqxtun      v5.8b,  v13.8h
    st1         {v3.8b}, [x3], x1
    sqxtun      v6.8b,  v14.8h
    st1         {v4.8b}, [x3], x1
    sqxtun      v7.8b,  v15.8h
    st1         {v5.8b}, [x3], x1
    st1         {v6.8b}, [x3], x1
    st1         {v7.8b}, [x3], x1
    ret
endfunc
