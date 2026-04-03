/*
 * AArch64 NEON optimised lossless video encoder DSP functions
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

#include "libavutil/aarch64/asm.S"

/*
 * void ff_diff_bytes_neon(uint8_t *dst, const uint8_t *src1,
 *                         const uint8_t *src2, intptr_t w)
 *
 * dst[i] = src1[i] - src2[i]  (modulo 256)
 *
 * x0=dst, x1=src1, x2=src2, x3=w (intptr_t, 64-bit)
 * 32 bytes per NEON iteration; scalar epilogue for remainder.
 */
function ff_diff_bytes_neon, export=1
        subs    x3,  x3,  #32
        b.lt    2f
1:
        ld1     {v0.16b, v1.16b}, [x1], #32
        ld1     {v2.16b, v3.16b}, [x2], #32
        sub     v0.16b, v0.16b, v2.16b
        sub     v1.16b, v1.16b, v3.16b
        st1     {v0.16b, v1.16b}, [x0], #32
        subs    x3,  x3,  #32
        b.ge    1b
2:
        adds    x3,  x3,  #32
        b.eq    4f
3:
        ldrb    w4,  [x1], #1
        ldrb    w5,  [x2], #1
        sub     w4,  w4,  w5
        strb    w4,  [x0], #1
        subs    x3,  x3,  #1
        b.ne    3b
4:
        ret
endfunc

/*
 * void ff_sub_left_predict_neon(uint8_t *dst, const uint8_t *src,
 *                               ptrdiff_t stride, ptrdiff_t width, int height)
 *
 * dst[i] = src[i] - src[i-1]  (finite difference, mod 256)
 * src[-1] for the first element of each row = 0x80 (C constant).
 *
 * x0=dst, x1=src, x2=stride (ptrdiff_t), x3=width (ptrdiff_t), w4=height
 *
 * Per row: 16B/iter. ext builds {prev, src[0..14]} so that
 * sub gives dst[i] = src[i] - prev_i for all 16 lanes simultaneously.
 * prev for the next 16-element block = src[15] (the last SOURCE byte,
 * saved via umov BEFORE the sub overwrites the register).
 */
function ff_sub_left_predict_neon, export=1
        sxtw    x4,  w4                     // sign-extend height to 64-bit
        mov     w7,  #0x80                  // prev = 0x80, initialized ONCE (persists across rows)

.Lslp_row_loop:
        cbz     x4,  .Lslp_done

        mov     x5,  x1                     // x5 = row src pointer
        mov     x6,  x3                     // x6 = remaining width this row

        subs    x6,  x6,  #16
        b.lt    .Lslp_scalar_row
.Lslp_vec_row:
        dup     v0.16b, w7                  // broadcast prev
        ld1     {v1.16b}, [x5], #16         // v1 = src[i..i+15]
        umov    w7,  v1.b[15]               // new prev = src[15]  (BEFORE sub)
        ext     v2.16b, v0.16b, v1.16b, #15 // {prev, src[0..14]}
        sub     v1.16b, v1.16b, v2.16b      // dst[i] = src[i] - prev_i
        st1     {v1.16b}, [x0], #16
        subs    x6,  x6,  #16
        b.ge    .Lslp_vec_row

.Lslp_scalar_row:
        adds    x6,  x6,  #16
        b.eq    .Lslp_row_advance
.Lslp_scalar_loop:
        ldrb    w8,  [x5], #1               // src[i]
        sub     w9,  w8,  w7                // dst = src[i] - prev
        uxtb    w9,  w9
        strb    w9,  [x0], #1
        mov     w7,  w8                     // prev = src[i]  (not dst[i])
        subs    x6,  x6,  #1
        b.ne    .Lslp_scalar_loop

.Lslp_row_advance:
        add     x1,  x1,  x2               // src += stride
        sub     x4,  x4,  #1
        b       .Lslp_row_loop

.Lslp_done:
        ret
endfunc

/*
 * void ff_sub_median_pred_neon(uint8_t *dst, const uint8_t *src1,
 *                              const uint8_t *src2, intptr_t w,
 *                              int *left, int *left_top)
 *
 * C reference:
 *   l = *left; lt = *left_top;
 *   for (i = 0; i < w; i++) {
 *       pred   = mid_pred(l, src1[i], (l + src1[i] - lt) & 0xFF);
 *       lt     = src1[i];
 *       l      = src2[i];
 *       dst[i] = l - pred;
 *   }
 *   *left = l; *left_top = lt;
 *
 * l  = src2[i-1] (initially *left)
 * lt = src1[i-1] (initially *left_top)
 *
 * Vectorized for 16-element blocks:
 *   l_vec[i]  = {l,  src2[0..14]}  via ext(dup(l),  src2, #15)
 *   lt_vec[i] = {lt, src1[0..14]}  via ext(dup(lt), src1, #15)
 *   tmp1      = (l_vec + src1 - lt_vec) & 0xFF
 *   pred      = umax(umin(l_vec, src1), umin(umax(l_vec, src1), tmp1))
 *   dst       = src2 - pred
 *   new l  = src2[15], new lt = src1[15]  (saved BEFORE src2/src1 regs overwritten)
 *
 * x0=dst, x1=src1, x2=src2, x3=w (intptr_t, 64-bit)
 * x4=int *left, x5=int *left_top
 * w7=l (running), w8=lt (running)
 */
function ff_sub_median_pred_neon, export=1
        ldrb    w7,  [x4]                   // w7 = l  = *left
        ldrb    w8,  [x5]                   // w8 = lt = *left_top

        subs    x3,  x3,  #16
        b.lt    .Lsmp_scalar
.Lsmp_vec_loop:
        ld1     {v1.16b}, [x1], #16         // src1[i..i+15]
        ld1     {v2.16b}, [x2], #16         // src2[i..i+15]

        // l_vec  = {l,  src2[0..14]}
        dup     v16.16b, w7
        ext     v6.16b,  v16.16b, v2.16b,  #15

        // lt_vec = {lt, src1[0..14]}
        dup     v17.16b, w8
        ext     v7.16b,  v17.16b, v1.16b,  #15

        // Save new running scalars (src bytes, not dst)
        umov    w7,  v2.b[15]               // new l  = src2[15]
        umov    w8,  v1.b[15]               // new lt = src1[15]

        // tmp1 = (l_vec + src1 - lt_vec) & 0xFF
        add     v5.16b,  v6.16b,  v1.16b   // l + src1
        sub     v5.16b,  v5.16b,  v7.16b   // tmp1

        // pred = mid_pred = umax(umin(l,src1), umin(umax(l,src1), tmp1))
        umin    v3.16b,  v6.16b,  v1.16b   // min(l, src1)
        umax    v4.16b,  v6.16b,  v1.16b   // max(l, src1)
        umin    v4.16b,  v4.16b,  v5.16b   // min(max(l,src1), tmp1)
        umax    v3.16b,  v3.16b,  v4.16b   // pred

        sub     v2.16b,  v2.16b,  v3.16b   // dst = src2 - pred
        st1     {v2.16b}, [x0], #16
        subs    x3,  x3,  #16
        b.ge    .Lsmp_vec_loop

.Lsmp_scalar:
        adds    x3,  x3,  #16
        b.eq    .Lsmp_done
.Lsmp_scalar_loop:
        ldrb    w9,  [x1], #1              // src1[i]
        ldrb    w10, [x2], #1              // src2[i]
        // tmp1 = (l + src1[i] - lt) & 0xFF
        add     w11, w7,  w9
        sub     w11, w11, w8
        uxtb    w11, w11
        // min(l, src1) → w12;  max(l, src1) → w13
        cmp     w7,  w9
        csel    w12, w7,  w9,  lo          // min(l, src1)
        csel    w13, w9,  w7,  lo          // max(l, src1)
        // min(max(l,src1), tmp1) → w13
        cmp     w13, w11
        csel    w13, w13, w11, lo          // min(max, tmp1)
        // pred = max(min(l,src1), min(max,tmp1)) → w12
        cmp     w12, w13
        csel    w12, w12, w13, hi          // max(min, min_of_max)
        // dst[i] = src2[i] - pred
        sub     w14, w10, w12
        uxtb    w14, w14
        strb    w14, [x0], #1
        // Update running state: lt = src1[i], l = src2[i]
        mov     w8,  w9                    // lt = src1[i]
        mov     w7,  w10                   // l  = src2[i]
        subs    x3,  x3,  #1
        b.ne    .Lsmp_scalar_loop

.Lsmp_done:
        strb    w7,  [x4]                  // *left     = l  = src2[w-1]
        strb    w8,  [x5]                  // *left_top = lt = src1[w-1]
        ret
endfunc
