/*
 * Copyright (c) 2025
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/aarch64/asm.S"

// void ff_sub_hfyu_median_pred_int16_neon(uint16_t *dst, const uint16_t *src1,
//   const uint16_t *src2, unsigned mask, int w, int *left, int *left_top)
//
// Computes per element:
//   est  = (l + t - lt) & mask
//   pred = clamp(est, min(l, t), max(l, t))   [unsigned median-of-3]
//   dst  = (src2[i] - pred) & mask
//   lt   = src1[i];  l = src2[i]
//
// The l/lt values for element i are src2[i-1]/src1[i-1] — no loop-carried
// dependency through dst — so the loop vectorises with an ext-based shift.
//
// ext v.16b, prev.16b, cur.16b, #14  =>  [ prev.h[7], cur.h[0..6] ]
// This constructs the one-element-back "shifted" vector each iteration.
//
// Registers:
//   v0  current t  (src1[i..i+7])
//   v1  current s  (src2[i..i+7])
//   v2  lt_shifted (ext of prev_s1 and v0)
//   v3  l_shifted  (ext of prev_s2 and v1)
//   v4  est / pred (temp, reused in place)
//   v5  umin(l, t)
//   v6  umax(l, t)
//   v7  dst result
//   v27 previous src1 vector (for ext into next iteration)
//   v28 previous src2 vector (for ext into next iteration)
//   v31 mask broadcast

function ff_sub_hfyu_median_pred_int16_neon, export=1
        ldr             w7, [x5]                        // l   = *left
        ldr             w8, [x6]                        // lt  = *left_top
        dup             v31.8h, w3                      // mask_vec

        // n_full = w & ~7  (number of elements processed by NEON)
        bic             w9, w4, #7
        cbz             w9, .Ldo_ands                   // no full 8-element chunks

        // Initialise prev vectors from scalar l / lt
        dup             v27.8h, w8                      // prev_s1 = dup(lt)
        dup             v28.8h, w7                      // prev_s2 = dup(l)

        // Load first 8-element chunk and build shifted vectors
        ld1             {v0.8h}, [x1], #16
        ld1             {v1.8h}, [x2], #16
        ext             v2.16b, v27.16b, v0.16b, #14    // [lt, src1[0..6]]
        ext             v3.16b, v28.16b, v1.16b, #14    // [l,  src2[0..6]]
        sub             w9, w9, #8

.Lneon_loop:
        // est = (l + t - lt) & mask
        add             v4.8h, v3.8h, v0.8h
        sub             v4.8h, v4.8h, v2.8h
        and             v4.16b, v4.16b, v31.16b

        // pred = clamp(est, umin(l,t), umax(l,t))
        umin            v5.8h, v3.8h, v0.8h
        umax            v6.8h, v3.8h, v0.8h
        umax            v4.8h, v4.8h, v5.8h
        umin            v4.8h, v4.8h, v6.8h

        // dst = (s - pred) & mask
        sub             v7.8h, v1.8h, v4.8h
        and             v7.16b, v7.16b, v31.16b
        st1             {v7.8h}, [x0], #16

        cbz             w9, .Lneon_done

        // Prepare next iteration: save current vectors, load next chunk
        mov             v27.16b, v0.16b
        mov             v28.16b, v1.16b
        ld1             {v0.8h}, [x1], #16
        ld1             {v1.8h}, [x2], #16
        ext             v2.16b, v27.16b, v0.16b, #14
        ext             v3.16b, v28.16b, v1.16b, #14
        sub             w9, w9, #8
        b               .Lneon_loop

.Lneon_done:
        // Extract l and lt from the last processed vectors (for tail or output)
        umov            w7, v1.h[7]                     // l  = src2[n_full-1]
        umov            w8, v0.h[7]                     // lt = src1[n_full-1]

.Ldo_ands:
        // tail = w & 7  (0 if w is a multiple of 8)
        ands            w4, w4, #7
        b.eq            .Lwrite_out

        // Scalar tail: process remaining w&7 elements
        // w7=l, w8=lt, w10=t(src1), w15=s(src2), w12=est/pred
.Lscalar_loop:
        ldrh            w10, [x1], #2                   // t = src1[i]
        ldrh            w15, [x2], #2                   // s = src2[i]

        // est = (l + t - lt) & mask
        add             w12, w7, w10
        sub             w12, w12, w8
        and             w12, w12, w3

        // pred = clamp(est, min(l,t), max(l,t))   [unsigned]
        cmp             w7, w10
        csel            w13, w7,  w10, lo               // min(l, t)
        csel            w14, w10, w7,  lo               // max(l, t)
        cmp             w12, w13
        csel            w12, w13, w12, lo               // max(est, min)
        cmp             w12, w14
        csel            w12, w14, w12, hi               // min(above, max) = pred

        // dst = (s - pred) & mask
        sub             w11, w15, w12
        and             w11, w11, w3
        strh            w11, [x0], #2

        // Advance l and lt for next element
        mov             w8, w10                         // lt = src1[i]
        mov             w7, w15                         // l  = src2[i]
        subs            w4, w4, #1
        b.ne            .Lscalar_loop

.Lwrite_out:
        str             w7, [x5]                        // *left     = l
        str             w8, [x6]                        // *left_top = lt
        ret
endfunc
