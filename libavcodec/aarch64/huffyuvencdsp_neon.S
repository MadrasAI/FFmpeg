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
// Per-element algorithm:
//   est  = (l + t - lt) & mask
//   pred = clamp(est, umin(l,t), umax(l,t))   [unsigned median-of-3]
//   dst  = (src2[i] - pred) & mask
//   lt   = src1[i];  l = src2[i]
//
// Vectorisation: l/lt depend only on source arrays, not on dst — no
// loop-carried hazard.  ext v.16b, prev.16b, cur.16b, #14 builds
// the one-element-back shifted vector: [ prev.h[7], cur.h[0..6] ].
//
// Loop structure: 2× software-pipelined unroll, 16 uint16 per iteration.
//
//   ldp q0,q2/q1,q3 loads both src1/src2 chunk-pairs in 2 instruction slots.
//   All 4 ext vectors (lt/l for chunk A and chunk B) are issued immediately
//   after the ldp — all operands arrive at the same load latency, so the 4
//   exts retire together in 1-2 cycles, fully hiding load latency behind the
//   previous iteration's compute.
//
//   Instruction budget per 16 elements (steady state):
//     20  compute + store (2 × 10)
//      1  cbz
//      2  mov (save t1/s1 as prev before overwriting)
//      2  ldp (replace 4 ld1)
//      4  ext
//      2  sub + b (loop control)
//     --
//     31  total  vs  38 for the 1× loop (2 × 19)
//
// Registers (2× loop):
//   v0   t0  src1[i..i+7]          v1   s0  src2[i..i+7]
//   v2   t1  src1[i+8..i+15]       v3   s1  src2[i+8..i+15]
//   v4   lt_shift_A → umin_A(reuse) v5   l_shift_A → umax_A(reuse)
//   v6   est_A → pred_A             v7   dst_A
//   v16  lt_shift_B                 v17  l_shift_B
//   v27  prev src1                  v28  prev src2
//   v31  mask broadcast

function ff_sub_hfyu_median_pred_int16_neon, export=1
        ldr             w7, [x5]                        // l   = *left
        ldr             w8, [x6]                        // lt  = *left_top
        dup             v31.8h, w3                      // mask_vec

        // n_full = w & ~7  (total elements for NEON)
        bic             w9, w4, #7
        cbz             w9, .Ldo_ands

        dup             v27.8h, w8                      // prev_s1 = dup(lt)
        dup             v28.8h, w7                      // prev_s2 = dup(l)

        // n16 = w & ~15  (elements for the 2× loop)
        bic             w10, w4, #15
        cbz             w10, .Lone_chunk                // < 16 elements: skip 2× loop

        // --------------- 2× software-pipelined prologue ---------------
        // Load first pair of 8-element chunks and build all 4 shifted vectors.
        // Both ldp results are ready at the same cycle (load latency), so all
        // four ext instructions can issue simultaneously at load_latency+1.
        ldp             q0, q2, [x1], #32               // t0, t1
        ldp             q1, q3, [x2], #32               // s0, s1
        ext             v4.16b, v27.16b, v0.16b, #14    // lt_shift_A = [prev_s1.h[7], t0.h[0..6]]
        ext             v5.16b, v28.16b, v1.16b, #14    // l_shift_A
        ext             v16.16b, v0.16b, v2.16b, #14    // lt_shift_B = [t0.h[7], t1.h[0..6]]
        ext             v17.16b, v1.16b, v3.16b, #14    // l_shift_B
        sub             w10, w10, #16

        // --------------- 2× unrolled main loop ---------------
        // Entry invariant: v0-v3 loaded, v4/v5/v16/v17 extended, w10 decremented.
.L2x_loop:
        // Chunk A: est_A = (l_A + t0 - lt_A) & mask
        // v4=lt_shift_A, v5=l_shift_A consumed then reused as min/max temps.
        // Use v6 as est accumulator to avoid overwriting v4 (lt_shift_A) too early.
        add             v6.8h, v5.8h, v0.8h             // l_A + t0
        sub             v6.8h, v6.8h, v4.8h             // - lt_A  (v4 consumed here)
        and             v6.16b, v6.16b, v31.16b          // est_A
        umin            v4.8h, v5.8h, v0.8h             // min(l_A, t0) — reuse v4
        umax            v5.8h, v5.8h, v0.8h             // max(l_A, t0) — reuse v5
        umax            v6.8h, v6.8h, v4.8h             // clamp low
        umin            v6.8h, v6.8h, v5.8h             // pred_A
        sub             v7.8h, v1.8h, v6.8h             // s0 - pred_A
        and             v7.16b, v7.16b, v31.16b
        st1             {v7.8h}, [x0], #16

        // Chunk B: v4/v5/v6/v7 all free; v16=lt_shift_B, v17=l_shift_B.
        add             v4.8h, v17.8h, v2.8h            // l_B + t1
        sub             v4.8h, v4.8h, v16.8h            // - lt_B
        and             v4.16b, v4.16b, v31.16b          // est_B
        umin            v5.8h, v17.8h, v2.8h
        umax            v6.8h, v17.8h, v2.8h
        umax            v4.8h, v4.8h, v5.8h
        umin            v4.8h, v4.8h, v6.8h             // pred_B
        sub             v7.8h, v3.8h, v4.8h             // s1 - pred_B
        and             v7.16b, v7.16b, v31.16b
        st1             {v7.8h}, [x0], #16

        cbz             w10, .L2x_done

        // Bottom of loop: save t1/s1 as prev (before ldp overwrites v2/v3),
        // load next pair, rebuild all 4 shifted vectors.
        mov             v27.16b, v2.16b                 // prev_s1 = t1
        mov             v28.16b, v3.16b                 // prev_s2 = s1
        ldp             q0, q2, [x1], #32
        ldp             q1, q3, [x2], #32
        ext             v4.16b, v27.16b, v0.16b, #14
        ext             v5.16b, v28.16b, v1.16b, #14
        ext             v16.16b, v0.16b, v2.16b, #14
        ext             v17.16b, v1.16b, v3.16b, #14
        sub             w10, w10, #16
        b               .L2x_loop

.L2x_done:
        // Always update l/lt scalars and prev vectors from last 2× pair.
        // If a 1× tail follows, .Lone_chunk will override w7/w8 with its own umov.
        mov             v27.16b, v2.16b
        mov             v28.16b, v3.16b
        umov            w7, v3.h[7]                     // l  = src2[n16-1]
        umov            w8, v2.h[7]                     // lt = src1[n16-1]
        tst             w9, #8                          // remaining 8-element chunk?
        b.ne            .Lone_chunk

        b               .Ldo_ands

        // --------------- 1× single-chunk path ---------------
        // Handles two entry points:
        //   (a) n16 == 0: entered via cbz from init; v27/v28 from dup
        //   (b) n16 >  0: entered via b.ne from .L2x_done; v27/v28 updated
.Lone_chunk:
        ld1             {v0.8h}, [x1], #16
        ld1             {v1.8h}, [x2], #16
        ext             v2.16b, v27.16b, v0.16b, #14
        ext             v3.16b, v28.16b, v1.16b, #14
        add             v4.8h, v3.8h, v0.8h
        sub             v4.8h, v4.8h, v2.8h
        and             v4.16b, v4.16b, v31.16b
        umin            v5.8h, v3.8h, v0.8h
        umax            v6.8h, v3.8h, v0.8h
        umax            v4.8h, v4.8h, v5.8h
        umin            v4.8h, v4.8h, v6.8h
        sub             v7.8h, v1.8h, v4.8h
        and             v7.16b, v7.16b, v31.16b
        st1             {v7.8h}, [x0], #16
        umov            w7, v1.h[7]                     // l  = src2[n_full-1]
        umov            w8, v0.h[7]                     // lt = src1[n_full-1]

        // --------------- scalar tail and output ---------------
.Ldo_ands:
        ands            w4, w4, #7
        b.eq            .Lwrite_out

.Lscalar_loop:
        ldrh            w10, [x1], #2                   // t = src1[i]
        ldrh            w15, [x2], #2                   // s = src2[i]
        add             w12, w7, w10
        sub             w12, w12, w8
        and             w12, w12, w3                    // est
        cmp             w7, w10
        csel            w13, w7,  w10, lo               // min(l, t)
        csel            w14, w10, w7,  lo               // max(l, t)
        cmp             w12, w13
        csel            w12, w13, w12, lo               // clamp low
        cmp             w12, w14
        csel            w12, w14, w12, hi               // pred
        sub             w11, w15, w12
        and             w11, w11, w3                    // dst
        strh            w11, [x0], #2
        mov             w8, w10                         // lt = src1[i]
        mov             w7, w15                         // l  = src2[i]
        subs            w4, w4, #1
        b.ne            .Lscalar_loop

.Lwrite_out:
        str             w7, [x5]                        // *left     = l
        str             w8, [x6]                        // *left_top = lt
        ret
endfunc
