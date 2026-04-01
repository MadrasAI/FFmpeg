/*
 * AArch64 NEON optimised lossless audio DSP functions
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

// int32_t ff_scalarproduct_and_madd_int16_neon(int16_t *v1 /*align16*/,
//                                              const int16_t *v2,
//                                              const int16_t *v3,
//                                              int order, int mul)
//
// Computes:  res = sum(v1[i] * v2[i])  for i in [0, order)
//            v1[i] += mul * v3[i]      (in-place, same loop)
//
// order is guaranteed to be a multiple of 16 (per header comment).
// v1 is guaranteed 16-byte aligned.  16 elements processed per iteration.

function ff_scalarproduct_and_madd_int16_neon, export=1
        movi    v16.4s, #0                  // accumulator lo half
        movi    v17.4s, #0                  // accumulator hi half
        dup     v18.8h, w4                  // broadcast mul to all 8 lanes
0:
        ldp     q0, q1, [x0]               // v1[0..7], v1[8..15]
        ld1     {v2.8h}, [x1], #16         // v2[0..7]
        ld1     {v3.8h}, [x1], #16         // v2[8..15]
        ld1     {v4.8h}, [x2], #16         // v3[0..7]
        ld1     {v5.8h}, [x2], #16         // v3[8..15]

        // dot product: acc += v1 * v2  (16-bit → 32-bit widening)
        smlal   v16.4s, v0.4h, v2.4h       // acc_lo += v1[0..3] * v2[0..3]
        smlal2  v16.4s, v0.8h, v2.8h       // acc_lo += v1[4..7] * v2[4..7]
        smlal   v17.4s, v1.4h, v3.4h       // acc_hi += v1[8..11] * v2[8..11]
        smlal2  v17.4s, v1.8h, v3.8h       // acc_hi += v1[12..15] * v2[12..15]

        // MADD: v1 += mul * v3  (16-bit wrapping)
        mul     v4.8h, v4.8h, v18.8h       // v3[0..7] * mul
        mul     v5.8h, v5.8h, v18.8h       // v3[8..15] * mul
        add     v0.8h, v0.8h, v4.8h        // v1[0..7]  += ...
        add     v1.8h, v1.8h, v5.8h        // v1[8..15] += ...

        stp     q0, q1, [x0], #32          // writeback v1[0..15], advance x0

        subs    w3, w3, #16
        b.gt    0b

        add     v16.4s, v16.4s, v17.4s     // merge two halves
        addv    s16, v16.4s                 // horizontal sum → s16
        fmov    w0, s16                     // move bits to integer return
        ret
endfunc


// int32_t ff_scalarproduct_and_madd_int32_neon(int16_t *v1 /*align16*/,
//                                              const int32_t *v2,
//                                              const int16_t *v3,
//                                              int order, int mul)
//
// Computes:  res = sum(sign_extend(v1[i]) * v2[i])  for i in [0, order)
//            v1[i] += mul * v3[i]                    (in-place, same loop)
//
// order is a multiple of 16.  v1 is 16-byte aligned.
// 8 elements per iteration (v2 is 32-bit; 8 elements = 32 bytes = 2 q-regs).

function ff_scalarproduct_and_madd_int32_neon, export=1
        movi    v16.4s, #0                  // accumulator lo half
        movi    v17.4s, #0                  // accumulator hi half
        dup     v18.8h, w4                  // broadcast mul

0:
        ld1     {v0.8h}, [x0]              // v1[0..7] (int16)
        ldp     q2, q3, [x1], #32          // v2[0..3], v2[4..7] (int32)
        ld1     {v4.8h}, [x2], #16         // v3[0..7] (int16)

        // sign-extend v1 halves to int32 for 32×32 multiply
        sshll   v6.4s, v0.4h, #0           // v1[0..3] → int32
        sshll2  v7.4s, v0.8h, #0           // v1[4..7] → int32

        // dot product: acc += sign_ext(v1) * v2  (32×32→32 low bits)
        mla     v16.4s, v6.4s, v2.4s       // acc_lo += v1[0..3] * v2[0..3]
        mla     v17.4s, v7.4s, v3.4s       // acc_hi += v1[4..7] * v2[4..7]

        // MADD: v1 += mul * v3  (16-bit wrapping)
        mul     v4.8h, v4.8h, v18.8h       // v3[0..7] * mul
        add     v0.8h, v0.8h, v4.8h        // v1[0..7] += ...

        st1     {v0.8h}, [x0], #16         // writeback v1[0..7]

        subs    w3, w3, #8
        b.gt    0b

        add     v16.4s, v16.4s, v17.4s     // merge two halves
        addv    s16, v16.4s                 // horizontal sum → s16
        fmov    w0, s16                     // integer return
        ret
endfunc
