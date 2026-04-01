/*
 * AArch64 NEON optimised DSP functions for G722 coding
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

// void ff_g722_apply_qmf_neon(const int16_t *prev_samples, int xout[2])
// x0 = prev_samples, x1 = xout
//
// Computes two 12-tap QMF filter outputs from 24 interleaved int16 samples:
//   xout[1] = dot(samples[0,2,4,...,22], [ 3,-11, 12, 32,-210, 951,3876,-805, 362,-156, 53,-11])
//   xout[0] = dot(samples[1,3,5,...,23], [-11, 53,-156,362,-805,3876, 951,-210,  32,  12,-11,  3])
//
// Strategy: ld2 deinterleaves even/odd samples; two coefficient tables in .rodata.
// Uses v0-v7 only (all caller-saved under AAPCS64).

function ff_g722_apply_qmf_neon, export=1
        // Load 24 int16 samples deinterleaved into even and odd streams.
        // ld2 {v0.8h, v1.8h} loads 16 int16s (32 bytes), deinterleaving:
        //   v0.8h = samples[0,2,4,6,8,10,12,14]
        //   v1.8h = samples[1,3,5,7,9,11,13,15]
        ld2     {v0.8h, v1.8h}, [x0], #32

        // ld2 {v2.4h, v3.4h} loads the remaining 8 int16s (16 bytes):
        //   v2.4h = samples[16,18,20,22]
        //   v3.4h = samples[17,19,21,23]
        ld2     {v2.4h, v3.4h}, [x0]

        // Load even coefficient table (12 x int16 = 24 bytes)
        movrel  x2, g722_qmf_even_coeffs
        ld1     {v4.8h}, [x2], #16     // v4.8h = [3,-11,12,32,-210,951,3876,-805]
        ld1     {v5.4h}, [x2]          // v5.4h = [362,-156,53,-11]

        // xout[1]: dot(even_samples, even_coeffs)
        smull   v6.4s, v0.4h, v4.4h    // s0*3, s2*-11, s4*12, s6*32
        smlal2  v6.4s, v0.8h, v4.8h    // += s8*-210, s10*951, s12*3876, s14*-805
        smlal   v6.4s, v2.4h, v5.4h    // += s16*362, s18*-156, s20*53, s22*-11
        addv    s6, v6.4s               // horizontal sum → xout[1]
        str     s6, [x1, #4]           // store xout[1] at offset 4

        // Load odd coefficient table (12 x int16 = 24 bytes)
        movrel  x2, g722_qmf_odd_coeffs
        ld1     {v4.8h}, [x2], #16     // v4.8h = [-11,53,-156,362,-805,3876,951,-210]
        ld1     {v5.4h}, [x2]          // v5.4h = [32,12,-11,3]

        // xout[0]: dot(odd_samples, odd_coeffs)
        smull   v6.4s, v1.4h, v4.4h    // s1*-11, s3*53, s5*-156, s7*362
        smlal2  v6.4s, v1.8h, v4.8h    // += s9*-805, s11*3876, s13*951, s15*-210
        smlal   v6.4s, v3.4h, v5.4h    // += s17*32, s19*12, s21*-11, s23*3
        addv    s6, v6.4s               // horizontal sum → xout[0]
        str     s6, [x1]               // store xout[0] at offset 0

        ret
endfunc

// QMF coefficients (ITU-T G.722 Table 11): 3,-11,12,32,-210,951,3876,-805,362,-156,53,-11
// Even table: coefficients for even-indexed samples (xout[1])
const g722_qmf_even_coeffs, align=4
        .short   3
        .short  -11
        .short   12
        .short   32
        .short  -210
        .short   951
        .short   3876
        .short  -805
        .short   362
        .short  -156
        .short   53
        .short  -11
endconst

// Odd table: coefficients for odd-indexed samples (xout[0])
const g722_qmf_odd_coeffs, align=4
        .short  -11
        .short   53
        .short  -156
        .short   362
        .short  -805
        .short   3876
        .short   951
        .short  -210
        .short   32
        .short   12
        .short  -11
        .short   3
endconst
