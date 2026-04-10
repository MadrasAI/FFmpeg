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
 * void ff_fcmul_add_neon(float *sum, const float *t, const float *c,
 *                        ptrdiff_t len)
 * x0=sum, x1=t, x2=c, x3=len
 *
 * Per complex pair n (0..len-1):
 *   sum[2n]   += t_re * c_re - t_im * c_im
 *   sum[2n+1] += t_re * c_im + t_im * c_re
 * Final scalar: sum[2*len] += t[2*len] * c[2*len]
 *
 * Main loop: 4 complex pairs per iteration (8 floats = 32 bytes each in/out).
 *
 * NEON:
 *   ld2 {v0.4s, v1.4s}, [x1], #32  → v0=[t_re0..3] v1=[t_im0..3]
 *   ld2 {v2.4s, v3.4s}, [x2], #32  → v2=[c_re0..3] v3=[c_im0..3]
 *   ld2 {v4.4s, v5.4s}, [x0]       → v4=[sum_re0..3] v5=[sum_im0..3]
 *   fmla v4, v0, v2   (sum_re += t_re * c_re)
 *   fmls v4, v1, v3   (sum_re -= t_im * c_im)
 *   fmla v5, v0, v3   (sum_im += t_re * c_im)
 *   fmla v5, v1, v2   (sum_im += t_im * c_re)
 *   st2 {v4.4s, v5.4s}, [x0], #32
 *
 * FMA rounding: matches the tolerance allowed by the checkasm test
 * (same approach as x86 FMA3 path).
 */
function ff_fcmul_add_neon, export=1
        // Main loop: 4 complex pairs at a time
        subs    x3, x3, #4
        b.lt    .Lfcmul_tail4

.Lfcmul_main:
        ld2     {v0.4s, v1.4s}, [x1], #32   // t: v0=re, v1=im
        ld2     {v2.4s, v3.4s}, [x2], #32   // c: v2=re, v3=im
        ld2     {v4.4s, v5.4s}, [x0]        // sum: v4=re, v5=im

        fmla    v4.4s, v0.4s, v2.4s         // sum_re += t_re * c_re
        fmls    v4.4s, v1.4s, v3.4s         // sum_re -= t_im * c_im
        fmla    v5.4s, v0.4s, v3.4s         // sum_im += t_re * c_im
        fmla    v5.4s, v1.4s, v2.4s         // sum_im += t_im * c_re

        st2     {v4.4s, v5.4s}, [x0], #32

        subs    x3, x3, #4
        b.ge    .Lfcmul_main

.Lfcmul_tail4:
        // Restore remaining count (x3 in range -4..-1 → remaining 0..3 pairs)
        adds    x3, x3, #4
        b.eq    .Lfcmul_scalar_final

.Lfcmul_tail_loop:
        // Scalar tail: one complex pair at a time
        ldr     s0, [x1], #4    // t_re
        ldr     s1, [x1], #4    // t_im
        ldr     s2, [x2], #4    // c_re
        ldr     s3, [x2], #4    // c_im
        ldr     s4, [x0]        // sum_re
        ldr     s5, [x0, #4]    // sum_im

        fmadd   s4, s0, s2, s4   // sum_re += t_re * c_re
        fnmsub  s4, s1, s3, s4   // sum_re -= t_im * c_im
        fmadd   s5, s0, s3, s5   // sum_im += t_re * c_im
        fmadd   s5, s1, s2, s5   // sum_im += t_im * c_re

        str     s4, [x0], #4
        str     s5, [x0], #4

        subs    x3, x3, #1
        b.gt    .Lfcmul_tail_loop

.Lfcmul_scalar_final:
        // Final extra element: sum[2*len] += t[2*len] * c[2*len]
        ldr     s0, [x1]
        ldr     s1, [x2]
        ldr     s2, [x0]
        fmadd   s2, s0, s1, s2
        str     s2, [x0]

        ret
endfunc

/*
 * void ff_dcmul_add_neon(double *sum, const double *t, const double *c,
 *                        ptrdiff_t len)
 * x0=sum, x1=t, x2=c, x3=len
 *
 * Same operation as fcmul_add but with float64 (double).
 * Main loop: 2 complex pairs per iteration (4 doubles = 32 bytes each).
 *
 * NEON:
 *   ld2 {v0.2d, v1.2d}, [x1], #32  → v0=[t_re0,t_re1] v1=[t_im0,t_im1]
 *   ld2 {v2.2d, v3.2d}, [x2], #32  → v2=[c_re0,c_re1] v3=[c_im0,c_im1]
 *   ld2 {v4.2d, v5.2d}, [x0]       → v4=[sum_re0,sum_re1] v5=[sum_im0,sum_im1]
 *   fmla v4, v0, v2
 *   fmls v4, v1, v3
 *   fmla v5, v0, v3
 *   fmla v5, v1, v2
 *   st2 {v4.2d, v5.2d}, [x0], #32
 */
function ff_dcmul_add_neon, export=1
        subs    x3, x3, #2
        b.lt    .Ldcmul_tail2

.Ldcmul_main:
        ld2     {v0.2d, v1.2d}, [x1], #32   // t: v0=re, v1=im
        ld2     {v2.2d, v3.2d}, [x2], #32   // c: v2=re, v3=im
        ld2     {v4.2d, v5.2d}, [x0]        // sum: v4=re, v5=im

        fmla    v4.2d, v0.2d, v2.2d         // sum_re += t_re * c_re
        fmls    v4.2d, v1.2d, v3.2d         // sum_re -= t_im * c_im
        fmla    v5.2d, v0.2d, v3.2d         // sum_im += t_re * c_im
        fmla    v5.2d, v1.2d, v2.2d         // sum_im += t_im * c_re

        st2     {v4.2d, v5.2d}, [x0], #32

        subs    x3, x3, #2
        b.ge    .Ldcmul_main

.Ldcmul_tail2:
        // Restore remaining count
        adds    x3, x3, #2
        b.eq    .Ldcmul_scalar_final

        // Scalar tail: one double complex pair
        ldr     d0, [x1], #8    // t_re
        ldr     d1, [x1], #8    // t_im
        ldr     d2, [x2], #8    // c_re
        ldr     d3, [x2], #8    // c_im
        ldr     d4, [x0]        // sum_re
        ldr     d5, [x0, #8]    // sum_im

        fmadd   d4, d0, d2, d4   // sum_re += t_re * c_re
        fnmsub  d4, d1, d3, d4   // sum_re -= t_im * c_im
        fmadd   d5, d0, d3, d5   // sum_im += t_re * c_im
        fmadd   d5, d1, d2, d5   // sum_im += t_im * c_re

        str     d4, [x0], #8
        str     d5, [x0], #8

.Ldcmul_scalar_final:
        // Final extra element: sum[2*len] += t[2*len] * c[2*len]
        ldr     d0, [x1]
        ldr     d1, [x2]
        ldr     d2, [x0]
        fmadd   d2, d0, d1, d2
        str     d2, [x0]

        ret
endfunc
