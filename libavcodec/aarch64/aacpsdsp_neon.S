/*
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

function ff_ps_add_squares_neon, export=1
1:      ld1             {v0.4s,v1.4s}, [x1], #32
        fmul            v0.4s, v0.4s, v0.4s
        fmul            v1.4s, v1.4s, v1.4s
        faddp           v2.4s, v0.4s, v1.4s
        ld1             {v3.4s}, [x0]
        fadd            v3.4s, v3.4s, v2.4s
        st1             {v3.4s}, [x0], #16
        subs            w2, w2, #4
        b.gt            1b
        ret
endfunc

function ff_ps_mul_pair_single_neon, export=1
1:      ld1             {v0.4s,v1.4s}, [x1], #32
        ld1             {v2.4s},       [x2], #16
        zip1            v3.4s, v2.4s, v2.4s
        zip2            v4.4s, v2.4s, v2.4s
        fmul            v0.4s, v0.4s, v3.4s
        fmul            v1.4s, v1.4s, v4.4s
        st1             {v0.4s,v1.4s}, [x0], #32
        subs            w3, w3, #4
        b.gt            1b
        ret
endfunc

function ff_ps_stereo_interpolate_neon, export=1
        ld1             {v0.4s}, [x2]
        ld1             {v1.4s}, [x3]
        zip1            v4.4s, v0.4s, v0.4s
        zip2            v5.4s, v0.4s, v0.4s
        zip1            v6.4s, v1.4s, v1.4s
        zip2            v7.4s, v1.4s, v1.4s
1:      ld1             {v2.2s}, [x0]
        ld1             {v3.2s}, [x1]
        fadd            v4.4s, v4.4s, v6.4s
        fadd            v5.4s, v5.4s, v7.4s
        mov             v2.d[1], v2.d[0]
        mov             v3.d[1], v3.d[0]
        fmul            v2.4s, v2.4s, v4.4s
        fmla            v2.4s, v3.4s, v5.4s
        st1             {v2.d}[0], [x0], #8
        st1             {v2.d}[1], [x1], #8
        subs            w4, w4, #1
        b.gt            1b
        ret
endfunc

function ff_ps_stereo_interpolate_ipdopd_neon, export=1
        ld1             {v0.4s,v1.4s}, [x2]
        ld1             {v6.4s,v7.4s}, [x3]
        fneg            v2.4s, v1.4s
        fneg            v3.4s, v7.4s
        zip1            v16.4s, v0.4s, v0.4s
        zip2            v17.4s, v0.4s, v0.4s
        zip1            v18.4s, v2.4s, v1.4s
        zip2            v19.4s, v2.4s, v1.4s
        zip1            v20.4s, v6.4s, v6.4s
        zip2            v21.4s, v6.4s, v6.4s
        zip1            v22.4s, v3.4s, v7.4s
        zip2            v23.4s, v3.4s, v7.4s
1:      ld1             {v2.2s}, [x0]
        ld1             {v3.2s}, [x1]
        fadd            v16.4s, v16.4s, v20.4s
        fadd            v17.4s, v17.4s, v21.4s
        mov             v2.d[1], v2.d[0]
        mov             v3.d[1], v3.d[0]
        fmul            v4.4s, v2.4s, v16.4s
        fmla            v4.4s, v3.4s, v17.4s
        fadd            v18.4s, v18.4s, v22.4s
        fadd            v19.4s, v19.4s, v23.4s
        ext             v2.16b, v2.16b, v2.16b, #4
        ext             v3.16b, v3.16b, v3.16b, #4
        fmla            v4.4s, v2.4s, v18.4s
        fmla            v4.4s, v3.4s, v19.4s
        st1             {v4.d}[0], [x0], #8
        st1             {v4.d}[1], [x1], #8
        subs            w4, w4, #1
        b.gt            1b
        ret
endfunc

function ff_ps_hybrid_analysis_neon, export=1
        lsl             x3, x3, #3
        ld2             {v0.4s,v1.4s}, [x1], #32
        ld2             {v2.2s,v3.2s}, [x1], #16
        ld1             {v24.2s},      [x1], #8
        ld2             {v4.2s,v5.2s}, [x1], #16
        ld2             {v6.4s,v7.4s}, [x1]
        rev64           v6.4s, v6.4s
        rev64           v7.4s, v7.4s
        ext             v6.16b, v6.16b, v6.16b, #8
        ext             v7.16b, v7.16b, v7.16b, #8
        rev64           v4.2s, v4.2s
        rev64           v5.2s, v5.2s
        mov             v2.d[1], v3.d[0]
        mov             v4.d[1], v5.d[0]
        mov             v5.d[1], v2.d[0]
        mov             v3.d[1], v4.d[0]
        fadd            v16.4s, v0.4s, v6.4s
        fadd            v17.4s, v1.4s, v7.4s
        fsub            v18.4s, v1.4s, v7.4s
        fsub            v19.4s, v0.4s, v6.4s
        fadd            v22.4s, v2.4s, v4.4s
        fsub            v23.4s, v5.4s, v3.4s
        trn1            v20.2d, v22.2d, v23.2d      // {re4+re8, re5+re7, im8-im4, im7-im5}
        trn2            v21.2d, v22.2d, v23.2d      // {im4+im8, im5+im7, re4-re8, re5-re7}
1:      ld2             {v2.4s,v3.4s}, [x2], #32
        ld2             {v4.2s,v5.2s}, [x2], #16
        ld1             {v6.2s},       [x2], #8
        add             x2, x2, #8
        mov             v4.d[1], v5.d[0]
        mov             v6.s[1], v6.s[0]
        fmul            v6.2s, v6.2s, v24.2s
        fmul            v0.4s, v2.4s, v16.4s
        fmul            v1.4s, v2.4s, v17.4s
        fmls            v0.4s, v3.4s, v18.4s
        fmla            v1.4s, v3.4s, v19.4s
        fmla            v0.4s, v4.4s, v20.4s
        fmla            v1.4s, v4.4s, v21.4s
        faddp           v0.4s, v0.4s, v1.4s
        faddp           v0.4s, v0.4s, v0.4s
        fadd            v0.2s, v0.2s, v6.2s
        st1             {v0.2s}, [x0], x3
        subs            w4, w4, #1
        b.gt            1b
        ret
endfunc

// void ff_ps_hybrid_synthesis_deint_neon(float out[2][38][64],
//                                        float (*in)[32][2],
//                                        int i, int len)
//
// For each column i (from arg_i to 63), scatter in[i][n][0..1] → out[0..1][n][i].
// 4-wide: process 4 columns at a time using zip1/zip2 deinterleave + 16-byte stores.
//
// Register map:
//  x0  = out[0][0][i]  (base for column i, advances +16 per 4-wide group)
//  x1  = in[i][0][0]   (base for current i-group, advances +1024 per 4-wide group)
//  w2  = 64 - i        (remaining column count, decrements by 4 per group)
//  w3  = len           (constant: number of n-values)
//  x4  = 256           (output stride: 64 floats × 4 bytes = bytes per n-row)
//  x5  = out[1][0][i]  (= x0 + 38×256, same column offsets as x0)
function ff_ps_hybrid_synthesis_deint_neon, export=1
        lsl             w4, w2, #2              // w4 = i*4 (zero-extended to x4)
        lsl             w5, w2, #8              // w5 = i*256 (zero-extended to x5)
        add             x0, x0, x4              // out += i*4
        add             x1, x1, x5              // in  += i*256
        neg             w2, w2
        add             w2, w2, #64             // w2 = 64 - i (remaining columns)
        mov             x4, #256                // output row stride
        add             x5, x0, #8192           // x5 = out + 8192 ...
        add             x5, x5, #1536           //    = out + 9728 = out[1][0][i]

        // Scalar prefix: handle (64-i)%4 columns to align to 4-wide boundary
        ands            w6, w2, #3              // w6 = remainder (0..3)
        sub             w2, w2, w6              // w2 = 4-aligned column count
        cbz             w6, 4f

1:      // scalar loop: 1 column per outer iteration
        mov             x7, x1
        mov             x8, x0
        mov             x9, x5
        mov             w10, w3
2:
        ld1             {v0.2s}, [x7], #8       // in[i][n][re, im]
        st1             {v0.s}[0], [x8], x4     // out[0][n][i], advance row
        st1             {v0.s}[1], [x9], x4     // out[1][n][i], advance row
        subs            w10, w10, #1
        b.gt            2b
        add             x0, x0, #4              // next column in out[0]
        add             x5, x5, #4              // next column in out[1]
        add             x1, x1, #256            // next i in in[]
        subs            w6, w6, #1
        b.gt            1b

4:      // 4-wide main loop
        cbz             w2, 8f

5:
        mov             x13, x1                 // save in[i] base before inner loop
        add             x7, x1, #256            // in[i+1][0][0]
        add             x8, x1, #512            // in[i+2][0][0]
        add             x9, x1, #768            // in[i+3][0][0]
        mov             x10, x0                 // snapshot out[0] row pointer
        mov             x11, x5                 // snapshot out[1] row pointer
        mov             w12, w3

6:
        ld1             {v0.2s}, [x1], #8       // {re0, im0}
        ld1             {v1.2s}, [x7], #8       // {re1, im1}
        ld1             {v2.2s}, [x8], #8       // {re2, im2}
        ld1             {v3.2s}, [x9], #8       // {re3, im3}
        zip1            v4.2s, v0.2s, v1.2s     // {re0, re1}
        zip1            v5.2s, v2.2s, v3.2s     // {re2, re3}
        zip2            v16.2s, v0.2s, v1.2s    // {im0, im1}
        zip2            v17.2s, v2.2s, v3.2s    // {im2, im3}
        mov             v4.d[1], v5.d[0]        // v4  = {re0, re1, re2, re3}
        mov             v16.d[1], v17.d[0]      // v16 = {im0, im1, im2, im3}
        st1             {v4.4s}, [x10], x4      // out[0][n][i..i+3], advance by 256
        st1             {v16.4s}, [x11], x4     // out[1][n][i..i+3], advance by 256
        subs            w12, w12, #1
        b.gt            6b

        add             x1, x13, #1024          // in: skip 4 rows (4×256)
        add             x0, x0, #16             // out[0]: advance 4 columns
        add             x5, x5, #16             // out[1]: advance 4 columns
        subs            w2, w2, #4
        b.gt            5b

8:
        ret
endfunc

// void ff_ps_hybrid_analysis_ileave_neon(float (*out)[32][2],
//                                         float L[2][38][64],
//                                         int i, int len)
//
// For each column i (from arg_i to 63), gather L[0..1][j][i] → out[i][j][0..1].
// 4-wide: load contiguous L[0/1][j][i..i+3], scatter via st2 single-element to
// 4 separate out[i+k] streams.
//
// Register map:
//  x0  = out[i][0][0]  (advances +1024 per 4-wide group = 4 rows × 256)
//  x1  = L[0][0][i]    (= L + i*4, advances +16 per 4-wide group = 4 columns)
//  w2  = 64 - i
//  w3  = len
//  x4  = 256           (L read stride: 64 floats × 4 bytes)
//  x5  = L[1][0][i]    (= x1 + 38×256 = x1 + 9728)
function ff_ps_hybrid_analysis_ileave_neon, export=1
        lsl             w4, w2, #8              // w4 = i*256 (zero-extended to x4)
        lsl             w5, w2, #2              // w5 = i*4 (zero-extended to x5)
        add             x0, x0, x4              // out += i*256
        add             x1, x1, x5              // L   += i*4
        neg             w2, w2
        add             w2, w2, #64             // w2 = 64 - i
        mov             x4, #256                // L read stride
        add             x5, x1, #8192           // x5 = L + 8192 ...
        add             x5, x5, #1536           //    = L + 9728 = L[1][0][i]

        ands            w6, w2, #3
        sub             w2, w2, w6
        cbz             w6, 4f

1:      // scalar: 1 column per outer iteration
        mov             x7, x1                  // L[0][0][i]
        mov             x8, x5                  // L[1][0][i]
        mov             x9, x0                  // out[i][0][0]
        mov             w10, w3
2:
        ld1             {v0.s}[0], [x7], x4     // L[0][j][i], advance to j+1
        ld1             {v0.s}[1], [x8], x4     // L[1][j][i], advance to j+1
        st1             {v0.2s}, [x9], #8       // out[i][j][0..1]
        subs            w10, w10, #1
        b.gt            2b
        add             x0, x0, #256            // next out[i+1]
        add             x1, x1, #4              // next L column
        add             x5, x5, #4
        subs            w6, w6, #1
        b.gt            1b

4:      // 4-wide main loop
        cbz             w2, 8f

5:
        mov             x12, x0                 // save out[i] base
        add             x6, x0, #256            // out[i+1][0][0]
        add             x7, x0, #512            // out[i+2][0][0]
        add             x8, x0, #768            // out[i+3][0][0]
        mov             x9, x1                  // save L[0] base
        mov             x10, x5                 // save L[1] base
        mov             w11, w3

6:
        ld1             {v0.4s}, [x1], x4       // L[0][j][i..i+3], advance to j+1
        ld1             {v1.4s}, [x5], x4       // L[1][j][i..i+3], advance to j+1
        st2             {v0.s, v1.s}[0], [x0], #8   // out[i][j][re,im]
        st2             {v0.s, v1.s}[1], [x6], #8   // out[i+1][j][re,im]
        st2             {v0.s, v1.s}[2], [x7], #8   // out[i+2][j][re,im]
        st2             {v0.s, v1.s}[3], [x8], #8   // out[i+3][j][re,im]
        subs            w11, w11, #1
        b.gt            6b

        add             x0, x12, #1024          // out[i+4][0][0]
        add             x1, x9,  #16            // L[0][0][i+4]
        add             x5, x10, #16            // L[1][0][i+4]
        subs            w2, w2, #4
        b.gt            5b

8:
        ret
endfunc
