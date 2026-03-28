/*
 * AArch64 NEON optimised lossless video DSP functions
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

// void ff_add_bytes_neon(uint8_t *dst, uint8_t *src, ptrdiff_t w)
// x0=dst (read+write), x1=src (read), x2=w
// dst[i] += src[i]; 32B/iteration; scalar epilogue for remainder.
function ff_add_bytes_neon, export=1
        subs    x2,  x2,  #32
        b.lt    2f
1:
        ld1     {v0.16b, v1.16b}, [x1], #32
        ld1     {v2.16b, v3.16b}, [x0]
        add     v0.16b, v0.16b,  v2.16b
        add     v1.16b, v1.16b,  v3.16b
        st1     {v0.16b, v1.16b}, [x0], #32
        subs    x2,  x2,  #32
        b.ge    1b
2:
        adds    x2,  x2,  #32
        b.eq    4f
3:
        ldrb    w3,  [x1], #1
        ldrb    w4,  [x0]
        add     w3,  w3,  w4
        strb    w3,  [x0], #1
        subs    x2,  x2,  #1
        b.ne    3b
4:
        ret
endfunc

// int ff_add_left_pred_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t w, int left)
// x0=dst, x1=src, x2=w, w3=left(carry)
// Returns: last output byte (new carry for next row).
// Each output[i] = (carry + src[0] + ... + src[i]) & 0xFF
function ff_add_left_pred_neon, export=1
        movi    v31.16b, #0          // zero register used as Vn in ext for right-shifts
        mov     w9,  w3              // w9 = carry

        subs    x2,  x2,  #16
        b.lt    2f
1:
        dup     v0.16b,  w9          // broadcast carry to all 16 lanes

        ld1     {v1.16b}, [x1], #16  // load 16 input bytes

        // 4-step prefix scan: v1[k] becomes sum(src[0..k]) after all steps.
        // Each step k doubles the range: shift-by-1, shift-by-2, shift-by-4, shift-by-8.
        // ext Vd.16b, Vn(vzero).16b, Vm(vsrc).16b, #(16-shift)
        // result[i] = (i < shift) ? 0 : v1[i-shift]
        ext     v2.16b,  v31.16b, v1.16b,  #15  // shift-by-1
        add     v1.16b,  v1.16b,  v2.16b
        ext     v2.16b,  v31.16b, v1.16b,  #14  // shift-by-2
        add     v1.16b,  v1.16b,  v2.16b
        ext     v2.16b,  v31.16b, v1.16b,  #12  // shift-by-4
        add     v1.16b,  v1.16b,  v2.16b
        ext     v2.16b,  v31.16b, v1.16b,  #8   // shift-by-8
        add     v1.16b,  v1.16b,  v2.16b

        add     v0.16b,  v0.16b,  v1.16b   // out[i] = carry + prefix_sum[i]
        st1     {v0.16b}, [x0], #16
        umov    w9,  v0.b[15]              // new carry = last output byte
        subs    x2,  x2,  #16
        b.ge    1b

2:
        adds    x2,  x2,  #16
        b.eq    4f
3:
        ldrb    w10, [x1], #1
        add     w9,  w9,  w10
        uxtb    w9,  w9
        strb    w9,  [x0], #1
        subs    x2,  x2,  #1
        b.ne    3b
4:
        mov     w0,  w9
        ret
endfunc
