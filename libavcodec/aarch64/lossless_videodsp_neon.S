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

/*
 * void ff_add_gradient_pred_neon(uint8_t *src, ptrdiff_t stride, ptrdiff_t width)
 *
 * In-place gradient predictor for lossless video coding.
 * src[i] = (src[i-stride] - src[i-stride-1] + src[i-1] + src[i]) & 0xFF
 * where src[i-1] is the UPDATED output of the previous iteration.
 *
 * Reformulated as a prefix sum: define g[i] = above[i] - above_left[i] + diff[i].
 * Then out[i] = (carry + g[0] + g[1] + ... + g[i]) & 0xFF,
 * with carry = src[-1] (pixel immediately left of the row start).
 *
 * x0=src (current row, read+write), x1=stride (ptrdiff_t), x2=width (ptrdiff_t)
 * stride is ptrdiff_t (64-bit) — no sxtw needed.
 * void return — no mov w0.
 */
function ff_add_gradient_pred_neon, export=1
        movi    v31.16b, #0                 // zero vector for ext prefix scan

        ldrb    w9,  [x0, #-1]              // initial carry = src[-1]
        sub     x3,  x0,  x1               // x3 = above row (src - stride)
        sub     x4,  x3,  #1               // x4 = above_left row (src - stride - 1)

        subs    x2,  x2,  #16
        b.lt    2f
1:
        ld1     {v1.16b}, [x3], #16         // above[0..15]
        ld1     {v2.16b}, [x4], #16         // above_left[0..15]
        ld1     {v3.16b}, [x0]              // diff[0..15] (current row, not yet updated)

        sub     v1.16b, v1.16b, v2.16b      // above - above_left
        add     v1.16b, v1.16b, v3.16b      // g[i] = above[i] - above_left[i] + diff[i]

        // 4-step Hillis-Steele prefix scan (in-place, same as add_left_pred_neon)
        ext     v2.16b, v31.16b, v1.16b, #15  // shift-by-1
        add     v1.16b, v1.16b, v2.16b
        ext     v2.16b, v31.16b, v1.16b, #14  // shift-by-2
        add     v1.16b, v1.16b, v2.16b
        ext     v2.16b, v31.16b, v1.16b, #12  // shift-by-4
        add     v1.16b, v1.16b, v2.16b
        ext     v2.16b, v31.16b, v1.16b, #8   // shift-by-8
        add     v1.16b, v1.16b, v2.16b

        dup     v0.16b,  w9                  // broadcast carry
        add     v0.16b,  v0.16b, v1.16b      // out[i] = carry + prefix_sum[i]

        st1     {v0.16b}, [x0], #16
        umov    w9,  v0.b[15]                // new carry = last output byte (uint8)

        subs    x2,  x2,  #16
        b.ge    1b
2:
        adds    x2,  x2,  #16
        b.eq    4f
3:
        ldrb    w10, [x3], #1                // above
        ldrb    w11, [x4], #1                // above_left
        ldrb    w12, [x0]                    // diff
        add     w9,  w9,  w10
        sub     w9,  w9,  w11
        add     w9,  w9,  w12
        uxtb    w9,  w9
        strb    w9,  [x0], #1
        subs    x2,  x2,  #1
        b.ne    3b
4:
        ret
endfunc

/*
 * int ff_add_left_pred_int16_neon(uint16_t *dst, const uint16_t *src,
 *                                  unsigned mask, ptrdiff_t w, unsigned acc)
 *
 * Cumulative sum with mask: dst[i] = acc = (acc + src[i]) & mask.
 * Returns final acc (masked).
 *
 * x0=dst, x1=src, w2=mask, x3=w (ptrdiff_t — use 64-bit), w4=acc (initial carry)
 *
 * Vectorized as an 8-element uint16 prefix scan per iteration (3 ext+add steps),
 * then add carry and apply mask. Mask is applied BEFORE extracting carry.
 */
function ff_add_left_pred_int16_neon, export=1
        movi    v31.16b, #0                 // zero vector for ext
        dup     v28.8h,  w2                 // mask broadcast (16-bit)
        mov     w9,  w4                     // w9 = carry (acc)

        subs    x3,  x3,  #8
        b.lt    2f
1:
        ld1     {v1.8h}, [x1], #16          // 8 × uint16 input elements

        // 3-step prefix scan for 8 uint16 lanes
        // shift-by-1 element (2 bytes): ext #(16-2) = #14
        ext     v2.16b, v31.16b, v1.16b, #14
        add     v1.8h,  v1.8h,  v2.8h
        // shift-by-2 elements (4 bytes): ext #(16-4) = #12
        ext     v2.16b, v31.16b, v1.16b, #12
        add     v1.8h,  v1.8h,  v2.8h
        // shift-by-4 elements (8 bytes): ext #(16-8) = #8
        ext     v2.16b, v31.16b, v1.16b, #8
        add     v1.8h,  v1.8h,  v2.8h

        dup     v0.8h,   w9                  // broadcast carry
        add     v0.8h,   v0.8h,  v1.8h       // out[i] = carry + prefix_sum[i]
        and     v0.16b,  v0.16b, v28.16b      // apply mask (BEFORE extracting carry)
        st1     {v0.8h}, [x0], #16
        umov    w9,  v0.h[7]                  // new carry = last masked output

        subs    x3,  x3,  #8
        b.ge    1b
2:
        adds    x3,  x3,  #8
        b.eq    4f
3:
        ldrh    w10, [x1], #2
        add     w9,  w9,  w10
        and     w9,  w9,  w2
        strh    w9,  [x0], #2
        subs    x3,  x3,  #1
        b.ne    3b
4:
        mov     w0,  w9
        ret
endfunc
