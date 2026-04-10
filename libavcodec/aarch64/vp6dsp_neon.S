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
 * void ff_vp6_filter_diag4_neon(uint8_t *dst, const uint8_t *src,
 *                               ptrdiff_t stride,
 *                               const int16_t *h_weights,
 *                               const int16_t *v_weights)
 * x0=dst, x1=src, x2=stride, x3=h_weights, x4=v_weights
 *
 * Two-pass separable 4-tap FIR filter:
 *   H pass: 11 rows × 8 pixels → uint8 tmp buffer (8×11=88 bytes, stack-allocated)
 *   V pass: 8 rows × 8 pixels → dst
 *
 * Algorithm (per pass, 8 pixels at once):
 *   sum[i] = s0[i]*w0 + s1[i]*w1 + s2[i]*w2 + s3[i]*w3
 *   out[i] = av_clip_uint8((sum[i] + 64) >> 7)
 *
 * Intermediate data stored as uint8 (post-clip), not int32 like the C reference,
 * saving 3/4 of stack and bus bandwidth. Correctness: vertical pass operates on
 * clipped [0..255] values identical to the C intermediate in that range.
 *
 * VP6 weight range: [-7..128], sum of |weights| ≤ 186.
 * Max sum = 255*186 + 64 = 47,494 → fits in int32 (no overflow).
 *
 * Register plan (all volatile — no callee-saved NEON registers used):
 *   H-pass loop (v0-v7, v16-v18 reused per iteration):
 *     v0    : 16-byte load from src-1
 *     v1-v3 : ext-shifted views for tap offsets +1,+2,+3
 *     v4-v7 : uint16-widened tap values (ushll from lower 8b)
 *     v16   : int32 sum for pixels 0-3
 *     v17   : int32 sum for pixels 4-7
 *   Persistent (setup before loops):
 *     v18   : {64, 64, 64, 64}           — rounding constant
 *     v19   : h_weights.4h               — H weight load scratch
 *     v20-v23 : h_w0..h_w3 broadcast 8h — H weight registers
 *     v24   : v_weights.4h               — V weight load scratch
 *     v25-v28 : v_w0..v_w3 broadcast 8h — V weight registers
 *   V-pass loop:
 *     v0-v3 : 4-register load of 4 tmp rows (8b each)
 *     v4-v7 : uint16-widened rows
 *     v16   : int32 sum for pixels 0-3
 *     v17   : int32 sum for pixels 4-7
 *
 * Stack: sub sp, #96  (11*8=88 bytes tmp, padded to 96 for 16-byte alignment)
 */
function ff_vp6_filter_diag4_neon, export=1
        /* Allocate stack tmp buffer: 11 rows × 8 uint8 = 88 bytes (pad to 96) */
        sub     sp, sp, #96

        /* src -= stride (need 1 row above the 8×8 output block) */
        sub     x1, x1, x2

        /* Load and broadcast H weights */
        ld1     {v19.4h}, [x3]
        dup     v20.8h, v19.h[0]               // h_w0
        dup     v21.8h, v19.h[1]               // h_w1
        dup     v22.8h, v19.h[2]               // h_w2
        dup     v23.8h, v19.h[3]               // h_w3

        /* Load and broadcast V weights */
        ld1     {v24.4h}, [x4]
        dup     v25.8h, v24.h[0]               // v_w0
        dup     v26.8h, v24.h[1]               // v_w1
        dup     v27.8h, v24.h[2]               // v_w2
        dup     v28.8h, v24.h[3]               // v_w3

        /* Rounding constant */
        movi    v18.4s, #64                    // {64, 64, 64, 64}

        /* x5 = tmp write pointer */
        mov     x5, sp

        /* --------------------------------------------------------- */
        /* Horizontal pass: 11 rows, 8 pixels/row                    */
        /* tmp[y*8 .. y*8+7] = av_clip_uint8((src[x-1..x+2]*h_w + 64) >> 7) */
        /* --------------------------------------------------------- */
        mov     w6, #11
.Lhpass_loop:
        /* Load 16 bytes from (src - 1): gives taps at offsets [-1, 0, +1, +2] */
        sub     x9, x1, #1
        ld1     {v0.16b}, [x9]

        /* Create shifted views via ext (wrap at byte 15 is harmless:
         * only bytes [0..7] of each result are used by ushll .8b) */
        ext     v1.16b, v0.16b, v0.16b, #1     // s1: src[0..7] in bytes [0..7]
        ext     v2.16b, v0.16b, v0.16b, #2     // s2: src[1..8]
        ext     v3.16b, v0.16b, v0.16b, #3     // s3: src[2..9]

        /* Zero-extend lower 8 uint8 values to uint16 (treated as int16 by smull:
         * safe because 0..255 < 32768 — unsigned/signed interpretations match) */
        ushll   v4.8h, v0.8b, #0               // s0: src[-1..6] as uint16
        ushll   v5.8h, v1.8b, #0               // s1: src[0..7]
        ushll   v6.8h, v2.8b, #0               // s2: src[1..8]
        ushll   v7.8h, v3.8b, #0               // s3: src[2..9]

        /* sum = s0*h_w0 + s1*h_w1 + s2*h_w2 + s3*h_w3 → int32 × 8 pixels */
        smull   v16.4s, v4.4h, v20.4h          // pixels 0-3: s0*h_w0
        smull2  v17.4s, v4.8h, v20.8h          // pixels 4-7
        smlal   v16.4s, v5.4h, v21.4h
        smlal2  v17.4s, v5.8h, v21.8h
        smlal   v16.4s, v6.4h, v22.4h
        smlal2  v17.4s, v6.8h, v22.8h
        smlal   v16.4s, v7.4h, v23.4h
        smlal2  v17.4s, v7.8h, v23.8h

        /* Add 64, arithmetic shift right 7 */
        add     v16.4s, v16.4s, v18.4s
        add     v17.4s, v17.4s, v18.4s
        sshr    v16.4s, v16.4s, #7
        sshr    v17.4s, v17.4s, #7

        /* Narrow: int32 → uint8 with saturation (implements av_clip_uint8) */
        sqxtun  v16.4h, v16.4s                 // int32 → uint16, signed-saturate
        sqxtun2 v16.8h, v17.4s                 // fill upper 4 halfwords
        uqxtn   v16.8b, v16.8h                 // uint16 → uint8, unsigned-saturate

        /* Store 8 bytes to tmp buffer */
        st1     {v16.8b}, [x5], #8

        add     x1, x1, x2                     // src += stride
        subs    w6, w6, #1
        b.ne    .Lhpass_loop

        /* --------------------------------------------------------- */
        /* Vertical pass: 8 output rows, 8 pixels/row               */
        /* dst[y*stride+x] = av_clip_uint8((tmp_row_4tap * v_w + 64) >> 7) */
        /*                                                           */
        /* For output row y: uses tmp rows [y, y+1, y+2, y+3]       */
        /* (matching C ref: t = tmp+8 makes output row 0 use         */
        /*  rows [0,1,2,3] in our compact uint8 tmp layout)          */
        /* --------------------------------------------------------- */
        mov     x5, sp                          // reset to start of tmp buffer
        mov     w6, #8
.Lvpass_loop:
        /* Load 4 consecutive tmp rows of 8 uint8 bytes (32 bytes total) */
        ld1     {v0.8b, v1.8b, v2.8b, v3.8b}, [x5]

        /* Zero-extend to uint16 */
        ushll   v4.8h, v0.8b, #0               // row y
        ushll   v5.8h, v1.8b, #0               // row y+1
        ushll   v6.8h, v2.8b, #0               // row y+2
        ushll   v7.8h, v3.8b, #0               // row y+3

        /* sum = t0*v_w0 + t1*v_w1 + t2*v_w2 + t3*v_w3 → int32 × 8 pixels */
        smull   v16.4s, v4.4h, v25.4h
        smull2  v17.4s, v4.8h, v25.8h
        smlal   v16.4s, v5.4h, v26.4h
        smlal2  v17.4s, v5.8h, v26.8h
        smlal   v16.4s, v6.4h, v27.4h
        smlal2  v17.4s, v6.8h, v27.8h
        smlal   v16.4s, v7.4h, v28.4h
        smlal2  v17.4s, v7.8h, v28.8h

        add     v16.4s, v16.4s, v18.4s
        add     v17.4s, v17.4s, v18.4s
        sshr    v16.4s, v16.4s, #7
        sshr    v17.4s, v17.4s, #7
        sqxtun  v16.4h, v16.4s
        sqxtun2 v16.8h, v17.4s
        uqxtn   v16.8b, v16.8h

        /* Store 8 pixels to dst, advance dst by stride (handles negative stride) */
        st1     {v16.8b}, [x0], x2

        add     x5, x5, #8                     // advance tmp: next row
        subs    w6, w6, #1
        b.ne    .Lvpass_loop

        /* Restore stack */
        add     sp, sp, #96
        ret
endfunc
