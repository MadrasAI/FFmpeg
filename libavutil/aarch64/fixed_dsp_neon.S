/*
 * ARM NEON optimised fixed-point DSP functions
 * Copyright (c) 2025 FFmpeg ARM64 contributors
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

#include "config.h"
#include "asm.S"

/*
 * Q31 fixed-point multiply: dst[i] = (src0[i] * src1[i] + 0x40000000) >> 31
 *
 * sqrdmulh Vd.4S, Vn.4S, Vm.4S computes (2*a*b + 2^31) >> 32 = (a*b + 2^30) >> 31.
 * This matches the C formula exactly for 24-bit inputs (no saturation possible).
 *
 * void ff_vector_fmul_fixed_neon(int *dst, const int *src0, const int *src1, int len)
 */
function ff_vector_fmul_fixed_neon, export=1
1:      subs            w3,  w3,  #16
        ld1             {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
        ld1             {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
        sqrdmulh        v0.4s,  v0.4s,  v4.4s
        sqrdmulh        v1.4s,  v1.4s,  v5.4s
        sqrdmulh        v2.4s,  v2.4s,  v6.4s
        sqrdmulh        v3.4s,  v3.4s,  v7.4s
        st1             {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
        b.ne            1b
        ret
endfunc

/*
 * dst[i] = src2[i] + (int)((src0[i]*src1[i] + 0x40000000) >> 31)
 *
 * void ff_vector_fmul_add_fixed_neon(int *dst, const int *src0, const int *src1,
 *                                    const int *src2, int len)
 */
function ff_vector_fmul_add_fixed_neon, export=1
1:      subs            w4,  w4,  #8
        ld1             {v0.4s, v1.4s}, [x1], #32
        ld1             {v2.4s, v3.4s}, [x2], #32
        ld1             {v4.4s, v5.4s}, [x3], #32
        sqrdmulh        v0.4s,  v0.4s,  v2.4s
        sqrdmulh        v1.4s,  v1.4s,  v3.4s
        add             v0.4s,  v0.4s,  v4.4s
        add             v1.4s,  v1.4s,  v5.4s
        st1             {v0.4s, v1.4s}, [x0], #32
        b.ne            1b
        ret
endfunc

/*
 * dst[i] = (int)((src0[i] * src1[len-1-i] + 0x40000000) >> 31)   (src1 reversed)
 *
 * void ff_vector_fmul_reverse_fixed_neon(int *dst, const int *src0, const int *src1, int len)
 */
function ff_vector_fmul_reverse_fixed_neon, export=1
        sxtw            x3,  w3
        add             x2,  x2,  x3, lsl #2    // src1 + len*4 = &src1[len]
        sub             x2,  x2,  #16            // &src1[len-4]
        mov             x4,  #-16
1:      subs            w3,  w3,  #4
        ld1             {v0.4s}, [x1], #16       // src0[i..i+3] forward
        ld1             {v1.4s}, [x2], x4        // src1[j-3..j] ascending, then backward
        rev64           v1.4s,  v1.4s
        ext             v1.16b, v1.16b, v1.16b, #8   // → [src1[j], src1[j-1], src1[j-2], src1[j-3]]
        sqrdmulh        v0.4s,  v0.4s,  v1.4s
        st1             {v0.4s}, [x0], #16
        b.ne            1b
        ret
endfunc

/*
 * v1[i] = old_v1[i] + v2[i]
 * v2[i] = old_v1[i] - v2[i]
 *
 * void ff_butterflies_fixed_neon(int *restrict v1, int *restrict v2, int len)
 */
function ff_butterflies_fixed_neon, export=1
1:      subs            w2,  w2,  #8
        ld1             {v0.4s, v1.4s}, [x0]         // v1[i..i+7] (no advance)
        ld1             {v2.4s, v3.4s}, [x1]         // v2[i..i+7]
        add             v4.4s,  v0.4s,  v2.4s        // new v1[i..i+3]
        sub             v5.4s,  v0.4s,  v2.4s        // new v2[i..i+3]
        add             v6.4s,  v1.4s,  v3.4s        // new v1[i+4..i+7]
        sub             v7.4s,  v1.4s,  v3.4s        // new v2[i+4..i+7]
        st1             {v4.4s}, [x0], #16
        st1             {v6.4s}, [x0], #16
        st1             {v5.4s}, [x1], #16
        st1             {v7.4s}, [x1], #16
        b.ne            1b
        ret
endfunc

/*
 * p = 0x40000000; for i: p += v1[i]*v2[i]; return (int)(p >> 31)
 *
 * 0x40000000 is not encodeable as an AArch64 add-immediate; use movz #0x4000 lsl #16.
 * Accumulate into 4×2d accumulators (smull/smlal2), reduce, then add rounding in integer.
 *
 * int ff_scalarproduct_fixed_neon(const int *v1, const int *v2, int len)
 */
function ff_scalarproduct_fixed_neon, export=1
        movi            v16.16b, #0
        movi            v17.16b, #0
        movi            v18.16b, #0
        movi            v19.16b, #0
1:      subs            w2,  w2,  #8
        ld1             {v0.4s, v1.4s}, [x0], #32
        ld1             {v2.4s, v3.4s}, [x1], #32
        smlal           v16.2d, v0.2s, v2.2s
        smlal2          v17.2d, v0.4s, v2.4s
        smlal           v18.2d, v1.2s, v3.2s
        smlal2          v19.2d, v1.4s, v3.4s
        b.ne            1b
        add             v16.2d, v16.2d, v17.2d
        add             v16.2d, v16.2d, v18.2d
        add             v16.2d, v16.2d, v19.2d
        addp            d16,    v16.2d
        mov             x0,     v16.d[0]
        movz            x1,     #0x4000, lsl #16    // 0x40000000 (add-imm form invalid)
        add             x0,     x0,  x1
        asr             x0,     x0,  #31
        ret
endfunc

/*
 * dst[i] = (s0[i]*wj[j]   - s1[j]*wi[i] + R) >> 31  (R = 0x40000000)
 * dst[j] = (s0[i]*wi[i]   + s1[j]*wj[j] + R) >> 31
 *
 * sqrdmulh is inapplicable: rounding covers the combined expression, not each term.
 * Use smull/smull2 + smlsl/smlsl2 for 64-bit intermediates, then sqshrn/sqshrn2.
 *
 * The bidirectional access pattern (src0/wi forward, src1/wj backward) requires
 * rev64+ext reversal on the backward loads and on the backward output store,
 * mirroring ff_vector_fmul_window_neon in float_dsp_neon.S.
 *
 * void ff_vector_fmul_window_fixed_neon(int32_t *dst, const int32_t *src0,
 *     const int32_t *src1, const int32_t *win, int len)
 */
function ff_vector_fmul_window_fixed_neon, export=1
        sxtw            x4,  w4

        // src1 backward: start at src1[len-4]
        sub             x5,  x4,  #4
        add             x6,  x2,  x5, lsl #2     // &src1[len-4]

        // wj backward: win[j] = (win+len)[j] for j=len-1..0 → start at win[2*len-4]
        add             x7,  x3,  x4, lsl #3     // win + 2*len*4
        sub             x7,  x7,  #16            // &win[2*len-4]

        // dst backward: start at dst[2*len-4]
        add             x8,  x0,  x4, lsl #3     // dst + 2*len*4
        sub             x8,  x8,  #16            // &dst[2*len-4]

        mov             x5,  #-16                // backward stride (int32 arrays)

        // Rounding constant R = 0x40000000 in both 64-bit lanes
        movz            x9,  #0x4000, lsl #16
        dup             v31.2d, x9

1:      // 4 forward elements
        ld1             {v0.4s}, [x1], #16       // s0[i..i+3]
        ld1             {v2.4s}, [x3], #16       // wi[i..i+3]

        // 4 backward elements: load ascending, reverse to descending j order
        ld1             {v1.4s}, [x6], x5        // src1[j-3..j] ascending
        rev64           v1.4s,  v1.4s
        ext             v1.16b, v1.16b, v1.16b, #8   // → [s1[j],s1[j-1],s1[j-2],s1[j-3]]

        ld1             {v3.4s}, [x7], x5        // wj[j-3..j] ascending
        rev64           v3.4s,  v3.4s
        ext             v3.16b, v3.16b, v3.16b, #8   // → [wj[j],wj[j-1],wj[j-2],wj[j-3]]

        // Forward output: dst[i..i+3] = (s0*wj - s1*wi + R) >> 31
        smull           v4.2d,  v0.2s, v3.2s     // s0[0,1] * wj[j,j-1]
        smull2          v5.2d,  v0.4s, v3.4s     // s0[2,3] * wj[j-2,j-3]
        smlsl           v4.2d,  v1.2s, v2.2s     // - s1[j,j-1] * wi[0,1]
        smlsl2          v5.2d,  v1.4s, v2.4s     // - s1[j-2,j-3] * wi[2,3]
        add             v4.2d,  v4.2d, v31.2d
        add             v5.2d,  v5.2d, v31.2d
        sqshrn          v16.2s, v4.2d, #31       // signed arithmetic >>31 + narrow
        sqshrn2         v16.4s, v5.2d, #31
        st1             {v16.4s}, [x0], #16

        // Backward output: dst[j..j-3] = (s0*wi + s1*wj + R) >> 31
        smull           v4.2d,  v0.2s, v2.2s     // s0[0,1] * wi[0,1]
        smull2          v5.2d,  v0.4s, v2.4s     // s0[2,3] * wi[2,3]
        smlal           v4.2d,  v1.2s, v3.2s     // + s1[j,j-1] * wj[j,j-1]
        smlal2          v5.2d,  v1.4s, v3.4s     // + s1[j-2,j-3] * wj[j-2,j-3]
        add             v4.2d,  v4.2d, v31.2d
        add             v5.2d,  v5.2d, v31.2d
        sqshrn          v17.2s, v4.2d, #31
        sqshrn2         v17.4s, v5.2d, #31
        // Result lanes: [dst[j], dst[j-1], dst[j-2], dst[j-3]] → reverse before storing backward
        rev64           v17.4s, v17.4s
        ext             v17.16b, v17.16b, v17.16b, #8
        st1             {v17.4s}, [x8], x5

        subs            w4,  w4,  #4
        b.ne            1b
        ret
endfunc

/*
 * dst[i] = av_clip_int16(((s0*wj-s1*wi+R)>>31 + round) >> bits)
 * dst[j] = av_clip_int16(((s0*wi+s1*wj+R)>>31 + round) >> bits)
 *
 * After the Q31 window computation, apply:
 *   - add round (= 1<<(bits-1) if bits>0, else 0)
 *   - arithmetic right shift by bits (sshl with negative register value)
 *   - saturating narrow to int16 (sqxtn)
 *
 * void ff_vector_fmul_window_scaled_fixed_neon(int16_t *dst, const int32_t *src0,
 *     const int32_t *src1, const int32_t *win, int len, uint8_t bits)
 */
function ff_vector_fmul_window_scaled_fixed_neon, export=1
        sxtw            x4,  w4

        // Backward pointers for int32 sources
        sub             x10, x4,  #4
        add             x6,  x2,  x10, lsl #2    // &src1[len-4]
        add             x7,  x3,  x4,  lsl #3
        sub             x7,  x7,  #16            // &win[2*len-4]

        // Backward pointer for int16 dst: (2*len-4)*2 bytes from x0
        add             x8,  x0,  x4,  lsl #2    // x0 + len*4 = x0 + 4*len bytes
        sub             x8,  x8,  #8             // &dst16[2*len-4]

        mov             x10, #-16                // int32 backward stride
        mov             x11, #-8                 // int16 backward stride

        // Rounding constant R = 0x40000000 in both 64-bit lanes
        movz            x9,  #0x4000, lsl #16
        dup             v31.2d, x9

        // bits-scaling: round = (bits > 0) ? (1 << (bits-1)) : 0
        cmp             w5,  #0
        b.eq            2f
        mov             w9,  #1
        lsl             w9,  w9,  w5             // 1 << bits
        lsr             w9,  w9,  #1             // 1 << (bits-1)
        dup             v30.4s, w9
        b               3f
2:      movi            v30.4s, #0
3:
        // Negative shift for sshl: sshl with negative count = arithmetic right shift
        neg             w9,  w5
        dup             v29.4s, w9               // {-bits} in each lane

1:      ld1             {v0.4s}, [x1], #16       // s0[i..i+3]
        ld1             {v2.4s}, [x3], #16       // wi[i..i+3]

        ld1             {v1.4s}, [x6], x10       // src1[j-3..j] ascending
        rev64           v1.4s,  v1.4s
        ext             v1.16b, v1.16b, v1.16b, #8

        ld1             {v3.4s}, [x7], x10       // wj[j-3..j] ascending
        rev64           v3.4s,  v3.4s
        ext             v3.16b, v3.16b, v3.16b, #8

        // Forward: dst16[i..i+3]
        smull           v4.2d,  v0.2s, v3.2s
        smull2          v5.2d,  v0.4s, v3.4s
        smlsl           v4.2d,  v1.2s, v2.2s
        smlsl2          v5.2d,  v1.4s, v2.4s
        add             v4.2d,  v4.2d, v31.2d
        add             v5.2d,  v5.2d, v31.2d
        sqshrn          v16.2s, v4.2d, #31
        sqshrn2         v16.4s, v5.2d, #31
        add             v16.4s, v16.4s, v30.4s   // + round
        sshl            v16.4s, v16.4s, v29.4s   // >> bits (sshl with negative)
        sqxtn           v16.4h, v16.4s            // saturate + narrow to int16
        st1             {v16.4h}, [x0], #8

        // Backward: dst16[j..j-3]
        smull           v4.2d,  v0.2s, v2.2s
        smull2          v5.2d,  v0.4s, v2.4s
        smlal           v4.2d,  v1.2s, v3.2s
        smlal2          v5.2d,  v1.4s, v3.4s
        add             v4.2d,  v4.2d, v31.2d
        add             v5.2d,  v5.2d, v31.2d
        sqshrn          v17.2s, v4.2d, #31
        sqshrn2         v17.4s, v5.2d, #31
        add             v17.4s, v17.4s, v30.4s
        sshl            v17.4s, v17.4s, v29.4s
        sqxtn           v17.4h, v17.4s
        // Reverse 4 int16 elements then store backward
        rev64           v17.4h, v17.4h
        st1             {v17.4h}, [x8], x11

        subs            w4,  w4,  #4
        b.ne            1b
        ret
endfunc
