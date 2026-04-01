/*
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

/*
 * void ff_dca_lfe_fir0_float_neon(float *pcm_samples, const int32_t *lfe_samples,
 *                                  const float *filter_coeff, ptrdiff_t npcmblocks)
 *
 * dec_select=0: factor=64, ncoeffs=8, nlfesamples=npcmblocks>>1, j=0..31
 *
 * Outer loop: convert 8 int32 LFE samples to float (and reversed), then for each j:
 *   a = dot(coeff[j*8..j*8+7],      [lfe[0],lfe[-1],...,lfe[-7]])  (reversed lfe)
 *   b = dot(coeff[248-j*8..255-j*8],[lfe[-7],...,lfe[-1],lfe[0]])  (forward lfe)
 *
 * Registers (all v0-v7, no callee-save violation):
 *   v0 = lfe_fwd_lo: [lfe[-7],lfe[-6],lfe[-5],lfe[-4]] as float
 *   v1 = lfe_fwd_hi: [lfe[-3],lfe[-2],lfe[-1],lfe[ 0]] as float
 *   v2 = lfe_rev_lo: [lfe[ 0],lfe[-1],lfe[-2],lfe[-3]] as float  (rev of v1)
 *   v3 = lfe_rev_hi: [lfe[-4],lfe[-5],lfe[-6],lfe[-7]] as float  (rev of v0)
 *   v4,v5 = coeff_a or coeff_b (loaded per j)
 *   v6 = dot-product accumulator for a (then reduction result [a_sum, b_sum, ...])
 *   v7 = dot-product accumulator for b
 */
function ff_dca_lfe_fir0_float_neon, export=1
        lsr             x4,  x3,  #1              // nlfesamples = npcmblocks / 2
        cbz             x4,  9f
        add             x8,  x2,  #(248*4)        // coeff_b_base = &coeff[248]
0:
        sub             x9,  x1,  #28             // &lfe[-7]
        ld1             {v0.4s, v1.4s}, [x9]      // v0=[lfe[-7..-4]], v1=[lfe[-3..0]] int32
        scvtf           v0.4s, v0.4s
        scvtf           v1.4s, v1.4s
        // Reverse v1 → v2: rev64.4s swaps pairs within each 64-bit lane,
        // then ext.16b #8 swaps the two 64-bit halves → full 4-element reversal.
        rev64           v2.4s, v1.4s              // [lfe[-2],lfe[-3],lfe[0],lfe[-1]]
        ext             v2.16b, v2.16b, v2.16b, #8 // [lfe[0],lfe[-1],lfe[-2],lfe[-3]]
        rev64           v3.4s, v0.4s              // [lfe[-6],lfe[-7],lfe[-4],lfe[-5]]
        ext             v3.16b, v3.16b, v3.16b, #8 // [lfe[-4],lfe[-5],lfe[-6],lfe[-7]]

        mov             x5,  xzr                  // j = 0
        mov             x6,  x2                   // coeff_a_ptr = &coeff[0]
        mov             x7,  x8                   // coeff_b_ptr = &coeff[248]

1:      // j loop (32 iterations)
        ld1             {v4.4s, v5.4s}, [x6], #32  // coeff_a[j*8..j*8+7], advance ptr
        fmul            v6.4s, v4.4s, v2.4s        // a  = coeff_a[0..3] * lfe_rev_lo
        fmla            v6.4s, v5.4s, v3.4s        // a += coeff_a[4..7] * lfe_rev_hi

        ld1             {v4.4s, v5.4s}, [x7]       // coeff_b[248-j*8..255-j*8]
        sub             x7,  x7,  #32
        fmul            v7.4s, v4.4s, v0.4s        // b  = coeff_b[0..3] * lfe_fwd_lo
        fmla            v7.4s, v5.4s, v1.4s        // b += coeff_b[4..7] * lfe_fwd_hi

        faddp           v6.4s, v6.4s, v7.4s        // [a0+a1, a2+a3, b0+b1, b2+b3]
        faddp           v6.4s, v6.4s, v6.4s        // [a_sum, b_sum, a_sum, b_sum]

        str             s6,  [x0, x5, lsl #2]      // pcm[j] = a_sum
        add             x9,  x0,  #128             // &pcm[32]
        add             x9,  x9,  x5, lsl #2
        st1             {v6.s}[1], [x9]            // pcm[32+j] = b_sum

        add             x5,  x5,  #1
        cmp             x5,  #32
        b.lo            1b

        add             x1,  x1,  #4              // lfe_samples++
        add             x0,  x0,  #256            // pcm_samples += 64 floats
        subs            x4,  x4,  #1
        b.ne            0b
9:
        ret
endfunc

/*
 * void ff_dca_lfe_fir1_float_neon(float *pcm_samples, const int32_t *lfe_samples,
 *                                  const float *filter_coeff, ptrdiff_t npcmblocks)
 *
 * dec_select=1: factor=128, ncoeffs=4, nlfesamples=npcmblocks>>2, j=0..63
 *
 * v0 = lfe_fwd: [lfe[-3],lfe[-2],lfe[-1],lfe[0]] as float
 * v1 = lfe_rev: [lfe[0],lfe[-1],lfe[-2],lfe[-3]] as float  (rev of v0)
 * v2 = coeff_a[j*4..j*4+3] (per j)
 * v4 = dot accumulator for a
 * v5 = dot accumulator for b
 */
function ff_dca_lfe_fir1_float_neon, export=1
        lsr             x4,  x3,  #2              // nlfesamples = npcmblocks / 4
        cbz             x4,  9f
        add             x8,  x2,  #(252*4)        // coeff_b_base = &coeff[252]
0:
        sub             x9,  x1,  #12             // &lfe[-3]
        ld1             {v0.4s}, [x9]              // v0=[lfe[-3],lfe[-2],lfe[-1],lfe[0]] int32
        scvtf           v0.4s, v0.4s
        // Reverse v0 → v1 (single 4S vector)
        rev64           v1.4s, v0.4s              // [lfe[-2],lfe[-3],lfe[0],lfe[-1]]
        ext             v1.16b, v1.16b, v1.16b, #8 // [lfe[0],lfe[-1],lfe[-2],lfe[-3]]

        mov             x5,  xzr                  // j = 0
        mov             x6,  x2                   // coeff_a_ptr = &coeff[0]
        mov             x7,  x8                   // coeff_b_ptr = &coeff[252]

1:      // j loop (64 iterations)
        ld1             {v2.4s}, [x6], #16        // coeff_a[j*4..j*4+3], advance ptr
        fmul            v4.4s, v2.4s, v1.4s       // a = coeff_a * lfe_rev

        ld1             {v2.4s}, [x7]
        sub             x7,  x7,  #16             // coeff_b_ptr -= 4 floats
        fmul            v5.4s, v2.4s, v0.4s       // b = coeff_b * lfe_fwd

        faddp           v4.4s, v4.4s, v5.4s       // [a0+a1, a2+a3, b0+b1, b2+b3]
        faddp           v4.4s, v4.4s, v4.4s       // [a_sum, b_sum, a_sum, b_sum]

        str             s4,  [x0, x5, lsl #2]     // pcm[j] = a_sum
        add             x9,  x0,  #256            // &pcm[64]
        add             x9,  x9,  x5, lsl #2
        st1             {v4.s}[1], [x9]           // pcm[64+j] = b_sum

        add             x5,  x5,  #1
        cmp             x5,  #64
        b.lo            1b

        add             x1,  x1,  #4              // lfe_samples++
        add             x0,  x0,  #512            // pcm_samples += 128 floats
        subs            x4,  x4,  #1
        b.ne            0b
9:
        ret
endfunc

/*
 * void ff_dca_lfe_fir_fixed_neon(int32_t *pcm_samples, const int32_t *lfe_samples,
 *                                const int32_t *filter_coeff, ptrdiff_t npcmblocks)
 *
 * Fixed-point FIR: 8 int32 coefficients, 8 int32 LFE samples, int64 accumulation.
 * norm23(a) = (a + (1<<22)) >> 23
 * clip23(a) = av_clip_intp2(a, 23)
 *
 * v0 = lfe_fwd_lo: [lfe[-7],lfe[-6],lfe[-5],lfe[-4]] as int32
 * v1 = lfe_fwd_hi: [lfe[-3],lfe[-2],lfe[-1],lfe[ 0]] as int32
 * v2 = lfe_rev_lo: [lfe[ 0],lfe[-1],lfe[-2],lfe[-3]] as int32  (rev of v1)
 * v3 = lfe_rev_hi: [lfe[-4],lfe[-5],lfe[-6],lfe[-7]] as int32  (rev of v0)
 * v4,v5 = coeff_a or coeff_b (per j, then reused as 2D temp)
 * v6 = int64 accumulator for a (then [a_sum, b_sum] after addp)
 * v7 = int64 accumulator for b (then round constant after addp)
 * x8 = round constant (1<<22) for norm23
 */
function ff_dca_lfe_fir_fixed_neon, export=1
        lsr             x4,  x3,  #1              // nlfesamples = npcmblocks / 2
        cbz             x4,  9f
        mov             x8,  #(1 << 22)           // round constant for norm23
        add             x10, x2,  #(248*4)        // coeff_b_base = &coeff[248]
0:
        sub             x9,  x1,  #28             // &lfe[-7]
        ld1             {v0.4s, v1.4s}, [x9]      // v0=[lfe[-7..-4]], v1=[lfe[-3..0]] int32
        // Reverse: same two-step per-register technique as float case
        rev64           v2.4s, v1.4s              // [lfe[-2],lfe[-3],lfe[0],lfe[-1]]
        ext             v2.16b, v2.16b, v2.16b, #8 // [lfe[0],lfe[-1],lfe[-2],lfe[-3]]
        rev64           v3.4s, v0.4s              // [lfe[-6],lfe[-7],lfe[-4],lfe[-5]]
        ext             v3.16b, v3.16b, v3.16b, #8 // [lfe[-4],lfe[-5],lfe[-6],lfe[-7]]

        mov             x5,  xzr                  // j = 0
        mov             x6,  x2                   // coeff_a_ptr = &coeff[0]
        mov             x7,  x10                  // coeff_b_ptr = &coeff[248]

1:      // j loop (32 iterations)
        // --- Compute a = sum_{k=0}^{7} coeff_a[k] * lfe_rev[k] ---
        ld1             {v4.4s, v5.4s}, [x6], #32 // coeff_a[j*8..j*8+7]
        smull           v6.2d, v4.2s, v2.2s       // [c0*l0, c1*l1]
        smlal2          v6.2d, v4.4s, v2.4s       // += [c2*l2, c3*l3]
        smull           v7.2d, v5.2s, v3.2s       // [c4*l4, c5*l5]
        smlal2          v7.2d, v5.4s, v3.4s       // += [c6*l6, c7*l7]
        add             v6.2d, v6.2d, v7.2d       // v6 = [a_even, a_odd]

        // --- Compute b = sum_{k=0}^{7} coeff_b[k] * lfe_fwd[k] ---
        ld1             {v4.4s, v5.4s}, [x7]      // coeff_b[248-j*8..255-j*8]
        sub             x7,  x7,  #32
        smull           v7.2d, v4.2s, v0.2s       // [c248*l[-7], c249*l[-6]]
        smlal2          v7.2d, v4.4s, v0.4s       // += [c250*l[-5], c251*l[-4]]
        smull           v4.2d, v5.2s, v1.2s       // reuse v4: [c252*l[-3], c253*l[-2]]
        smlal2          v4.2d, v5.4s, v1.4s       // += [c254*l[-1], c255*l[0]]
        add             v7.2d, v7.2d, v4.2d       // v7 = [b_even, b_odd]

        // --- addp to get [a_sum, b_sum] ---
        addp            v6.2d, v6.2d, v7.2d       // [a_sum, b_sum]

        // --- norm23: (x + (1<<22)) >> 23 ---
        dup             v7.2d, x8                 // [1<<22, 1<<22]
        add             v6.2d, v6.2d, v7.2d
        sshr            v6.2d, v6.2d, #23

        // --- clip23: av_clip_intp2(x, 23) via sqxtn + sqshl/sshr ---
        sqxtn           v6.2s, v6.2d              // int64 → int32 with saturation
        sqshl           v6.2s, v6.2s, #8          // saturating left shift 8: clips at 2^23
        sshr            v6.2s, v6.2s, #8          // arithmetic right shift 8: restores scale

        str             s6,  [x0, x5, lsl #2]     // pcm[j] = a_clipped
        add             x9,  x0,  #128            // &pcm[32]
        add             x9,  x9,  x5, lsl #2
        st1             {v6.s}[1], [x9]           // pcm[32+j] = b_clipped

        add             x5,  x5,  #1
        cmp             x5,  #32
        b.lo            1b

        add             x1,  x1,  #4              // lfe_samples++
        add             x0,  x0,  #256            // pcm_samples += 64 int32
        subs            x4,  x4,  #1
        b.ne            0b
9:
        ret
endfunc
