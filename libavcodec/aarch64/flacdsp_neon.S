/*
 * AArch64 NEON optimised FLAC DSP functions
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

// void ff_flac_wasted_32_neon(int32_t *decoded, int wasted, int len)
// x0=decoded, w1=wasted, w2=len (multiple of 16)
// Shift each int32 element left by wasted bits in-place.
function ff_flac_wasted_32_neon, export=1
        dup             v0.4s,  w1
1:
        ld1             {v1.4s, v2.4s, v3.4s, v4.4s}, [x0]
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        st1             {v1.4s, v2.4s, v3.4s, v4.4s}, [x0], #64
        subs            w2,  w2,  #16
        b.gt            1b
        ret
endfunc

// void ff_flac_wasted_33_neon(int64_t *decoded, const int32_t *residual,
//                             int wasted, int len)
// x0=decoded(out int64), x1=residual(in int32), w2=wasted, w3=len (multiple of 4)
// Sign-extend each int32 residual to int64, then left-shift by wasted.
function ff_flac_wasted_33_neon, export=1
        sxtw            x2,  w2
        dup             v0.2d,  x2
1:
        ld1             {v1.4s},  [x1], #16
        sshll           v2.2d,  v1.2s,  #0
        sshll2          v3.2d,  v1.4s,  #0
        sshl            v2.2d,  v2.2d,  v0.2d
        sshl            v3.2d,  v3.2d,  v0.2d
        st1             {v2.2d, v3.2d}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// ---------------------------------------------------------------------------
// Decorrelate helpers (S32 output — interleaved int32 stereo)
// Signature: void fn(uint8_t **out, int32_t **in, int channels, int len, int shift)
//            x0=out**, x1=in**, w2=channels(unused), w3=len, w4=shift
// ---------------------------------------------------------------------------

// left-side: out[0]=a<<shift, out[1]=(a-b)<<shift
function ff_flac_decorrelate_ls_32_neon, export=1
        ldr             x2,  [x1, #8]       // in[1]
        ldr             x1,  [x1]           // in[0]
        ldr             x0,  [x0]           // out[0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16
        ld1             {v2.4s}, [x2], #16
        sub             v3.4s,  v1.4s,  v2.4s   // a - b
        sshl            v1.4s,  v1.4s,  v0.4s   // a << shift
        sshl            v3.4s,  v3.4s,  v0.4s   // (a-b) << shift
        zip1            v4.4s,  v1.4s,  v3.4s
        zip2            v5.4s,  v1.4s,  v3.4s
        st1             {v4.4s, v5.4s}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// right-side: out[0]=(a+b)<<shift, out[1]=b<<shift
function ff_flac_decorrelate_rs_32_neon, export=1
        ldr             x2,  [x1, #8]
        ldr             x1,  [x1]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16
        ld1             {v2.4s}, [x2], #16
        add             v3.4s,  v1.4s,  v2.4s   // a + b
        sshl            v3.4s,  v3.4s,  v0.4s   // (a+b) << shift
        sshl            v2.4s,  v2.4s,  v0.4s   // b << shift
        zip1            v4.4s,  v3.4s,  v2.4s
        zip2            v5.4s,  v3.4s,  v2.4s
        st1             {v4.4s, v5.4s}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// mid-side: a -= b>>1; out[0]=(a+b)<<shift, out[1]=a<<shift
function ff_flac_decorrelate_ms_32_neon, export=1
        ldr             x2,  [x1, #8]
        ldr             x1,  [x1]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16      // a
        ld1             {v2.4s}, [x2], #16      // b
        sshr            v3.4s,  v2.4s,  #1      // b >> 1
        sub             v1.4s,  v1.4s,  v3.4s   // a -= b>>1
        add             v3.4s,  v1.4s,  v2.4s   // a + b (left)
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v1.4s,  v1.4s,  v0.4s
        zip1            v4.4s,  v3.4s,  v1.4s
        zip2            v5.4s,  v3.4s,  v1.4s
        st1             {v4.4s, v5.4s}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// ---------------------------------------------------------------------------
// Decorrelate helpers (S16 output — interleaved int16 stereo)
// Same signatures; output narrowed to int16 via sqxtn.
// ---------------------------------------------------------------------------

// left-side S16
function ff_flac_decorrelate_ls_16_neon, export=1
        ldr             x2,  [x1, #8]
        ldr             x1,  [x1]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16
        ld1             {v2.4s}, [x2], #16
        sub             v3.4s,  v1.4s,  v2.4s
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        xtn             v1.4h,  v1.4s
        xtn             v3.4h,  v3.4s
        zip1            v4.4h,  v1.4h,  v3.4h
        zip2            v5.4h,  v1.4h,  v3.4h
        st1             {v4.4h, v5.4h}, [x0], #16
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// right-side S16
function ff_flac_decorrelate_rs_16_neon, export=1
        ldr             x2,  [x1, #8]
        ldr             x1,  [x1]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16
        ld1             {v2.4s}, [x2], #16
        add             v3.4s,  v1.4s,  v2.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        xtn             v3.4h,  v3.4s
        xtn             v2.4h,  v2.4s
        zip1            v4.4h,  v3.4h,  v2.4h
        zip2            v5.4h,  v3.4h,  v2.4h
        st1             {v4.4h, v5.4h}, [x0], #16
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// mid-side S16
function ff_flac_decorrelate_ms_16_neon, export=1
        ldr             x2,  [x1, #8]
        ldr             x1,  [x1]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x1], #16
        ld1             {v2.4s}, [x2], #16
        sshr            v3.4s,  v2.4s,  #1
        sub             v1.4s,  v1.4s,  v3.4s
        add             v3.4s,  v1.4s,  v2.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v1.4s,  v1.4s,  v0.4s
        xtn             v3.4h,  v3.4s
        xtn             v1.4h,  v1.4s
        zip1            v4.4h,  v3.4h,  v1.4h
        zip2            v5.4h,  v3.4h,  v1.4h
        st1             {v4.4h, v5.4h}, [x0], #16
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// void ff_flac_lpc_32_neon(int32_t *decoded, const int coeffs[32],
//                          int pred_order, int qlevel, int len)
// x0=decoded, x1=coeffs, w2=pred_order, w3=qlevel, w4=len
// Vectorizes the inner dot-product loop across pred_order coefficients.
// smlal  v0.2d, vA.2s, vB.2s: v0.d[0] += sext(vA.s[0])*sext(vB.s[0]);
//                              v0.d[1] += sext(vA.s[1])*sext(vB.s[1])
// smlal2 v0.2d, vA.4s, vB.4s: v0.d[0] += sext(vA.s[2])*sext(vB.s[2]);
//                              v0.d[1] += sext(vA.s[3])*sext(vB.s[3])
// Together they process all 4 loaded lanes into v0.2d per SIMD iteration.
// addp d0, v0.2d: d0 = v0.d[0]+v0.d[1] (fold 2-lane int64 accumulator to scalar)
function ff_flac_lpc_32_neon, export=1
        subs    w4,  w4,  w2            // output_count = len - pred_order; sets flags
        b.le    9f
        sxtw    x2,  w2                 // sign-extend pred_order for addr arith
        add     x0,  x0,  x2,  lsl #2  // x0 -> decoded[pred_order] (first output)
1:      movi    v0.2d,  #0              // clear 2-lane int64 SIMD accumulator
        mov     x5,  x1                 // x5 = &coeffs[0], reset each outer iter
        sub     x6,  x0,  x2,  lsl #2  // x6 = &decoded[i-pred_order]
        mov     x7,  x2                 // x7 = remaining coeff count
        mov     x8,  #0                 // x8 = scalar int64 accumulator (tail)
        subs    x7,  x7,  #4
        b.lt    3f
2:      ld1     {v1.4s},  [x5], #16    // 4 coefficients
        ld1     {v2.4s},  [x6], #16    // 4 history samples
        smlal   v0.2d,  v1.2s,  v2.2s // v0.d[0]+=c[0]*h[0]; v0.d[1]+=c[1]*h[1]
        smlal2  v0.2d,  v1.4s,  v2.4s // v0.d[0]+=c[2]*h[2]; v0.d[1]+=c[3]*h[3]
        subs    x7,  x7,  #4
        b.ge    2b
3:      adds    x7,  x7,  #4
        b.eq    5f
4:      ldr     w9,   [x5], #4
        ldr     w10,  [x6], #4
        smull   x9,   w9,  w10          // x9 = (int64)coeff * (int64)hist
        add     x8,  x8,  x9
        subs    x7,  x7,  #1
        b.ne    4b
5:      addp    d0,  v0.2d             // d0 = v0.d[0]+v0.d[1]
        fmov    x9,  d0               // move 64-bit result to GP register
        add     x8,  x8,  x9         // combine SIMD and scalar tail sums
        asr     x8,  x8,  x3         // >> qlevel
        ldr     w9,   [x0]
        add     w9,  w9,  w8         // decoded[i] += (int32)(sum >> qlevel)
        str     w9,   [x0], #4
        subs    w4,  w4,  #1
        b.ne    1b
9:      ret
endfunc

// void ff_flac_lpc_16_neon(int32_t *decoded, const int coeffs[32],
//                          int pred_order, int qlevel, int len)
// x0=decoded, x1=coeffs, w2=pred_order, w3=qlevel, w4=len
// Vectorizes the inner dot-product loop across pred_order coefficients.
// mla v0.4s,v1.4s,v2.4s: v0.s[k] += v1.s[k]*v2.s[k] mod 2^32 for k=0..3
//   matches C behavior: int s0 += (SUINT)c * (SUINT)d  (wraps mod 2^32)
// addv s0, v0.4s: horizontal 32-bit sum of all 4 lanes into scalar s0 (wraps mod 2^32)
function ff_flac_lpc_16_neon, export=1
        subs    w4,  w4,  w2            // output_count = len - pred_order; sets flags
        b.le    9f
        sxtw    x2,  w2
        add     x0,  x0,  x2,  lsl #2
1:      movi    v0.4s,  #0              // clear 4-lane int32 SIMD accumulator
        mov     x5,  x1
        sub     x6,  x0,  x2,  lsl #2
        mov     x7,  x2
        mov     w8,  #0                 // w8 = scalar int32 accumulator (tail)
        subs    x7,  x7,  #4
        b.lt    3f
2:      ld1     {v1.4s},  [x5], #16
        ld1     {v2.4s},  [x6], #16
        mla     v0.4s,  v1.4s,  v2.4s // v0.s[k] += coeff[k]*hist[k] mod 2^32
        subs    x7,  x7,  #4
        b.ge    2b
3:      adds    x7,  x7,  #4
        b.eq    5f
4:      ldr     w9,   [x5], #4
        ldr     w10,  [x6], #4
        madd    w8,  w9,  w10,  w8    // w8 += coeff*hist (32-bit wrapping)
        subs    x7,  x7,  #1
        b.ne    4b
5:      addv    s0,  v0.4s            // horizontal 32-bit sum into scalar s0
        fmov    w9,  s0              // move 32-bit result to GP register
        add     w8,  w8,  w9        // combine SIMD and scalar tail sums
        asr     w8,  w8,  w3        // >> qlevel (32-bit arithmetic right shift)
        ldr     w9,   [x0]
        add     w9,  w9,  w8
        str     w9,   [x0], #4
        subs    w4,  w4,  #1
        b.ne    1b
9:      ret
endfunc
