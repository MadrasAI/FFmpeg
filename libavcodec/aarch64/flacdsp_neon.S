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

// ---------------------------------------------------------------------------
// Independent channel decorrelate (S32 output — N-channel interleaved int32)
// Signature: void fn(uint8_t **out, int32_t **in, int channels, int len, int shift)
//            x0=out**, x1=in**, w2=channels(unused here), w3=len, w4=shift
// Algorithm: for j in 0..len, for i in 0..N: out[0][j*N+i] = in[i][j] << shift
// ---------------------------------------------------------------------------

// 2-channel S32: st2 {v1.4s, v2.4s} interleaves as ch0,ch1,ch0,ch1,...
// 4 samples per iteration → 2*4*4 = 32 bytes output
function ff_flac_decorrelate_indep2_32_neon, export=1
        ldr             x5,  [x1]           // in[0]
        ldr             x6,  [x1, #8]       // in[1]
        ldr             x0,  [x0]           // out[0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x5], #16
        ld1             {v2.4s}, [x6], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        st2             {v1.4s, v2.4s}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// 4-channel S32: st4 {v1.4s..v4.4s} produces 4-way interleave of 4 samples
// 4 samples per iteration → 4*4*4 = 64 bytes output
function ff_flac_decorrelate_indep4_32_neon, export=1
        ldr             x5,  [x1]
        ldr             x6,  [x1, #8]
        ldr             x7,  [x1, #16]
        ldr             x8,  [x1, #24]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x5], #16
        ld1             {v2.4s}, [x6], #16
        ld1             {v3.4s}, [x7], #16
        ld1             {v4.4s}, [x8], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        st4             {v1.4s, v2.4s, v3.4s, v4.4s}, [x0], #64
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// 8-channel S32: two-level zip transpose, then 4 paired st1 stores.
// Registers: v0=shift, v1-v7=ch0-6, v16=ch7 (all caller-saved, avoid v8-v15).
// Round 1 (.4s): zip pairs → v17-v24 (all caller-saved).
// Round 2 (.2d): zip → v1-v6,v25-v26 (consecutive pairs per sample).
// 4 samples per iteration → 8*4*4 = 128 bytes output.
function ff_flac_decorrelate_indep8_32_neon, export=1
        ldr             x5,  [x1]
        ldr             x6,  [x1, #8]
        ldr             x7,  [x1, #16]
        ldr             x8,  [x1, #24]
        ldr             x9,  [x1, #32]
        ldr             x10, [x1, #40]
        ldr             x11, [x1, #48]
        ldr             x12, [x1, #56]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s},  [x5],  #16
        ld1             {v2.4s},  [x6],  #16
        ld1             {v3.4s},  [x7],  #16
        ld1             {v4.4s},  [x8],  #16
        ld1             {v5.4s},  [x9],  #16
        ld1             {v6.4s},  [x10], #16
        ld1             {v7.4s},  [x11], #16
        ld1             {v16.4s}, [x12], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        sshl            v5.4s,  v5.4s,  v0.4s
        sshl            v6.4s,  v6.4s,  v0.4s
        sshl            v7.4s,  v7.4s,  v0.4s
        sshl            v16.4s, v16.4s, v0.4s
        // Round 1: pair interleave at .4s → v17-v24 (all caller-saved)
        zip1            v17.4s, v1.4s,  v2.4s   // [ch0[0],ch1[0],ch0[1],ch1[1]]
        zip2            v18.4s, v1.4s,  v2.4s   // [ch0[2],ch1[2],ch0[3],ch1[3]]
        zip1            v19.4s, v3.4s,  v4.4s   // [ch2[0],ch3[0],ch2[1],ch3[1]]
        zip2            v20.4s, v3.4s,  v4.4s   // [ch2[2],ch3[2],ch2[3],ch3[3]]
        zip1            v21.4s, v5.4s,  v6.4s   // [ch4[0],ch5[0],ch4[1],ch5[1]]
        zip2            v22.4s, v5.4s,  v6.4s   // [ch4[2],ch5[2],ch4[3],ch5[3]]
        zip1            v23.4s, v7.4s,  v16.4s  // [ch6[0],ch7[0],ch6[1],ch7[1]]
        zip2            v24.4s, v7.4s,  v16.4s  // [ch6[2],ch7[2],ch6[3],ch7[3]]
        // Round 2: quad interleave at .2d — consecutive pairs for each sample
        zip1            v1.2d,  v17.2d, v19.2d  // sample 0 ch0-3
        zip1            v2.2d,  v21.2d, v23.2d  // sample 0 ch4-7
        zip2            v3.2d,  v17.2d, v19.2d  // sample 1 ch0-3
        zip2            v4.2d,  v21.2d, v23.2d  // sample 1 ch4-7
        zip1            v5.2d,  v18.2d, v20.2d  // sample 2 ch0-3
        zip1            v6.2d,  v22.2d, v24.2d  // sample 2 ch4-7
        zip2            v25.2d, v18.2d, v20.2d  // sample 3 ch0-3
        zip2            v26.2d, v22.2d, v24.2d  // sample 3 ch4-7
        // Store: 4 samples x 8 channels x 4 bytes = 128 bytes
        st1             {v1.4s,  v2.4s},  [x0], #32   // sample 0
        st1             {v3.4s,  v4.4s},  [x0], #32   // sample 1
        st1             {v5.4s,  v6.4s},  [x0], #32   // sample 2
        st1             {v25.4s, v26.4s}, [x0], #32   // sample 3
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// ---------------------------------------------------------------------------
// Independent channel decorrelate (S16 output — N-channel interleaved int16)
// Same signature. Input is int32; sshl then xtn narrows to int16 (truncating).
// xtn matches C cast semantics: (int16_t)(int32_t)(x << shift).
// ---------------------------------------------------------------------------

// 2-channel S16: sshl+xtn+st2 .4h — 4 samples × 2 channels × 2 bytes = 16 bytes
function ff_flac_decorrelate_indep2_16_neon, export=1
        ldr             x5,  [x1]
        ldr             x6,  [x1, #8]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x5], #16
        ld1             {v2.4s}, [x6], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        xtn             v1.4h,  v1.4s
        xtn             v2.4h,  v2.4s
        st2             {v1.4h, v2.4h}, [x0], #16
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// 4-channel S16: sshl+xtn+st4 .4h — 4 samples × 4 channels × 2 bytes = 32 bytes
function ff_flac_decorrelate_indep4_16_neon, export=1
        ldr             x5,  [x1]
        ldr             x6,  [x1, #8]
        ldr             x7,  [x1, #16]
        ldr             x8,  [x1, #24]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s}, [x5], #16
        ld1             {v2.4s}, [x6], #16
        ld1             {v3.4s}, [x7], #16
        ld1             {v4.4s}, [x8], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        xtn             v1.4h,  v1.4s
        xtn             v2.4h,  v2.4s
        xtn             v3.4h,  v3.4s
        xtn             v4.4h,  v4.4s
        st4             {v1.4h, v2.4h, v3.4h, v4.4h}, [x0], #32
        subs            w3,  w3,  #4
        b.gt            1b
        ret
endfunc

// 8-channel S16: three-level zip transpose, then 4 st1 {vN.8h} stores.
// Registers: v0=shift, v1-v7=ch0-6, v16=ch7 (avoid callee-saved v8-v15).
// Round 1 (pair at .8h): zip1 → v17-v20.
// Round 2 (quad at .4s): zip1+zip2 → reuse v1-v4.
// Round 3 (8-way at .2d): zip1+zip2 → v5-v7, v16.
// 4 samples per iteration → 4 × 8 × 2 = 64 bytes output.
function ff_flac_decorrelate_indep8_16_neon, export=1
        ldr             x5,  [x1]
        ldr             x6,  [x1, #8]
        ldr             x7,  [x1, #16]
        ldr             x8,  [x1, #24]
        ldr             x9,  [x1, #32]
        ldr             x10, [x1, #40]
        ldr             x11, [x1, #48]
        ldr             x12, [x1, #56]
        ldr             x0,  [x0]
        dup             v0.4s,  w4
1:
        ld1             {v1.4s},  [x5],  #16
        ld1             {v2.4s},  [x6],  #16
        ld1             {v3.4s},  [x7],  #16
        ld1             {v4.4s},  [x8],  #16
        ld1             {v5.4s},  [x9],  #16
        ld1             {v6.4s},  [x10], #16
        ld1             {v7.4s},  [x11], #16
        ld1             {v16.4s}, [x12], #16
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        sshl            v5.4s,  v5.4s,  v0.4s
        sshl            v6.4s,  v6.4s,  v0.4s
        sshl            v7.4s,  v7.4s,  v0.4s
        sshl            v16.4s, v16.4s, v0.4s
        xtn             v1.4h,  v1.4s
        xtn             v2.4h,  v2.4s
        xtn             v3.4h,  v3.4s
        xtn             v4.4h,  v4.4s
        xtn             v5.4h,  v5.4s
        xtn             v6.4h,  v6.4s
        xtn             v7.4h,  v7.4s
        xtn             v16.4h, v16.4s
        // Round 1: pair interleave at .8h → v17-v20
        // v17.8h = [ch0[0],ch1[0],ch0[1],ch1[1],ch0[2],ch1[2],ch0[3],ch1[3]]
        zip1            v17.8h, v1.8h,  v2.8h
        zip1            v18.8h, v3.8h,  v4.8h
        zip1            v19.8h, v5.8h,  v6.8h
        zip1            v20.8h, v7.8h,  v16.8h
        // Round 2: quad interleave at .4s → reuse v1-v4
        // v1.4s = [ch0-3 s0|s0, ch0-3 s1|s1] (2 samples, 4 channels, as .4s lanes of 2×int16)
        zip1            v1.4s,  v17.4s, v18.4s   // ch0-3 samples 0,1
        zip2            v2.4s,  v17.4s, v18.4s   // ch0-3 samples 2,3
        zip1            v3.4s,  v19.4s, v20.4s   // ch4-7 samples 0,1
        zip2            v4.4s,  v19.4s, v20.4s   // ch4-7 samples 2,3
        // Round 3: 8-interleave at .2d → v5-v7, v16
        zip1            v5.2d,  v1.2d,  v3.2d    // sample 0: ch0-7
        zip2            v6.2d,  v1.2d,  v3.2d    // sample 1: ch0-7
        zip1            v7.2d,  v2.2d,  v4.2d    // sample 2: ch0-7
        zip2            v16.2d, v2.2d,  v4.2d    // sample 3: ch0-7
        // Store: 4 samples x 8 channels x 2 bytes = 64 bytes
        st1             {v5.8h},  [x0], #16
        st1             {v6.8h},  [x0], #16
        st1             {v7.8h},  [x0], #16
        st1             {v16.8h}, [x0], #16
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
