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
