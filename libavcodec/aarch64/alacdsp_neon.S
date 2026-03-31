/*
 * AArch64 NEON optimised ALAC DSP functions
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

// void ff_alac_decorrelate_stereo_neon(int32_t *buffer[2], int nb_samples,
//                                      int decorr_shift, int decorr_left_weight)
// x0=buffer[2], w1=nb_samples, w2=decorr_shift, w3=decorr_left_weight
//
// C reference:
//   a = buffer[0][i]; b = buffer[1][i];
//   a -= (int)(b * decorr_left_weight) >> decorr_shift;
//   b += a;               // uses updated a
//   buffer[0][i] = b;     // new_b
//   buffer[1][i] = a;     // new_a
//
// Register plan (all caller-saved):
//   x5=buf0, x6=buf1
//   v0.4s = dup(weight), v1.4s = dup(-shift)
//   v2-v7 = data
// 8 int32s/iter; scalar tail for remainder.
function ff_alac_decorrelate_stereo_neon, export=1
        ldr             x5,  [x0]           // buf0 = buffer[0]
        ldr             x6,  [x0, #8]       // buf1 = buffer[1]
        dup             v0.4s,  w3          // weight splat
        neg             w7,  w2             // -shift (w2=decorr_shift preserved)
        dup             v1.4s,  w7          // -shift splat for sshl right-shift
        subs            w1,  w1,  #8
        b.lt            2f
1:      ld1             {v2.4s, v3.4s}, [x5]            // a[0..7]
        ld1             {v4.4s, v5.4s}, [x6]            // b[0..7]
        mul             v6.4s,  v4.4s,  v0.4s           // b * weight
        mul             v7.4s,  v5.4s,  v0.4s
        sshl            v6.4s,  v6.4s,  v1.4s           // (b*weight) >> shift
        sshl            v7.4s,  v7.4s,  v1.4s
        sub             v2.4s,  v2.4s,  v6.4s           // new_a = a - correction
        sub             v3.4s,  v3.4s,  v7.4s
        add             v4.4s,  v4.4s,  v2.4s           // new_b = b + new_a
        add             v5.4s,  v5.4s,  v3.4s
        st1             {v4.4s, v5.4s}, [x5], #32       // buffer[0] = new_b
        st1             {v2.4s, v3.4s}, [x6], #32       // buffer[1] = new_a
        subs            w1,  w1,  #8
        b.ge            1b
2:      adds            w1,  w1,  #8
        b.eq            9f
        umov            w3,  v0.s[0]        // weight back to GPR for scalar tail
3:      ldr             w8,  [x5]           // a
        ldr             w9,  [x6]           // b
        mul             w10, w9,  w3        // b * weight
        asr             w10, w10, w2        // >> decorr_shift
        sub             w8,  w8,  w10       // new_a = a - correction
        add             w9,  w9,  w8        // new_b = b + new_a
        str             w9,  [x5], #4       // buffer[0] = new_b
        str             w8,  [x6], #4       // buffer[1] = new_a
        subs            w1,  w1,  #1
        b.ne            3b
9:      ret
endfunc

// void ff_alac_append_extra_bits_stereo_neon(int32_t *buffer[2],
//                                            int32_t *extra_bits_buffer[2],
//                                            int extra_bits, int channels,
//                                            int nb_samples)
// x0=buffer[2], x1=extra_bits_buffer[2], w2=extra_bits, w3=channels(=2), w4=nb_samples
//
// C reference: buffer[ch][i] = ((unsigned)buffer[ch][i] << extra_bits) | ebb[ch][i]
// Stereo: process both channels. 8 int32s/iter per channel.
// Register plan:
//   x5=buf0, x6=buf1, x7=ebb0, x8=ebb1
//   v0.4s = dup(extra_bits)
function ff_alac_append_extra_bits_stereo_neon, export=1
        ldr             x5,  [x0]           // buf0
        ldr             x6,  [x0, #8]       // buf1
        ldr             x7,  [x1]           // ebb0
        ldr             x8,  [x1, #8]       // ebb1
        dup             v0.4s,  w2          // extra_bits splat
        subs            w4,  w4,  #8
        b.lt            2f
1:      ld1             {v1.4s, v2.4s}, [x5]            // buf0[0..7]
        ld1             {v3.4s, v4.4s}, [x6]            // buf1[0..7]
        sshl            v1.4s,  v1.4s,  v0.4s           // << extra_bits
        sshl            v2.4s,  v2.4s,  v0.4s
        sshl            v3.4s,  v3.4s,  v0.4s
        sshl            v4.4s,  v4.4s,  v0.4s
        ld1             {v5.4s, v6.4s},   [x7], #32      // ebb0[0..7]
        ld1             {v16.4s, v17.4s}, [x8], #32     // ebb1[0..7]
        orr             v1.16b, v1.16b, v5.16b
        orr             v2.16b, v2.16b, v6.16b
        orr             v3.16b, v3.16b, v16.16b
        orr             v4.16b, v4.16b, v17.16b
        st1             {v1.4s, v2.4s}, [x5], #32
        st1             {v3.4s, v4.4s}, [x6], #32
        subs            w4,  w4,  #8
        b.ge            1b
2:      adds            w4,  w4,  #8
        b.eq            9f
        umov            w2,  v0.s[0]        // extra_bits back to GPR for scalar tail
3:      ldr             w3,  [x5]
        lsl             w3,  w3,  w2
        ldr             w9,  [x7], #4
        orr             w3,  w3,  w9
        str             w3,  [x5], #4
        ldr             w3,  [x6]
        lsl             w3,  w3,  w2
        ldr             w9,  [x8], #4
        orr             w3,  w3,  w9
        str             w3,  [x6], #4
        subs            w4,  w4,  #1
        b.ne            3b
9:      ret
endfunc

// void ff_alac_append_extra_bits_mono_neon(int32_t *buffer[2],
//                                          int32_t *extra_bits_buffer[2],
//                                          int extra_bits, int channels,
//                                          int nb_samples)
// x0=buffer[2], x1=extra_bits_buffer[2], w2=extra_bits, w3=channels(=1), w4=nb_samples
//
// Mono: process channel 0 only. 8 int32s/iter.
function ff_alac_append_extra_bits_mono_neon, export=1
        ldr             x5,  [x0]           // buf0
        ldr             x7,  [x1]           // ebb0
        dup             v0.4s,  w2
        subs            w4,  w4,  #8
        b.lt            2f
1:      ld1             {v1.4s, v2.4s}, [x5]
        sshl            v1.4s,  v1.4s,  v0.4s
        sshl            v2.4s,  v2.4s,  v0.4s
        ld1             {v3.4s, v4.4s}, [x7], #32
        orr             v1.16b, v1.16b, v3.16b
        orr             v2.16b, v2.16b, v4.16b
        st1             {v1.4s, v2.4s}, [x5], #32
        subs            w4,  w4,  #8
        b.ge            1b
2:      adds            w4,  w4,  #8
        b.eq            9f
        umov            w2,  v0.s[0]
3:      ldr             w3,  [x5]
        lsl             w3,  w3,  w2
        ldr             w9,  [x7], #4
        orr             w3,  w3,  w9
        str             w3,  [x5], #4
        subs            w4,  w4,  #1
        b.ne            3b
9:      ret
endfunc
