/*
 * AArch64 NEON optimised audio functions
 * Copyright (c) 2008 Mans Rullgard <mans@mansr.com>
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

// int32_t ff_scalarproduct_int16_neon(const int16_t *v1, const int16_t *v2, int len)
// x0 = v1 (no alignment guarantee), x1 = v2 (align 16), w2 = len (multiple of 16)
// Each iteration: consume 32 int16 (2 Q-regs per source), accumulate into v4-v7.4s
function ff_scalarproduct_int16_neon, export=1
        movi            v4.4s,  #0
        movi            v5.4s,  #0
        movi            v6.4s,  #0
        movi            v7.4s,  #0
1:
        ld1             {v0.8h, v1.8h}, [x0], #32
        ld1             {v2.8h, v3.8h}, [x1], #32
        smlal           v4.4s,  v0.4h,  v2.4h
        smlal2          v5.4s,  v0.8h,  v2.8h
        smlal           v6.4s,  v1.4h,  v3.4h
        smlal2          v7.4s,  v1.8h,  v3.8h
        subs            w2,  w2,  #16
        b.gt            1b
        // Reduce 16 int32 lanes (v4-v7) to one scalar
        addp            v4.4s,  v4.4s,  v5.4s
        addp            v6.4s,  v6.4s,  v7.4s
        addp            v4.4s,  v4.4s,  v6.4s
        addv            s0,     v4.4s
        mov             w0,  v0.s[0]
        ret
endfunc

// void ff_vector_clip_int32_neon(int32_t *dst, const int32_t *src,
//                                int32_t min, int32_t max, unsigned int len)
// x0=dst, x1=src, w2=min, w3=max, w4=len (multiple of 8, >0)
function ff_vector_clip_int32_neon, export=1
        dup             v0.4s,  w2
        dup             v1.4s,  w3
1:
        ld1             {v2.4s, v3.4s}, [x1], #32
        smax            v2.4s,  v2.4s,  v0.4s
        smax            v3.4s,  v3.4s,  v0.4s
        smin            v2.4s,  v2.4s,  v1.4s
        smin            v3.4s,  v3.4s,  v1.4s
        st1             {v2.4s, v3.4s}, [x0], #32
        subs            w4,  w4,  #8
        b.gt            1b
        ret
endfunc

// void ff_vector_clipf_neon(float *dst, const float *src,
//                           int len, float min, float max)
// x0=dst, x1=src, w2=len (multiple of 8), s0=min (via AAPCS64), s1=max
function ff_vector_clipf_neon, export=1
        dup             v0.4s,  v0.s[0]
        dup             v1.4s,  v1.s[0]
1:
        ld1             {v2.4s, v3.4s}, [x1], #32
        fmax            v2.4s,  v2.4s,  v0.4s
        fmax            v3.4s,  v3.4s,  v0.4s
        fmin            v2.4s,  v2.4s,  v1.4s
        fmin            v3.4s,  v3.4s,  v1.4s
        st1             {v2.4s, v3.4s}, [x0], #32
        subs            w2,  w2,  #8
        b.gt            1b
        ret
endfunc
