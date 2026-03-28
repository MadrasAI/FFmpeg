/*
 * AArch64 NEON optimised byte-swap DSP functions
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

// void ff_bswap_buf_neon(uint32_t *dst, const uint32_t *src, int w)
// x0=dst, x1=src, w2=w (count of uint32s)
// Byte-reverse each uint32 element: [b3,b2,b1,b0] -> [b0,b1,b2,b3]
// 8 elements (32 bytes) per SIMD iteration; scalar epilogue for remainder.
function ff_bswap_buf_neon, export=1
        subs    w2,  w2,  #8
        b.lt    2f
1:
        ld1     {v0.4s, v1.4s}, [x1], #32
        rev32   v0.16b, v0.16b
        rev32   v1.16b, v1.16b
        st1     {v0.4s, v1.4s}, [x0], #32
        subs    w2,  w2,  #8
        b.ge    1b
2:
        adds    w2,  w2,  #8
        b.eq    4f
3:
        ldr     w3,  [x1], #4
        rev     w3,  w3
        str     w3,  [x0], #4
        subs    w2,  w2,  #1
        b.ne    3b
4:
        ret
endfunc

// void ff_bswap16_buf_neon(uint16_t *dst, const uint16_t *src, int len)
// x0=dst, x1=src, w2=len (count of uint16s)
// Byte-reverse each uint16 element: [b1,b0] -> [b0,b1]
// 16 elements (32 bytes) per SIMD iteration; scalar epilogue for remainder.
function ff_bswap16_buf_neon, export=1
        subs    w2,  w2,  #16
        b.lt    2f
1:
        ld1     {v0.8h, v1.8h}, [x1], #32
        rev16   v0.16b, v0.16b
        rev16   v1.16b, v1.16b
        st1     {v0.8h, v1.8h}, [x0], #32
        subs    w2,  w2,  #16
        b.ge    1b
2:
        adds    w2,  w2,  #16
        b.eq    4f
3:
        ldrh    w3,  [x1], #2
        rev16   w3,  w3
        strh    w3,  [x0], #2
        subs    w2,  w2,  #1
        b.ne    3b
4:
        ret
endfunc
