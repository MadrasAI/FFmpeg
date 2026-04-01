/*
 * AArch64 NEON optimised OpenEXR DSP functions
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

// void ff_reorder_pixels_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t size)
//
// Interleaves the two halves of src into dst:
//   dst[2i]   = src[i]
//   dst[2i+1] = src[half_size + i]
//
// Assumes half_size = size/2 is a multiple of 16 (matches x86 SSE2 assumption).

function ff_reorder_pixels_neon, export=1
        lsr     x2, x2, #1              // half_size = size / 2
        add     x3, x1, x2             // src2 = src + half_size
0:
        ld1     {v0.16b}, [x1], #16    // load 16 bytes from first half
        ld1     {v1.16b}, [x3], #16    // load 16 bytes from second half
        zip1    v2.16b, v0.16b, v1.16b // interleave lower 8 of each
        zip2    v3.16b, v0.16b, v1.16b // interleave upper 8 of each
        stp     q2, q3, [x0], #32      // store 32 interleaved bytes
        subs    x2, x2, #16
        b.gt    0b
        ret
endfunc


// void ff_predictor_neon(uint8_t *src, ptrdiff_t size)
//
// Running prefix-sum with bias: out[i] = in[i] + out[i-1] - 128
// (equivalent to out[i] = carry + sum(in[j] ^ 0x80, j=0..i))
//
// Algorithm (matches x86 SSSE3 predictor):
//   1. XOR input with 0x80 to apply the -128 bias once per element
//   2. Compute parallel prefix-sum within each 16-byte block using
//      four rounds of (ext + add) with shifts of 1, 2, 4, 8 bytes
//   3. Add carry from previous block (all bytes of carry vector = last output)
//   4. Broadcast last byte of result as carry for next block
//
// Initial carry = 0x80 so that out[0] = in[0] (0x80 bias cancels the carry).
// size must be a multiple of 16.
//
// Register map:
//   v0  = XOR mask {0x80,...}
//   v1  = carry (all bytes = carry value from last block)
//   v2  = zero vector (source for ext shifts)
//   v3  = working data / accumulating prefix-sum
//   v4  = shifted temporary

function ff_predictor_neon, export=1
        movi    v0.16b, #0x80           // v0 = XOR mask
        movi    v2.16b, #0              // v2 = zero (for ext shifts)
        mov     v1.16b, v0.16b          // v1 = carry = {0x80,...}
        add     x2, x0, x1             // end = src + size

0:
        ld1     {v3.16b}, [x0]         // load 16 source bytes
        eor     v3.16b, v3.16b, v0.16b // XOR with 0x80

        // Parallel prefix-sum: 4 rounds of (shift-left-N + add)
        // ext vd.16b, va.16b, vb.16b, #n  →  {va[n..15], vb[0..n-1]}
        // ext v4.16b, v2.16b, v3.16b, #(16-N)  shifts v3 left by N bytes

        ext     v4.16b, v2.16b, v3.16b, #15   // v4 = v3 << 1 byte
        add     v3.16b, v3.16b, v4.16b
        ext     v4.16b, v2.16b, v3.16b, #14   // v4 = v3 << 2 bytes
        add     v3.16b, v3.16b, v4.16b
        ext     v4.16b, v2.16b, v3.16b, #12   // v4 = v3 << 4 bytes
        add     v3.16b, v3.16b, v4.16b
        ext     v4.16b, v2.16b, v3.16b, #8    // v4 = v3 << 8 bytes (saved for add below)

        // Accumulate prefix-sum + shifted portion into carry.
        // For positions 0..7:  v4[i]=0,      carry += v3[i]           (correct prefix sum)
        // For positions 8..15: v4[i]=v3[i-8], carry += v3[i]+v3[i-8]  (tiled windows = full sum)
        add     v1.16b, v1.16b, v3.16b
        add     v1.16b, v1.16b, v4.16b

        st1     {v1.16b}, [x0], #16    // store result
        dup     v1.16b, v1.b[15]       // broadcast last byte as carry for next block

        cmp     x0, x2
        b.lt    0b
        ret
endfunc
