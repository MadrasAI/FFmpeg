/*
 * AArch64 NEON optimised UTVideo DSP functions
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

// void ff_restore_rgb_planes_neon(uint8_t *src_r, uint8_t *src_g, uint8_t *src_b,
//     ptrdiff_t linesize_r, ptrdiff_t linesize_g, ptrdiff_t linesize_b,
//     int width, int height)
//
// Reverses YCoCg-R decorrelation for 8-bit RGB planes (in-place on r and b):
//   src_r[i] = src_r[i] + src_g[i] - 0x80
//   src_b[i] = src_b[i] + src_g[i] - 0x80
//   (src_g unchanged)
//
// x0=src_r, x1=src_g, x2=src_b, x3=linesize_r, x4=linesize_g, x5=linesize_b
// w6=width, w7=height
// All vector registers used (v0-v2, v7) are caller-saved (AAPCS64).

function ff_restore_rgb_planes_neon, export=1
        movi    v7.16b, #0x80           // broadcast 0x80 to all 16 bytes

        sxtw    x6, w6                  // width → 64-bit signed
        sxtw    x7, w7                  // height → 64-bit signed

        // End-of-row pointer trick: advance pointers to one past end of row.
        // Inner loop uses negative index x8 in [-width .. 0).
        add     x0, x0, x6
        add     x1, x1, x6
        add     x2, x2, x6

0:      // outer loop: iterate over rows
        neg     x8, x6                  // x8 = -width

1:      // inner loop: 16 bytes (16 uint8) per SIMD iteration
        // Check whether a full 16-byte chunk fits within the remaining row.
        adds    x9, x8, #16
        b.gt    3f                      // partial tail → scalar epilogue

        ldr     q0, [x0, x8]            // load 16 r pixels
        ldr     q1, [x1, x8]            // load 16 g pixels
        ldr     q2, [x2, x8]            // load 16 b pixels
        sub     v1.16b, v1.16b, v7.16b  // g_adj = g - 0x80
        add     v0.16b, v0.16b, v1.16b  // r_out = r + g_adj (uint8 wrapping)
        add     v2.16b, v2.16b, v1.16b  // b_out = b + g_adj (uint8 wrapping)
        str     q0, [x0, x8]            // store r_out
        str     q2, [x2, x8]            // store b_out
        mov     x8, x9                  // x8 += 16
        b.lt    1b                      // continue while x8 < 0
        b       4f                      // row done (x8 == 0)

3:      // scalar epilogue: process remaining (width % 16) pixels one at a time
        ldrb    w10, [x0, x8]           // r
        ldrb    w11, [x1, x8]           // g
        ldrb    w12, [x2, x8]           // b
        sub     w11, w11, #0x80         // g_adj = g - 0x80 (32-bit arithmetic)
        add     w10, w10, w11           // r_out = r + g_adj
        add     w12, w12, w11           // b_out = b + g_adj
        uxtb    w10, w10                // truncate to uint8
        uxtb    w12, w12
        strb    w10, [x0, x8]
        strb    w12, [x2, x8]
        add     x8, x8, #1
        cbnz    x8, 3b                  // loop until x8 == 0

4:      // advance to next row
        add     x0, x0, x3
        add     x1, x1, x4
        add     x2, x2, x5
        subs    x7, x7, #1
        b.gt    0b
        ret
endfunc

// void ff_restore_rgb_planes10_neon(uint16_t *src_r, uint16_t *src_g, uint16_t *src_b,
//     ptrdiff_t linesize_r, ptrdiff_t linesize_g, ptrdiff_t linesize_b,
//     int width, int height)
//
// Reverses YCoCg-R decorrelation for 10-bit RGB planes (uint16 storage, in-place on r and b):
//   src_r[i] = (src_r[i] + src_g[i] - 0x200) & 0x3FF
//   src_b[i] = (src_b[i] + src_g[i] - 0x200) & 0x3FF
//   (src_g unchanged)
//
// linesize and width are in uint16 element units; assembly converts to bytes.
// x0=src_r, x1=src_g, x2=src_b, x3=linesize_r, x4=linesize_g, x5=linesize_b
// w6=width, w7=height

function ff_restore_rgb_planes10_neon, export=1
        mov     w9, #512
        dup     v6.8h, w9               // broadcast 0x0200 to all 8 halfwords
        mov     w9, #1023
        dup     v7.8h, w9               // broadcast 0x03FF to all 8 halfwords

        sxtw    x6, w6                  // width in elements → 64-bit
        sxtw    x7, w7                  // height → 64-bit

        // Convert linesize from uint16 element units to bytes (×2)
        lsl     x3, x3, #1
        lsl     x4, x4, #1
        lsl     x5, x5, #1
        // Width in bytes
        lsl     x6, x6, #1

        // End-of-row pointer trick (all pointers now measured in bytes)
        add     x0, x0, x6
        add     x1, x1, x6
        add     x2, x2, x6

0:      // outer loop: iterate over rows
        neg     x8, x6                  // x8 = -(width in bytes)

1:      // inner loop: 16 bytes (8 uint16) per SIMD iteration
        adds    x9, x8, #16
        b.gt    3f                      // partial tail → scalar epilogue

        ldr     q0, [x0, x8]            // load 8 r pixels
        ldr     q1, [x1, x8]            // load 8 g pixels
        ldr     q2, [x2, x8]            // load 8 b pixels
        sub     v1.8h, v1.8h, v6.8h    // g_adj = g - 0x200
        add     v0.8h, v0.8h, v1.8h    // r_out = r + g_adj
        add     v2.8h, v2.8h, v1.8h    // b_out = b + g_adj
        and     v0.16b, v0.16b, v7.16b // r_out &= 0x03FF (10-bit mask)
        and     v2.16b, v2.16b, v7.16b // b_out &= 0x03FF
        str     q0, [x0, x8]
        str     q2, [x2, x8]
        mov     x8, x9                  // x8 += 16
        b.lt    1b
        b       4f

3:      // scalar epilogue: process remaining (width_bytes % 16) pixels one at a time
        ldrh    w10, [x0, x8]           // r (uint16)
        ldrh    w11, [x1, x8]           // g
        ldrh    w12, [x2, x8]           // b
        sub     w11, w11, #0x200        // g_adj = g - 0x200
        add     w10, w10, w11           // r_out = r + g_adj
        add     w12, w12, w11           // b_out = b + g_adj
        and     w10, w10, #0x3FF        // & 0x3FF
        and     w12, w12, #0x3FF
        strh    w10, [x0, x8]
        strh    w12, [x2, x8]
        add     x8, x8, #2              // advance by 2 bytes (1 uint16)
        cbnz    x8, 3b

4:      // advance to next row
        add     x0, x0, x3
        add     x1, x1, x4
        add     x2, x2, x5
        subs    x7, x7, #1
        b.gt    0b
        ret
endfunc
