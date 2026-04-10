/*
 * Copyright (c) 2026 Ramaseshan M S
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
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
 * void ff_eq_process_neon(const uint8_t *src,  // x0
 *                         uint8_t *dst,         // x1
 *                         short contrast,       // w2 (sign-extended)
 *                         short brightness,     // w3 (sign-extended)
 *                         int w)                // w4
 *
 * Per-pixel: out = clip_uint8((src * contrast >> 12) + brightness)
 *
 * Equivalent to x86 process_one_line_sse2 but using NEON smull/sshr/sqxtn/sqxtun.
 * Main loop: 16 uint8 pixels per iteration.
 * Tail: scalar loop for remaining 0-15 pixels.
 *
 * NEON pipeline per 16 pixels:
 *   ushll  → extend bytes to int16
 *   smull  → int16 * int16 contrast → int32 (4 lanes per smull)
 *   sshr#12 → arithmetic shift right 12
 *   sqxtn  → narrow int32→int16 with signed saturation
 *   add    → add brightness (int16); no overflow for realistic contrast/brightness
 *   sqxtun → signed int16 → uint8 with saturation (0-255 clamp)
 *
 * v16.8h = [contrast]*8, v17.8h = [brightness]*8 (set once at entry)
 */
function ff_eq_process_neon, export=1
        dup     v16.8h, w2          // contrast broadcast to 8 int16 lanes
        dup     v17.8h, w3          // brightness broadcast to 8 int16 lanes
        sxtw    x4, w4              // w → 64-bit

        // Number of bytes in aligned 16-byte chunks
        and     x5, x4, #~15
        cbz     x5, .Leq_tail

.Leq_main:
        ld1     {v0.16b}, [x0], #16

        // Lower 8 bytes: extend to int16, multiply by contrast, shift right 12
        ushll   v1.8h,  v0.8b,   #0       // zero-extend bytes 0-7 to int16
        smull   v2.4s,  v1.4h,  v16.4h    // (byte[0-3] * contrast) → int32
        smull2  v3.4s,  v1.8h,  v16.8h    // (byte[4-7] * contrast) → int32
        sshr    v2.4s,  v2.4s,  #12
        sshr    v3.4s,  v3.4s,  #12
        sqxtn   v2.4h,  v2.4s             // narrow int32→int16, saturate
        sqxtn2  v2.8h,  v3.4s            // pack bytes 4-7 into upper lanes of v2

        // Upper 8 bytes: same treatment
        ushll2  v4.8h,  v0.16b,  #0       // zero-extend bytes 8-15 to int16
        smull   v5.4s,  v4.4h,  v16.4h
        smull2  v6.4s,  v4.8h,  v16.8h
        sshr    v5.4s,  v5.4s,  #12
        sshr    v6.4s,  v6.4s,  #12
        sqxtn   v5.4h,  v5.4s
        sqxtn2  v5.8h,  v6.4s

        // Add brightness (int16 wrapping — safe for realistic contrast/brightness ranges)
        add     v2.8h,  v2.8h,  v17.8h
        add     v5.8h,  v5.8h,  v17.8h

        // Saturate to uint8 [0, 255] and pack 16 bytes
        sqxtun  v2.8b,  v2.8h
        sqxtun2 v2.16b, v5.8h

        st1     {v2.16b}, [x1], #16
        subs    x5, x5, #16
        b.gt    .Leq_main

.Leq_tail:
        // Remaining 0-15 pixels — scalar
        ands    x5, x4, #15
        b.eq    .Leq_end

        sxth    w2, w2              // sign-extend contrast in case upper bits are garbage
        sxth    w3, w3

.Leq_tail_loop:
        ldrb    w6, [x0], #1
        mul     w6, w6, w2          // src * contrast
        asr     w6, w6, #12         // >> 12
        add     w6, w6, w3          // + brightness
        // Clamp to [0, 255]
        cmp     w6, #0
        csel    w6, wzr, w6, lt
        cmp     w6, #255
        mov     w7, #255
        csel    w6, w7,  w6, gt
        strb    w6, [x1], #1
        subs    x5, x5, #1
        b.gt    .Leq_tail_loop

.Leq_end:
        ret
endfunc
