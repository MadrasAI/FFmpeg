/*
 * Copyright (c) 2026 Ramaseshan M S
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
 * void ff_threshold8_neon(const uint8_t *in,        // x0
 *                         const uint8_t *threshold, // x1
 *                         const uint8_t *min,       // x2
 *                         const uint8_t *max,       // x3
 *                         uint8_t *out,             // x4
 *                         ptrdiff_t ilinesize,      // x5
 *                         ptrdiff_t tlinesize,      // x6
 *                         ptrdiff_t flinesize,      // x7
 *                         ptrdiff_t slinesize,      // [sp+0]
 *                         ptrdiff_t olinesize,      // [sp+8]
 *                         int w,                    // [sp+16]
 *                         int h)                    // [sp+24]
 *
 * Per-pixel: out[x] = (in[x] <= threshold[x]) ? min[x] : max[x]
 *
 * w must be a multiple of 16 (8-bit pixels). The C wrapper handles tail.
 *
 * NEON: cmhs  threshold>=in (unsigned) → select mask
 *       bsl   mask=1→min, mask=0→max
 *
 * x19-x23 hold per-row stride residuals (stride - w_bytes) for each pointer.
 * Stack frame: 48 bytes (save x19–x23 + 8-byte pad for 16-byte alignment).
 */
function ff_threshold8_neon, export=1
        // Allocate 48-byte frame; save x19–x23
        stp             x19, x20, [sp, #-48]!
        stp             x21, x22, [sp, #16]
        str             x23,      [sp, #32]

        // Stack-passed args are now at sp+48
        ldp             x8,  x9,  [sp, #48]    // x8=slinesize, x9=olinesize
        ldr             w10, [sp, #64]          // w  (32-bit load: avoids garbage upper bits)
        ldr             w11, [sp, #72]          // h

        sxtw            x10, w10               // w → 64-bit byte count

        // Per-row residuals: advance each pointer by (stride - w) after inner loop
        sub             x19, x5,  x10          // ilinesize - w
        sub             x20, x6,  x10          // tlinesize - w
        sub             x21, x7,  x10          // flinesize - w
        sub             x22, x8,  x10          // slinesize - w
        sub             x23, x9,  x10          // olinesize - w

.Lrow8:
        mov             x12, x10               // column byte counter = w

.Lcol8:
        ld1             {v0.16b}, [x0], #16    // in[col..col+15]
        ld1             {v1.16b}, [x1], #16    // threshold[col..col+15]
        ld1             {v2.16b}, [x2], #16    // min[col..col+15]
        ld1             {v3.16b}, [x3], #16    // max[col..col+15]
        cmhs            v4.16b,  v1.16b, v0.16b  // mask: threshold >= in (unsigned)
        bsl             v4.16b,  v2.16b, v3.16b  // select: mask=1→min, mask=0→max
        st1             {v4.16b}, [x4], #16
        subs            x12, x12, #16
        b.gt            .Lcol8

        // Advance all pointers to next row
        add             x0,  x0,  x19
        add             x1,  x1,  x20
        add             x2,  x2,  x21
        add             x3,  x3,  x22
        add             x4,  x4,  x23
        subs            w11, w11, #1
        b.gt            .Lrow8

        ldp             x21, x22, [sp, #16]
        ldr             x23,      [sp, #32]
        ldp             x19, x20, [sp], #48
        ret
endfunc

/*
 * void ff_threshold16_neon(const uint8_t *in,        // x0 (uint16 data)
 *                          const uint8_t *threshold, // x1
 *                          const uint8_t *min,       // x2
 *                          const uint8_t *max,       // x3
 *                          uint8_t *out,             // x4
 *                          ptrdiff_t ilinesize,      // x5
 *                          ptrdiff_t tlinesize,      // x6
 *                          ptrdiff_t flinesize,      // x7
 *                          ptrdiff_t slinesize,      // [sp+0]
 *                          ptrdiff_t olinesize,      // [sp+8]
 *                          int w,                    // [sp+16] (pixel count)
 *                          int h)                    // [sp+24]
 *
 * w must be a multiple of 8 (uint16 pixels = 16 bytes). C wrapper handles tail.
 * All strides are in bytes (as passed by the caller).
 * cmhs.8h: unsigned compare per uint16 lane.
 */
function ff_threshold16_neon, export=1
        stp             x19, x20, [sp, #-48]!
        stp             x21, x22, [sp, #16]
        str             x23,      [sp, #32]

        ldp             x8,  x9,  [sp, #48]
        ldr             w10, [sp, #64]          // w in uint16 pixels
        ldr             w11, [sp, #72]          // h

        sxtw            x10, w10
        lsl             x10, x10, #1            // w_bytes = w * 2

        sub             x19, x5,  x10
        sub             x20, x6,  x10
        sub             x21, x7,  x10
        sub             x22, x8,  x10
        sub             x23, x9,  x10

.Lrow16:
        mov             x12, x10

.Lcol16:
        ld1             {v0.8h}, [x0], #16
        ld1             {v1.8h}, [x1], #16
        ld1             {v2.8h}, [x2], #16
        ld1             {v3.8h}, [x3], #16
        cmhs            v4.8h,   v1.8h,  v0.8h  // threshold >= in (unsigned uint16)
        bsl             v4.16b,  v2.16b, v3.16b  // select min or max
        st1             {v4.8h}, [x4], #16
        subs            x12, x12, #16
        b.gt            .Lcol16

        add             x0,  x0,  x19
        add             x1,  x1,  x20
        add             x2,  x2,  x21
        add             x3,  x3,  x22
        add             x4,  x4,  x23
        subs            w11, w11, #1
        b.gt            .Lrow16

        ldp             x21, x22, [sp, #16]
        ldr             x23,      [sp, #32]
        ldp             x19, x20, [sp], #48
        ret
endfunc
