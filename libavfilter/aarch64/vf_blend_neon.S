/*
 * AArch64 NEON optimised blend filter functions
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

/*
 * Blend function signature (all modes, 8-bit and 16-bit):
 *
 * void ff_blend_<mode>_neon(
 *     const uint8_t *top,             x0
 *     ptrdiff_t      top_linesize,    x1
 *     const uint8_t *bottom,          x2
 *     ptrdiff_t      bottom_linesize, x3
 *     uint8_t       *dst,             x4
 *     ptrdiff_t      dst_linesize,    x5
 *     ptrdiff_t      width,           x6  (bytes)
 *     ptrdiff_t      height,          x7  (rows)
 *     FilterParams  *param,           [sp+0]  (unused — opacity==1 at dispatch)
 *     SliceParams   *sliceparam)      [sp+8]  (unused)
 *
 * Register allocation within functions:
 *   x8  = top row pointer
 *   x9  = bottom row pointer
 *   x10 = dst row pointer
 *   x11 = width (remaining bytes for current row)
 *   x12 = loop counter (pairs of 16, then tail bytes/elements)
 *   x13, x14, x15, x16 = scalar temporaries
 *
 * All NEON registers v0–v7 (caller-saved) used freely.
 *
 * Inner loop: process 16 bytes per iteration.
 * Tail: process 1 byte at a time (or 1 uint16 = 2 bytes for 16-bit modes).
 */

/* ── xor ─────────────────────────────────────────────────────────── */
function ff_blend_xor_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        eor     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        eor     w13, w13, w14
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── or ──────────────────────────────────────────────────────────── */
function ff_blend_or_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        orr     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        orr     w13, w13, w14
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── and ─────────────────────────────────────────────────────────── */
function ff_blend_and_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        and     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        and     w13, w13, w14
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── addition: saturating add, clip to [0,255] ───────────────────── */
function ff_blend_addition_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        uqadd   v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        mov     w15, #255
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14
        cmp     w13, #255
        csel    w13, w13, w15, ls
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── subtract: saturating sub, clip to [0,255] ───────────────────── */
function ff_blend_subtract_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        uqsub   v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        subs    w13, w13, w14
        csel    w13, w13, wzr, pl
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── darken: min(a,b) ───────────────────────────────────────────── */
function ff_blend_darken_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        umin    v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        cmp     w13, w14
        csel    w13, w13, w14, lo
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── lighten: max(a,b) ──────────────────────────────────────────── */
function ff_blend_lighten_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        umax    v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        cmp     w13, w14
        csel    w13, w13, w14, hi
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── average: floor((a+b)/2)
 * Algebraically equivalent to x86 pavgb-with-double-XOR.
 * uhadd computes floor((a+b)/2) directly.
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_average_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        uhadd   v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14
        lsr     w13, w13, #1
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── difference: |a - b|
 * uabd is exact unsigned absolute difference.
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_difference_neon, export=1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        uabd    v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        subs    w13, w13, w14
        cneg    w13, w13, mi
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── grainextract: clip(a + 128 - b) → saturating uint8
 * 16-bit intermediates: ushll, add 128, sub b, sqxtun (signed → unsigned clamp)
 * sqxtun is correct: intermediate can be negative (e.g. a=0, b=255 → -127)
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_grainextract_neon, export=1
        cbz     x7, 99f
        movi    v7.8h, #128
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        ushll   v2.8h, v0.8b,  #0
        ushll   v3.8h, v1.8b,  #0
        add     v2.8h, v2.8h, v7.8h    // top + 128
        sub     v2.8h, v2.8h, v3.8h    // - bottom  (signed result)
        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        add     v4.8h, v4.8h, v7.8h
        sub     v4.8h, v4.8h, v5.8h
        sqxtun  v0.8b,  v2.8h          // signed saturating narrow to uint8
        sqxtun2 v0.16b, v4.8h
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        mov     w15, #255
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, #128
        sub     w13, w13, w14
        cmp     w13, #0
        csel    w13, w13, wzr, ge
        cmp     w13, w15
        csel    w13, w13, w15, ls
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── grainmerge: clip(a + b - 128)
 * Same 16-bit approach: intermediate can be > 255 or < 0.
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_grainmerge_neon, export=1
        cbz     x7, 99f
        movi    v7.8h, #128
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        ushll   v2.8h, v0.8b,  #0
        ushll   v3.8h, v1.8b,  #0
        add     v2.8h, v2.8h, v3.8h    // a + b
        sub     v2.8h, v2.8h, v7.8h    // - 128
        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        add     v4.8h, v4.8h, v5.8h
        sub     v4.8h, v4.8h, v7.8h
        sqxtun  v0.8b,  v2.8h
        sqxtun2 v0.16b, v4.8h
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        mov     w15, #255
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14
        sub     w13, w13, #128
        cmp     w13, #0
        csel    w13, w13, wzr, ge
        cmp     w13, w15
        csel    w13, w13, w15, ls
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── multiply: floor(a*b/255) ≈ (a*b+1+(a*b+1>>8))>>8
 * Same approximation as x86 MULTIPLY macro.
 * umull expands 8 uint8 lanes to 8 uint16 lanes.
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_multiply_neon, export=1
        cbz     x7, 99f
        movi    v7.8h, #1
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        umull   v2.8h, v0.8b,  v1.8b   // a*b, range [0,65025]
        add     v2.8h, v2.8h, v7.8h    // +1
        ushr    v3.8h, v2.8h, #8       // >>8
        add     v2.8h, v2.8h, v3.8h    // +((a*b+1)>>8)
        ushr    v2.8h, v2.8h, #8       // >>8 → approx a*b/255 in [0,255]
        umull2  v4.8h, v0.16b, v1.16b
        add     v4.8h, v4.8h, v7.8h
        ushr    v5.8h, v4.8h, #8
        add     v4.8h, v4.8h, v5.8h
        ushr    v4.8h, v4.8h, #8
        xtn     v0.8b,  v2.8h          // narrow: values in [0,255]
        xtn2    v0.16b, v4.8h
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        mul     w13, w13, w14
        add     w13, w13, #1
        lsr     w14, w13, #8
        add     w13, w13, w14
        lsr     w13, w13, #8
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── screen: 255 - (255-a)*(255-b)/255 = mvn(multiply(mvn(a),mvn(b)))
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_screen_neon, export=1
        cbz     x7, 99f
        movi    v7.8h, #1
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        mvn     v0.16b, v0.16b          // 255 - a
        mvn     v1.16b, v1.16b          // 255 - b
        umull   v2.8h, v0.8b,  v1.8b
        add     v2.8h, v2.8h, v7.8h
        ushr    v3.8h, v2.8h, #8
        add     v2.8h, v2.8h, v3.8h
        ushr    v2.8h, v2.8h, #8
        umull2  v4.8h, v0.16b, v1.16b
        add     v4.8h, v4.8h, v7.8h
        ushr    v5.8h, v4.8h, #8
        add     v4.8h, v4.8h, v5.8h
        ushr    v4.8h, v4.8h, #8
        xtn     v0.8b,  v2.8h
        xtn2    v0.16b, v4.8h
        mvn     v0.16b, v0.16b          // 255 - product
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        eor     w13, w13, #255          // 255 - a
        eor     w14, w14, #255          // 255 - b
        mul     w13, w13, w14
        add     w13, w13, #1
        lsr     w14, w13, #8
        add     w13, w13, w14
        lsr     w13, w13, #8
        eor     w13, w13, #255          // 255 - product
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── hardmix: (a+b >= 255) ? 255 : 0
 * 16-bit: ushll + add + cmhi #254 (> 254 ↔ >= 255) + xtn
 * Specialist correction: cmhi #254, NOT #255 (off-by-one at a+b=255 boundary)
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_hardmix_neon, export=1
        cbz     x7, 99f
        movi    v6.8h, #255             // constant 255 for cmhs comparison
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        ushll   v2.8h, v0.8b,  #0
        ushll   v3.8h, v1.8b,  #0
        add     v2.8h, v2.8h, v3.8h    // sum_lo ∈ [0, 510]
        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        add     v4.8h, v4.8h, v5.8h    // sum_hi
        // cmhs: unsigned >=, fires when sum >= 255 ✓ (specialist-corrected boundary)
        cmhs    v2.8h, v2.8h, v6.8h
        cmhs    v4.8h, v4.8h, v6.8h
        // 0xFFFF → 0xFF (255), 0x0000 → 0x00 (0)
        xtn     v0.8b,  v2.8h
        xtn2    v0.16b, v4.8h
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        mov     w15, #255
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14
        cmp     w13, #255
        csel    w13, w15, wzr, hs      // 255 if sum >= 255, else 0
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── phoenix: min(a,b) - max(a,b) + 255 = 255 - |a-b|
 * NEON: umin, umax, movi 255, uqsub(255-max), uqadd(+min)
 * Specialist note: explicit constant register for uqsub (no imm operand)
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_phoenix_neon, export=1
        cbz     x7, 99f
        movi    v7.16b, #255
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        umin    v2.16b, v0.16b, v1.16b  // min(a,b)
        umax    v3.16b, v0.16b, v1.16b  // max(a,b)
        uqsub   v3.16b, v7.16b, v3.16b  // 255 - max (always >= 0)
        uqadd   v3.16b, v3.16b, v2.16b  // + min (always <= 255)
        st1     {v3.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        cmp     w13, w14
        csel    w15, w13, w14, lo       // min
        csel    w16, w13, w14, hi       // max
        mov     w17, #255
        sub     w17, w17, w16           // 255 - max
        add     w17, w17, w15           // + min → in [0,255]
        strb    w17, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── extremity: |(a+b) - 255|
 * 16-bit: ushll + add + uabd vs constant 255 + xtn
 * Result ∈ [0,255] → safe narrow with xtn (no saturation)
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_extremity_neon, export=1
        cbz     x7, 99f
        movi    v7.8h, #255
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        ushll   v2.8h, v0.8b,  #0      // a_lo
        ushll   v3.8h, v1.8b,  #0      // b_lo
        add     v2.8h, v2.8h, v3.8h    // a+b ∈ [0,510]
        uabd    v2.8h, v2.8h, v7.8h    // |(a+b) - 255| ∈ [0,255]
        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        add     v4.8h, v4.8h, v5.8h
        uabd    v4.8h, v4.8h, v7.8h
        xtn     v0.8b,  v2.8h          // safe: values fit in uint8
        xtn2    v0.16b, v4.8h
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14           // a+b
        subs    w13, w13, #255          // a+b - 255 (with sign in flags)
        cneg    w13, w13, mi            // |a+b - 255|
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── negation: 255 - |(a+b) - 255| = 255 - extremity(a,b)
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_negation_neon, export=1
        cbz     x7, 99f
        movi    v7.8h,  #255
        movi    v6.16b, #255
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        ushll   v2.8h, v0.8b,  #0
        ushll   v3.8h, v1.8b,  #0
        add     v2.8h, v2.8h, v3.8h
        uabd    v2.8h, v2.8h, v7.8h    // extremity_lo
        ushll2  v4.8h, v0.16b, #0
        ushll2  v5.8h, v1.16b, #0
        add     v4.8h, v4.8h, v5.8h
        uabd    v4.8h, v4.8h, v7.8h    // extremity_hi
        xtn     v0.8b,  v2.8h
        xtn2    v0.16b, v4.8h          // extremity result
        uqsub   v0.16b, v6.16b, v0.16b // 255 - extremity
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrb    w13, [x8], #1
        ldrb    w14, [x9], #1
        add     w13, w13, w14
        subs    w13, w13, #255
        cneg    w13, w13, mi            // extremity
        mov     w14, #255
        sub     w13, w14, w13           // 255 - extremity
        strb    w13, [x10], #1
        subs    x12, x12, #1
        b.ne    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ════════════════════════════════════════════════════════════════════
 * 16-BIT MODES (depth == 16)
 * Width is in bytes (caller doubles pixel count before passing).
 * Inner loop: 16 bytes = 8 uint16 lanes per iteration.
 * Tail: 2 bytes = 1 uint16 at a time.
 * ════════════════════════════════════════════════════════════════════ */

/* ── addition_16 ────────────────────────────────────────────────── */
function ff_blend_addition_16_neon, export=1
        // width is in pixels; convert to bytes (x86 does: add widthq, widthq)
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        uqadd   v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, w14
        cmp     w13, w15
        csel    w13, w13, w15, ls
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── subtract_16 ────────────────────────────────────────────────── */
function ff_blend_subtract_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        uqsub   v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        subs    w13, w13, w14
        csel    w13, w13, wzr, pl
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── and_16 ─────────────────────────────────────────────────────── */
function ff_blend_and_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        and     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        and     w13, w13, w14
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── or_16 ──────────────────────────────────────────────────────── */
function ff_blend_or_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        orr     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        orr     w13, w13, w14
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── xor_16 ─────────────────────────────────────────────────────── */
function ff_blend_xor_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.16b}, [x8], #16
        ld1     {v1.16b}, [x9], #16
        eor     v0.16b, v0.16b, v1.16b
        st1     {v0.16b}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        eor     w13, w13, w14
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── average_16: floor((a+b)/2) using uhadd on .8h ──────────────── */
function ff_blend_average_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        uhadd   v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, w14
        lsr     w13, w13, #1
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── darken_16: umin on .8h ─────────────────────────────────────── */
function ff_blend_darken_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        umin    v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        cmp     w13, w14
        csel    w13, w13, w14, lo
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── lighten_16: umax on .8h ────────────────────────────────────── */
function ff_blend_lighten_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        umax    v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        cmp     w13, w14
        csel    w13, w13, w14, hi
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── grainextract_16: clip_u16(a + 32768 - b)
 * 32-bit intermediates: ushll→u32, add 32768, sub b, sqxtun→u16
 * sqxtun: signed int32 → saturating uint16 (clamp to [0,65535])
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_grainextract_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
        movz    w13, #0x8000            // 32768
        dup     v7.4s, w13
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        ushll   v2.4s, v0.4h, #0       // top_lo → u32
        ushll   v3.4s, v1.4h, #0       // bot_lo → u32
        add     v2.4s, v2.4s, v7.4s    // + 32768
        sub     v2.4s, v2.4s, v3.4s    // - bottom (signed result in s32 range)
        ushll2  v4.4s, v0.8h, #0
        ushll2  v5.4s, v1.8h, #0
        add     v4.4s, v4.4s, v7.4s
        sub     v4.4s, v4.4s, v5.4s
        sqxtun  v0.4h,  v2.4s          // s32 → saturating u16
        sqxtun2 v0.8h,  v4.4s
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, #32768
        sub     w13, w13, w14
        cmp     w13, #0
        csel    w13, w13, wzr, ge
        cmp     w13, w15
        csel    w13, w13, w15, ls
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── grainmerge_16: clip_u16(a + b - 32768) ─────────────────────── */
function ff_blend_grainmerge_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
        movz    w13, #0x8000
        dup     v7.4s, w13
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        ushll   v2.4s, v0.4h, #0
        ushll   v3.4s, v1.4h, #0
        add     v2.4s, v2.4s, v3.4s    // a + b
        sub     v2.4s, v2.4s, v7.4s    // - 32768
        ushll2  v4.4s, v0.8h, #0
        ushll2  v5.4s, v1.8h, #0
        add     v4.4s, v4.4s, v5.4s
        sub     v4.4s, v4.4s, v7.4s
        sqxtun  v0.4h,  v2.4s
        sqxtun2 v0.8h,  v4.4s
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, w14
        sub     w13, w13, #32768
        cmp     w13, #0
        csel    w13, w13, wzr, ge
        cmp     w13, w15
        csel    w13, w13, w15, ls
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── phoenix_16: min(a,b) - max(a,b) + 65535
 * movi .16b, #0xFF gives all bytes = 0xFF → .8h lanes = 0xFFFF = 65535
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_phoenix_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
        movi    v7.16b, #0xFF           // all bytes 0xFF → .8h = 0xFFFF
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        umin    v2.8h, v0.8h, v1.8h    // min(a,b)
        umax    v3.8h, v0.8h, v1.8h    // max(a,b)
        uqsub   v3.8h, v7.8h, v3.8h    // 65535 - max
        uqadd   v3.8h, v3.8h, v2.8h    // + min → in [0,65535]
        st1     {v3.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        cmp     w13, w14
        csel    w16, w13, w14, lo       // min
        csel    w17, w13, w14, hi       // max
        sub     w15, w15, w17           // 65535 - max (reloading w15 each iter is ok)
        add     w15, w15, w16           // + min
        strh    w15, [x10], #2
        movz    w15, #0xFFFF
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── difference_16: |a - b| using uabd on .8h ───────────────────── */
function ff_blend_difference_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        uabd    v0.8h, v0.8h, v1.8h
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        subs    w13, w13, w14
        cneg    w13, w13, mi
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── extremity_16: |(a+b) - 65535|
 * 32-bit intermediates (a+b range [0, 131070] — exceeds uint16).
 * ────────────────────────────────────────────────────────────────── */
function ff_blend_extremity_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
        movz    w13, #0xFFFF
        dup     v7.4s, w13              // v7.4s = {65535,...}
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        ushll   v2.4s, v0.4h, #0
        ushll   v3.4s, v1.4h, #0
        add     v2.4s, v2.4s, v3.4s    // a+b in u32
        uabd    v2.4s, v2.4s, v7.4s    // |(a+b) - 65535| ∈ [0,65535]
        ushll2  v4.4s, v0.8h, #0
        ushll2  v5.4s, v1.8h, #0
        add     v4.4s, v4.4s, v5.4s
        uabd    v4.4s, v4.4s, v7.4s
        xtn     v0.4h,  v2.4s          // safe narrow: values ≤ 65535
        xtn2    v0.8h,  v4.4s
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, w14           // a+b ∈ [0, 131070]
        subs    w13, w13, w15           // a+b - 65535
        cneg    w13, w13, mi            // |a+b - 65535|
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc

/* ── negation_16: 65535 - |(a+b) - 65535| = 65535 - extremity_16 ── */
function ff_blend_negation_16_neon, export=1
        lsl     x6, x6, #1
        cbz     x7, 99f
        movz    w13, #0xFFFF
        dup     v7.4s, w13
        movi    v6.16b, #0xFF           // .8h = 0xFFFF = 65535
1:      mov     x8, x0
        mov     x9, x2
        mov     x10, x4
        mov     x11, x6
        lsr     x12, x11, #4
        cbz     x12, 3f
2:      ld1     {v0.8h}, [x8], #16
        ld1     {v1.8h}, [x9], #16
        ushll   v2.4s, v0.4h, #0
        ushll   v3.4s, v1.4h, #0
        add     v2.4s, v2.4s, v3.4s
        uabd    v2.4s, v2.4s, v7.4s    // extremity_lo
        ushll2  v4.4s, v0.8h, #0
        ushll2  v5.4s, v1.8h, #0
        add     v4.4s, v4.4s, v5.4s
        uabd    v4.4s, v4.4s, v7.4s    // extremity_hi
        xtn     v0.4h,  v2.4s
        xtn2    v0.8h,  v4.4s          // extremity
        uqsub   v0.8h,  v6.8h, v0.8h  // 65535 - extremity
        st1     {v0.8h}, [x10], #16
        subs    x12, x12, #1
        b.ne    2b
3:      ands    x12, x11, #15
        b.eq    5f
        movz    w15, #0xFFFF
4:      ldrh    w13, [x8], #2
        ldrh    w14, [x9], #2
        add     w13, w13, w14
        subs    w13, w13, w15
        cneg    w13, w13, mi            // extremity
        sub     w13, w15, w13           // 65535 - extremity
        strh    w13, [x10], #2
        subs    x12, x12, #2
        b.gt    4b
5:      add     x0, x0, x1
        add     x2, x2, x3
        add     x4, x4, x5
        subs    x7, x7, #1
        b.ne    1b
99:     ret
endfunc
