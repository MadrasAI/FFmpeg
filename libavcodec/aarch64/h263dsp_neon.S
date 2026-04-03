/*
 * H.263 loop filter NEON optimisations for AArch64
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
 * h263_filter_tent macro
 *
 * Vectorized H.263 loop filter kernel: tent function + d2 clamp.
 * Operates on 8 lanes simultaneously in int16.
 *
 * Register layout:
 *   v4.8h  = p0 (unsigned, widened)
 *   v5.8h  = p1 (unsigned, widened)
 *   v6.8h  = p2 (unsigned, widened)
 *   v7.8h  = p3 (unsigned, widened)
 *   v16.8h = strength (broadcast)
 *   v17.8h = 2 * strength (broadcast, pre-computed)
 *
 * Temporaries: v0-v3 (used by h_filter for LD4/ST4; overwritten here),
 *              v18-v26.
 *
 * Outputs (int16, not yet narrowed):
 *   v4.8h  = p0'   (p0 - d2,  wrapping — use xtn)
 *   v5.8h  = p1'   (p1 + d1,  saturating — use sqxtun)
 *   v6.8h  = p2'   (p2 - d1,  saturating — use sqxtun)
 *   v7.8h  = p3'   (p3 + d2,  wrapping — use xtn)
 *
 * Algorithm:
 *   d      = (p0 - p3 + 4*(p2 - p1)) / 8   [truncating toward zero]
 *   d1     = tent(d, strength)
 *              where tent(d, s):
 *                |d| >= 2s  →  0
 *                s <= |d| < 2s  →  sign(d)*(2s - |d|)
 *                |d| < s  →  d
 *   d2     = clip((p0 - p3)/4, -|d1|/2, |d1|/2)  [truncating, /4]
 */
.macro h263_filter_tent
    // ── d = (p0 - p3 + 4*(p2 - p1)) / 8  [truncating] ─────────────────
    sub     v0.8h, v4.8h, v7.8h        // p0 - p3  (saved for d2 later)
    sub     v1.8h, v6.8h, v5.8h        // p2 - p1
    shl     v1.8h, v1.8h, #2           // 4*(p2 - p1)
    add     v2.8h, v0.8h, v1.8h        // sum = p0-p3 + 4*(p2-p1)
    // Truncating >>3: add bias 7 to negative values before arithmetic shift
    sshr    v3.8h, v2.8h, #15          // sign mask: 0x0000 if >=0, 0xFFFF if <0
    ushr    v3.8h, v3.8h, #13          // bias: 0 or 7
    add     v2.8h, v2.8h, v3.8h
    sshr    v2.8h, v2.8h, #3           // d

    // ── Tent function → d1 ──────────────────────────────────────────────
    neg     v18.8h, v17.8h             // -2*strength
    smax    v19.8h, v2.8h, v18.8h      // max(d, -2s)
    smin    v19.8h, v19.8h, v17.8h     // d_clamped = clip(d, -2s, 2s)
    abs     v20.8h, v19.8h             // |d_clamped|
    sub     v21.8h, v17.8h, v20.8h     // tent = 2s - |d| (non-negative in range)
    // signed_tent = sign(d_clamped) * tent via XOR-and-subtract trick
    sshr    v22.8h, v19.8h, #15        // sign mask: 0 if >=0, -1 if <0
    eor     v21.16b, v21.16b, v22.16b  // tent ^ sign
    sub     v21.8h, v21.8h, v22.8h     // signed_tent = (tent^sign) - sign
    // blend: |d| < s → d_clamped,  s <= |d| < 2s → signed_tent,  |d| >= 2s → 0 (by clamp)
    cmgt    v23.8h, v16.8h, v20.8h     // 0xFFFF where |d| < s (inner region)
    bsl     v23.16b, v19.16b, v21.16b  // d1

    // ── d2 = clip((p0-p3)/4, -ad1, ad1)  [truncating] ──────────────────
    abs     v24.8h, v23.8h             // |d1|
    sshr    v24.8h, v24.8h, #1         // ad1 = |d1| >> 1
    neg     v25.8h, v24.8h             // -ad1
    // (p0-p3)/4 truncating: v0 = p0-p3
    sshr    v26.8h, v0.8h, #15         // sign mask of (p0-p3)
    ushr    v26.8h, v26.8h, #14        // bias: 0 or 3 (for /4 truncating)
    add     v26.8h, v0.8h, v26.8h
    sshr    v26.8h, v26.8h, #2         // (p0-p3)/4 truncating
    smin    v26.8h, v26.8h, v24.8h     // clip upper to ad1
    smax    v26.8h, v26.8h, v25.8h     // d2 = clip to [-ad1, ad1]

    // ── Apply filter deltas ──────────────────────────────────────────────
    add     v5.8h, v5.8h, v23.8h       // p1 += d1
    sub     v6.8h, v6.8h, v23.8h       // p2 -= d1
    sub     v4.8h, v4.8h, v26.8h       // p0 -= d2
    add     v7.8h, v7.8h, v26.8h       // p3 += d2
.endm

/*
 * ff_h263_v_loop_filter_neon(uint8_t *src, int stride, int qscale)
 *
 * Vertical filter: operates on 8 columns in a single row.
 * p0 = src[x - 2*stride], p1 = src[x - stride],
 * p2 = src[x],            p3 = src[x + stride]    for x in [0, 7]
 *
 * x0 = src (points to p2 row, column 0)
 * x1 = stride (int, sign-extended)
 * w2 = qscale
 */
function ff_h263_v_loop_filter_neon, export=1
    sxtw    x1, w1                     // sign-extend stride (callers may leave upper 32 bits garbage)

    // strength = ff_h263_loop_filter_strength[qscale]
    adrp    x3, ff_h263_loop_filter_strength
    add     x3, x3, :lo12:ff_h263_loop_filter_strength
    ldrb    w3, [x3, w2, uxtw]
    dup     v16.8h, w3                 // strength broadcast
    shl     v17.8h, v16.8h, #1         // 2*strength broadcast

    // Load 4 rows of 8 bytes each
    sub     x4, x0, x1, lsl #1         // x4 = &p0 row (src - 2*stride)
    ld1     {v4.8b}, [x4], x1          // v4 = p0 row; x4 → p1 row
    ld1     {v5.8b}, [x4], x1          // v5 = p1 row; x4 → p2 row
    ld1     {v6.8b}, [x4], x1          // v6 = p2 row; x4 → p3 row
    ld1     {v7.8b}, [x4]              // v7 = p3 row; x4 stays at p3 row

    // Widen uint8 → int16
    uxtl    v4.8h, v4.8b
    uxtl    v5.8h, v5.8b
    uxtl    v6.8h, v6.8b
    uxtl    v7.8h, v7.8b

    h263_filter_tent

    // Narrow and store
    // p1 and p2: saturating clamp to [0, 255] (matches C's `& 256` / `~(p>>31)` idiom)
    sqxtun  v5.8b, v5.8h
    sqxtun  v6.8b, v6.8h
    // p0 and p3: wrapping narrow — C assigns directly to uint8 with no saturation
    xtn     v4.8b, v4.8h
    xtn     v7.8b, v7.8h

    // Restore pointer to p0 row: x4 is at p3, subtract 3 strides
    sub     x4, x4, x1, lsl #1         // x4 - 2*stride = p1 row
    sub     x4, x4, x1                 // x4 - stride   = p0 row
    st1     {v4.8b}, [x4], x1
    st1     {v5.8b}, [x4], x1
    st1     {v6.8b}, [x4], x1
    st1     {v7.8b}, [x4]
    ret
endfunc

/*
 * ff_h263_h_loop_filter_neon(uint8_t *src, int stride, int qscale)
 *
 * Horizontal filter: operates on 8 rows, 4 pixels each.
 * Per row y: p0=src[y*stride-2], p1=src[y*stride-1],
 *            p2=src[y*stride+0], p3=src[y*stride+1]
 *
 * Gather using LD4 single-lane across 8 rows → no transpose needed.
 * After gather: v0.8b=p0[0..7], v1.8b=p1[0..7], v2.8b=p2[0..7], v3.8b=p3[0..7]
 *
 * x0 = src (points to p2 of row 0)
 * x1 = stride (int, sign-extended)
 * w2 = qscale
 */
function ff_h263_h_loop_filter_neon, export=1
    sxtw    x1, w1                     // sign-extend stride

    // strength = ff_h263_loop_filter_strength[qscale]
    adrp    x3, ff_h263_loop_filter_strength
    add     x3, x3, :lo12:ff_h263_loop_filter_strength
    ldrb    w3, [x3, w2, uxtw]
    dup     v16.8h, w3                 // strength broadcast
    shl     v17.8h, v16.8h, #1         // 2*strength broadcast

    // Point to p0 of row 0 (src - 2)
    sub     x0, x0, #2
    mov     x4, x0                     // save base for stores

    // Gather 8 rows × 4 bytes via LD4 single-lane (post-indexed by stride)
    ld4     {v0.b, v1.b, v2.b, v3.b}[0], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[1], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[2], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[3], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[4], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[5], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[6], [x0], x1
    ld4     {v0.b, v1.b, v2.b, v3.b}[7], [x0]

    // Widen uint8 → int16
    uxtl    v4.8h, v0.8b               // p0 for all 8 rows
    uxtl    v5.8h, v1.8b               // p1 for all 8 rows
    uxtl    v6.8h, v2.8b               // p2 for all 8 rows
    uxtl    v7.8h, v3.8b               // p3 for all 8 rows

    h263_filter_tent

    // Narrow: p1/p2 saturating, p0/p3 wrapping (back into v0-v3 for ST4)
    sqxtun  v1.8b, v5.8h               // p1 saturating narrow
    sqxtun  v2.8b, v6.8h               // p2 saturating narrow
    xtn     v0.8b, v4.8h               // p0 wrapping narrow
    xtn     v3.8b, v7.8h               // p3 wrapping narrow

    // Scatter back via ST4 single-lane
    mov     x0, x4
    st4     {v0.b, v1.b, v2.b, v3.b}[0], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[1], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[2], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[3], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[4], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[5], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[6], [x0], x1
    st4     {v0.b, v1.b, v2.b, v3.b}[7], [x0]
    ret
endfunc
