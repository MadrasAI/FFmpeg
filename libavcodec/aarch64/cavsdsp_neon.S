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

/*
 * CAVS (AVS1-P2) QPEL motion compensation — AArch64 NEON
 *
 * Implements mc positions [2] (H hpel), [4] (V qpel_l), [8] (V hpel),
 * [12] (V qpel_r) for 8×8 and 16×16 block sizes (put and avg).
 *
 * Filter coefficients (from cavsdsp.c):
 *   hpel:   [0, -1, 5, 5, -1, 0]  op: (x + 4) >> 3   (int16 accumulation)
 *   qpel_r: [0, -7, 42, 96, -2, -1] op: (x + 64) >> 7  (int32 accumulation)
 *   qpel_l: time-reversal of qpel_r via mc03 with negative stride
 *
 * Register plan (leaf functions — all v0-v31 available, no callee saves):
 *
 * H hpel:
 *   x4      : src-1 (base, post-incremented by stride each row)
 *   v0.16b  : 16-byte load from src-1
 *   v1..v3  : ext-shifted views (src[0..7], src[1..8], src[2..9])
 *   v4..v7  : uint16-widened tap values (ushll)
 *   v8..v9  : int16 arithmetic intermediates
 *   v10     : result bytes (sqrshrun output)
 *
 * V hpel sliding window (4 registers):
 *   v0..v3.8b : rows r[y-1], r[y], r[y+1], r[y+2]  (8-byte, lower half used)
 *   x4         : source pointer for next row load
 *   v4..v9     : uint16-widened rows and arithmetic
 *
 * V qpel_r sliding window (5 registers):
 *   v0..v3.8b  : rows r[y-1], r[y], r[y+1], r[y+2]  (pre-loaded)
 *   v4.8b      : newly loaded row r[y+3]
 *   v27..v31.8h: broadcast constants {1, 7, 42, 96, 2}
 *   v5..v7, v20..v21 : uint16-widened rows
 *   v22..v23.4s: int32 accumulators (lo/hi 4 lanes)
 *   v24.8h     : int16 narrowed result (sqrshrn/sqrshrn2)
 *   v25.8b     : final uint8 result (sqxtun)
 */

#include "libavutil/aarch64/asm.S"

/* =====================================================================
 * H hpel: filter = -sm1 + 5*(s0+s1) - s2,  out = (x+4)>>3 clip_uint8
 * void ff_put_cavs_qpel8_mc20_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
 * void ff_put_cavs_qpel8_h_neon  (uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h)
 * ===================================================================== */

// Temporaries use v16-v19 (not callee-saved); avoid v8-v15 (callee-saved in AAPCS64).
.macro cavs_qpel8_h_row avg
        ld1     {v0.16b}, [x4], x2         // 16 bytes from src-1, advance src ptr
        ext     v1.16b, v0.16b, v0.16b, #1 // src[0..15]
        ext     v2.16b, v0.16b, v0.16b, #2 // src[1..16]
        ext     v3.16b, v0.16b, v0.16b, #3 // src[2..17]
        uxtl    v4.8h, v0.8b               // sm1 = src[-1] as uint16
        uxtl    v5.8h, v1.8b               // s0  = src[0]
        uxtl    v6.8h, v2.8b               // s1  = src[1]
        uxtl    v7.8h, v3.8b               // s2  = src[2]
        add     v16.8h, v5.8h, v6.8h       // s0+s1
        add     v4.8h,  v4.8h, v7.8h       // sm1+s2
        shl     v17.8h, v16.8h, #2         // 4*(s0+s1)
        add     v17.8h, v17.8h, v16.8h     // 5*(s0+s1)
        sub     v17.8h, v17.8h, v4.8h      // 5*(s0+s1)-(sm1+s2)
        sqrshrun v18.8b, v17.8h, #3        // clip_uint8((x+4)>>3)
  .ifc \avg, avg
        ld1     {v19.8b}, [x0]             // load existing dst
        urhadd  v18.8b, v18.8b, v19.8b    // (new+old+1)>>1
  .endif
        st1     {v18.8b}, [x0], x2         // store 8 pixels, advance dst
.endm

function ff_put_cavs_qpel8_mc20_neon, export=1
        mov     w3, #8
        sub     x4, x1, #1
0:      cavs_qpel8_h_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_put_cavs_qpel8_h_neon, export=1
        sub     x4, x1, #1
0:      cavs_qpel8_h_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_mc20_neon, export=1
        mov     w3, #8
        sub     x4, x1, #1
0:      cavs_qpel8_h_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_h_neon, export=1
        sub     x4, x1, #1
0:      cavs_qpel8_h_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

/* =====================================================================
 * V hpel: filter = -r[y-1] + 5*(r[y]+r[y+1]) - r[y+2], (x+4)>>3
 * Sliding 4-register window: v0=r[y-1], v1=r[y], v2=r[y+1]; load r[y+2] each iter.
 * void ff_put_cavs_qpel8_mc02_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
 * void ff_put_cavs_qpel8_v2_neon  (uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h)
 * ===================================================================== */

.macro cavs_qpel8_v2_setup
        sub     x4, x1, x2                 // x4 = src - stride = row[-1]
        ld1     {v0.8b}, [x4], x2          // v0 = r[-1], x4 → src
        ld1     {v1.8b}, [x4], x2          // v1 = r[0]
        ld1     {v2.8b}, [x4], x2          // v2 = r[1], x4 → src+2*stride
.endm

// Temporaries use v16-v19 (not callee-saved); avoid v8-v15 (callee-saved in AAPCS64).
.macro cavs_qpel8_v2_row avg
        ld1     {v3.8b}, [x4], x2          // load r[y+2]
        uxtl    v4.8h, v0.8b               // r[y-1] as uint16
        uxtl    v5.8h, v1.8b               // r[y]
        uxtl    v6.8h, v2.8b               // r[y+1]
        uxtl    v7.8h, v3.8b               // r[y+2]
        add     v16.8h, v5.8h, v6.8h       // r[y]+r[y+1]
        add     v4.8h,  v4.8h, v7.8h       // r[y-1]+r[y+2]
        shl     v17.8h, v16.8h, #2         // 4*(r[y]+r[y+1])
        add     v17.8h, v17.8h, v16.8h     // 5*(r[y]+r[y+1])
        sub     v17.8h, v17.8h, v4.8h      // 5*(r[y]+r[y+1])-(r[y-1]+r[y+2])
        sqrshrun v18.8b, v17.8h, #3        // clip_uint8((x+4)>>3)
  .ifc \avg, avg
        ld1     {v19.8b}, [x0]
        urhadd  v18.8b, v18.8b, v19.8b
  .endif
        st1     {v18.8b}, [x0], x2
        // Rotate sliding window: v0←v1, v1←v2, v2←v3
        mov     v0.8b, v1.8b
        mov     v1.8b, v2.8b
        mov     v2.8b, v3.8b
.endm

function ff_put_cavs_qpel8_mc02_neon, export=1
        mov     w3, #8
        cavs_qpel8_v2_setup
0:      cavs_qpel8_v2_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_put_cavs_qpel8_v2_neon, export=1
        cavs_qpel8_v2_setup
0:      cavs_qpel8_v2_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_mc02_neon, export=1
        mov     w3, #8
        cavs_qpel8_v2_setup
0:      cavs_qpel8_v2_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_v2_neon, export=1
        cavs_qpel8_v2_setup
0:      cavs_qpel8_v2_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

/* =====================================================================
 * V qpel_r: filter = -7*r[y-1] + 42*r[y] + 96*r[y+1] - 2*r[y+2] - r[y+3]
 *           out = (x + 64) >> 7  clip_uint8  (needs int32 accumulation)
 *
 * Sliding 5-register window: v0=r[y-1], v1=r[y], v2=r[y+1], v3=r[y+2];
 *                            load r[y+3] into v4 each iteration.
 * Constants (broadcast 8h): v27=1, v28=7, v29=42, v30=96, v31=2
 *
 * void ff_put_cavs_qpel8_mc03_neon(uint8_t *dst, const uint8_t *src, ptrdiff_t stride)
 * void ff_put_cavs_qpel8_v3_neon  (uint8_t *dst, const uint8_t *src, ptrdiff_t stride, int h)
 * ===================================================================== */

.macro cavs_qpel8_v3_setup
        // Load filter constants into persistent registers
        movi    v27.8h, #1                 // coeff 1
        movi    v28.8h, #7                 // coeff 7
        movi    v29.8h, #42                // coeff 42
        movi    v30.8h, #96                // coeff 96
        movi    v31.8h, #2                 // coeff 2
        // Pre-load rows r[-1], r[0], r[1], r[2] into sliding window
        sub     x4, x1, x2                 // x4 = src - stride = r[-1]
        ld1     {v0.8b}, [x4], x2          // v0 = r[-1], x4 → src
        ld1     {v1.8b}, [x4], x2          // v1 = r[0]
        ld1     {v2.8b}, [x4], x2          // v2 = r[1]
        ld1     {v3.8b}, [x4], x2          // v3 = r[2], x4 → src+3*stride
.endm

.macro cavs_qpel8_v3_row avg
        ld1     {v4.8b}, [x4], x2          // load r[y+3]
        // Widen all 5 rows to int16 (values 0..255, safe as uint16=int16)
        uxtl    v5.8h,  v0.8b              // r[y-1]
        uxtl    v6.8h,  v1.8b              // r[y]
        uxtl    v7.8h,  v2.8b              // r[y+1]
        uxtl    v20.8h, v3.8b              // r[y+2]
        uxtl    v21.8h, v4.8b              // r[y+3]
        // Accumulate into int32: 42*r[y] + 96*r[y+1] - 7*r[y-1] - 2*r[y+2] - 1*r[y+3]
        smull   v22.4s, v6.4h, v29.4h     // 42*r[y]  pixels 0-3
        smull2  v23.4s, v6.8h, v29.8h     // 42*r[y]  pixels 4-7
        smlal   v22.4s, v7.4h, v30.4h     // +96*r[y+1]
        smlal2  v23.4s, v7.8h, v30.8h
        smlsl   v22.4s, v5.4h, v28.4h     // -7*r[y-1]
        smlsl2  v23.4s, v5.8h, v28.8h
        smlsl   v22.4s, v20.4h, v31.4h    // -2*r[y+2]
        smlsl2  v23.4s, v20.8h, v31.8h
        smlsl   v22.4s, v21.4h, v27.4h    // -1*r[y+3]
        smlsl2  v23.4s, v21.8h, v27.8h
        // (x + 64) >> 7 with signed saturation to int16, then clip to uint8
        sqrshrn  v24.4h, v22.4s, #7       // pixels 0-3: (x+64)>>7 → int16
        sqrshrn2 v24.8h, v23.4s, #7       // pixels 4-7
        sqxtun   v25.8b, v24.8h           // int16 → uint8 with saturation
  .ifc \avg, avg
        ld1     {v26.8b}, [x0]
        urhadd  v25.8b, v25.8b, v26.8b
  .endif
        st1     {v25.8b}, [x0], x2
        // Rotate sliding window: v0←v1, v1←v2, v2←v3, v3←v4
        mov     v0.8b, v1.8b
        mov     v1.8b, v2.8b
        mov     v2.8b, v3.8b
        mov     v3.8b, v4.8b
.endm

function ff_put_cavs_qpel8_mc03_neon, export=1
        mov     w3, #8
        cavs_qpel8_v3_setup
0:      cavs_qpel8_v3_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_put_cavs_qpel8_v3_neon, export=1
        cavs_qpel8_v3_setup
0:      cavs_qpel8_v3_row put
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_mc03_neon, export=1
        mov     w3, #8
        cavs_qpel8_v3_setup
0:      cavs_qpel8_v3_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc

function ff_avg_cavs_qpel8_v3_neon, export=1
        cavs_qpel8_v3_setup
0:      cavs_qpel8_v3_row avg
        subs    w3, w3, #1
        b.ne    0b
        ret
endfunc
