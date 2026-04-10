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
 * void ff_add_dirac_obmc{8,16,32}_neon(uint16_t *dst, const uint8_t *src,
 *                                      int stride, const uint8_t *obmc_weight,
 *                                      int yblen)
 * x0=dst, x1=src, x2=stride (int), x3=obmc_weight, x4=yblen (int)
 *
 * Per-row operation:
 *   dst[x] += src[x] * obmc_weight[x]   (wrapping uint16 addition — matches C)
 *   dst    += stride    (uint16_t* advance: stride * 2 bytes)
 *   src    += stride    (uint8_t* advance:  stride bytes)
 *   obmc_weight += 32   (OBMC_STRIDE fixed at 32)
 *
 * Precision: src[x], obmc_weight[x] in [0..255]; max product = 255*255 = 65025
 * fits in uint16 (< 65536), so umull Vd.8h, Vn.8b, Vm.8b is exact.
 * Accumulation into uint16 dst wraps on overflow — matching C uint16_t += semantics.
 *
 * AAPCS64 notes:
 *   sxtw x2, w2  — stride is int; sign-extend to 64-bit before use as offset.
 *   subs w4, w4, #1; b.gt — 32-bit comparison avoids upper-bits garbage in x4
 *     from checkasm_checked_call trampoline.
 *   Registers used: v0-v7, v16-v19 (all caller-saved); x5 (caller-saved scratch).
 */

/* ------------------------------------------------------------------ */
/* ff_add_dirac_obmc8_neon — 8 pixels wide                            */
/* ------------------------------------------------------------------ */
function ff_add_dirac_obmc8_neon, export=1
        sxtw    x2, w2                          // sign-extend stride
        lsl     x5, x2, #1                      // x5 = stride*2 (byte advance for dst)
.Lobmc8_loop:
        ld1     {v0.8b}, [x1]                   // load 8 uint8 src
        ld1     {v1.8b}, [x3]                   // load 8 uint8 obmc_weight
        umull   v2.8h, v0.8b, v1.8b             // src*obmc → uint16 (fits: max 65025)
        ld1     {v3.8h}, [x0]                   // load 8 uint16 dst
        add     v3.8h, v3.8h, v2.8h             // dst += product (wrapping uint16)
        st1     {v3.8h}, [x0]                   // store
        add     x0, x0, x5                      // dst += stride*2 bytes
        add     x1, x1, x2                      // src += stride bytes
        add     x3, x3, #32                     // obmc_weight += OBMC_STRIDE
        subs    w4, w4, #1
        b.gt    .Lobmc8_loop
        ret
endfunc

/* ------------------------------------------------------------------ */
/* ff_add_dirac_obmc16_neon — 16 pixels wide                          */
/* ------------------------------------------------------------------ */
function ff_add_dirac_obmc16_neon, export=1
        sxtw    x2, w2
        lsl     x5, x2, #1
.Lobmc16_loop:
        ld1     {v0.16b}, [x1]                  // load 16 uint8 src
        ld1     {v1.16b}, [x3]                  // load 16 uint8 obmc_weight
        umull   v2.8h, v0.8b, v1.8b             // src[0..7] * obmc[0..7]
        umull2  v3.8h, v0.16b, v1.16b           // src[8..15] * obmc[8..15]
        ld1     {v4.8h, v5.8h}, [x0]            // load 16 uint16 dst
        add     v4.8h, v4.8h, v2.8h
        add     v5.8h, v5.8h, v3.8h
        st1     {v4.8h, v5.8h}, [x0]
        add     x0, x0, x5
        add     x1, x1, x2
        add     x3, x3, #32
        subs    w4, w4, #1
        b.gt    .Lobmc16_loop
        ret
endfunc

/* ------------------------------------------------------------------ */
/* ff_add_dirac_obmc32_neon — 32 pixels wide                          */
/* ------------------------------------------------------------------ */
function ff_add_dirac_obmc32_neon, export=1
        sxtw    x2, w2
        lsl     x5, x2, #1
.Lobmc32_loop:
        ld1     {v0.16b, v1.16b}, [x1]          // load 32 uint8 src
        ld1     {v2.16b, v3.16b}, [x3]          // load 32 uint8 obmc_weight
        umull   v4.8h, v0.8b, v2.8b             // src[0..7]  * obmc[0..7]
        umull2  v5.8h, v0.16b, v2.16b           // src[8..15] * obmc[8..15]
        umull   v6.8h, v1.8b, v3.8b             // src[16..23] * obmc[16..23]
        umull2  v7.8h, v1.16b, v3.16b           // src[24..31] * obmc[24..31]
        ld1     {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]  // load 32 uint16 dst
        add     v16.8h, v16.8h, v4.8h
        add     v17.8h, v17.8h, v5.8h
        add     v18.8h, v18.8h, v6.8h
        add     v19.8h, v19.8h, v7.8h
        st1     {v16.8h, v17.8h, v18.8h, v19.8h}, [x0]
        add     x0, x0, x5
        add     x1, x1, x2
        add     x3, x3, #32
        subs    w4, w4, #1
        b.gt    .Lobmc32_loop
        ret
endfunc
