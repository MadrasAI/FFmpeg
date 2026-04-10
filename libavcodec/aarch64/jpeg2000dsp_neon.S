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
 * ICT float constants: {1.402, 0.34413, 0.71414, 1.772}
 * Stored locally to avoid GOT access for an external symbol.
 */
const ff_ict_params_neon, align=4
        .float 1.402, 0.34413, 0.71414, 1.772
endconst

/*
 * void ff_rct_int_neon(void *src0, void *src1, void *src2, int csize)
 * x0=src0  x1=src1  x2=src2  w3=csize (int32 element count)
 *
 * Reversible Color Transform (integer):
 *   i1 = src0 - ((src1 + src2) >> 2)
 *   i0 = i1 + src2
 *   i2 = i1 + src1
 *
 * Main loop: 4 int32/iter using sshr + sub + add.
 * Guard: cmp w3,#4; b.lt .tail before each ld1 (prevents overread).
 */
function ff_rct_int_neon, export=1
        cmp     w3, #4
        b.lt    .Lrct_tail

.Lrct_main:
        ld1     {v0.4s}, [x0]
        ld1     {v1.4s}, [x1]
        ld1     {v2.4s}, [x2]
        add     v3.4s, v1.4s, v2.4s      // src1 + src2
        sshr    v3.4s, v3.4s, #2         // (src1+src2)>>2 (arithmetic)
        sub     v4.4s, v0.4s, v3.4s      // i1 = src0 - ((src1+src2)>>2)
        add     v5.4s, v4.4s, v2.4s      // i0 = i1 + src2
        add     v6.4s, v4.4s, v1.4s      // i2 = i1 + src1
        st1     {v5.4s}, [x0], #16       // store i0 → src0
        st1     {v4.4s}, [x1], #16       // store i1 → src1
        st1     {v6.4s}, [x2], #16       // store i2 → src2
        sub     w3, w3, #4               // non-flag-setting; guard next load
        cmp     w3, #4
        b.ge    .Lrct_main

.Lrct_tail:
        cbz     w3, .Lrct_done
.Lrct_tail_loop:
        ldr     w4, [x0]
        ldr     w5, [x1]
        ldr     w6, [x2]
        add     w7, w5, w6               // src1 + src2
        asr     w7, w7, #2              // >> 2 (arithmetic)
        sub     w8, w4, w7              // i1
        add     w4, w8, w6              // i0 = i1 + src2
        add     w9, w8, w5              // i2 = i1 + src1
        str     w4, [x0], #4
        str     w8, [x1], #4
        str     w9, [x2], #4
        subs    w3, w3, #1
        b.gt    .Lrct_tail_loop

.Lrct_done:
        ret
endfunc

/*
 * void ff_ict_float_neon(void *src0, void *src1, void *src2, int csize)
 * x0=src0  x1=src1  x2=src2  w3=csize
 *
 * Irreversible Color Transform (float):
 *   i0 = src0 + 1.402   * src2
 *   i1 = src0 - 0.34413 * src1 - 0.71414 * src2
 *   i2 = src0 + 1.772   * src1
 *
 * Constants loaded from local rodata into v17-v20 (broadcast).
 * Main loop: 4 float/iter using fmla/fmls.
 */
function ff_ict_float_neon, export=1
        adrp    x4, ff_ict_params_neon
        add     x4, x4, :lo12:ff_ict_params_neon
        ld1     {v16.4s}, [x4]
        dup     v17.4s, v16.s[0]         // 1.402
        dup     v18.4s, v16.s[1]         // 0.34413
        dup     v19.4s, v16.s[2]         // 0.71414
        dup     v20.4s, v16.s[3]         // 1.772

        cmp     w3, #4
        b.lt    .Lict_tail

.Lict_main:
        ld1     {v0.4s}, [x0]
        ld1     {v1.4s}, [x1]
        ld1     {v2.4s}, [x2]
        mov     v3.16b, v0.16b
        fmla    v3.4s, v2.4s, v17.4s    // i0 = src0 + 1.402*src2
        mov     v4.16b, v0.16b
        fmls    v4.4s, v1.4s, v18.4s    // i1 = src0 - 0.34413*src1
        fmls    v4.4s, v2.4s, v19.4s    //    - 0.71414*src2
        mov     v5.16b, v0.16b
        fmla    v5.4s, v1.4s, v20.4s    // i2 = src0 + 1.772*src1
        st1     {v3.4s}, [x0], #16
        st1     {v4.4s}, [x1], #16
        st1     {v5.4s}, [x2], #16
        sub     w3, w3, #4
        cmp     w3, #4
        b.ge    .Lict_main

.Lict_tail:
        cbz     w3, .Lict_done
.Lict_tail_loop:
        ldr     s0, [x0]
        ldr     s1, [x1]
        ldr     s2, [x2]
        fmul    s3, s2, s17              // 1.402 * src2
        fadd    s3, s3, s0              // i0 = src0 + 1.402*src2
        fmul    s4, s1, s18              // 0.34413 * src1
        fmul    s5, s2, s19              // 0.71414 * src2
        fsub    s4, s0, s4              // src0 - 0.34413*src1
        fsub    s4, s4, s5              // i1 -= 0.71414*src2
        fmul    s5, s1, s20              // 1.772 * src1
        fadd    s5, s5, s0              // i2 = src0 + 1.772*src1
        str     s3, [x0], #4
        str     s4, [x1], #4
        str     s5, [x2], #4
        subs    w3, w3, #1
        b.gt    .Lict_tail_loop

.Lict_done:
        ret
endfunc
