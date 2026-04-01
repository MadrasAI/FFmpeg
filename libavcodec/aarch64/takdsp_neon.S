/*
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
 * void ff_tak_decorrelate_ls_neon(const int32_t *p1, int32_t *p2, int length)
 *   p2[i] = p1[i] + p2[i]
 *   x0=p1 (ro), x1=p2 (rw), w2=length
 */
function ff_tak_decorrelate_ls_neon, export=1
        subs            w2,  w2,  #8
        b.lt            2f
1:
        ld1             {v0.4s, v1.4s}, [x0], #32
        ld1             {v2.4s, v3.4s}, [x1]
        add             v2.4s, v0.4s, v2.4s
        add             v3.4s, v1.4s, v3.4s
        st1             {v2.4s, v3.4s}, [x1], #32
        subs            w2,  w2,  #8
        b.ge            1b
2:
        adds            w2,  w2,  #8
        b.eq            9f
3:
        ldr             w5,  [x0], #4
        ldr             w6,  [x1]
        add             w6,  w5,  w6
        str             w6,  [x1], #4
        subs            w2,  w2,  #1
        b.ne            3b
9:
        ret
endfunc

/*
 * void ff_tak_decorrelate_sr_neon(int32_t *p1, const int32_t *p2, int length)
 *   p1[i] = p2[i] - p1[i]
 *   x0=p1 (rw), x1=p2 (ro), w2=length
 */
function ff_tak_decorrelate_sr_neon, export=1
        subs            w2,  w2,  #8
        b.lt            2f
1:
        ld1             {v0.4s, v1.4s}, [x0]
        ld1             {v2.4s, v3.4s}, [x1], #32
        sub             v0.4s, v2.4s, v0.4s
        sub             v1.4s, v3.4s, v1.4s
        st1             {v0.4s, v1.4s}, [x0], #32
        subs            w2,  w2,  #8
        b.ge            1b
2:
        adds            w2,  w2,  #8
        b.eq            9f
3:
        ldr             w5,  [x0]
        ldr             w6,  [x1], #4
        sub             w5,  w6,  w5
        str             w5,  [x0], #4
        subs            w2,  w2,  #1
        b.ne            3b
9:
        ret
endfunc

/*
 * void ff_tak_decorrelate_sm_neon(int32_t *p1, int32_t *p2, int length)
 *   a = p1[i] - (p2[i] >> 1); p1[i] = a; p2[i] = a + p2[i]
 *   x0=p1 (rw), x1=p2 (rw), w2=length
 */
function ff_tak_decorrelate_sm_neon, export=1
        subs            w2,  w2,  #8
        b.lt            2f
1:
        ld1             {v0.4s, v1.4s}, [x0]        // a = p1[i:i+8]
        ld1             {v2.4s, v3.4s}, [x1]        // b = p2[i:i+8]
        sshr            v4.4s, v2.4s, #1            // b >> 1
        sshr            v5.4s, v3.4s, #1
        sub             v0.4s, v0.4s, v4.4s         // a -= b>>1
        sub             v1.4s, v1.4s, v5.4s
        add             v2.4s, v0.4s, v2.4s         // b = a + b_original
        add             v3.4s, v1.4s, v3.4s
        st1             {v0.4s, v1.4s}, [x0], #32
        st1             {v2.4s, v3.4s}, [x1], #32
        subs            w2,  w2,  #8
        b.ge            1b
2:
        adds            w2,  w2,  #8
        b.eq            9f
3:
        ldr             w5,  [x0]                   // a = p1[i]
        ldr             w6,  [x1]                   // b = p2[i]
        asr             w7,  w6,  #1                // b >> 1
        sub             w5,  w5,  w7                // a -= b>>1
        str             w5,  [x0], #4
        add             w6,  w5,  w6                // b = a + b_original
        str             w6,  [x1], #4
        subs            w2,  w2,  #1
        b.ne            3b
9:
        ret
endfunc

/*
 * void ff_tak_decorrelate_sf_neon(int32_t *p1, const int32_t *p2,
 *                                 int length, int dshift, int dfactor)
 *   b = (uint32_t)((int)(dfactor*(uint32_t)(b>>dshift)+128)>>8)<<dshift
 *   p1[i] = b - p1[i]
 *   x0=p1 (rw), x1=p2 (ro), w2=length, w3=dshift, w4=dfactor
 *
 * Register map (all v0-v7 — no callee-saved v8-v15 used):
 *   v0 = {dfactor, dfactor, dfactor, dfactor}  (constant)
 *   v1 = {128, 128, 128, 128}                  (constant)
 *   v2 = {-dshift, ...}                        (for sshl arithmetic right shift)
 *   v3 = {dshift, ...}                         (for sshl left shift)
 *   v4 = p1[i:i+4] (a)
 *   v5 = p2[i:i+4] (b, working)
 */
function ff_tak_decorrelate_sf_neon, export=1
        dup             v0.4s, w4                   // v0 = {dfactor,...}
        movi            v1.4s, #128                 // v1 = {128,...}
        neg             w5,  w3                     // -dshift
        dup             v2.4s, w5                   // v2 = {-dshift,...}
        dup             v3.4s, w3                   // v3 = {+dshift,...}
        subs            w2,  w2,  #4
        b.lt            2f
1:
        ld1             {v4.4s}, [x0]               // a = p1[i:i+4]
        ld1             {v5.4s}, [x1], #16          // b = p2[i:i+4]
        sshl            v5.4s, v5.4s, v2.4s         // b >>= dshift (arithmetic, sshl(-dshift))
        mul             v5.4s, v5.4s, v0.4s         // b *= dfactor (32x32->32 lower bits)
        add             v5.4s, v5.4s, v1.4s         // b += 128
        sshr            v5.4s, v5.4s, #8            // b >>= 8 (arithmetic truncation, not rounding)
        sshl            v5.4s, v5.4s, v3.4s         // b <<= dshift
        sub             v4.4s, v5.4s, v4.4s         // result = b - a
        st1             {v4.4s}, [x0], #16
        subs            w2,  w2,  #4
        b.ge            1b
2:
        adds            w2,  w2,  #4
        b.eq            9f
3:
        ldr             w5,  [x1], #4               // b = p2[i]
        ldr             w6,  [x0]                   // a = p1[i]
        asr             w5,  w5,  w3                // b >>= dshift
        mul             w5,  w5,  w4                // b *= dfactor
        add             w5,  w5,  #128
        asr             w5,  w5,  #8                // b >>= 8
        lsl             w5,  w5,  w3                // b <<= dshift
        sub             w5,  w5,  w6                // b - a
        str             w5,  [x0], #4
        subs            w2,  w2,  #1
        b.ne            3b
9:
        ret
endfunc
