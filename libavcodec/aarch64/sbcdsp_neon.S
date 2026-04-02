/*
 * Bluetooth low-complexity, subband codec (SBC)
 * AArch64 NEON optimizations
 *
 * Copyright (C) 2017  Aurelien Jacobs <aurel@gnuage.org>
 * Copyright (C) 2008-2010  Nokia Corporation
 * Copyright (C) 2004-2010  Marcel Holtmann <marcel@holtmann.org>
 * Copyright (C) 2004-2005  Henryk Ploetz <henryk@ploetzli.ch>
 * Copyright (C) 2005-2006  Brad Midgley <bmidgley@xmission.com>
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

/**
 * @file
 * SBC AArch64 NEON optimizations
 * Ported from libavcodec/arm/sbcdsp_neon.S
 */

#include "libavutil/aarch64/asm.S"

#define SBC_PROTO_FIXED_SCALE 16

/*
 * ff_sbc_analyze_4_neon(const int16_t *in, int32_t *out, const int16_t *consts)
 * x0=in, x1=out, x2=consts
 *
 * ARM32→AArch64 register map:
 *   d16,d17 = q8  → v8      d18,d19 = q9  → v9
 *   d20,d21 = q10 → v10     d22,d23 = q11 → v11
 *   q0,q1 accumulators → v0,v1
 */
function ff_sbc_analyze_4_neon, export=1
        stp     d8,  d9,  [sp, #-0x20]!
        stp     d10, d11, [sp, #0x10]

        ld1     {v8.8h},  [x0], #16
        ld1     {v10.8h}, [x2], #16

        smull   v0.4s,  v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smull2  v1.4s,  v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16

        smlal   v0.4s,  v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smlal2  v1.4s,  v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v0.4s,  v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smlal2  v1.4s,  v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16

        smlal   v0.4s,  v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smlal2  v1.4s,  v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v0.4s,  v8.4h,  v10.4h
        smlal2  v1.4s,  v8.8h,  v10.8h

        // ARM32: vpadd d0,d0,d1 + vpadd d1,d2,d3
        addp    v0.4s,  v0.4s,  v1.4s

        rshrn   v0.4h,  v0.4s,  #SBC_PROTO_FIXED_SCALE

        ld1     {v8.8h, v9.8h}, [x2], #32

        // ARM32: vdup.i32 d1,d0[1]; vdup.i32 d0,d0[0]
        // After rshrn: v0 lower 64-bits = [s0,s1,s2,s3] as int16
        // v0.s[0] as int32 = {s1<<16|s0}, v0.s[1] = {s3<<16|s2}
        dup     v1.4s,  v0.s[1]            // v1.8h = [s2,s3,s2,s3,s2,s3,s2,s3]
        dup     v0.4s,  v0.s[0]            // v0.8h = [s0,s1,s0,s1,s0,s1,s0,s1]

        smull   v10.4s, v8.4h,  v0.4h
        smull2  v11.4s, v8.8h,  v0.8h
        smlal   v10.4s, v9.4h,  v1.4h
        smlal2  v11.4s, v9.8h,  v1.8h

        addp    v0.4s,  v10.4s, v11.4s

        st1     {v0.4s}, [x1]

        ldp     d10, d11, [sp, #0x10]
        ldp     d8,  d9,  [sp], #0x20
        ret
endfunc

/*
 * ff_sbc_analyze_8_neon(const int16_t *in, int32_t *out, const int16_t *consts)
 * x0=in, x1=out, x2=consts
 *
 * ARM32 accumulators q12-q15 (d24-d31) → v12-v15
 * ARM32 temporaries  q8-q11  (d16-d23) → v8-v11
 */
function ff_sbc_analyze_8_neon, export=1
        stp     d8,  d9,  [sp, #-0x40]!
        stp     d10, d11, [sp, #0x10]
        stp     d12, d13, [sp, #0x20]
        stp     d14, d15, [sp, #0x30]

        ld1     {v8.8h},  [x0], #16
        ld1     {v10.8h}, [x2], #16

        smull   v12.4s, v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smull2  v13.4s, v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16
        smull   v14.4s, v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smull2  v15.4s, v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v12.4s, v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smlal2  v13.4s, v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16
        smlal   v14.4s, v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smlal2  v15.4s, v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v12.4s, v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smlal2  v13.4s, v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16
        smlal   v14.4s, v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smlal2  v15.4s, v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v12.4s, v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smlal2  v13.4s, v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16
        smlal   v14.4s, v9.4h,  v11.4h
        ld1     {v8.8h},  [x0], #16
        smlal2  v15.4s, v9.8h,  v11.8h
        ld1     {v10.8h}, [x2], #16

        smlal   v12.4s, v8.4h,  v10.4h
        ld1     {v9.8h},  [x0], #16
        smlal2  v13.4s, v8.8h,  v10.8h
        ld1     {v11.8h}, [x2], #16

        smlal   v14.4s, v9.4h,  v11.4h
        smlal2  v15.4s, v9.8h,  v11.8h

        // ARM32: 4× vpadd.s32 on d24..d31
        addp    v0.4s,  v12.4s, v13.4s
        addp    v1.4s,  v14.4s, v15.4s

        rshrn   v0.4h,  v0.4s,  #SBC_PROTO_FIXED_SCALE   // d0 = [s0,s1,s2,s3]
        rshrn   v1.4h,  v1.4s,  #SBC_PROTO_FIXED_SCALE   // d1 = [s4,s5,s6,s7]

        // Broadcast each int32 lane (= int16 pair) to fill all 8 int16 lanes
        dup     v3.4s,  v1.s[1]            // v3.8h = [s6,s7] × 4
        dup     v2.4s,  v1.s[0]            // v2.8h = [s4,s5] × 4
        dup     v1.4s,  v0.s[1]            // v1.8h = [s2,s3] × 4
        dup     v0.4s,  v0.s[0]            // v0.8h = [s0,s1] × 4

        ld1     {v8.8h},  [x2], #16
        smull   v12.4s, v8.4h,  v0.4h
        ld1     {v9.8h},  [x2], #16
        smull2  v13.4s, v8.8h,  v0.8h
        smull   v14.4s, v9.4h,  v0.4h
        smull2  v15.4s, v9.8h,  v0.8h

        ld1     {v8.8h},  [x2], #16
        smlal   v12.4s, v8.4h,  v1.4h
        ld1     {v9.8h},  [x2], #16
        smlal2  v13.4s, v8.8h,  v1.8h
        smlal   v14.4s, v9.4h,  v1.4h
        smlal2  v15.4s, v9.8h,  v1.8h

        ld1     {v8.8h},  [x2], #16
        smlal   v12.4s, v8.4h,  v2.4h
        ld1     {v9.8h},  [x2], #16
        smlal2  v13.4s, v8.8h,  v2.8h
        smlal   v14.4s, v9.4h,  v2.4h
        smlal2  v15.4s, v9.8h,  v2.8h

        ld1     {v8.8h},  [x2], #16
        smlal   v12.4s, v8.4h,  v3.4h
        ld1     {v9.8h},  [x2], #16
        smlal2  v13.4s, v8.8h,  v3.8h
        smlal   v14.4s, v9.4h,  v3.4h
        smlal2  v15.4s, v9.8h,  v3.8h

        addp    v0.4s,  v12.4s, v13.4s
        addp    v1.4s,  v14.4s, v15.4s

        st1     {v0.4s, v1.4s}, [x1]

        ldp     d14, d15, [sp, #0x30]
        ldp     d12, d13, [sp, #0x20]
        ldp     d10, d11, [sp, #0x10]
        ldp     d8,  d9,  [sp], #0x40
        ret
endfunc

/*
 * ff_sbc_calc_scalefactors_neon(
 *   const int32_t sb_sample_f[16][2][8],  // x0
 *   uint32_t scale_factor[2][8],           // x1
 *   int blocks,                            // w2
 *   int channels,                          // w3
 *   int subbands)                          // w4
 *
 * AArch64 5th arg is in x4 (not on stack as in ARM32).
 * x0-x18 are caller-saved; no integer register push/pop needed.
 */
function ff_sbc_calc_scalefactors_neon, export=1
        stp     d8,  d9,  [sp, #-0x30]!
        stp     d10, d11, [sp, #0x10]
        stp     d14, d15, [sp, #0x20]

        mov     x11, #64               // inc = 64 bytes per block

        movi    v14.4s, #1
        movi    v15.4s, #16            // 31 - SCALE_OUT_BITS = 16

        mov     x9,  #0                // ch = 0
1:
        add     x5,  x0,  x9,  lsl #5  // in_loop_1  = sb_sample_f + ch*32
        add     x7,  x1,  x9,  lsl #5  // out_loop_1 = scale_factor + ch*32

        mov     x10, #0                // sb = 0
2:
        add     x6,  x5,  x10, lsl #2  // in  = in_loop_1 + sb*4
        add     x8,  x7,  x10, lsl #2  // out = out_loop_1 + sb*4
        movi    v0.4s, #0
        movi    v1.4s, #0x80, lsl #8   // v1 = 0x8000
        add     v1.4s,  v1.4s,  v14.4s // v1 = 0x8001 = (1<<SCALE_OUT_BITS)+1

        and     w12, w2, #~3            // w12 = blocks rounded down to mult of 4
        cbz     w12, 5f
3:
        ld1     {v8.4s},  [x6], x11
        abs     v8.4s,   v8.4s
        ld1     {v9.4s},  [x6], x11
        abs     v9.4s,   v9.4s
        ld1     {v10.4s}, [x6], x11
        abs     v10.4s,  v10.4s
        ld1     {v11.4s}, [x6], x11
        abs     v11.4s,  v11.4s
        smax    v0.4s,   v0.4s,  v8.4s
        smax    v1.4s,   v1.4s,  v9.4s
        smax    v0.4s,   v0.4s,  v10.4s
        smax    v1.4s,   v1.4s,  v11.4s
        subs    w12, w12, #4
        b.gt    3b
5:
        ands    w12, w2, #3             // tail = blocks % 4
        b.eq    4f
6:
        ld1     {v8.4s},  [x6], x11
        abs     v8.4s,   v8.4s
        smax    v0.4s,   v0.4s,  v8.4s
        subs    w12, w12, #1
        b.gt    6b
4:

        smax    v0.4s,  v0.4s,  v1.4s
        sub     v0.4s,  v0.4s,  v14.4s
        clz     v0.4s,  v0.4s
        sub     v0.4s,  v15.4s, v0.4s
        st1     {v0.4s}, [x8]

        add     x10, x10, #4
        cmp     w10, w4
        b.lt    2b

        add     x9,  x9,  #1
        cmp     w9,  w3
        b.lt    1b

        ldp     d14, d15, [sp, #0x20]
        ldp     d10, d11, [sp, #0x10]
        ldp     d8,  d9,  [sp], #0x30
        ret
endfunc

/*
 * calc_scalefactors macro — used inside ff_sbc_calc_scalefactors_j_neon
 *
 * Inputs (registers set by caller):
 *   x5  = channel 0 sample pointer
 *   x6  = channel 1 sample pointer
 *   x2  = blocks count
 *   x11 = stride (64)
 *   v14.4s = 1
 *   v13.4s = 16 (31 - SCALE_OUT_BITS)
 *   v0.4s  = (1<<SCALE_OUT_BITS)+1 initial value
 *
 * Outputs:
 *   v0,v1 = scale factors without joint stereo
 *   v2,v3 = scale factors with joint stereo
 *   v15   = joint stereo selection mask (set by caller after macro)
 *
 * Uses x15 as the per-invocation loop counter (does not conflict with
 * x0-x14 used in the outer function body).
 */
.macro calc_scalefactors
        mov     v1.16b, v0.16b
        mov     v2.16b, v0.16b
        mov     v3.16b, v0.16b
        mov     x15, x2                // loop counter = blocks
1:
        ld1     {v9.4s},  [x6], x11
        bic     v11.16b, v9.16b,  v14.16b
        ld1     {v8.4s},  [x5], x11
        shadd   v10.4s,  v8.4s,  v11.4s
        shsub   v11.4s,  v8.4s,  v11.4s
        abs     v8.4s,   v8.4s
        abs     v9.4s,   v9.4s
        abs     v10.4s,  v10.4s
        abs     v11.4s,  v11.4s
        smax    v0.4s,   v0.4s,  v8.4s
        smax    v1.4s,   v1.4s,  v9.4s
        smax    v2.4s,   v2.4s,  v10.4s
        smax    v3.4s,   v3.4s,  v11.4s
        subs    x15, x15, #1
        b.gt    1b
        sub     v0.4s,  v0.4s,  v14.4s
        sub     v1.4s,  v1.4s,  v14.4s
        sub     v2.4s,  v2.4s,  v14.4s
        sub     v3.4s,  v3.4s,  v14.4s
        clz     v0.4s,  v0.4s
        clz     v1.4s,  v1.4s
        clz     v2.4s,  v2.4s
        clz     v3.4s,  v3.4s
        sub     v0.4s,  v13.4s, v0.4s
        sub     v1.4s,  v13.4s, v1.4s
        sub     v2.4s,  v13.4s, v2.4s
        sub     v3.4s,  v13.4s, v3.4s
.endm

/*
 * update_joint_stereo_samples macro
 *
 * ARM32 used partial d-register vbif (d22 before loop, d23 at start of loop
 * body). AArch64 restructures to full 128-bit bif/vbif per iteration to avoid
 * lane-indexed bit/bif which is not supported in AArch64 NEON.
 * The net result is identical; only the interleaving differs.
 *
 * Inputs:
 *   x5,x6 = pointers after calc_scalefactors (block-past end)
 *   x12,x13 = additional ch pointers (r8,r7 equivalents)
 *   x10 = 0 (zero constant)
 *   x11 = 64 (stride, restored on exit)
 *   x2  = blocks
 *   v14 = 1
 *   v15 = joint stereo mask
 */
.macro update_joint_stereo_samples
        sub     x12, x6,  x11
        sub     x13, x5,  x11
        sub     x6,  x6,  x11, lsl #1
        sub     x5,  x5,  x11, lsl #1
        ld1     {v9.4s},  [x6]
        bic     v11.16b, v9.16b,  v14.16b
        ld1     {v8.4s},  [x5]
        ld1     {v1.4s},  [x12]
        bic     v3.16b,  v1.16b,  v14.16b
        ld1     {v0.4s},  [x13]
        shsub   v10.4s,  v8.4s,  v11.4s
        shadd   v11.4s,  v8.4s,  v11.4s
        shsub   v2.4s,   v0.4s,  v3.4s
        shadd   v3.4s,   v0.4s,  v3.4s
        // ARM32: vbif q10,q9,q15; vbif d22,d16,d30 (partial q11)
        // AArch64: do full 128-bit bif on both q10 and q11 upfront
        bif     v10.16b, v9.16b,  v15.16b
        bif     v11.16b, v8.16b,  v15.16b
        sub     x11, x10, x11, lsl #1   // x11 = -(stride*2)
        sub     w16, w2,  #2             // loop counter = blocks-2 (use w16)
2:
        st1     {v10.4s}, [x6], x11
        // ARM32: vbif d4,d2,d30; vbif d5,d3,d31 → full 128-bit bif v2,v1,v15
        bif     v2.16b,  v1.16b,  v15.16b
        ld1     {v9.4s},  [x6]
        st1     {v11.4s}, [x5], x11
        bif     v3.16b,  v0.16b,  v15.16b
        ld1     {v8.4s},  [x5]
        st1     {v2.4s},  [x12], x11
        bic     v11.16b, v9.16b,  v14.16b
        ld1     {v1.4s},  [x12]
        st1     {v3.4s},  [x13], x11
        bic     v3.16b,  v1.16b,  v14.16b
        ld1     {v0.4s},  [x13]
        shsub   v10.4s,  v8.4s,  v11.4s
        shadd   v11.4s,  v8.4s,  v11.4s
        shsub   v2.4s,   v0.4s,  v3.4s
        shadd   v3.4s,   v0.4s,  v3.4s
        bif     v10.16b, v9.16b,  v15.16b
        bif     v11.16b, v8.16b,  v15.16b
        subs    w16, w16, #2
        b.gt    2b
        // ARM32: sub r11, r10, r11, asr #1  →  x11 = 0 - (x11 >> 1)
        // x11 is currently -(stride*2), asr #1 gives -stride, negate → +stride
        sub     x11, x10, x11, asr #1
        st1     {v10.4s}, [x6]
        bif     v2.16b,  v1.16b,  v15.16b
        st1     {v11.4s}, [x5]
        bif     v3.16b,  v0.16b,  v15.16b
        st1     {v2.4s},  [x12]
        st1     {v3.4s},  [x13]
.endm

/*
 * ff_sbc_calc_scalefactors_j_neon(
 *   int32_t sb_sample_f[16][2][8],    // x0
 *   uint32_t scale_factor[2][8],       // x1
 *   int blocks,                        // w2
 *   int subbands)                      // w3
 *
 * Returns: int joint  (w0)
 *
 * Saves: d8-d15 (NEON callee-saved halves), x19 (used as ff_sbcdsp mask ptr)
 * x0-x18 are caller-saved; x15,x16 used as loop counters in macros.
 * x19 is callee-saved and used to preserve the mask pointer across macro calls.
 */
function ff_sbc_calc_scalefactors_j_neon, export=1
        stp     d8,  d9,  [sp, #-0x50]!
        stp     d10, d11, [sp, #0x10]
        stp     d12, d13, [sp, #0x20]
        stp     d14, d15, [sp, #0x30]
        str     x19,      [sp, #0x40]

        movrel  x4,  ff_sbcdsp_joint_bits_mask
        mov     x10, #0
        mov     x11, #64

        movi    v14.4s, #1
        movi    v13.4s, #16            // 31 - SCALE_OUT_BITS

        cmp     w3, #4
        b.ne    8f

4:      // 4 subbands
        add     x5,  x0,  #0
        add     x6,  x0,  #32
        add     x7,  x1,  #0
        add     x8,  x1,  #32
        movi    v0.4s, #0x80, lsl #8
        add     v0.4s, v0.4s, v14.4s

        calc_scalefactors

        add     v15.4s, v0.4s, v1.4s
        add     v9.4s,  v2.4s, v3.4s
        ins     v15.s[3], w10           // last subband → no joint
        ld1     {v8.4s}, [x4], #16
        cmgt    v15.4s, v15.4s, v9.4s

        and     v8.16b, v8.16b, v15.16b
        bit     v0.16b, v2.16b, v15.16b
        addp    v8.4s,  v8.4s,  v8.4s
        bit     v1.16b, v3.16b, v15.16b
        addp    v8.4s,  v8.4s,  v8.4s
        st1     {v0.4s}, [x7]
        st1     {v1.4s}, [x8]
        mov     w0, v8.s[0]

        update_joint_stereo_samples
        b       9f

8:      // 8 subbands
        add     x5,  x0,  #16
        add     x6,  x0,  #48
        add     x7,  x1,  #16
        add     x8,  x1,  #48
        movi    v0.4s, #0x80, lsl #8
        add     v0.4s, v0.4s, v14.4s

        calc_scalefactors

        add     v15.4s, v0.4s, v1.4s
        add     v9.4s,  v2.4s, v3.4s
        ins     v15.s[3], w10
        ld1     {v8.4s}, [x4], #16
        cmgt    v15.4s, v15.4s, v9.4s

        and     v8.16b, v8.16b, v15.16b
        bit     v0.16b, v2.16b, v15.16b
        addp    v8.4s,  v8.4s,  v8.4s
        bit     v1.16b, v3.16b, v15.16b
        st1     {v0.4s}, [x7]
        st1     {v1.4s}, [x8]
        addp    v12.4s, v8.4s,  v8.4s  // save partial joint in v12

        update_joint_stereo_samples

        add     x5,  x0,  #0
        add     x6,  x0,  #32
        add     x7,  x1,  #0
        add     x8,  x1,  #32
        movi    v0.4s, #0x80, lsl #8
        add     v0.4s, v0.4s, v14.4s

        calc_scalefactors

        add     v15.4s, v0.4s, v1.4s
        add     v9.4s,  v2.4s, v3.4s
        ld1     {v8.4s}, [x4], #16
        cmgt    v15.4s, v15.4s, v9.4s

        and     v8.16b, v8.16b, v15.16b
        bit     v0.16b, v2.16b, v15.16b
        addp    v8.4s,  v8.4s,  v8.4s
        bit     v1.16b, v3.16b, v15.16b
        addp    v8.4s,  v8.4s,  v8.4s
        st1     {v0.4s}, [x7]
        add     v8.4s,  v8.4s,  v12.4s
        st1     {v1.4s}, [x8]
        mov     w0, v8.s[0]

        update_joint_stereo_samples
9:
        ldr     x19,      [sp, #0x40]
        ldp     d14, d15, [sp, #0x30]
        ldp     d12, d13, [sp, #0x20]
        ldp     d10, d11, [sp, #0x10]
        ldp     d8,  d9,  [sp], #0x50
        ret
endfunc

/*
 * ff_sbc_enc_process_input_4s_neon(
 *   int position,                       // w0 (return value)
 *   const uint8_t *pcm,                 // x1
 *   int16_t X[2][SBC_X_BUFFER_SIZE],    // x2
 *   int nsamples,                       // w3
 *   int nchannels)                      // w4
 *
 * AArch64: 5th arg in x4 (not on stack). x0-x18 caller-saved; no saves needed.
 * NEON: v0,v1 = perm table rows; v2-v7,v16-v19 = data (all caller-saved).
 */
function ff_sbc_enc_process_input_4s_neon, export=1
        movrel  x5,  ff_sbc_input_perm_4

        // Handle X buffer wraparound: copy 36 int16 (72 bytes) if position < nsamples
        cmp     w0,  w3
        b.ge    1f
        add     x7,  x2,  #576                 // &X[0][SBC_X_BUFFER_SIZE-40] = X+288*2
        add     x6,  x2,  w0, sxtw #1
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.4h}, [x6]                  // 8 bytes (= d-register width)
        st1     {v0.4h}, [x7]
        cmp     w4,  #1
        b.le    2f
        add     x7,  x2,  #1232                // &X[1][SBC_X_BUFFER_SIZE-40]
        add     x6,  x2,  #656
        add     x6,  x6,  w0, sxtw #1
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.4h}, [x6]
        st1     {v0.4h}, [x7]
2:
        mov     w0,  #288
1:
        add     x6,  x2,  w0, sxtw #1          // &X[0][position]
        add     x7,  x6,  #656                 // &X[1][position]

        // Load perm table: two 8-byte rows into d0, d1
        ldr     d0,  [x5]
        ldr     d1,  [x5, #8]

        cmp     w4,  #1
        b.le    8f                             // mono

7:      // Stereo: ld2 de-interleave
        sub     x6,  x6,  #16
        sub     x7,  x7,  #16
        sub     w0,  w0,  #8
        ld2     {v2.4h, v3.4h}, [x1], #16     // v2=ch0[0..3], v3=ch1[0..3]
        ld2     {v4.4h, v5.4h}, [x1], #16     // v4=ch0[4..7], v5=ch1[4..7]
        ins     v2.d[1], v4.d[0]              // v2.16b = ch0 all 8 samples
        ins     v3.d[1], v5.d[0]              // v3.16b = ch1 all 8 samples
        tbl     v16.8b, {v2.16b}, v0.8b
        tbl     v17.8b, {v2.16b}, v1.8b
        tbl     v18.8b, {v3.16b}, v0.8b
        tbl     v19.8b, {v3.16b}, v1.8b
        st1     {v16.8b, v17.8b}, [x6]
        st1     {v18.8b, v19.8b}, [x7]
        subs    w3,  w3,  #8
        b.gt    7b
        ret

8:      // Mono
1:
        sub     x6,  x6,  #16
        sub     w0,  w0,  #8
        ld1     {v2.16b}, [x1], #16
        tbl     v16.8b, {v2.16b}, v0.8b
        tbl     v17.8b, {v2.16b}, v1.8b
        st1     {v16.8b, v17.8b}, [x6]
        subs    w3,  w3,  #8
        b.gt    1b
        ret
endfunc

/*
 * ff_sbc_enc_process_input_8s_neon(
 *   int position,     // w0
 *   const uint8_t *pcm, // x1
 *   int16_t X[2][SBC_X_BUFFER_SIZE], // x2
 *   int nsamples,     // w3
 *   int nchannels)    // w4
 */
function ff_sbc_enc_process_input_8s_neon, export=1
        movrel  x5,  ff_sbc_input_perm_8

        // Handle X buffer wraparound: copy 72 int16 (144 bytes)
        cmp     w0,  w3
        b.ge    1f
        add     x7,  x2,  #512                 // &X[0][SBC_X_BUFFER_SIZE-72]
        add     x6,  x2,  w0, sxtw #1
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.8h, v1.8h}, [x6], #32
        st1     {v0.8h, v1.8h}, [x7], #32
        cmp     w4,  #1
        b.le    2f
        add     x7,  x2,  #1168                // &X[1][SBC_X_BUFFER_SIZE-72]
        add     x6,  x2,  #656
        add     x6,  x6,  w0, sxtw #1
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x6], #64
        st1     {v0.8h, v1.8h, v2.8h, v3.8h}, [x7], #64
        ld1     {v0.8h, v1.8h}, [x6], #32
        st1     {v0.8h, v1.8h}, [x7], #32
2:
        mov     w0,  #256
1:
        add     x6,  x2,  w0, sxtw #1
        add     x7,  x6,  #656

        // Load perm table: four 8-byte rows
        ldr     d0,  [x5]
        ldr     d1,  [x5, #8]
        ldr     d2,  [x5, #16]
        ldr     d3,  [x5, #24]

        cmp     w4,  #1
        b.le    8f                             // mono

7:      // Stereo: ld2 de-interleave two 32-byte blocks
        sub     x6,  x6,  #32
        sub     x7,  x7,  #32
        sub     w0,  w0,  #16
        ld2     {v4.8h, v5.8h}, [x1], #32     // v4=L[0..7], v5=R[0..7]
        ld2     {v6.8h, v7.8h}, [x1], #32     // v6=L[8..15], v7=R[8..15]
        // Rearrange: {v4,v5}→ch0, {v6,v7}→ch1
        mov     v16.16b, v5.16b
        mov     v5.16b,  v6.16b               // v5 ← L[8..15]
        mov     v6.16b,  v16.16b              // v6 ← R[0..7]
        // Now v4=L[0..7], v5=L[8..15], v6=R[0..7], v7=R[8..15]
        tbl     v16.8b, {v4.16b, v5.16b}, v0.8b
        tbl     v17.8b, {v4.16b, v5.16b}, v1.8b
        tbl     v18.8b, {v4.16b, v5.16b}, v2.8b
        tbl     v19.8b, {v4.16b, v5.16b}, v3.8b
        st1     {v16.8b, v17.8b, v18.8b, v19.8b}, [x6]
        tbl     v16.8b, {v6.16b, v7.16b}, v0.8b
        tbl     v17.8b, {v6.16b, v7.16b}, v1.8b
        tbl     v18.8b, {v6.16b, v7.16b}, v2.8b
        tbl     v19.8b, {v6.16b, v7.16b}, v3.8b
        st1     {v16.8b, v17.8b, v18.8b, v19.8b}, [x7]
        subs    w3,  w3,  #16
        b.gt    7b
        ret

8:      // Mono
1:
        sub     x6,  x6,  #32
        sub     w0,  w0,  #16
        ld1     {v4.16b, v5.16b}, [x1], #32
        tbl     v16.8b, {v4.16b, v5.16b}, v0.8b
        tbl     v17.8b, {v4.16b, v5.16b}, v1.8b
        tbl     v18.8b, {v4.16b, v5.16b}, v2.8b
        tbl     v19.8b, {v4.16b, v5.16b}, v3.8b
        st1     {v16.8b, v17.8b, v18.8b, v19.8b}, [x6]
        subs    w3,  w3,  #16
        b.gt    1b
        ret
endfunc
