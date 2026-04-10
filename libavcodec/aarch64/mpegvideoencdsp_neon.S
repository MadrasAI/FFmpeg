/*
 * Copyright (c) 2024 Ramiro Polla
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

function ff_pix_sum16_neon, export=1
// x0  const uint8_t *pix
// x1  ptrdiff_t line_size

        add             x2, x0, x1
        lsl             x1, x1, #1
        movi            v0.16b, #0
        mov             w3, #16

1:
        ld1             {v1.16b}, [x0], x1
        ld1             {v2.16b}, [x2], x1
        subs            w3, w3, #2
        uadalp          v0.8h, v1.16b
        uadalp          v0.8h, v2.16b
        b.ne            1b

        uaddlv          s0, v0.8h
        fmov            w0, s0

        ret
endfunc

function ff_pix_norm1_neon, export=1
// x0  const uint8_t *pix
// x1  ptrdiff_t line_size

        movi            v0.16b, #0
        mov             w2, #16

1:
        ld1             {v1.16b}, [x0], x1
        subs            w2, w2, #1
        umull           v2.8h, v1.8b,  v1.8b
        umull2          v3.8h, v1.16b, v1.16b
        uadalp          v0.4s, v2.8h
        uadalp          v0.4s, v3.8h
        b.ne            1b

        uaddlv          d0, v0.4s
        fmov            w0, s0

        ret
endfunc

#if HAVE_DOTPROD
ENABLE_DOTPROD

function ff_pix_norm1_neon_dotprod, export=1
// x0  const uint8_t *pix
// x1  ptrdiff_t line_size

        movi            v0.16b, #0
        mov             w2, #16

1:
        ld1             {v1.16b}, [x0], x1
        ld1             {v2.16b}, [x0], x1
        udot            v0.4s, v1.16b, v1.16b
        subs            w2, w2, #2
        udot            v0.4s, v2.16b, v2.16b
        b.ne            1b

        uaddlv          d0, v0.4s
        fmov            w0, s0

        ret
endfunc

DISABLE_DOTPROD
#endif

/*
 * void ff_denoise_dct_neon(int16_t block[64], int dct_error_sum[64],
 *                          const uint16_t dct_offset[64])
 *
 * Vectorized denoiser: for each element,
 *   abs_level = abs(block[i])
 *   dct_error_sum[i] += abs_level            (accumulate into int32)
 *   block[i] = sign(block[i]) * max(0, abs_level - dct_offset[i])
 *
 * 8 elements per iteration × 8 iterations = 64 elements.
 * uqsub.8h replaces the x86 psubusw: max(0, abs-offset) in one instruction.
 * Sign restore: eor+sub (two's-complement conditional negate).
 */
function ff_denoise_dct_neon, export=1
        // x0=block (int16[64]), x1=dct_error_sum (int32[64]), x2=dct_offset (uint16[64])
        mov     w3, #8
1:
        ld1     {v0.8h},         [x0]           // block[i..i+7]
        ld1     {v1.8h},         [x2], #16       // dct_offset[i..i+7]
        ld1     {v4.4s, v5.4s},  [x1]           // dct_error_sum[i..i+7] (8×int32)
        cmlt    v2.8h,  v0.8h, #0               // sign mask: 0xFFFF where < 0, else 0
        abs     v3.8h,  v0.8h                   // abs(block[i])
        uaddw   v4.4s,  v4.4s, v3.4h            // sum[i..i+3] += abs (lower 4 lanes)
        uaddw2  v5.4s,  v5.4s, v3.8h            // sum[i+4..i+7] += abs (upper 4 lanes)
        uqsub   v6.8h,  v3.8h, v1.8h            // max(0, abs-offset) unsigned sat sub
        eor     v6.16b, v6.16b, v2.16b          // flip bits if original was negative
        sub     v6.8h,  v6.8h,  v2.8h           // +1 via sub(-1) → two's complement done
        st1     {v6.8h},         [x0], #16       // store denoised block
        st1     {v4.4s, v5.4s},  [x1], #32       // store updated error sums
        subs    w3, w3, #1
        b.ne    1b
        ret
endfunc

/*
 * void ff_add_8x8basis_neon(int16_t rem[64], const int16_t basis[64], int scale)
 *
 * rem[i] += (basis[i] * scale + 512) >> 10
 *
 * With scale ∈ [-2048, 2047] and basis ∈ [-15760, 15760], the product fits in
 * int32.  sshll widens basis to int32 for the multiply; sshr+sqxtn narrows
 * back to int16 after rounding.
 */
function ff_add_8x8basis_neon, export=1
        // x0=rem (int16[64]), x1=basis (int16[64]), w2=scale (int32)
        dup     v16.4s, w2                  // broadcast scale to all 4 lanes
        mov     w3, #512
        dup     v17.4s, w3                  // rounding constant 512 = 1 << 9
        mov     w3, #8
1:
        ld1     {v0.8h}, [x1], #16          // basis[i..i+7]
        ld1     {v1.8h}, [x0]               // rem[i..i+7]
        sshll   v2.4s,  v0.4h,  #0          // widen lower 4 basis elements to int32
        sshll2  v3.4s,  v0.8h,  #0          // widen upper 4 basis elements to int32
        mul     v2.4s,  v2.4s,  v16.4s      // basis * scale (int32)
        mul     v3.4s,  v3.4s,  v16.4s
        add     v2.4s,  v2.4s,  v17.4s      // + 512 (rounding)
        add     v3.4s,  v3.4s,  v17.4s
        sshr    v2.4s,  v2.4s,  #10         // arithmetic >> 10
        sshr    v3.4s,  v3.4s,  #10
        sqxtn   v4.4h,  v2.4s               // saturating narrow to int16
        sqxtn2  v4.8h,  v3.4s
        add     v1.8h,  v1.8h,  v4.8h       // rem += delta
        st1     {v1.8h}, [x0], #16
        subs    w3, w3, #1
        b.ne    1b
        ret
endfunc

/*
 * void ff_draw_edges_lr_neon(uint8_t *buf, ptrdiff_t wrap, int width,
 *                            int height, int w)
 *
 * For each of `height` rows: replicate the first byte of the row into
 * w bytes to the left, and the last byte into w bytes to the right.
 * w must be 4, 8, or 16.  wrap can be negative (upside-down buffers).
 */
function ff_draw_edges_lr_neon, export=1
        // x0=buf, x1=wrap (signed 64-bit), w2=width, w3=height, w4=w
        sxtw    x2,  w2                     // sign-extend width
        sxtw    x3,  w3                     // sign-extend height
        sxtw    x4,  w4                     // sign-extend w (edge width)
        neg     x5,  x4                     // x5 = -w (left-store offset)

        cmp     w4,  #16
        b.eq    .Llr16_loop
        cmp     w4,  #8
        b.eq    .Llr8_loop

        // w == 4: store 4 bytes each side
.Llr4_loop:
        ldrb    w6,  [x0]                   // first byte of row
        add     x7,  x0,  x2               // ptr + width
        ldrb    w8,  [x7, #-1]              // last byte of row
        dup     v0.8b, w6
        dup     v1.8b, w8
        str     s0,  [x0, x5]              // store 4 bytes left of row
        str     s1,  [x7]                  // store 4 bytes right of row
        add     x0,  x0,  x1              // advance by stride
        subs    x3,  x3,  #1
        b.ne    .Llr4_loop
        ret

        // w == 8: store 8 bytes each side
.Llr8_loop:
        ldrb    w6,  [x0]
        add     x7,  x0,  x2
        ldrb    w8,  [x7, #-1]
        dup     v0.8b, w6
        dup     v1.8b, w8
        str     d0,  [x0, x5]
        str     d1,  [x7]
        add     x0,  x0,  x1
        subs    x3,  x3,  #1
        b.ne    .Llr8_loop
        ret

        // w == 16: store 16 bytes each side
.Llr16_loop:
        ldrb    w6,  [x0]
        add     x7,  x0,  x2
        ldrb    w8,  [x7, #-1]
        dup     v0.16b, w6
        dup     v1.16b, w8
        str     q0,  [x0, x5]              // 16-byte store left
        str     q1,  [x7]                  // 16-byte store right
        add     x0,  x0,  x1
        subs    x3,  x3,  #1
        b.ne    .Llr16_loop
        ret
endfunc
