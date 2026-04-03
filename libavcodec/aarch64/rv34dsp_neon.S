/*
 * RV30/40 IDCT NEON optimisations for AArch64
 * Ported from libavcodec/arm/rv34dsp_neon.S
 * Copyright (c) 2011 Janne Grunau <janne-libav@jannau.net>
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
 * rv34_inv_transform macro
 *
 * Computes the 4x4 16-bit IDCT in two passes: row-wise then column-wise.
 * Input:  \block_reg = pointer to int16_t block[16], stored row-major.
 *         v16.8h = block[0..7]  (row0 in lower, row1 in upper)
 *         v17.8h = block[8..15] (row2 in lower, row3 in upper)
 * Output: v1 = z0+z3, v2 = z1+z2, v3 = z1-z2, v5 = z0-z3 (all 4s int32)
 *
 * Register usage: v0-v5, v16-v23 (all caller-saved — no save/restore needed).
 *
 * ARM32 → AArch64 idioms used:
 *   vshll.s16 q, d_upper, #N  → sshll2 v.4s, v.8h, #N   (upper half)
 *   vmull.s16 q, d_lower, d   → smull  v.4s, v.4h, v.4h  (lower half)
 *   vmlal.s16 q, d_lower, d   → smlal  v.4s, v.4h, v.4h
 *   vmlsl.s16 q, d_lower, d   → smlsl  v.4s, v.4h, v.4h
 *   vsubw.s16 q, q, d_upper   → ssubw2 v.4s, v.4s, v.8h  (upper half)
 *   vaddw.s16 q, q, d_upper   → saddw2 v.4s, v.4s, v.8h
 *   vmul.s32 q, q, d[0]       → mul    v.4s, v.4s, v.s[0] (indexed scalar)
 *   vtrn.32+vswp 4x4 transpose → trn1/trn2 + zip1/zip2
 */
.macro rv34_inv_transform block_reg
        ld1             {v16.8h, v17.8h}, [\block_reg]
        // v16.8h: row0 in lower 4h, row1 in upper 4h
        // v17.8h: row2 in lower 4h, row3 in upper 4h

        // --- Row-wise IDCT ---
        movi            v0.4h,  #13

        // z2/z3 intermediates for rows 1 and 3
        sshll2          v18.4s, v16.8h,  #3         // row1*8  (z2 partial)
        sshll2          v19.4s, v16.8h,  #4         // row1*16 (z3 partial)
        sshll2          v20.4s, v17.8h,  #3         // row3*8
        sshll2          v21.4s, v17.8h,  #4         // row3*16

        // z0/z1 from rows 0 and 2
        smull           v22.4s, v16.4h,  v0.4h      // row0*13
        smlal           v22.4s, v17.4h,  v0.4h      // z0 = (row0+row2)*13
        smull           v23.4s, v16.4h,  v0.4h      // row0*13
        smlsl           v23.4s, v17.4h,  v0.4h      // z1 = (row0-row2)*13

        // Complete z2/z3 with *7 and *17 patterns
        ssubw2          v18.4s, v18.4s,  v16.8h     // z2 = row1*7
        saddw2          v19.4s, v19.4s,  v16.8h     // z3 partial = row1*17
        ssubw2          v20.4s, v20.4s,  v17.8h     // row3*7
        saddw2          v21.4s, v21.4s,  v17.8h     // row3*17

        // Final z2/z3 combining rows 1 and 3
        add             v19.4s, v19.4s,  v20.4s     // z3 = 17*row1 + 7*row3
        sub             v18.4s, v18.4s,  v21.4s     // z2 = 7*row1 - 17*row3

        // Row results: v1=z0+z3, v2=z1+z2, v3=z1-z2, v4=z0-z3
        add             v1.4s,  v22.4s,  v19.4s
        add             v2.4s,  v23.4s,  v18.4s
        sub             v3.4s,  v23.4s,  v18.4s
        sub             v4.4s,  v22.4s,  v19.4s

        // --- 4×4 int32 transpose (replaces ARM32 vtrn.32+vswp sequence) ---
        // v1=row0, v2=row1, v3=row2, v4=row3  →  col0..col3
        trn1            v16.4s, v1.4s,  v2.4s
        trn2            v17.4s, v1.4s,  v2.4s
        trn1            v18.4s, v3.4s,  v4.4s
        trn2            v19.4s, v3.4s,  v4.4s
        zip1            v1.2d,  v16.2d, v18.2d      // col 0
        zip1            v2.2d,  v17.2d, v19.2d      // col 1
        zip2            v3.2d,  v16.2d, v18.2d      // col 2
        zip2            v4.2d,  v17.2d, v19.2d      // col 3

        // --- Column-wise IDCT ---
        movi            v0.2s,  #13                 // 32-bit constant

        add             v16.4s, v1.4s,  v3.4s       // z0 = col0 + col2
        sub             v17.4s, v1.4s,  v3.4s       // z1 = col0 - col2

        shl             v18.4s, v2.4s,  #3          // col1*8
        shl             v19.4s, v2.4s,  #4          // col1*16
        mul             v20.4s, v17.4s, v0.s[0]     // z1*13
        shl             v21.4s, v4.4s,  #4          // col3*16
        add             v19.4s, v19.4s, v2.4s       // col1*17
        shl             v22.4s, v4.4s,  #3          // col3*8
        sub             v18.4s, v18.4s, v2.4s       // col1*7
        add             v21.4s, v21.4s, v4.4s       // col3*17
        mul             v23.4s, v16.4s, v0.s[0]     // z0*13
        sub             v22.4s, v22.4s, v4.4s       // col3*7
        sub             v18.4s, v18.4s, v21.4s      // z2 = col1*7 - col3*17
        add             v19.4s, v19.4s, v22.4s      // z3 = col1*17 + col3*7

        // Column results: v1=z0+z3, v2=z1+z2, v3=z1-z2, v5=z0-z3
        add             v1.4s,  v23.4s, v19.4s
        add             v2.4s,  v20.4s, v18.4s
        sub             v3.4s,  v20.4s, v18.4s
        sub             v5.4s,  v23.4s, v19.4s
.endm

/*
 * void ff_rv34_idct_add_neon(uint8_t *dst, ptrdiff_t stride, int16_t *block)
 *
 * Runs the 4×4 IDCT then adds residuals to dst in-place (uint8 clamp).
 * Also zeroes the block after reading it.
 */
function ff_rv34_idct_add_neon, export=1
        mov             x3,  x0                     // save dst base for stores

        rv34_inv_transform  x2                      // block in x2; results: v1,v2,v3,v5

        // Pack column int32 results into pairs of int16 via rounded narrow shift
        // rshrn+rshrn2 packs two col vectors into one 128-bit register:
        //   v6 = [col0>>10 (lower 4h) | col1>>10 (upper 4h)] = ARM32 q8
        //   v7 = [col2>>10 (lower 4h) | col3>>10 (upper 4h)] = ARM32 q9
        rshrn           v6.4h,  v1.4s,  #10
        rshrn2          v6.8h,  v2.4s,  #10
        rshrn           v7.4h,  v3.4s,  #10
        rshrn2          v7.8h,  v5.4s,  #10

        // vtrn.32 q8,q9 equivalent: interleave 32-bit words between v6 and v7
        trn1            v16.4s, v6.4s,  v7.4s
        trn2            v7.4s,  v6.4s,  v7.4s
        mov             v6.16b, v16.16b

        // vtrn.16 d16,d17 (lower↔upper halves of v6) → get row vectors
        // d16=lower(v6)=[c0r01,c2r01], d17=upper(v6)=[c1r01,c3r01]
        // After trn.16: d16→row0=[c0r0,c1r0,c2r0,c3r0], d17→row1
        ins             v16.d[0], v6.d[1]
        trn1            v17.4h, v6.4h,  v16.4h     // row 0
        trn2            v18.4h, v6.4h,  v16.4h     // row 1
        ins             v17.d[1], v18.d[0]          // v17 = [row0|row1] int16

        // vtrn.16 d18,d19 (lower↔upper halves of v7) → rows 2,3
        ins             v16.d[0], v7.d[1]
        trn1            v18.4h, v7.4h,  v16.4h     // row 2
        trn2            v19.4h, v7.4h,  v16.4h     // row 3
        ins             v18.d[1], v19.d[0]          // v18 = [row2|row3] int16

        // Zero the input block (ARM32: vst1.16 {q12}, [r2]! × 2)
        movi            v16.16b, #0
        st1             {v16.8h}, [x2], #16
        st1             {v16.8h}, [x2]

        // Load dst pixel rows and interleave into pairs
        // ld1r broadcasts 4 bytes to fill a 2s vector (like ARM32 vld1.32 {d[]})
        ld1r            {v4.2s},  [x0], x1          // [row0_pix, row0_pix]
        ld1r            {v5.2s},  [x0], x1          // [row1_pix, row1_pix]
        ld1r            {v6.2s},  [x0], x1          // [row2_pix, row2_pix]
        ld1r            {v7.2s},  [x0], x1          // [row3_pix, row3_pix]
        zip1            v4.2s,  v4.2s, v5.2s        // [row0_pix, row1_pix]
        zip1            v5.2s,  v6.2s, v7.2s        // [row2_pix, row3_pix]

        // Add IDCT residuals (int16) to pixels (uint8→int16) and saturate
        uaddw           v0.8h, v17.8h, v4.8b
        uaddw           v1.8h, v18.8h, v5.8b
        sqxtun          v4.8b, v0.8h
        sqxtun          v5.8b, v1.8h

        st1             {v4.s}[0], [x3], x1         // row 0
        st1             {v4.s}[1], [x3], x1         // row 1
        st1             {v5.s}[0], [x3], x1         // row 2
        st1             {v5.s}[1], [x3], x1         // row 3
        ret
endfunc

/*
 * void ff_rv34_inv_transform_noround_neon(int16_t *block)
 *
 * 4×4 IDCT with multiply-by-3 and truncating shift-11 (no rounding).
 */
function ff_rv34_inv_transform_noround_neon, export=1
        rv34_inv_transform  x0                      // results: v1,v2,v3,v5

        // Multiply each result by 3 (shl #1 + add) then truncating narrow >>11
        shl             v16.4s, v1.4s,  #1
        shl             v17.4s, v2.4s,  #1
        shl             v18.4s, v3.4s,  #1
        shl             v19.4s, v5.4s,  #1
        add             v16.4s, v16.4s, v1.4s       // (z0+z3)*3
        add             v17.4s, v17.4s, v2.4s       // (z1+z2)*3
        add             v18.4s, v18.4s, v3.4s       // (z1-z2)*3
        add             v19.4s, v19.4s, v5.4s       // (z0-z3)*3

        shrn            v0.4h, v16.4s, #11
        shrn            v1.4h, v17.4s, #11
        shrn            v2.4h, v18.4s, #11
        shrn            v3.4h, v19.4s, #11

        // vst4.16 {d[i], ...}[i] × 4: store rows of 4×16-bit interleaved
        st4             {v0.h, v1.h, v2.h, v3.h}[0], [x0], #8
        st4             {v0.h, v1.h, v2.h, v3.h}[1], [x0], #8
        st4             {v0.h, v1.h, v2.h, v3.h}[2], [x0], #8
        st4             {v0.h, v1.h, v2.h, v3.h}[3], [x0], #8
        ret
endfunc

/*
 * void ff_rv34_idct_dc_add_neon(uint8_t *dst, ptrdiff_t stride, int dc)
 *
 * DC-only IDCT: dc_scaled = (dc * 13*13 + 0x200) >> 10
 * Adds dc_scaled to all 16 pixels in the 4×4 block, clamps to [0,255].
 */
function ff_rv34_idct_dc_add_neon, export=1
        mov             x3,  x0                     // save dst pointer for stores

        dup             v0.4h, w2                   // [dc, dc, dc, dc] as int16
        movi            v1.4h, #169                 // [169, 169, 169, 169] = 13*13
        smull           v2.4s, v0.4h, v1.4h         // [dc*169, ...] as int32
        rshrn           v0.4h, v2.4s, #10           // [(dc*169+0x200)>>10, ...]
        ins             v0.d[1], v0.d[0]            // replicate: v0.8h = dc_scaled×8

        // Load 4 rows, keep [row0,row2] and [row1,row3] separately
        // (ARM32 never does vtrn on d28,d29 in this function — no row swap needed)
        ld1r            {v1.2s}, [x0], x1           // [row0_pix, row0_pix]
        ld1r            {v2.2s}, [x0], x1           // [row1_pix, row1_pix]
        ld1             {v1.s}[1], [x0], x1         // v1 = [row0_pix, row2_pix]
        ld1             {v2.s}[1], [x0], x1         // v2 = [row1_pix, row3_pix]

        uaddw           v3.8h, v0.8h, v1.8b         // [dc+row0 | dc+row2] rows 0,2
        uaddw           v4.8h, v0.8h, v2.8b         // [dc+row1 | dc+row3] rows 1,3
        sqxtun          v1.8b, v3.8h
        sqxtun          v2.8b, v4.8h

        st1             {v1.s}[0], [x3], x1         // row 0
        st1             {v2.s}[0], [x3], x1         // row 1
        st1             {v1.s}[1], [x3], x1         // row 2
        st1             {v2.s}[1], [x3], x1         // row 3
        ret
endfunc

/*
 * void ff_rv34_inv_transform_noround_dc_neon(int16_t *block)
 *
 * DC-only no-round transform: val = (block[0] * 13*13*3) >> 11
 * Fills all 16 int16_t block elements with val.
 */
function ff_rv34_inv_transform_noround_dc_neon, export=1
        ld1r            {v1.4h}, [x0]               // block[0] replicated × 4
        movi            v2.8h,  #0xFB               // 251 per lane
        orr             v2.8h,  #0x01, lsl #8       // 251|256 = 507 = 13*13*3
        smull           v3.4s,  v1.4h, v2.4h        // block[0]*507 per lane
        shrn            v0.4h,  v3.4s, #11          // (block[0]*507)>>11
        ins             v0.d[1], v0.d[0]            // replicate × 8
        st1             {v0.8h}, [x0], #16          // block[0..7]
        st1             {v0.8h}, [x0]               // block[8..15]
        ret
endfunc
