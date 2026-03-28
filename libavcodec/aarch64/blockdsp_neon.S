/*
 * AArch64 NEON optimised block functions
 * Copyright (c) 2008 Mans Rullgard <mans@mansr.com>
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

// void ff_clear_block_neon(int16_t *block)
// Zero one 8x8 int16 block = 64 * 2 = 128 bytes = 8 x 16-byte stores
function ff_clear_block_neon, export=1
        movi            v0.16b, #0
        .rept           8
        st1             {v0.16b}, [x0], #16
        .endr
        ret
endfunc

// void ff_clear_blocks_neon(int16_t *blocks)
// Zero six 8x8 int16 blocks = 6*128 = 768 bytes = 48 x 16-byte stores
function ff_clear_blocks_neon, export=1
        movi            v0.16b, #0
        .rept           48
        st1             {v0.16b}, [x0], #16
        .endr
        ret
endfunc

// void ff_fill_block16_neon(uint8_t *block, uint8_t value, ptrdiff_t line_size, int h)
// x0 = block, w1 = value, x2 = line_size, w3 = h
function ff_fill_block16_neon, export=1
        dup             v0.16b, w1
1:      st1             {v0.16b}, [x0], x2
        subs            w3,  w3,  #1
        b.ne            1b
        ret
endfunc

// void ff_fill_block8_neon(uint8_t *block, uint8_t value, ptrdiff_t line_size, int h)
// x0 = block, w1 = value, x2 = line_size, w3 = h
function ff_fill_block8_neon, export=1
        dup             v0.8b, w1
1:      st1             {v0.8b}, [x0], x2
        subs            w3,  w3,  #1
        b.ne            1b
        ret
endfunc
