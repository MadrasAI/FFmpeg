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
 * void ff_mlp_filter_channel_aarch64(int32_t *state, const int32_t *coeff,
 *                                    int firorder, int iirorder,
 *                                    unsigned int filter_shift, int32_t mask,
 *                                    int blocksize, int32_t *sample_buffer)
 *
 * Register map:
 *   x0  = firbuf (state arg; decremented by 4 each outer iteration)
 *   x1  = fircoeff (coeff arg; constant)
 *   x2  = firorder (zero-extended from w2 at entry)
 *   x3  = iirorder (zero-extended from w3 at entry)
 *   w4  = filter_shift (used as asr amount; AArch64 asr masks to low 6 bits)
 *   w5  = mask
 *   x6  = blocksize (zero-extended from w6 at entry; outer loop counter)
 *   x7  = sample_buffer (incremented by MAX_CHANNELS*4 = 32 each outer iter)
 *   x8  = iirbuf = state + IIRBUF_OFFSET (decremented by 4 each outer iter)
 *   x9  = iircoeff = coeff + IIRCOEFF_OFFSET (constant)
 *   x10 = accum (int64; accumulated per-sample, reset to 0 each outer iter)
 *   w11 = residual = *sample_buffer (int32_t)
 *   x13 = inner loop order counter
 *   w14 = temp: firbuf[order] or iirbuf[order]
 *   w15 = temp: fircoeff[order] or iircoeff[order]
 *   x16 = accum + residual (int64); after mask: w16 = result (int32_t, upper bits 0)
 *   w17 = result - accum (int32_t, truncated); stored to iirbuf
 *
 * Constants (from mlp.h):
 *   MAX_BLOCKSIZE = 160, MAX_FIR_ORDER = 8, MAX_CHANNELS = 8
 *   IIRBUF_OFFSET   = (MAX_BLOCKSIZE + MAX_FIR_ORDER) * sizeof(int32_t) = 168*4 = 672
 *   IIRCOEFF_OFFSET = MAX_FIR_ORDER * sizeof(int32_t) = 8*4 = 32
 *   MAX_CHANNELS_STRIDE = MAX_CHANNELS * sizeof(int32_t) = 8*4 = 32
 */

#define IIRBUF_OFFSET           672
#define IIRCOEFF_OFFSET         32
#define MAX_CHANNELS_STRIDE     32

function ff_mlp_filter_channel_aarch64, export=1
        uxtw    x2,  w2                         // zero-extend firorder (prevent upper-32 garbage)
        uxtw    x3,  w3                         // zero-extend iirorder
        uxtw    x6,  w6                         // zero-extend blocksize
        cbz     x6,  9f                         // early exit if blocksize == 0

        add     x8,  x0,  #IIRBUF_OFFSET        // x8 = iirbuf
        add     x9,  x1,  #IIRCOEFF_OFFSET      // x9 = iircoeff

0:      // outer loop: process one sample
        ldr     w11, [x7]                       // residual = *sample_buffer
        mov     x10, xzr                        // accum = 0

        // FIR dot product: accum += sum(firbuf[i] * fircoeff[i], i=0..firorder-1)
        cbz     x2,  2f
        mov     x13, xzr
1:
        ldr     w14, [x0,  x13, lsl #2]         // firbuf[order]
        ldr     w15, [x1,  x13, lsl #2]         // fircoeff[order]
        smaddl  x10, w14, w15, x10              // accum += (int64_t)firbuf[order] * fircoeff[order]
        add     x13, x13, #1
        cmp     x13, x2
        b.lo    1b

2:      // IIR dot product: accum += sum(iirbuf[i] * iircoeff[i], i=0..iirorder-1)
        cbz     x3,  4f
        mov     x13, xzr
3:
        ldr     w14, [x8,  x13, lsl #2]         // iirbuf[order]
        ldr     w15, [x9,  x13, lsl #2]         // iircoeff[order]
        smaddl  x10, w14, w15, x10              // accum += (int64_t)iirbuf[order] * iircoeff[order]
        add     x13, x13, #1
        cmp     x13, x3
        b.lo    3b

4:
        asr     x10, x10, x4                    // accum >>= filter_shift (arithmetic)
        add     x16, x10, w11, sxtw             // x16 = accum + sign_extend(residual)
        and     w16, w16, w5                    // w16 = result = (accum + residual) & mask

        sub     x0,  x0,  #4
        str     w16, [x0]                       // *--firbuf = result

        sub     w17, w16, w10                   // w17 = (int32_t)(result - accum)
        sub     x8,  x8,  #4
        str     w17, [x8]                       // *--iirbuf = result - accum

        str     w16, [x7]                       // *sample_buffer = result
        add     x7,  x7,  #MAX_CHANNELS_STRIDE  // sample_buffer += MAX_CHANNELS

        subs    x6,  x6,  #1
        b.ne    0b

9:
        ret
endfunc
