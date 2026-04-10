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
 * void ff_postscale_slice_neon(float *buffer, int length,
 *                              float postscale, float min, float max)
 * x0=buffer  w1=length  s0=postscale (v0.s[0])  s1=min (v1.s[0])  s2=max (v2.s[0])
 *
 * Per element: buffer[i] = clamp(buffer[i] * postscale, min, max)
 * Main loop: 4 floats per iteration (fmul + fmax + fmin).
 */
function ff_postscale_slice_neon, export=1
        dup     v16.4s, v0.s[0]    // broadcast postscale
        dup     v17.4s, v1.s[0]    // broadcast min
        dup     v18.4s, v2.s[0]    // broadcast max

        sxtw    x1, w1             // length → 64-bit
        lsr     x2, x1, #2         // x2 = length / 4 (4-float groups)
        cbz     x2, .Lpost_tail

.Lpost_main:
        ld1     {v0.4s}, [x0]
        fmul    v0.4s, v0.4s, v16.4s
        fmax    v0.4s, v0.4s, v17.4s
        fmin    v0.4s, v0.4s, v18.4s
        st1     {v0.4s}, [x0], #16
        subs    x2, x2, #1
        b.gt    .Lpost_main

.Lpost_tail:
        ands    x2, x1, #3         // remaining 0-3 floats
        b.eq    .Lpost_end
.Lpost_tail_loop:
        ldr     s0, [x0]
        fmul    s0, s0, s16
        fmax    s0, s0, s17
        fmin    s0, s0, s18
        str     s0, [x0], #4
        subs    x2, x2, #1
        b.gt    .Lpost_tail_loop

.Lpost_end:
        ret
endfunc

/*
 * void ff_verti_slice_neon(float *buffer, int width, int height,
 *                          int slice_start, int slice_end, int steps,
 *                          float nu, float bscale)
 * x0=buffer  w1=width  w2=height  w3=slice_start  w4=slice_end  w5=steps
 * s0=nu (v0.s[0])  s1=bscale (v1.s[0])
 *
 * For column groups of 4 across [slice_start, slice_end):
 *   For each step:
 *     Scale first row by bscale.
 *     Downward IIR:  row[i][0..3] += nu * row[i-1][0..3]
 *     Scale last row by bscale.
 *     Upward IIR:    row[i-1][0..3] += nu * row[i][0..3]
 * Then scalar tail for remaining 1-3 columns.
 *
 * Does NOT use callee-saved registers — all within x0-x15, v0-v7, v16-v18.
 * v16.4s = nu  v17.4s = bscale  s16 = nu (for scalar)  s17 = bscale (for scalar)
 */
function ff_verti_slice_neon, export=1
        sxtw    x1, w1             // width
        sxtw    x2, w2             // height
        sxtw    x3, w3             // slice_start
        sxtw    x4, w4             // slice_end
        sxtw    x5, w5             // steps

        dup     v16.4s, v0.s[0]   // nu broadcast (NEON 4s)
        dup     v17.4s, v1.s[0]   // bscale broadcast
        // Also keep scalar versions for tail loop (s16=v16.s[0], already set)

        lsl     x6, x1, #2        // width_bytes = width * 4

        // ptr = buffer + slice_start * 4
        add     x7, x0, x3, lsl #2

        // number of 4-column NEON groups: ngroups = (slice_end - slice_start) / 4
        sub     x8, x4, x3        // total_cols
        lsr     x8, x8, #2        // ngroups = total_cols >> 2
        cbz     x8, .Lvert_scalar_setup

        // --- 4-column NEON loop ---
.Lvert4_col_loop:
        mov     x9, x5            // step counter = steps

.Lvert4_step_loop:
        cbz     x9, .Lvert4_col_done

        // Scale first row: ptr[0..3] *= bscale
        ld1     {v0.4s}, [x7]
        fmul    v0.4s, v0.4s, v17.4s
        st1     {v0.4s}, [x7]

        // Downward IIR: row[i] += nu * row[i-1], for i=1..height-1
        mov     x10, x7           // x10 = prev row ptr (starts at row 0)
        add     x11, x7, x6       // x11 = curr row ptr (row 1)
        mov     x12, x2           // row counter = height
        sub     x12, x12, #1     // = height - 1

.Lvert4_down:
        cbz     x12, .Lvert4_down_done
        ld1     {v1.4s}, [x10]
        ld1     {v2.4s}, [x11]
        fmla    v2.4s, v1.4s, v16.4s
        st1     {v2.4s}, [x11]
        add     x10, x10, x6
        add     x11, x11, x6
        subs    x12, x12, #1
        b.gt    .Lvert4_down

.Lvert4_down_done:
        // x10 now points to the last row; scale it
        ld1     {v0.4s}, [x10]
        fmul    v0.4s, v0.4s, v17.4s
        st1     {v0.4s}, [x10]

        // Upward IIR: row[i-1] += nu * row[i], for i=height-1..1
        mov     x11, x10          // x11 = curr (= last row, the "below" ptr)
        sub     x10, x10, x6      // x10 = the row above last (second to last)
        mov     x12, x2
        sub     x12, x12, #1

.Lvert4_up:
        cbz     x12, .Lvert4_up_done
        ld1     {v1.4s}, [x11]    // below row
        ld1     {v2.4s}, [x10]    // current row
        fmla    v2.4s, v1.4s, v16.4s
        st1     {v2.4s}, [x10]
        sub     x10, x10, x6
        sub     x11, x11, x6
        subs    x12, x12, #1
        b.gt    .Lvert4_up

.Lvert4_up_done:
        subs    x9, x9, #1
        b.gt    .Lvert4_step_loop

.Lvert4_col_done:
        add     x7, x7, #16       // ptr += 4 floats (4 columns)
        subs    x8, x8, #1        // ngroups--
        b.gt    .Lvert4_col_loop

.Lvert_scalar_setup:
        // scalar tail: remaining (total_cols & 3) columns
        // x7 now points to the first unprocessed column
        // total remaining = slice_end - current_x
        // current_x = slice_start + ngroups_done * 4
        // Compute remaining: slice_end*4 = x4*4 bytes, x7 is current byte ptr
        add     x8, x0, x4, lsl #2   // end byte ptr = buffer + slice_end*4
        cmp     x7, x8
        b.ge    .Lvert_done

.Lvert1_col_loop:
        // One scalar column at a time
        mov     x9, x5            // step counter

.Lvert1_step_loop:
        cbz     x9, .Lvert1_col_done

        // Scale first row (1 float)
        ldr     s0, [x7]
        fmul    s0, s0, s17
        str     s0, [x7]

        // Downward IIR (scalar)
        mov     x10, x7
        add     x11, x7, x6
        mov     x12, x2
        sub     x12, x12, #1

.Lvert1_down:
        cbz     x12, .Lvert1_down_done
        ldr     s1, [x10]
        ldr     s2, [x11]
        fmadd   s2, s1, s16, s2
        str     s2, [x11]
        add     x10, x10, x6
        add     x11, x11, x6
        subs    x12, x12, #1
        b.gt    .Lvert1_down

.Lvert1_down_done:
        // Scale last row
        ldr     s0, [x10]
        fmul    s0, s0, s17
        str     s0, [x10]

        // Upward IIR (scalar)
        mov     x11, x10
        sub     x10, x10, x6
        mov     x12, x2
        sub     x12, x12, #1

.Lvert1_up:
        cbz     x12, .Lvert1_up_done
        ldr     s1, [x11]
        ldr     s2, [x10]
        fmadd   s2, s1, s16, s2
        str     s2, [x10]
        sub     x10, x10, x6
        sub     x11, x11, x6
        subs    x12, x12, #1
        b.gt    .Lvert1_up

.Lvert1_up_done:
        subs    x9, x9, #1
        b.gt    .Lvert1_step_loop

.Lvert1_col_done:
        add     x7, x7, #4        // advance 1 column (4 bytes)
        cmp     x7, x8            // compare with end ptr
        b.lt    .Lvert1_col_loop

.Lvert_done:
        ret
endfunc
