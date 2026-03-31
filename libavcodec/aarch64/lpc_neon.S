/*
 * AArch64 NEON optimised LPC DSP functions
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

        .section .rodata
        .align  4
pd_one_zero:
        .double 1.0, 0.0

        .text

/*
 * void ff_lpc_apply_welch_window_neon(const int32_t *data, ptrdiff_t len,
 *                                      double *w_data)
 *
 * Applies Welch window: w_data[i] = data[i] * (1 - w[i]^2)
 *
 * Restructured as two sequential forward passes to avoid symmetric indexing:
 *   Pass 1: w_data[0..n2-1],   w[i] = c - 1 - i  (decrement)
 *   Middle: w_data[n2] = 0.0   (odd length only)
 *   Pass 2: w_data[n2..len-1], w[i] = c - n2 + i  (increment)
 * where c = 2.0 / (len - 1), n2 = len / 2.
 *
 * Uses float64x2_t (2 f64 per vector). int32→f64 via sshll + scvtf.
 * All registers v0-v7 and v16-v21 (caller-saved only).
 */
function ff_lpc_apply_welch_window_neon, export=1
        cmp     x1,  #1
        b.le    .Lwelch_special

        // c = 2.0 / (len - 1)
        sub     x3,  x1,  #1
        scvtf   d0,  x3                  // (double)(len-1)
        fmov    d1,  #2.0
        fdiv    d0,  d1,  d0             // d0 = c

        // n2 = len >> 1  (stored in x3)
        asr     x3,  x1,  #1            // x3 = n2

        // v16 = {1.0, 1.0}  — permanent constant for 1 - w*w
        fmov    d16, #1.0
        dup     v16.2d, v16.d[0]

        // ================================================================
        // Pass 1: w_data[0..n2-1], w = c-1, c-2, c-3, ...
        // v19 = current weight pair {w0, w1}
        // v20 = step {-2.0, -2.0}
        // ================================================================
        fmov    d17, #1.0
        fsub    d17, d0,  d17            // c - 1.0
        fmov    d18, #2.0
        fsub    d18, d0,  d18            // c - 2.0
        ins     v19.d[0], v17.d[0]
        ins     v19.d[1], v18.d[0]      // v19 = {c-1, c-2}
        fmov    d20, #2.0
        fneg    d20, d20
        dup     v20.2d, v20.d[0]        // v20 = {-2.0, -2.0}

        mov     x4,  x3                 // x4 = remaining = n2
        lsr     x5,  x4,  #1           // x5 = pairs = n2/2
        cbz     x5,  .Lpass1_tail

.Lpass1_loop:
        ld1     {v1.2s},  [x0], #8     // 2x int32
        sshll   v1.2d,   v1.2s,  #0   // sign-extend → 2x int64
        scvtf   v1.2d,   v1.2d        // int64 → f64
        fmul    v2.2d,   v19.2d, v19.2d  // w*w
        fsub    v2.2d,   v16.2d, v2.2d   // 1 - w*w
        fmul    v1.2d,   v1.2d,  v2.2d   // data * coeff
        st1     {v1.2d}, [x2], #16
        fadd    v19.2d,  v19.2d, v20.2d  // w -= 2.0
        subs    x5,  x5,  #1
        b.ne    .Lpass1_loop

.Lpass1_tail:
        ands    x5,  x4,  #1
        b.eq    .Lpass1_done
        // 1 remaining element
        ldr     w6,  [x0], #4
        scvtf   d1,  w6                 // int32 → f64 via GPR
        fmov    d3,  d19                 // current w (low lane = v19.d[0])
        fmul    d3,  d3,  d3            // w*w
        fmov    d4,  #1.0
        fsub    d3,  d4,  d3            // 1 - w*w
        fmul    d1,  d1,  d3
        str     d1,  [x2], #8

.Lpass1_done:
        // Odd length: write 0.0 at w_data[n2], advance data past data[n2]
        tst     x1,  #1
        b.eq    .Lpass2_setup
        str     xzr, [x2], #8          // 0.0 (IEEE754 bit pattern = 0)
        add     x0,  x0,  #4           // skip data[n2]

.Lpass2_setup:
        // ================================================================
        // Pass 2: w_data[n2..len-1], w = c-n2, c-n2+1, c-n2+2, ...
        // ================================================================
        scvtf   d17, x3                 // (double)n2
        fsub    d17, d0,  d17           // c - n2
        fmov    d18, #1.0
        fadd    d18, d17, d18           // c - n2 + 1
        ins     v19.d[0], v17.d[0]
        ins     v19.d[1], v18.d[0]     // v19 = {c-n2, c-n2+1}
        fmov    d20, #2.0
        dup     v20.2d, v20.d[0]       // v20 = {+2.0, +2.0}

        mov     x4,  x3                // x4 = remaining = n2
        lsr     x5,  x4,  #1          // pairs
        cbz     x5,  .Lpass2_tail

.Lpass2_loop:
        ld1     {v1.2s},  [x0], #8
        sshll   v1.2d,   v1.2s,  #0
        scvtf   v1.2d,   v1.2d
        fmul    v2.2d,   v19.2d, v19.2d
        fsub    v2.2d,   v16.2d, v2.2d
        fmul    v1.2d,   v1.2d,  v2.2d
        st1     {v1.2d}, [x2], #16
        fadd    v19.2d,  v19.2d, v20.2d
        subs    x5,  x5,  #1
        b.ne    .Lpass2_loop

.Lpass2_tail:
        ands    x5,  x4,  #1
        b.eq    .Lpass2_done
        ldr     w6,  [x0]
        scvtf   d1,  w6
        fmov    d3,  d19                // current w (low lane = v19.d[0])
        fmul    d3,  d3,  d3
        fmov    d4,  #1.0
        fsub    d3,  d4,  d3
        fmul    d1,  d1,  d3
        str     d1,  [x2]

.Lpass2_done:
        ret

.Lwelch_special:
        // len == 0: nothing; len == 1: w_data[0] = 0.0
        cbz     x1,  .Lwelch_ret
        str     xzr, [x2]
.Lwelch_ret:
        ret
endfunc

/*
 * void ff_lpc_compute_autocorr_neon(const double *data, ptrdiff_t len,
 *                                    int lag, double *autoc)
 *
 * Computes autocorrelation at lags 0..lag inclusive.
 * autoc[j] = 1.0 + sum_{i=j}^{len-1}  data[i] * data[i-j]
 * autoc[j+1] = 1.0 + sum_{i=j}^{len-1} data[i] * data[i-j-1]  (when j+1 <= lag)
 *
 * Outer loop: j = 0, 2, ..., < lag (step 2), using fused float64x2_t fmla.
 * Accumulator initialized to {1.0, 0.0}:
 *   after faddp: (1 + sum_even) + (0 + sum_odd) = 1 + total  ✓
 *
 * Tail: ldr Dd zeroes upper lane → fmla with {val, 0} only updates lane 0 ✓
 *
 * Even-lag final entry: j == lag, inner loop starts at i = lag-1.
 *
 * All NEON in v0-v7, v16-v18 (caller-saved only).
 */
function ff_lpc_compute_autocorr_neon, export=1
        // lag is 'int' (32-bit): zero-extend to avoid garbage in upper 32 bits
        // (checkasm stack-clobber leaves 0xDEADBEEF in [sp+20..23] next to lag)
        uxtw    x2,  w2

        // Load address of {1.0, 0.0} constant
        adrp    x4,  pd_one_zero
        add     x4,  x4,  :lo12:pd_one_zero

        // Outer loop setup: j_bytes = 0, step 16; limit = lag * 8 bytes
        mov     x5,  xzr               // x5 = j_bytes = 0
        lsl     x6,  x2,  #3          // x6 = lag * 8 bytes (exclusive limit)

.Louter:
        cmp     x5,  x6
        b.ge    .Louter_done

        // Pointer setup for inner loop
        add     x7,  x0,  x5           // ptr_a = data + j  (advances)
        mov     x8,  x0                // ptr_b0 = data[0]  (advances)
        sub     x9,  x0,  #8          // ptr_b1 = data[-1] (advances)

        // remaining = len - j (integer)
        lsr     x10, x5,  #3          // j (integer)
        sub     x10, x1,  x10         // x10 = len - j

        // Init accumulators: v17 = v18 = {1.0, 0.0}
        ld1     {v17.2d}, [x4]
        ld1     {v18.2d}, [x4]

        // Inner loop: 2 doubles at a time
        lsr     x11, x10, #1          // pairs
        cbz     x11, .Linner_tail

.Linner:
        ld1     {v0.2d}, [x7], #16    // data[i], data[i+1]
        ld1     {v1.2d}, [x8], #16    // data[i-j], data[i+1-j]
        ld1     {v2.2d}, [x9], #16    // data[i-j-1], data[i+1-j-1]
        fmla    v17.2d, v0.2d, v1.2d  // sum0 += a * b0
        fmla    v18.2d, v0.2d, v2.2d  // sum1 += a * b1
        subs    x11, x11, #1
        b.ne    .Linner

.Linner_tail:
        tst     x10, #1
        b.eq    .Lreduce
        // ldr Dd zeroes upper lane → v*.d[1] += 0*0 = 0 (no effect)
        ldr     d0,  [x7]             // v0 = {data[i], 0.0}
        ldr     d1,  [x8]             // v1 = {data[i-j], 0.0}
        ldr     d2,  [x9]             // v2 = {data[i-j-1], 0.0}
        fmla    v17.2d, v0.2d, v1.2d
        fmla    v18.2d, v0.2d, v2.2d

.Lreduce:
        // Scalar faddp: d = v.2d lane0 + lane1
        faddp   d17, v17.2d           // sum0 reduced
        faddp   d18, v18.2d           // sum1 reduced
        add     x12, x3,  x5          // &autoc[j]
        str     d17, [x12]
        str     d18, [x12, #8]

        add     x5,  x5,  #16         // j_bytes += 16 (j += 2)
        b       .Louter

.Louter_done:
        // Even-lag case: j == lag → compute autoc[lag] separately
        // Inner loop: for i = lag-1..len-1: sum += data[i] * data[i-lag]
        cmp     x5,  x6               // x5 == lag*8 iff even lag
        b.ne    .Lautocorr_done

        // ptr_a = data + (lag-1)
        sub     x7,  x6,  #8          // (lag-1)*8
        add     x7,  x0,  x7          // data + (lag-1) doubles
        // ptr_b0 = data - 1 (data[i-lag] when i=lag-1 → data[-1])
        sub     x8,  x0,  #8

        // remaining = len - lag + 1
        sub     x10, x1,  x2          // len - lag
        add     x10, x10, #1          // len - lag + 1

        ld1     {v17.2d}, [x4]        // init {1.0, 0.0}

        lsr     x11, x10, #1
        cbz     x11, .Llast_tail

.Llast_inner:
        ld1     {v0.2d}, [x7], #16
        ld1     {v1.2d}, [x8], #16
        fmla    v17.2d, v0.2d, v1.2d
        subs    x11, x11, #1
        b.ne    .Llast_inner

.Llast_tail:
        tst     x10, #1
        b.eq    .Llast_reduce
        ldr     d0,  [x7]
        ldr     d1,  [x8]
        fmla    v17.2d, v0.2d, v1.2d

.Llast_reduce:
        faddp   d17, v17.2d
        add     x12, x3,  x6          // &autoc[lag]
        str     d17, [x12]

.Lautocorr_done:
        ret
endfunc
