/*
 * Copyright (c) 2025 FFmpeg contributors
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <string.h>

#include "checkasm.h"
#include "libavcodec/mlpdsp.h"
#include "libavcodec/mlp.h"
#include "libavcodec/mathops.h"
#include "libavutil/common.h"
#include "libavutil/mem_internal.h"

/*
 * State buffer layout (from mlpdec.c filter_channel()):
 *
 *   int32_t state_buffer[NUM_FILTERS][MAX_BLOCKSIZE + MAX_FIR_ORDER];
 *   int32_t *firbuf = state_buffer[FIR] + MAX_BLOCKSIZE;   <- `state` arg
 *   int32_t *iirbuf = state_buffer[IIR] + MAX_BLOCKSIZE;   <- state + MAX_BLOCKSIZE + MAX_FIR_ORDER
 *
 * The C implementation reads firbuf[0..firorder-1] and iirbuf[0..iirorder-1]
 * as history, then decrements both pointers for each of `blocksize` samples.
 * Allocate 2*(MAX_BLOCKSIZE + MAX_FIR_ORDER) and pass buf + MAX_BLOCKSIZE.
 */
#define STATE_SIZE  (2 * (MAX_BLOCKSIZE + MAX_FIR_ORDER))
#define BLOCKSIZE   40   /* typical TrueHD/MLP blocksize */

/*
 * Valid (firorder, iirorder) pairs.
 * MLP spec enforces firorder + iirorder <= MAX_FIR_ORDER (= 8).
 * (8,4) is excluded — 8+4=12 > 8 violates the constraint checked in mlpdec.c.
 */
static const struct { int fir; int iir; } filter_cases[] = {
    { 0, 0 },
    { 4, 0 },
    { 8, 0 },
    { 0, 4 },
    { 4, 4 },
    { 1, 0 },
    { 2, 1 },
    { 3, 3 },
};

static void check_mlp_filter_channel(void)
{
    /* Use 16-bit-range random values to prevent int64_t accumulator overflow.
     * Worst case: 8 products of (2^15)*(2^15) = 8 * 2^30 = 2^33 — safe. */
    LOCAL_ALIGNED_32(int32_t, state_ref, [STATE_SIZE]);
    LOCAL_ALIGNED_32(int32_t, state_new, [STATE_SIZE]);
    LOCAL_ALIGNED_16(int32_t, coeff,     [MAX_FIR_ORDER + MAX_IIR_ORDER]);
    /* sample_buffer is strided by MAX_CHANNELS=8 per sample */
    LOCAL_ALIGNED_32(int32_t, sbuf_ref,  [BLOCKSIZE * MAX_CHANNELS]);
    LOCAL_ALIGNED_32(int32_t, sbuf_new,  [BLOCKSIZE * MAX_CHANNELS]);
    MLPDSPContext c;

    ff_mlpdsp_init(&c);
    if (!check_func(c.mlp_filter_channel, "mlp_filter_channel"))
        return;

    declare_func(void, int32_t *, const int32_t *,
                 int, int, unsigned int, int32_t, int, int32_t *);

    for (int t = 0; t < FF_ARRAY_ELEMS(filter_cases); t++) {
        int firorder     = filter_cases[t].fir;
        int iirorder     = filter_cases[t].iir;
        unsigned int filter_shift = rnd() & 0x7;   /* 0..7 */
        int32_t mask     = -1;                      /* most common: all bits */

        /* Randomize state — bounded to 16-bit signed to avoid 64-bit overflow */
        for (int i = 0; i < STATE_SIZE; i++)
            state_ref[i] = state_new[i] = sign_extend(rnd(), 16);

        /* Randomize coefficients — bounded to 16-bit signed */
        for (int i = 0; i < MAX_FIR_ORDER + MAX_IIR_ORDER; i++)
            coeff[i] = sign_extend(rnd(), 16);

        /* Randomize sample_buffer — only the [0] element per stride is used */
        for (int i = 0; i < BLOCKSIZE; i++)
            sbuf_ref[i * MAX_CHANNELS] = sbuf_new[i * MAX_CHANNELS] =
                sign_extend(rnd(), 24);

        call_ref(state_ref + MAX_BLOCKSIZE, coeff,
                 firorder, iirorder, filter_shift, mask,
                 BLOCKSIZE, sbuf_ref);
        call_new(state_new + MAX_BLOCKSIZE, coeff,
                 firorder, iirorder, filter_shift, mask,
                 BLOCKSIZE, sbuf_new);

        /* Compare sample_buffer outputs (only the strided elements matter) */
        for (int i = 0; i < BLOCKSIZE; i++) {
            if (sbuf_ref[i * MAX_CHANNELS] != sbuf_new[i * MAX_CHANNELS]) {
                fail();
                break;
            }
        }
        /* Compare full state (both FIR and IIR regions) */
        if (memcmp(state_ref, state_new, STATE_SIZE * sizeof(int32_t)))
            fail();
    }

    /* Benchmark with worst-case filter orders (fir=4, iir=4) */
    {
        for (int i = 0; i < STATE_SIZE; i++)
            state_ref[i] = sign_extend(rnd(), 16);
        for (int i = 0; i < MAX_FIR_ORDER + MAX_IIR_ORDER; i++)
            coeff[i] = sign_extend(rnd(), 16);
        for (int i = 0; i < BLOCKSIZE; i++)
            sbuf_ref[i * MAX_CHANNELS] = sign_extend(rnd(), 24);
        bench_new(state_ref + MAX_BLOCKSIZE, coeff,
                  4, 4, 4, -1, BLOCKSIZE, sbuf_ref);
    }

    report("mlp_filter_channel");
}

void checkasm_check_mlpdsp(void)
{
    check_mlp_filter_channel();
}
