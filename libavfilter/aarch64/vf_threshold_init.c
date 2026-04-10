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

#include <stdint.h>
#include <stddef.h>

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavfilter/threshold.h"

/*
 * AArch64 NEON declarations.
 * w must be a multiple of 16 (8-bit) or 8 (16-bit) pixels.
 * The ASM functions apply the threshold to all h rows.
 */
void ff_threshold8_neon(const uint8_t *in, const uint8_t *threshold,
                        const uint8_t *min, const uint8_t *max,
                        uint8_t *out,
                        ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                        ptrdiff_t flinesize, ptrdiff_t slinesize,
                        ptrdiff_t olinesize, int w, int h);
void ff_threshold16_neon(const uint8_t *in, const uint8_t *threshold,
                         const uint8_t *min, const uint8_t *max,
                         uint8_t *out,
                         ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                         ptrdiff_t flinesize, ptrdiff_t slinesize,
                         ptrdiff_t olinesize, int w, int h);

/*
 * C wrappers: align w to NEON lane width, call NEON for bulk,
 * call C scalar for the remaining unaligned tail.
 *
 * 8-bit: NEON processes multiples of 16 pixels.
 * 16-bit: NEON processes multiples of 8 pixels.
 *
 * Tail handling: fall back to C threshold8/threshold16 for remaining pixels.
 * All line sizes are passed through unchanged; only the aligned w portion
 * is given to NEON and the tail portion (offset by aligned_w) to C.
 */

static void threshold8_c_ref(const uint8_t *in, const uint8_t *threshold,
                              const uint8_t *min, const uint8_t *max,
                              uint8_t *out,
                              ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                              ptrdiff_t flinesize, ptrdiff_t slinesize,
                              ptrdiff_t olinesize, int w, int h)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)
            out[x] = in[x] <= threshold[x] ? min[x] : max[x];
        in        += ilinesize;
        threshold += tlinesize;
        min       += flinesize;
        max       += slinesize;
        out       += olinesize;
    }
}

static void threshold8_neon(const uint8_t *in, const uint8_t *threshold,
                             const uint8_t *min, const uint8_t *max,
                             uint8_t *out,
                             ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                             ptrdiff_t flinesize, ptrdiff_t slinesize,
                             ptrdiff_t olinesize, int w, int h)
{
    int w_aligned = w & ~15;
    if (w_aligned > 0)
        ff_threshold8_neon(in, threshold, min, max, out,
                           ilinesize, tlinesize, flinesize, slinesize, olinesize,
                           w_aligned, h);
    if (w > w_aligned)
        threshold8_c_ref(in + w_aligned, threshold + w_aligned,
                         min + w_aligned, max + w_aligned, out + w_aligned,
                         ilinesize, tlinesize, flinesize, slinesize, olinesize,
                         w - w_aligned, h);
}

static void threshold16_c_ref(const uint8_t *iin, const uint8_t *tthreshold,
                               const uint8_t *ffirst, const uint8_t *ssecond,
                               uint8_t *oout,
                               ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                               ptrdiff_t flinesize, ptrdiff_t slinesize,
                               ptrdiff_t olinesize, int w, int h)
{
    const uint16_t *in        = (const uint16_t *)iin;
    const uint16_t *threshold = (const uint16_t *)tthreshold;
    const uint16_t *min       = (const uint16_t *)ffirst;
    const uint16_t *max       = (const uint16_t *)ssecond;
    uint16_t       *out       = (uint16_t *)oout;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++)
            out[x] = in[x] <= threshold[x] ? min[x] : max[x];
        in        += ilinesize / 2;
        threshold += tlinesize / 2;
        min       += flinesize / 2;
        max       += slinesize / 2;
        out       += olinesize / 2;
    }
}

static void threshold16_neon(const uint8_t *in, const uint8_t *threshold,
                              const uint8_t *min, const uint8_t *max,
                              uint8_t *out,
                              ptrdiff_t ilinesize, ptrdiff_t tlinesize,
                              ptrdiff_t flinesize, ptrdiff_t slinesize,
                              ptrdiff_t olinesize, int w, int h)
{
    /* Align to 8 pixels (= 16 bytes for uint16) */
    int w_aligned = w & ~7;
    if (w_aligned > 0)
        ff_threshold16_neon(in, threshold, min, max, out,
                            ilinesize, tlinesize, flinesize, slinesize, olinesize,
                            w_aligned, h);
    if (w > w_aligned) {
        int tail   = w - w_aligned;
        int offset = w_aligned * 2;          /* byte offset into each row */
        threshold16_c_ref(in + offset, threshold + offset,
                          min + offset, max + offset, out + offset,
                          ilinesize, tlinesize, flinesize, slinesize, olinesize,
                          tail, h);
    }
}

av_cold void ff_threshold_init_aarch64(ThresholdContext *s)
{
    int cpu_flags = av_get_cpu_flags();
    if (have_neon(cpu_flags)) {
        if (s->depth == 8)
            s->threshold = threshold8_neon;
        else
            s->threshold = threshold16_neon;
    }
}
