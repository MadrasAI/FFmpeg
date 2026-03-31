/*
 * AArch64 NEON optimised FLAC DSP functions
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

#include <stdint.h>

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavutil/samplefmt.h"
#include "libavcodec/flacdsp.h"

void ff_flac_wasted_32_neon(int32_t *decoded, int wasted, int len);
void ff_flac_wasted_33_neon(int64_t *decoded, const int32_t *residual,
                            int wasted, int len);
void ff_flac_lpc_16_neon(int32_t *decoded, const int coeffs[32],
                         int pred_order, int qlevel, int len);
void ff_flac_lpc_32_neon(int32_t *decoded, const int coeffs[32],
                         int pred_order, int qlevel, int len);

/* Saved C fallbacks — written once at init, read-only thereafter. */
static void (*s_lpc16_c)(int32_t *, const int [32], int, int, int);
static void (*s_lpc32_c)(int32_t *, const int [32], int, int, int);

/* NEON is only profitable for pred_order >= 20; delegate smaller orders to C. */
static void flac_lpc_16_aarch64(int32_t *decoded, const int coeffs[32],
                                int pred_order, int qlevel, int len)
{
    if (pred_order >= 20)
        ff_flac_lpc_16_neon(decoded, coeffs, pred_order, qlevel, len);
    else
        s_lpc16_c(decoded, coeffs, pred_order, qlevel, len);
}

static void flac_lpc_32_aarch64(int32_t *decoded, const int coeffs[32],
                                int pred_order, int qlevel, int len)
{
    if (pred_order >= 20)
        ff_flac_lpc_32_neon(decoded, coeffs, pred_order, qlevel, len);
    else
        s_lpc32_c(decoded, coeffs, pred_order, qlevel, len);
}

void ff_flac_decorrelate_ls_32_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_rs_32_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_ms_32_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_ls_16_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_rs_16_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_ms_16_neon(uint8_t **out, int32_t **in, int channels,
                                    int len, int shift);
void ff_flac_decorrelate_indep2_32_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);
void ff_flac_decorrelate_indep4_32_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);
void ff_flac_decorrelate_indep8_32_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);
void ff_flac_decorrelate_indep2_16_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);
void ff_flac_decorrelate_indep4_16_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);
void ff_flac_decorrelate_indep8_16_neon(uint8_t **out, int32_t **in, int channels,
                                        int len, int shift);

av_cold void ff_flacdsp_init_aarch64(FLACDSPContext *c, enum AVSampleFormat fmt,
                                     int channels)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->wasted32 = ff_flac_wasted_32_neon;
        c->wasted33 = ff_flac_wasted_33_neon;
        s_lpc16_c   = c->lpc16;          /* save C ptr before overwriting */
        s_lpc32_c   = c->lpc32;
        c->lpc16    = flac_lpc_16_aarch64;
        c->lpc32    = flac_lpc_32_aarch64;

        if (fmt == AV_SAMPLE_FMT_S32) {
            if      (channels == 2) c->decorrelate[0] = ff_flac_decorrelate_indep2_32_neon;
            else if (channels == 4) c->decorrelate[0] = ff_flac_decorrelate_indep4_32_neon;
            else if (channels == 8) c->decorrelate[0] = ff_flac_decorrelate_indep8_32_neon;
            c->decorrelate[1] = ff_flac_decorrelate_ls_32_neon;
            c->decorrelate[2] = ff_flac_decorrelate_rs_32_neon;
            c->decorrelate[3] = ff_flac_decorrelate_ms_32_neon;
        } else if (fmt == AV_SAMPLE_FMT_S16) {
            if      (channels == 2) c->decorrelate[0] = ff_flac_decorrelate_indep2_16_neon;
            else if (channels == 4) c->decorrelate[0] = ff_flac_decorrelate_indep4_16_neon;
            else if (channels == 8) c->decorrelate[0] = ff_flac_decorrelate_indep8_16_neon;
            c->decorrelate[1] = ff_flac_decorrelate_ls_16_neon;
            c->decorrelate[2] = ff_flac_decorrelate_rs_16_neon;
            c->decorrelate[3] = ff_flac_decorrelate_ms_16_neon;
        }
    }
}
