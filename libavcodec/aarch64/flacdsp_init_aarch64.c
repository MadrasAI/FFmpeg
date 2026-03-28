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

av_cold void ff_flacdsp_init_aarch64(FLACDSPContext *c, enum AVSampleFormat fmt,
                                     int channels)
{
    int cpu_flags = av_get_cpu_flags();

    if (have_neon(cpu_flags)) {
        c->wasted32 = ff_flac_wasted_32_neon;
        c->wasted33 = ff_flac_wasted_33_neon;

        if (fmt == AV_SAMPLE_FMT_S32) {
            c->decorrelate[1] = ff_flac_decorrelate_ls_32_neon;
            c->decorrelate[2] = ff_flac_decorrelate_rs_32_neon;
            c->decorrelate[3] = ff_flac_decorrelate_ms_32_neon;
        } else if (fmt == AV_SAMPLE_FMT_S16) {
            c->decorrelate[1] = ff_flac_decorrelate_ls_16_neon;
            c->decorrelate[2] = ff_flac_decorrelate_rs_16_neon;
            c->decorrelate[3] = ff_flac_decorrelate_ms_16_neon;
        }
    }
}
