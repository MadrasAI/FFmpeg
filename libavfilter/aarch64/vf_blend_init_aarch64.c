/*
 * AArch64 NEON optimised blend filter init
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

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/aarch64/cpu.h"
#include "libavfilter/blend.h"

#define BLEND_FUNC(name) \
void ff_blend_##name##_neon(const uint8_t *top, ptrdiff_t top_linesize,       \
                             const uint8_t *bottom, ptrdiff_t bottom_linesize, \
                             uint8_t *dst, ptrdiff_t dst_linesize,             \
                             ptrdiff_t width, ptrdiff_t height,                \
                             FilterParams *param, SliceParams *sliceparam);

/* 8-bit modes */
BLEND_FUNC(xor)
BLEND_FUNC(or)
BLEND_FUNC(and)
BLEND_FUNC(addition)
BLEND_FUNC(subtract)
BLEND_FUNC(darken)
BLEND_FUNC(lighten)
BLEND_FUNC(average)
BLEND_FUNC(difference)
BLEND_FUNC(grainextract)
BLEND_FUNC(grainmerge)
BLEND_FUNC(multiply)
BLEND_FUNC(screen)
BLEND_FUNC(hardmix)
BLEND_FUNC(phoenix)
BLEND_FUNC(extremity)
BLEND_FUNC(negation)

/* 16-bit modes */
BLEND_FUNC(addition_16)
BLEND_FUNC(subtract_16)
BLEND_FUNC(and_16)
BLEND_FUNC(or_16)
BLEND_FUNC(xor_16)
BLEND_FUNC(average_16)
BLEND_FUNC(darken_16)
BLEND_FUNC(lighten_16)
BLEND_FUNC(grainextract_16)
BLEND_FUNC(grainmerge_16)
BLEND_FUNC(phoenix_16)
BLEND_FUNC(difference_16)
BLEND_FUNC(extremity_16)
BLEND_FUNC(negation_16)

av_cold void ff_blend_init_aarch64(FilterParams *param, int depth)
{
    int cpu_flags = av_get_cpu_flags();

    if (!have_neon(cpu_flags) || param->opacity != 1)
        return;

    if (depth == 8) {
        switch (param->mode) {
        case BLEND_XOR:          param->blend = ff_blend_xor_neon;          break;
        case BLEND_OR:           param->blend = ff_blend_or_neon;           break;
        case BLEND_AND:          param->blend = ff_blend_and_neon;          break;
        case BLEND_ADDITION:     param->blend = ff_blend_addition_neon;     break;
        case BLEND_SUBTRACT:     param->blend = ff_blend_subtract_neon;     break;
        case BLEND_DARKEN:       param->blend = ff_blend_darken_neon;       break;
        case BLEND_LIGHTEN:      param->blend = ff_blend_lighten_neon;      break;
        case BLEND_AVERAGE:      param->blend = ff_blend_average_neon;      break;
        case BLEND_DIFFERENCE:   param->blend = ff_blend_difference_neon;   break;
        case BLEND_GRAINEXTRACT: param->blend = ff_blend_grainextract_neon; break;
        case BLEND_GRAINMERGE:   param->blend = ff_blend_grainmerge_neon;   break;
        case BLEND_MULTIPLY:     param->blend = ff_blend_multiply_neon;     break;
        case BLEND_SCREEN:       param->blend = ff_blend_screen_neon;       break;
        case BLEND_HARDMIX:      param->blend = ff_blend_hardmix_neon;      break;
        case BLEND_PHOENIX:      param->blend = ff_blend_phoenix_neon;      break;
        case BLEND_EXTREMITY:    param->blend = ff_blend_extremity_neon;    break;
        case BLEND_NEGATION:     param->blend = ff_blend_negation_neon;     break;
        default: break;
        }
    } else if (depth == 16) {
        switch (param->mode) {
        case BLEND_ADDITION:     param->blend = ff_blend_addition_16_neon;     break;
        case BLEND_SUBTRACT:     param->blend = ff_blend_subtract_16_neon;     break;
        case BLEND_AND:          param->blend = ff_blend_and_16_neon;          break;
        case BLEND_OR:           param->blend = ff_blend_or_16_neon;           break;
        case BLEND_XOR:          param->blend = ff_blend_xor_16_neon;          break;
        case BLEND_AVERAGE:      param->blend = ff_blend_average_16_neon;      break;
        case BLEND_DARKEN:       param->blend = ff_blend_darken_16_neon;       break;
        case BLEND_LIGHTEN:      param->blend = ff_blend_lighten_16_neon;      break;
        case BLEND_GRAINEXTRACT: param->blend = ff_blend_grainextract_16_neon; break;
        case BLEND_GRAINMERGE:   param->blend = ff_blend_grainmerge_16_neon;   break;
        case BLEND_PHOENIX:      param->blend = ff_blend_phoenix_16_neon;      break;
        case BLEND_DIFFERENCE:   param->blend = ff_blend_difference_16_neon;   break;
        case BLEND_EXTREMITY:    param->blend = ff_blend_extremity_16_neon;    break;
        case BLEND_NEGATION:     param->blend = ff_blend_negation_16_neon;     break;
        default: break;
        }
    }
}
