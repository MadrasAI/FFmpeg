# subsystems
OBJS-$(CONFIG_JPEG2000_DECODER)         += aarch64/jpeg2000dsp_init.o
OBJS-$(CONFIG_ALAC_DECODER)             += aarch64/alacdsp_init_aarch64.o
OBJS-$(CONFIG_SBC_ENCODER)              += aarch64/sbcdsp_init_aarch64.o
OBJS-$(CONFIG_VP3DSP)                   += aarch64/vp3dsp_init_aarch64.o
OBJS-$(CONFIG_AUDIODSP)                 += aarch64/audiodsp_init_aarch64.o
OBJS-$(CONFIG_DCA_DECODER)              += aarch64/dcadsp_init_aarch64.o
OBJS-$(CONFIG_FLAC_DECODER)             += aarch64/flacdsp_init_aarch64.o
OBJS-$(CONFIG_G722DSP)                  += aarch64/g722dsp_init_aarch64.o
OBJS-$(CONFIG_EXR_DECODER)              += aarch64/exrdsp_init_aarch64.o
OBJS-$(CONFIG_UTVIDEO_DECODER)          += aarch64/utvideodsp_init_aarch64.o
OBJS-$(CONFIG_MLP_DECODER)             += aarch64/mlpdsp_init_aarch64.o
OBJS-$(CONFIG_TAK_DECODER)             += aarch64/takdsp_init_aarch64.o
OBJS-$(CONFIG_LPC)                      += aarch64/lpc_init.o
OBJS-$(CONFIG_LLAUDDSP)                 += aarch64/lossless_audiodsp_init_aarch64.o
OBJS-$(CONFIG_LLVIDDSP)                 += aarch64/lossless_videodsp_init_aarch64.o
OBJS-$(CONFIG_BLOCKDSP)                 += aarch64/blockdsp_init_aarch64.o
OBJS-$(CONFIG_BSWAPDSP)                 += aarch64/bswapdsp_init_aarch64.o
OBJS-$(CONFIG_AC3DSP)                   += aarch64/ac3dsp_init_aarch64.o
OBJS-$(CONFIG_FDCTDSP)                  += aarch64/fdctdsp_init_aarch64.o
OBJS-$(CONFIG_FMTCONVERT)               += aarch64/fmtconvert_init.o
OBJS-$(CONFIG_H264CHROMA)               += aarch64/h264chroma_init_aarch64.o
OBJS-$(CONFIG_H264DSP)                  += aarch64/h264dsp_init_aarch64.o
OBJS-$(CONFIG_HUFFYUVDSP)               += aarch64/huffyuvdsp_init_aarch64.o
OBJS-$(CONFIG_HUFFYUVENCDSP)            += aarch64/huffyuvencdsp_init_aarch64.o
OBJS-$(CONFIG_H264PRED)                 += aarch64/h264pred_init.o
OBJS-$(CONFIG_H264QPEL)                 += aarch64/h264qpel_init_aarch64.o
OBJS-$(CONFIG_HPELDSP)                  += aarch64/hpeldsp_init_aarch64.o
OBJS-$(CONFIG_IDCTDSP)                  += aarch64/idctdsp_init_aarch64.o
OBJS-$(CONFIG_ME_CMP)                   += aarch64/me_cmp_init_aarch64.o
OBJS-$(CONFIG_MPEGAUDIODSP)             += aarch64/mpegaudiodsp_init.o
OBJS-$(CONFIG_MPEGVIDEOENCDSP)          += aarch64/mpegvideoencdsp_init.o
OBJS-$(CONFIG_NEON_CLOBBER_TEST)        += aarch64/neontest.o
OBJS-$(CONFIG_PIXBLOCKDSP)              += aarch64/pixblockdsp_init_aarch64.o
OBJS-$(CONFIG_PNG_DECODER)              += aarch64/pngdsp_init.o
OBJS-$(CONFIG_VIDEODSP)                 += aarch64/videodsp_init.o
OBJS-$(CONFIG_VP8DSP)                   += aarch64/vp8dsp_init_aarch64.o

# decoders/encoders
OBJS-$(CONFIG_AAC_DECODER)              += aarch64/aacpsdsp_init_aarch64.o \
                                           aarch64/sbrdsp_init_aarch64.o
OBJS-$(CONFIG_AAC_ENCODER)              += aarch64/aacencdsp_init.o
OBJS-$(CONFIG_DCA_DECODER)              += aarch64/synth_filter_init.o
OBJS-$(CONFIG_OPUS_DECODER)             += aarch64/opusdsp_init.o
OBJS-$(CONFIG_H263DSP)                  += aarch64/h263dsp_init_aarch64.o
OBJS-$(CONFIG_LLVIDENCDSP)              += aarch64/llvidencdsp_init_aarch64.o
OBJS-$(CONFIG_RV34DSP)                  += aarch64/rv34dsp_init_aarch64.o
OBJS-$(CONFIG_RV40_DECODER)             += aarch64/rv40dsp_init_aarch64.o
OBJS-$(CONFIG_VC1DSP)                   += aarch64/vc1dsp_init_aarch64.o
OBJS-$(CONFIG_VORBIS_DECODER)           += aarch64/vorbisdsp_init.o
OBJS-$(CONFIG_VP9_DECODER)              += aarch64/vp9dsp_init_10bpp_aarch64.o \
                                           aarch64/vp9dsp_init_12bpp_aarch64.o \
                                           aarch64/vp9mc_aarch64.o             \
                                           aarch64/vp9dsp_init_aarch64.o

# ARMv8 optimizations

# subsystems
ARMV8-OBJS-$(CONFIG_VIDEODSP)           += aarch64/videodsp.o

# NEON optimizations

# subsystems
NEON-OBJS-$(CONFIG_JPEG2000_DECODER)    += aarch64/jpeg2000dsp_neon.o
NEON-OBJS-$(CONFIG_H263DSP)             += aarch64/h263dsp_neon.o
NEON-OBJS-$(CONFIG_LLVIDENCDSP)         += aarch64/llvidencdsp_neon.o
NEON-OBJS-$(CONFIG_RV34DSP)             += aarch64/rv34dsp_neon.o
NEON-OBJS-$(CONFIG_ALAC_DECODER)        += aarch64/alacdsp_neon.o
NEON-OBJS-$(CONFIG_SBC_ENCODER)         += aarch64/sbcdsp_neon.o
NEON-OBJS-$(CONFIG_VP3DSP)              += aarch64/vp3dsp_neon.o
NEON-OBJS-$(CONFIG_AUDIODSP)            += aarch64/audiodsp_neon.o
NEON-OBJS-$(CONFIG_DCA_DECODER)         += aarch64/dcadsp_neon.o
NEON-OBJS-$(CONFIG_FLAC_DECODER)        += aarch64/flacdsp_neon.o
NEON-OBJS-$(CONFIG_G722DSP)             += aarch64/g722dsp_neon.o
NEON-OBJS-$(CONFIG_EXR_DECODER)         += aarch64/exrdsp_neon.o
NEON-OBJS-$(CONFIG_UTVIDEO_DECODER)     += aarch64/utvideodsp_neon.o
NEON-OBJS-$(CONFIG_MLP_DECODER)         += aarch64/mlpdsp_aarch64.o
NEON-OBJS-$(CONFIG_TAK_DECODER)         += aarch64/takdsp_neon.o
NEON-OBJS-$(CONFIG_LPC)                 += aarch64/lpc_neon.o
NEON-OBJS-$(CONFIG_LLAUDDSP)            += aarch64/lossless_audiodsp_neon.o
NEON-OBJS-$(CONFIG_LLVIDDSP)            += aarch64/lossless_videodsp_neon.o
NEON-OBJS-$(CONFIG_BLOCKDSP)            += aarch64/blockdsp_neon.o
NEON-OBJS-$(CONFIG_BSWAPDSP)            += aarch64/bswapdsp_neon.o
NEON-OBJS-$(CONFIG_AAC_DECODER)         += aarch64/sbrdsp_neon.o
NEON-OBJS-$(CONFIG_AAC_ENCODER)         += aarch64/aacencdsp_neon.o
NEON-OBJS-$(CONFIG_AC3DSP)              += aarch64/ac3dsp_neon.o
NEON-OBJS-$(CONFIG_FDCTDSP)             += aarch64/fdctdsp_neon.o
NEON-OBJS-$(CONFIG_FMTCONVERT)          += aarch64/fmtconvert_neon.o
NEON-OBJS-$(CONFIG_H264CHROMA)          += aarch64/h264cmc_neon.o
NEON-OBJS-$(CONFIG_H264DSP)             += aarch64/h264dsp_neon.o              \
                                           aarch64/h264idct_neon.o
NEON-OBJS-$(CONFIG_H264PRED)            += aarch64/h264pred_neon.o
NEON-OBJS-$(CONFIG_H264QPEL)            += aarch64/h264qpel_neon.o             \
                                           aarch64/hpeldsp_neon.o
NEON-OBJS-$(CONFIG_HPELDSP)             += aarch64/hpeldsp_neon.o
NEON-OBJS-$(CONFIG_HUFFYUVDSP)          += aarch64/huffyuvdsp_neon.o
NEON-OBJS-$(CONFIG_HUFFYUVENCDSP)       += aarch64/huffyuvencdsp_neon.o
NEON-OBJS-$(CONFIG_IDCTDSP)             += aarch64/idctdsp_neon.o              \
                                           aarch64/simple_idct_neon.o
NEON-OBJS-$(CONFIG_ME_CMP)              += aarch64/me_cmp_neon.o
NEON-OBJS-$(CONFIG_MPEGAUDIODSP)        += aarch64/mpegaudiodsp_neon.o
NEON-OBJS-$(CONFIG_MPEGVIDEOENCDSP)     += aarch64/mpegvideoencdsp_neon.o
NEON-OBJS-$(CONFIG_PIXBLOCKDSP)         += aarch64/pixblockdsp_neon.o
NEON-OBJS-$(CONFIG_PNG_DECODER)         += aarch64/pngdsp_neon.o
NEON-OBJS-$(CONFIG_VC1DSP)              += aarch64/vc1dsp_neon.o
NEON-OBJS-$(CONFIG_VP8DSP)              += aarch64/vp8dsp_neon.o

# decoders/encoders
NEON-OBJS-$(CONFIG_AAC_DECODER)         += aarch64/aacpsdsp_neon.o
NEON-OBJS-$(CONFIG_DCA_DECODER)         += aarch64/synth_filter_neon.o
NEON-OBJS-$(CONFIG_OPUS_DECODER)        += aarch64/opusdsp_neon.o
NEON-OBJS-$(CONFIG_VORBIS_DECODER)      += aarch64/vorbisdsp_neon.o
NEON-OBJS-$(CONFIG_VP9_DECODER)         += aarch64/vp9intrapred_neon.o          \
                                           aarch64/vp9itxfm_16bpp_neon.o       \
                                           aarch64/vp9itxfm_neon.o             \
                                           aarch64/vp9lpf_16bpp_neon.o         \
                                           aarch64/vp9lpf_neon.o               \
                                           aarch64/vp9mc_16bpp_neon.o          \
                                           aarch64/vp9mc_neon.o
NEON-OBJS-$(CONFIG_HEVC_DECODER)        += aarch64/hevcdsp_deblock_neon.o      \
                                           aarch64/hevcdsp_dequant_neon.o      \
                                           aarch64/hevcdsp_idct_neon.o         \
                                           aarch64/hevcdsp_init_aarch64.o      \
                                           aarch64/hevcpred_neon.o             \
                                           aarch64/hevcpred_init_aarch64.o     \
                                           aarch64/h26x/epel_neon.o            \
                                           aarch64/h26x/qpel_neon.o            \
                                           aarch64/h26x/sao_neon.o
