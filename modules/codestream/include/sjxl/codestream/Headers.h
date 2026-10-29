// Copyright (c) 2026 jesus luque.
//
// The codestream syntax the CPU writes: signature, SizeHeader, ImageMetadata,
// FrameHeader, TOC, and the DcGlobal quantizer fields. Everything else in a
// frame is emitted by kernels (shaders/sjxl/enc).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "sjxl/codestream/BitWriter.h"

namespace sjxl::codestream {

constexpr uint32_t kBlockDim = 8;
constexpr uint32_t kGroupDimBlocks = 32;      ///< AC group: 256 px
constexpr uint32_t kDcGroupDimBlocks = 256;   ///< DC group: 2048 px
constexpr uint32_t kColorTileDimBlocks = 8;   ///< chroma-from-luma tile: 64 px

/// Signature + SizeHeader + ImageMetadata (all_default: 8-bit sRGB, xyb) +
/// transform_data (all_default). Byte-aligned on exit.
/// ImageMetadata animation: ticks per second as a fraction, loops (0: forever).
struct Animation {
    uint32_t tpsNumerator = 10;
    uint32_t tpsDenominator = 1;
    uint32_t numLoops = 0;
};

/// A frame's place in the codestream: whether it is the last one and, in an
/// animation, its duration in ticks.
struct FrameInfo {
    bool     isLast = true;
    bool     animated = false;   ///< the ImageMetadata has an animation: the duration field is written
    uint32_t duration = 0;
};

/// `gamma` 0: the all_default sRGB encoding; else a gamma transfer (e.g. 0.45455)
/// over sRGB primaries and D65. With `animation` the metadata carries the
/// animation header (extra_fields).
void writeCodestreamHeaders(BitWriter& w, uint32_t xsize, uint32_t ysize, float gamma = 0.0f, uint32_t bitDepth = 8, bool alpha = false,
                            const Animation* animation = nullptr);
/// Signature, SizeHeader and the ImageMetadata of an 8- or 16-bit sRGB image
/// that is not XYB encoded (a lossless Modular frame), with an unassociated
/// alpha channel of the same depth when `alpha`. Byte-aligned on exit.
void writeCodestreamHeadersRgb(BitWriter& w, uint32_t xsize, uint32_t ysize, uint32_t bitDepth, bool alpha, float gamma = 0.0f,
                               const Animation* animation = nullptr);
/// A single-pass, full-frame, last Modular frame in RGB (no colour transform,
/// group_size_shift 1, LoopFilter off) with `numExtraChannels` unsampled extra
/// channels. Not byte-aligned on exit.
void writeModularFrameHeader(BitWriter& w, uint32_t numExtraChannels, const FrameInfo& info = FrameInfo{}, uint32_t groupSizeShift = 1);

/// libjxl's modular group size (enc_frame.cc GetGroupSizeShift): 128 px groups
/// up to 128 px, 256 up to 256, 512 up to 400 px, 256 beyond.
[[nodiscard]] constexpr uint32_t modularGroupSizeShift(uint32_t xsize, uint32_t ysize) {
    if (xsize <= 128 && ysize <= 128) return 0;
    if (xsize <= 256 && ysize <= 256) return 1;
    if (xsize <= 400 && ysize <= 400) return 2;
    return 1;
}

struct FrameParams {
    bool     gaborish = true;      ///< LoopFilter.gab
    uint32_t epfIters = 2;         ///< LoopFilter.epf_iters (0..3)
    uint32_t xQmScale = 2;         ///< neutral chroma quant-matrix scale
    uint32_t bQmScale = 2;
    /// Passes (frame_header.cc Passes): the AC coefficients split over
    /// `numPasses` sections per group; `shift[i]` quantisation shift of pass i
    /// (0 for the last); `downsample[j]` the factor pass `lastPass[j]` completes.
    uint32_t numPasses = 1;
    std::array<uint32_t, 11> shift{};
    uint32_t numDownsample = 0;
    std::array<uint32_t, 4> downsample{}, lastPass{};
    /// kUseDcFrame: the DC comes from the DC frame before this one (dc_level 1);
    /// the DC groups carry no DC stream and the header no upsampling fields.
    bool useDcFrame = false;
    /// A frame of a non-XYB image (JPEG recompression): `ycbcr` is the
    /// do_YCbCr field, `chromaMode` the YCbCrChromaSubsampling channel modes
    /// (X, Y, B), `skipDcSmoothing` flag kSkipAdaptiveDCSmoothing.
    bool xyb = true;
    bool ycbcr = false;
    std::array<uint32_t, 3> chromaMode{};
    bool skipDcSmoothing = false;
};

/// The DC frame of a VarDCT frame: a Modular frame in XYB at 1/8 size (frame
/// type kDCFrame, dc_level 1), no filters, not displayed. Not byte-aligned on exit.
void writeDcFrameHeader(BitWriter& w, uint32_t groupSizeShift = 1);

/// A single-pass, full-frame, last VarDCT frame in XYB. Not byte-aligned on exit.
void writeFrameHeader(BitWriter& w, const FrameParams& params, uint32_t numExtraChannels = 0, const FrameInfo& info = FrameInfo{});

/// The TOC (no permutation) for `sectionBytes`, one entry per section, or one
/// entry when the frame has a single group and pass. Byte-aligned on entry and exit.
void writeToc(BitWriter& w, std::span<const uint32_t> sectionBytes);

/// Frame geometry derived from the image size.
struct FrameDim {
    uint32_t xsize = 0, ysize = 0;
    uint32_t xsizeBlocks = 0, ysizeBlocks = 0;   ///< ceil(size / 8)
    uint32_t xsizePadded = 0, ysizePadded = 0;   ///< blocks * 8
    uint32_t xsizeGroups = 0, ysizeGroups = 0;   ///< AC groups
    uint32_t xsizeDcGroups = 0, ysizeDcGroups = 0;
    uint32_t groupDim = 256;                     ///< AC group size in pixels: 128 << group_size_shift
    [[nodiscard]] uint32_t numGroups() const { return xsizeGroups * ysizeGroups; }
    [[nodiscard]] uint32_t numDcGroups() const { return xsizeDcGroups * ysizeDcGroups; }
    [[nodiscard]] uint32_t numBlocks() const { return xsizeBlocks * ysizeBlocks; }
    /// One TOC entry: the whole frame is one section, bit-packed.
    [[nodiscard]] bool singleSection(uint32_t numPasses = 1) const { return numGroups() == 1 && numPasses == 1; }
    /// Sections in codestream order: DcGlobal, DcGroups, AcGlobal, AcGroups per pass.
    [[nodiscard]] uint32_t numSections(uint32_t numPasses = 1) const { return 2 + numDcGroups() + numGroups() * numPasses; }
};
/// `groupSizeShift` is the frame header field (1 for 256 px groups).
/// `maxHShift`/`maxVShift`: the largest chroma subsampling shifts (block counts round up to their multiples).
[[nodiscard]] FrameDim frameDim(uint32_t xsize, uint32_t ysize, uint32_t groupSizeShift = 1, uint32_t maxHShift = 0, uint32_t maxVShift = 0);

}   // namespace sjxl::codestream
