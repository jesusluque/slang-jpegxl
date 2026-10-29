// Copyright (c) 2026 jesus luque.
//
// The codestream syntax the CPU writes: signature, SizeHeader, ImageMetadata,
// FrameHeader, TOC, and the DcGlobal quantizer fields. Everything else in a
// frame is emitted by kernels (shaders/sjxl/enc).
#pragma once

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
void writeCodestreamHeaders(BitWriter& w, uint32_t xsize, uint32_t ysize);
/// Signature, SizeHeader and the ImageMetadata of an 8- or 16-bit sRGB image
/// that is not XYB encoded (a lossless Modular frame), with an unassociated
/// alpha channel of the same depth when `alpha`. Byte-aligned on exit.
void writeCodestreamHeadersRgb(BitWriter& w, uint32_t xsize, uint32_t ysize, uint32_t bitDepth, bool alpha);
/// A single-pass, full-frame, last Modular frame in RGB (no colour transform,
/// group_size_shift 1, LoopFilter off) with `numExtraChannels` unsampled extra
/// channels. Not byte-aligned on exit.
void writeModularFrameHeader(BitWriter& w, uint32_t numExtraChannels);

struct FrameParams {
    bool     gaborish = true;      ///< LoopFilter.gab
    uint32_t epfIters = 2;         ///< LoopFilter.epf_iters (0..3)
    uint32_t xQmScale = 2;         ///< neutral chroma quant-matrix scale
    uint32_t bQmScale = 2;
};

/// A single-pass, full-frame, last VarDCT frame in XYB. Not byte-aligned on exit.
void writeFrameHeader(BitWriter& w, const FrameParams& params);

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
    [[nodiscard]] bool singleSection() const { return numGroups() == 1; }
    /// Sections in codestream order: DcGlobal, DcGroups, AcGlobal, AcGroups.
    [[nodiscard]] uint32_t numSections() const { return 2 + numDcGroups() + numGroups(); }
};
/// `groupSizeShift` is the frame header field (1 for 256 px groups).
/// `maxHShift`/`maxVShift`: the largest chroma subsampling shifts (block counts round up to their multiples).
[[nodiscard]] FrameDim frameDim(uint32_t xsize, uint32_t ysize, uint32_t groupSizeShift = 1, uint32_t maxHShift = 0, uint32_t maxVShift = 0);

}   // namespace sjxl::codestream
