// Copyright (c) 2026 jesus luque.
//
// The VarDCT encoder. Every numeric stage is a kernel (shaders/sjxl/enc); the
// CPU uploads the pixels, writes the headers and the TOC from the section
// sizes the device reports, and appends the body the device emitted.
#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

#include "sjxl/codestream/Headers.h"
#include "sjxl/image/Jpeg.h"
#include "sjxl/core/Result.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/ComputeKernel.h"
#include "sjxl/gpu/Device.h"
#include "sjxl/gpu/ShaderLibrary.h"
#include "sjxl/gpu/algo/PrefixSum.h"
#include "sjxl/image/Image.h"

namespace sjxl::enc {

struct Options {
    /// Lossless: a Modular frame (YCoCg RCT, Gradient predictor, fixed MA tree,
    /// ANS) of the image's channels (RGB or RGBA) at its bit depth (8 or 16);
    /// the distance and the VarDCT options are ignored.
    bool lossless = false;
    /// Butteraugli distance: smaller is higher quality (1.0 ~ libjxl quality 90).
    float distance = 1.0f;
    /// Per-block quant field from libjxl's masking model; off: one value per frame.
    bool adaptiveQuant = true;
    /// Chroma-from-luma per 64 px tile; off: X and B against Y with the default factors.
    bool chromaFromLuma = true;
    /// Gaborish + EPF as libjxl chooses them for the distance, with the inverse
    /// Gaborish applied before the DCT; off: LoopFilter disabled.
    bool loopFilter = true;
    /// Most AC histogram clusters (1..32); 1 codes every context with one histogram.
    uint32_t acClusters = 32;
    /// Transform selection (DCT8 to 32x32 per 32x32 superblock); off: DCT8 everywhere.
    bool acStrategy = true;
    /// The selection's distortion weight: bits per squared quant step of rounding error.
    float acsLambda = 128.0f;
    /// The 8x8 special transforms (IDENTITY, DCT2x2, DCT4x4, DCT4x8, DCT8x4, AFV) compete with DCT8.
    bool acsSpecials = true;
    /// Progressive AC (lossy): 0 one pass; 1 libjxl's qprogressive (every
    /// coefficient at half precision, then the rest: 2 passes); 2 libjxl's
    /// progressive (the 2x2 lowest frequencies, then 3x3, then all: 3 passes).
    uint32_t progressive = 0;
    /// A DC frame first (libjxl's progressive_dc 1): the DC image as a Modular
    /// XYB frame at 1/8 size, the VarDCT frame taking its DC from it (lossy
    /// RGB only: libjxl does not combine DC frames with extra channels either).
    bool progressiveDc = false;
    /// Modular frames: learn the MA tree from the image (libjxl's ComputeBestTree
    /// at its hare tier) instead of the fixed WP / gradient trees per channel.
    bool learnedTree = true;
    /// Animations: a frame codes only the rectangle that changed since the frame
    /// before it (a partial frame blended over the saved canvas); identical frames
    /// extend the previous frame's duration. Off: every frame a full canvas.
    bool cropFrames = true;
};

/// What the last encode measured, for tests and the CLI.
struct Stats {
    uint32_t              frames = 0;      ///< animations: frames emitted (identical frames merge)
    uint32_t              headBytes = 0;
    uint32_t              bodyBytes = 0;
    std::vector<uint32_t> sectionBytes;
    uint32_t              numAcClusters = 0; ///< AC histogram clusters the device chose
    uint32_t              globalScale = 0;   ///< the quantizer
    uint32_t              quantDc = 0;
    bool                  gaborish = false;
    uint32_t              epfIters = 0;
    uint32_t              alphaPredictor = 0;   ///< lossy with alpha: 0 weighted, 1 gradient
    uint32_t              dcFrameBytes = 0;     ///< progressive DC: the DC frame (header, TOC, body)
    uint32_t              palette = 0;          ///< lossless: colours of the palette over every channel (0: none)
    uint32_t              compacted = 0;        ///< lossless: channels given a compaction palette
};

class Encoder {
public:
    [[nodiscard]] static Result<std::unique_ptr<Encoder>> create(std::shared_ptr<gpu::ShaderLibrary> library);

    [[nodiscard]] Result<std::vector<uint8_t>> encode(const image::Rgb8& image, const Options& options);
    /// An animation: every frame a full canvas replacing the previous one, shown
    /// for `durations[i]` ticks at `ticksPerSecond`; all frames share the first
    /// one's size, channels and depth. `numLoops` 0 repeats forever.
    [[nodiscard]] Result<std::vector<uint8_t>> encodeAnimation(const std::vector<image::Rgb8>& frames, const std::vector<uint32_t>& durations,
                                                               uint32_t ticksPerSecond, uint32_t numLoops, const Options& options);
    /// JPEG recompression: the JPEG's quantised DCT coefficients as a VarDCT
    /// frame in YCbCr (or RGB) with its quantisation tables, decoded by a kernel
    /// (baseline Huffman scans). Lossless in the coefficients, as cjxl does.
    [[nodiscard]] Result<std::vector<uint8_t>> encodeJpeg(const image::JpegFile& jpeg, const Options& options);
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    Encoder() = default;
    /// One frame's bytes (frame header, TOC, body), byte-aligned, without the codestream header.
    /// `rgbDevice`: the image's bytes already on the device (a cropped frame); `image` then
    /// carries the size, channels and depth only.
    [[nodiscard]] Result<std::vector<uint8_t>> encodeFrame(const image::Rgb8& image, const Options& options, const codestream::FrameInfo& info,
                                                           const image::JpegFile* jpeg = nullptr, const gpu::Buffer* rgbDevice = nullptr);
    [[nodiscard]] Result<std::vector<uint8_t>> encodeLossless(const image::Rgb8& image, const Options& options, const codestream::FrameInfo& info,
                                                              const gpu::Buffer* rgbDevice = nullptr);
    /// A Modular frame of `planes` (channels of width x height int32 samples,
    /// already on the device): the frame header, TOC and body. `rct` writes the
    /// YCoCg RCT over channels 0..2 (the planes must hold its output); a DC frame
    /// (`dcQuant` set) is an XYB frame at 1/8 size scaled by those factors.
    struct ModularChannel {
        uint32_t width = 0, height = 0, offset = 0;   ///< samples in the planes buffer
        bool     grouped = false;                     ///< coded in the group streams (else the global stream)
    };
    struct ModularTransform {
        uint32_t id = 1, beginC = 0, numC = 1, nbColors = 0;   ///< id 1: palette (nb_deltas 0, predictor Zero)
    };
    struct ModularSpec {
        uint32_t width = 0, height = 0, bitDepth = 8;   ///< the image (the grouped channels' size)
        uint32_t groupSizeShift = 1;                    ///< groups of 128 << shift px
        std::vector<ModularChannel>  channels;          ///< meta channels first
        std::vector<ModularTransform> transforms;       ///< the global transforms (palettes), in order
        uint32_t nbMeta = 0;
        bool     rct = false;          ///< an RCT per group over channels nbMeta..+2 (groupRct)
        bool     dcFrame = false;
        std::array<float, 3> dcQuant{};
        [[nodiscard]] uint32_t totalSamples() const {
            uint32_t n = 0;
            for (const ModularChannel& c : channels) n += c.width * c.height;
            return n;
        }
    };
    [[nodiscard]] Result<std::vector<uint8_t>> encodeModularFrame(const ModularSpec& spec, const gpu::Buffer& planes, const Options& options,
                                                                  const codestream::FrameInfo& info, const gpu::Buffer* groupRct = nullptr);

    std::shared_ptr<gpu::ShaderLibrary> library_;
    gpu::Device*                        device_ = nullptr;
    gpu::ComputeKernel                  rgbToXyb_, aqDiff_, aqMap_, aqRawField_;
    gpu::ComputeKernel                  gaborishInverse_, cflTile_, quantize_;
    gpu::ComputeKernel                  tables_, buildOrders_, selectStrategies_, largeQf_, largeRows_, largeCols_, mergeLarge_, adjustQuantField_, dctRows_, dctCols_, dcFromLlf_, specialTransform_, passNonZeroCount_, acTokenize_;
    gpu::ComputeKernel                  dcTokenize_, metaTokenize_, dcGroupSections_, clusterContexts_, accumulateClusters_, acSections_;
    gpu::ComputeKernel                  propStats_, propThresholds_, treeHistograms_, treeSplit_, treeAssign_, modSamples_, modTokenizeTree_;
    gpu::ComputeKernel                  unpackPlanes_, rctCost_, rctApply_, dcFramePlanes_, modTokenize_;
    gpu::ComputeKernel                  paletteCount_, paletteBuild_, paletteIndex_, paletteVerify_, compactStats_, compactBuild_, compactApply_;
    gpu::ComputeKernel                  rleScan_, rleHistograms_, uintConfigHist_, uintConfigChoose_;
    gpu::ComputeKernel                  jpegDecodeScan_, jpegToCoeffs_, jpegBlockCtx_, quantTableBlob_, changeBounds_, cropBytes_;
    gpu::ComputeKernel                  extractAlpha_, channelEntropy_, remapTokens_, modGroupSections_, modularGlobalBlob_;
    gpu::ComputeKernel                  dcGroupPre_, dcGroupMid_, globalBlobs_, layout_, blobEmit_;
    gpu::PrefixSum                      prefixSum_;
    gpu::Buffer                         dequantAll_;   // every quant table kind, 3 channels each (kTableBase layout)
    gpu::Buffer                         orders_;       // the natural coefficient order of every bucket
    gpu::Buffer                         dummy_;     // bound where a pass has no body yet
    gpu::Buffer                         dummyInt_;  // an int buffer for the same purpose
    Stats                               stats_;
};

}   // namespace sjxl::enc
