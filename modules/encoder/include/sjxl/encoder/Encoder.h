// Copyright (c) 2026 jesus luque.
//
// The VarDCT encoder. Every numeric stage is a kernel (shaders/sjxl/enc); the
// CPU uploads the pixels, writes the headers and the TOC from the section
// sizes the device reports, and appends the body the device emitted.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

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
    /// ANS); the distance and the VarDCT options are ignored.
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
};

/// What the last encode measured, for tests and the CLI.
struct Stats {
    uint32_t              headBytes = 0;
    uint32_t              bodyBytes = 0;
    std::vector<uint32_t> sectionBytes;
    uint32_t              numAcClusters = 0; ///< AC histogram clusters the device chose
    uint32_t              globalScale = 0;   ///< the quantizer
    uint32_t              quantDc = 0;
    bool                  gaborish = false;
    uint32_t              epfIters = 0;
};

class Encoder {
public:
    [[nodiscard]] static Result<std::unique_ptr<Encoder>> create(std::shared_ptr<gpu::ShaderLibrary> library);

    [[nodiscard]] Result<std::vector<uint8_t>> encode(const image::Rgb8& image, const Options& options);
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    Encoder() = default;
    [[nodiscard]] Result<std::vector<uint8_t>> encodeLossless(const image::Rgb8& image, const Options& options);

    std::shared_ptr<gpu::ShaderLibrary> library_;
    gpu::Device*                        device_ = nullptr;
    gpu::ComputeKernel                  rgbToXyb_, aqDiff_, aqMap_, aqRawField_;
    gpu::ComputeKernel                  gaborishInverse_, dct_, cflTile_, quantize_;
    gpu::ComputeKernel                  acItems_, dcItems_, metaItems_, clusterContexts_, acSections_;
    gpu::ComputeKernel                  rctYCoCg_, modHist_, modGroupSections_, modularGlobalBlob_;
    gpu::ComputeKernel                  dcGroupBlobs_, globalBlobs_, layout_, blobEmit_;
    gpu::PrefixSum                      prefixSum_;
    gpu::Buffer                         dequant_;   // 3 * 64 floats
    gpu::Buffer                         order_;     // 64 uints
    gpu::Buffer                         dummy_;     // bound where a pass has no body yet
    gpu::Buffer                         dummyInt_;  // an int buffer for the same purpose
    Stats                               stats_;
};

}   // namespace sjxl::enc
