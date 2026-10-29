// Copyright (c) 2026 jesus luque.
//
// The VarDCT and Modular frame decoder. The CPU parses headers and the TOC (codestream::parse);
// every section is decoded by a kernel thread, and dequantisation, inverse
// transforms, loop filters and the colour transform are kernels
// (shaders/sjxl/dec). Features not implemented fail with Unsupported.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "sjxl/codestream/Parse.h"
#include "sjxl/core/Result.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/ComputeKernel.h"
#include "sjxl/gpu/ShaderLibrary.h"
#include "sjxl/image/Image.h"

namespace sjxl::dec {

/// A reference-only frame kept for the patches of a later frame: its XYB planes.
struct RefFrame {
    std::optional<gpu::Buffer> planes;   // XYB: 3 planes of paddedWidth * paddedHeight; canvas: (3 + extra channels) planes of width * height
    uint32_t paddedWidth = 0, paddedHeight = 0, width = 0, height = 0;
    bool inXyb = true;                   // saved before the colour transform (patches) or after (blending canvases)
};

/// A displayed frame of the image (one for stills, several for animations).
struct Frame {
    image::Rgb8 image;
    uint32_t    duration = 0;   // animation ticks (tps numerator / denominator from the metadata)
};

struct Stats {
    uint32_t dcGlobalBits = 0;
    uint32_t usedAcs = 0;
};

class Decoder {
public:
    [[nodiscard]] static Result<std::unique_ptr<Decoder>> create(std::shared_ptr<gpu::ShaderLibrary> library);

    /// Headers only, no device work. Stops at a permuted TOC (Parsed::incomplete).
    [[nodiscard]] static Result<codestream::Parsed> info(std::span<const uint8_t> file) { return codestream::parse(file); }
    /// Headers of every frame, finishing permuted TOCs with the device.
    [[nodiscard]] Result<codestream::Parsed> parseAll(std::span<const uint8_t> file);

    /// The first displayed frame (the image, for a still).
    [[nodiscard]] Result<image::Rgb8> decode(std::span<const uint8_t> file);
    /// Every displayed frame, composited; one for a still image.
    [[nodiscard]] Result<std::vector<Frame>> decodeAll(std::span<const uint8_t> file);
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    Decoder() = default;
    /// One frame; a reference-only frame stores its planes in `refs`, a DC frame in
    /// `dcFrames[level]`, and both return an empty image.
    struct FrameIndices {
        uint32_t visible = 0, nonvisible = 0;   // dec_cache.h: seeds of the noise generators
    };
    struct FrameOutput {
        image::Rgb8 image;        // when emitted
        bool        emitted = false;
    };
    [[nodiscard]] Result<FrameOutput> run(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame, bool withLz77Windows,
                                          std::array<RefFrame, 4>& refs, std::array<RefFrame, 5>& dcFrames, FrameIndices indices);
    /// The permutation of a permuted TOC, decoded by a kernel: the order and the bit after it.
    struct TocPermutation {
        std::vector<uint32_t> order;
        size_t endBit = 0;
    };
    [[nodiscard]] Result<TocPermutation> decodeTocPermutation(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame);
    /// The bit after the ICC stream at parsed.iccBit, decoded by a kernel.
    [[nodiscard]] Result<size_t> decodeIccEnd(const codestream::Parsed& parsed);

    std::shared_ptr<gpu::ShaderLibrary> library_;
    gpu::Device*                        device_ = nullptr;
    gpu::ComputeKernel globals_, dcGroups_, acGroups_, tables_, smoothDc_, computeSigma_, dequantCoeffs_, llfFromDc_, idctRows_, idctCols_, transformSpecial_,
        gaborish_, epf_, output_, modularGlobals_, modularDcGroups_, modularAcGroups_, modularFinalize_, modularToPlanes_, ecToPlanes_, bucketPatches_, applyPatches_, tocPermutation_, dcFromFrame_, upsample_, noiseRandom_, noiseConvolve_, noiseAdd_, chromaUpsample_, icc_, toOutputPlanes_, blendCanvas_,
        splineSegments_, splineOffsets_, splineRowCounts_, splineRowStarts_, splineScatter_, splineDraw_, buildNaturalOrders_, finishOrders_, hostDcBlocks_, hostDcVarblocks_,
        hostDcTiles_, hostModCopy_, hostModRct_, singleSection_;
    std::optional<gpu::Buffer> naturalOrders_;   // the natural coefficient orders, built by the first VarDCT frame
    Stats stats_;
};

}   // namespace sjxl::dec
