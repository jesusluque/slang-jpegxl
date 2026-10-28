// Copyright (c) 2026 jesus luque.
//
// The VarDCT and Modular frame decoder. The CPU parses headers and the TOC (codestream::parse);
// every section is decoded by a kernel thread, and dequantisation, inverse
// transforms, loop filters and the colour transform are kernels
// (shaders/sjxl/dec). Features not implemented fail with Unsupported.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "sjxl/codestream/Parse.h"
#include "sjxl/core/Result.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/ComputeKernel.h"
#include "sjxl/gpu/ShaderLibrary.h"
#include "sjxl/image/Image.h"

namespace sjxl::dec {

struct Stats {
    uint32_t dcGlobalBits = 0;
    uint32_t usedAcs = 0;
};

class Decoder {
public:
    [[nodiscard]] static Result<std::unique_ptr<Decoder>> create(std::shared_ptr<gpu::ShaderLibrary> library);

    /// Headers only, no device work.
    [[nodiscard]] static Result<codestream::Parsed> info(std::span<const uint8_t> file) { return codestream::parse(file); }

    [[nodiscard]] Result<image::Rgb8> decode(std::span<const uint8_t> file);
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    Decoder() = default;
    [[nodiscard]] Result<image::Rgb8> run(const codestream::Parsed& parsed, bool withLz77Windows);

    std::shared_ptr<gpu::ShaderLibrary> library_;
    gpu::Device*                        device_ = nullptr;
    gpu::ComputeKernel globals_, dcGroups_, acGroups_, tables_, smoothDc_, computeSigma_, dequantCoeffs_, llfFromDc_, idctRows_, idctCols_, transformSpecial_,
        gaborish_, epf_, output_, modularGlobals_, modularDcGroups_, modularAcGroups_, modularFinalize_, modularToPlanes_;
    Stats stats_;
};

}   // namespace sjxl::dec
