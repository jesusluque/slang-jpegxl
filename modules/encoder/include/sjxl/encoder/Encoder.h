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
    /// Butteraugli distance: smaller is higher quality (1.0 ~ libjxl quality 90).
    float distance = 1.0f;
};

/// What the last encode measured, for tests and the CLI.
struct Stats {
    uint32_t              headBytes = 0;
    uint32_t              bodyBytes = 0;
    std::vector<uint32_t> sectionBytes;
};

class Encoder {
public:
    [[nodiscard]] static Result<std::unique_ptr<Encoder>> create(std::shared_ptr<gpu::ShaderLibrary> library);

    [[nodiscard]] Result<std::vector<uint8_t>> encode(const image::Rgb8& image, const Options& options);
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    Encoder() = default;

    std::shared_ptr<gpu::ShaderLibrary> library_;
    gpu::Device*                        device_ = nullptr;
    gpu::ComputeKernel                  rgbToXyb_, dctQuant_, acItems_, dcItems_, metaItems_;
    gpu::ComputeKernel                  dcGroupBlobs_, globalBlobs_, layout_, blobEmit_;
    gpu::PrefixSum                      prefixSum_;
    gpu::Buffer                         dequant_;   // 3 * 64 floats
    gpu::Buffer                         order_;     // 64 uints
    gpu::Buffer                         dummy_;     // bound where a pass has no body yet
    Stats                               stats_;
};

}   // namespace sjxl::enc
