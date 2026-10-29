// Copyright (c) 2026 jesus luque.
//
// Image comparison on the device: a kernel measures, the caller reads a few
// numbers. What tests and `sjxl compare` use.
#pragma once

#include <cstdint>
#include <memory>

#include "sjxl/core/Result.h"
#include "sjxl/gpu/ShaderLibrary.h"
#include "sjxl/image/Image.h"

namespace sjxl::metrics {

struct Metrics {
    double   psnr = 0;          ///< all channels, on the images' sample scale (255 or 65535); infinity when identical
    uint32_t maxAbsDiff = 0;    ///< in sample units
    uint32_t maxAtX = 0, maxAtY = 0, maxAtChannel = 0;   ///< where the largest difference is
    uint64_t sumSquares = 0;
};

/// Both images must have the same size.
[[nodiscard]] Result<Metrics> compare(gpu::ShaderLibrary& library, const image::Rgb8& a, const image::Rgb8& b);

}   // namespace sjxl::metrics
