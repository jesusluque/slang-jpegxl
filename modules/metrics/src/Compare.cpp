// Copyright (c) 2026 jesus luque.
#include "sjxl/metrics/Compare.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/ComputeKernel.h"

namespace sjxl::metrics {

Result<Metrics> compare(gpu::ShaderLibrary& library, const image::Rgb8& a, const image::Rgb8& b) {
    if (a.channels != b.channels) {
        return Error::make(ErrorCode::InvalidArgument, "channel counts differ: {} vs {}", a.channels, b.channels);
    }
    if (a.bitDepth != b.bitDepth) {
        return Error::make(ErrorCode::InvalidArgument, "bit depths differ: {} vs {}", a.bitDepth, b.bitDepth);
    }
    if (a.width != b.width || a.height != b.height) {
        return Error::make(ErrorCode::InvalidArgument, "sizes differ: {}x{} vs {}x{}", a.width, a.height, b.width,
                           b.height);
    }
    gpu::Device& dev = library.device();
    SJXL_ASSIGN(auto kernel, gpu::ComputeKernel::create(library, "sjxl/metrics/compare", "compareSamples"));
    SJXL_ASSIGN(auto locate, gpu::ComputeKernel::create(library, "sjxl/metrics/compare", "locateMax"));
    const auto upload = [&](const image::Rgb8& img, const char* label) -> Result<gpu::Buffer> {
        std::vector<uint32_t> words((img.pixels.size() + 3) / 4, 0);
        std::memcpy(words.data(), img.pixels.data(), img.pixels.size());
        return gpu::Buffer::fromSpan<uint32_t>(dev, words, label);
    };
    SJXL_ASSIGN(auto ba, upload(a, "compare.a"));
    SJXL_ASSIGN(auto bb, upload(b, "compare.b"));
    const uint32_t samples = static_cast<uint32_t>(a.pixels.size() / a.bytesPerSample());
    const uint32_t groups = (samples + 255) / 256;
    SJXL_ASSIGN(auto sums, gpu::Buffer::ofType<uint32_t>(dev, groups, "compare.sums"));
    const std::array<uint32_t, 2> init{0u, 0xFFFFFFFFu};
    SJXL_ASSIGN(auto maxDiff, gpu::Buffer::fromSpan<uint32_t>(dev, init, "compare.max"));
    gpu::CommandBatch batch(dev);
    const auto bind = [&](rhi::ShaderCursor c) {
        c["a"].setBinding(ba.rhi());
        c["b"].setBinding(bb.rhi());
        c["groupSums"].setBinding(sums.rhi());
        c["maxDiff"].setBinding(maxDiff.rhi());
        c["params"]["samples"].setData(samples);
        c["params"]["bytesPerSample"].setData(a.bytesPerSample());
    };
    kernel.dispatch(batch, {samples, 1, 1}, bind);
    locate.dispatch(batch, {samples, 1, 1}, bind);
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const auto sumsHost, sums.readAll<uint32_t>(dev));
    SJXL_ASSIGN(const auto maxHost, maxDiff.readAll<uint32_t>(dev));
    Metrics m;
    for (uint32_t s : sumsHost) m.sumSquares += s;
    m.maxAbsDiff = maxHost[0];
    const uint32_t at = maxHost[1] == 0xFFFFFFFFu ? 0 : maxHost[1];
    m.maxAtChannel = at % a.channels;
    m.maxAtX = (at / a.channels) % a.width;
    m.maxAtY = (at / a.channels) / a.width;
    const double peak = a.bitDepth == 16 ? 65535.0 : 255.0;
    if (m.sumSquares == 0) {
        m.psnr = std::numeric_limits<double>::infinity();
    } else {
        const double mse = static_cast<double>(m.sumSquares) / static_cast<double>(samples);
        m.psnr = 10.0 * std::log10(peak * peak / mse);
    }
    return m;
}

}   // namespace sjxl::metrics
