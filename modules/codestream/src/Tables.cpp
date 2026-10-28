// Copyright (c) 2026 jesus luque.
#include "sjxl/codestream/Tables.h"

#include <algorithm>
#include <cmath>

namespace sjxl::codestream {
namespace {

// libjxl quantizer.h / enc_adaptive_quantization.cc constants.
constexpr float kAcQuant = 0.725f;
constexpr float kDcQuant = 1.095924047623553f;
constexpr float kDcQuantPow = 0.83f;
constexpr float kDcMul = 0.3f;
constexpr float kQuantFieldTarget = 5.0f;
constexpr int kGlobalScaleDenom = 1 << 16;
constexpr int kGlobalScaleNumerator = 4096;
constexpr int kQuantMax = 256;

// libjxl quant_weights.cc Mult: how one distance band follows the previous.
float mult(float v) { return v > 0.0f ? 1.0f + v : 1.0f / (1.0f - v); }

}   // namespace

QuantParams quantParamsForDistance(float distance) {
    const float quantAc = kAcQuant / distance;
    const float btDc = std::max(0.5f * distance,
                                std::min(distance, kDcMul * std::pow((1.0f / kDcMul) * distance, kDcQuantPow)));
    const float quantDcFloat = std::min(kDcQuant / btDc, 50.0f);

    // ComputeGlobalScaleAndQuant with a uniform field: median == quantAc, mad == 0.
    float scale = static_cast<float>(kGlobalScaleDenom) * quantAc / kQuantFieldTarget;
    scale = std::clamp(scale, 1.0f, static_cast<float>(1 << 15));
    int globalScale = static_cast<int>(scale);
    const int scaledQuantDc = static_cast<int>(quantDcFloat * kGlobalScaleNumerator * 1.6f);
    if (globalScale > scaledQuantDc) {
        globalScale = scaledQuantDc <= 0 ? 1 : scaledQuantDc;
    }
    const float invGlobalScale = static_cast<float>(kGlobalScaleDenom) / static_cast<float>(globalScale);
    float fval = quantDcFloat * invGlobalScale + 0.5f;
    fval = std::min(static_cast<float>(1 << 16), fval);
    const int quantDc = static_cast<int>(fval);
    int q = static_cast<int>(std::lround(quantAc * invGlobalScale));
    q = std::clamp(q, 1, kQuantMax);

    QuantParams p;
    p.globalScale = static_cast<uint32_t>(globalScale);
    p.quantDc = static_cast<uint32_t>(quantDc);
    p.quantField = static_cast<uint32_t>(q);
    p.globalScaleFloat = static_cast<float>(globalScale) * (1.0f / kGlobalScaleDenom);
    p.invGlobalScale = invGlobalScale;
    return p;
}

std::array<float, 3 * 64> dct8DequantWeights() {
    // DequantMatricesLibraryDef::DCT(): six distance bands per channel.
    constexpr float kBands[3][6] = {
        {3150.0f, 0.0f, -0.4f, -0.4f, -0.4f, -2.0f},
        {560.0f, 0.0f, -0.3f, -0.3f, -0.3f, -0.3f},
        {512.0f, -2.0f, -1.0f, 0.0f, -1.0f, -2.0f},
    };
    constexpr size_t kNumBands = 6;
    constexpr float kSqrt2 = 1.41421356237f;
    std::array<float, 3 * 64> out{};
    for (size_t c = 0; c < 3; ++c) {
        float bands[kNumBands] = {kBands[c][0]};
        for (size_t i = 1; i < kNumBands; ++i) {
            bands[i] = bands[i - 1] * mult(kBands[c][i]);
        }
        const float scale = (kNumBands - 1) / (kSqrt2 + 1e-6f);
        const float rcp = scale / 7.0f;
        for (uint32_t y = 0; y < 8; ++y) {
            for (uint32_t x = 0; x < 8; ++x) {
                const float dx = x * rcp;
                const float dy = y * rcp;
                const float pos = std::sqrt(dx * dx + dy * dy);
                const int idx = static_cast<int>(pos);
                const float frac = pos - static_cast<float>(idx);
                const float a = bands[idx];
                const float b = bands[std::min<size_t>(idx + 1, kNumBands - 1)];
                // Interpolate: a * (b/a)^frac. The stored matrix is its reciprocal.
                const float weight = a * std::pow(b / a, frac);
                out[c * 64 + y * 8 + x] = 1.0f / weight;
            }
        }
    }
    return out;
}

std::array<uint32_t, 64> dct8NaturalOrder() {
    // ac_strategy.cc CoeffOrderAndLut with cx = cy = 1: order[val] = y*8 + x.
    std::array<uint32_t, 64> order{};
    size_t cur = 1;
    for (size_t i = 0; i < 8; i++) {
        for (size_t j = 0; j <= i; j++) {
            size_t x = j;
            size_t y = i - j;
            if (i % 2) std::swap(x, y);
            const size_t val = (x == 0 && y == 0) ? 0 : cur++;
            order[val] = static_cast<uint32_t>(y * 8 + x);
        }
    }
    for (size_t ip = 7; ip > 0; ip--) {
        const size_t i = ip - 1;
        for (size_t j = 0; j <= i; j++) {
            size_t x = 7 - (i - j);
            size_t y = 7 - j;
            if (i % 2) std::swap(x, y);
            order[cur++] = static_cast<uint32_t>(y * 8 + x);
        }
    }
    return order;
}

}   // namespace sjxl::codestream
