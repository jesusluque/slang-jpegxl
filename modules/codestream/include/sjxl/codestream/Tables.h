// Copyright (c) 2026 jesus luque.
//
// The numeric constants of the format that kernels need as tables: libjxl's
// formulas evaluated once on the CPU at start-up (they are constants of the
// specification, not image data) and uploaded. tests/codestream checks them
// against the values cujpegxl extracted from libjxl itself.
#pragma once

#include <array>
#include <cstdint>

namespace sjxl::codestream {

/// Quantizer state for a frame, from a Butteraugli distance exactly as
/// libjxl's encoder derives it for a uniform quant field.
struct QuantParams {
    uint32_t globalScale = 1;      ///< serialized; kGlobalScaleDenom = 2^16
    uint32_t quantDc = 1;          ///< serialized
    uint32_t quantField = 1;       ///< the uniform per-block AC quant (1..256)
    float    globalScaleFloat = 0; ///< globalScale / 2^16
    float    invGlobalScale = 0;   ///< 2^16 / globalScale
};
[[nodiscard]] QuantParams quantParamsForDistance(float distance);
/// Quantizer::ComputeGlobalScaleAndQuant(quantDc, quantAc, 0) and the raw field value of `quantAc`.
[[nodiscard]] QuantParams quantParamsFor(float quantAc, float quantDc);

/// libjxl InitialQuantDC: the DC quantiser (before the global scale) for a distance.
[[nodiscard]] float initialQuantDc(float distance);
/// libjxl kAcQuant / distance: the unit of the adaptive quant field.
[[nodiscard]] float initialQuantAc(float distance);

/// libjxl kInvDCQuant, channels X, Y, B.
constexpr std::array<float, 3> kInvDcQuant = {4096.0f, 512.0f, 256.0f};

/// The default DCT8 dequantization weights (libjxl DequantMatrices::Matrix),
/// [channel X,Y,B][k], k = row*8 + col in the coefficient block's memory
/// layout (symmetric for DCT8, so row/column naming does not matter).
[[nodiscard]] std::array<float, 3 * 64> dct8DequantWeights();

/// The natural (zig-zag) coefficient order of an 8x8 DCT block: order[k] is
/// the block index coded k-th. order[0] is the DC slot, never coded.
[[nodiscard]] std::array<uint32_t, 64> dct8NaturalOrder();

}   // namespace sjxl::codestream
