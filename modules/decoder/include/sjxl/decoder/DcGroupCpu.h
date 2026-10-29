// Copyright (c) 2026 jesus luque.
//
// The DC group sections of a VarDCT frame decoded on the CPU: one ANS stream
// per 2048 px group carries the DC values (modular, weighted predictor) and the
// AcMetadata, a chain of a few hundred thousand symbols whose contexts depend
// on the samples decoded before them. One GPU thread pays every latency of
// that chain in full (1 µs a sample); a CPU core walks it in a fraction of the
// time. This is the project's one exception to "no CPU arithmetic on coded
// data", taken by decision (the GPU path stays and is the reference).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "sjxl/codestream/Parse.h"
#include "sjxl/core/Result.h"

namespace sjxl::dec::cpu {

/// A DC group's rectangle in blocks and the chroma subsampling shifts, indexed
/// by physical channel (X, Y, B), as the GPU kernel sees them.
struct DcGroupGeom {
    uint32_t x0 = 0, y0 = 0, w = 0, h = 0;
    std::array<uint32_t, 3> hshift{}, vshift{};
};

/// What the GPU kernel reads instead of decoding: the DC channels Y, X, B (each
/// at its subsampled size), the chroma-from-luma tiles, the strategy/quant rows
/// and the sharpness, as the modular streams produced them.
struct DcGroupData {
    uint32_t             extraPrecision = 0;
    uint32_t             count = 0;   // AcMetadata entries (varblocks)
    std::vector<int32_t> words;       // Y, X, B planes; ytox, ytob (tiles); acs (count x 2); sharpness (w x h)
};

/// Decodes every DC group of `frame`. `globalTreeBit` is where DcGlobal's global
/// modular part starts (the GPU globals thread reports it). Returns false, with
/// nothing decoded, when a stream uses something this port does not do (LZ77,
/// transforms, extra channels): the caller then keeps the GPU path.
[[nodiscard]] Result<bool> decodeDcGroups(std::span<const uint8_t> codestream, const codestream::ParsedFrame& frame,
                                          size_t globalTreeBit, std::span<const DcGroupGeom> geoms, std::vector<DcGroupData>& out);

}   // namespace sjxl::dec::cpu
