// Copyright (c) 2026 jesus luque.
//
// The encoder's serial entropy chains on the CPU, by decision (docs/decisions.md
// "M5: the encoder's serial chains"): the DC tokens through the weighted
// predictor (one chain per DC group) and the ANS streams of every section (one
// chain per AC group and two per DC group), which a lone GPU thread ran at
// 1-1.5 µs a token. Tokens, histograms, clustering, the ANS tables and the
// header blobs stay on the GPU; this code reads them back and writes the frame
// body bit for bit as the kernels do (the [enc-cpu] test compares the files).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "sjxl/core/Result.h"

namespace sjxl::enc::cpu {

constexpr uint32_t kNoToken = 0xFFFFFFFFu;

struct Rect {
    uint32_t x = 0, y = 0, w = 0, h = 0;   // blocks
};

struct FrameGeom {
    uint32_t                nblocks = 0, blocksX = 0, blocksY = 0;
    uint32_t                numGroups = 0, numDcGroups = 0, numPasses = 1;
    bool                    useDcFrame = false;
    bool                    packed = false;   // one TOC entry: the sections follow each other bit-packed
    std::array<uint32_t, 3> hshift{}, vshift{};   // per physical channel X, Y, B
    std::span<const Rect>   acRects;
    std::span<const Rect>   dcRects;
};

/// dcTokenize: the quantised DC of every DC group through libjxl's kWPFixedDC
/// tree (the weighted predictor, one chain per group, groups in parallel).
/// Writes dcTokens[c * nblocks + block] = ctx << 18 | value at the coded
/// positions (a subsampled channel codes one block per 2^shift) and kNoToken
/// elsewhere. `dcTree` is the tree in the kernels' 8-word node layout.
void dcTokenize(const FrameGeom& geom, std::span<const int32_t> dcQuant, std::span<const uint32_t> dcTree, std::vector<uint32_t>& dcTokens);

struct SectionsInput {
    FrameGeom                  geom;
    std::span<const uint32_t>  acs;            // nblocks: strategy << 1 | first
    std::span<const uint32_t>  nonZeros;       // [pass][c][block]
    std::span<const uint32_t>  acTokens;       // [pass][c][block][64]
    std::span<const uint32_t>  numTokens;      // [pass][c][block]
    std::span<const uint32_t>  nzCtx;          // [pass][c][block]
    std::span<const uint32_t>  dcTokens;       // [c][block]
    std::span<const uint32_t>  metaTokens;     // numDcGroups * kMetaItemsPerGroup
    std::span<const uint32_t>  ctxMap;         // AC sets (kNumAcContexts each), then the DC groups' sets (kSetContexts each)
    std::span<const uint32_t>  ansInfo;        // code region r: freq | start << 16 per symbol at r * 256
    std::span<const uint32_t>  ansReverse;     // code region r: slots at r * 4096
    std::span<const uint32_t>  blobs;          // region r at r * kBlobStrideWords
    std::span<const uint32_t>  blobBits;
    uint32_t                   qtBlobRegion = 0, acPostRegion = 0;
};

struct SectionsOutput {
    std::vector<uint32_t> sectionBytes;   // per TOC entry (codestream order)
    std::vector<uint32_t> acOnlyBits;     // per (pass, group): the AC stream's bits (debug)
    std::vector<uint32_t> dcPartStart;    // per DC group: pre, DC stream, mid, meta stream bit offsets (debug)
    std::vector<uint8_t>  body;
};

/// Every section as the kernels would place it: DcGlobal, the DC groups (pre
/// blob, DC stream, mid blob, AcMetadata stream), AcGlobal, the AC groups per
/// pass. The sections are coded in parallel and concatenated byte-aligned.
[[nodiscard]] Result<SectionsOutput> emitSections(const SectionsInput& in);

}   // namespace sjxl::enc::cpu
