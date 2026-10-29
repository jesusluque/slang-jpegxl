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

// --- the lossless (Modular) encoder's chains ---------------------------------------------

struct ModChannel {
    uint32_t w = 0, h = 0, offset = 0;   // size and offset in the planes
    bool     grouped = false;             // coded in the group streams (else in the global stream, by group 0)
};

struct ModGeom {
    uint32_t                     width = 0, height = 0, groupDim = 0, groupsX = 0, numGroups = 0;
    uint32_t                     nbMeta = 0;        // meta channels (group streams number their channels from the first non-meta one)
    uint32_t                     streamBase = 0;    // stream id of group 0's stream (property 1)
    uint32_t                     totalSamples = 0;
    std::span<const ModChannel>  channels;
};

struct ModSamplesParams {
    float                      fraction = 0.25f;
    uint32_t                   capacity = 0;
    std::array<uint32_t, 4>    propIds{};
    std::span<const int32_t>   thresholds;   // [16][64] per property, count at 16 * 64 + property
};

/// modSamples: the tree learner's samples (GatherTreeData) over every group, packed
/// as the kernel's uint2 (x | y << 32), in group order, at most `capacity` of them.
void modSamples(const ModGeom& geom, std::span<const int32_t> planes, const ModSamplesParams& params, std::vector<uint64_t>& samples);

/// modTokenizeTree: every sample's token under the learned tree (the kernels' 8-word
/// nodes, split values as property values); tokens[at] = ctx << 18 | value.
void modTokenizeTree(const ModGeom& geom, std::span<const int32_t> planes, std::span<const uint32_t> treeNodes, std::vector<uint32_t>& tokens);

/// modTokenize: one predictor pass under a fixed tree (the WP-property or gradient
/// tree) into tokens[pass * totalSamples + at].
void modTokenizeFixed(const ModGeom& geom, std::span<const int32_t> planes, std::span<const uint32_t> tree, uint32_t pass, std::vector<uint32_t>& tokens);

struct RleResult {
    std::vector<uint32_t> streamTokens;   // the compact lists (totalSamples slots; stream s at streamStart[s], streamLen[s] long)
    std::vector<uint32_t> streamLen;
    uint32_t              decrease16 = 0;   // estimated bit saving x 16 (the kernel's rleStats[0])
    uint32_t              total = 0;        // tokens over every stream (rleStats[1])
};

/// lz77.slang rleScan over every stream (the groups', then the global one): mode 1
/// run-length coded with libjxl's per-run estimate from `ctxHist` (numContexts
/// contexts of 128 bins), mode 0 a plain copy.
void rleScan(const ModGeom& geom, std::span<const uint32_t> tokens, std::span<const uint32_t> ctxHist, uint32_t numContexts, uint32_t mode,
             std::span<const uint32_t> streamStart, RleResult& out);

struct ModSectionsInput {
    ModGeom                    geom;
    uint32_t                   numDcGroups = 0;
    bool                       fitsGroup = false;   // one section: the global blob holds the whole stream
    bool                       lz77 = false;
    uint32_t                   rctBeginC = 0;
    std::span<const uint32_t>  groupRct;        // per group: its RCT type (empty: none)
    std::span<const uint32_t>  streamTokens, streamStart, streamLen;
    std::span<const uint32_t>  ctxMap;          // leaf -> cluster
    std::span<const uint32_t>  ansInfo, ansReverse;
    std::span<const uint32_t>  clusterCfg;      // per code region: split | msb << 8 | lsb << 16
    std::span<const uint32_t>  blob0;           // DcGlobal's words
    uint32_t                   blob0Bits = 0;
};

/// modular.slang modGroupSections + the layout: DcGlobal, the (empty) DC groups
/// and AcGlobal, then every group's section (GroupHeader with its RCT, the ANS
/// stream over its compact token list), byte-aligned.
[[nodiscard]] Result<SectionsOutput> emitModularSections(const ModSectionsInput& in);

}   // namespace sjxl::enc::cpu
