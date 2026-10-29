// Copyright (c) 2026 jesus luque.
//
// Two of libjxl's modular-stream refinements at its hare tier, over the
// tokens of a Modular frame: run-length LZ77 (enc_lz77.cc ApplyLZ77_RLE: a run
// of a stream's token values repeating the previous value becomes a length
// token in the first token's context plus a distance token in an extra
// context, when the estimated saving beats libjxl's threshold) and the
// hybrid-uint configuration per cluster (enc_ans.cc ChooseUintConfigs, the
// fast candidate set), chosen by the cost of the cluster's histogram under
// each candidate. The streams' tokens end up in compact per-stream lists the
// section writers read backwards.
module enc_lz77;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/enc/ans.slangh"

static const uint kHistBins = 128;
static const uint kMinSymbol = 224;
static const uint kMinLength = 3;
static const uint kLengthFlag = 0x80000000u;    // compact token: an LZ77 length (value = run - kMinLength)
static const uint kNumCandidates = 4;
static const uint kCodeStride = 256;
static const uint kThreads = 64;
static const uint kAlpha = 256;

struct Lz77Params {
    uint numStreams;      // group streams, then the global stream last
    uint numContexts;     // leaves (the distance context is numContexts)
    uint mode;            // rleScan: 1 run-length tokens, 0 plain copy
    uint numClusters;
    uint totalTokens;     // over every stream (the compact lists' capacity)
    uint lz77;            // 1: the streams carry length tokens
    uint regionBase;      // the set's first code region in hist / clusterCfg
    uint ctxBase;         // the set's first context in ctxMap
    uint numChannels;
    uint groupDim;
    uint groupsX;
    uint numGroups;
    uint width;
    uint height;
    uint histogramTries;  // ANS histogram precisions tried per candidate (runtime: no unrolling)
    uint numCandidatesRt; // kNumCandidates as a runtime value (same reason)
};

StructuredBuffer<uint>     modTokens;      // per sample: ctx << 18 | value
StructuredBuffer<uint4>    chanTable;      // per channel: w, h, offset, grouped
StructuredBuffer<uint>     streamStart;    // per stream: first slot of its compact list
RWStructuredBuffer<uint>   streamLen;      // per stream: tokens in the list
RWStructuredBuffer<uint>   streamTokens;   // the compact lists: ctx << 18 | value, kLengthFlag for lengths
RWStructuredBuffer<uint>   ctxHist;        // (numContexts + 1) * kHistBins: input to the scan (costs), rebuilt by rleHistograms
RWStructuredBuffer<uint>   rleStats;       // [0] bit decrease (x16), [1] total symbols
StructuredBuffer<uint>     ctxMap;         // context -> cluster (set at ctxBase)
RWStructuredBuffer<uint>   cfgHist;        // [cluster][candidate][kAlpha] token counts of the normal tokens
RWStructuredBuffer<uint>   cfgExtra;       // [cluster][candidate] raw bits; [numClusters * kNumCandidates + cluster * kNumCandidates + k] invalid flags
RWStructuredBuffer<uint>   lengthHist;     // [cluster][kAlpha] the length tokens' symbols
RWStructuredBuffer<uint>   hist;           // the code regions' final histograms (regionBase + cluster) * kCodeStride
RWStructuredBuffer<uint>   clusterCfg;     // per code region: split | msb << 8 | lsb << 16
RWStructuredBuffer<uint>   maxToken;       // [0] the largest symbol over the set
ConstantBuffer<Lz77Params> params;

static const uint3 kCandidates[kNumCandidates] = {uint3(4, 2, 0), uint3(4, 1, 2), uint3(0, 0, 0), uint3(2, 0, 1)};

/// The stream's channels and the rectangle each contributes, in order: the
/// stream's token list is channel-major, raster within the rectangle.
bool streamRect(uint stream, uint c, out uint x0, out uint y0, out uint w, out uint h, out uint stride, out uint offset) {
    const uint4 ch = chanTable[c];
    stride = ch.x;
    offset = ch.z;
    const bool grouped = ch.w != 0;
    if (stream == params.numGroups) {   // the global stream
        x0 = 0; y0 = 0; w = ch.x; h = ch.y;
        return !grouped && w > 0 && h > 0;
    }
    if (!grouped) { x0 = 0; y0 = 0; w = 0; h = 0; return false; }
    x0 = (stream % params.groupsX) * params.groupDim;
    y0 = (stream / params.groupsX) * params.groupDim;
    if (x0 >= ch.x || y0 >= ch.y) { w = 0; h = 0; return false; }
    w = min(params.groupDim, ch.x - x0);
    h = min(params.groupDim, ch.y - y0);
    return true;
}

/// The cost in bits of a token under the pre-LZ77 statistics (SymbolCostEstimator).
float symbolBits(uint ctx, uint tok) {
    uint total = 0;
    for (uint s = 0; s < kHistBins; ++s) total += ctxHist[ctx * kHistBins + s];
    const uint c = ctxHist[ctx * kHistBins + min(tok, kHistBins - 1)];
    if (c == 0) return float(kAnsLogTabSize);
    if (c == total) return 0.0f;
    return -log2(float(c) / float(total));
}

/// One thread per stream: its tokens in order into the compact list; in mode 1
/// a run of tokens repeating the value before it becomes a length token (the
/// run's first context) plus a distance token when the saving beats libjxl's
/// per-run estimate (CeilLog2Nonzero(len + 1) + 1 bits). The run's tokens are
/// appended as they come and rewound when the run is taken.
[shader("compute")]
[numthreads(1, 1, 1)]
void rleScan(uint3 tid: SV_DispatchThreadID) {
    const uint stream = tid.x;
    if (stream >= params.numStreams) return;
    const uint base = streamStart[stream];
    uint out = 0;
    uint total = 0;
    float decrease = 0.0f;
    uint prevValue = 0;
    bool havePrev = false;
    uint runLen = 0, runCtx = 0;
    float runCost = 0.0f;
    for (uint c = 0; c <= params.numChannels; ++c) {
        uint x0 = 0, y0 = 0, w = 0, h = 0, stride = 1, offset = 0;
        const bool last = c == params.numChannels;
        if (!last && !streamRect(stream, c, x0, y0, w, h, stride, offset)) continue;
        const uint count = last ? 1 : w * h;   // one extra iteration flushes the final run
        for (uint i = 0; i < count; ++i) {
            uint t = 0;
            if (!last) {
                const uint lx = i % w, ly = i / w;
                t = modTokens[offset + (y0 + ly) * stride + x0 + lx];
                ++total;
            }
            const uint value = t & 0x3FFFFu, ctx = t >> 18;
            if (params.mode == 1 && !last && havePrev && value == prevValue) {
                if (runLen == 0) runCtx = ctx;
                uint tok, nbits, raw;
                hybridEncode(value, tok, nbits, raw);
                runCost += symbolBits(ctx, tok) + float(nbits);
                ++runLen;
                streamTokens[base + out++] = t;
                continue;
            }
            if (runLen != 0) {
                const uint lz77Len = runLen >= kMinLength ? runLen - kMinLength : 0;
                const float lz77Cost = runLen >= kMinLength ? float(firstbithigh(lz77Len + 1) + 1 + 1) : 0.0f;
                if (runLen >= kMinLength && runCost > lz77Cost) {
                    out -= runLen;   // the run's plain tokens, replaced
                    streamTokens[base + out++] = kLengthFlag | (runCtx << 18) | lz77Len;
                    streamTokens[base + out++] = (params.numContexts << 18) | 1u;   // distance 1 (special distance index 1)
                    decrease += runCost - lz77Cost;
                }
                runLen = 0;
                runCost = 0.0f;
            }
            if (!last) {
                streamTokens[base + out++] = t;
                prevValue = value;
                havePrev = true;
            }
        }
    }
    streamLen[stream] = out;
    uint before;
    InterlockedAdd(rleStats[0], uint(decrease * 16.0f), before);
    InterlockedAdd(rleStats[1], total, before);
}

/// One thread per compact token: the context histograms after run-length coding.
[shader("compute")]
[numthreads(64, 1, 1)]
void rleHistograms(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.totalTokens) return;
    // Find the stream: the lists are consecutive; skip the gaps after each list.
    uint stream = 0;
    for (; stream + 1 < params.numStreams; ++stream) {
        if (tid.x < streamStart[stream + 1]) break;
    }
    if (tid.x - streamStart[stream] >= streamLen[stream]) return;
    const uint t = streamTokens[tid.x];
    const uint ctx = (t & ~kLengthFlag) >> 18;
    const uint value = t & 0x3FFFFu;
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    if ((t & kLengthFlag) != 0) tok += kMinSymbol;
    uint before;
    InterlockedAdd(ctxHist[ctx * kHistBins + min(tok, kHistBins - 1)], 1u, before);
}

/// One thread per compact token: its symbol under every candidate configuration
/// into its cluster's histograms (length tokens aside, into lengthHist).
[shader("compute")]
[numthreads(64, 1, 1)]
void uintConfigHist(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.totalTokens) return;
    uint stream = 0;
    for (; stream + 1 < params.numStreams; ++stream) {
        if (tid.x < streamStart[stream + 1]) break;
    }
    if (tid.x - streamStart[stream] >= streamLen[stream]) return;
    const uint t = streamTokens[tid.x];
    const uint ctx = (t & ~kLengthFlag) >> 18;
    const uint value = t & 0x3FFFFu;
    const uint cluster = ctxMap[params.ctxBase + ctx];
    uint before;
    if ((t & kLengthFlag) != 0) {
        uint tok, nbits, raw;
        hybridEncode(value, tok, nbits, raw);
        InterlockedAdd(lengthHist[cluster * kAlpha + min(tok + kMinSymbol, kAlpha - 1)], 1u, before);
        return;
    }
    const uint limit = params.lz77 != 0 ? kMinSymbol : kAlpha;
    for (uint k = 0; k < kNumCandidates; ++k) {
        uint tok, nbits, raw;
        hybridEncodeCfg(value, kCandidates[k].x, kCandidates[k].y, kCandidates[k].z, tok, nbits, raw);
        if (tok >= limit) {
            cfgExtra[params.numClusters * kNumCandidates + cluster * kNumCandidates + k] = 1;
            continue;
        }
        InterlockedAdd(cfgHist[(cluster * kNumCandidates + k) * kAlpha + tok], 1u, before);
        if (nbits != 0) InterlockedAdd(cfgExtra[cluster * kNumCandidates + k], nbits, before);
    }
}

groupshared uint gMaxTok;

/// One thread-group (one working thread) per cluster: the candidate whose
/// histogram cost (ANSPopulationCost: the cheapest normalised description plus
/// the data) plus raw bits plus the configuration's own bits is lowest; its
/// counts (the length tokens added) become the cluster's code region.
[shader("compute")]
[numthreads(1, 1, 1)]
void uintConfigChoose(uint3 gid: SV_GroupID) {
    const uint cluster = gid.x;
    if (cluster >= params.numClusters) return;
    float bestCost = 3.4e38f;
    uint best = 0;
    for (uint k = 0; k < min(params.numCandidatesRt, kNumCandidates); ++k) {
        if (cfgExtra[params.numClusters * kNumCandidates + cluster * kNumCandidates + k] != 0) continue;
        uint length = 1;
        for (uint s = 0; s < kAlpha; ++s) {
            gAnsCounts[s] = cfgHist[(cluster * kNumCandidates + k) * kAlpha + s];
            if (gAnsCounts[s] != 0) length = s + 1;
        }
        float cost = ansBestHistogramCost(hist, length, params.histogramTries) + float(cfgExtra[cluster * kNumCandidates + k]);
        cost += float(ceilLog2Nonzero(kCandidates[k].x + 1) + ceilLog2Nonzero(kCandidates[k].x - kCandidates[k].y + 1));
        if (cost < bestCost) { bestCost = cost; best = k; }
    }
    const uint region = params.regionBase + cluster;
    clusterCfg[region] = kCandidates[best].x | (kCandidates[best].y << 8) | (kCandidates[best].z << 16);
    uint maxTok = 0;
    for (uint s = 0; s < kAlpha; ++s) {
        const uint c = cfgHist[(cluster * kNumCandidates + best) * kAlpha + s] + lengthHist[cluster * kAlpha + s];
        hist[region * kCodeStride + s] = c;
        if (c != 0) maxTok = s;
    }
    uint before;
    InterlockedMax(maxToken[0], maxTok, before);
}
