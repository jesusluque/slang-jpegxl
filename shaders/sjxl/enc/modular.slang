// Copyright (c) 2026 jesus luque.
//
// The lossless Modular encoder's kernels over a channel table: the planes
// (int samples, one region per channel: meta channels such as palettes and
// the image channels), the RCT chosen per group, the DC frame's planes, the
// tree learner's samples, the tokenisers (one thread per group, the weighted
// predictor state serial per channel), and the group sections as ANS streams
// measured in mode 1 and placed backwards in mode 2 as ac_sections.slang does.
// A channel that fits a group (or is a meta channel) belongs to the global
// stream and is handled by group 0; the others are split into group
// rectangles with the decoder's edge rules inside each rectangle.
module enc_modular;

#include "sjxl/common/bitio.slangh"
#include "sjxl/enc/modular.slangh"
#include "sjxl/enc/wp.slangh"
#include "sjxl/enc/ans.slangh"

struct ModParams {
    uint width;       // the image (the grouped channels' size)
    uint height;
    uint groupDim;
    uint groupsX;
    uint numGroups;
    uint numLeaves;
    uint mode;        // modGroupSections: 1 count, 2 emit
    uint numChannels; // channels in chanTable (meta channels included)
    uint bitDepth;    // 8 or 16 (sampleAt, the DC frame's planes)
    uint wpWordsPerGroup;
    uint pass;        // modTokenize: which predictor pass (tokens and histograms land at its offset)
    uint ctxBase;     // remapTokens / modTokenizeTree: the first context of this stream's set in ctxHist
    float dcQuantX;   // dcFramePlanes: the DC frame's dc_quant per channel (as the decoder's F16 holds it)
    float dcQuantY;
    float dcQuantB;
    float sampleFraction;   // modSamples: the pixel fraction kept
    uint  sampleCapacity;
    uint  streamBase;       // stream id of group 0's stream (property 1)
    uint  rct;              // modGroupSections: 1 when the groups carry their RCT (groupRct)
    uint  rctBeginC;        // the RCT's first channel (after the meta channels)
    uint  totalSamples;     // samples over every channel
    uint  nbMeta;           // meta channels (the group streams number their channels from the first non-meta one)
    uint  lz77;             // modGroupSections: 1 when the lists carry length tokens (kLengthFlag)
    uint  pad1, pad2, pad3;
    uint4 propIds;          // modSamples: the candidate properties
};

StructuredBuffer<uint>     rgb8;          // width * height * numChannels samples of bitDepth / 8 bytes, little-endian
StructuredBuffer<float>    dcValue;       // dcFramePlanes: the VarDCT frame's DC image, [c][block], channels X, Y, B
StructuredBuffer<uint4>    chanTable;     // per channel: w, h, offset in the planes, grouped (1: coded in the group streams)
StructuredBuffer<int>      thresholds;    // modSamples: [16][64] per property, count at 16 * 64 + property
RWStructuredBuffer<uint2>  samples;       // modSamples output
RWStructuredBuffer<uint>   sampleCount;   // [0]
RWStructuredBuffer<float>  rctCosts;      // rctCost: [group][candidate] estimated bits
RWStructuredBuffer<uint>   groupRct;      // rctApply: the RCT type chosen per group
RWStructuredBuffer<int>    planes;        // the channels' samples (written by unpackPlanes, rctApply, dcFramePlanes, extractAlpha)
StructuredBuffer<int>      planesIn;      // the same buffer, read by the token kernels
StructuredBuffer<uint>     treeNodes;     // kModTreeWords per node
RWStructuredBuffer<uint>   wpScratch;     // numGroups * wpWordsPerGroup
RWStructuredBuffer<uint>   passTokens;    // 2 passes * totalSamples: ctx << 18 | value per predictor tree
RWStructuredBuffer<uint>   passHist;      // 2 passes * numChannels * kSetContexts * kHistBins
RWStructuredBuffer<float>  passBits;      // 2 passes * numChannels: the entropy estimate
StructuredBuffer<uint>     choice;        // per channel: the pass chosen
StructuredBuffer<uint>     leafMap;       // per channel * kSetContexts: the chosen pass's leaf -> the combined tree's leaf
RWStructuredBuffer<uint>   modTokens;     // totalSamples: the final ctx << 18 | value
RWStructuredBuffer<uint>   ctxHist;       // numLeaves * kHistBins
StructuredBuffer<uint>     ctxMap;        // leaf -> cluster
StructuredBuffer<uint>     ansInfo;
StructuredBuffer<uint>     ansReverse;
StructuredBuffer<uint>     groupStart;    // mode 2: bit offset of each group's section
StructuredBuffer<uint>     streamTokens;  // the compact per-stream token lists (lz77.slang rleScan)
StructuredBuffer<uint>     streamStart;   // per stream (groups, then the global one): first slot
StructuredBuffer<uint>     streamLen;     // per stream: tokens
StructuredBuffer<uint>     clusterCfg;    // per code region: the hybrid-uint config split | msb << 8 | lsb << 16
RWStructuredBuffer<uint>   groupBits;     // mode 1 output
RWStructuredBuffer<uint>   body;          // mode 2 output
ConstantBuffer<ModParams>  params;

static const uint kHistBins = 128;
static const uint kSetContexts = 64;

uint byteAt(uint index) {
    return (rgb8[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}

/// Sample `index` of the input (8-bit, or 16-bit little-endian).
int sampleAt(uint index) {
    if (params.bitDepth == 8) return int(byteAt(index));
    return int(byteAt(2 * index) | (byteAt(2 * index + 1) << 8));
}

struct Chan {
    uint w, h, offset;
    bool grouped;
};

Chan chanAt(uint c) {
    const uint4 e = chanTable[c];
    Chan ch;
    ch.w = e.x;
    ch.h = e.y;
    ch.offset = e.z;
    ch.grouped = e.w != 0;
    return ch;
}

/// The rectangle of channel `ch` that group g codes: the group's part of a
/// grouped channel, the whole channel for group 0 otherwise (none for the others).
bool groupRect(Chan ch, uint g, out uint x0, out uint y0, out uint w, out uint h) {
    if (!ch.grouped) {
        x0 = 0; y0 = 0; w = ch.w; h = ch.h;
        return g == 0 && w > 0 && h > 0;
    }
    x0 = (g % params.groupsX) * params.groupDim;
    y0 = (g / params.groupsX) * params.groupDim;
    if (x0 >= ch.w || y0 >= ch.h) { w = 0; h = 0; return false; }
    w = min(params.groupDim, ch.w - x0);
    h = min(params.groupDim, ch.h - y0);
    return true;
}

/// One thread per pixel: the input's channels as flat int planes (channel c at c * width * height).
[shader("compute")]
[numthreads(16, 16, 1)]
void unpackPlanes(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint i = tid.y * params.width + tid.x;
    const uint nc = params.numChannels;
    const uint plane = params.width * params.height;
    for (uint c = 0; c < nc; ++c) planes[c * plane + i] = sampleAt(i * nc + c);
}

// --- the RCT per group ------------------------------------------------------------------

/// libjxl tries four RCTs per group at its hare tier (enc_modular.cc): none,
/// YCoCg, type 5 (R, G - (R + B) / 2, B - R) and type 10 (G, B - G, R - G).
static const uint kRctCandidates = 4;
static const uint kRctTypes[kRctCandidates] = {0, 6, 5, 10};

/// FwdRct: the three channels of a pixel under `rctType` (permutation * 7 + transform).
int3 forwardRct(int r, int g, int b, uint rctType) {
    const uint permutation = rctType / 7;
    const uint transform = rctType % 7;
    const int c[3] = {r, g, b};
    const int first = c[permutation % 3];
    int second = c[(permutation + 1 + permutation / 3) % 3];
    int third = c[(permutation + 2 - permutation / 3) % 3];
    if (transform == 6) {
        const int o1 = first - third;
        const int tmp = third + (o1 >> 1);
        const int o2 = second - tmp;
        return int3(tmp + (o2 >> 1), o1, o2);
    }
    const uint sec = transform >> 1;
    if (sec == 1) second = second - first;
    else if (sec == 2) second = second - ((first + third) >> 1);
    if ((transform & 1) != 0) third = third - first;
    return int3(first, second, third);
}

/// The three colour channels (rctBeginC..+2) of pixel (x, y) under the RCT.
int3 rctSampleAt(uint x, uint y, uint rctType) {
    const Chan c0 = chanAt(params.rctBeginC), c1 = chanAt(params.rctBeginC + 1), c2 = chanAt(params.rctBeginC + 2);
    const uint i = y * params.width + x;
    return forwardRct(planesIn[c0.offset + i], planesIn[c1.offset + i], planesIn[c2.offset + i], rctType);
}

static const uint kRctContexts = 18;
static const uint kRctTokens = 64;
static const uint kRctThreads = 64;
groupshared uint gRctHist[kRctContexts * kRctTokens];
groupshared uint gRctExtra;

/// EstimateCost (enc_modular_simd.cc): one thread-group per (group, candidate):
/// per channel, the Shannon entropy of the ClampedGradient residual tokens in
/// 18 contexts by the neighbours' spread, plus the raw bits.
[shader("compute")]
[numthreads(kRctThreads, 1, 1)]
void rctCost(uint3 gid: SV_GroupID, uint3 ltid: SV_GroupThreadID) {
    const uint g = gid.x / kRctCandidates;
    const uint cand = gid.x % kRctCandidates;
    if (g >= params.numGroups) return;
    const uint tid = ltid.x;
    const uint rctType = kRctTypes[cand];
    const uint x0 = (g % params.groupsX) * params.groupDim;
    const uint y0 = (g / params.groupsX) * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    const uint cutoffs[17] = {0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500};
    float cost = 0.0f;
    for (uint c = 0; c < 3; ++c) {
        for (uint i = tid; i < kRctContexts * kRctTokens; i += kRctThreads) gRctHist[i] = 0;
        if (tid == 0) gRctExtra = 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint p = tid; p < w * h; p += kRctThreads) {
            const uint lx = p % w, ly = p / w;
            const uint x = x0 + lx, y = y0 + ly;
            const int v = rctSampleAt(x, y, rctType)[c];
            const int left = lx > 0 ? rctSampleAt(x - 1, y, rctType)[c] : (ly > 0 ? rctSampleAt(x, y - 1, rctType)[c] : 0);
            const int top = ly > 0 ? rctSampleAt(x, y - 1, rctType)[c] : left;
            const int topleft = (lx > 0 && ly > 0) ? rctSampleAt(x - 1, y - 1, rctType)[c] : left;
            const uint maxDiff = uint(max(max(left, top), topleft) - min(min(left, top), topleft));
            uint ctx = 0;
            for (uint k = 0; k < 17; ++k) ctx += maxDiff < cutoffs[k] ? 1 : 0;
            uint tok, nbits, raw;
            hybridEncode(packSigned(v - modClampedGradient(left, top, topleft)), tok, nbits, raw);
            uint before;
            InterlockedAdd(gRctHist[ctx * kRctTokens + min(tok, kRctTokens - 1)], 1u, before);
            if (nbits != 0) InterlockedAdd(gRctExtra, nbits, before);
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid == 0) {
            for (uint ctx = 0; ctx < kRctContexts; ++ctx) {
                float total = 0.0f;
                for (uint t = 0; t < kRctTokens; ++t) total += float(gRctHist[ctx * kRctTokens + t]);
                if (total == 0.0f) continue;
                for (uint t = 0; t < kRctTokens; ++t) {
                    const float n = float(gRctHist[ctx * kRctTokens + t]);
                    if (n > 0.0f) cost += n * log2(total / n);
                }
            }
            cost += float(gRctExtra);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0) rctCosts[g * kRctCandidates + cand] = cost;
}

/// One thread per pixel: the three colour channels under the group's cheapest
/// RCT (ties to the first candidate, libjxl's order), in place. The first
/// pixel of a group records the choice for the group's header.
[shader("compute")]
[numthreads(16, 16, 1)]
void rctApply(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint g = (tid.y / params.groupDim) * params.groupsX + tid.x / params.groupDim;
    uint best = 0;
    for (uint k = 1; k < kRctCandidates; ++k) {
        if (rctCosts[g * kRctCandidates + k] < rctCosts[g * kRctCandidates + best]) best = k;
    }
    const uint rctType = kRctTypes[best];
    const Chan c0 = chanAt(params.rctBeginC), c1 = chanAt(params.rctBeginC + 1), c2 = chanAt(params.rctBeginC + 2);
    const uint i = tid.y * params.width + tid.x;
    const int3 t = forwardRct(planesIn[c0.offset + i], planesIn[c1.offset + i], planesIn[c2.offset + i], rctType);
    planes[c0.offset + i] = t.x;
    planes[c1.offset + i] = t.y;
    planes[c2.offset + i] = t.z;
    if (tid.x % params.groupDim == 0 && tid.y % params.groupDim == 0) groupRct[g] = rctType;
}

/// The GroupHeader of a modular stream: use_global_tree, the default WP header
/// and its RCT (if any) over channels rctBeginC..+2: num_transforms, id kRCT,
/// begin_c and rct_type as U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10)).
uint groupHeaderBits(uint rctType) {
    if (rctType == 0) return 4;
    const uint typeBits = rctType == 6 ? 2 : rctType < 4 ? 4 : rctType < 18 ? 6 : 8;
    return 2 + 2 + 2 + 5 + typeBits;
}

void emitGroupHeader(RWStructuredBuffer<uint> body, uint pos, uint rctType, uint beginC) {
    emitAtomic(body, pos, 2, 0x3u);   // use_global_tree, wp all_default
    if (rctType == 0) {
        emitAtomic(body, pos + 2, 2, 0);   // num_transforms = 0
        return;
    }
    emitAtomic(body, pos + 2, 2, 1);   // num_transforms = 1 (U32 selector 1: Val(1))
    emitAtomic(body, pos + 4, 2, 0);   // Transform id: RCT
    emitAtomic(body, pos + 6, 2, 0);   // begin_c: selector 0 (Bits(3)) ...
    emitAtomic(body, pos + 8, 3, beginC);
    if (rctType == 6) {
        emitAtomic(body, pos + 11, 2, 0);
    } else if (rctType < 4) {
        emitAtomic(body, pos + 11, 2, 1);
        emitAtomic(body, pos + 13, 2, rctType);
    } else if (rctType < 18) {
        emitAtomic(body, pos + 11, 2, 2);
        emitAtomic(body, pos + 13, 4, rctType - 2);
    } else {
        emitAtomic(body, pos + 11, 2, 3);
        emitAtomic(body, pos + 13, 6, rctType - 10);
    }
}

// --- other plane producers ---------------------------------------------------------------

/// One thread per block: the DC frame's channels from the DC image. A modular
/// XYB frame holds Y, X and B - Y as integers that the decoder scales by its
/// dc_quant per channel (dec_modular.cc: "XYB is encoded as YX(B-Y)"), so each
/// channel is the DC rounded to its own dc_quant grid and B's integer is
/// relative to Y's.
[shader("compute")]
[numthreads(64, 1, 1)]
void dcFramePlanes(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;   // blocks
    if (tid.x >= n) return;
    const float fx = f16tof32(f32tof16(params.dcQuantX * 128.0f)) / 128.0f;
    const float fy = f16tof32(f32tof16(params.dcQuantY * 128.0f)) / 128.0f;
    const float fb = f16tof32(f32tof16(params.dcQuantB * 128.0f)) / 128.0f;
    const int yi = int(round(dcValue[1 * n + tid.x] / fy));
    const int xi = int(round(dcValue[0 * n + tid.x] / fx));
    const int bi = int(round(dcValue[2 * n + tid.x] / fb));
    planes[0 * n + tid.x] = yi;
    planes[1 * n + tid.x] = xi;
    planes[2 * n + tid.x] = bi - yi;
}

/// One thread per pixel: channel 3 (alpha) of the input as plane 0, for a
/// VarDCT frame's extra channel.
[shader("compute")]
[numthreads(16, 16, 1)]
void extractAlpha(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint i = tid.y * params.width + tid.x;
    planes[i] = sampleAt(i * params.numChannels + 3);
}

// --- tokenisation ------------------------------------------------------------------------

/// The neighbours of (lx, ly) in a rectangle of a channel with row stride
/// `stride` (at = the sample's index), with the decoder's edge rules.
struct Neighbours {
    int left, top, topleft, topright, toptop, leftleft;
};

Neighbours neighboursAt(uint at, uint stride, uint lx, uint ly, uint w) {
    Neighbours n;
    n.left = lx > 0 ? planesIn[at - 1] : (ly > 0 ? planesIn[at - stride] : 0);
    n.top = ly > 0 ? planesIn[at - stride] : n.left;
    n.topleft = (lx > 0 && ly > 0) ? planesIn[at - stride - 1] : n.left;
    n.topright = (lx + 1 < w && ly > 0) ? planesIn[at - stride + 1] : n.top;
    n.toptop = ly > 1 ? planesIn[at - 2 * stride] : n.top;
    n.leftleft = lx > 1 ? planesIn[at - 2] : n.left;
    return n;
}

/// One thread per group: its rectangles of every channel in raster order
/// through the weighted predictor state and the pass's fixed tree, the tokens
/// kept per sample and counted per channel and context. Two passes (the WP
/// tree, the gradient tree) feed the per-channel choice libjxl's
/// Predictor::Best makes.
[shader("compute")]
[numthreads(1, 1, 1)]
void modTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    for (uint c = 0; c < params.numChannels; ++c) {
        const Chan ch = chanAt(c);
        uint x0, y0, w, h;
        if (!groupRect(ch, g, x0, y0, w, h)) continue;
        const uint chanProp = ch.grouped ? c - params.nbMeta : c;   // a group stream's channels start at the first non-meta one
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            for (uint lx = 0; lx < w; ++lx) {
                const uint at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int v = planesIn[at];
                const Neighbours n = neighboursAt(at, ch.w, lx, ly, w);
                int wpProp;
                const int64_t pred = wpPredict(wpScratch, wp, lx, ly, int64_t(n.top), int64_t(n.left), int64_t(n.topright), int64_t(n.topleft),
                                               int64_t(n.toptop), wpProp);
                uint predictor;
                int offset;
                const uint ctx = modTreeLeaf(treeNodes, chanProp, ly, n.top, n.left, n.topleft, wpProp, predictor, offset);
                int64_t guess = int64_t(offset);
                if (predictor == kModPredWeighted) guess += pred;
                else if (predictor == kModPredGradient) guess += int64_t(modClampedGradient(n.left, n.top, n.topleft));
                else if (predictor == kModPredLeft) guess += int64_t(n.left);
                const uint value = packSigned(int(int64_t(v) - guess));
                passTokens[params.pass * params.totalSamples + at] = (ctx << 18) | value;
                uint tok, nbits, raw;
                hybridEncode(value, tok, nbits, raw);
                uint before;
                InterlockedAdd(passHist[((params.pass * params.numChannels + c) * kSetContexts + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
                wpUpdateErrors(wpScratch, wp, int64_t(v), lx, ly);
            }
        }
    }
}

// --- the learned tree ------------------------------------------------------------------

static const uint kThreshStride = 64;
static const uint kNumProperties = 16;

/// libjxl's 16 properties (context_predict.h): channel, stream, y, x, |N|, |W|,
/// N, W, W minus the previous pixel's gradient, the gradient W + N - NW, W - NW,
/// NW - N, N - NE, N - NN, W - WW, and the WP property.
void modProperties(uint c, uint stream, uint lx, uint ly, Neighbours n, int prevGradient, int wpProp, out int props[kNumProperties]) {
    props[0] = int(c);
    props[1] = int(stream);
    props[2] = int(ly);
    props[3] = int(lx);
    props[4] = abs(n.top);
    props[5] = abs(n.left);
    props[6] = n.top;
    props[7] = n.left;
    props[8] = n.left - prevGradient;
    props[9] = int(uint(n.left) + uint(n.top) - uint(n.topleft));
    props[10] = n.left - n.topleft;
    props[11] = n.topleft - n.top;
    props[12] = n.top - n.topright;
    props[13] = n.top - n.toptop;
    props[14] = n.left - n.leftleft;
    props[15] = wpProp;
}

uint propIdOf(uint k) { return k == 0 ? params.propIds.x : k == 1 ? params.propIds.y : k == 2 ? params.propIds.z : params.propIds.w; }

/// QuantMap: the number of the property's thresholds below the value.
uint quantizeProperty(uint propId, int v) {
    const uint count = uint(thresholds[kNumProperties * kThreshStride + propId]);
    uint q = 0;
    for (uint i = 0; i < count; ++i) {
        if (thresholds[propId * kThreshStride + i] < v) ++q;
    }
    return q;
}

uint hashSample(uint i) {
    uint x = i * 2654435761u + 0x9E3779B9u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

/// One thread per group: the tree learner's samples (GatherTreeData): for a
/// fraction of the samples, the residual tokens of the Weighted and Gradient
/// predictors and the quantised candidate properties.
[shader("compute")]
[numthreads(1, 1, 1)]
void modSamples(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    for (uint c = 0; c < params.numChannels; ++c) {
        const Chan ch = chanAt(c);
        uint x0, y0, w, h;
        if (!groupRect(ch, g, x0, y0, w, h)) continue;
        const uint stream = ch.grouped ? params.streamBase + g : 0;
        const uint chanProp = ch.grouped ? c - params.nbMeta : c;
        const float fraction = max(params.sampleFraction, min(1.0f, 1024.0f / float(w * h)));
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            int prevGradient = 0;
            for (uint lx = 0; lx < w; ++lx) {
                const uint at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int v = planesIn[at];
                const Neighbours n = neighboursAt(at, ch.w, lx, ly, w);
                int wpProp;
                const int64_t wpPred = wpPredict(wpScratch, wp, lx, ly, int64_t(n.top), int64_t(n.left), int64_t(n.topright), int64_t(n.topleft),
                                                 int64_t(n.toptop), wpProp);
                if (float(hashSample(at) & 0xFFFFFFu) < fraction * 16777216.0f) {
                    int props[kNumProperties];
                    modProperties(chanProp, stream, lx, ly, n, prevGradient, wpProp, props);
                    // TreeSamples::AddSample tokenises with HybridUintConfig(4, 1, 2).
                    uint tokW, nbitsW, raw, tokG, nbitsG;
                    hybridEncodeCfg(packSigned(int(int64_t(v) - wpPred)), 4, 1, 2, tokW, nbitsW, raw);
                    hybridEncodeCfg(packSigned(v - modClampedGradient(n.left, n.top, n.topleft)), 4, 1, 2, tokG, nbitsG, raw);
                    uint2 s;
                    s.x = min(tokW, 255u) | (min(nbitsW, 255u) << 8) | (min(tokG, 255u) << 16) | (min(nbitsG, 255u) << 24);
                    s.y = 0;
                    for (uint k = 0; k < 4; ++k) s.y |= min(quantizeProperty(propIdOf(k), props[propIdOf(k)]), 255u) << (8 * k);
                    uint idx;
                    InterlockedAdd(sampleCount[0], 1u, idx);
                    if (idx < params.sampleCapacity) samples[idx] = s;
                }
                prevGradient = int(uint(n.left) + uint(n.top) - uint(n.topleft));
                wpUpdateErrors(wpScratch, wp, int64_t(v), lx, ly);
            }
        }
    }
}

/// One thread per group: the tokens under the learned tree (treeNodes in the
/// decoder's layout, split values as property values), with the leaf's
/// predictor, into modTokens and ctxHist.
[shader("compute")]
[numthreads(1, 1, 1)]
void modTokenizeTree(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    for (uint c = 0; c < params.numChannels; ++c) {
        const Chan ch = chanAt(c);
        uint x0, y0, w, h;
        if (!groupRect(ch, g, x0, y0, w, h)) continue;
        const uint stream = ch.grouped ? params.streamBase + g : 0;
        const uint chanProp = ch.grouped ? c - params.nbMeta : c;
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            int prevGradient = 0;
            for (uint lx = 0; lx < w; ++lx) {
                const uint at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int v = planesIn[at];
                const Neighbours n = neighboursAt(at, ch.w, lx, ly, w);
                int wpProp;
                const int64_t wpPred = wpPredict(wpScratch, wp, lx, ly, int64_t(n.top), int64_t(n.left), int64_t(n.topright), int64_t(n.topleft),
                                                 int64_t(n.toptop), wpProp);
                int props[kNumProperties];
                modProperties(chanProp, stream, lx, ly, n, prevGradient, wpProp, props);
                uint node = 0;
                for (;;) {
                    const int prop = int(treeNodes[node * kModTreeWords + 0]);
                    if (prop < 0) break;
                    const int pv = uint(prop) < kNumProperties ? props[uint(prop)] : 0;
                    node = pv > int(treeNodes[node * kModTreeWords + 1]) ? treeNodes[node * kModTreeWords + 2] : treeNodes[node * kModTreeWords + 3];
                }
                const uint ctx = treeNodes[node * kModTreeWords + 4];
                const uint predictor = treeNodes[node * kModTreeWords + 5];
                int64_t guess = int64_t(int(treeNodes[node * kModTreeWords + 6]));
                if (predictor == kModPredWeighted) guess += wpPred;
                else if (predictor == kModPredGradient) guess += int64_t(modClampedGradient(n.left, n.top, n.topleft));
                else if (predictor == kModPredLeft) guess += int64_t(n.left);
                else if (predictor == 2) guess += int64_t(n.top);
                const uint value = packSigned(int(int64_t(v) - guess));
                modTokens[at] = (ctx << 18) | value;
                uint tok, nbits, raw;
                hybridEncode(value, tok, nbits, raw);
                uint before;
                InterlockedAdd(ctxHist[(params.ctxBase + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
                prevGradient = props[9];
                wpUpdateErrors(wpScratch, wp, int64_t(v), lx, ly);
            }
        }
    }
}

/// One thread per (pass, channel): the bits its tokens would take under an
/// ideal code per context, sum of n log2(N / n).
[shader("compute")]
[numthreads(64, 1, 1)]
void channelEntropy(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= 2 * params.numChannels) return;
    float bits = 0.0f;
    for (uint ctx = 0; ctx < kSetContexts; ++ctx) {
        const uint base = (tid.x * kSetContexts + ctx) * kHistBins;
        float total = 0.0f;
        for (uint sym = 0; sym < kHistBins; ++sym) total += float(passHist[base + sym]);
        if (total == 0.0f) continue;
        for (uint sym = 0; sym < kHistBins; ++sym) {
            const float n = float(passHist[base + sym]);
            if (n > 0.0f) bits += n * log2(total / n);
        }
    }
    passBits[tid.x] = bits;
}

/// The channel a flat sample index belongs to.
uint channelOfSample(uint i) {
    for (uint c = 0; c + 1 < params.numChannels; ++c) {
        if (i < chanTable[c + 1].z) return c;
    }
    return params.numChannels - 1;
}

/// One thread per sample: the chosen pass's token with its leaf renumbered
/// into the combined tree, and the final context histograms.
[shader("compute")]
[numthreads(64, 1, 1)]
void remapTokens(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.totalSamples) return;
    const uint c = channelOfSample(tid.x);
    const uint t = passTokens[choice[c] * params.totalSamples + tid.x];
    const uint ctx = leafMap[c * kSetContexts + (t >> 18)];
    const uint value = t & 0x3FFFFu;
    modTokens[tid.x] = (ctx << 18) | value;
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    uint before;
    InterlockedAdd(ctxHist[(params.ctxBase + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
}

/// One thread per sample: the histograms of tokens the CPU produced
/// (EntropyCpu.cpp): mode 0 the final tokens into ctxHist, mode 1 the pass's
/// tokens into passHist (modTokenize's layout).
[shader("compute")]
[numthreads(64, 1, 1)]
void modTokenCount(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.totalSamples) return;
    uint before;
    if (params.mode == 0) {
        const uint t = modTokens[tid.x];
        uint tok, nbits, raw;
        hybridEncode(t & 0x3FFFFu, tok, nbits, raw);
        InterlockedAdd(ctxHist[(params.ctxBase + (t >> 18)) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
    } else {
        const uint c = channelOfSample(tid.x);
        const uint t = passTokens[params.pass * params.totalSamples + tid.x];
        uint tok, nbits, raw;
        hybridEncode(t & 0x3FFFFu, tok, nbits, raw);
        InterlockedAdd(passHist[((params.pass * params.numChannels + c) * kSetContexts + (t >> 18)) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
    }
}

// --- the group sections ------------------------------------------------------------------

struct AnsWriter {
    uint mode;
    uint state;
    uint pos;
    uint bits;
};

static const uint kLengthFlag = 0x80000000u;
static const uint kMinSymbol = 224;

/// A compact token: a length token's symbol is kMinSymbol + its (4, 2, 0) token,
/// a normal one its cluster's configuration.
void put(inout AnsWriter aw, uint t) {
    const uint ctx = (t & ~kLengthFlag) >> 18;
    const uint value = t & 0x3FFFFu;
    const uint cluster = ctxMap[ctx];
    uint tok, nbits, raw;
    if ((t & kLengthFlag) != 0) {
        hybridEncode(value, tok, nbits, raw);
        tok += kMinSymbol;
    } else {
        const uint cfg = clusterCfg[cluster];
        hybridEncodeCfg(value, cfg & 0xFFu, (cfg >> 8) & 0xFFu, (cfg >> 16) & 0xFFu, tok, nbits, raw);
    }
    const uint e = ansInfo[cluster * kAnsInfoStride + tok];
    if (aw.mode == 2) {
        aw.pos -= nbits;
        emitAtomic(body, aw.pos, nbits, raw);
    }
    uint ansBits, ansN;
    ansPut(aw.state, e & 0xFFFFu, e >> 16, ansReverse, cluster * kAnsReverseStride, ansN, ansBits);
    if (aw.mode == 2) {
        aw.pos -= ansN;
        emitAtomic(body, aw.pos, ansN, ansBits);
    }
    aw.bits += nbits + ansN;
}

/// One thread per group: its section (GroupHeader, then the grouped channels'
/// rectangles as one ANS stream, channels in order, raster order each).
[shader("compute")]
[numthreads(1, 1, 1)]
void modGroupSections(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    const uint rctType = params.rct != 0 ? groupRct[g] : 0;
    const uint headerBits = groupHeaderBits(rctType);
    AnsWriter aw;
    aw.mode = params.mode;
    aw.state = kAnsSignature << 16;
    aw.bits = headerBits + 32;
    aw.pos = aw.mode == 2 ? groupStart[g] + groupBits[g] : 0;
    const uint base = streamStart[g];
    for (int i = int(streamLen[g]) - 1; i >= 0; --i) put(aw, streamTokens[base + uint(i)]);
    if (aw.mode == 1) {
        groupBits[g] = aw.bits;
    } else {
        emitGroupHeader(body, groupStart[g], rctType, params.rctBeginC);
        emitAtomic(body, groupStart[g] + headerBits, 32, aw.state);
    }
}
