// Copyright (c) 2026 jesus luque.
//
// The lossless Modular encoder's kernels: the RCT (YCoCg, type 6) from 8- or
// 16-bit RGB (an alpha channel is carried as the fourth channel), one thread per
// group tokenising its channels through the weighted predictor and the frame's
// fixed tree (the WP error state is serial, so the residuals are too), and the
// group sections as ANS streams (GroupHeader with the global tree and no
// transforms, then the channels' rectangles), one thread per group, measured in
// mode 1 and placed backwards in mode 2 as ac_sections.slang does.
module enc_modular;

#include "sjxl/common/bitio.slangh"
#include "sjxl/enc/modular.slangh"
#include "sjxl/enc/wp.slangh"
#include "sjxl/enc/ans.slangh"

struct ModParams {
    uint width;
    uint height;
    uint groupDim;
    uint groupsX;
    uint numGroups;
    uint numLeaves;
    uint mode;        // modGroupSections: 1 count, 2 emit
    uint numChannels; // 3, or 4 with alpha
    uint bitDepth;    // 8 or 16
    uint wpWordsPerGroup;
    uint pass;        // modTokenize: which predictor pass (tokens and histograms land at its offset)
    uint ctxBase;     // remapTokens: the first context of this stream's set in ctxHist
    float dcQuantX;   // dcFramePlanes: the DC frame's dc_quant per channel (as the decoder's F16 holds it)
    float dcQuantY;
    float dcQuantB;
    // The learned tree (tree.slang): modSamples picks pixels at `sampleFraction` and
    // quantises the candidate properties `propIds` with `thresholds`; modTokenizeTree
    // walks the final tree over every property.
    float sampleFraction;
    uint  sampleCapacity;
    uint  streamBase;     // stream id of group 0 (property 1)
    uint  rct;            // modGroupSections: 1 when the groups carry their RCT (groupRct)
    uint4 propIds;
};

StructuredBuffer<uint>     rgb8;          // width * height * numChannels samples of bitDepth / 8 bytes, little-endian
StructuredBuffer<int>      thresholds;    // modSamples: [16][64] per property, count at 16 * 64 + property
RWStructuredBuffer<uint2>  samples;       // modSamples output
RWStructuredBuffer<uint>   sampleCount;   // [0]
RWStructuredBuffer<float>  rctCosts;      // rctCost: [group][candidate] estimated bits
RWStructuredBuffer<uint>   groupRct;      // rctApply: the RCT type chosen per group
StructuredBuffer<float>    dcValue;       // dcFramePlanes: the VarDCT frame's DC image, [c][block], channels X, Y, B
RWStructuredBuffer<int>    planes;        // numChannels * width * height: Y, Co, Cg[, alpha] (written by rctYCoCg)
StructuredBuffer<int>      planesIn;      // the same buffer, read by the token kernels
StructuredBuffer<uint>     treeNodes;     // kModTreeWords per node
RWStructuredBuffer<uint>   wpScratch;     // numGroups * wpWordsPerGroup
RWStructuredBuffer<uint>   passTokens;    // 2 passes * numChannels * width * height: ctx << 18 | value per predictor tree
RWStructuredBuffer<uint>   passHist;      // 2 passes * numChannels * kSetContexts * kHistBins
RWStructuredBuffer<float>  passBits;      // 2 passes * numChannels: the entropy estimate
StructuredBuffer<uint>     choice;        // per channel: the pass chosen
StructuredBuffer<uint>     leafMap;       // per channel * kSetContexts: the chosen pass's leaf -> the combined tree's leaf
RWStructuredBuffer<uint>   modTokens;     // numChannels * width * height: the final ctx << 18 | value
RWStructuredBuffer<uint>   ctxHist;       // numLeaves * kHistBins
StructuredBuffer<uint>     ctxMap;        // leaf -> cluster
StructuredBuffer<uint>     ansInfo;
StructuredBuffer<uint>     ansReverse;
StructuredBuffer<uint>     groupStart;    // mode 2: bit offset of each group's section
RWStructuredBuffer<uint>   groupBits;     // mode 1 output
RWStructuredBuffer<uint>   body;          // mode 2 output
ConstantBuffer<ModParams>  params;

static const uint kHistBins = 128;
static const uint kSetContexts = 64;

uint byteAt(uint index) {
    return (rgb8[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}

/// Sample `index` of the input as an integer.
int sampleAt(uint index) {
    if (params.bitDepth == 8) return int(byteAt(index));
    return int(byteAt(2 * index) | (byteAt(2 * index + 1) << 8));
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
    int3 in = int3(r, g, b);
    // RctPermute: inputs {in[p % 3], in[(p + 1 + p / 3) % 3], in[(p + 2 - p / 3) % 3]}.
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

int3 rctSampleAt(uint x, uint y, uint rctType) {
    const uint i = (y * params.width + x) * params.numChannels;
    return forwardRct(sampleAt(i), sampleAt(i + 1), sampleAt(i + 2), rctType);
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

/// One thread per pixel: the planes under the group's cheapest RCT (ties to
/// the first candidate, libjxl's order); alpha copied. The first pixel of a
/// group records the choice for the group's header.
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
    const uint i = tid.y * params.width + tid.x;
    const uint nc = params.numChannels;
    const int3 t = forwardRct(sampleAt(i * nc + 0), sampleAt(i * nc + 1), sampleAt(i * nc + 2), rctType);
    const uint plane = params.width * params.height;
    planes[0 * plane + i] = t.x;
    planes[1 * plane + i] = t.y;
    planes[2 * plane + i] = t.z;
    if (nc > 3) planes[3 * plane + i] = sampleAt(i * nc + 3);
    if (tid.x % params.groupDim == 0 && tid.y % params.groupDim == 0) groupRct[g] = rctType;
}

/// The GroupHeader of a modular stream: use_global_tree, the default WP header
/// and its RCT (if any) over channels 0..2: num_transforms, id kRCT, begin_c 0
/// and rct_type as U32(Val(6), Bits(2), BitsOffset(4, 2), BitsOffset(6, 10)).
uint groupHeaderBits(uint rctType) {
    if (rctType == 0) return 4;
    const uint typeBits = rctType == 6 ? 2 : rctType < 4 ? 4 : rctType < 18 ? 6 : 8;
    return 2 + 2 + 2 + 5 + typeBits;
}

void emitGroupHeader(RWStructuredBuffer<uint> body, uint pos, uint rctType) {
    emitAtomic(body, pos, 2, 0x3u);   // use_global_tree, wp all_default
    if (rctType == 0) {
        emitAtomic(body, pos + 2, 2, 0);   // num_transforms = 0
        return;
    }
    emitAtomic(body, pos + 2, 2, 1);   // num_transforms = 1 (U32 selector 1: Val(1))
    emitAtomic(body, pos + 4, 2, 0);   // Transform id: RCT
    emitAtomic(body, pos + 6, 5, 0);   // begin_c: selector 0 (Bits(3)) = 0
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

/// One thread per group: its channels in raster order through the weighted
/// predictor state and the pass's tree (the decoder's edge rules inside the
/// group's rectangle), the tokens kept per sample and counted per channel and
/// context. Two passes (the WP tree, the gradient tree) feed the per-channel
/// choice libjxl's Predictor::Best makes.
[shader("compute")]
[numthreads(1, 1, 1)]
void modTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    const uint x0 = (g % params.groupsX) * params.groupDim;
    const uint y0 = (g / params.groupsX) * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    const uint plane = params.width * params.height;
    for (uint c = 0; c < params.numChannels; ++c) {
        const uint base = c * plane;
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            for (uint lx = 0; lx < w; ++lx) {
                const uint x = x0 + lx, y = y0 + ly;
                const uint at = base + y * params.width + x;
                const int v = planesIn[at];
                const int left = lx > 0 ? planesIn[at - 1] : (ly > 0 ? planesIn[at - params.width] : 0);
                const int top = ly > 0 ? planesIn[at - params.width] : left;
                const int topleft = (lx > 0 && ly > 0) ? planesIn[at - params.width - 1] : left;
                const int topright = (lx + 1 < w && ly > 0) ? planesIn[at - params.width + 1] : top;
                const int toptop = ly > 1 ? planesIn[at - 2 * params.width] : top;
                int wpProp;
                const int64_t pred = wpPredict(wpScratch, wp, lx, ly, int64_t(top), int64_t(left), int64_t(topright), int64_t(topleft), int64_t(toptop), wpProp);
                uint predictor;
                int offset;
                const uint ctx = modTreeLeaf(treeNodes, c, ly, top, left, topleft, wpProp, predictor, offset);
                int64_t guess = int64_t(offset);
                if (predictor == kModPredWeighted) guess += pred;
                else if (predictor == kModPredGradient) guess += int64_t(modClampedGradient(left, top, topleft));
                else if (predictor == kModPredLeft) guess += int64_t(left);
                const uint value = packSigned(int(int64_t(v) - guess));
                passTokens[params.pass * params.numChannels * plane + at] = (ctx << 18) | value;
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

/// The neighbours of (lx, ly) in a group rectangle with the decoder's edge rules.
struct Neighbours {
    int left, top, topleft, topright, toptop, leftleft;
};

Neighbours neighboursAt(uint base, uint at, uint lx, uint ly, uint w) {
    Neighbours n;
    n.left = lx > 0 ? planesIn[at - 1] : (ly > 0 ? planesIn[at - params.width] : 0);
    n.top = ly > 0 ? planesIn[at - params.width] : n.left;
    n.topleft = (lx > 0 && ly > 0) ? planesIn[at - params.width - 1] : n.left;
    n.topright = (lx + 1 < w && ly > 0) ? planesIn[at - params.width + 1] : n.top;
    n.toptop = ly > 1 ? planesIn[at - 2 * params.width] : n.top;
    n.leftleft = lx > 1 ? planesIn[at - 2] : n.left;
    return n;
}

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
/// fraction of the pixels, the residual tokens of the Weighted and Gradient
/// predictors and the quantised candidate properties.
[shader("compute")]
[numthreads(1, 1, 1)]
void modSamples(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    const uint x0 = (g % params.groupsX) * params.groupDim;
    const uint y0 = (g / params.groupsX) * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    const uint plane = params.width * params.height;
    const uint stream = params.streamBase + g;
    const float fraction = max(params.sampleFraction, min(1.0f, 1024.0f / float(w * h)));
    for (uint c = 0; c < params.numChannels; ++c) {
        const uint base = c * plane;
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            int prevGradient = 0;
            for (uint lx = 0; lx < w; ++lx) {
                const uint x = x0 + lx, y = y0 + ly;
                const uint at = base + y * params.width + x;
                const int v = planesIn[at];
                const Neighbours n = neighboursAt(base, at, lx, ly, w);
                int wpProp;
                const int64_t wpPred = wpPredict(wpScratch, wp, lx, ly, int64_t(n.top), int64_t(n.left), int64_t(n.topright), int64_t(n.topleft),
                                                 int64_t(n.toptop), wpProp);
                if (float(hashSample(at) & 0xFFFFFFu) < fraction * 16777216.0f) {
                    int props[kNumProperties];
                    modProperties(c, stream, lx, ly, n, prevGradient, wpProp, props);
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
    const uint x0 = (g % params.groupsX) * params.groupDim;
    const uint y0 = (g / params.groupsX) * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    const uint plane = params.width * params.height;
    const uint stream = params.streamBase + g;
    for (uint c = 0; c < params.numChannels; ++c) {
        const uint base = c * plane;
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint ly = 0; ly < h; ++ly) {
            int prevGradient = 0;
            for (uint lx = 0; lx < w; ++lx) {
                const uint x = x0 + lx, y = y0 + ly;
                const uint at = base + y * params.width + x;
                const int v = planesIn[at];
                const Neighbours n = neighboursAt(base, at, lx, ly, w);
                int wpProp;
                const int64_t wpPred = wpPredict(wpScratch, wp, lx, ly, int64_t(n.top), int64_t(n.left), int64_t(n.topright), int64_t(n.topleft),
                                                 int64_t(n.toptop), wpProp);
                int props[kNumProperties];
                modProperties(c, stream, lx, ly, n, prevGradient, wpProp, props);
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

/// One thread per sample: the chosen pass's token with its leaf renumbered
/// into the combined tree, and the final context histograms.
[shader("compute")]
[numthreads(64, 1, 1)]
void remapTokens(uint3 tid: SV_DispatchThreadID) {
    const uint plane = params.width * params.height;
    if (tid.x >= params.numChannels * plane) return;
    const uint c = tid.x / plane;
    const uint t = passTokens[choice[c] * params.numChannels * plane + tid.x];
    const uint ctx = leafMap[c * kSetContexts + (t >> 18)];
    const uint value = t & 0x3FFFFu;
    modTokens[tid.x] = (ctx << 18) | value;
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    uint before;
    InterlockedAdd(ctxHist[(params.ctxBase + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
}

struct AnsWriter {
    uint mode;
    uint state;
    uint pos;
    uint bits;
};

void put(inout AnsWriter aw, uint ctx, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    const uint cluster = ctxMap[ctx];
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

[shader("compute")]
[numthreads(1, 1, 1)]
void modGroupSections(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) return;
    const uint x0 = (g % params.groupsX) * params.groupDim;
    const uint y0 = (g / params.groupsX) * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    AnsWriter aw;
    aw.mode = params.mode;
    aw.state = kAnsSignature << 16;
    const uint headerBits = groupHeaderBits(params.rct != 0 ? groupRct[g] : 0);
    aw.bits = headerBits + 32;
    aw.pos = aw.mode == 2 ? groupStart[g] + groupBits[g] : 0;
    const uint plane = params.width * params.height;
    for (int c = int(params.numChannels) - 1; c >= 0; --c) {
        for (int y = int(h) - 1; y >= 0; --y) {
            for (int x = int(w) - 1; x >= 0; --x) {
                const uint t = modTokens[uint(c) * plane + (y0 + uint(y)) * params.width + x0 + uint(x)];
                put(aw, t >> 18, t & 0x3FFFFu);
            }
        }
    }
    if (aw.mode == 1) {
        groupBits[g] = aw.bits;
    } else {
        // GroupHeader (the group's RCT when the frame has one), then the state.
        emitGroupHeader(body, groupStart[g], params.rct != 0 ? groupRct[g] : 0);
        emitAtomic(body, groupStart[g] + headerBits, 32, aw.state);
    }
}
