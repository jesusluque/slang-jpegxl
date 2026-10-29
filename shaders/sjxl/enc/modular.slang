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
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint>     rgb8;          // width * height * numChannels samples of bitDepth / 8 bytes, little-endian
RWStructuredBuffer<int>    planes;        // numChannels * width * height: Y, Co, Cg[, alpha] (written by rctYCoCg)
StructuredBuffer<int>      planesIn;      // the same buffer, read by the token kernels
StructuredBuffer<uint>     treeNodes;     // kModTreeWords per node
RWStructuredBuffer<uint>   wpScratch;     // numGroups * wpWordsPerGroup
RWStructuredBuffer<uint>   modTokens;     // numChannels * width * height: ctx << 18 | value
RWStructuredBuffer<uint>   ctxHist;       // numLeaves * kHistBins
StructuredBuffer<uint>     ctxMap;        // leaf -> cluster
StructuredBuffer<uint>     ansInfo;
StructuredBuffer<uint>     ansReverse;
StructuredBuffer<uint>     groupStart;    // mode 2: bit offset of each group's section
RWStructuredBuffer<uint>   groupBits;     // mode 1 output
RWStructuredBuffer<uint>   body;          // mode 2 output
ConstantBuffer<ModParams>  params;

static const uint kHistBins = 128;

uint byteAt(uint index) {
    return (rgb8[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}

/// Sample `index` of the input as an integer.
int sampleAt(uint index) {
    if (params.bitDepth == 8) return int(byteAt(index));
    return int(byteAt(2 * index) | (byteAt(2 * index + 1) << 8));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rctYCoCg(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint i = tid.y * params.width + tid.x;
    const uint nc = params.numChannels;
    const int r = sampleAt(i * nc + 0);
    const int g = sampleAt(i * nc + 1);
    const int b = sampleAt(i * nc + 2);
    // enc_rct.cc FwdRctRow<6>
    const int co = r - b;
    const int tmp = b + (co >> 1);
    const int cg = g - tmp;
    const uint plane = params.width * params.height;
    planes[0 * plane + i] = tmp + (cg >> 1);
    planes[1 * plane + i] = co;
    planes[2 * plane + i] = cg;
    if (nc > 3) planes[3 * plane + i] = sampleAt(i * nc + 3);
}

/// One thread per group: its channels in raster order through the weighted
/// predictor and the tree (the decoder's edge rules inside the group's
/// rectangle), the tokens kept per pixel and counted per context.
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
                modTokens[at] = (ctx << 18) | value;
                uint tok, nbits, raw;
                hybridEncode(value, tok, nbits, raw);
                uint before;
                InterlockedAdd(ctxHist[ctx * kHistBins + min(tok, kHistBins - 1)], 1u, before);
                wpUpdateErrors(wpScratch, wp, int64_t(v), lx, ly);
            }
        }
    }
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

static const uint kGroupHeaderBits = 4;   // use_global_tree 1, wp all_default 1, num_transforms = 0: U32 selector 0

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
    aw.bits = kGroupHeaderBits + 32;
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
        // GroupHeader: use_global_tree = 1, weighted header all_default = 1, no transforms (2 zero bits), then the state.
        emitAtomic(body, groupStart[g], kGroupHeaderBits, 0x3u);
        emitAtomic(body, groupStart[g] + kGroupHeaderBits, 32, aw.state);
    }
}
