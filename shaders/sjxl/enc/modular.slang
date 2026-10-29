// Copyright (c) 2026 jesus luque.
//
// The lossless Modular encoder's per-pixel and per-group kernels: the RCT
// (YCoCg, type 6) from sRGB8, the residual token histograms per context, and
// the group sections as ANS streams (GroupHeader with the global tree and no
// transforms, then the three channels' rectangles), one thread per group,
// measured in mode 1 and placed backwards in mode 2 as ac_sections.slang does.
module enc_modular;

#include "sjxl/common/bitio.slangh"
#include "sjxl/enc/modular.slangh"
#include "sjxl/enc/ans.slangh"

struct ModParams {
    uint width;
    uint height;
    uint groupDim;
    uint groupsX;
    uint numGroups;
    uint numLeaves;
    uint mode;        // modGroupSections: 1 count, 2 emit
    uint pad0;
};

StructuredBuffer<uint>     rgb8;          // width * height * 3 bytes
RWStructuredBuffer<int>    planes;        // 3 * width * height: Y, Co, Cg (written by rctYCoCg)
StructuredBuffer<int>      planesIn;      // the same buffer, read by the token kernels
StructuredBuffer<uint>     treeNodes;     // kModTreeWords per node
RWStructuredBuffer<uint>   ctxHist;       // numLeaves * kAcAlphabet(64)
StructuredBuffer<uint>     ctxMap;        // leaf -> cluster
StructuredBuffer<uint>     ansInfo;
StructuredBuffer<uint>     ansReverse;
StructuredBuffer<uint>     groupStart;    // mode 2: bit offset of each group's section
RWStructuredBuffer<uint>   groupBits;     // mode 1 output
RWStructuredBuffer<uint>   body;          // mode 2 output
ConstantBuffer<ModParams>  params;

static const uint kModAlphabet = 64;

uint byteAt(uint index) {
    return (rgb8[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void rctYCoCg(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint i = tid.y * params.width + tid.x;
    const int r = int(byteAt(i * 3 + 0));
    const int g = int(byteAt(i * 3 + 1));
    const int b = int(byteAt(i * 3 + 2));
    // enc_rct.cc FwdRctRow<6>
    const int co = r - b;
    const int tmp = b + (co >> 1);
    const int cg = g - tmp;
    const uint plane = params.width * params.height;
    planes[0 * plane + i] = tmp + (cg >> 1);
    planes[1 * plane + i] = co;
    planes[2 * plane + i] = cg;
}

/// One thread per pixel and channel: the token counted in its leaf's context,
/// with the edge rules of the group the pixel will be coded in.
[shader("compute")]
[numthreads(64, 1, 1)]
void modHist(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= 3 * n) return;
    const uint c = tid.x / n;
    const uint i = tid.x % n;
    const uint x = i % params.width, y = i / params.width;
    const uint x0 = (x / params.groupDim) * params.groupDim;
    const uint y0 = (y / params.groupDim) * params.groupDim;
    uint ctx, value;
    modToken(planesIn, params.width, params.height, x0, y0, c, x - x0, y - y0, treeNodes, ctx, value);
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    uint before;
    InterlockedAdd(ctxHist[ctx * kModAlphabet + min(tok, kModAlphabet - 1)], 1u, before);
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
    for (int c = 2; c >= 0; --c) {
        for (int y = int(h) - 1; y >= 0; --y) {
            for (int x = int(w) - 1; x >= 0; --x) {
                uint ctx, value;
                modToken(planesIn, params.width, params.height, x0, y0, uint(c), uint(x), uint(y), treeNodes, ctx, value);
                put(aw, ctx, value);
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
