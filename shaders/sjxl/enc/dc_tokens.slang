// Copyright (c) 2026 jesus luque.
//
// The DC group sections' two modular streams as tokens and as ANS streams.
//   dcTokenize   one thread per DC group: the quantised DC of Y, X, B in raster
//                order, predicted with the weighted predictor and put in the
//                context of libjxl's kWPFixedDC tree (a search over the WP
//                property), the tokens kept per block and counted per context.
//   metaTokenize one thread per AcMetadata item: the YtoX / YtoB tiles
//                (Gradient), the strategy and qf of every varblock (Zero and
//                Left) and the EPF sharpness of every block (4, in leaves with
//                offset 4: a single-symbol cluster, so no bits) in libjxl's
//                kACMeta tree. With ANS every token must be in the stream, zero
//                cost or not: the decoder's state walks through each one.
//   dcGroupSections one thread per DC group, modes 1 (count) and 2 (emit): the
//                DC stream and the AcMetadata stream as ANS chains, backwards,
//                placed from each part's end as the AC sections are.
module enc_dc_tokens;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/common/transform_basis.slangh"
#include "sjxl/common/ac_context.slangh"
#include "sjxl/enc/modular.slangh"
#include "sjxl/enc/wp.slangh"
#include "sjxl/enc/ans.slangh"

static const uint kMetaTiles = 32 * 32;
static const uint kMetaBlocks = 256 * 256;
static const uint kMetaItemsPerGroup = 2 * kMetaTiles + 3 * kMetaBlocks;   // YtoX, YtoB tiles; ACS row, QF row; EPF
static const uint kHistBins = 128;
static const uint kSetContexts = 64;      // contexts per histogram set (tree leaves)
static const uint kNoToken = 0xFFFFFFFFu;

struct DcTokenParams {
    uint nblocks;
    uint blocksX;
    uint blocksY;
    uint numDcGroups;
    uint tilesX;
    uint mode;              // dcGroupSections: 1 count, 2 emit
    uint dcTreeNodes;
    uint metaTreeNodes;
    uint wpWordsPerGroup;
    uint numAcSets;         // the AC passes' sets precede the DC groups' sets
    uint useDcFrame;        // kUseDcFrame: no DC stream in the DC group sections
    uint pad2;
    uint4 hshift;           // chroma subsampling per channel (X, Y, B): a channel's DC grid is the group's >> shift
    uint4 vshift;
};

uint shiftOf(uint4 v, uint c) { return c == 0 ? v.x : c == 1 ? v.y : v.z; }

StructuredBuffer<uint4>    dcGroups;      // bx0, by0, w, h (blocks)
StructuredBuffer<int>      dcQuant;       // [c][block]
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     rawQf;
StructuredBuffer<int>      cmapX;
StructuredBuffer<int>      cmapB;
StructuredBuffer<uint>     dcTree;        // kModTreeWords per node, decode order
StructuredBuffer<uint>     metaTree;
RWStructuredBuffer<uint>   wpScratch;     // numDcGroups * wpWordsPerGroup
RWStructuredBuffer<uint>   dcTokens;      // [c][block]: ctx << 18 | value
RWStructuredBuffer<uint>   metaTokens;    // per item: ctx << 18 | value, or kNoToken
RWStructuredBuffer<uint>   ctxHist;       // sets of kSetContexts * kHistBins; set 1 + 2g DC, 2 + 2g meta
StructuredBuffer<uint>     ctxMap;        // sets of kSetContexts
StructuredBuffer<uint>     ansInfo;       // regions of 256: set k's cluster c at (k * 32 + c)
StructuredBuffer<uint>     ansReverse;
StructuredBuffer<uint>     dcPartStart;   // mode 2: pre, DC stream, mid, meta stream bit offsets per group
RWStructuredBuffer<uint>   dcBits;        // mode 1: the DC stream's bits per group
RWStructuredBuffer<uint>   metaBits;      // mode 1: the AcMetadata stream's bits
RWStructuredBuffer<uint>   body;
ConstantBuffer<DcTokenParams> params;

uint dcSet(uint g) { return params.numAcSets + 2 * g; }
uint metaSet(uint g) { return params.numAcSets + 1 + 2 * g; }
/// The AC sets (one per pass, kNumAcContexts each) come first in ctxHist / ctxMap; the DC group sets follow, kSetContexts each.
uint setCtxBase(uint set) { return params.numAcSets * kNumAcContexts + (set - params.numAcSets) * kSetContexts; }

void countToken(uint set, uint ctx, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    uint before;
    InterlockedAdd(ctxHist[(setCtxBase(set) + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
}

/// The tree walk over the properties the two trees read: 0 channel, 2 y, 6 N, 7 W, 15 WP.
uint treeLeaf(StructuredBuffer<uint> tree, uint c, uint y, int top, int left, int wpProp, out uint predictor, out int offset) {
    uint node = 0;
    for (;;) {
        const int prop = int(tree[node * kModTreeWords + 0]);
        if (prop < 0) break;
        int v = 0;
        if (prop == 0) v = int(c);
        else if (prop == 2) v = int(y);
        else if (prop == 6) v = top;
        else if (prop == 7) v = left;
        else if (prop == 15) v = wpProp;
        node = v > int(tree[node * kModTreeWords + 1]) ? tree[node * kModTreeWords + 2] : tree[node * kModTreeWords + 3];
    }
    predictor = tree[node * kModTreeWords + 5];
    offset = int(tree[node * kModTreeWords + 6]);
    return tree[node * kModTreeWords + 4];
}

static const uint kPredZero = 0, kPredLeft = 1, kPredWeighted = 6, kPredGradient = 5;

[shader("compute")]
[numthreads(1, 1, 1)]
void dcTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numDcGroups) return;
    const uint4 rect = dcGroups[g];
    const uint physical[3] = {1, 0, 2};
    for (uint ci = 0; ci < 3; ++ci) {
        const uint c = physical[ci];
        const uint plane = c * params.nblocks;
        // The channel's DC grid in this group: the group's blocks >> its shifts; a
        // subsampled channel's value sits at the co-located full-grid block.
        const uint hs = shiftOf(params.hshift, c), vs = shiftOf(params.vshift, c);
        const uint w = rect.z >> hs, h = rect.w >> vs;
        const uint strideX = 1u << hs, strideY = params.blocksX << vs;
        WpState wp = wpInit(wpScratch, g * params.wpWordsPerGroup, w);
        for (uint y = 0; y < h; ++y) {
            for (uint x = 0; x < w; ++x) {
                const uint block = (rect.y + (y << vs)) * params.blocksX + rect.x + (x << hs);
                const int v = dcQuant[plane + block];
                const int left = x > 0 ? dcQuant[plane + block - strideX] : (y > 0 ? dcQuant[plane + block - strideY] : 0);
                const int top = y > 0 ? dcQuant[plane + block - strideY] : left;
                const int topleft = (x > 0 && y > 0) ? dcQuant[plane + block - strideY - strideX] : left;
                const int topright = (x + 1 < w && y > 0) ? dcQuant[plane + block - strideY + strideX] : top;
                const int toptop = y > 1 ? dcQuant[plane + block - 2 * strideY] : top;
                int wpProp;
                const int64_t pred = wpPredict(wpScratch, wp, x, y, int64_t(top), int64_t(left), int64_t(topright), int64_t(topleft), int64_t(toptop), wpProp);
                uint predictor;
                int offset;
                const uint ctx = treeLeaf(dcTree, ci, y, top, left, wpProp, predictor, offset);
                int64_t guess = int64_t(offset);
                if (predictor == kPredWeighted) guess += pred;
                else if (predictor == kPredGradient) guess += int64_t(modClampedGradient(left, top, topleft));
                else if (predictor == kPredLeft) guess += int64_t(left);
                const uint value = packSigned(int(int64_t(v) - guess));
                dcTokens[plane + block] = (ctx << 18) | value;
                countToken(dcSet(g), ctx, value);
                wpUpdateErrors(wpScratch, wp, int64_t(v), x, y);
            }
        }
    }
}

/// The strategy or qf value of the first block at or before `block` in raster
/// order within the group rectangle: the left neighbour in the ACS/QF channel.
bool previousVarblock(uint4 rect, uint bx, uint by, out uint prevBlock) {
    for (;;) {
        if (bx == 0) {
            if (by == 0) return false;
            --by;
            bx = rect.z - 1;
        } else {
            --bx;
        }
        const uint b = (rect.y + by) * params.blocksX + rect.x + bx;
        if ((acs[b] & 1) != 0) { prevBlock = b; return true; }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void metaTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.numDcGroups * kMetaItemsPerGroup) return;
    const uint g = i / kMetaItemsPerGroup;
    const uint l = i % kMetaItemsPerGroup;
    const uint4 rect = dcGroups[g];
    const uint tilesX = (rect.z + 7) / 8;
    const uint tilesY = (rect.w + 7) / 8;
    metaTokens[i] = kNoToken;
    uint c, y, ctx, predictor;
    int value, left, top, topleft, offset;
    if (l < 2 * kMetaTiles) {
        // YtoX (channel 0) then YtoB (channel 1): Gradient over the tile grid.
        const uint t = l % kMetaTiles;
        const uint tx = t % 32, ty = t / 32;
        if (tx >= tilesX || ty >= tilesY) return;
        c = l < kMetaTiles ? 0 : 1;
        const uint gx0 = rect.x / 8, gy0 = rect.y / 8;
        StructuredBuffer<int> m = c == 0 ? cmapX : cmapB;
        value = m[(gy0 + ty) * params.tilesX + gx0 + tx];
        left = tx > 0 ? m[(gy0 + ty) * params.tilesX + gx0 + tx - 1] : (ty > 0 ? m[(gy0 + ty - 1) * params.tilesX + gx0 + tx] : 0);
        top = ty > 0 ? m[(gy0 + ty - 1) * params.tilesX + gx0 + tx] : left;
        topleft = (tx > 0 && ty > 0) ? m[(gy0 + ty - 1) * params.tilesX + gx0 + tx - 1] : left;
        y = ty;
    } else if (l >= 2 * kMetaTiles + 2 * kMetaBlocks) {
        // Channel 3: the EPF sharpness, 4 at every block.
        const uint b = l - 2 * kMetaTiles - 2 * kMetaBlocks;
        const uint bx = b % 256, by = b / 256;
        if (bx >= rect.z || by >= rect.w) return;
        c = 3;
        y = by;
        value = 4;
        left = bx > 0 ? 4 : (by > 0 ? 4 : 0);
        top = by > 0 ? 4 : left;
        topleft = (bx > 0 && by > 0) ? 4 : left;
    } else {
        // Channel 2, two rows of `count`: the strategies (row 0) then the qf - 1 (row 1) of the varblocks.
        const uint b = (l - 2 * kMetaTiles) % kMetaBlocks;
        const bool qfRow = (l - 2 * kMetaTiles) >= kMetaBlocks;
        const uint bx = b % 256, by = b / 256;
        if (bx >= rect.z || by >= rect.w) return;
        const uint block = (rect.y + by) * params.blocksX + rect.x + bx;
        if ((acs[block] & 1) == 0) return;
        c = 2;
        y = qfRow ? 1 : 0;
        const int acsValue = int(acs[block] >> 1);
        const int qfValue = int(rawQf[block]) - 1;
        value = qfRow ? qfValue : acsValue;
        uint prev;
        const bool hasLeft = previousVarblock(rect, bx, by, prev);
        if (!qfRow) {   // row 0: left = previous strategy or 0; top = left; topleft = left
            left = hasLeft ? int(acs[prev] >> 1) : 0;
            top = left;
            topleft = left;
        } else {        // row 1: left = previous qf, or the strategy above at x = 0; top = this varblock's strategy
            top = acsValue;
            left = hasLeft ? int(rawQf[prev]) - 1 : top;
            topleft = hasLeft ? int(acs[prev] >> 1) : left;
        }
    }
    ctx = treeLeaf(metaTree, c, y, top, left, 0, predictor, offset);
    int guess = offset;
    if (predictor == kPredGradient) guess += modClampedGradient(left, top, topleft);
    else if (predictor == kPredLeft) guess += left;
    const uint v = packSigned(value - guess);
    metaTokens[i] = (ctx << 18) | v;
    countToken(metaSet(g), ctx, v);
}

struct AnsWriter {
    uint mode;
    uint state;
    uint pos;
    uint bits;
    uint set;
};

void put(inout AnsWriter aw, uint ctx, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    const uint cluster = ctxMap[setCtxBase(aw.set) + ctx];
    const uint region = aw.set * 32 + cluster;
    const uint e = ansInfo[region * kAnsInfoStride + tok];
    if (aw.mode == 2) {
        aw.pos -= nbits;
        emitAtomic(body, aw.pos, nbits, raw);
    }
    uint ansBits, ansN;
    ansPut(aw.state, e & 0xFFFFu, e >> 16, ansReverse, region * kAnsReverseStride, ansN, ansBits);
    if (aw.mode == 2) {
        aw.pos -= ansN;
        emitAtomic(body, aw.pos, ansN, ansBits);
    }
    aw.bits += nbits + ansN;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void dcGroupSections(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numDcGroups) return;
    const uint4 rect = dcGroups[g];
    // The DC stream: channels B, X, Y backwards, raster backwards.
    AnsWriter aw;
    aw.mode = params.mode;
    aw.state = kAnsSignature << 16;
    aw.bits = 32;
    aw.set = dcSet(g);
    aw.pos = aw.mode == 2 ? dcPartStart[g * 4 + 1] + dcBits[g] : 0;
    const uint physical[3] = {1, 0, 2};
    if (params.useDcFrame == 0) {
        for (int ci = 2; ci >= 0; --ci) {
            const uint c = physical[ci];
            const uint plane = c * params.nblocks;
            const uint hs = shiftOf(params.hshift, c), vs = shiftOf(params.vshift, c);
            const uint w = rect.z >> hs, h = rect.w >> vs;
            for (int y = int(h) - 1; y >= 0; --y) {
                for (int x = int(w) - 1; x >= 0; --x) {
                    const uint t = dcTokens[plane + (rect.y + (uint(y) << vs)) * params.blocksX + rect.x + (uint(x) << hs)];
                    put(aw, t >> 18, t & 0x3FFFFu);
                }
            }
        }
        if (aw.mode == 1) dcBits[g] = aw.bits;
        else emitAtomic(body, dcPartStart[g * 4 + 1], 32, aw.state);
    } else if (aw.mode == 1) {
        dcBits[g] = 0;
    }
    // The AcMetadata stream: the items backwards.
    AnsWriter mw;
    mw.mode = params.mode;
    mw.state = kAnsSignature << 16;
    mw.bits = 32;
    mw.set = metaSet(g);
    mw.pos = mw.mode == 2 ? dcPartStart[g * 4 + 3] + metaBits[g] : 0;
    for (int l = int(kMetaItemsPerGroup) - 1; l >= 0; --l) {
        const uint t = metaTokens[g * kMetaItemsPerGroup + uint(l)];
        if (t == kNoToken) continue;
        put(mw, t >> 18, t & 0x3FFFFu);
    }
    if (mw.mode == 1) metaBits[g] = mw.bits;
    else emitAtomic(body, dcPartStart[g * 4 + 3], 32, mw.state);
}
