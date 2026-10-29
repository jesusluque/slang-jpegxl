// Copyright (c) 2026 jesus luque.
//
// The varblock transforms and the AC tokens:
//   buildOrders       one thread: the natural coefficient order of every bucket.
//   adjustQuantField  one thread per first block: AdjustQuantField, the field
//                     of a varblock is the max of its blocks' (mixed with the
//                     mean at large distances).
//   dctRows / dctCols one thread per pixel and channel: the separable forward
//                     scaled DCT of every varblock, coefficients to libjxl's
//                     memory layout.
//   dcFromLlf         one thread per block and channel: the DC image value of
//                     a covered block from its varblock's lowest frequencies,
//                     the exact inverse of the decoder's LowestFrequenciesFromDC.
//   acTokenize        one thread per first block and channel: the varblock's
//                     tokens (context, value) in coding order, the non-zero
//                     count's context, and the context histograms.
module enc_transform;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/enc/ac_tokens.slangh"
#include "sjxl/enc/special.slangh"

struct TransformParams {
    uint  paddedWidth;
    uint  paddedHeight;
    uint  blocksX;
    uint  blocksY;
    float distance;    // AdjustQuantField's mean/max mix
    // Progressive passes (enc_progressive_split.cc SplitACCoefficients): this pass's
    // definition and what the earlier passes already coded.
    uint  pass;        // index: per-pass outputs live at pass * 3 * nblocks (tokens: * 64)
    uint  shift;       // this pass's quantisation shift
    uint  prevShift;   // the previous pass's shift (0: none, or it was exact)
    uint  ncoeffs;     // side of the coefficient square this pass covers per 8x8 block (2, 3, 8)
    uint  prevNcoeffs; // side completed by earlier exact passes (1: nothing)
    uint  ctxBase;     // this pass's histogram set in ctxHist
    uint  numBlockCtxs; // block contexts of the BlockCtxMap (15 default; a JPEG transcode's from bctxMap)
    uint  jpegCtx;      // 1: the block context comes from bctxMap and the block's luma DC
    uint4 hshift;      // chroma subsampling per channel (X, Y, B): a subsampled channel codes one block per 2x2 (JPEG recompression)
    uint4 vshift;
};

uint shiftOf(uint4 v, uint c) { return c == 0 ? v.x : c == 1 ? v.y : v.z; }

StructuredBuffer<float>    xyb;           // 3 planes (after the inverse Gaborish)
StructuredBuffer<uint>     acs;           // per block: strategy << 1 | first
StructuredBuffer<uint>     firstBlock;    // per block
RWStructuredBuffer<uint>   orders;        // natural orders, kCoeffOrderLimit * 64
RWStructuredBuffer<float>  rowPass;       // 3 planes: the row transforms
RWStructuredBuffer<float>  coeffs;        // [c][block][slot]
RWStructuredBuffer<float>  dcValue;       // [c][block]
RWStructuredBuffer<uint>   rawQf;         // per block; the varblock's value at its first block
StructuredBuffer<int>      acQuant;       // [c][block][slot]
StructuredBuffer<uint>     nonZeros;      // [c][block], at first blocks
RWStructuredBuffer<uint>   acTokens;      // [pass][c][block][slot]: ctx << 18 | value, at coefficient index k's slot
RWStructuredBuffer<uint>   numTokens;     // [pass][c][block]: coefficient tokens of the varblock (at its first block)
RWStructuredBuffer<uint>   nzCtx;         // [pass][c][block]: the non-zero count's context
RWStructuredBuffer<uint>   passNonZeros;  // [pass][c][block]: non-zero coded coefficients of the pass
RWStructuredBuffer<uint>   ctxHist;       // numPasses * kNumAcContexts * kHistBins
StructuredBuffer<uint>     bctxMap;       // jpegBlockCtx's map: [0] DC contexts, [2..9) luma DC thresholds, [16..) contexts
StructuredBuffer<int>      dcQuant;       // [c][block]: the quantised DC (the luma one picks the DC context)
ConstantBuffer<TransformParams> params;

static const uint kHistBins = 128;

uint nblocks() { return params.blocksX * params.blocksY; }
uint plane() { return params.paddedWidth * params.paddedHeight; }

[shader("compute")]
[numthreads(1, 1, 1)]
void buildOrders(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    // One strategy per bucket, as kStrategyOrder maps them.
    const uint strategyOfOrder[13] = {0, 1, 4, 5, 6, 8, 10, 18, 19, 21, 22, 24, 25};
    for (uint ord = 0; ord < kNumOrders; ++ord) {
        const uint s = strategyOfOrder[ord];
        naturalCoeffOrder(kCoveredX[s], kCoveredY[s], orders, orderOffset(ord));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void adjustQuantField(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= nblocks() || (acs[block] & 1) == 0) return;
    const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
    if (vb.covered == 1) return;
    // enc_adaptive_quantization.cc AdjustQuantField.
    float mixer = 1.0f;
    if (params.distance > 1.54138f) {
        mixer = max(0.0f, 1.0f - (params.distance - 1.54138f) * 0.56391f);
    }
    float mx = 0.0f, mean = 0.0f;
    for (uint y = 0; y < vb.cy; ++y) {
        for (uint x = 0; x < vb.cx; ++x) {
            const float q = float(rawQf[(vb.fby + y) * params.blocksX + vb.fbx + x]);
            mx = max(mx, q);
            mean += q;
        }
    }
    mean /= float(vb.covered);
    float v = mx;
    if (vb.covered >= 4) v = mx * mixer + (1.0f - mixer) * mean;
    rawQf[block] = uint(clamp(int(v + 0.5f), 1, 256));
}

/// The varblock containing pixel (px, py).
Varblock varblockAt(uint px, uint py) {
    const uint block = (py / 8) * params.blocksX + px / 8;
    const uint first = firstBlock[block];
    return varblockOf(first, acs[first] >> 1, params.blocksX);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dctRows(uint3 tid: SV_DispatchThreadID) {
    const uint n = plane();
    if (tid.x >= 3 * n) return;
    const uint c = tid.x / n;
    const uint p = tid.x % n;
    const uint px = p % params.paddedWidth, py = p / params.paddedWidth;
    const Varblock vb = varblockAt(px, py);
    const uint x0 = vb.fbx * 8;
    const uint u = px - x0;   // horizontal frequency of this output
    const uint rowBase = c * n + py * params.paddedWidth + x0;
    float sum = 0.0f;
    for (uint x = 0; x < vb.nx; ++x) {
        sum += xyb[rowBase + x] * fdctBasis(u, x, vb.nx);
    }
    rowPass[tid.x] = sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dctCols(uint3 tid: SV_DispatchThreadID) {
    const uint n = plane();
    if (tid.x >= 3 * n) return;
    const uint c = tid.x / n;
    const uint p = tid.x % n;
    const uint px = p % params.paddedWidth, py = p / params.paddedWidth;
    const Varblock vb = varblockAt(px, py);
    const uint y0 = vb.fby * 8;
    const uint u = px - vb.fbx * 8;
    const uint v = py - y0;   // vertical frequency of this output
    float sum = 0.0f;
    for (uint y = 0; y < vb.ny; ++y) {
        sum += rowPass[c * n + (y0 + y) * params.paddedWidth + px] * fdctBasis(v, y, vb.ny);
    }
    coeffs[coeffSlot(vb, c, nblocks(), params.blocksX, memIndex(v, u, vb.ny, vb.nx))] = sum;
}

/// One thread per block and channel with a special 8x8 strategy: its forward
/// transform replaces the DCT the separable passes wrote.
[shader("compute")]
[numthreads(64, 1, 1)]
void specialTransform(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    const uint strategy = acs[block] >> 1;
    if ((acs[block] & 1) == 0 || !isSpecialStrategy(strategy)) return;
    const uint px = (block % params.blocksX) * 8, py = (block / params.blocksX) * 8;
    float pix[64];
    for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) pix[y * 8 + x] = xyb[c * plane() + (py + y) * params.paddedWidth + px + x];
    float coef[64];
    forwardSpecial(strategy, pix, coef);
    for (uint k = 0; k < 64; ++k) coeffs[(c * nb + block) * 64 + k] = coef[k];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dcFromLlf(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    const uint first = firstBlock[block];
    const Varblock vb = varblockOf(first, acs[first] >> 1, params.blocksX);
    const uint bx = block % params.blocksX - vb.fbx;
    const uint by = block / params.blocksX - vb.fby;
    // The decoder recomputes LLF(v, u) = fdct(dc)(v, u) * rs(cy, v) * rs(cx, u),
    // so dc = idct(LLF / (rs rs)).
    float sum = 0.0f;
    for (uint v = 0; v < vb.cy; ++v) {
        for (uint u = 0; u < vb.cx; ++u) {
            const float llf = coeffs[coeffSlot(vb, c, nb, params.blocksX, memIndex(v, u, vb.ny, vb.nx))];
            sum += llf / (resampleScale(vb.cy, v) * resampleScale(vb.cx, u)) * idctBasis(v, by, vb.cy) * idctBasis(u, bx, vb.cx);
        }
    }
    dcValue[tid.x] = sum;
}

/// v >> shift rounding towards zero (SplitACCoefficients shift_right_round0).
int shiftRound0(int v, uint shift) {
    const int neg = v < 0 ? 1 : 0;
    return (v + ((neg << shift) - neg)) >> shift;
}

/// The value this pass codes for the quantised coefficient v at memory index m
/// of the varblock: zero outside the pass's frequency square (ncoeffs per 8x8
/// block in each direction, scaled by the covered blocks) or inside what earlier
/// exact passes covered; otherwise what the previous pass's shift left, at this
/// pass's shift.
int passValue(int v, uint m, Varblock vb) {
    uint vf, uf;   // vertical and horizontal frequency
    memFreq(m, vb.ny, vb.nx, vf, uf);
    if (uf >= vb.cx * params.ncoeffs || vf >= vb.cy * params.ncoeffs) return 0;
    if (uf < vb.cx * params.prevNcoeffs && vf < vb.cy * params.prevNcoeffs) return 0;
    if (params.prevShift != 0) v -= shiftRound0(v, params.prevShift) << params.prevShift;
    return shiftRound0(v, params.shift);
}

/// One thread per first block and channel: the pass's non-zero count of the
/// varblock (nonZeros holds the single-pass count, which quantize measured).
[shader("compute")]
[numthreads(64, 1, 1)]
void passNonZeroCount(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    if ((acs[block] & 1) == 0) return;
    const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
    const uint hs = shiftOf(params.hshift, c), vs = shiftOf(params.vshift, c);
    if (((vb.fbx >> hs) << hs) != vb.fbx || ((vb.fby >> vs) << vs) != vb.fby) return;   // not this channel's block
    uint nz = 0;
    for (uint m = 0; m < vb.size; ++m) {
        if (isLlf(vb, m)) continue;
        if (passValue(acQuant[coeffSlot(vb, c, nb, params.blocksX, m)], m, vb) != 0) ++nz;
    }
    passNonZeros[(params.pass * 3 + c) * nb + block] = nz;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void acTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    if ((acs[block] & 1) == 0) return;
    const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
    const uint hs = shiftOf(params.hshift, c), vs = shiftOf(params.vshift, c);
    if (((vb.fbx >> hs) << hs) != vb.fbx || ((vb.fby >> vs) << vs) != vb.fby) return;   // not this channel's block
    uint blockCtx = blockContextFor(c, vb.strategy);
    if (params.jpegCtx != 0) {
        // BlockCtxMap::Context with luma DC thresholds only (order 0: every block is DCT8).
        const uint numDc = bctxMap[0];
        const int qy = dcQuant[nb + block];
        uint dcIdx = 0;
        for (uint t = 0; t + 1 < numDc; ++t) {
            if (qy > int(bctxMap[2 + t])) ++dcIdx;
        }
        blockCtx = bctxMap[16 + (c < 2 ? (c ^ 1) : 2) * 13 * numDc + dcIdx];
    }
    const uint nbc = params.numBlockCtxs;
    const uint pb = params.pass * 3 * nb;
    uint nz = passNonZeros[pb + c * nb + block];
    // The non-zero count's context from the pass's group-local grid at the channel's
    // resolution (groups are 32 blocks; a subsampled channel's neighbours are 1 << shift away).
    const uint gx = (vb.fbx >> hs) % (32u >> hs), gy = (vb.fby >> vs) % (32u >> vs);
    const uint left = gx > 0 ? gridValueAt(block - (1u << hs), acs, firstBlock, passNonZeros, pb + c * nb) : 0;
    const uint top = gy > 0 ? gridValueAt(block - (params.blocksX << vs), acs, firstBlock, passNonZeros, pb + c * nb) : 0;
    const uint predicted = predictNonZeros(gx > 0, gy > 0, left, top);
    const uint nzc = nonZeroBucket(predicted) * nbc + blockCtx;
    nzCtx[pb + c * nb + block] = nzc;
    {
        uint tok, nbits, raw;
        hybridEncode(nz, tok, nbits, raw);
        uint before;
        InterlockedAdd(ctxHist[(params.ctxBase + nzc) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
    }
    const uint histoOffset = nbc * kNonZeroBuckets + kZeroDensityContextCount * blockCtx;
    const uint ord = orderOffset(kStrategyOrder[vb.strategy]);
    uint prev = nz > vb.size / 16 ? 0 : 1;
    uint k = vb.covered;
    for (; k < vb.size && nz != 0; ++k) {
        const uint idx = orders[ord + k];
        const int v = passValue(acQuant[coeffSlot(vb, c, nb, params.blocksX, idx)], idx, vb);
        const uint ctx = histoOffset + zeroDensityContextVar(nz, k, vb.covered, vb.log2Covered, prev);
        const uint value = packSigned(v);
        acTokens[pb * 64 + coeffSlot(vb, c, nb, params.blocksX, k)] = (ctx << 18) | value;
        uint tok, nbits, raw;
        hybridEncode(value, tok, nbits, raw);
        uint before;
        InterlockedAdd(ctxHist[(params.ctxBase + ctx) * kHistBins + min(tok, kHistBins - 1)], 1u, before);
        prev = v != 0 ? 1 : 0;
        nz -= prev;
    }
    numTokens[pb + c * nb + block] = k - vb.covered;
}
