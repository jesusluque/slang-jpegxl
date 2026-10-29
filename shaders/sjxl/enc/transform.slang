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

struct TransformParams {
    uint  paddedWidth;
    uint  paddedHeight;
    uint  blocksX;
    uint  blocksY;
    float distance;    // AdjustQuantField's mean/max mix
    uint  pad0, pad1, pad2;
};

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
RWStructuredBuffer<uint>   acTokens;      // [c][block][slot]: ctx << 18 | value, at coefficient index k's slot
RWStructuredBuffer<uint>   numTokens;     // [c][block]: coefficient tokens of the varblock (at its first block)
RWStructuredBuffer<uint>   nzCtx;         // [c][block]: the non-zero count's context
RWStructuredBuffer<uint>   ctxHist;       // kNumAcContexts * kHistBins
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

[shader("compute")]
[numthreads(64, 1, 1)]
void acTokenize(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    if ((acs[block] & 1) == 0) return;
    const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
    const uint blockCtx = blockContextFor(c, vb.strategy);
    uint nz = nonZeros[c * nb + block];
    // The non-zero count's context from the group-local grid (groups are 32 blocks).
    const uint gx = vb.fbx % 32, gy = vb.fby % 32;
    const uint left = gx > 0 ? gridValue(block - 1, acs, firstBlock, nonZeros, c, nb) : 0;
    const uint top = gy > 0 ? gridValue(block - params.blocksX, acs, firstBlock, nonZeros, c, nb) : 0;
    const uint predicted = predictNonZeros(gx > 0, gy > 0, left, top);
    const uint nzc = nonZeroBucket(predicted) * kDefaultNumBlockCtxs + blockCtx;
    nzCtx[c * nb + block] = nzc;
    {
        uint tok, nbits, raw;
        hybridEncode(nz, tok, nbits, raw);
        uint before;
        InterlockedAdd(ctxHist[nzc * kHistBins + min(tok, kHistBins - 1)], 1u, before);
    }
    const uint histoOffset = kDefaultNumBlockCtxs * kNonZeroBuckets + kZeroDensityContextCount * blockCtx;
    const uint ord = orderOffset(kStrategyOrder[vb.strategy]);
    uint prev = nz > vb.size / 16 ? 0 : 1;
    uint k = vb.covered;
    for (; k < vb.size && nz != 0; ++k) {
        const uint idx = orders[ord + k];
        const int v = acQuant[coeffSlot(vb, c, nb, params.blocksX, idx)];
        const uint ctx = histoOffset + zeroDensityContextVar(nz, k, vb.covered, vb.log2Covered, prev);
        const uint value = packSigned(v);
        acTokens[coeffSlot(vb, c, nb, params.blocksX, k)] = (ctx << 18) | value;
        uint tok, nbits, raw;
        hybridEncode(value, tok, nbits, raw);
        uint before;
        InterlockedAdd(ctxHist[ctx * kHistBins + min(tok, kHistBins - 1)], 1u, before);
        prev = v != 0 ? 1 : 0;
        nz -= prev;
    }
    numTokens[c * nb + block] = k - vb.covered;
}
