// Copyright (c) 2026 jesus luque.
//
// The 64-pixel transform families (DCT64X64, two DCT64X32 (32 wide, 64 tall),
// two DCT32X64 (64 wide, 32 tall)) scored per 64x64 region against the four
// 32x32 superblock layouts strategy.slang chose. Too large for one
// thread-group's memory, they go through the plane-sized rowPass buffer:
//   largeQf     one thread per candidate: validity and AdjustQuantField.
//   largeRows   one thread per pixel: the row DCT of the family's varblock.
//   largeCols   one thread per pixel: the column DCT, quantisation with the
//               candidate's field and the kind's table, and the cost
//               accumulated with atomics (the same proxy as strategy.slang).
//   mergeLarge  one thread per region: the cheapest of 64x64, two 64x32, two
//               32x64 or the superblocks' layouts; a win rewrites the region's
//               strategy map.
module enc_large;

#include "sjxl/common/transform_basis.slangh"

static const uint kCandsPerRegion = 5;   // 64x64; 64x32 left, right; 32x64 top, bottom
static const float kCostScale = 256.0f;
static const float kCostDelta = 10.833273317067883f;
static const float kZerosMul = 9.3089059022677905f;
static const float kEntropyMul64 = 2.25f;

struct LargeParams {
    uint  blocksX;
    uint  blocksY;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  regionsX;
    uint  family;      // largeRows / largeCols: 0 64x64, 1 64x32, 2 32x64
    float lambda;
    float distance;
    uint  largeMask;   // bit f: family f may be chosen
    uint  channelPass; // largeCols: 0 the Y channel (its coefficients kept in largeY), 1 X and B as chroma-from-luma residuals
    uint  tilesX;      // chroma-from-luma tiles across (64 px: a region is a tile)
    uint  threeChannels;   // 1: X and B scored too
    uint  deadZone;    // 1: quantise with the coder's dead zones (libjxl's estimate rounds)
};

StructuredBuffer<float>    xyb;          // 3 planes
StructuredBuffer<uint>     rawQf;
StructuredBuffer<float>    qf;           // per block: the float field (quant_norm16)
StructuredBuffer<int>      cmapX;        // per tile: YtoX
StructuredBuffer<int>      cmapB;        // per tile: YtoB
RWStructuredBuffer<float>  largeY;       // scratch plane: the Y coefficient of every pixel position
StructuredBuffer<float>    dequant;
StructuredBuffer<uint>     quantizer;    // [3] globalScaleFloat
StructuredBuffer<float>    sbCost;       // per 32x32 superblock
RWStructuredBuffer<float>  rowPass;      // scratch, three planes
RWStructuredBuffer<float>  candQf;       // regions * kCandsPerRegion (0: invalid)
RWStructuredBuffer<uint>   candBits;     // fixed point
RWStructuredBuffer<uint>   candDist;
RWStructuredBuffer<uint>   candNz;       // per channel: [c][candidate]
RWStructuredBuffer<uint>   acs;
RWStructuredBuffer<uint>   firstBlock;
ConstantBuffer<LargeParams> params;

static const uint kFamStrategy[3] = {18, 19, 20};
static const uint kFamNx[3] = {64, 32, 64};   // pixels
static const uint kFamNy[3] = {64, 64, 32};

/// Candidate c of a region: family, varblock origin in blocks, covered size.
void candidate(uint region, uint c, out uint fam, out uint fbx, out uint fby, out uint cx, out uint cy) {
    const uint rx = region % params.regionsX, ry = region / params.regionsX;
    fam = c == 0 ? 0 : (c <= 2 ? 1 : 2);
    cx = kFamNx[fam] / 8;
    cy = kFamNy[fam] / 8;
    fbx = rx * 8 + (fam == 1 ? (c - 1) * 4 : 0);
    fby = ry * 8 + (fam == 2 ? (c - 3) * 4 : 0);
}

float entropyOf(float q) {
    const float a = abs(q);
    return a == 0.0f ? 0.0f : kCostDelta * sqrt(a);
}

float nzCost(uint nz) {
    const uint nbits = ceilLog2Nonzero(nz + 1) + 1;
    return kZerosMul * float(ceilLog2Nonzero(nbits + 17) + nbits);
}

static const float kXWeight = 4.0f;   // EstimateEntropy's ringing weight of X: 1 + min(3, blocks / 8), 4 from 32 blocks up

uint numCandidates() { return params.regionsX * ((params.blocksY + 7) / 8) * kCandsPerRegion; }

float quantise(float val, bool lowFreq) {
    if (params.deadZone == 0) return round(val);
    const float threshold = lowFreq ? 0.56f : 0.62f;
    return abs(val) < threshold ? 0.0f : round(val);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void largeQf(uint3 tid: SV_DispatchThreadID) {
    const uint regions = params.regionsX * ((params.blocksY + 7) / 8);
    if (tid.x >= regions * kCandsPerRegion) return;
    const uint region = tid.x / kCandsPerRegion, c = tid.x % kCandsPerRegion;
    uint fam, fbx, fby, cx, cy;
    candidate(region, c, fam, fbx, fby, cx, cy);
    candBits[tid.x] = 0;
    candDist[tid.x] = 0;
    const uint numCands = regions * kCandsPerRegion;
    for (uint ch = 0; ch < 3; ++ch) candNz[ch * numCands + tid.x] = 0;
    const bool valid = fbx + cx <= params.blocksX && fby + cy <= params.blocksY && ((params.largeMask >> fam) & 1u) != 0;
    if (!valid) { candQf[tid.x] = 0.0f; return; }
    // quant_norm16 of the float field over the covered blocks.
    float sum = 0.0f;
    for (uint y = 0; y < cy; ++y) {
        for (uint x = 0; x < cx; ++x) {
            float v = qf[(fby + y) * params.blocksX + fbx + x];
            v *= v; v *= v; v *= v;
            sum += v * v;
        }
    }
    candQf[tid.x] = pow(sum / float(cx * cy), 1.0f / 16.0f);
}

/// The candidate index and varblock origin (pixels) of pixel (px, py) in the family being scored.
uint candidateAt(uint px, uint py, out uint x0, out uint y0) {
    const uint rx = px / 64, ry = py / 64;
    const uint region = ry * params.regionsX + rx;
    if (params.family == 0) { x0 = rx * 64; y0 = ry * 64; return region * kCandsPerRegion; }
    if (params.family == 1) { const uint h = (px % 64) / 32; x0 = rx * 64 + h * 32; y0 = ry * 64; return region * kCandsPerRegion + 1 + h; }
    const uint h = (py % 64) / 32;
    x0 = rx * 64;
    y0 = ry * 64 + h * 32;
    return region * kCandsPerRegion + 3 + h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void largeRows(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.paddedWidth * params.paddedHeight;
    if (tid.x >= 3 * n) return;
    const uint c = tid.x / n, pixel = tid.x % n;
    if (c != 1 && params.threeChannels == 0) return;
    const uint px = pixel % params.paddedWidth, py = pixel / params.paddedWidth;
    uint x0, y0;
    const uint cand = candidateAt(px, py, x0, y0);
    if (candQf[cand] == 0.0f) return;
    const uint nx = kFamNx[params.family];
    const uint u = px - x0;
    const uint rowBase = c * n + py * params.paddedWidth + x0;
    float sum = 0.0f;
    for (uint x = 0; x < nx; ++x) sum += xyb[rowBase + x] * fdctBasis(u, x, nx);
    rowPass[c * n + pixel] = sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void largeCols(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.paddedWidth * params.paddedHeight;
    // Pass 0: Y (n threads); pass 1: X then B (2 n threads), residuals against Y's coefficients.
    const uint c = params.channelPass == 0 ? 1 : (tid.x < n ? 0 : 2);
    const uint pixel = tid.x % n;
    if (tid.x >= (params.channelPass == 0 ? n : 2 * n)) return;
    const uint px = pixel % params.paddedWidth, py = pixel / params.paddedWidth;
    uint x0, y0;
    const uint cand = candidateAt(px, py, x0, y0);
    const float qfv = candQf[cand];
    if (qfv == 0.0f) return;
    const uint nx = kFamNx[params.family], ny = kFamNy[params.family];
    const uint cx = nx / 8, cy = ny / 8;
    const uint u = px - x0, v = py - y0;
    float coef = 0.0f;
    for (uint y = 0; y < ny; ++y) coef += rowPass[c * n + (y0 + y) * params.paddedWidth + px] * fdctBasis(v, y, ny);
    if (c == 1) largeY[pixel] = coef;
    if (v < cy && u < cx) return;   // lowest frequencies: coded as DC
    const uint strategy = kFamStrategy[params.family];
    const uint tableBase = kTableBase[kStrategyToTable[strategy]];
    const uint size = nx * ny;
    const float quantMul = qfv;   // the float field (raw value x global scale)
    const uint tile = (py / 64) * params.tilesX + px / 64;
    const float factor = c == 0 ? float(cmapX[tile]) / 84.0f : (c == 2 ? float(cmapB[tile]) / 84.0f + 1.0f : 0.0f);
    const float val = (coef - factor * (c == 1 ? 0.0f : largeY[pixel])) * quantMul / dequant[tableBase + c * size + memIndex(v, u, ny, nx)];
    const float q = quantise(val, v < ny / 2 && u < nx / 2);
    const float err = val - q;
    const float w = c == 0 ? kXWeight : 1.0f;
    uint before;
    InterlockedAdd(candBits[cand], uint(w * entropyOf(q) * kCostScale), before);
    InterlockedAdd(candDist[cand], uint(w * min(err * err, 64.0f) * kCostScale), before);
    if (q != 0.0f) InterlockedAdd(candNz[c * numCandidates() + cand], 1u, before);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mergeLarge(uint3 tid: SV_DispatchThreadID) {
    const uint regionsY = (params.blocksY + 7) / 8;
    if (tid.x >= params.regionsX * regionsY) return;
    const uint region = tid.x;
    const uint rx = region % params.regionsX, ry = region / params.regionsX;
    const float kInvalid = 1.0e30f;
    float cost[kCandsPerRegion];
    for (uint c = 0; c < kCandsPerRegion; ++c) {
        const uint i = region * kCandsPerRegion + c;
        const uint numCands = numCandidates();
        float nzBits = nzCost(candNz[numCands + i]);
        if (params.threeChannels != 0) nzBits += kXWeight * nzCost(candNz[i]) + nzCost(candNz[2 * numCands + i]);
        cost[c] = candQf[i] != 0.0f ? kEntropyMul64 * (float(candBits[i]) / kCostScale + nzBits) + params.lambda * float(candDist[i]) / kCostScale
                                    : kInvalid;
    }
    // The superblocks' layouts: the 2x2 superblocks of the region that exist.
    const uint superblocksX = (params.blocksX + 3) / 4;
    const uint superblocksY = (params.blocksY + 3) / 4;
    float small = 0.0f;
    for (uint sy = 0; sy < 2; ++sy) {
        for (uint sx = 0; sx < 2; ++sx) {
            const uint sbx = rx * 2 + sx, sby = ry * 2 + sy;
            if (sbx < superblocksX && sby < superblocksY) small += sbCost[sby * superblocksX + sbx];
        }
    }
    uint choice = 0;   // 0 keep, 1 64x64, 2 two 64x32, 3 two 32x64
    float best = small;
    if (cost[0] < best) { choice = 1; best = cost[0]; }
    if (cost[1] + cost[2] < best) { choice = 2; best = cost[1] + cost[2]; }
    if (cost[3] + cost[4] < best) { choice = 3; best = cost[3] + cost[4]; }
    if (choice == 0) return;
    const uint bx0 = rx * 8, by0 = ry * 8;
    for (uint i = 0; i < 64; ++i) {
        const uint lx = i % 8, ly = i / 8;
        const uint bx = bx0 + lx, by = by0 + ly;
        if (bx >= params.blocksX || by >= params.blocksY) continue;
        uint s, fbx, fby;
        if (choice == 1) { s = 18; fbx = bx0; fby = by0; }
        else if (choice == 2) { s = 19; fbx = bx0 + (lx / 4) * 4; fby = by0; }
        else { s = 20; fbx = bx0; fby = by0 + (ly / 4) * 4; }
        const uint first = fby * params.blocksX + fbx;
        const uint b = by * params.blocksX + bx;
        acs[b] = (s << 1) | (b == first ? 1u : 0u);
        firstBlock[b] = first;
    }
}
