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
    uint  pad0, pad1, pad2;
};

StructuredBuffer<float>    xyb;          // 3 planes; only Y is read
StructuredBuffer<uint>     rawQf;
StructuredBuffer<float>    dequant;
StructuredBuffer<uint>     quantizer;    // [3] globalScaleFloat
StructuredBuffer<float>    sbCost;       // per 32x32 superblock
RWStructuredBuffer<float>  rowPass;      // scratch, one plane used
RWStructuredBuffer<float>  candQf;       // regions * kCandsPerRegion (0: invalid)
RWStructuredBuffer<uint>   candBits;     // fixed point
RWStructuredBuffer<uint>   candDist;
RWStructuredBuffer<uint>   candNz;
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
    candNz[tid.x] = 0;
    const bool valid = fbx + cx <= params.blocksX && fby + cy <= params.blocksY && ((params.largeMask >> fam) & 1u) != 0;
    if (!valid) { candQf[tid.x] = 0.0f; return; }
    float mx = 0.0f, mean = 0.0f;
    for (uint y = 0; y < cy; ++y) {
        for (uint x = 0; x < cx; ++x) {
            const float q = float(rawQf[(fby + y) * params.blocksX + fbx + x]);
            mx = max(mx, q);
            mean += q;
        }
    }
    mean /= float(cx * cy);
    float mixer = 1.0f;
    if (params.distance > 1.54138f) mixer = max(0.0f, 1.0f - (params.distance - 1.54138f) * 0.56391f);
    candQf[tid.x] = mx * mixer + (1.0f - mixer) * mean;
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
    if (tid.x >= n) return;
    const uint px = tid.x % params.paddedWidth, py = tid.x / params.paddedWidth;
    uint x0, y0;
    const uint cand = candidateAt(px, py, x0, y0);
    if (candQf[cand] == 0.0f) return;
    const uint nx = kFamNx[params.family];
    const uint u = px - x0;
    const uint rowBase = n + py * params.paddedWidth + x0;   // the Y plane
    float sum = 0.0f;
    for (uint x = 0; x < nx; ++x) sum += xyb[rowBase + x] * fdctBasis(u, x, nx);
    rowPass[tid.x] = sum;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void largeCols(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.paddedWidth * params.paddedHeight;
    if (tid.x >= n) return;
    const uint px = tid.x % params.paddedWidth, py = tid.x / params.paddedWidth;
    uint x0, y0;
    const uint cand = candidateAt(px, py, x0, y0);
    const float qf = candQf[cand];
    if (qf == 0.0f) return;
    const uint nx = kFamNx[params.family], ny = kFamNy[params.family];
    const uint cx = nx / 8, cy = ny / 8;
    const uint u = px - x0, v = py - y0;
    if (v < cy && u < cx) return;   // lowest frequencies: coded as DC
    float coef = 0.0f;
    for (uint y = 0; y < ny; ++y) coef += rowPass[(y0 + y) * params.paddedWidth + px] * fdctBasis(v, y, ny);
    const uint strategy = kFamStrategy[params.family];
    const uint tableBase = kTableBase[kStrategyToTable[strategy]];
    const uint size = nx * ny;
    const float quantMul = asfloat(quantizer[3]) * qf;
    const float val = coef * quantMul / dequant[tableBase + 1 * size + memIndex(v, u, ny, nx)];
    const float threshold = (v < ny / 2 && u < nx / 2) ? 0.56f : 0.62f;
    const float q = abs(val) < threshold ? 0.0f : round(val);
    const float err = val - q;
    uint before;
    InterlockedAdd(candBits[cand], uint(entropyOf(q) * kCostScale), before);
    InterlockedAdd(candDist[cand], uint(min(err * err, 64.0f) * kCostScale), before);
    if (q != 0.0f) InterlockedAdd(candNz[cand], 1u, before);
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
        cost[c] = candQf[i] != 0.0f ? kEntropyMul64 * (float(candBits[i]) / kCostScale + nzCost(candNz[i])) + params.lambda * float(candDist[i]) / kCostScale
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
