// Copyright (c) 2026 jesus luque.
//
// The AC strategy of every block: one thread-group per 32x32 superblock scores
// seven transform layouts of its Y channel by a rate-distortion proxy and picks
// the cheapest hierarchy: each 16x16 quadrant is a DCT16X16, two DCT16X8
// (8 wide, 16 tall), two DCT8X16 (16 wide, 8 tall) or four DCT8; the
// superblock is a DCT32X32, two DCT32X16 (16 wide, 32 tall), two DCT16X32 (32
// wide, 16 tall) or its quadrants. Every family has 1024 coefficients, four
// per thread, computed as a separable scaled DCT from shared basis tables;
// each coefficient is quantised with the varblock's field value (the max of
// its blocks', AdjustQuantField) and the kind's dequant table, and costs
// bits (a token estimate: 0.3 for a zero, 3 + 2 log2 |q| otherwise) plus
// lambda times its squared rounding error in quant steps; the lowest
// frequencies are coded as DC and skipped. Candidates outside the frame's
// block grid are not considered.
module enc_strategy;

#include "sjxl/common/transform_basis.slangh"
#include "sjxl/enc/special.slangh"

static const uint kThreads = 256;
static const uint kNumCandidates = 41;
static const float kCostScale = 256.0f;   // fixed point of the shared accumulators
// libjxl's EstimateEntropy: cost_delta * sqrt|q| per coefficient (zeros free),
// zeros_mul * (log2 bits of the non-zero count) per block-channel, a multiplier
// per transform against ringing, and the distortion weight.
static const float kCostDelta = 10.833273317067883f;
static const float kZerosMul = 9.3089059022677905f;

struct StrategyParams {
    uint  blocksX;
    uint  blocksY;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  superblocksX;
    float lambda;      // distortion weight, in bits per squared quant step
    float distance;    // AdjustQuantField's mean/max mix
    uint  enabled;     // 0: every block DCT8
    uint  familyMask;  // bit f: family f (0 32x32, 1 32x16, 2 16x32, 3 16x16, 4 16x8, 5 8x16, 6 8x8) may be chosen
    uint  specials;    // 1: the 8x8 special transforms compete with DCT8
    uint  pad1, pad2;
};

StructuredBuffer<float>    xyb;          // 3 planes; only Y is read
StructuredBuffer<uint>     rawQf;        // per block
StructuredBuffer<float>    dequant;      // every kind's tables
StructuredBuffer<uint>     quantizer;    // [3] globalScaleFloat as bits
RWStructuredBuffer<uint>   acs;          // per block: strategy << 1 | first
RWStructuredBuffer<uint>   firstBlock;   // per block
RWStructuredBuffer<float>  sbCost;       // per superblock: the cost of the chosen layout (for the 64-pixel families)
ConstantBuffer<StrategyParams> params;

groupshared float gY[1024];
groupshared float gTmp[1024];
groupshared float gB8[64];
groupshared float gB16[256];
groupshared float gB32[1024];
groupshared uint  gBits[kNumCandidates];   // the entropy term, fixed point
groupshared uint  gDist[kNumCandidates];
groupshared uint  gNz[kNumCandidates];
groupshared float gQf[kNumCandidates];
groupshared uint  gValid[kNumCandidates];
groupshared uint  gBest8[16];          // the 8x8 strategy of every block (DCT8 or a special one)
groupshared float gBest8Cost[16];

// Families: strategy, varblock pixels, varblocks across and down, first candidate.
static const uint kFamStrategy[7] = {5, 10, 11, 4, 6, 7, 0};
static const uint kFamNx[7] = {32, 16, 32, 16, 8, 16, 8};
static const uint kFamNy[7] = {32, 32, 16, 16, 16, 8, 8};
static const uint kFamBase[7] = {0, 1, 3, 5, 9, 17, 25};

float basis(uint n, uint k, uint t) {
    if (n == 8) return gB8[k * 8 + t];
    if (n == 16) return gB16[k * 16 + t];
    return gB32[k * 32 + t];
}

/// The candidate's varblock position (in 8x8 blocks within the superblock) and size.
void candidateGeometry(uint cand, out uint fam, out uint vbx, out uint vby, out uint nx, out uint ny) {
    fam = 6;
    for (uint f = 0; f < 6; ++f) {
        if (cand < kFamBase[f + 1]) { fam = f; break; }
    }
    nx = kFamNx[fam];
    ny = kFamNy[fam];
    const uint vb = cand - kFamBase[fam];
    const uint nvx = 32 / nx;
    vbx = vb % nvx;
    vby = vb / nvx;
}

float entropyOf(float q) {
    const float a = abs(q);
    return a == 0.0f ? 0.0f : kCostDelta * sqrt(a);
}

/// The non-zero count's share: #bits of the count and #bits of that (EstimateEntropy).
float nzCost(uint nz) {
    const uint nbits = ceilLog2Nonzero(nz + 1) + 1;
    return kZerosMul * float(ceilLog2Nonzero(nbits + 17) + nbits);
}

/// EstimateEntropy's entropy_mul of a strategy, relative to DCT8's 0.8, with the
/// distance adjustments of the 8x8 search (favour DCT2X2/IDENTITY at high
/// quality, avoid the other specials at low quality).
float entropyMul(uint strategy, float distance) {
    float m = 1.0f;
    switch (strategy) {
    case 1: m = 1.0427542510634957f / 0.8f; break;
    case 2: m = 0.95f / 0.8f; break;
    case 3: m = 1.08f / 0.8f; break;
    case 12: case 13: m = 0.85931637428340035f / 0.8f; break;
    case 14: case 15: case 16: case 17: m = 0.81779489591359944f / 0.8f; break;
    case 6: case 7: m = 1.21f; break;
    case 4: m = 1.34f; break;
    case 10: case 11: m = 1.49f; break;
    case 5: m = 1.48f; break;
    case 18: case 19: case 20: m = 2.25f; break;
    default: m = 1.0f;
    }
    if ((strategy == 1 || strategy == 2) && distance < 5.0f) {
        const float w = (5.0f - distance) / 5.0f;
        m -= 0.4f * w * w;
    }
    if (strategy == 3 || (strategy >= 12 && strategy <= 17)) {
        if (distance > 4.0f) m += 0.5f * (distance < 12.0f ? 8.0f / (distance - 4.0f) : 1.0f);
    }
    return m;
}

float candidateCost(float ent, float dist, uint nz, uint strategy, float lambda, float distance) {
    return entropyMul(strategy, distance) * (ent + nzCost(nz)) + lambda * dist;
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void selectStrategies(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint tid = gtid.x;
    const uint sbx = gid.x % params.superblocksX;
    const uint sby = gid.x / params.superblocksX;
    const uint bx0 = sbx * 4, by0 = sby * 4;
    if (bx0 >= params.blocksX || by0 >= params.blocksY) return;
    const uint nblocks = params.blocksX * params.blocksY;

    if (params.enabled == 0) {
        if (tid == 0) sbCost[gid.x] = 0.0f;
        for (uint i = tid; i < 16; i += kThreads) {
            const uint bx = bx0 + i % 4, by = by0 + i / 4;
            if (bx < params.blocksX && by < params.blocksY) {
                const uint b = by * params.blocksX + bx;
                acs[b] = 1u;
                firstBlock[b] = b;
            }
        }
        return;
    }

    // Shared tables: the Y tile (0 beyond the block grid) and the DCT bases.
    const uint plane = params.paddedWidth * params.paddedHeight;
    for (uint i = tid; i < 1024; i += kThreads) {
        const uint x = bx0 * 8 + i % 32, y = by0 * 8 + i / 32;
        gY[i] = (x < params.paddedWidth && y < params.paddedHeight) ? xyb[plane + y * params.paddedWidth + x] : 0.0f;
        gB32[i] = fdctBasis(i / 32, i % 32, 32);
        if (i < 256) gB16[i] = fdctBasis(i / 16, i % 16, 16);
        if (i < 64) gB8[i] = fdctBasis(i / 8, i % 8, 8);
    }
    // Per candidate: validity and the varblock's field value (AdjustQuantField).
    if (tid < kNumCandidates) {
        uint fam, vbx, vby, nx, ny;
        candidateGeometry(tid, fam, vbx, vby, nx, ny);
        const uint cx = nx / 8, cy = ny / 8;
        const uint fbx = bx0 + vbx * cx, fby = by0 + vby * cy;
        const bool valid = fbx + cx <= params.blocksX && fby + cy <= params.blocksY && ((params.familyMask >> fam) & 1u) != 0;
        gValid[tid] = valid ? 1u : 0u;
        float mx = 0.0f, mean = 0.0f;
        if (valid) {
            for (uint y = 0; y < cy; ++y) {
                for (uint x = 0; x < cx; ++x) {
                    const float q = float(rawQf[(fby + y) * params.blocksX + fbx + x]);
                    mx = max(mx, q);
                    mean += q;
                }
            }
            mean /= float(cx * cy);
        }
        float mixer = 1.0f;
        if (params.distance > 1.54138f) mixer = max(0.0f, 1.0f - (params.distance - 1.54138f) * 0.56391f);
        gQf[tid] = cx * cy >= 4 ? mx * mixer + (1.0f - mixer) * mean : mx;
        gBits[tid] = 0;
        gDist[tid] = 0;
        gNz[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const float gsf = asfloat(quantizer[3]);
    for (uint fam = 0; fam < 7; ++fam) {
        const uint nx = kFamNx[fam], ny = kFamNy[fam];
        const uint nvx = 32 / nx;
        const uint strategy = kFamStrategy[fam];
        const uint cx = nx / 8, cy = ny / 8;
        const uint size = nx * ny;
        const uint tableBase = kTableBase[kStrategyToTable[strategy]];
        // Row pass: tmp[y][x0 + u] for every varblock of the family.
        for (uint o = tid; o < 1024; o += kThreads) {
            const uint vb = o / size, local = o % size;
            const uint x0 = (vb % nvx) * nx, y0 = (vb / nvx) * ny;
            const uint y = local / nx, u = local % nx;
            float sum = 0.0f;
            for (uint x = 0; x < nx; ++x) sum += gY[(y0 + y) * 32 + x0 + x] * basis(nx, u, x);
            gTmp[(y0 + y) * 32 + x0 + u] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
        // Column pass, quantisation and cost.
        for (uint o = tid; o < 1024; o += kThreads) {
            const uint vb = o / size, local = o % size;
            const uint cand = kFamBase[fam] + vb;
            if (gValid[cand] == 0) continue;
            const uint x0 = (vb % nvx) * nx, y0 = (vb / nvx) * ny;
            const uint v = local / nx, u = local % nx;
            if (v < cy && u < cx) continue;   // lowest frequencies: coded as DC
            float coef = 0.0f;
            for (uint y = 0; y < ny; ++y) coef += gTmp[(y0 + y) * 32 + x0 + u] * basis(ny, v, y);
            const uint m = memIndex(v, u, ny, nx);
            const float quantMul = gsf * gQf[cand];
            const float val = coef * quantMul / dequant[tableBase + 1 * size + m];
            const float threshold = (v < ny / 2 && u < nx / 2) ? 0.56f : 0.62f;
            const float q = abs(val) < threshold ? 0.0f : round(val);
            const float err = val - q;
            uint before;
            InterlockedAdd(gBits[cand], uint(entropyOf(q) * kCostScale), before);
            InterlockedAdd(gDist[cand], uint(min(err * err, 64.0f) * kCostScale), before);
            if (q != 0.0f) InterlockedAdd(gNz[cand], 1u, before);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // The 8x8 special transforms: one thread per block tries every special
    // strategy (the whole transform and its cost) and keeps the cheapest.
    if (tid < 16) {
        const uint b = tid;
        gBest8[b] = 0;
        gBest8Cost[b] = gValid[25 + b] != 0
                            ? candidateCost(float(gBits[25 + b]) / kCostScale, float(gDist[25 + b]) / kCostScale, gNz[25 + b], 0, params.lambda, params.distance)
                            : 0.0f;
        if (params.specials != 0 && gValid[25 + b] != 0) {
            const uint kSpecials[9] = {1, 2, 3, 12, 13, 14, 15, 16, 17};
            const uint x0 = (b % 4) * 8, y0 = (b / 4) * 8;
            float pix[64];
            for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) pix[y * 8 + x] = gY[(y0 + y) * 32 + x0 + x];
            const float quantMul = gsf * gQf[25 + b];
            for (uint si = 0; si < 9; ++si) {
                const uint strategy = kSpecials[si];
                float coef[64];
                forwardSpecial(strategy, pix, coef);
                const uint tableBase = kTableBase[kStrategyToTable[strategy]];
                float ent = 0.0f, dist = 0.0f;
                uint nz = 0;
                for (uint m = 1; m < 64; ++m) {
                    const float val = coef[m] * quantMul / dequant[tableBase + 1 * 64 + m];
                    const float threshold = ((m / 8) < 4 && (m % 8) < 4) ? 0.56f : 0.62f;
                    const float q = abs(val) < threshold ? 0.0f : round(val);
                    const float err = val - q;
                    ent += entropyOf(q);
                    dist += min(err * err, 64.0f);
                    if (q != 0.0f) ++nz;
                }
                const float cost = candidateCost(ent, dist, nz, strategy, params.lambda, params.distance);
                if (cost < gBest8Cost[b]) {
                    gBest8Cost[b] = cost;
                    gBest8[b] = strategy;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid != 0) return;
    // An invalid candidate never wins; a block beyond the grid costs nothing (it
    // is not coded), so the four-DCT8 option of every quadrant stays finite.
    const float kInvalid = 1.0e30f;
    float cost[kNumCandidates];
    for (uint c = 0; c < kNumCandidates; ++c) {
        uint fam, vbx, vby, nx, ny;
        candidateGeometry(c, fam, vbx, vby, nx, ny);
        cost[c] = gValid[c] != 0 ? candidateCost(float(gBits[c]) / kCostScale, float(gDist[c]) / kCostScale, gNz[c], kFamStrategy[fam],
                                                 params.lambda, params.distance)
                                 : (c >= 25 ? 0.0f : kInvalid);
    }
    for (uint b = 0; b < 16; ++b) {
        if (gValid[25 + b] != 0) cost[25 + b] = gBest8Cost[b];   // the block's best 8x8 transform
    }
    // Per quadrant: 16x16, two 16x8 (vertical pairs), two 8x16 (horizontal pairs), or four 8x8.
    uint quadChoice[4];
    float quadCost[4];
    for (uint q = 0; q < 4; ++q) {
        const uint qx = q % 2, qy = q / 2;
        const float c8 = cost[25 + (2 * qy) * 4 + 2 * qx] + cost[25 + (2 * qy) * 4 + 2 * qx + 1] +
                         cost[25 + (2 * qy + 1) * 4 + 2 * qx] + cost[25 + (2 * qy + 1) * 4 + 2 * qx + 1];
        const float c16 = cost[5 + q];
        const float cV = cost[9 + qy * 4 + 2 * qx] + cost[9 + qy * 4 + 2 * qx + 1];
        const float cH = cost[17 + (2 * qy) * 2 + qx] + cost[17 + (2 * qy + 1) * 2 + qx];
        quadChoice[q] = 0;
        quadCost[q] = c8;
        if (c16 < quadCost[q]) { quadChoice[q] = 1; quadCost[q] = c16; }
        if (cV < quadCost[q]) { quadChoice[q] = 2; quadCost[q] = cV; }
        if (cH < quadCost[q]) { quadChoice[q] = 3; quadCost[q] = cH; }
    }
    const float cQ = quadCost[0] + quadCost[1] + quadCost[2] + quadCost[3];
    const float c32 = cost[0];
    const float cLR = cost[1] + cost[2];
    const float cTB = cost[3] + cost[4];
    uint whole = 0;   // 0 quadrants, 1 32x32, 2 two 32x16, 3 two 16x32
    float best = cQ;
    if (c32 < best) { whole = 1; best = c32; }
    if (cLR < best) { whole = 2; best = cLR; }
    if (cTB < best) { whole = 3; best = cTB; }
    sbCost[gid.x] = best;

    // Write the layout: a varblock of strategy s with first block (fbx, fby).
    const uint strategies[4] = {0, 4, 6, 7};
    for (uint i = 0; i < 16; ++i) {
        const uint bx = bx0 + i % 4, by = by0 + i / 4;
        if (bx >= params.blocksX || by >= params.blocksY) continue;
        const uint b = by * params.blocksX + bx;
        uint s, fbx, fby;
        const uint lx = i % 4, ly = i / 4;
        if (whole == 1) { s = 5; fbx = bx0; fby = by0; }
        else if (whole == 2) { s = 10; fbx = bx0 + (lx / 2) * 2; fby = by0; }
        else if (whole == 3) { s = 11; fbx = bx0; fby = by0 + (ly / 2) * 2; }
        else {
            const uint q = (ly / 2) * 2 + lx / 2;
            const uint choice = quadChoice[q];
            s = strategies[choice];
            if (choice == 0) { s = gBest8[i]; fbx = bx; fby = by; }
            else if (choice == 1) { fbx = bx0 + (lx / 2) * 2; fby = by0 + (ly / 2) * 2; }
            else if (choice == 2) { fbx = bx; fby = by0 + (ly / 2) * 2; }           // 8 wide, 16 tall
            else { fbx = bx0 + (lx / 2) * 2; fby = by; }                           // 16 wide, 8 tall
        }
        const uint first = fby * params.blocksX + fbx;
        acs[b] = (s << 1) | (b == first ? 1u : 0u);
        firstBlock[b] = first;
    }
}
