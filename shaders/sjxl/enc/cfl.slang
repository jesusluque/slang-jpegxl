// Copyright (c) 2026 jesus luque.
//
// Chroma-from-luma per 64x64 tile (libjxl enc_chroma_from_luma.cc ComputeTile
// with the fast least-squares FindBestMultiplier): the X and B coefficients of
// the tile's varblocks, scaled by the quantisation they will see, are regressed
// on Y's, and the slope is stored in units of 1/84 as the AcMetadata YtoX /
// YtoB values. One thread per tile.
module enc_cfl;

#include "sjxl/enc/ac_tokens.slangh"

struct CflParams {
    uint blocksX;
    uint blocksY;
    uint tilesX;
    uint tilesY;
};

StructuredBuffer<float>    coeffs;     // [c][block][slot]
StructuredBuffer<float>    dequant;    // every kind's tables
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     rawQf;      // the varblock's field value at its first block
StructuredBuffer<uint>     quantizer;  // [3] globalScaleFloat as bits
RWStructuredBuffer<int>    cmapX;      // tiles: YtoX
RWStructuredBuffer<int>    cmapB;      // tiles: YtoB
ConstantBuffer<CflParams>  params;

static const float kInvColorFactor = 1.0f / 84.0f;
static const float kYToBRatio = 1.0f;
static const float kDistanceMultiplierAC = 1e-9f;
static const float kStrangeMultiplier = 128.0f;

/// FindBestMultiplier's fast path on the accumulated sums, then its
/// towards-zero bias, rounding and clamp.
int bestMultiplier(float ca, float cb, uint num) {
    if (num == 0) {
        return 0;
    }
    float x = -cb / (ca + float(num) * kDistanceMultiplierAC * 0.5f);
    const float towardsZero = 2.6f;
    if (x >= towardsZero) {
        x -= towardsZero;
    } else if (x <= -towardsZero) {
        x += towardsZero;
    } else {
        x = 0.0f;
    }
    return int(clamp(round(x), -128.0f, 127.0f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cflTile(uint3 tid: SV_DispatchThreadID) {
    const uint tile = tid.x;
    if (tile >= params.tilesX * params.tilesY) {
        return;
    }
    const uint nblocks = params.blocksX * params.blocksY;
    const uint tx = tile % params.tilesX;
    const uint ty = tile / params.tilesX;
    const uint bx1 = min((tx + 1) * 8, params.blocksX);
    const uint by1 = min((ty + 1) * 8, params.blocksY);
    const float globalScaleFloat = asfloat(quantizer[3]);
    float caX = 0.0f, cbX = 0.0f, caB = 0.0f, cbB = 0.0f;
    uint num = 0;
    for (uint by = ty * 8; by < by1; ++by) {
        for (uint bx = tx * 8; bx < bx1; ++bx) {
            const uint block = by * params.blocksX + bx;
            if ((acs[block] & 1) == 0) continue;
            const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
            if (vb.fbx + vb.cx > bx1 || vb.fby + vb.cy > by1) continue;   // not wholly in the tile
            const uint tableBase = kTableBase[kStrategyToTable[vb.strategy]];
            const float q = globalScaleFloat * kStrangeMultiplier * float(rawQf[block]);
            for (uint m = 0; m < vb.size; ++m) {
                if (isLlf(vb, m)) continue;   // zeroed in libjxl: their terms are 0
                const float y = coeffs[coeffSlot(vb, 1, nblocks, params.blocksX, m)];
                const float x = coeffs[coeffSlot(vb, 0, nblocks, params.blocksX, m)];
                const float b = coeffs[coeffSlot(vb, 2, nblocks, params.blocksX, m)];
                const float qmX = q / dequant[tableBase + 0 * vb.size + m];
                const float qmB = q / dequant[tableBase + 2 * vb.size + m];
                // residual = a * mult + b, with mult in 1/84 units
                const float aX = kInvColorFactor * y * qmX;
                const float rX = 0.0f * y * qmX - x * qmX;
                caX += aX * aX;
                cbX += aX * rX;
                const float aB = kInvColorFactor * y * qmB;
                const float rB = kYToBRatio * y * qmB - b * qmB;
                caB += aB * aB;
                cbB += aB * rB;
            }
            num += vb.size;
        }
    }
    cmapX[tile] = bestMultiplier(caX, cbX, num);
    cmapB[tile] = bestMultiplier(caB, cbB, num);
}
