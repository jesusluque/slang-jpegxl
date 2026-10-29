// Copyright (c) 2026 jesus luque.
//
// The forward DCT of every block (one thread each, three channels, float
// coefficients kept for the two consumers) and chroma-from-luma per 64x64 tile
// (libjxl enc_chroma_from_luma.cc ComputeTile with the fast least-squares
// FindBestMultiplier): the X and B coefficients of the tile's blocks, scaled
// by the quantisation they will see, are regressed on Y's, and the slope is
// stored in units of 1/84 as the AcMetadata YtoX / YtoB values.
module enc_cfl;

#include "sjxl/enc/dct.slangh"

struct CflParams {
    uint paddedWidth;
    uint paddedHeight;
    uint blocksX;
    uint blocksY;
    uint tilesX;
    uint tilesY;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float>    xyb;        // 3 planes, after the inverse Gaborish
RWStructuredBuffer<float>  coeffs;     // [c][block][k]
StructuredBuffer<float>    dequant;    // 3 * 64 dequant weights (X, Y, B)
StructuredBuffer<uint>     rawQf;      // nblocks
StructuredBuffer<uint>     quantizer;  // [3] globalScaleFloat as bits
RWStructuredBuffer<int>    cmapX;      // tiles: YtoX
RWStructuredBuffer<int>    cmapB;      // tiles: YtoB
ConstantBuffer<CflParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void dct(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    const uint nblocks = params.blocksX * params.blocksY;
    if (block >= nblocks) {
        return;
    }
    const uint px = (block % params.blocksX) * 8;
    const uint py = (block / params.blocksX) * 8;
    const uint plane = params.paddedWidth * params.paddedHeight;
    for (uint c = 0; c < 3; ++c) {
        float coef[64];
        forwardDct8x8(xyb, c * plane + py * params.paddedWidth + px, params.paddedWidth, coef);
        const uint base = (c * nblocks + block) * 64;
        for (uint k = 0; k < 64; ++k) {
            coeffs[base + k] = coef[k];
        }
    }
}

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
            const float q = globalScaleFloat * kStrangeMultiplier * float(rawQf[block]);
            // The LLF (k = 0) is zeroed; its terms would be 0 anyway.
            for (uint k = 1; k < 64; ++k) {
                const float y = coeffs[(1 * nblocks + block) * 64 + k];
                const float x = coeffs[(0 * nblocks + block) * 64 + k];
                const float b = coeffs[(2 * nblocks + block) * 64 + k];
                const float qmX = q / dequant[0 * 64 + k];
                const float qmB = q / dequant[2 * 64 + k];
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
            num += 64;
        }
    }
    cmapX[tile] = bestMultiplier(caX, cbX, num);
    cmapB[tile] = bestMultiplier(caB, cbB, num);
}
