// Copyright (c) 2026 jesus luque.
//
// One thread per 8x8 block: quantisation of its three channels' DCT
// coefficients in libjxl's conventions (enc_group.cc QuantizeRoundtripYBlockAC
// at the hare tier: AdjustQuantBlockAC per channel raises the block's quant
// where the block is flat or its high frequencies would all round to zero,
// lowers it in busy blocks, and tunes Y's dead zone; then QuantizeBlockAC).
//
// q = round(coef * qf * globalScale * qmMultiplier / dequantWeight) with
// libjxl's dead zone (the x/b_qm_scale header fields scale the chroma steps),
// the chroma channels coded as residuals against the bias-adjusted
// reconstruction of Y times the tile's chroma-from-luma factor (ytox / 84 for
// X, 1 + ytob / 84 for B), as the decoder adds it back. DC: q = round(dc *
// kInvDcQuant[c] * globalScale * quantDc), the chroma DC against Y's
// dequantised DC with the default DC factors (0 for X, 1 for B), which is
// what the decoder's DequantDC adds.
module enc_quantize;

struct QuantizeParams {
    uint  blocksX;
    uint  blocksY;
    uint  tilesX;
    float xQmMultiplier;   // 1.25 ^ (x_qm_scale - 2): the decoder's X steps are that much finer
    float bQmMultiplier;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

StructuredBuffer<float>    coeffs;      // [c][block][k]
StructuredBuffer<float>    dequant;     // 3 * 64 dequant weights (X, Y, B)
RWStructuredBuffer<uint>   rawQf;       // nblocks, 1..256: read, adjusted per block, written back
StructuredBuffer<uint>     quantizer;   // [1] quantDc, [3] globalScaleFloat as bits
StructuredBuffer<int>      cmapX;       // tiles
StructuredBuffer<int>      cmapB;
RWStructuredBuffer<int>    acQuant;     // 3 * nblocks * 64, [c][block][k]
RWStructuredBuffer<int>    dcQuant;     // 3 * nblocks
RWStructuredBuffer<uint>   nonZeros;    // 3 * nblocks: coded coefficients that are not zero
ConstantBuffer<QuantizeParams> params;

// libjxl's dead zone: a coefficient below `threshold` quant steps is zero.
int quantizeDeadzone(float x, float threshold) {
    const float a = abs(x);
    if (a < threshold) {
        return 0;
    }
    const int q = int(floor(a + 0.5f));
    return x < 0.0f ? -q : q;
}

// enc_group.cc: X and B dead zones are 0.58 in the low-frequency quadrant
// and 0.62 elsewhere; Y's start at 0.58 / 0.64 and AdjustQuantBlockAC tunes them.
static const float kChromaThresholds[4] = {0.58f, 0.62f, 0.62f, 0.62f};
static const float kInitialThresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};

/// Quadrant of coefficient k: bit 0 the horizontal half, bit 1 the vertical.
uint quadrant(uint k) {
    const uint y = k / 8, x = k % 8;
    return (y >= 4 ? 2u : 0u) + (x >= 4 ? 1u : 0u);
}

/// AdjustQuantBlockAC for an 8x8 DCT block of channel c: the block's quant
/// (in and out) and, for Y, its dead-zone thresholds.
void adjustQuantBlock(uint c, uint base, float globalScaleFloat, float qmMultiplier, inout float thresholds[4], inout int quant) {
    const float qac = globalScaleFloat * float(quant);
    float sumHighest = 0.0f;
    float sumOfVals = 0.0f;
    float hfNonZeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float hfMaxError[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 1; k < 64; ++k) {
        const uint y = k / 8, x = k % 8;
        const uint hfix = quadrant(k);
        const float val = coeffs[base + k] * (qac * qmMultiplier / dequant[c * 64 + k]);
        const float v = abs(val) < thresholds[hfix] ? 0.0f : round(val);
        const float error = abs(val - v);
        sumOfVals += abs(v);
        if (c == 1 && v == 0.0f) {
            hfMaxError[hfix] = max(hfMaxError[hfix], error);
        }
        if (v != 0.0f) {
            hfNonZeros[hfix] += abs(v);
            const bool inCorner = y >= 7 && x >= 7;
            const bool onBorder = y == 7 || x == 7;
            const bool inLargerCorner = x >= 4 && y >= 4;
            if (inCorner || (onBorder && inLargerCorner)) {
                sumHighest += abs(val);
            }
        }
    }
    if (c == 1 && sumOfVals * 8.0f < 1.0f) {
        const float kLimit = 0.46f;
        const int origQuant = quant;
        int newQuant = quant;
        for (int i = 1; i < 4; ++i) {
            if (hfNonZeros[i] == 0.0f && hfMaxError[i] > kLimit) {
                newQuant = origQuant + 1;
                break;
            }
        }
        quant = newQuant;
        const float ratio = 0.9999f * float(newQuant) / float(origQuant);
        if (hfNonZeros[3] == 0.0f && hfMaxError[3] > kLimit) {
            thresholds[3] = ratio * hfMaxError[3];
        } else if ((hfNonZeros[1] == 0.0f && hfMaxError[1] > kLimit) || (hfNonZeros[2] == 0.0f && hfMaxError[2] > kLimit)) {
            thresholds[1] = ratio * max(hfMaxError[1], hfMaxError[2]);
            thresholds[2] = thresholds[1];
        } else if (hfNonZeros[0] == 0.0f && hfMaxError[0] > kLimit) {
            thresholds[0] = ratio * hfMaxError[0];
        }
    }
    {   // High-frequency patterns with no medium-frequency masking.
        const float all = hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3] + 1.0f;
        const float mul[3] = {70.0f, 30.0f, 60.0f};
        if (mul[c] * sumHighest >= all) {
            quant += int(mul[c] * sumHighest / all);
            quant = min(quant, 255);
        }
    }
    // A flat 8x8 block: a little finer, against visible block boundaries.
    if (hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3] < 11.0f) {
        quant = min(quant + 1, 255);
    }
    {   // Coarser in highly active areas.
        const float minHf = min(min(hfNonZeros[0], hfNonZeros[1]), min(hfNonZeros[2], hfNonZeros[3]));
        const int activity = minHf < 15.0f ? int(minHf) : 15;
        int qp = quant - activity;
        if (c == 1) {
            for (int i = 1; i < 4; ++i) {
                thresholds[i] += 0.01f * float(activity);
            }
        }
        qp = max(qp, max(4, quant / 2));
        quant = qp;
    }
}

// quantizer.h kDefaultQuantBias: what the decoder reconstructs q as, in steps.
static const float kQuantBias[4] = {1.0f - 0.05465007330715401f, 1.0f - 0.07005449891748593f,
                                    1.0f - 0.049935103337343655f, 0.145f};
float adjustQuantBias(uint c, int q) {
    if (q == 0) return 0.0f;
    if (q == 1) return kQuantBias[c];
    if (q == -1) return -kQuantBias[c];
    const float f = float(q);
    return f - kQuantBias[3] / f;
}

static const float kInvDcQuant[3] = {4096.0f, 512.0f, 256.0f};
static const float kInvColorFactor = 1.0f / 84.0f;

[shader("compute")]
[numthreads(64, 1, 1)]
void quantize(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    const uint nblocks = params.blocksX * params.blocksY;
    if (block >= nblocks) {
        return;
    }
    const uint bx = block % params.blocksX;
    const uint by = block / params.blocksX;
    const uint tile = (by / 8) * params.tilesX + bx / 8;
    const float globalScaleFloat = asfloat(quantizer[3]);
    const float dcMul = globalScaleFloat * float(quantizer[1]);

    // QuantizeRoundtripYBlockAC: each channel adjusts the block's quant from
    // the field's value; the largest wins. Y's dead zones come out of its pass.
    const int quantOrig = int(rawQf[block]);
    int quant = quantOrig;
    float thresholdsY[4] = kInitialThresholds;
    {
        const uint channels[3] = {1, 0, 2};
        const float multipliers[3] = {1.0f, params.xQmMultiplier, params.bQmMultiplier};
        for (uint i = 0; i < 3; ++i) {
            const uint c = channels[i];
            float thresholds[4] = kInitialThresholds;
            int q = quantOrig;
            adjustQuantBlock(c, (c * nblocks + block) * 64, globalScaleFloat, multipliers[i], thresholds, q);
            if (c == 1) {
                thresholdsY = thresholds;
            }
            quant = max(quant, q);
        }
    }
    rawQf[block] = uint(quant);
    const float quantMul = globalScaleFloat * float(quant);

    // Y first: X and B are coded relative to its reconstruction.
    float yRecon[64];
    float yDc;
    {
        const uint base = (1 * nblocks + block) * 64;
        uint nz = 0;
        for (uint k = 0; k < 64; ++k) {
            const float w = dequant[1 * 64 + k];
            const int q = quantizeDeadzone(coeffs[base + k] * quantMul / w, thresholdsY[quadrant(k)]);
            acQuant[base + k] = q;
            yRecon[k] = adjustQuantBias(1, q) * w / quantMul;
            if (k != 0 && q != 0) ++nz;
        }
        nonZeros[1 * nblocks + block] = nz;
        yDc = coeffs[base];
    }
    const float dcMulY = kInvDcQuant[1] * dcMul;
    const int qdcY = int(round(yDc * dcMulY));
    const float yDcRecon = float(qdcY) / dcMulY;
    dcQuant[1 * nblocks + block] = qdcY;

    for (uint ci = 0; ci < 2; ++ci) {
        const uint c = ci == 0 ? 0 : 2;
        const uint base = (c * nblocks + block) * 64;
        const float acFactor = c == 2 ? 1.0f + float(cmapB[tile]) * kInvColorFactor : float(cmapX[tile]) * kInvColorFactor;
        const float dcFactor = c == 2 ? 1.0f : 0.0f;
        const float qmMul = quantMul * (c == 2 ? params.bQmMultiplier : params.xQmMultiplier);
        uint nz = 0;
        for (uint k = 0; k < 64; ++k) {
            const float w = dequant[c * 64 + k];
            const float residual = coeffs[base + k] - acFactor * yRecon[k];
            const int q = quantizeDeadzone(residual * qmMul / w, kChromaThresholds[quadrant(k)]);
            acQuant[base + k] = q;
            if (k != 0 && q != 0) ++nz;
        }
        nonZeros[c * nblocks + block] = nz;
        const float dcResidual = coeffs[base] - dcFactor * yDcRecon;
        dcQuant[c * nblocks + block] = int(round(dcResidual * kInvDcQuant[c] * dcMul));
    }
}
