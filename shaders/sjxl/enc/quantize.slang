// Copyright (c) 2026 jesus luque.
//
// One thread per varblock (its first block): quantisation of its three
// channels' coefficients in libjxl's conventions (enc_group.cc
// QuantizeRoundtripYBlockAC at the hare tier: AdjustQuantBlockAC per channel
// raises the varblock's quant where it is flat or its high frequencies would
// all round to zero, lowers it in busy blocks, tunes Y's dead zone; then
// QuantizeBlockAC).
//
// q = round(coef * qf * globalScale * qmMultiplier / dequantWeight) with
// libjxl's dead zone (the x/b_qm_scale header fields scale the chroma steps),
// the chroma channels coded as residuals against the bias-adjusted
// reconstruction of Y times the tile's chroma-from-luma factor (ytox / 84 for
// X, 1 + ytob / 84 for B), as the decoder adds it back. The lowest frequencies
// are not coded here: the DC image (dcValue, from dcFromLlf) is quantised per
// 8x8 block, q = round(dc * kInvDcQuant[c] * globalScale * quantDc), the
// chroma DC against Y's dequantised DC with the default DC factors (0 for X,
// 1 for B), which is what the decoder's DequantDC adds.
module enc_quantize;

#include "sjxl/enc/ac_tokens.slangh"

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

StructuredBuffer<float>    coeffs;      // [c][block][slot]
StructuredBuffer<float>    dcValue;     // [c][block]
StructuredBuffer<float>    dequant;     // every kind's tables (kTableBase)
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     firstBlock;
RWStructuredBuffer<uint>   rawQf;       // per block, 1..256: the varblock's at its first block, adjusted here
StructuredBuffer<uint>     quantizer;   // [1] quantDc, [3] globalScaleFloat as bits
StructuredBuffer<int>      cmapX;       // tiles
StructuredBuffer<int>      cmapB;
RWStructuredBuffer<float>  yRecon;      // [block][slot]: Y's dequantised coefficients, scratch
RWStructuredBuffer<int>    acQuant;     // [c][block][slot]
RWStructuredBuffer<int>    dcQuant;     // [c][block]
RWStructuredBuffer<uint>   nonZeros;    // [c][block]: coded coefficients that are not zero (at first blocks)
ConstantBuffer<QuantizeParams> params;

uint nblocks() { return params.blocksX * params.blocksY; }

// libjxl's dead zone: a coefficient below `threshold` quant steps is zero.
int quantizeDeadzone(float x, float threshold) {
    const float a = abs(x);
    if (a < threshold) {
        return 0;
    }
    const int q = int(floor(a + 0.5f));
    return x < 0.0f ? -q : q;
}

static const float kChromaThresholds[4] = {0.58f, 0.62f, 0.62f, 0.62f};
static const float kInitialThresholds[4] = {0.58f, 0.64f, 0.64f, 0.64f};

/// libjxl's view of a varblock's coefficient memory: rows of (8 * max(cx, cy)) columns.
void memRowCol(Varblock vb, uint m, out uint row, out uint col) {
    const uint cols = 8 * max(vb.cx, vb.cy);
    row = m / cols;
    col = m % cols;
}

/// Quadrant of memory index m: bit 0 the second half of the columns, bit 1 of the rows.
uint quadrantOf(Varblock vb, uint m) {
    uint row, col;
    memRowCol(vb, m, row, col);
    const uint rows = 8 * min(vb.cx, vb.cy);
    const uint cols = 8 * max(vb.cx, vb.cy);
    return (row >= rows / 2 ? 2u : 0u) + (col >= cols / 2 ? 1u : 0u);
}

/// AdjustQuantBlockAC for channel c: the varblock's quant (in and out) and, for Y, its dead-zone thresholds.
void adjustQuantBlock(Varblock vb, uint c, uint tableBase, float globalScaleFloat, float qmMultiplier, inout float thresholds[4],
                      inout int quant) {
    const uint partial = (1u << 1) | (1u << 2) | (1u << 3) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16) | (1u << 17);
    if (((1u << vb.strategy) & partial) != 0) return;
    const float qac = globalScaleFloat * float(quant);
    const uint xsize = max(vb.cx, vb.cy), ysize = min(vb.cx, vb.cy);   // libjxl's layout: xsize >= ysize
    if (vb.covered > 1) {
        for (int i = 0; i < 4; ++i) {
            thresholds[i] -= clamp(0.003f * float(vb.covered), 0.0f, 0.08f);
            thresholds[i] = max(thresholds[i], 0.54f);
        }
    }
    float sumHighest = 0.0f;
    float sumOfError = 0.0f;
    float sumOfVals = 0.0f;
    float hfNonZeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float hfMaxError[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint nb = nblocks();
    for (uint m = 0; m < vb.size; ++m) {
        uint row, col;
        memRowCol(vb, m, row, col);
        if (col < xsize && row < ysize) continue;   // the lowest frequencies
        const uint hfix = quadrantOf(vb, m);
        const float val = coeffs[coeffSlot(vb, c, nb, params.blocksX, m)] * (qac * qmMultiplier / dequant[tableBase + c * vb.size + m]);
        const float v = abs(val) < thresholds[hfix] ? 0.0f : round(val);
        const float error = abs(val - v);
        sumOfError += error;
        sumOfVals += abs(v);
        if (c == 1 && v == 0.0f) {
            hfMaxError[hfix] = max(hfMaxError[hfix], error);
        }
        if (v != 0.0f) {
            hfNonZeros[hfix] += abs(v);
            const bool inCorner = row >= 7 * ysize && col >= 7 * xsize;
            const bool onBorder = row == ysize * 8 - 1 || col == xsize * 8 - 1;
            const bool inLargerCorner = col >= 4 * xsize && row >= 4 * ysize;
            if (inCorner || (onBorder && inLargerCorner)) {
                sumHighest += abs(val);
            }
        }
    }
    if (c == 1 && sumOfVals * 8.0f < float(vb.covered)) {
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
    if (vb.strategy == 0 && hfNonZeros[0] + hfNonZeros[1] + hfNonZeros[2] + hfNonZeros[3] < 11.0f) {
        quant = min(quant + 1, 255);   // a flat 8x8 block: finer, against visible block boundaries
    }
    if (vb.strategy == 4 || vb.strategy == 5 || vb.strategy == 10 || vb.strategy == 11 || vb.strategy >= 18) {
        // Large transforms: a step up when the rounding error is large against the values.
        const float kMul1[4][3] = {{0.22080615753848404f, 0.45797479824262011f, 0.29859235095977965f},
                                   {0.70109486510286834f, 0.16185281305512639f, 0.14387691730035473f},
                                   {0.114985964456218638f, 0.44656840441027695f, 0.10587658215149048f},
                                   {0.46849665264409396f, 0.41239077937781954f, 0.088667407767185444f}};
        const float kMul2[4][3] = {{0.27450281941822197f, 1.1255766549984996f, 0.98950459134128388f},
                                   {0.4652168675598285f, 0.40945807983455818f, 0.36581899811751367f},
                                   {0.28034972424715715f, 0.9182653201929738f, 1.5581531543057416f},
                                   {0.26873118114033728f, 0.68863712390392484f, 1.2082185408666786f}};
        const float kQuantNormalizer = 2.2942708343284721f;
        sumOfError *= kQuantNormalizer;
        sumOfVals *= kQuantNormalizer;
        uint ix = 3;
        if (vb.strategy == 10 || vb.strategy == 11) ix = 1;
        else if (vb.strategy == 4) ix = 0;
        else if (vb.strategy == 5) ix = 2;
        const float limit = kMul1[ix][c] * float(vb.covered * 64) + kMul2[ix][c] * sumOfVals;
        int step = int(sumOfError / limit);
        step = clamp(step, 0, 2);
        if (sumOfError > limit) {
            quant = min(quant + step, 255);
        }
    }
    {   // Coarser in highly active areas.
        const float minHf = min(min(hfNonZeros[0], hfNonZeros[1]), min(hfNonZeros[2], hfNonZeros[3]));
        const int div = int(vb.covered);
        const int activity = minHf < 15.0f * float(div) ? (int(minHf) + div / 2) / div : 15;
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
    const uint nb = nblocks();
    if (block >= nb) return;
    const float globalScaleFloat = asfloat(quantizer[3]);
    const float dcMul = globalScaleFloat * float(quantizer[1]);
    // Every block has a DC: Y's, then X and B against Y's dequantised DC.
    const float dcMulY = kInvDcQuant[1] * dcMul;
    const int qdcY = int(round(dcValue[1 * nb + block] * dcMulY));
    const float yDcRecon = float(qdcY) / dcMulY;
    dcQuant[1 * nb + block] = qdcY;
    dcQuant[0 * nb + block] = int(round(dcValue[0 * nb + block] * kInvDcQuant[0] * dcMul));
    dcQuant[2 * nb + block] = int(round((dcValue[2 * nb + block] - yDcRecon) * kInvDcQuant[2] * dcMul));
    if ((acs[block] & 1) == 0) return;

    const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
    const uint tableBase = kTableBase[kStrategyToTable[vb.strategy]];
    const uint tile = (vb.fby / 8) * params.tilesX + vb.fbx / 8;

    // QuantizeRoundtripYBlockAC: each channel adjusts the varblock's quant from
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
            adjustQuantBlock(vb, c, tableBase, globalScaleFloat, multipliers[i], thresholds, q);
            if (c == 1) thresholdsY = thresholds;
            quant = max(quant, q);
        }
    }
    rawQf[block] = uint(quant);
    const float quantMul = globalScaleFloat * float(quant);

    // Chroma dead zones for larger blocks (QuantizeBlockAC).
    float chromaThresholds[4] = kChromaThresholds;
    if (vb.covered >= 4) {
        for (int i = 0; i < 4; ++i) chromaThresholds[i] = max(chromaThresholds[i] - 0.00744f * float(vb.covered), 0.5f);
    }

    // Y first: X and B are coded relative to its reconstruction.
    {
        uint nz = 0;
        for (uint m = 0; m < vb.size; ++m) {
            const uint slot = coeffSlot(vb, 1, nb, params.blocksX, m);
            const float w = dequant[tableBase + 1 * vb.size + m];
            const int q = quantizeDeadzone(coeffs[slot] * quantMul / w, thresholdsY[quadrantOf(vb, m)]);
            acQuant[slot] = q;
            yRecon[slot - 1 * nb * 64] = adjustQuantBias(1, q) * w / quantMul;
            if (q != 0 && !isLlf(vb, m)) ++nz;
        }
        nonZeros[1 * nb + block] = nz;
    }
    for (uint ci = 0; ci < 2; ++ci) {
        const uint c = ci == 0 ? 0 : 2;
        const float acFactor = c == 2 ? 1.0f + float(cmapB[tile]) * kInvColorFactor : float(cmapX[tile]) * kInvColorFactor;
        const float qmMul = quantMul * (c == 2 ? params.bQmMultiplier : params.xQmMultiplier);
        uint nz = 0;
        for (uint m = 0; m < vb.size; ++m) {
            const uint slot = coeffSlot(vb, c, nb, params.blocksX, m);
            const uint ySlot = coeffSlot(vb, 1, nb, params.blocksX, m) - 1 * nb * 64;
            const float w = dequant[tableBase + c * vb.size + m];
            const float residual = coeffs[slot] - acFactor * yRecon[ySlot];
            const int q = quantizeDeadzone(residual * qmMul / w, chromaThresholds[quadrantOf(vb, m)]);
            acQuant[slot] = q;
            if (q != 0 && !isLlf(vb, m)) ++nz;
        }
        nonZeros[c * nb + block] = nz;
    }
}
