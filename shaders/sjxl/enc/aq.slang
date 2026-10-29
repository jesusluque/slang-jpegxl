// Copyright (c) 2026 jesus luque.
//
// Adaptive quantisation: libjxl's InitialQuantField (enc_adaptive_quantization.cc
// AdaptiveQuantizationMap) and the quantizer's global scale, on the device.
//
//   aqDiff      one thread per 4x4 cell: the masking measure of the Y plane
//               (gamma-corrected Laplacian, squared, limited, MaskingSqrt),
//               summed over the cell -- libjxl's pre_erosion image.
//   aqMap       one thread per block: FuzzyErosion of the cells (the four
//               smallest of each 3x3 neighbourhood, weighted), then the
//               per-block modulations (ComputeMask, GammaModulation,
//               HfModulation, BlueModulation) -> the float quant field.
//   aqRawField  one thread per block: the integer field, 1..256, from the
//               quantizer's global scale.
//
// The field is read from the XYB planes before the inverse Gaborish, as
// libjxl does. The global scale is not derived from the field: libjxl's
// speeds up to squirrel set it from 0.39 / distance (enc_heuristics.cc) and
// only the butteraugli iterations of kitten and slower use the field's
// median, so the quantizer is a pure function of the distance and the host
// uploads it.
module enc_aq;

struct AqParams {
    uint  paddedWidth;
    uint  paddedHeight;
    uint  blocksX;
    uint  blocksY;
    uint  cellsX;          // paddedWidth / 4
    uint  cellsY;
    uint  uniform;         // 1: a flat field of `scale` (no adaptive quantisation)
    float distance;        // butteraugli target the field is computed for
    float scale;           // kAcQuant / distance: the field's unit
    float pad0;
};

StructuredBuffer<float>    xyb;          // 3 planes
RWStructuredBuffer<float>  preErosion;   // cellsX * cellsY
RWStructuredBuffer<float>  qf;           // nblocks: the float field
StructuredBuffer<uint>     quantizer;    // [0] globalScale [1] quantDc [2] invGlobalScale [3] globalScaleFloat (floats as bits)
RWStructuredBuffer<uint>   rawQf;        // nblocks: 1..256
ConstantBuffer<AqParams>   params;

static const float kInvLog2e = 0.6931471805599453f;
static const float kSGmul = 226.77216153508914f;
static const float kSGmul2 = 1.0f / 73.377132366608819f;
static const float kSGRetMul = kSGmul2 * 18.6580932135f * kInvLog2e;
static const float kSGVOffset = 7.7825991679894591f;

// enc_adaptive_quantization.cc RatioOfDerivativesOfCubicRootToSimpleGamma.
float ratioOfDerivatives(float v, bool invert) {
    const float kEpsilon = 1e-2f;
    v = max(v, 0.0f);
    const float numMul = kSGRetMul * 3.0f * kSGmul;
    const float vOffset = kSGVOffset * kInvLog2e + kEpsilon;
    const float denMul = kInvLog2e * kSGmul;
    const float v2 = v * v;
    const float num = numMul * v2 + kEpsilon;
    const float den = denMul * v * v2 + vOffset;
    return invert ? num / den : den / num;
}

float maskingSqrt(float v) {
    const float kLogOffset = 27.505837037000106f;
    const float kMul = 211.66567973503678f;
    return 0.25f * sqrt(v * sqrt(kMul * 1e8f) + kLogOffset);
}

uint plane() { return params.paddedWidth * params.paddedHeight; }

float yAt(int x, int y) {
    x = clamp(x, 0, int(params.paddedWidth) - 1);
    y = clamp(y, 0, int(params.paddedHeight) - 1);
    return xyb[plane() + uint(y) * params.paddedWidth + uint(x)];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aqDiff(uint3 tid: SV_DispatchThreadID) {
    const uint cell = tid.x;
    if (cell >= params.cellsX * params.cellsY) {
        return;
    }
    const int cx = int(cell % params.cellsX) * 4;
    const int cy = int(cell / params.cellsX) * 4;
    const float matchGammaOffset = 0.019f;
    const float limit = 0.2f;
    float sum = 0.0f;
    for (int dy = 0; dy < 4; ++dy) {
        for (int dx = 0; dx < 4; ++dx) {
            const int x = cx + dx, y = cy + dy;
            const float v = yAt(x, y);
            const float base = 0.25f * (yAt(x, y + 1) + yAt(x, y - 1) + yAt(x - 1, y) + yAt(x + 1, y));
            const float gammac = ratioOfDerivatives(v + matchGammaOffset, false);
            float diff = gammac * (v - base);
            diff = min(diff * diff, limit);
            sum += maskingSqrt(diff);
        }
    }
    preErosion[cell] = sum * 0.25f;
}

float cellAt(int x, int y) {
    x = clamp(x, 0, int(params.cellsX) - 1);
    y = clamp(y, 0, int(params.cellsY) - 1);
    return preErosion[uint(y) * params.cellsX + uint(x)];
}

void storeMin4(float v, inout float m0, inout float m1, inout float m2, inout float m3) {
    if (v < m3) {
        if (v < m0) {
            m3 = m2; m2 = m1; m1 = m0; m0 = v;
        } else if (v < m1) {
            m3 = m2; m2 = m1; m1 = v;
        } else if (v < m2) {
            m3 = m2; m2 = v;
        } else {
            m3 = v;
        }
    }
}

void sortSwap(inout float a, inout float b) {
    if (a > b) {
        const float t = a;
        a = b;
        b = t;
    }
}

/// FuzzyErosion at one cell: the four smallest of its 3x3 neighbourhood, weighted.
float erosion(int x, int y, float kMul[4]) {
    float m0 = cellAt(x, y), m1 = cellAt(x - 1, y), m2 = cellAt(x + 1, y), m3 = cellAt(x - 1, y - 1);
    sortSwap(m0, m1); sortSwap(m0, m2); sortSwap(m0, m3);
    sortSwap(m1, m2); sortSwap(m1, m3); sortSwap(m2, m3);
    storeMin4(cellAt(x, y - 1), m0, m1, m2, m3);
    storeMin4(cellAt(x + 1, y - 1), m0, m1, m2, m3);
    storeMin4(cellAt(x - 1, y + 1), m0, m1, m2, m3);
    storeMin4(cellAt(x, y + 1), m0, m1, m2, m3);
    storeMin4(cellAt(x + 1, y + 1), m0, m1, m2, m3);
    return kMul[0] * m0 + kMul[1] * m1 + kMul[2] * m2 + kMul[3] * m3;
}

float computeMask(float outVal) {
    const float kBase = -0.7647f;
    const float kMul4 = 9.4708735624378946f;
    const float kMul2 = 17.35036561631863f;
    const float kOffset2 = 302.59587815579727f;
    const float kMul3 = 6.7943250517376494f;
    const float kOffset3 = 3.7179635626140772f;
    const float kOffset4 = 0.25f * kOffset3;
    const float kMul0 = 0.80061762862741759f;
    const float v1 = max(outVal * kMul0, 1e-3f);
    const float v2 = 1.0f / (v1 + kOffset2);
    const float v3 = 1.0f / (v1 * v1 + kOffset3);
    const float v4 = 1.0f / (v1 * v1 + kOffset4);
    return kBase + kMul4 * v4 + kMul2 * v2 + kMul3 * v3;
}

float pixel(uint c, uint x, uint y) {
    return xyb[c * plane() + y * params.paddedWidth + x];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aqMap(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= params.blocksX * params.blocksY) {
        return;
    }
    if (params.uniform != 0) {
        qf[block] = params.scale;
        return;
    }
    const uint bx = block % params.blocksX;
    const uint by = block / params.blocksX;

    // FuzzyErosion weights for this distance.
    const float kMulBase[4] = {0.125f, 0.1f, 0.09f, 0.06f};
    const float kMulAdd[4] = {0.0f, -0.1f, -0.09f, -0.06f};
    const float kTotal = 0.29959705784054957f;
    const float mul = params.distance < 2.0f ? (2.0f - params.distance) * 0.5f : 0.0f;
    float kMul[4];
    float normSum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        kMul[i] = kMulBase[i] + mul * kMulAdd[i];
        normSum += kMul[i];
    }
    for (int i = 0; i < 4; ++i) {
        kMul[i] *= kTotal / normSum;
    }
    const int cx = int(bx) * 2, cy = int(by) * 2;
    const float aq = erosion(cx, cy, kMul) + erosion(cx + 1, cy, kMul) + erosion(cx, cy + 1, kMul) + erosion(cx + 1, cy + 1, kMul);

    // PerBlockModulations.
    const float baseLevel = 0.48f * params.scale;
    float dampen = 1.0f;
    if (params.distance >= 2.0f) {
        dampen = max(0.0f, 1.0f - (params.distance - 2.0f) / 12.0f);
    }
    const float outMul = params.scale * dampen;
    const float outAdd = (1.0f - dampen) * baseLevel;

    const uint px = bx * 8, py = by * 8;
    float val = computeMask(aq);
    {   // GammaModulation
        const float kBias = 0.16f;
        float overall = 0.0f;
        for (uint dy = 0; dy < 8; ++dy) {
            for (uint dx = 0; dx < 8; ++dx) {
                const float iny = pixel(1, px + dx, py + dy) + kBias;
                const float inx = pixel(0, px + dx, py + dy);
                overall += ratioOfDerivatives(iny - inx, true);
                overall += ratioOfDerivatives(iny + inx, true);
            }
        }
        overall *= 0.5f / 64.0f;
        val += 0.1005613337192697f * log2(overall);
    }
    float hf;
    {   // HfModulation
        const float valmin = 0.0206f;
        float sum = 0.0f;
        for (uint dy = 0; dy < 8; ++dy) {
            const uint ny = dy == 7 ? dy : dy + 1;
            for (uint dx = 0; dx < 8; ++dx) {
                const float p = pixel(1, px + dx, py + dy);
                if (dx < 7) {
                    sum += min(valmin, abs(p - pixel(1, px + dx + 1, py + dy)));
                }
                sum += min(valmin, abs(p - pixel(1, px + dx, py + ny)));
            }
        }
        hf = val + (-0.38f * sum + 0.42f);
    }
    float blue;
    {   // BlueModulation
        const float kLimit = 0.010474084867598155f;
        const float kOffset = 0.0031994768654636393f;
        float sum = 0.0f;
        for (uint dy = 0; dy < 8; ++dy) {
            for (uint dx = 0; dx < 8; ++dx) {
                const float pxv = pixel(0, px + dx, py + dy);
                const float pb = pixel(2, px + dx, py + dy);
                const float pyEff = pixel(1, px + dx, py + dy) + kOffset + abs(pxv);
                if (pb > pyEff) {
                    sum += min(pb - pyEff, kLimit);
                }
            }
        }
        if (sum >= 32.0f * kLimit) {
            sum = 64.0f * kLimit - sum;
        }
        const float kMaxLimit = 15.463398341612438f;
        sum = min(sum, kMaxLimit * kLimit);
        blue = val + sum * 0.90590804735610064f;
    }
    const float outVal = min(hf, blue);
    qf[block] = exp2(outVal * 1.442695041f) * outMul + outAdd;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aqRawField(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= params.blocksX * params.blocksY) {
        return;
    }
    const float invGlobalScale = asfloat(quantizer[2]);
    rawQf[block] = uint(clamp(int(qf[block] * invGlobalScale + 0.5f), 1, 256));
}
