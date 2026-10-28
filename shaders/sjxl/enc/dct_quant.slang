// Copyright (c) 2026 jesus luque.
//
// One thread per 8x8 block: the forward DCT of its three XYB channels and
// their quantisation, in libjxl's conventions.
//
// DCT: the orthonormal 2D DCT-II scaled by 1/8 (so coefficient 0 is the block
// mean), stored coef[fx * 8 + fy] with fx the horizontal frequency -- libjxl's
// transposed raster order (cujpegxl dct_impl.cuh, checked against libjxl).
// The 1D transform is libjxl's recursive butterfly: even outputs are the
// half-size DCT of v[i] + v[N-1-i], odd outputs the half-size DCT of
// (v[i] - v[N-1-i]) / (2 cos(pi (i + 0.5) / N)) with neighbours added.
//
// Quantisation: q = round(coef * qf * globalScale / dequantWeight) with
// libjxl's dead zone (enc_group.cc thresholds), the chroma channels coded as
// residuals against the reconstructed Y (the decoder adds base_b * Y to B;
// base_x is 0). DC: q = round(dc * kInvDcQuant[c] * globalScale * quantDc).
module enc_dct_quant;

struct DctParams {
    uint  paddedWidth;
    uint  paddedHeight;
    uint  blocksX;
    uint  blocksY;
    float quantMul;      // quantField * globalScaleFloat
    float dcMul;         // globalScaleFloat * quantDc
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float>    xyb;             // 3 planes
StructuredBuffer<float>    dequant;         // 3 * 64 dequant weights (X, Y, B)
RWStructuredBuffer<int>    acQuant;         // 3 * nblocks * 64, [c][block][k]
RWStructuredBuffer<int>    dcQuant;         // 3 * nblocks
RWStructuredBuffer<uint>   nonZeros;        // 3 * nblocks: coded coefficients that are not zero
ConstantBuffer<DctParams>  params;

static const float kSqrt2 = 1.41421356237309504880f;
// 1 / (2 cos(pi (i + 0.5) / N)) for N = 4 (2 values) and N = 8 (4 values).
static const float kOdd4[2] = {0.54119610014619701f, 1.3065629648763764f};
static const float kOdd8[4] = {0.50979557910415918f, 0.60134488693504529f, 0.89997622313641557f, 2.5629154477415055f};

// In place: v <- 2 * A2 v (DCT-2 of two values, unnormalised).
void dct2(inout float v0, inout float v1) {
    const float a = v0 + v1;
    const float b = v0 - v1;
    v0 = a;
    v1 = b;
}

// In place: v <- 4 * A4 v.
void dct4(inout float v[4]) {
    float even[2] = {v[0] + v[3], v[1] + v[2]};
    dct2(even[0], even[1]);
    float odd[2] = {(v[0] - v[3]) * kOdd4[0], (v[1] - v[2]) * kOdd4[1]};
    dct2(odd[0], odd[1]);
    odd[0] = odd[0] * kSqrt2 + odd[1];
    v[0] = even[0];
    v[1] = odd[0];
    v[2] = even[1];
    v[3] = odd[1];
}

// In place: v <- 8 * A8 v.
void dct8(inout float v[8]) {
    float even[4];
    float odd[4];
    for (int i = 0; i < 4; ++i) {
        even[i] = v[i] + v[7 - i];
        odd[i] = (v[i] - v[7 - i]) * kOdd8[i];
    }
    dct4(even);
    dct4(odd);
    odd[0] = odd[0] * kSqrt2 + odd[1];
    odd[1] += odd[2];
    odd[2] += odd[3];
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = even[i];
        v[2 * i + 1] = odd[i];
    }
}

/// coef[fx * 8 + fy] of the block whose top-left pixel is (px, py) in plane c.
void forwardDct8x8(uint c, uint px, uint py, out float coef[64]) {
    const uint plane = params.paddedWidth * params.paddedHeight;
    float rows[64];   // rows[fx * 8 + r]
    for (uint r = 0; r < 8; ++r) {
        float v[8];
        const uint base = c * plane + (py + r) * params.paddedWidth + px;
        for (uint x = 0; x < 8; ++x) {
            v[x] = xyb[base + x];
        }
        dct8(v);
        for (uint fx = 0; fx < 8; ++fx) {
            rows[fx * 8 + r] = v[fx];
        }
    }
    for (uint fx = 0; fx < 8; ++fx) {
        float v[8];
        for (uint r = 0; r < 8; ++r) {
            v[r] = rows[fx * 8 + r];
        }
        dct8(v);
        for (uint fy = 0; fy < 8; ++fy) {
            coef[fx * 8 + fy] = v[fy] * (1.0f / 64.0f);
        }
    }
}

// libjxl's dead zone: a coefficient below `threshold` quant steps is zero.
int quantizeDeadzone(float x, float threshold) {
    const float a = abs(x);
    if (a < threshold) {
        return 0;
    }
    const int q = int(floor(a + 0.5f));
    return x < 0.0f ? -q : q;
}

// enc_group.cc thresholds for the speed tiers without AdjustQuantBlockAC: Y
// 0.56 in the low-frequency quadrant and 0.62 elsewhere; X and B 0.58 / 0.62.
float deadzone(uint c, uint k) {
    const bool lowQuadrant = (k & 0x24u) == 0;   // fx < 4 and fy < 4
    if (c == 1) {
        return lowQuadrant ? 0.56f : 0.62f;
    }
    return lowQuadrant ? 0.58f : 0.62f;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void dctQuant(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    const uint nblocks = params.blocksX * params.blocksY;
    if (block >= nblocks) {
        return;
    }
    const uint bx = block % params.blocksX;
    const uint by = block / params.blocksX;
    const uint px = bx * 8;
    const uint py = by * 8;

    // Y first: X and B are coded relative to its reconstruction.
    float coefY[64];
    forwardDct8x8(1, px, py, coefY);
    float yRecon[64];
    {
        uint nz = 0;
        for (uint k = 0; k < 64; ++k) {
            const float w = dequant[1 * 64 + k];
            const int q = quantizeDeadzone(coefY[k] * params.quantMul / w, deadzone(1, k));
            acQuant[(1 * nblocks + block) * 64 + k] = q;
            yRecon[k] = adjustQuantBias(1, q) * w / params.quantMul;
            if (k != 0 && q != 0) ++nz;
        }
        nonZeros[1 * nblocks + block] = nz;
    }
    // DC: coefficient 0 is the block mean. B's DC is coded against Y's
    // dequantised DC (the decoder adds base_b = 1 times Y to B).
    const float dcMulY = kInvDcQuant[1] * params.dcMul;
    const int qdcY = int(round(coefY[0] * dcMulY));
    const float yDcRecon = float(qdcY) / dcMulY;
    dcQuant[1 * nblocks + block] = qdcY;

    for (uint ci = 0; ci < 2; ++ci) {
        const uint c = ci == 0 ? 0 : 2;
        float coef[64];
        forwardDct8x8(c, px, py, coef);
        const float cfl = c == 2 ? 1.0f : 0.0f;
        uint nz = 0;
        for (uint k = 0; k < 64; ++k) {
            const float w = dequant[c * 64 + k];
            const float residual = coef[k] - cfl * yRecon[k];
            const int q = quantizeDeadzone(residual * params.quantMul / w, deadzone(c, k));
            acQuant[(c * nblocks + block) * 64 + k] = q;
            if (k != 0 && q != 0) ++nz;
        }
        nonZeros[c * nblocks + block] = nz;
        const float dcResidual = coef[0] - cfl * yDcRecon;
        dcQuant[c * nblocks + block] = int(round(dcResidual * kInvDcQuant[c] * params.dcMul));
    }
}
