// Copyright (c) 2026 jesus luque.
//
// After the DC groups: AdaptiveDCSmoothing over the DC image, and the EPF
// sigma per block (epf.cc ComputeSigma). libjxl compressed_dc.cc.
module dec_dc_finalize;

StructuredBuffer<uint>    arena;      // globals header: kGsMulDc, kGsGlobalScaleF
StructuredBuffer<float>   dc;         // 3 planes of nblocks
RWStructuredBuffer<float> dcSmooth;   // 3 planes of nblocks
StructuredBuffer<uint>    acs;
StructuredBuffer<uint>    qf;
StructuredBuffer<uint>    epfSharp;
RWStructuredBuffer<float> sigma;      // per block: 1 / sigma, or 0 when EPF is off

struct DcFinalizeParams {
    uint  blocksX;
    uint  blocksY;
    uint  skipSmoothing;
    uint  epfIters;
    float epfQuantMul;
    float sharpLut[8];
    float pad0, pad1, pad2;
};
ConstantBuffer<DcFinalizeParams> params;

static const uint kGsGlobalScaleF = 4;
static const uint kGsMulDc = 9;
static const float kW1 = 0.20345139757231578f;
static const float kW2 = 0.0334829185968739f;
static const float kW0 = 1.0f - 4.0f * (kW1 + kW2);
static const float kInvSigmaNum = -1.1715728752538099024f;

float dcAt(uint c, uint x, uint y) { return dc[c * params.blocksX * params.blocksY + y * params.blocksX + x]; }

[shader("compute")]
[numthreads(16, 16, 1)]
void smoothDc(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x, y = tid.y;
    if (x >= params.blocksX || y >= params.blocksY) return;
    const uint nblocks = params.blocksX * params.blocksY;
    const uint at = y * params.blocksX + x;
    if (params.skipSmoothing != 0 || x == 0 || y == 0 || x + 1 == params.blocksX || y + 1 == params.blocksY) {
        for (uint c = 0; c < 3; ++c) dcSmooth[c * nblocks + at] = dc[c * nblocks + at];
        return;
    }
    float mc[3], sm[3];
    float gap = 0.5f;
    for (uint c = 0; c < 3; ++c) {
        const float tl = dcAt(c, x - 1, y - 1), tc = dcAt(c, x, y - 1), tr = dcAt(c, x + 1, y - 1);
        const float ml = dcAt(c, x - 1, y), mr = dcAt(c, x + 1, y);
        const float bl = dcAt(c, x - 1, y + 1), bc = dcAt(c, x, y + 1), br = dcAt(c, x + 1, y + 1);
        mc[c] = dcAt(c, x, y);
        const float corner = (tl + tr) + (bl + br);
        const float side = (ml + mr) + (tc + bc);
        sm[c] = corner * kW2 + side * kW1 + mc[c] * kW0;
        const float dcFactor = asfloat(arena[kGsMulDc + c]);
        gap = max(gap, abs((mc[c] - sm[c]) / dcFactor));
    }
    const float factor = max(0.0f, 3.0f - 4.0f * gap);
    for (uint c = 0; c < 3; ++c) dcSmooth[c * nblocks + at] = mc[c] + (sm[c] - mc[c]) * factor;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeSigma(uint3 tid: SV_DispatchThreadID) {
    const uint block = tid.x;
    if (block >= params.blocksX * params.blocksY) return;
    if (params.epfIters == 0) {
        sigma[block] = 0.0f;
        return;
    }
    // Every covered block carries its varblock's qf (set by the DC group decode).
    const float quantScale = asfloat(arena[kGsGlobalScaleF]);
    const float sigmaQuant = params.epfQuantMul / (quantScale * float(qf[block]) * kInvSigmaNum);
    float s = sigmaQuant * params.sharpLut[min(epfSharp[block], 7u)];
    s = min(-1e-4f, s);
    sigma[block] = 1.0f / s;
}
