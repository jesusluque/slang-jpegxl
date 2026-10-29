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
    float constantInvSigma;   // modular frames: kInvSigmaNum / epf_sigma_for_modular for every block
    uint  useConstantSigma;
    uint  dcFramePaddedWidth; // kUseDcFrame: the DC frame's planes, one pixel per block
    uint4 hshift;             // chroma subsampling per channel: smoothing runs per channel on its own grid
    uint4 vshift;
};
ConstantBuffer<DcFinalizeParams> params;

StructuredBuffer<float> dcFrame;   // a DC frame's 3 planes (kUseDcFrame), dcFramePaddedWidth * height

/// kUseDcFrame: the DC image is the DC frame's pixels (passes_state.cc: shared->dc = &dc_frames[level]),
/// written to dcSmooth since no smoothing runs (FinalizeDC skips it).
[shader("compute")]
[numthreads(64, 1, 1)]
void dcFromFrame(uint3 tid: SV_DispatchThreadID) {
    const uint nblocks = params.blocksX * params.blocksY;
    if (tid.x >= nblocks) return;
    const uint bx = tid.x % params.blocksX;
    const uint by = tid.x / params.blocksX;
    const uint plane = params.dcFramePaddedWidth * ((params.blocksY + 7) / 8 * 8);
    for (uint c = 0; c < 3; ++c) dcSmooth[c * nblocks + tid.x] = dcFrame[c * plane + by * params.dcFramePaddedWidth + bx];
}

static const uint kGsGlobalScaleF = 4;
static const uint kGsMulDc = 9;
static const float kW1 = 0.20345139757231578f;
static const float kW2 = 0.0334829185968739f;
static const float kW0 = 1.0f - 4.0f * (kW1 + kW2);
static const float kInvSigmaNum = -1.1715728752538099024f;

float dcAt(uint c, uint x, uint y) { return dc[c * params.blocksX * params.blocksY + y * params.blocksX + x]; }
uint hsOf(uint c) { return c == 0 ? params.hshift.x : c == 1 ? params.hshift.y : params.hshift.z; }
uint vsOf(uint c) { return c == 0 ? params.vshift.x : c == 1 ? params.vshift.y : params.vshift.z; }
/// A subsampled channel's value at its grid position (sx, sy): the co-located full-grid block.
float dcAtC(uint c, uint sx, uint sy) { return dcAt(c, sx << hsOf(c), sy << vsOf(c)); }

[shader("compute")]
[numthreads(16, 16, 1)]
void smoothDc(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x, y = tid.y;
    if (x >= params.blocksX || y >= params.blocksY) return;
    const uint nblocks = params.blocksX * params.blocksY;
    const uint at = y * params.blocksX + x;
    const bool subsampled = (params.hshift.x | params.hshift.y | params.hshift.z | params.vshift.x | params.vshift.y | params.vshift.z) != 0;
    if (subsampled && params.skipSmoothing == 0) {
        // compressed_dc.cc, the per-channel variant: each channel on its own grid.
        for (uint c = 0; c < 3; ++c) {
            const uint hs = hsOf(c), vs = vsOf(c);
            if ((x & ((1u << hs) - 1)) != 0 || (y & ((1u << vs) - 1)) != 0) continue;   // not a position of this channel
            const uint sx = x >> hs, sy = y >> vs;
            const uint gw = params.blocksX >> hs, gh = params.blocksY >> vs;
            if (gw <= 2 || gh <= 2 || sx == 0 || sy == 0 || sx + 1 == gw || sy + 1 == gh) {
                dcSmooth[c * nblocks + at] = dc[c * nblocks + at];
                continue;
            }
            const float tl = dcAtC(c, sx - 1, sy - 1), tc = dcAtC(c, sx, sy - 1), tr = dcAtC(c, sx + 1, sy - 1);
            const float ml = dcAtC(c, sx - 1, sy), mr = dcAtC(c, sx + 1, sy);
            const float bl = dcAtC(c, sx - 1, sy + 1), bc = dcAtC(c, sx, sy + 1), br = dcAtC(c, sx + 1, sy + 1);
            const float mc = dcAtC(c, sx, sy);
            const float sm = ((tl + tr) + (bl + br)) * kW2 + ((ml + mr) + (tc + bc)) * kW1 + mc * kW0;
            const float gap = max(0.5f, abs((mc - sm) / asfloat(arena[kGsMulDc + c])));
            const float factor = max(0.0f, 3.0f - 4.0f * gap);
            dcSmooth[c * nblocks + at] = mc + (sm - mc) * factor;
        }
        return;
    }
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
    if (params.useConstantSigma != 0) {
        sigma[block] = params.constantInvSigma;
        return;
    }
    // Every covered block carries its varblock's qf (set by the DC group decode).
    const float quantScale = asfloat(arena[kGsGlobalScaleF]);
    const float sigmaQuant = params.epfQuantMul / (quantScale * float(qf[block]) * kInvSigmaNum);
    float s = sigmaQuant * params.sharpLut[min(epfSharp[block], 7u)];
    s = min(-1e-4f, s);
    sigma[block] = 1.0f / s;
}
