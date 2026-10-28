// Copyright (c) 2026 jesus luque.
//
// The decoded modular channels (integers in the arena) to the three float
// planes the filters and the output stage read: libjxl
// ModularImageToDecodedRect. RGB or grey samples scale by 1 / (2^bits - 1);
// an XYB modular frame holds Y, X, B with the DC dequant factors as scale and
// B stored relative to Y. One thread per pixel.
module dec_modular_out;

StructuredBuffer<uint>    arena;
RWStructuredBuffer<float> xyb;   // 3 planes of paddedWidth * paddedHeight

struct ModularOutParams {
    uint  width;
    uint  height;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  xybMode;        // 1: Y, X, B channels with the DC dequant factors (kGsDcQuant)
    float sampleScale;    // 1 / (2^bits - 1) for RGB / grey
    uint  pad0, pad1;
};
ConstantBuffer<ModularOutParams> params;

static const uint kGsDcQuant = 6;
static const uint kGsModChansOff = 48;
static const uint kGsModColorChannels = 52;
static const uint kChanWords = 8;
static const uint kChanData = 4;
static const uint kChanStride = 5;

int sample(uint chansOff, uint c, uint x, uint y) {
    const uint o = chansOff + c * kChanWords;
    return int(arena[arena[o + kChanData] + y * arena[o + kChanStride] + x]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void modularToPlanes(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint x = tid.x % params.width;
    const uint y = tid.x / params.width;
    const uint chansOff = arena[kGsModChansOff];
    const uint colorChannels = arena[kGsModColorChannels];
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = y * params.paddedWidth + x;
    if (params.xybMode != 0) {
        const int sy = sample(chansOff, 0, x, y);
        const int sx = sample(chansOff, 1, x, y);
        const int sb = sample(chansOff, 2, x, y);
        xyb[at] = float(sx) * asfloat(arena[kGsDcQuant + 0]);
        xyb[plane + at] = float(sy) * asfloat(arena[kGsDcQuant + 1]);
        xyb[2 * plane + at] = float(sb + sy) * asfloat(arena[kGsDcQuant + 2]);
        return;
    }
    for (uint c = 0; c < 3; ++c) {
        const int v = sample(chansOff, colorChannels == 1 ? 0 : c, x, y);
        xyb[c * plane + at] = float(v) * params.sampleScale;
    }
}
