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
    float sampleScale;    // 1 / (2^bits - 1) for RGB / grey, or for the extra channel (ecToPlanes)
    uint  floatBits;      // floating point samples: their width (16, 24, 32; 0 for integers) ...
    uint  floatExpBits;   // ... and exponent bits: the ints are the float's bit patterns (int_to_float)
    uint  ecChannel;      // ecToPlanes: index in the channel list
    uint  ecPlane;        // ecToPlanes: destination plane index
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

/// dec_modular.cc int_to_float: a custom float of `floatBits` bits with `floatExpBits`
/// exponent bits, stored as an int, back to binary32.
float intToFloat(uint f) {
    const uint bits = params.floatBits, expBits = params.floatExpBits;
    if (bits == 32) return asfloat(f);
    const int expBias = (1 << (expBits - 1)) - 1;
    const uint signShift = bits - 1;
    const uint mantBits = bits - expBits - 1;
    const uint mantShift = 23 - mantBits;
    const uint signbit = f >> signShift;
    f &= (1u << signShift) - 1;
    if (f == 0) return signbit != 0 ? -0.0f : 0.0f;
    int exp = int(f >> mantBits);
    uint mantissa = f & ((1u << mantBits) - 1);
    if (exp == (1 << expBits) - 1) {   // inf / nan
        return asfloat((signbit != 0 ? 0x80000000u : 0u) | (0xFFu << 23) | (mantissa << mantShift));
    }
    mantissa <<= mantShift;
    if (exp == 0 && expBits < 8) {   // subnormal: normalise
        while ((mantissa & 0x800000u) == 0) {
            mantissa <<= 1;
            --exp;
        }
        ++exp;
        mantissa &= 0x7FFFFFu;
    }
    exp = exp - expBias + 127;
    if (exp < 0) return 0.0f;
    return asfloat((signbit != 0 ? 0x80000000u : 0u) | (uint(exp) << 23) | mantissa);
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
        xyb[c * plane + at] = params.floatBits != 0 ? intToFloat(uint(v)) : float(v) * params.sampleScale;
    }
}

/// One extra channel (integers) to its plane in `xyb` (here: the extra channel planes
/// buffer), width x height at the channel's own size, scaled by 1 / (2^bits - 1).
[shader("compute")]
[numthreads(64, 1, 1)]
void ecToPlanes(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint x = tid.x % params.width;
    const uint y = tid.x / params.width;
    const uint chansOff = arena[kGsModChansOff];
    const uint plane = params.paddedWidth * params.paddedHeight;
    xyb[params.ecPlane * plane + y * params.paddedWidth + x] = float(sample(chansOff, params.ecChannel, x, y)) * params.sampleScale;
}
