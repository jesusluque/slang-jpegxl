// Copyright (c) 2026 jesus luque.
//
// Noise synthesis (frame flag kNoise): libjxl dec_noise.cc and
// render_pipeline/stage_noise.cc. Three random planes at the output size,
// each group_dim x group_dim tile from its own xorshift128+ generator seeded
// by (visible frame index, nonvisible frame index, x0, y0), filled row by row
// in batches of 16 values (bits >> 9 | 0x3F800000: floats in [1, 2)). A 5x5
// convolution (0.16 * neighbours - 3.84 * centre) with mirrored borders, then
// the noise is added in XYB with a strength from an 8-point LUT of the
// intensity and the colour correlation base ratios.
module dec_noise;

StructuredBuffer<uint>    arena;    // the LUT (8 floats) at arena[kGsNoiseLutOff]; cmap base ratios in the header
RWStructuredBuffer<float> noise;    // 3 random planes, then (noiseConvolve) the convolved ones in `conv`
RWStructuredBuffer<float> conv;
RWStructuredBuffer<float> xyb;      // the frame's planes (noiseAdd modifies in place)

struct NoiseParams {
    uint width, height;               // output size
    uint paddedWidth, paddedHeight;
    uint groupDim;                    // tile size of the generators
    uint visibleFrame, nonvisibleFrame;
    uint pad0;
};
ConstantBuffer<NoiseParams> params;

static const uint kGsCflBaseX = 13;
static const uint kGsCflBaseB = 14;
static const uint kGsNoiseLutOff = 62;

uint64_t splitMix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Rng {
    uint64_t s0[8];
    uint64_t s1[8];
};

Rng rngCreate(uint seed1, uint seed2, uint seed3, uint seed4) {
    Rng r;
    r.s0[0] = splitMix64(((uint64_t(seed1) << 32) + seed2) + 0x9E3779B97F4A7C15ull);
    r.s1[0] = splitMix64(((uint64_t(seed3) << 32) + seed4) + 0x9E3779B97F4A7C15ull);
    for (uint i = 1; i < 8; ++i) {
        r.s0[i] = splitMix64(r.s0[i - 1]);
        r.s1[i] = splitMix64(r.s1[i - 1]);
    }
    return r;
}

/// Fill: 8 x 64 bits = 16 random 32-bit words, low word first.
void rngFill(inout Rng r, out uint bits[16]) {
    for (uint i = 0; i < 8; ++i) {
        uint64_t s1 = r.s0[i];
        const uint64_t s0 = r.s1[i];
        const uint64_t b = s1 + s0;
        r.s0[i] = s0;
        s1 ^= s1 << 23;
        bits[2 * i] = uint(b & 0xFFFFFFFFull);
        bits[2 * i + 1] = uint(b >> 32);
        s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
        r.s1[i] = s1;
    }
}

float bitsToFloat(uint b) { return asfloat((b >> 9) | 0x3F800000u); }

/// One thread per tile: the three random planes of the tile, in libjxl's order.
[shader("compute")]
[numthreads(1, 1, 1)]
void noiseRandom(uint3 tid: SV_DispatchThreadID) {
    const uint tilesX = (params.width + params.groupDim - 1) / params.groupDim;
    const uint tilesY = (params.height + params.groupDim - 1) / params.groupDim;
    if (tid.x >= tilesX || tid.y >= tilesY) return;
    const uint x0 = tid.x * params.groupDim;
    const uint y0 = tid.y * params.groupDim;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    Rng rng = rngCreate(params.visibleFrame, params.nonvisibleFrame, x0, y0);
    const uint plane = params.paddedWidth * params.paddedHeight;
    uint bits[16];
    for (uint c = 0; c < 3; ++c) {
        for (uint y = 0; y < h; ++y) {
            const uint row = c * plane + (y0 + y) * params.paddedWidth + x0;
            uint x = 0;
            for (; x + 16 < w; x += 16) {
                rngFill(rng, bits);
                for (uint i = 0; i < 16; ++i) noise[row + x + i] = bitsToFloat(bits[i]);
            }
            rngFill(rng, bits);
            for (uint i = 0; x + i < w && i < 16; ++i) noise[row + x + i] = bitsToFloat(bits[i]);
        }
    }
}

int mirror(int v, int size) {
    while (v < 0 || v >= size) {
        if (v < 0) v = -v - 1;
        else v = 2 * size - 1 - v;
    }
    return v;
}

float noiseAt(uint c, int x, int y) {
    const int mx = mirror(x, int(params.width));
    const int my = mirror(y, int(params.height));
    return noise[c * params.paddedWidth * params.paddedHeight + uint(my) * params.paddedWidth + uint(mx)];
}

/// ConvolveNoiseStage: 0.16 * the 24 neighbours - 3.84 * the centre.
[shader("compute")]
[numthreads(16, 16, 1)]
void noiseConvolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint plane = params.paddedWidth * params.paddedHeight;
    for (uint c = 0; c < 3; ++c) {
        float others = 0.0f;
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                if (dx == 0 && dy == 0) continue;
                others += noiseAt(c, int(tid.x) + dx, int(tid.y) + dy);
            }
        }
        conv[c * plane + tid.y * params.paddedWidth + tid.x] = others * 0.16f - 3.84f * noiseAt(c, int(tid.x), int(tid.y));
    }
}

float noiseStrength(float x) {
    const uint lut = arena[kGsNoiseLutOff];
    const float kScale = 6.0f;   // kNumNoisePoints - 2
    float scaled = max(0.0f, x * kScale);
    float fl = floor(scaled);
    float frac = scaled - fl;
    if (scaled >= kScale + 1.0f) { fl = kScale; frac = 1.0f; }
    const uint i = uint(fl);
    const float lo = asfloat(arena[lut + i]);
    const float hi = asfloat(arena[lut + i + 1]);
    const float v = lo + (hi - lo) * frac;
    return max(0.0f, min(v, 1.0f));
}

/// AddNoiseStage, in XYB.
[shader("compute")]
[numthreads(16, 16, 1)]
void noiseAdd(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = tid.y * params.paddedWidth + tid.x;
    const float vx = xyb[at], vy = xyb[plane + at], vb = xyb[2 * plane + at];
    const float inG = vy - vx, inR = vy + vx;
    const float strengthG = noiseStrength(inG * 0.5f);
    const float strengthR = noiseStrength(inR * 0.5f);
    const float norm = 0.22f;
    const float rndR = conv[at] * norm, rndG = conv[plane + at] * norm, rndC = conv[2 * plane + at] * norm;
    const float kRGCorr = 0.9921875f, kRGNCorr = 0.0078125f;
    const float redNoise = strengthR * (kRGNCorr * rndR + kRGCorr * rndC);
    const float greenNoise = strengthG * (kRGNCorr * rndG + kRGCorr * rndC);
    const float rgNoise = redNoise + greenNoise;
    const float ytox = asfloat(arena[kGsCflBaseX]);
    const float ytob = asfloat(arena[kGsCflBaseB]);
    xyb[at] = vx + ytox * rgNoise + (redNoise - greenNoise);
    xyb[plane + at] = vy + rgNoise;
    xyb[2 * plane + at] = vb + ytob * rgNoise;
}
