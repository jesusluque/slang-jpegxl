// Copyright (c) 2026 jesus luque.
//
// From quantised coefficients to XYB pixels: dequantisation with the quant
// bias and chroma-from-luma (one thread per coefficient), then the lowest
// frequencies from the DC image and the inverse transform of each varblock
// (one thread per varblock and channel). libjxl dec_group.cc DequantBlock,
// dec_transforms-inl.h TransformToPixels / LowestFrequenciesFromDC.
//
// Coefficient slots are those of acgroup.slangh: coefficient k of a varblock
// with first block B lives at slot (c, B's covered block k / 64, k % 64).
module dec_transforms;

StructuredBuffer<uint>     arena;      // globals header
StructuredBuffer<int>      coeffs;     // 3 * nblocks * 64 quantised
RWStructuredBuffer<float>  dequantF;   // same layout, dequantised and chroma-corrected
StructuredBuffer<float>    dequant;    // the matrices (dec_tables)
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     qf;
StructuredBuffer<uint>     firstBlock;
StructuredBuffer<int>      ytox;
StructuredBuffer<int>      ytob;
StructuredBuffer<float>    dcSmooth;   // 3 planes of nblocks
RWStructuredBuffer<float>  xyb;        // 3 planes of paddedWidth * paddedHeight
RWStructuredBuffer<uint>   flags;      // [0] |= 1 when a strategy beyond 32 px was met

struct TransformParams {
    uint  blocksX;
    uint  blocksY;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  tilesX;
    float xDm;          // 1.25 ^ (2 - x_qm_scale)
    float bDm;
    uint  pad0;
    float quantBias[4];
};
ConstantBuffer<TransformParams> params;

static const uint kGsInvGlobalScale = 3;
static const uint kGsCflBaseX = 13;
static const uint kGsCflBaseB = 14;
static const uint kGsCflColorScale = 19;
static const uint kStrategyToTable[27] = {0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16};
static const uint kCoveredX[27] = {1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32};
static const uint kCoveredY[27] = {1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16};
static const uint kTableBase[18] = {0, 192, 384, 576, 768, 1536, 4608, 4992, 5760, 7296, 7488, 7680, 19968, 26112, 75264, 99840, 296448, 394752};
static const float kSqrt2 = 1.41421356237309504880f;

float adjustQuantBias(uint c, int q) {
    if (q == 0) return 0.0f;
    if (q == 1) return params.quantBias[c];
    if (q == -1) return -params.quantBias[c];
    const float f = float(q);
    return f - params.quantBias[3] / f;
}

uint nblocks() { return params.blocksX * params.blocksY; }

/// Slot of coefficient k of the varblock whose first block is `first` (cx covered columns).
uint slotOf(uint c, uint first, uint cx, uint k) {
    const uint q = k >> 6;
    const uint block = first + (q / cx) * params.blocksX + (q % cx);
    return (c * nblocks() + block) * 64 + (k & 63);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void dequantCoeffs(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    const uint nb = nblocks();
    if (s >= 3 * nb * 64) return;
    const uint c = s / (nb * 64);
    const uint rem = s - c * nb * 64;
    const uint block = rem >> 6;
    const uint i = rem & 63;
    const uint first = firstBlock[block];
    const uint strategy = acs[first] >> 1;
    if (strategy >= 27) { dequantF[s] = 0.0f; return; }
    const uint cx = kCoveredX[strategy];
    const uint cy = kCoveredY[strategy];
    const uint fbx = first % params.blocksX;
    const uint fby = first / params.blocksX;
    const uint bx = block % params.blocksX;
    const uint by = block / params.blocksX;
    const uint q = (by - fby) * cx + (bx - fbx);
    const uint k = q * 64 + i;
    const uint size = 64 * cx * cy;
    const uint tableBase = kTableBase[kStrategyToTable[strategy]];
    const float mul = asfloat(arena[kGsInvGlobalScale]) / float(qf[first]);
    const float dm = c == 0 ? params.xDm : (c == 2 ? params.bDm : 1.0f);
    float v = adjustQuantBias(c, coeffs[s]) * dequant[tableBase + c * size + k] * mul * dm;
    if (c != 1) {
        const float vy = adjustQuantBias(1, coeffs[(1 * nb + block) * 64 + i]) * dequant[tableBase + 1 * size + k] * mul;
        const uint tile = (fby / 8) * params.tilesX + fbx / 8;
        const float scale = asfloat(arena[kGsCflColorScale]);
        const float factor = c == 0 ? asfloat(arena[kGsCflBaseX]) + float(ytox[tile]) * scale
                                    : asfloat(arena[kGsCflBaseB]) + float(ytob[tile]) * scale;
        v += vy * factor;
    }
    dequantF[s] = v;
}

// --- inverse transforms --------------------------------------------------------------

/// 1D inverse of libjxl's scaled DCT: x[t] = sum_k c(k) X[k] cos(pi (t + 1/2) k / N), c(0) = 1, else sqrt 2.
float idctBasis(uint k, uint t, uint n) {
    const float c = k == 0 ? 1.0f : kSqrt2;
    return c * cos(3.14159265358979323846f * (float(t) + 0.5f) * float(k) / float(n));
}

/// libjxl's forward scaled DCT basis: g(0) = 1/N, g(k) = sqrt(2)/N.
float fdctBasis(uint k, uint t, uint n) {
    const float g = (k == 0 ? 1.0f : kSqrt2) / float(n);
    return g * cos(3.14159265358979323846f * (float(t) + 0.5f) * float(k) / float(n));
}

/// Coefficient memory of an ny x nx (pixel) transform (dct-inl.h ComputeScaledDCT): a
/// wide block stores [vertical][horizontal] frequencies, a tall or square one
/// [horizontal][vertical]; both have the shorter axis first.
uint memIndex(uint v, uint u, uint ny, uint nx) {
    return ny < nx ? v * nx + u : u * ny + v;
}

// DCTResampleScales<cy, 8 cy> (the decoder's set, values above 1); the encoder's
// DCFromLowestFrequencies uses the inverse set <8 cy, cy>.
static const float kResample2[2] = {1.0f, 1.108937353592731823f};
static const float kResample4[4] = {1.0f, 1.025760096781116015f, 1.108937353592731823f, 1.270559368765487251f};
float resampleScale(uint n, uint k) {
    if (n == 2) return kResample2[k];
    if (n == 4) return kResample4[k];
    return 1.0f;
}

/// Generic separable IDCT of `coef` (memory layout above) into pixels at
/// (px, py) of channel c.
void idctToPixels(float coef[1024], uint ny, uint nx, uint c, uint px, uint py) {
    float tmp[1024];
    for (uint v = 0; v < ny; ++v) {
        for (uint tx = 0; tx < nx; ++tx) {
            float sum = 0.0f;
            for (uint u = 0; u < nx; ++u) sum += coef[memIndex(v, u, ny, nx)] * idctBasis(u, tx, nx);
            tmp[v * nx + tx] = sum;
        }
    }
    const uint plane = params.paddedWidth * params.paddedHeight;
    for (uint ty = 0; ty < ny; ++ty) {
        for (uint tx = 0; tx < nx; ++tx) {
            float sum = 0.0f;
            for (uint v = 0; v < ny; ++v) sum += tmp[v * nx + tx] * idctBasis(v, ty, ny);
            xyb[c * plane + (py + ty) * params.paddedWidth + px + tx] = sum;
        }
    }
}

/// A small transform held in `coef` (same conventions), written to pixels.
void idctSmall(float small[32], uint ny, uint nx, uint c, uint px, uint py) {
    float coef[1024];
    for (uint i = 0; i < ny * nx; ++i) coef[i] = small[i];
    idctToPixels(coef, ny, nx, c, px, py);
}

static const float kAfvBasis[16][16] = {
    {0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f},
    {0.876902929799142f, 0.2206518106944235f, -0.10140050393753763f, -0.1014005039375375f, 0.2206518106944236f, -0.10140050393753777f, -0.10140050393753772f, -0.10140050393753763f, -0.10140050393753758f, -0.10140050393753769f, -0.1014005039375375f, -0.10140050393753768f, -0.10140050393753768f, -0.10140050393753759f, -0.10140050393753763f, -0.10140050393753741f},
    {0.0f, 0.0f, 0.40670075830260755f, 0.44444816619734445f, 0.0f, 0.0f, 0.19574399372042936f, 0.2929100136981264f, -0.40670075830260716f, -0.19574399372042872f, 0.0f, 0.11379074460448091f, -0.44444816619734384f, -0.29291001369812636f, -0.1137907446044814f, 0.0f},
    {0.0f, 0.0f, -0.21255748058288748f, 0.3085497062849767f, 0.0f, 0.4706702258572536f, -0.1621205195722993f, 0.0f, -0.21255748058287047f, -0.16212051957228327f, -0.47067022585725277f, -0.1464291867126764f, 0.3085497062849487f, 0.0f, -0.14642918671266536f, 0.4251149611657548f},
    {0.0f, -0.7071067811865474f, 0.0f, 0.0f, 0.7071067811865476f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-0.4105377591765233f, 0.6235485373547691f, -0.06435071657946274f, -0.06435071657946266f, 0.6235485373547694f, -0.06435071657946284f, -0.0643507165794628f, -0.06435071657946274f, -0.06435071657946272f, -0.06435071657946279f, -0.06435071657946266f, -0.06435071657946277f, -0.06435071657946277f, -0.06435071657946273f, -0.06435071657946274f, -0.0643507165794626f},
    {0.0f, 0.0f, -0.4517556589999482f, 0.15854503551840063f, 0.0f, -0.04038515160822202f, 0.0074182263792423875f, 0.39351034269210167f, -0.45175565899994635f, 0.007418226379244351f, 0.1107416575309343f, 0.08298163094882051f, 0.15854503551839705f, 0.3935103426921022f, 0.0829816309488214f, -0.45175565899994796f},
    {0.0f, 0.0f, -0.304684750724869f, 0.5112616136591823f, 0.0f, 0.0f, -0.290480129728998f, -0.06578701549142804f, 0.304684750724884f, 0.2904801297290076f, 0.0f, -0.23889773523344604f, -0.5112616136592012f, 0.06578701549142545f, 0.23889773523345467f, 0.0f},
    {0.0f, 0.0f, 0.3017929516615495f, 0.25792362796341184f, 0.0f, 0.16272340142866204f, 0.09520022653475037f, 0.0f, 0.3017929516615503f, 0.09520022653475055f, -0.16272340142866173f, -0.35312385449816297f, 0.25792362796341295f, 0.0f, -0.3531238544981624f, -0.6035859033230976f},
    {0.0f, 0.0f, 0.40824829046386274f, 0.0f, 0.0f, 0.0f, 0.0f, -0.4082482904638628f, -0.4082482904638635f, 0.0f, 0.0f, -0.40824829046386296f, 0.0f, 0.4082482904638634f, 0.408248290463863f, 0.0f},
    {0.0f, 0.0f, 0.1747866975480809f, 0.0812611176717539f, 0.0f, 0.0f, -0.3675398009862027f, -0.307882213957909f, -0.17478669754808135f, 0.3675398009862011f, 0.0f, 0.4826689115059883f, -0.08126111767175039f, 0.30788221395790305f, -0.48266891150598584f, 0.0f},
    {0.0f, 0.0f, -0.21105601049335784f, 0.18567180916109802f, 0.0f, 0.0f, 0.49215859013738733f, -0.38525013709251915f, 0.21105601049335806f, -0.49215859013738905f, 0.0f, 0.17419412659916217f, -0.18567180916109904f, 0.3852501370925211f, -0.1741941265991621f, 0.0f},
    {0.0f, 0.0f, -0.14266084808807264f, -0.3416446842253372f, 0.0f, 0.7367497537172237f, 0.24627107722075148f, -0.08574019035519306f, -0.14266084808807344f, 0.24627107722075137f, 0.14883399227113567f, -0.04768680350229251f, -0.3416446842253373f, -0.08574019035519267f, -0.047686803502292804f, -0.14266084808807242f},
    {0.0f, 0.0f, -0.13813540350758585f, 0.3302282550303788f, 0.0f, 0.08755115000587084f, -0.07946706605909573f, -0.4613374887461511f, -0.13813540350758294f, -0.07946706605910261f, 0.49724647109535086f, 0.12538059448563663f, 0.3302282550303805f, -0.4613374887461554f, 0.12538059448564315f, -0.13813540350758452f},
    {0.0f, 0.0f, -0.17437602599651067f, 0.0702790691196284f, 0.0f, -0.2921026642334881f, 0.3623817333531167f, 0.0f, -0.1743760259965108f, 0.36238173335311646f, 0.29210266423348785f, -0.4326608024727445f, 0.07027906911962818f, 0.0f, -0.4326608024727457f, 0.34875205199302267f},
    {0.0f, 0.0f, 0.11354987314994337f, -0.07417504595810355f, 0.0f, 0.19402893032594343f, -0.435190496523228f, 0.21918684838857466f, 0.11354987314994257f, -0.4351904965232251f, 0.5550443808910661f, -0.25468277124066463f, -0.07417504595810233f, 0.2191868483885728f, -0.25468277124066413f, 0.1135498731499429f}};

void putPixel(uint c, uint x, uint y, float v) {
    xyb[c * params.paddedWidth * params.paddedHeight + y * params.paddedWidth + x] = v;
}

void transformSpecial(uint strategy, float coef[1024], uint c, uint px, uint py) {
    float small[32];
    if (strategy == 1) {   // IDENTITY
        const float b00 = coef[0], b01 = coef[1], b10 = coef[8], b11 = coef[9];
        const float dcs[4] = {b00 + b01 + b10 + b11, b00 + b01 - b10 - b11, b00 - b01 + b10 - b11, b00 - b01 - b10 + b11};
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                float residualSum = 0.0f;
                for (uint iy = 0; iy < 4; ++iy) {
                    for (uint ix = 0; ix < 4; ++ix) {
                        if (ix == 0 && iy == 0) continue;
                        residualSum += coef[(y + iy * 2) * 8 + x + ix * 2];
                    }
                }
                const float center = dcs[y * 2 + x] - residualSum * (1.0f / 16.0f);
                for (uint iy = 0; iy < 4; ++iy) {
                    for (uint ix = 0; ix < 4; ++ix) {
                        if (ix == 1 && iy == 1) { putPixel(c, px + 4 * x + 1, py + 4 * y + 1, center); continue; }
                        putPixel(c, px + x * 4 + ix, py + y * 4 + iy, coef[(y + iy * 2) * 8 + x + ix * 2] + center);
                    }
                }
                putPixel(c, px + x * 4, py + y * 4, coef[(y + 2) * 8 + x + 2] + center);
            }
        }
        return;
    }
    if (strategy == 2) {   // DCT2X2: IDCT2TopBlock<2>, <4>, <8>
        float blk[64];
        for (uint i = 0; i < 64; ++i) blk[i] = coef[i];
        for (uint s = 2; s <= 8; s *= 2) {
            float tmp[64];
            const uint n2 = s / 2;
            for (uint y = 0; y < n2; ++y) {
                for (uint x = 0; x < n2; ++x) {
                    const float c00 = blk[y * 8 + x], c01 = blk[y * 8 + n2 + x];
                    const float c10 = blk[(y + n2) * 8 + x], c11 = blk[(y + n2) * 8 + n2 + x];
                    tmp[y * 2 * 8 + x * 2] = c00 + c01 + c10 + c11;
                    tmp[y * 2 * 8 + x * 2 + 1] = c00 + c01 - c10 - c11;
                    tmp[(y * 2 + 1) * 8 + x * 2] = c00 - c01 + c10 - c11;
                    tmp[(y * 2 + 1) * 8 + x * 2 + 1] = c00 - c01 - c10 + c11;
                }
            }
            for (uint y = 0; y < s; ++y) for (uint x = 0; x < s; ++x) blk[y * 8 + x] = tmp[y * 8 + x];
        }
        for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) putPixel(c, px + x, py + y, blk[y * 8 + x]);
        return;
    }
    if (strategy == 3) {   // DCT4X4
        const float b00 = coef[0], b01 = coef[1], b10 = coef[8], b11 = coef[9];
        const float dcs[4] = {b00 + b01 + b10 + b11, b00 + b01 - b10 - b11, b00 - b01 + b10 - b11, b00 - b01 - b10 + b11};
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                for (uint i = 0; i < 32; ++i) small[i] = 0.0f;
                small[0] = dcs[y * 2 + x];
                for (uint iy = 0; iy < 4; ++iy) {
                    for (uint ix = 0; ix < 4; ++ix) {
                        if (ix == 0 && iy == 0) continue;
                        small[iy * 4 + ix] = coef[(y + iy * 2) * 8 + x + ix * 2];
                    }
                }
                idctSmall(small, 4, 4, c, px + x * 4, py + y * 4);
            }
        }
        return;
    }
    if (strategy == 12) {   // DCT4X8: two 4-row x 8-col transforms
        const float dcs[2] = {coef[0] + coef[8], coef[0] - coef[8]};
        for (uint y = 0; y < 2; ++y) {
            for (uint i = 0; i < 32; ++i) small[i] = 0.0f;
            small[0] = dcs[y];
            for (uint iy = 0; iy < 4; ++iy) {
                for (uint ix = 0; ix < 8; ++ix) {
                    if (ix == 0 && iy == 0) continue;
                    small[iy * 8 + ix] = coef[(y + iy * 2) * 8 + ix];
                }
            }
            idctSmall(small, 4, 8, c, px, py + y * 4);
        }
        return;
    }
    if (strategy == 13) {   // DCT8X4: two 8-row x 4-col transforms, memory [4][8]
        const float dcs[2] = {coef[0] + coef[8], coef[0] - coef[8]};
        for (uint x = 0; x < 2; ++x) {
            for (uint i = 0; i < 32; ++i) small[i] = 0.0f;
            small[0] = dcs[x];
            for (uint iy = 0; iy < 4; ++iy) {
                for (uint ix = 0; ix < 8; ++ix) {
                    if (ix == 0 && iy == 0) continue;
                    small[iy * 8 + ix] = coef[(x + iy * 2) * 8 + ix];
                }
            }
            idctSmall(small, 8, 4, c, px + x * 4, py);
        }
        return;
    }
    if (strategy >= 14 && strategy <= 17) {   // AFV
        const uint afvKind = strategy - 14;
        const uint afvX = afvKind & 1;
        const uint afvY = afvKind / 2;
        const float b00 = coef[0], b01 = coef[1], b10 = coef[8];
        const float dcs[3] = {(b00 + b10 + b01) * 4.0f, b00 + b10 - b01, b00 - b10};
        float afvCoef[16];
        afvCoef[0] = dcs[0];
        for (uint iy = 0; iy < 4; ++iy) {
            for (uint ix = 0; ix < 4; ++ix) {
                if (ix == 0 && iy == 0) continue;
                afvCoef[iy * 4 + ix] = coef[iy * 2 * 8 + ix * 2];
            }
        }
        float blk[16];
        for (uint i = 0; i < 16; ++i) {
            float sum = 0.0f;
            for (uint j = 0; j < 16; ++j) sum += afvCoef[j] * kAfvBasis[j][i];
            blk[i] = sum;
        }
        for (uint iy = 0; iy < 4; ++iy) {
            for (uint ix = 0; ix < 4; ++ix) {
                putPixel(c, px + afvX * 4 + ix, py + afvY * 4 + iy, blk[(afvY == 1 ? 3 - iy : iy) * 4 + (afvX == 1 ? 3 - ix : ix)]);
            }
        }
        for (uint i = 0; i < 32; ++i) small[i] = 0.0f;
        small[0] = dcs[1];
        for (uint iy = 0; iy < 4; ++iy) {
            for (uint ix = 0; ix < 4; ++ix) {
                if (ix == 0 && iy == 0) continue;
                small[iy * 4 + ix] = coef[iy * 2 * 8 + ix * 2 + 1];
            }
        }
        idctSmall(small, 4, 4, c, px + (afvX == 1 ? 0 : 4), py + afvY * 4);
        for (uint i = 0; i < 32; ++i) small[i] = 0.0f;
        small[0] = dcs[2];
        for (uint iy = 0; iy < 4; ++iy) {
            for (uint ix = 0; ix < 8; ++ix) {
                if (ix == 0 && iy == 0) continue;
                small[iy * 8 + ix] = coef[(1 + iy * 2) * 8 + ix];
            }
        }
        idctSmall(small, 4, 8, c, px, py + (afvY == 1 ? 0 : 4));
        return;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void inverseTransform(uint3 tid: SV_DispatchThreadID) {
    const uint nb = nblocks();
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x - c * nb;
    const uint entry = acs[block];
    if ((entry & 1) == 0 || entry == 0xFFu) return;
    const uint strategy = entry >> 1;
    if (strategy >= 18) {
        uint before;
        InterlockedOr(flags[0], 1u, before);
        return;
    }
    const uint cx = kCoveredX[strategy];
    const uint cy = kCoveredY[strategy];
    const uint size = 64 * cx * cy;
    const uint bx = block % params.blocksX;
    const uint by = block / params.blocksX;
    float coef[1024];
    for (uint k = 0; k < size; ++k) coef[k] = dequantF[slotOf(c, block, cx, k)];
    // LowestFrequenciesFromDC: the cy x cx DC patch, scaled DCT, resample scales.
    const uint ny = cy * 8, nx = cx * 8;
    for (uint v = 0; v < cy; ++v) {
        for (uint u = 0; u < cx; ++u) {
            float sum = 0.0f;
            for (uint y = 0; y < cy; ++y) {
                for (uint x = 0; x < cx; ++x) {
                    const float dcv = dcSmooth[c * nb + (by + y) * params.blocksX + bx + x];
                    sum += dcv * fdctBasis(v, y, cy) * fdctBasis(u, x, cx);
                }
            }
            coef[memIndex(v, u, ny, nx)] = sum * resampleScale(cy, v) * resampleScale(cx, u);
        }
    }
    const uint px = bx * 8, py = by * 8;
    if (strategy == 0 || (strategy >= 4 && strategy <= 11)) {
        idctToPixels(coef, ny, nx, c, px, py);
    } else {
        transformSpecial(strategy, coef, c, px, py);
    }
}
