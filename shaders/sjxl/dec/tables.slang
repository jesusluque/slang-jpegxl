// Copyright (c) 2026 jesus luque.
//
// The dequantisation matrices, one thread per entry, from the encodings the
// globals thread decoded (or libjxl's default library when all_default).
// libjxl quant_weights.cc ComputeQuantTable and DequantMatricesLibraryDef.
//
// Layout follows libjxl: tables in kind order 0..16, each 3 channels of
// (8 * required_size_x) rows by (8 * required_size_y) columns, row-major;
// the stored value is the dequant multiplier (1 / weight).
module dec_tables;

StructuredBuffer<uint>    arena;      // the globals region: encodings at arena[kGsQuantEncOff]
RWStructuredBuffer<float> dequant;    // 394752 floats

static const uint kGsQuantEncOff = 44;
static const uint kQeWords = 176;
static const uint kQeMode = 0, kQeNumBands = 1, kQeBands = 2, kQeNumBandsAfv = 53, kQeBandsAfv = 54, kQeIdWeights = 105,
                  kQeDct2Weights = 114, kQeDct4x8Mul = 132, kQeDct4Mul = 135, kQeAfvWeights = 141, kQeRawOff = 168, kQeRawDen = 169;

static const uint kTableSizeX[17] = {1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16};
static const uint kTableSizeY[17] = {1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32};
// First float of each table (3 channels each): cumulative 3 * 64 * x * y.
static const uint kTableBase[18] = {0, 192, 384, 576, 768, 1536, 4608, 4992, 5760, 7296, 7488, 7680, 19968, 26112, 75264, 99840,
                                    296448, 394752};

struct TableParams {
    uint totalEntries;
    uint pad0, pad1, pad2;
};
ConstantBuffer<TableParams> params;

float mult(float v) { return v > 0.0f ? 1.0f + v : 1.0f / (1.0f - v); }

/// libjxl Interpolate over `bands` (already chained): a * (b/a)^frac.
float interpolate(float scaledPos, float bands[17], uint numBands) {
    const uint idx = min(uint(scaledPos), numBands - 2);
    const float frac = scaledPos - float(idx);
    const float a = bands[idx];
    const float b = bands[idx + 1];
    return a * pow(b / a, frac);
}

/// GetQuantWeights at (x, y) of a ROWS x COLS table.
float dctWeight(float distanceBands[17], uint numBands, uint rows, uint cols, uint x, uint y) {
    float bands[17];
    bands[0] = distanceBands[0];
    for (uint i = 1; i < numBands; ++i) bands[i] = bands[i - 1] * mult(distanceBands[i]);
    if (numBands == 1) return bands[0];
    const float scale = float(numBands - 1) / (1.41421356237f + 1e-6f);
    const float rcpcol = scale / float(cols - 1);
    const float rcprow = scale / float(rows - 1);
    const float dx = float(x) * rcpcol;
    const float dy = float(y) * rcprow;
    const float dist = sqrt(dx * dx + dy * dy);
    return interpolate(dist, bands, numBands);
}

struct Enc {
    uint  mode;
    uint  numBands;
    float bands[17];        // this channel
    uint  numBandsAfv;
    float bandsAfv[17];
    float idw[3];
    float dct2w[6];
    float dct4x8mul;
    float dct4mul[2];
    float afvw[9];
};

// libjxl's default library (DequantMatricesLibraryDef), per kind and channel.
void libraryBands(uint kind, uint c, out float bands[17], out uint numBands) {
    for (uint i = 0; i < 17; ++i) bands[i] = 0.0f;
    numBands = 1;
    switch (kind) {
    case 0: {
        const float b[3][6] = {{3150.0f, 0.0f, -0.4f, -0.4f, -0.4f, -2.0f}, {560.0f, 0.0f, -0.3f, -0.3f, -0.3f, -0.3f}, {512.0f, -2.0f, -1.0f, 0.0f, -1.0f, -2.0f}};
        numBands = 6; for (uint i = 0; i < 6; ++i) bands[i] = b[c][i]; break;
    }
    case 3: case 10: {   // DCT4X4 params (AFV uses them as its 4x4 params too: handled by caller)
        const float b[3][4] = {{2200.0f, 0.0f, 0.0f, 0.0f}, {392.0f, 0.0f, 0.0f, 0.0f}, {112.0f, -0.25f, -0.25f, -0.5f}};
        numBands = 4; for (uint i = 0; i < 4; ++i) bands[i] = b[c][i]; break;
    }
    case 4: {
        const float b[3][7] = {{8996.8725711814115328f, -1.3000777393353804f, -0.49424529824571225f, -0.439093774457103443f, -0.6350101832695744f, -0.90177264050827612f, -1.6162099239887414f},
                               {3191.48366296844234752f, -0.67424582104194355f, -0.80745813428471001f, -0.44925837484843441f, -0.35865440981033403f, -0.31322389111877305f, -0.37615025315725483f},
                               {1157.50408145487200256f, -2.0531423165804414f, -1.4f, -0.50687130033378396f, -0.42708730624733904f, -1.4856834539296244f, -4.9209142884401604f}};
        numBands = 7; for (uint i = 0; i < 7; ++i) bands[i] = b[c][i]; break;
    }
    case 5: {
        const float b[3][8] = {{15718.40830982518931456f, -1.025f, -0.98f, -0.9012f, -0.4f, -0.48819395464f, -0.421064f, -0.27f},
                               {7305.7636810695983104f, -0.8041958212306401f, -0.7633036457487539f, -0.55660379990111464f, -0.49785304658857626f, -0.43699592683512467f, -0.40180866526242109f, -0.27321683125358037f},
                               {3803.53173721215041536f, -3.060733579805728f, -2.0413270132490346f, -2.0235650159727417f, -0.5495389509954993f, -0.4f, -0.4f, -0.3f}};
        numBands = 8; for (uint i = 0; i < 8; ++i) bands[i] = b[c][i]; break;
    }
    case 6: {
        const float b[3][7] = {{7240.7734393502f, -0.7f, -0.7f, -0.2f, -0.2f, -0.2f, -0.5f}, {1448.15468787004f, -0.5f, -0.5f, -0.5f, -0.2f, -0.2f, -0.2f},
                               {506.854140754517f, -1.4f, -0.2f, -0.5f, -0.5f, -1.5f, -3.6f}};
        numBands = 7; for (uint i = 0; i < 7; ++i) bands[i] = b[c][i]; break;
    }
    case 7: {
        const float b[3][8] = {{16283.2494710648897f, -1.7812845336559429f, -1.6309059012653515f, -1.0382179034313539f, -0.85f, -0.7f, -0.9f, -1.2360638576849587f},
                               {5089.15750884921511936f, -0.320049391452786891f, -0.35362849922161446f, -0.30340000000000003f, -0.61f, -0.5f, -0.5f, -0.6f},
                               {3397.77603275308720128f, -0.321327362693153371f, -0.34507619223117997f, -0.70340000000000003f, -0.9f, -1.0f, -1.0f, -1.1754605576265209f}};
        numBands = 8; for (uint i = 0; i < 8; ++i) bands[i] = b[c][i]; break;
    }
    case 8: {
        const float b[3][8] = {{13844.97076442300573f, -0.97113799999999995f, -0.658f, -0.42026f, -0.22712f, -0.2206f, -0.226f, -0.6f},
                               {4798.964084220744293f, -0.61125308982767057f, -0.83770786552491361f, -0.79014862079498627f, -0.2692727459704829f, -0.38272769465388551f, -0.22924222653091453f, -0.20719098826199578f},
                               {1807.236946760964614f, -1.2f, -1.2f, -0.7f, -0.7f, -0.7f, -0.4f, -0.5f}};
        numBands = 8; for (uint i = 0; i < 8; ++i) bands[i] = b[c][i]; break;
    }
    case 9: {
        const float b[3][4] = {{2198.050556016380522f, -0.96269623020744692f, -0.76194253026666783f, -0.6551140670773547f},
                               {764.3655248643528689f, -0.92630200888366945f, -0.9675229603596517f, -0.27845290869168118f},
                               {527.107573587542228f, -1.4594385811273854f, -1.450082094097871593f, -1.5843722511996204f}};
        numBands = 4; for (uint i = 0; i < 4; ++i) bands[i] = b[c][i]; break;
    }
    case 11: case 12: case 13: case 14: case 15: case 16: {
        const bool square = kind == 11 || kind == 13 || kind == 15;
        const float scale = kind == 11 ? 0.9f : kind == 12 ? 0.65f : kind == 13 ? 1.8f : kind == 14 ? 1.3f : kind == 15 ? 3.6f : 2.6f;
        const float first[2][3] = {{23629.073922049845f, 8611.3238710010046f, 4492.2486445538634f}, {26629.073922049845f, 9311.3238710010046f, 4992.2486445538634f}};
        const float rest[3][7] = {{-1.025f, -0.78f, -0.65012f, -0.19041574084286472f, -0.20819395464f, -0.421064f, -0.32733845535848671f},
                                  {-0.3041958212306401f, -0.3633036457487539f, -0.35660379990111464f, -0.3443074455424403f, -0.33699592683512467f, -0.30180866526242109f, -0.27321683125358037f},
                                  {-1.2f, -1.2f, -0.8f, -0.7f, -0.7f, -0.4f, -0.5f}};
        numBands = 8;
        bands[0] = scale * first[square ? 1 : 0][c];
        for (uint i = 1; i < 8; ++i) bands[i] = rest[c][i - 1];
        break;
    }
    default: break;
    }
}

Enc loadEncoding(uint kind, uint c) {
    Enc e;
    const uint qe = arena[kGsQuantEncOff] + kind * kQeWords;
    e.mode = arena[qe + kQeMode];
    for (uint i = 0; i < 17; ++i) { e.bands[i] = 0.0f; e.bandsAfv[i] = 0.0f; }
    e.numBands = 1;
    e.numBandsAfv = 1;
    for (uint i = 0; i < 3; ++i) e.idw[i] = 0.0f;
    for (uint i = 0; i < 6; ++i) e.dct2w[i] = 0.0f;
    e.dct4x8mul = 1.0f;
    e.dct4mul[0] = e.dct4mul[1] = 1.0f;
    for (uint i = 0; i < 9; ++i) e.afvw[i] = 0.0f;
    if (e.mode == 0) {
        // Library: the mode a kind's default encoding has, with its constants.
        switch (kind) {
        case 1: {
            e.mode = 1;
            const float w[3][3] = {{280.0f, 3160.0f, 3160.0f}, {60.0f, 864.0f, 864.0f}, {18.0f, 200.0f, 200.0f}};
            for (uint i = 0; i < 3; ++i) e.idw[i] = w[c][i];
            break;
        }
        case 2: {
            e.mode = 2;
            const float w[3][6] = {{3840.0f, 2560.0f, 1280.0f, 640.0f, 480.0f, 300.0f}, {960.0f, 640.0f, 320.0f, 180.0f, 140.0f, 120.0f}, {640.0f, 320.0f, 128.0f, 64.0f, 32.0f, 16.0f}};
            for (uint i = 0; i < 6; ++i) e.dct2w[i] = w[c][i];
            break;
        }
        case 3:
            e.mode = 3;
            libraryBands(3, c, e.bands, e.numBands);
            break;
        case 9:
            e.mode = 4;
            libraryBands(9, c, e.bands, e.numBands);
            break;
        case 10: {
            e.mode = 5;
            libraryBands(9, c, e.bands, e.numBands);          // DCT4X8 params
            libraryBands(3, c, e.bandsAfv, e.numBandsAfv);    // DCT4X4 params
            const float w[3][9] = {{3072.0f, 3072.0f, 256.0f, 256.0f, 256.0f, 414.0f, 0.0f, 0.0f, 0.0f}, {1024.0f, 1024.0f, 50.0f, 50.0f, 50.0f, 58.0f, 0.0f, 0.0f, 0.0f},
                                   {384.0f, 384.0f, 12.0f, 12.0f, 12.0f, 22.0f, -0.25f, -0.25f, -0.25f}};
            for (uint i = 0; i < 9; ++i) e.afvw[i] = w[c][i];
            break;
        }
        default:
            e.mode = 6;
            libraryBands(kind, c, e.bands, e.numBands);
            break;
        }
        return e;
    }
    e.numBands = arena[qe + kQeNumBands];
    for (uint i = 0; i < 17; ++i) e.bands[i] = asfloat(arena[qe + kQeBands + c * 17 + i]);
    e.numBandsAfv = arena[qe + kQeNumBandsAfv];
    for (uint i = 0; i < 17; ++i) e.bandsAfv[i] = asfloat(arena[qe + kQeBandsAfv + c * 17 + i]);
    for (uint i = 0; i < 3; ++i) e.idw[i] = asfloat(arena[qe + kQeIdWeights + c * 3 + i]);
    for (uint i = 0; i < 6; ++i) e.dct2w[i] = asfloat(arena[qe + kQeDct2Weights + c * 6 + i]);
    e.dct4x8mul = asfloat(arena[qe + kQeDct4x8Mul + c]);
    e.dct4mul[0] = asfloat(arena[qe + kQeDct4Mul + c * 2]);
    e.dct4mul[1] = asfloat(arena[qe + kQeDct4Mul + c * 2 + 1]);
    for (uint i = 0; i < 9; ++i) e.afvw[i] = asfloat(arena[qe + kQeAfvWeights + c * 9 + i]);
    return e;
}

/// The weight (before inversion) at (x, y) of a kind's table for channel c.
float weightAt(uint kind, uint c, uint x, uint y, Enc e) {
    const uint rows = 8 * kTableSizeX[kind];
    const uint cols = 8 * kTableSizeY[kind];
    switch (e.mode) {
    case 1: {   // identity
        if (y == 0 && x == 1) return e.idw[1];
        if (y == 1 && x == 0) return e.idw[1];
        if (y == 1 && x == 1) return e.idw[2];
        return e.idw[0];
    }
    case 2: {   // DCT2
        if ((y == 0 && x == 1) || (y == 1 && x == 0)) return e.dct2w[0];
        if (y == 1 && x == 1) return e.dct2w[1];
        if (y < 2 && x >= 2 && x < 4) return e.dct2w[2];
        if (y >= 2 && y < 4 && x < 2) return e.dct2w[2];
        if (y >= 2 && y < 4 && x >= 2 && x < 4) return e.dct2w[3];
        if (y < 4 && x >= 4) return e.dct2w[4];
        if (y >= 4 && x < 4) return e.dct2w[4];
        if (y >= 4 && x >= 4) return e.dct2w[5];
        return e.dct2w[0];   // (0,0): unused
    }
    case 3: {   // DCT4
        float w = dctWeight(e.bands, e.numBands, 4, 4, x / 2, y / 2);
        if (y == 0 && x == 1) w /= e.dct4mul[0];
        if (y == 1 && x == 0) w /= e.dct4mul[0];
        if (y == 1 && x == 1) w /= e.dct4mul[1];
        return w;
    }
    case 4: {   // DCT4X8
        float w = dctWeight(e.bands, e.numBands, 4, 8, x, y / 2);
        if (y == 1 && x == 0) w /= e.dct4x8mul;
        return w;
    }
    case 5: {   // AFV
        if (x == 0 && y == 0) return 1.0f;
        if ((y & 1) == 1) {
            // odd rows: 4x8 weights
            return dctWeight(e.bands, e.numBands, 4, 8, x, y / 2);
        }
        if ((x & 1) == 1) {
            // even rows, odd columns: 4x4 weights
            return dctWeight(e.bandsAfv, e.numBandsAfv, 4, 4, x / 2, y / 2);
        }
        // even/even: the AFV positions
        const uint ax = x / 2, ay = y / 2;
        if (ax == 0 && ay == 1) return e.afvw[0];
        if (ax == 1 && ay == 0) return e.afvw[1];
        if (ax == 0 && ay == 2) return e.afvw[2];
        if (ax == 2 && ay == 0) return e.afvw[3];
        if (ax == 2 && ay == 2) return e.afvw[4];
        const float kFreqs[16] = {0.0f, 0.0f, 0.8517778890324296f, 5.37778436506804f, 0.0f, 0.0f, 4.734747904497923f, 5.449245381693219f,
                                  1.6598270267479331f, 4.0f, 7.275749096817861f, 10.423227632456525f, 2.662932286148962f, 7.630657783650829f,
                                  8.962388608184032f, 12.97166202570235f};
        float bands[17];
        bands[0] = e.afvw[5];
        for (uint i = 1; i < 4; ++i) bands[i] = bands[i - 1] * mult(e.afvw[i + 5]);
        const float lo = 0.8517778890324296f;
        const float hi = 12.97166202570235f - lo + 1e-6f;
        const float pos = (kFreqs[ay * 4 + ax] - lo) * 3.0f / hi;
        return interpolate(pos, bands, 4);
    }
    case 6:
        return dctWeight(e.bands, e.numBands, rows, cols, x, y);
    default:
        return 1.0f;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeDequantTable(uint3 tid: SV_DispatchThreadID) {
    const uint entry = tid.x;
    if (entry >= params.totalEntries) return;
    uint kind = 0;
    while (kind + 1 < 17 && entry >= kTableBase[kind + 1]) ++kind;
    const uint rows = 8 * kTableSizeX[kind];
    const uint cols = 8 * kTableSizeY[kind];
    const uint num = rows * cols;
    const uint rel = entry - kTableBase[kind];
    const uint c = rel / num;
    const uint i = rel % num;
    const uint y = i / cols;
    const uint x = i % cols;
    const Enc e = loadEncoding(kind, c);
    const float w = weightAt(kind, c, x, y, e);
    dequant[entry] = 1.0f / w;
}
