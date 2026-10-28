// Copyright (c) 2026 jesus luque.
//
// The loop filters, each a whole-image pass over the three XYB planes with
// mirrored borders at the frame edge: Gaborish (stage_gaborish.cc) and the
// three edge-preserving filters (stage_epf.cc) with the per-block sigma.
module dec_filters;

StructuredBuffer<float>   src;     // 3 planes, paddedWidth * paddedHeight, image is width x height
RWStructuredBuffer<float> dst;
StructuredBuffer<float>   sigma;   // per block, 1 / sigma

struct FilterParams {
    uint  width;
    uint  height;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  blocksX;
    uint  stage;            // 0 EPF0, 1 EPF1, 2 EPF2 (for epf)
    float sigmaScale;       // pass sigma scale * 1.65
    float borderSadMul;
    float channelScale[3];
    float pad0;
    float gabWeights[12];   // per channel: w0, w1, w2 normalised (3 used of 4)
};
ConstantBuffer<FilterParams> params;

static const float kMinSigma = -3.90524291751269967465540850526868f;

int mirror(int v, int size) {
    while (v < 0 || v >= size) {
        if (v < 0) v = -v - 1;
        else v = 2 * size - 1 - v;
    }
    return v;
}

float at(uint c, int x, int y) {
    const int mx = mirror(x, int(params.width));
    const int my = mirror(y, int(params.height));
    return src[c * params.paddedWidth * params.paddedHeight + uint(my) * params.paddedWidth + uint(mx)];
}

void put(uint c, uint x, uint y, float v) {
    dst[c * params.paddedWidth * params.paddedHeight + y * params.paddedWidth + x] = v;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaborish(uint3 tid: SV_DispatchThreadID) {
    const int x = int(tid.x), y = int(tid.y);
    if (tid.x >= params.width || tid.y >= params.height) return;
    for (uint c = 0; c < 3; ++c) {
        const float w0 = params.gabWeights[c * 4 + 0];
        const float w1 = params.gabWeights[c * 4 + 1];
        const float w2 = params.gabWeights[c * 4 + 2];
        const float m = at(c, x, y);
        const float sum1 = at(c, x - 1, y) + at(c, x + 1, y) + at(c, x, y - 1) + at(c, x, y + 1);
        const float sum2 = at(c, x - 1, y - 1) + at(c, x + 1, y - 1) + at(c, x - 1, y + 1) + at(c, x + 1, y + 1);
        put(c, tid.x, tid.y, m * w0 + sum1 * w1 + sum2 * w2);
    }
}

float sigmaAt(int bx, int by) {
    const int mbx = clamp(bx, 0, int(params.blocksX) - 1);
    const int mby = clamp(by, 0, int((params.height + 7) / 8) - 1);
    return sigma[uint(mby) * params.blocksX + uint(mbx)];
}

/// Sum over the plus shape of |P(center + o) - P(center + n + o)|, all channels scaled.
float plusSad(int x, int y, int nx, int ny) {
    const int2 plus[5] = {int2(0, 0), int2(-1, 0), int2(0, -1), int2(1, 0), int2(0, 1)};
    float total = 0.0f;
    for (uint c = 0; c < 3; ++c) {
        float sad = 0.0f;
        for (uint i = 0; i < 5; ++i) {
            sad += abs(at(c, x + plus[i].x, y + plus[i].y) - at(c, x + nx + plus[i].x, y + ny + plus[i].y));
        }
        total += sad * params.channelScale[c];
    }
    return total;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void epf(uint3 tid: SV_DispatchThreadID) {
    const int x = int(tid.x), y = int(tid.y);
    if (tid.x >= params.width || tid.y >= params.height) return;
    const float s = sigmaAt(x / 8, y / 8);
    if (s < kMinSigma) {
        for (uint c = 0; c < 3; ++c) put(c, tid.x, tid.y, at(c, x, y));
        return;
    }
    const uint ix = tid.x % 8, iy = tid.y % 8;
    const float sm = params.sigmaScale;
    const float bsm = sm * params.borderSadMul;
    const bool border = ix == 0 || ix == 7 || iy == 0 || iy == 7;
    const float invSigma = s * (border ? bsm : sm);
    float X = at(0, x, y), Y = at(1, x, y), B = at(2, x, y);
    const float cx = X, cy = Y, cb = B;
    float w = 1.0f;
    if (params.stage == 0) {
        const int2 offs[12] = {int2(0, -2), int2(-1, -1), int2(0, -1), int2(1, -1), int2(-2, 0), int2(-1, 0),
                               int2(1, 0),  int2(2, 0),   int2(-1, 1), int2(0, 1),  int2(1, 1),  int2(0, 2)};
        for (uint i = 0; i < 12; ++i) {
            const float sad = plusSad(x, y, offs[i].x, offs[i].y);
            const float weight = max(0.0f, sad * invSigma + 1.0f);
            w += weight;
            X += weight * at(0, x + offs[i].x, y + offs[i].y);
            Y += weight * at(1, x + offs[i].x, y + offs[i].y);
            B += weight * at(2, x + offs[i].x, y + offs[i].y);
        }
    } else if (params.stage == 1) {
        const int2 offs[4] = {int2(0, -1), int2(-1, 0), int2(1, 0), int2(0, 1)};
        for (uint i = 0; i < 4; ++i) {
            const float sad = plusSad(x, y, offs[i].x, offs[i].y);
            const float weight = max(0.0f, sad * invSigma + 1.0f);
            w += weight;
            X += weight * at(0, x + offs[i].x, y + offs[i].y);
            Y += weight * at(1, x + offs[i].x, y + offs[i].y);
            B += weight * at(2, x + offs[i].x, y + offs[i].y);
        }
    } else {
        const int2 offs[4] = {int2(0, -1), int2(-1, 0), int2(1, 0), int2(0, 1)};
        for (uint i = 0; i < 4; ++i) {
            const float nx = at(0, x + offs[i].x, y + offs[i].y);
            const float ny = at(1, x + offs[i].x, y + offs[i].y);
            const float nb = at(2, x + offs[i].x, y + offs[i].y);
            const float sad = abs(nx - cx) * params.channelScale[0] + abs(ny - cy) * params.channelScale[1] + abs(nb - cb) * params.channelScale[2];
            const float weight = max(0.0f, sad * invSigma + 1.0f);
            w += weight;
            X += weight * nx;
            Y += weight * ny;
            B += weight * nb;
        }
    }
    const float inv = 1.0f / w;
    put(0, tid.x, tid.y, X * inv);
    put(1, tid.x, tid.y, Y * inv);
    put(2, tid.x, tid.y, B * inv);
}
