// Copyright (c) 2026 jesus luque.
//
// The inverse Gaborish (libjxl enc_gaborish.cc): a 5x5 symmetric convolution
// that pre-sharpens the XYB planes so that the decoder's Gaborish blur returns
// the original. One thread per padded pixel, borders mirrored.
module enc_gaborish;

struct GaborishParams {
    uint paddedWidth;
    uint paddedHeight;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float>      src;    // 3 planes
RWStructuredBuffer<float>    dst;    // 3 planes
ConstantBuffer<GaborishParams> params;

// enc_gaborish.cc kGaborish, mul = 1 (VarDCT).
static const float kGaborish[5] = {-0.09495815671340026f, -0.041031725066768575f, 0.013710004822696948f,
                                   0.006510206083837737f, -0.0014789063378272242f};

int mirror(int v, int size) {
    if (v < 0) return -v - 1;
    if (v >= size) return 2 * size - v - 1;
    return v;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void gaborishInverse(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.paddedWidth || tid.y >= params.paddedHeight) {
        return;
    }
    const int w = int(params.paddedWidth);
    const int h = int(params.paddedHeight);
    const float normalize = 1.0f / (1.0f + 4.0f * (kGaborish[0] + kGaborish[1] + kGaborish[2] + kGaborish[4] + 2.0f * kGaborish[3]));
    // WeightsSymmetric5: c (centre), r (+-1 on an axis), R (+-2 on an axis),
    // d (+-1, +-1), D (+-2, +-2), L (+-1, +-2 either way).
    const float c = normalize;
    const float r = normalize * kGaborish[0];
    const float R = normalize * kGaborish[2];
    const float d = normalize * kGaborish[1];
    const float D = normalize * kGaborish[4];
    const float L = normalize * kGaborish[3];
    const uint plane = params.paddedWidth * params.paddedHeight;
    const int x = int(tid.x), y = int(tid.y);
    int xs[5], ys[5];
    for (int i = 0; i < 5; ++i) {
        xs[i] = mirror(x + i - 2, w);
        ys[i] = mirror(y + i - 2, h);
    }
    const float weights[25] = {D, L, R, L, D,
                               L, d, r, d, L,
                               R, r, c, r, R,
                               L, d, r, d, L,
                               D, L, R, L, D};
    for (uint ch = 0; ch < 3; ++ch) {
        float sum = 0.0f;
        for (int j = 0; j < 5; ++j) {
            const uint rowBase = ch * plane + uint(ys[j]) * params.paddedWidth;
            for (int i = 0; i < 5; ++i) {
                sum += weights[j * 5 + i] * src[rowBase + uint(xs[i])];
            }
        }
        dst[ch * plane + tid.y * params.paddedWidth + tid.x] = sum;
    }
}
