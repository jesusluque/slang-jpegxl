// Copyright (c) 2026 jesus luque.
//
// Upsampling by 2, 4 or 8 (frame header `upsampling`): every output pixel is a
// 5x5 weighted sum of the coded image around its source pixel, clamped to the
// window's range, with the weights of libjxl's render_pipeline/stage_upsampling.cc.
// The 15 / 55 / 210 stored weights are the triangular half of the kernels of
// the top-left sub-positions; the others follow by symmetry. Borders mirror at
// the coded size, as the render pipeline pads. One thread per output pixel.
module dec_upsample;

StructuredBuffer<float>   src;       // 3 planes, paddedWidth * paddedHeight, image width x height (coded)
RWStructuredBuffer<float> dst;       // 3 planes, outPaddedWidth * outPaddedHeight
StructuredBuffer<float>   weights;   // 280 floats: 2x at 0, 4x at 15, 8x at 70

struct UpsampleParams {
    uint width, height;               // coded size
    uint paddedWidth, paddedHeight;
    uint outWidth, outHeight;         // upsampled size
    uint outPaddedWidth, outPaddedHeight;
    uint shift;                       // 1, 2 or 3
    uint channels;                    // planes to process (3 for colour, 1 for an extra channel)
    uint srcPlane;                    // first source plane index
    uint dstPlane;                    // first destination plane index
};
ConstantBuffer<UpsampleParams> params;

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
    return src[(params.srcPlane + c) * params.paddedWidth * params.paddedHeight + uint(my) * params.paddedWidth + uint(mx)];
}

/// Weight of input tap (py, px) of the kernel for output sub-position (oy, ox).
float kernelWeight(uint oy, uint ox, uint py, uint px) {
    const uint n = 1u << params.shift;
    const uint h = n / 2;
    const uint ky = oy < h ? oy : n - 1 - oy;
    const uint kx = ox < h ? ox : n - 1 - ox;
    const uint fy = oy < h ? py : 4 - py;
    const uint fx = ox < h ? px : 4 - px;
    const uint j = 5 * ky + fy;
    const uint i = 5 * kx + fx;
    const uint my = min(i, j);
    const uint mx = max(i, j);
    const uint base = params.shift == 1 ? 0 : params.shift == 2 ? 15 : 70;
    return weights[base + 5 * h * my - my * (my - 1) / 2 + mx - my];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void upsample(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.outWidth || tid.y >= params.outHeight) return;
    const uint n = 1u << params.shift;
    const int sx = int(tid.x >> params.shift);
    const int sy = int(tid.y >> params.shift);
    const uint ox = tid.x & (n - 1);
    const uint oy = tid.y & (n - 1);
    const uint plane = params.outPaddedWidth * params.outPaddedHeight;
    for (uint c = 0; c < params.channels; ++c) {
        float lo = 1e30f, hi = -1e30f, sum = 0.0f;
        for (uint py = 0; py < 5; ++py) {
            for (uint px = 0; px < 5; ++px) {
                const float v = at(c, sx - 2 + int(px), sy - 2 + int(py));
                lo = min(lo, v);
                hi = max(hi, v);
                sum += v * kernelWeight(oy, ox, py, px);
            }
        }
        dst[(params.dstPlane + c) * plane + tid.y * params.outPaddedWidth + tid.x] = clamp(sum, lo, hi);
    }
}
