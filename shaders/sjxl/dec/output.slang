// Copyright (c) 2026 jesus luque.
//
// XYB to 8-bit sRGB, one thread per output word (four bytes of the packed
// RGB8 image). dec_xyb-inl.h XybToRgb, then the sRGB transfer function.
module dec_output;

StructuredBuffer<float>  xyb;     // 3 planes, paddedWidth * paddedHeight
StructuredBuffer<float>  alpha;   // the alpha plane (channels == 4), same layout, at plane `alphaPlane`
RWStructuredBuffer<uint> rgb8;    // width * height * channels bytes, packed
RWStructuredBuffer<float> outPlanes;   // toOutputPlanes: 3 planes, width * height, the output encoding in [0, 1]

struct OutputParams {
    uint  width;
    uint  height;
    uint  paddedWidth;
    uint  paddedHeight;
    float inverseMatrix[12];   // 3x3 row-major, scaled by 255 / intensity_target, padded rows of 4
    float negBias[4];          // opsin biases (negative) and padding
    float negBiasCbrt[4];
    uint  transfer;            // 0 sRGB, 1 linear, 2 power (gamma)
    float gamma;               // the exponent for transfer 2 (e.g. 1/2.2)
    uint  mode;                // 0: XYB planes; 1: planes already hold the output encoding in [0, 1]; 2: YCbCr (Cb, Y, Cr planes)
    uint  channels;            // 3 (RGB) or 4 (RGBA)
    uint  alphaPlane;          // plane index of the alpha channel within `alpha`
    uint  bytesPerSample;      // 1: 8-bit, 2: 16-bit little-endian
    uint  pad3;
};
ConstantBuffer<OutputParams> params;

/// stage_from_linear.cc: out-of-range values are kept (blending happens after this, the
/// clamp comes with the 8/16-bit packing). sRGB is applied to the magnitude with the
/// sign restored (TF_SRGB::EncodedFromDisplay); gamma gives 0 for non-positive input.
float encodeFromLinear(float v) {
    if (params.transfer == 1) return v;
    if (params.transfer == 2) return v <= 0.0f ? 0.0f : pow(v, params.gamma);
    const float a = abs(v);
    const float e = a <= 0.0031308f ? 12.92f * a : 1.055f * pow(a, 1.0f / 2.4f) - 0.055f;
    return v < 0.0f ? -e : e;
}

/// Colour channel c of pixel (x, y) in the output encoding, [0, 1] (not yet clamped for mode 0).
float outputValue(uint c, uint x, uint y) {
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = y * params.paddedWidth + x;
    if (params.mode == 1) return xyb[c * plane + at];
    if (params.mode == 2) {   // stage_ycbcr.cc
        const float yv = xyb[plane + at] + 128.0f / 255.0f, cb = xyb[at], cr = xyb[2 * plane + at];
        return c == 0 ? yv + 1.402f * cr : c == 1 ? yv + (-0.114f * 1.772f / 0.587f) * cb + (-0.299f * 1.402f / 0.587f) * cr : yv + 1.772f * cb;
    }
    const float ox = xyb[at], oy = xyb[plane + at], ob = xyb[2 * plane + at];
    float gr = oy + ox - params.negBiasCbrt[0];
    float gg = oy - ox - params.negBiasCbrt[1];
    float gb = ob - params.negBiasCbrt[2];
    const float mr = gr * gr * gr + params.negBias[0];
    const float mg = gg * gg * gg + params.negBias[1];
    const float mb = gb * gb * gb + params.negBias[2];
    const float lin = params.inverseMatrix[c * 4 + 0] * mr + params.inverseMatrix[c * 4 + 1] * mg + params.inverseMatrix[c * 4 + 2] * mb;
    return encodeFromLinear(lin);
}

/// One thread per pixel: the three output-encoding planes (for blending and canvases).
[shader("compute")]
[numthreads(64, 1, 1)]
void toOutputPlanes(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint x = tid.x % params.width, y = tid.x / params.width;
    for (uint c = 0; c < 3; ++c) outPlanes[c * n + tid.x] = outputValue(c, x, y);
}

/// Sample `index` (pixel * channels + channel) as an integer on the output scale.
uint sampleFor(uint index) {
    const float peak = params.bytesPerSample == 2 ? 65535.0f : 255.0f;
    const uint pixel = index / params.channels;
    const uint c = index % params.channels;
    const uint x = pixel % params.width;
    const uint y = pixel / params.width;
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = y * params.paddedWidth + x;
    if (c == 3) return uint(round(clamp(alpha[params.alphaPlane * plane + at], 0.0f, 1.0f) * peak));
    return uint(round(clamp(outputValue(c, x, y), 0.0f, 1.0f) * peak));
}

uint byteFor(uint index) {
    if (params.bytesPerSample == 1) return sampleFor(index);
    return (sampleFor(index / 2) >> ((index & 1) * 8)) & 0xFFu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void xybToSrgb8(uint3 tid: SV_DispatchThreadID) {
    const uint total = params.width * params.height * params.channels * params.bytesPerSample;
    const uint word = tid.x;
    if (word * 4 >= total) return;
    uint packed = 0;
    for (uint b = 0; b < 4; ++b) {
        const uint index = word * 4 + b;
        if (index < total) packed |= byteFor(index) << (8 * b);
    }
    rgb8[word] = packed;
}
