// Copyright (c) 2026 jesus luque.
//
// 8-bit RGB (sRGB, or a gamma transfer) to XYB, one thread per padded pixel. The image is padded to whole
// 8x8 blocks by replicating its last row and column (libjxl pads the same way
// before the DCT). libjxl enc_xyb.cc: opsin absorbance, cube root with bias.
module enc_xyb;

struct XybParams {
    uint  width;         // source
    uint  height;
    uint  paddedWidth;   // multiples of 8
    uint  paddedHeight;
    uint  transfer;      // 0 sRGB, 2 power (gamma)
    float gammaInv;      // linear = v ^ gammaInv for transfer 2
    uint  numChannels;   // 3, or 4 with alpha (skipped here)
    uint  bitDepth;      // 8 or 16
};

StructuredBuffer<uint>     rgb8;    // width*height*numChannels samples of bitDepth/8 bytes, packed little-endian
RWStructuredBuffer<float>  xyb;     // 3 planes of paddedWidth*paddedHeight
ConstantBuffer<XybParams>  params;

float srgbToLinear(float v) {
    return v <= 0.04045f ? v * (1.0f / 12.92f) : pow((v + 0.055f) * (1.0f / 1.055f), 2.4f);
}

float toLinear(float v) {
    return params.transfer == 2 ? pow(v, params.gammaInv) : srgbToLinear(v);
}

uint byteAt(uint index) {
    return (rgb8[index >> 2] >> ((index & 3) * 8)) & 0xFFu;
}

/// Sample `index` as 0..1 (16-bit samples are little-endian pairs).
float sampleAt(uint index) {
    if (params.bitDepth == 8) return float(byteAt(index)) * (1.0f / 255.0f);
    return float(byteAt(2 * index) | (byteAt(2 * index + 1) << 8)) * (1.0f / 65535.0f);
}

// cms/opsin_params.h
static const float kM00 = 0.30f, kM01 = 0.622f, kM02 = 0.078f;
static const float kM10 = 0.23f, kM11 = 0.692f, kM12 = 0.078f;
static const float kM20 = 0.24342268924547819f, kM21 = 0.20476744424496821f, kM22 = 0.55180986650955360f;
static const float kBias = 0.0037930732552754493f;

[shader("compute")]
[numthreads(16, 16, 1)]
void rgbToXyb(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x;
    const uint y = tid.y;
    if (x >= params.paddedWidth || y >= params.paddedHeight) {
        return;
    }
    const uint sx = min(x, params.width - 1);
    const uint sy = min(y, params.height - 1);
    const uint p = (sy * params.width + sx) * params.numChannels;
    const float r = toLinear(sampleAt(p + 0));
    const float g = toLinear(sampleAt(p + 1));
    const float b = toLinear(sampleAt(p + 2));

    float m0 = max(kM00 * r + kM01 * g + kM02 * b + kBias, 0.0f);
    float m1 = max(kM10 * r + kM11 * g + kM12 * b + kBias, 0.0f);
    float m2 = max(kM20 * r + kM21 * g + kM22 * b + kBias, 0.0f);
    const float cbrtBias = pow(kBias, 1.0f / 3.0f);
    m0 = pow(m0, 1.0f / 3.0f) - cbrtBias;
    m1 = pow(m1, 1.0f / 3.0f) - cbrtBias;
    m2 = pow(m2, 1.0f / 3.0f) - cbrtBias;

    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint i = y * params.paddedWidth + x;
    xyb[0 * plane + i] = 0.5f * (m0 - m1);   // X
    xyb[1 * plane + i] = 0.5f * (m0 + m1);   // Y
    xyb[2 * plane + i] = m2;                 // B
}
