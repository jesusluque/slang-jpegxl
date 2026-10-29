// Copyright (c) 2026 jesus luque.
//
// Frame composition (stage_blending.cc, blending.cc PerformBlending): a displayed
// frame, converted to the output encoding, is placed at its origin on the canvas
// of the image's size over the background canvas of its blending source (zeros
// when none). Colour channels and extra channels have their own modes; alpha
// blending uses the alpha extra channel (premultiplied or not). Outside the
// frame's rectangle the background shows through. One thread per canvas pixel.
module dec_blend;

StructuredBuffer<float>   fg;        // the frame: 3 planes of fw * fh (output encoding)
StructuredBuffer<float>   fgEc;      // the frame's extra channels: numEc planes of fgPaddedWidth * fgPaddedHeight (frame size region)
StructuredBuffer<float>   bg;        // the background canvas: (3 + numEc) planes of width * height; unused when hasBg == 0
RWStructuredBuffer<float> canvas;    // the result, same layout as bg

struct BlendParams {
    uint  width, height;             // canvas (image) size
    uint  fw, fh;                    // frame size
    int   x0, y0;                    // frame origin on the canvas
    uint  fgPaddedWidth, fgPaddedHeight;
    uint  numEc;
    uint  hasBg;
    uint  alphaIndex;                // extra channel used by colour blending (mode 2 / 3)
    uint  alphaPremultiplied;
    uint4 modes;                     // x: colour mode; y: clamp (bit 0)
    uint4 ecModes;                   // per extra channel: mode | clamp << 8 | alpha channel << 16
};
ConstantBuffer<BlendParams> params;

float clamp01(float v) { return clamp(v, 0.0f, 1.0f); }

[shader("compute")]
[numthreads(64, 1, 1)]
void blendCanvas(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const int x = int(tid.x % params.width), y = int(tid.x / params.width);
    const uint nc = 3 + params.numEc;
    float bgv[7], fgv[7], outv[7];
    for (uint c = 0; c < nc; ++c) bgv[c] = params.hasBg != 0 ? bg[c * n + tid.x] : 0.0f;
    const int fx = x - params.x0, fy = y - params.y0;
    const bool inside = fx >= 0 && fy >= 0 && fx < int(params.fw) && fy < int(params.fh);
    if (!inside) {
        for (uint c = 0; c < nc; ++c) canvas[c * n + tid.x] = bgv[c];
        return;
    }
    for (uint c = 0; c < 3; ++c) fgv[c] = fg[c * params.fw * params.fh + uint(fy) * params.fw + uint(fx)];
    for (uint e = 0; e < params.numEc; ++e) fgv[3 + e] = fgEc[e * params.fgPaddedWidth * params.fgPaddedHeight + uint(fy) * params.fgPaddedWidth + uint(fx)];
    const bool hasAlpha = params.alphaIndex < params.numEc;
    // Extra channels first (libjxl blends them before the colour, which uses the pre-blend alpha).
    for (uint e = 0; e < params.numEc; ++e) {
        const uint word = e == 0 ? params.ecModes.x : e == 1 ? params.ecModes.y : e == 2 ? params.ecModes.z : params.ecModes.w;
        const uint mode = word & 0xFFu;
        const bool cl = ((word >> 8) & 1u) != 0;
        const uint a = (word >> 16) & 0xFFu;
        const float fa = cl ? clamp01(fgv[3 + a]) : fgv[3 + a];
        const float ba = bgv[3 + a];
        float o;
        if (mode == 0) o = fgv[3 + e];
        else if (mode == 1) o = bgv[3 + e] + fgv[3 + e];
        else if (mode == 4) o = bgv[3 + e] * (cl ? clamp01(fgv[3 + e]) : fgv[3 + e]);
        else if (mode == 2) {
            if (!hasAlpha) o = fgv[3 + e];
            else if (e == a) o = 1.0f - (1.0f - fa) * (1.0f - ba);
            else if (params.alphaPremultiplied != 0) o = fgv[3 + e] + bgv[3 + e] * (1.0f - fa);
            else {   // alpha.cc: the reciprocal is computed once and multiplied
                const float na = 1.0f - (1.0f - fa) * (1.0f - ba);
                const float rna = na > 0.0f ? 1.0f / na : 0.0f;
                o = (fgv[3 + e] * fa + bgv[3 + e] * ba * (1.0f - fa)) * rna;
            }
        } else o = !hasAlpha ? bgv[3 + e] + fgv[3 + e] : (e == a ? bgv[3 + e] : bgv[3 + e] + fgv[3 + e] * fa);   // alpha-weighted add (alpha.cc keeps the alpha channel itself)
        outv[3 + e] = o;
    }
    const uint mode = params.modes.x;
    const bool cl = (params.modes.y & 1u) != 0;
    const float fa = hasAlpha ? (cl ? clamp01(fgv[3 + params.alphaIndex]) : fgv[3 + params.alphaIndex]) : 1.0f;
    const float ba = hasAlpha ? bgv[3 + params.alphaIndex] : 0.0f;
    for (uint c = 0; c < 3; ++c) {
        float o;
        if (mode == 0) o = fgv[c];
        else if (mode == 1) o = bgv[c] + fgv[c];
        else if (mode == 4) o = bgv[c] * (cl ? clamp01(fgv[c]) : fgv[c]);
        else if (mode == 2) {
            if (!hasAlpha) o = fgv[c];
            else if (params.alphaPremultiplied != 0) o = fgv[c] + bgv[c] * (1.0f - fa);
            else {
                const float na = 1.0f - (1.0f - fa) * (1.0f - ba);
                const float rna = na > 0.0f ? 1.0f / na : 0.0f;
                o = (fgv[c] * fa + bgv[c] * ba * (1.0f - fa)) * rna;
            }
        } else o = hasAlpha ? bgv[c] + fgv[c] * fa : bgv[c] + fgv[c];
        outv[c] = o;
    }
    for (uint c = 0; c < nc; ++c) canvas[c * n + tid.x] = outv[c];
}
