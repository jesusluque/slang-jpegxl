// Copyright (c) 2026 jesus luque.
//
// Animations of partial frames: where two frames differ, and the bytes of a
// frame's sub-rectangle laid out as the uploaded image of that rectangle.
module enc_crop;

struct CropParams {
    uint srcWidth;        // the full frame in pixels
    uint srcHeight;
    uint bytesPerPixel;   // channels x bytes per sample
    uint x0;              // the rectangle (pixels)
    uint y0;
    uint width;
    uint height;
    uint pad0;
};

StructuredBuffer<uint>     src;      // the full frame's bytes, little-endian words
StructuredBuffer<uint>     other;    // changeBounds: the frame to compare with (same layout)
RWStructuredBuffer<uint>   dst;      // cropBytes: the rectangle's bytes, little-endian words
RWStructuredBuffer<uint>   bounds;   // changeBounds: [0] min x, [1] min y, [2] max x, [3] max y of the differing pixels
ConstantBuffer<CropParams> params;

uint byteAt(StructuredBuffer<uint> buf, uint i) { return (buf[i >> 2] >> ((i & 3) * 8)) & 0xFFu; }

/// One thread per pixel of the full frame: the bounding box of the pixels whose
/// bytes differ between `src` and `other` (bounds start at {~0, ~0, 0, 0}).
[shader("compute")]
[numthreads(8, 8, 1)]
void changeBounds(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x, y = tid.y;
    if (x >= params.srcWidth || y >= params.srcHeight) return;
    const uint base = (y * params.srcWidth + x) * params.bytesPerPixel;
    bool differs = false;
    for (uint b = 0; b < params.bytesPerPixel; ++b) {
        if (byteAt(src, base + b) != byteAt(other, base + b)) differs = true;
    }
    if (!differs) return;
    uint prev;
    InterlockedMin(bounds[0], x, prev);
    InterlockedMin(bounds[1], y, prev);
    InterlockedMax(bounds[2], x, prev);
    InterlockedMax(bounds[3], y, prev);
}

/// One thread per word of the rectangle's bytes.
[shader("compute")]
[numthreads(64, 1, 1)]
void cropBytes(uint3 tid: SV_DispatchThreadID) {
    const uint rowBytes = params.width * params.bytesPerPixel;
    const uint total = rowBytes * params.height;
    const uint first = tid.x * 4;
    if (first >= total) return;
    uint word = 0;
    for (uint k = 0; k < 4; ++k) {
        const uint i = first + k;
        if (i >= total) break;
        const uint y = i / rowBytes;
        const uint rem = i - y * rowBytes;
        const uint srcIndex = ((params.y0 + y) * params.srcWidth + params.x0) * params.bytesPerPixel + rem;
        word |= byteAt(src, srcIndex) << (k * 8);
    }
    dst[tid.x] = word;
}
