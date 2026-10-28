// Copyright (c) 2026 jesus luque.
//
// Patches: each position of the dictionary (decoded by the globals thread,
// frame.slangh readPatchDictionary) blends a rectangle of a reference frame
// into the XYB planes, after the loop filters and before the colour transform
// (libjxl PatchDictionary::AddOneRow / PerformBlending, colour channels only).
// libjxl applies the positions covering a pixel in dictionary order, which
// matters when they overlap (additive patches do overlap), so the work is one
// thread per pixel applying its covering positions in order. A single-thread
// pass first buckets the positions by 32-row band so each pixel scans only the
// positions near it.
module dec_patches;

RWStructuredBuffer<uint>  arena;
RWStructuredBuffer<float> xyb;    // the frame's 3 planes, paddedWidth * paddedHeight
StructuredBuffer<float>   ref0;   // reference frames' planes (slot 0..3), their own padded size
StructuredBuffer<float>   ref1;
StructuredBuffer<float>   ref2;
StructuredBuffer<float>   ref3;

struct PatchParams {
    uint  width;          // frame size in pixels
    uint  height;
    uint  paddedWidth;
    uint  paddedHeight;
    uint  arenaCap;       // words available to the bucket pass (the globals region)
    uint  pad0, pad1, pad2;
    uint4 refPaddedW;
    uint4 refPaddedH;
};
ConstantBuffer<PatchParams> params;

static const uint kGsPatchNumRef = 56, kGsPatchRefOff = 57, kGsPatchNumPos = 58, kGsPatchPosOff = 59, kGsArenaTop = 60;
static const uint kGsPatchBandsOff = 61;   // bucket pass output: counts[bands], then lists[bands * numPos]; 0 when it did not fit
static const uint kPatchRefWords = 5, kPatchPosWords = 4;
static const uint kBandRows = 32;

uint numBands() { return (params.height + kBandRows - 1) / kBandRows; }

/// One thread: positions by row band, in dictionary order.
[shader("compute")]
[numthreads(1, 1, 1)]
void bucketPatches(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint numPos = arena[kGsPatchNumPos];
    const uint bands = numBands();
    const uint top = arena[kGsArenaTop];
    arena[kGsPatchBandsOff] = 0;
    if (numPos == 0) return;
    if (top + bands + bands * numPos > params.arenaCap) return;   // the pixel pass scans every position instead
    const uint counts = top;
    const uint lists = top + bands;
    for (uint b = 0; b < bands; ++b) arena[counts + b] = 0;
    const uint posOff = arena[kGsPatchPosOff];
    const uint refOff = arena[kGsPatchRefOff];
    for (uint p = 0; p < numPos; ++p) {
        const uint po = posOff + p * kPatchPosWords;
        const uint py = arena[po + 1];
        const uint h = arena[refOff + arena[po + 2] * kPatchRefWords + 4];
        const uint b0 = py / kBandRows;
        const uint b1 = min((py + h - 1) / kBandRows, bands - 1);
        for (uint b = b0; b <= b1; ++b) {
            arena[lists + b * numPos + arena[counts + b]] = p;
            arena[counts + b] += 1;
        }
    }
    arena[kGsPatchBandsOff] = counts;
}

float refSample(uint ref, uint c, uint x, uint y) {
    const uint pw = ref == 0 ? params.refPaddedW.x : ref == 1 ? params.refPaddedW.y : ref == 2 ? params.refPaddedW.z : params.refPaddedW.w;
    const uint ph = ref == 0 ? params.refPaddedH.x : ref == 1 ? params.refPaddedH.y : ref == 2 ? params.refPaddedH.z : params.refPaddedH.w;
    const uint at = c * pw * ph + y * pw + x;
    switch (ref) {
    case 0: return ref0[at];
    case 1: return ref1[at];
    case 2: return ref2[at];
    default: return ref3[at];
    }
}

void applyOne(uint p, uint x, uint y, inout float v[3]) {
    const uint po = arena[kGsPatchPosOff] + p * kPatchPosWords;
    const uint px = arena[po + 0], py = arena[po + 1];
    if (x < px || y < py) return;
    const uint ro = arena[kGsPatchRefOff] + arena[po + 2] * kPatchRefWords;
    const uint w = arena[ro + 3], h = arena[ro + 4];
    if (x >= px + w || y >= py + h) return;
    const uint mode = arena[po + 3] & 0xFFu;
    if (mode == 0) return;   // kNone
    const bool clampFg = (arena[po + 3] >> 8) != 0;
    const uint ref = arena[ro + 0], rx = arena[ro + 1] + (x - px), ry = arena[ro + 2] + (y - py);
    for (uint c = 0; c < 3; ++c) {
        const float fg = refSample(ref, c, rx, ry);
        if (mode == 1 || mode == 4 || mode == 5) v[c] = fg;                                  // replace; blend above/below without alpha
        else if (mode == 2 || mode == 6 || mode == 7) v[c] += fg;                            // add; alpha-weighted add without alpha
        else v[c] *= clampFg ? clamp(fg, 0.0f, 1.0f) : fg;                                   // mul
    }
}

/// One thread per pixel of the frame.
[shader("compute")]
[numthreads(64, 1, 1)]
void applyPatches(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint x = tid.x % params.width;
    const uint y = tid.x / params.width;
    const uint numPos = arena[kGsPatchNumPos];
    if (numPos == 0) return;
    const uint plane = params.paddedWidth * params.paddedHeight;
    const uint at = y * params.paddedWidth + x;
    float v[3] = {xyb[at], xyb[plane + at], xyb[2 * plane + at]};
    const uint counts = arena[kGsPatchBandsOff];
    if (counts != 0) {
        const uint bands = numBands();
        const uint b = y / kBandRows;
        const uint list = counts + bands + b * numPos;
        const uint count = arena[counts + b];
        for (uint i = 0; i < count; ++i) applyOne(arena[list + i], x, y, v);
    } else {
        for (uint p = 0; p < numPos; ++p) applyOne(p, x, y, v);
    }
    xyb[at] = v[0];
    xyb[plane + at] = v[1];
    xyb[2 * plane + at] = v[2];
}
