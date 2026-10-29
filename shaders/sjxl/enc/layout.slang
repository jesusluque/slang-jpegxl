// Copyright (c) 2026 jesus luque.
//
// Where every section lands in the frame body, and the blobs copied there.
// Sections in codestream order: DcGlobal, the DC groups, AcGlobal, the AC
// groups. With several TOC entries each section starts on a byte; with one
// (a single group and pass) the whole frame is one bit-packed section.
module enc_layout;

#include "sjxl/common/bitio.slangh"

static const uint kBlobStrideWords = 8192;

struct LayoutParams {
    uint numGroups;
    uint numDcGroups;
    uint packed;    // 1: single section
    uint pad0;
};

StructuredBuffer<uint>      blobBits;
StructuredBuffer<uint>      acSectionBits;   // bits of each AC group section (ANS)
StructuredBuffer<uint>      dcBits;          // bits of each DC group's DC stream (ANS)
StructuredBuffer<uint>      metaBits;        // bits of each DC group's AcMetadata stream
StructuredBuffer<uint>      blobs;
RWStructuredBuffer<uint>    sectionStart;   // bit offset
RWStructuredBuffer<uint>    sectionBits;
RWStructuredBuffer<uint>    sectionBytes;
RWStructuredBuffer<uint>    dcPartStart;    // per DC group: pre, DC items, mid, meta items
RWStructuredBuffer<uint>    totals;         // [0] bytes, [1] bits
RWStructuredBuffer<uint>    body;
ConstantBuffer<LayoutParams> params;

void place(uint s, uint bits, inout uint pos, out uint start) {
    start = params.packed != 0 ? pos : ((pos + 7) & ~7u);
    sectionStart[s] = start;
    sectionBits[s] = bits;
    sectionBytes[s] = (bits + 7) / 8;
    pos = start + bits;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void layoutSections(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint pos = 0;
    uint s = 0;
    uint start;
    place(s++, blobBits[0], pos, start);
    for (uint g = 0; g < params.numDcGroups; ++g) {
        const uint pre = blobBits[2 + 2 * g];
        const uint dcT = dcBits[g];
        const uint mid = blobBits[3 + 2 * g];
        const uint metaT = metaBits[g];
        place(s++, pre + dcT + mid + metaT, pos, start);
        dcPartStart[g * 4 + 0] = start;
        dcPartStart[g * 4 + 1] = start + pre;
        dcPartStart[g * 4 + 2] = start + pre + dcT;
        dcPartStart[g * 4 + 3] = start + pre + dcT + mid;
    }
    place(s++, blobBits[1], pos, start);
    for (uint g = 0; g < params.numGroups; ++g) {
        place(s++, acSectionBits[g], pos, start);
    }
    totals[1] = pos;
    totals[0] = (pos + 7) / 8;
}

/// One thread per blob region: its bits copied to where the layout put them.
[shader("compute")]
[numthreads(1, 1, 1)]
void blobEmit(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= 2 + 2 * params.numDcGroups) {
        return;
    }
    uint dest;
    if (r == 0) {
        dest = sectionStart[0];
    } else if (r == 1) {
        dest = sectionStart[1 + params.numDcGroups];
    } else {
        const uint g = (r - 2) / 2;
        dest = dcPartStart[g * 4 + (((r - 2) % 2 == 0) ? 0 : 2)];
    }
    const uint bits = blobBits[r];
    const uint base = r * kBlobStrideWords;
    for (uint i = 0; i < bits; i += 32) {
        const uint n = min(32u, bits - i);
        emitAtomic(body, dest + i, n, blobs[base + i / 32]);
    }
}
