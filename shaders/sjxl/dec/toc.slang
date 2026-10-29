// Copyright (c) 2026 jesus luque.
//
// A permuted TOC (progressive orderings): the permutation is entropy coded,
// so one thread decodes it and the host reads back the order and where the
// section sizes start (bookkeeping the CPU then finishes). libjxl toc.cc
// ReadToc / coeff_order.cc DecodePermutation.
module dec_toc;

StructuredBuffer<uint>    cs;
RWStructuredBuffer<uint>  arena;

#include "sjxl/dec/entropy.slangh"
#include "sjxl/dec/modular.slangh"
#include "sjxl/dec/frame.slangh"

RWStructuredBuffer<uint>  tocOut;   // [0] error, [1] bit position after the permutation, [2..] the permutation

struct TocParams {
    uint tocBit;       // first bit of the permutation (after the flag bit)
    uint tocEntries;
    uint arenaCap;
    uint pad0;
};
ConstantBuffer<TocParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeTocPermutation(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    Dec d;
    d.br.pos = params.tocBit;
    d.br.end = 0xFFFFFFFFu;
    d.base = kGsHeaderSize;
    d.top = d.base;
    d.cap = params.arenaCap - kLz77WindowSize;   // an LZ77 window at the top of the arena
    d.error = kErrNone;
    const uint dest = alloc(d, params.tocEntries);
    const uint endBit = readTocPermutation(d, params.tocEntries, dest, d.cap);
    tocOut[0] = d.error;
    tocOut[1] = endBit;
    for (uint i = 0; i < params.tocEntries; ++i) tocOut[2 + i] = d.error == kErrNone ? arena[dest + i] : i;
}

/// The byte contexts of the ICC codec (icc_codec_common.cc ICCANSContext).
uint iccByteKind1(uint b) {
    if ((b >= 97 && b <= 122) || (b >= 65 && b <= 90)) return 0;   // a-z, A-Z
    if ((b >= 48 && b <= 57) || b == 46 || b == 44) return 1;       // 0-9 . ,
    if (b == 0) return 2;
    if (b == 1) return 3;
    if (b < 16) return 4;
    if (b == 255) return 6;
    if (b > 240) return 5;
    return 7;
}
uint iccByteKind2(uint b) {
    if ((b >= 97 && b <= 122) || (b >= 65 && b <= 90)) return 0;
    if ((b >= 48 && b <= 57) || b == 46 || b == 44) return 1;
    if (b < 16) return 2;
    if (b > 240) return 3;
    return 4;
}
uint iccContext(uint i, uint b1, uint b2) { return i <= 128 ? 0 : 1 + iccByteKind1(b1) + iccByteKind2(b2) * 8; }

/// ICCReader: the encoded size (U64), histograms over 41 contexts, one symbol per
/// byte with a context from the two previous bytes, the final-state check. Reports the
/// bit after the stream; the bytes themselves (the predicted ICC) are not kept, since
/// pixels do not depend on them.
[shader("compute")]
[numthreads(1, 1, 1)]
void decodeIcc(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    Dec d;
    d.br.pos = params.tocBit;
    d.br.end = 0xFFFFFFFFu;
    d.base = kGsHeaderSize;
    d.top = d.base;
    d.cap = params.arenaCap - kLz77WindowSize;   // an LZ77 window at the top of the arena
    d.error = kErrNone;
    const uint window = d.cap;
    const uint64_t encSize64 = brU64Low(d.br);
    if (encSize64 > 268435456ull) { tocOut[0] = kErrCorrupt; return; }
    const uint encSize = uint(encSize64);
    const uint codeSet = decodeHistograms(d, 41, false, window);
    if (d.error == kErrNone) {
        SymReader r = makeSymReader(d, codeSet, window, 0);
        uint b1 = 0, b2 = 0;
        for (uint i = 0; i < encSize && d.error == kErrNone; ++i) {
            const uint b = readHybridUint(d, r, iccContext(i, b1, b2)) & 0xFFu;
            b2 = b1;
            b1 = b;
        }
        if (d.error == kErrNone && !checkFinalState(r)) d.error = kErrCorrupt;
    }
    tocOut[0] = d.error;
    tocOut[1] = d.br.pos;
    tocOut[2] = encSize;
}
