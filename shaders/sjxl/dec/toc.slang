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
    d.cap = params.arenaCap;
    d.error = kErrNone;
    const uint dest = alloc(d, params.tocEntries);
    const uint endBit = readTocPermutation(d, params.tocEntries, dest);
    tocOut[0] = d.error;
    tocOut[1] = endBit;
    for (uint i = 0; i < params.tocEntries; ++i) tocOut[2 + i] = d.error == kErrNone ? arena[dest + i] : i;
}
