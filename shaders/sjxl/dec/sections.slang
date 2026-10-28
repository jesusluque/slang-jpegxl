// Copyright (c) 2026 jesus luque.
//
// The entropy-coded sections of a VarDCT frame, one thread per section:
// decodeGlobals (DcGlobal and AcGlobal, and with a single TOC entry the DC
// and AC group too, since they share one bit stream), decodeDcGroups (one
// thread per DC group), decodeAcGroups (one thread per AC group).
module dec_sections;

StructuredBuffer<uint>    cs;
RWStructuredBuffer<uint>  arena;

#include "sjxl/dec/entropy.slangh"
#include "sjxl/dec/modular.slangh"
#include "sjxl/dec/frame.slangh"

RWStructuredBuffer<float> dc;
RWStructuredBuffer<uint>  quantDcBucket;
RWStructuredBuffer<uint>  acs;
RWStructuredBuffer<uint>  qf;
RWStructuredBuffer<uint>  epfSharp;
RWStructuredBuffer<int>   ytox;
RWStructuredBuffer<int>   ytob;
RWStructuredBuffer<uint>  firstBlock;   // nblocks: the varblock's first block
RWStructuredBuffer<int>   coeffs;

#include "sjxl/dec/dcgroup.slangh"
#include "sjxl/dec/acgroup.slangh"

struct SectionParams {
    uint blocksX;
    uint blocksY;
    uint numGroups;
    uint numDcGroups;
    uint singleSection;   // 1: one TOC entry; the globals thread decodes everything
    uint globalsCap;      // arena words for the globals thread (from 0)
    uint dcArenaBase;     // DC group g owns [dcArenaBase + g * dcArenaWords, +dcArenaWords)
    uint dcArenaWords;
    uint acArenaBase;     // AC group g owns [acArenaBase + g * acArenaWords, ...)
    uint acArenaWords;
    uint windowBase;      // LZ77 windows: 0xFFFFFFFF when none; else window i at windowBase + i * kLz77WindowSize
    uint windowCount;     // windows available: one per thread in order globals, DC groups, AC groups
};

StructuredBuffer<uint2>   sections;   // bit start, bit end of each TOC entry, codestream order
RWStructuredBuffer<uint>  status;     // error code per thread: [0] globals, [1 + g] DC groups, [1 + nDc + g] AC groups
ConstantBuffer<SectionParams> params;

uint windowFor(uint thread) {
    if (params.windowBase == 0xFFFFFFFFu || thread >= params.windowCount) return 0xFFFFFFFFu;
    return params.windowBase + thread * kLz77WindowSize;
}

Dec makeDec(uint section, uint base, uint cap) {
    Dec d;
    d.br.pos = sections[section].x;
    d.br.end = sections[section].y;
    d.base = base;
    d.top = base;
    d.cap = cap;
    d.error = kErrNone;
    return d;
}

DcGroupGeom dcGeom(uint g) {
    DcGroupGeom geom;
    const uint gx = g % ((params.blocksX + 255) / 256);
    const uint gy = g / ((params.blocksX + 255) / 256);
    geom.x0 = gx * 256;
    geom.y0 = gy * 256;
    geom.w = min(256u, params.blocksX - geom.x0);
    geom.h = min(256u, params.blocksY - geom.y0);
    geom.blocksX = params.blocksX;
    geom.tilesX = (params.blocksX + 7) / 8;
    geom.numDcGroups = params.numDcGroups;
    geom.group = g;
    return geom;
}

AcGroupGeom acGeom(uint g) {
    AcGroupGeom geom;
    const uint gx = g % ((params.blocksX + 31) / 32);
    const uint gy = g / ((params.blocksX + 31) / 32);
    geom.x0 = gx * 32;
    geom.y0 = gy * 32;
    geom.w = min(32u, params.blocksX - geom.x0);
    geom.h = min(32u, params.blocksY - geom.y0);
    geom.blocksX = params.blocksX;
    geom.nblocks = params.blocksX * params.blocksY;
    geom.group = g;
    return geom;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeGlobals(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    for (uint i = 0; i < kGsHeaderSize; ++i) arena[i] = 0;
    arena[kGsHeaderSize - 1] = params.blocksX * params.blocksY;   // nblocks, read by the DC group body
    Dec d = makeDec(0, kGsHeaderSize, params.globalsCap);
    const uint window = windowFor(0);
    decodeDcGlobal(d, window);
    arena[kGsDcGlobalBits] = d.br.pos - sections[0].x;
    if (d.error == kErrNone && params.singleSection != 0) {
        decodeDcGroup(d, dcGeom(0), window);
    }
    if (d.error == kErrNone) {
        if (params.singleSection == 0) {
            d.br.pos = sections[1 + params.numDcGroups].x;
            d.br.end = sections[1 + params.numDcGroups].y;
        }
        decodeAcGlobal(d, params.numGroups, window);
        arena[kGsAcGlobalBits] = d.br.pos;
    }
    if (d.error == kErrNone && params.singleSection != 0) {
        decodeAcGroup(d, acGeom(0), window);
    }
    status[0] = d.error;
    arena[kGsError] = d.error;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeDcGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numDcGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + g] = arena[kGsError]; return; }
    const uint base = params.dcArenaBase + g * params.dcArenaWords;
    Dec d = makeDec(1 + g, base, base + params.dcArenaWords);
    decodeDcGroup(d, dcGeom(g), windowFor(1 + g));
    status[1 + g] = d.error;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeAcGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + params.numDcGroups + g] = arena[kGsError]; return; }
    const uint base = params.acArenaBase + g * params.acArenaWords;
    Dec d = makeDec(2 + params.numDcGroups + g, base, base + params.acArenaWords);
    decodeAcGroup(d, acGeom(g), windowFor(1 + params.numDcGroups + g));
    status[1 + params.numDcGroups + g] = d.error;
}
