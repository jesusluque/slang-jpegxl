// Copyright (c) 2026 jesus luque.
//
// The entropy-coded sections of a frame, one thread per section. VarDCT:
// decodeGlobals (DcGlobal and AcGlobal, and with a single TOC entry the DC
// and AC group too, since they share one bit stream), decodeDcGroups (one
// thread per DC group), decodeAcGroups (one thread per AC group). Modular:
// decodeModularGlobals, decodeModularDcGroups, decodeModularAcGroups and
// modularFinalize (modframe.slangh).
module dec_sections;

StructuredBuffer<uint>    cs;
RWStructuredBuffer<uint>  arena;

#include "sjxl/dec/entropy.slangh"
#include "sjxl/dec/modular.slangh"
#include "sjxl/dec/frame.slangh"
#include "sjxl/dec/modframe.slangh"

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
    uint groupDim;        // AC group size in pixels (128 << group_size_shift); DC groups are 8x that
    uint xsize;           // frame size in pixels
    uint ysize;
    uint colorChannels;   // modular: 1 (grey) or 3
    uint bitDepth;        // modular: bits per sample
    uint frameFlags;      // FrameHeader flags (bit 1: patches)
    uint pad1, pad2;
    uint4 refW;           // reference frame sizes per slot (0: none)
    uint4 refH;
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

uint groupBlocks() { return params.groupDim / 8; }
uint xsizeGroups() { return (params.blocksX + groupBlocks() - 1) / groupBlocks(); }
uint xsizeDcGroups() { return (params.blocksX + params.groupDim - 1) / params.groupDim; }

DcGroupGeom dcGeom(uint g) {
    DcGroupGeom geom;
    const uint gx = g % xsizeDcGroups();
    const uint gy = g / xsizeDcGroups();
    geom.x0 = gx * params.groupDim;
    geom.y0 = gy * params.groupDim;
    geom.w = min(params.groupDim, params.blocksX - geom.x0);
    geom.h = min(params.groupDim, params.blocksY - geom.y0);
    geom.blocksX = params.blocksX;
    geom.tilesX = (params.blocksX + 7) / 8;
    geom.numDcGroups = params.numDcGroups;
    geom.group = g;
    geom.groupBlocks = groupBlocks();
    return geom;
}

AcGroupGeom acGeom(uint g) {
    AcGroupGeom geom;
    const uint gb = groupBlocks();
    const uint gx = g % xsizeGroups();
    const uint gy = g / xsizeGroups();
    geom.x0 = gx * gb;
    geom.y0 = gy * gb;
    geom.w = min(gb, params.blocksX - geom.x0);
    geom.h = min(gb, params.blocksY - geom.y0);
    geom.blocksX = params.blocksX;
    geom.nblocks = params.blocksX * params.blocksY;
    geom.group = g;
    return geom;
}

// --- VarDCT -------------------------------------------------------------------------------

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeGlobals(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    for (uint i = 0; i < kGsHeaderSize; ++i) arena[i] = 0;
    arena[kGsHeaderSize - 1] = params.blocksX * params.blocksY;   // nblocks, read by the DC group body
    Dec d = makeDec(0, kGsHeaderSize, params.globalsCap);
    const uint window = windowFor(0);
    uint stage = 1;
    decodeDcGlobal(d, window, params.frameFlags, params.xsize, params.ysize, params.refW, params.refH);
    arena[kGsDcGlobalBits] = d.br.pos - sections[0].x;
    if (d.error == kErrNone && params.singleSection != 0) {
        stage = 2;
        decodeDcGroup(d, dcGeom(0), window);
    }
    if (d.error == kErrNone) {
        stage = 3;
        if (params.singleSection == 0) {
            d.br.pos = sections[1 + params.numDcGroups].x;
            d.br.end = sections[1 + params.numDcGroups].y;
        }
        decodeAcGlobal(d, params.numGroups, window);
        arena[kGsAcGlobalBits] = d.br.pos;
    }
    if (d.error == kErrNone && params.singleSection != 0) {
        stage = 4;
        decodeAcGroup(d, acGeom(0), window);
    }
    status[0] = d.error | (d.error != kErrNone ? (stage << 8) : 0);
    arena[kGsError] = d.error;
    arena[kGsArenaTop] = d.top;
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

// --- Modular ------------------------------------------------------------------------------

ModularFrameGeom modularGeom() {
    ModularFrameGeom g;
    g.xsize = params.xsize;
    g.ysize = params.ysize;
    g.groupDim = params.groupDim;
    g.colorChannels = params.colorChannels;
    g.bitDepth = params.bitDepth;
    return g;
}

/// Pixel rectangle of DC group g (groupDim * 8 pixels) or AC group g (groupDim pixels).
/// Unclipped, as libjxl's mrect: each (shifted) channel clips it to its own size,
/// which is not the same as shifting the clipped rectangle for odd sizes.
void groupRect(uint g, bool dcGroup, out uint x0, out uint y0, out uint w, out uint h) {
    const uint dim = dcGroup ? params.groupDim * 8 : params.groupDim;
    const uint perRow = (params.xsize + dim - 1) / dim;
    x0 = (g % perRow) * dim;
    y0 = (g / perRow) * dim;
    w = dim;
    h = dim;
}

/// Stream ids (ModularStreamId): ModularDC(g) = 1 + nDC + g; ModularAC(g, pass 0) = 1 + 3 nDC + 17 + g.
uint modularDcStream(uint g) { return 1 + params.numDcGroups + g; }
uint modularAcStream(uint g) { return 1 + 3 * params.numDcGroups + 17 + g; }

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeModularGlobals(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    for (uint i = 0; i < kGsHeaderSize; ++i) arena[i] = 0;
    Dec d = makeDec(0, kGsHeaderSize, params.globalsCap);
    const uint window = windowFor(0);
    uint stage = 1;
    if ((params.frameFlags & kFramePatches) != 0) {
        readPatchDictionary(d, params.xsize, params.ysize, params.refW, params.refH, window);
    }
    if (d.error == kErrNone) decodeModularGlobal(d, modularGeom(), window);
    arena[kGsDcGlobalBits] = d.br.pos - sections[0].x;
    if (d.error == kErrNone && params.singleSection != 0) {
        // One section: DC group 0, (an empty AcGlobal), AC group 0 follow in the same bits.
        uint x0, y0, w, h;
        stage = 2;
        groupRect(0, true, x0, y0, w, h);
        decodeModularGroup(d, x0, y0, w, h, 3, 1000, modularDcStream(0), window);
        if (d.error == kErrNone) {
            stage = 4;
            groupRect(0, false, x0, y0, w, h);
            decodeModularGroup(d, x0, y0, w, h, 0, 2, modularAcStream(0), window);
        }
    }
    status[0] = d.error | (d.error != kErrNone ? (stage << 8) : 0);
    arena[kGsError] = d.error;
    arena[kGsArenaTop] = d.top;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeModularDcGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numDcGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + g] = arena[kGsError]; return; }
    const uint base = params.dcArenaBase + g * params.dcArenaWords;
    Dec d = makeDec(1 + g, base, base + params.dcArenaWords);
    uint x0, y0, w, h;
    groupRect(g, true, x0, y0, w, h);
    decodeModularGroup(d, x0, y0, w, h, 3, 1000, modularDcStream(g), windowFor(1 + g));
    status[1 + g] = d.error;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeModularAcGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + params.numDcGroups + g] = arena[kGsError]; return; }
    const uint base = params.acArenaBase + g * params.acArenaWords;
    Dec d = makeDec(2 + params.numDcGroups + g, base, base + params.acArenaWords);
    uint x0, y0, w, h;
    groupRect(g, false, x0, y0, w, h);
    decodeModularGroup(d, x0, y0, w, h, 0, 2, modularAcStream(g), windowFor(1 + params.numDcGroups + g));
    status[1 + params.numDcGroups + g] = d.error;
}

/// After every group: the inverse transforms over the full image. Reports through
/// status[0] like the globals thread (stage 5).
[shader("compute")]
[numthreads(1, 1, 1)]
void modularFinalizeFrame(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    for (uint t = 0; t < 1 + params.numDcGroups + params.numGroups; ++t) {
        if ((status[t] & 0xFFu) != kErrNone) return;
    }
    Dec d;
    d.br.pos = 0;
    d.br.end = 0;
    d.base = params.acArenaBase;   // the AC groups' storage is free by now (palettes allocate full channels)
    d.top = d.base;
    d.cap = params.acArenaBase + params.numGroups * params.acArenaWords;
    d.error = kErrNone;
    modularFinalize(d);
    if (d.error != kErrNone) status[0] = d.error | (5u << 8);
}
