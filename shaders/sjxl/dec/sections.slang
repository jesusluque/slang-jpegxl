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
StructuredBuffer<int>     hostDc;         // DC groups decoded on the CPU (DcGroupCpu.cpp): per group extraPrecision, count, usedAcs, numVarblocks, the channels, the varblock triplets
StructuredBuffer<uint>    hostDcBase;     // [g]: word offset of DC group g in hostDc; [numDcGroups + g]: its first varblock index (numDcGroups + 1 entries)
RWStructuredBuffer<uint>  naturalOrders;  // kCoeffOrderLimit * 64: the natural coefficient orders in libjxl's slot layout (built once per decoder)

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
    uint numPasses;       // 1..11; AC group g of pass p is section 2 + nDC + p * nAC + g
    uint numEc;           // extra channels (<= 4): modular in every frame
    uint4 refW;           // reference frame sizes per slot (0: none)
    uint4 refH;
    uint4 passShiftA;     // passes.shift[0..3], [4..7], [8..11]
    uint4 passShiftB;
    uint4 passShiftC;
    int4  passMinA;       // modular: GetDownsamplingBracket per pass
    int4  passMinB;
    int4  passMinC;
    uint  perfSkip;       // SJXL_PERF_SKIP timing experiments (entropy.slangh gPerfSkip)
    uint  wideSamples;    // 1: samples beyond 16 bits (the 64-bit modular arithmetic)
    int4  passMaxA;
    int4  passMaxB;
    int4  passMaxC;
    uint4 ecW;            // extra channel sizes (ceil(output size / ec_upsampling)) and shifts
    uint4 ecH;
    uint4 ecShift;
    uint4 hshift;         // YCbCr chroma subsampling per channel (x: X/Cb, y: Y, z: B/Cr); zeros for 4:4:4 and XYB
    uint4 vshift;
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
    d.br.cacheWord = 0xFFFFFFFFu;
    d.base = base;
    d.top = base;
    d.cap = cap;
    d.error = kErrNone;
    return d;
}

uint vec4At(uint4 a, uint4 b, uint4 c, uint i) {
    const uint4 v = i < 4 ? a : (i < 8 ? b : c);
    const uint j = i & 3;
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}
int ivec4At(int4 a, int4 b, int4 c, uint i) {
    const int4 v = i < 4 ? a : (i < 8 ? b : c);
    const uint j = i & 3;
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}
uint passShift(uint p) { return vec4At(params.passShiftA, params.passShiftB, params.passShiftC, p); }
int passMinShift(uint p) { return ivec4At(params.passMinA, params.passMinB, params.passMinC, p); }
int passMaxShift(uint p) { return ivec4At(params.passMaxA, params.passMaxB, params.passMaxC, p); }
uint acSection(uint pass, uint g) { return 2 + params.numDcGroups + pass * params.numGroups + g; }

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
    geom.useDcFrame = (params.frameFlags & 32u) != 0 ? 1u : 0u;
    geom.windowOff = windowFor(1 + g);
    geom.hshift = params.hshift;
    geom.vshift = params.vshift;
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
    geom.hshift = params.hshift;
    geom.vshift = params.vshift;
    return geom;
}

// --- Modular ------------------------------------------------------------------------------

ModularFrameGeom modularGeom(uint colorChannels) {
    ModularFrameGeom g;
    g.xsize = params.xsize;
    g.ysize = params.ysize;
    g.groupDim = params.groupDim;
    g.colorChannels = colorChannels;
    g.bitDepth = params.bitDepth;
    g.numEc = params.numEc;
    g.ecW = params.ecW;
    g.ecH = params.ecH;
    g.ecShift = params.ecShift;
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

/// Stream ids (ModularStreamId): ModularDC(g) = 1 + nDC + g; ModularAC(g, pass) = 1 + 3 nDC + 17 + nAC * pass + g.
uint modularDcStream(uint g) { return 1 + params.numDcGroups + g; }
uint modularAcStream(uint g, uint pass) { return 1 + 3 * params.numDcGroups + 17 + params.numGroups * pass + g; }

// --- VarDCT -------------------------------------------------------------------------------

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeGlobals(uint3 tid: SV_DispatchThreadID) {
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
    gDeferOrders = params.singleSection == 0 ? 1u : 0u;
    if (tid.x != 0) return;
    for (uint i = 0; i < kGsHeaderSize; ++i) arena[i] = 0;
    arena[kGsHeaderSize - 1] = params.blocksX * params.blocksY;   // nblocks, read by the DC group body
    Dec d = makeDec(0, kGsHeaderSize, params.globalsCap);
    const uint window = windowFor(0);
    uint stage = 1;
    decodeDcGlobal(d, window, params.frameFlags, params.xsize, params.ysize, params.refW, params.refH, params.numEc);
    if (d.error == kErrNone) decodeModularImage(d, modularGeom(0), window);   // the extra channels
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
            d.br.cacheWord = 0xFFFFFFFFu;
        }
        decodeAcGlobal(d, params.numGroups, params.numDcGroups, params.numPasses, window);
        arena[kGsAcGlobalBits] = d.br.pos;
    }
    if (d.error == kErrNone && params.singleSection != 0) {
        stage = 4;
        decodeAcGroup(d, acGeom(0), 0, passShift(0), window);
        if (d.error == kErrNone) decodeModularGroup(d, 0, 0, params.groupDim, params.groupDim, passMinShift(0), passMaxShift(0), modularAcStream(0, 0), window);
    }
    status[0] = d.error | (d.error != kErrNone ? (stage << 8) : 0);
    arena[kGsError] = d.error;
    arena[kGsArenaTop] = d.top;
}

static const uint kNumOrderBuckets = 13;   // kCoeffOrderOffset has 3 slots per bucket plus the end

// --- Coefficient orders --------------------------------------------------------------------
// The natural orders are the same for every frame: built once into naturalOrders (one
// thread per slot), then finishOrders fills every pass's order table in parallel, mapping
// the permutation indices the globals thread left for the orders the frame codes.

[shader("compute")]
[numthreads(1, 1, 1)]
void buildNaturalOrders(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;   // 3 per order bucket
    if (slot >= 3 * kNumOrderBuckets) return;
    const uint ord = slot / 3;
    uint strategy = 0;
    for (uint o = 0; o < kNumStrategies; ++o) if (kStrategyOrder[o] == ord) { strategy = o; break; }
    naturalCoeffOrder(kCoveredX[strategy], kCoveredY[strategy], naturalOrders, kCoeffOrderOffset[slot] * 64);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void finishOrders(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    const uint p = tid.y;
    if (w >= kCoeffOrderLimit * 64 || p >= params.numPasses || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) return;
    const uint blk = w / 64;
    uint slot = 0;
    while (slot + 1 < 3 * kNumOrderBuckets && kCoeffOrderOffset[slot + 1] <= blk) ++slot;
    const uint ord = slot / 3;
    const uint t = passTable(p);
    const uint ordersOff = arena[t + 0];
    const uint usedOrders = arena[t + 2];
    if ((usedOrders & (1u << ord)) == 0) arena[ordersOff + w] = naturalOrders[w];
    else arena[ordersOff + w] = naturalOrders[kCoeffOrderOffset[slot] * 64 + arena[ordersOff + w]];
}

// --- DC groups decoded on the CPU -----------------------------------------------------------
// DcGroupCpu.cpp decodes the modular streams and walks the strategies; these kernels do what
// the DC group thread does after its streams, one thread per block, varblock and tile.

struct HostDcLayout {
    uint base;   // extraPrecision, count, usedAcs, numVarblocks
    uint y, x, b, wY, hY, wX, hX, wB, hB;
    uint ytox, ytob, tw, th;
    uint acsCh, sharp, vars;
};

HostDcLayout hostDcLayout(DcGroupGeom g) {
    HostDcLayout L;
    L.base = hostDcBase[g.group];
    uint pos = L.base + 4;
    L.wY = g.w >> shiftOf(g.hshift, 1); L.hY = g.h >> shiftOf(g.vshift, 1); L.y = pos; pos += L.wY * L.hY;
    L.wX = g.w >> shiftOf(g.hshift, 0); L.hX = g.h >> shiftOf(g.vshift, 0); L.x = pos; pos += L.wX * L.hX;
    L.wB = g.w >> shiftOf(g.hshift, 2); L.hB = g.h >> shiftOf(g.vshift, 2); L.b = pos; pos += L.wB * L.hB;
    L.tw = (g.w + 7) >> 3; L.th = (g.h + 7) >> 3;
    L.ytox = pos; pos += L.tw * L.th;
    L.ytob = pos; pos += L.tw * L.th;
    const uint count = uint(hostDc[L.base + 1]);
    L.acsCh = pos; pos += count * 2;
    L.sharp = pos; pos += g.w * g.h;
    L.vars = pos;
    return L;
}

uint dcGroupOfBlock(uint x, uint y) { return (y / params.groupDim) * xsizeDcGroups() + x / params.groupDim; }

[shader("compute")]
[numthreads(64, 1, 1)]
void hostDcBlocks(uint3 tid: SV_DispatchThreadID) {
    const uint nblocks = params.blocksX * params.blocksY;
    const uint block = tid.x;
    if (block >= nblocks || arena[kGsError] != kErrNone) return;
    const uint x = block % params.blocksX, y = block / params.blocksX;
    const DcGroupGeom g = dcGeom(dcGroupOfBlock(x, y));
    const HostDcLayout L = hostDcLayout(g);
    const uint ix = x - g.x0, iy = y - g.y0;
    const bool is444 = (g.hshift.x | g.hshift.y | g.hshift.z | g.vshift.x | g.vshift.y | g.vshift.z) == 0;
    const float mul = 1.0f / float(1u << uint(hostDc[L.base]));
    const int qy = hostDc[L.y + (iy >> shiftOf(g.vshift, 1)) * L.wY + (ix >> shiftOf(g.hshift, 1))];
    const int qx = hostDc[L.x + (iy >> shiftOf(g.vshift, 0)) * L.wX + (ix >> shiftOf(g.hshift, 0))];
    const int qb = hostDc[L.b + (iy >> shiftOf(g.vshift, 2)) * L.wB + (ix >> shiftOf(g.hshift, 2))];
    // The same products in the same order as decodeDcGroup (bit-identical floats).
    const float facX = gsF(kGsMulDc + 0) * mul;
    const float facY = gsF(kGsMulDc + 1) * mul;
    const float facB = gsF(kGsMulDc + 2) * mul;
    const float fy = float(qy) * facY;
    dc[1 * nblocks + block] = fy;
    dc[0 * nblocks + block] = is444 ? fma(fy, gsF(kGsCflDcFactorX), float(qx) * facX) : float(qx) * facX;
    dc[2 * nblocks + block] = is444 ? fma(fy, gsF(kGsCflDcFactorB), float(qb) * facB) : float(qb) * facB;
    uint bucket = 0;
    if (arena[kGsBctxNumDcCtxs] > 1) {
        uint bx = 0, by = 0, bb = 0;
        const uint nX = arena[kGsBctxDcCount + 0], nY = arena[kGsBctxDcCount + 1], nB = arena[kGsBctxDcCount + 2];
        for (uint t = 0; t < nX; ++t) if (qx > int(arena[arena[kGsBctxDcOff + 0] + t])) ++bx;
        for (uint t = 0; t < nY; ++t) if (qy > int(arena[arena[kGsBctxDcOff + 1] + t])) ++by;
        for (uint t = 0; t < nB; ++t) if (qb > int(arena[arena[kGsBctxDcOff + 2] + t])) ++bb;
        bucket = bx;
        bucket = bucket * (nB + 1) + bb;
        bucket = bucket * (nY + 1) + by;
    }
    quantDcBucket[block] = bucket;
    epfSharp[block] = uint(hostDc[L.sharp + iy * g.w + ix]);
    if (ix == 0 && iy == 0) {
        uint before;
        InterlockedOr(arena[kGsUsedAcs], uint(hostDc[L.base + 2]), before);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void hostDcVarblocks(uint3 tid: SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= hostDcBase[2 * params.numDcGroups] || arena[kGsError] != kErrNone) return;
    uint gi = 0;
    while (gi + 1 < params.numDcGroups && hostDcBase[params.numDcGroups + gi + 1] <= v) ++gi;
    const uint i = v - hostDcBase[params.numDcGroups + gi];
    const DcGroupGeom g = dcGeom(gi);
    const HostDcLayout L = hostDcLayout(g);
    const uint block = uint(hostDc[L.vars + 3 * i]);
    const uint strategy = uint(hostDc[L.vars + 3 * i + 1]);
    const uint qfStored = uint(hostDc[L.vars + 3 * i + 2]);
    const uint x = block % params.blocksX, y = block / params.blocksX;
    const uint cx = kCoveredX[strategy], cy = kCoveredY[strategy];
    for (uint by = 0; by < cy; ++by) {
        for (uint bx = 0; bx < cx; ++bx) {
            const uint b = (y + by) * params.blocksX + x + bx;
            acs[b] = (strategy << 1) | ((bx | by) == 0 ? 1u : 0u);
            qf[b] = qfStored;
            firstBlock[b] = block;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void hostDcTiles(uint3 tid: SV_DispatchThreadID) {
    const uint tilesX = (params.blocksX + 7) / 8, tilesY = (params.blocksY + 7) / 8;
    const uint tile = tid.x;
    if (tile >= tilesX * tilesY || arena[kGsError] != kErrNone) return;
    const uint tx = tile % tilesX, ty = tile / tilesX;
    const DcGroupGeom g = dcGeom(dcGroupOfBlock(tx * 8, ty * 8));
    const HostDcLayout L = hostDcLayout(g);
    const uint ltx = tx - (g.x0 >> 3), lty = ty - (g.y0 >> 3);
    ytox[tile] = clamp(hostDc[L.ytox + lty * L.tw + ltx], -128, 127);
    ytob[tile] = clamp(hostDc[L.ytob + lty * L.tw + ltx], -128, 127);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeDcGroups(uint3 tid: SV_DispatchThreadID) {
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
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
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
    const uint g = tid.x;
    if (g >= params.numGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + params.numDcGroups + g] = arena[kGsError]; return; }
    const uint base = params.acArenaBase + g * params.acArenaWords;
    Dec d = makeDec(acSection(0, g), base, base + params.acArenaWords);
    uint x0, y0, w, h;
    groupRect(g, false, x0, y0, w, h);
    for (uint pass = 0; pass < params.numPasses && d.error == kErrNone; ++pass) {
        d.br.pos = sections[acSection(pass, g)].x;
        d.br.end = sections[acSection(pass, g)].y;
        d.br.cacheWord = 0xFFFFFFFFu;
        decodeAcGroup(d, acGeom(g), pass, passShift(pass), windowFor(1 + params.numDcGroups + g));
        if (d.error == kErrNone) {   // the extra channels' modular group of this pass (ProcessACGroup)
            decodeModularGroup(d, x0, y0, w, h, passMinShift(pass), passMaxShift(pass), modularAcStream(g, pass), windowFor(1 + params.numDcGroups + g));
        }
    }
    status[1 + params.numDcGroups + g] = d.error;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeModularGlobals(uint3 tid: SV_DispatchThreadID) {
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
    if (tid.x != 0) return;
    for (uint i = 0; i < kGsHeaderSize; ++i) arena[i] = 0;
    Dec d = makeDec(0, kGsHeaderSize, params.globalsCap);
    const uint window = windowFor(0);
    uint stage = 1;
    if ((params.frameFlags & kFramePatches) != 0) {
        readPatchDictionary(d, params.xsize, params.ysize, params.refW, params.refH, params.numEc, window);
    }
    if (d.error == kErrNone && (params.frameFlags & kFrameSplines) != 0) {
        readSplines(d, params.xsize * params.ysize, window);
        gsSetF(kGsCflBaseB, 1.0f);   // the default colour correlation the splines are dequantised with (y_to_x 0, y_to_b 1)
    }
    if (d.error == kErrNone && (params.frameFlags & kFrameNoise) != 0) readNoise(d);
    if (d.error == kErrNone) decodeModularGlobal(d, modularGeom(params.colorChannels), window);
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
            decodeModularGroup(d, x0, y0, w, h, passMinShift(0), passMaxShift(0), modularAcStream(0, 0), window);
        }
    }
    status[0] = d.error | (d.error != kErrNone ? (stage << 8) : 0);
    arena[kGsError] = d.error;
    arena[kGsArenaTop] = d.top;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeModularDcGroups(uint3 tid: SV_DispatchThreadID) {
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
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
    gPerfSkip = params.perfSkip;
    gWideSamples = params.wideSamples;
    const uint g = tid.x;
    if (g >= params.numGroups || params.singleSection != 0) return;
    if (arena[kGsError] != kErrNone) { status[1 + params.numDcGroups + g] = arena[kGsError]; return; }
    const uint base = params.acArenaBase + g * params.acArenaWords;
    Dec d = makeDec(acSection(0, g), base, base + params.acArenaWords);
    uint x0, y0, w, h;
    groupRect(g, false, x0, y0, w, h);
    for (uint pass = 0; pass < params.numPasses && d.error == kErrNone; ++pass) {
        d.br.pos = sections[acSection(pass, g)].x;
        d.br.end = sections[acSection(pass, g)].y;
        d.br.cacheWord = 0xFFFFFFFFu;
        decodeModularGroup(d, x0, y0, w, h, passMinShift(pass), passMaxShift(pass), modularAcStream(g, pass), windowFor(1 + params.numDcGroups + g));
    }
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
    d.br.cacheWord = 0xFFFFFFFFu;
    d.base = params.acArenaBase;   // the AC groups' storage is free by now (palettes allocate full channels)
    d.top = d.base;
    d.cap = params.acArenaBase + params.numGroups * params.acArenaWords;
    d.error = kErrNone;
    modularFinalize(d);
    if (d.error != kErrNone) status[0] = d.error | (5u << 8);
}
