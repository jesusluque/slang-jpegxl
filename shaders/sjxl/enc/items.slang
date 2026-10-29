// Copyright (c) 2026 jesus luque.
//
// The entropy-coded items of a frame, one thread each, in three passes chosen
// by params.mode: 0 counts token symbols into histograms, 1 measures each
// item's bits with the prefix codes built from them, 2 emits the bits at the
// offset the prefix sum of those lengths gave the item. The three passes walk
// the same tokens, so the layout is decided once, here.
//
// The AC groups are tokenised by transform.slang and coded with ANS by
// ac_sections.slang; the DC and AcMetadata streams are the prefix-coded items
// here.
//
// Items are laid out regularly, padded to the largest group, so an item index
// says where it belongs without a table: DC items g * 196608 + ci * 65536 + by * 256 + bx;
// AcMetadata items g * 133120 + [0,1024) YtoX tiles, [1024,2048) YtoB tiles,
// then 65536 strategies and 65536 quant-field values (the EPF channel costs
// no bits: its alphabet has one symbol, sharpness 4 through the leaf offset). Items outside their group's rect
// are 0 bits.
module enc_items;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/common/bitio.slangh"

static const uint kDcItemsPerGroup = 256 * 256 * 3;
static const uint kMetaTiles = 32 * 32;
static const uint kMetaBlocks = 256 * 256;
static const uint kMetaItemsPerGroup = 2 * kMetaTiles + 2 * kMetaBlocks;
static const uint kCodeStride = 256;
static const uint kAcClusters = 32;   // code regions 0..31 are the AC clusters

struct ItemParams {
    uint nblocks;
    uint blocksX;
    uint blocksY;
    uint numGroups;
    uint numDcGroups;
    uint mode;          // 0 histogram, 1 count, 2 emit
    uint tilesX;        // chroma-from-luma tiles across the frame
    uint pad0;
};

StructuredBuffer<uint4>    dcGroups;      // DC groups: the same
StructuredBuffer<int>      dcQuant;       // [c][block]
StructuredBuffer<uint>     rawQf;         // [block]: the quant field, 1..256
StructuredBuffer<int>      cmapX;         // [tile]: YtoX
StructuredBuffer<int>      cmapB;         // [tile]: YtoB
StructuredBuffer<uint>     codes;         // depth << 16 | bits; region r at r * 256
RWStructuredBuffer<uint>   hist;          // same regions as codes
StructuredBuffer<uint>     acs;           // per block: strategy << 1 | first
RWStructuredBuffer<uint>   itemBits;      // mode 1 output
StructuredBuffer<uint>     prefix;        // mode 2: exclusive prefix of itemBits
StructuredBuffer<uint>     sectionStart;  // mode 2: bit offset of each section
StructuredBuffer<uint>     dcPartStart;   // mode 2: per DC group: pre, dc items, mid, meta items
RWStructuredBuffer<uint>   body;          // mode 2 output
ConstantBuffer<ItemParams> params;

/// Code regions: AC clusters 0..31; DC group g: 32 + 2g (DC), 33 + 2g (AcMetadata).
uint dcCodeRegion(uint g) { return kAcClusters + 2 * g; }
uint metaCodeRegion(uint g) { return kAcClusters + 1 + 2 * g; }

struct Cursor {
    uint mode;
    uint codeBase;   // region * 256
    uint pos;        // bits so far (mode 1 and 2); absolute in mode 2
};

/// The token's code bits and raw bits, measured or emitted.
void codeToken(inout Cursor cur, uint codeBase, uint tok, uint nbits, uint raw) {
    const uint entry = codes[codeBase + tok];
    const uint depth = codeDepth(entry);
    if (cur.mode == 2) {
        emitAtomic(body, cur.pos, depth, codeBits(entry));
        emitAtomic(body, cur.pos + depth, nbits, raw);
    }
    cur.pos += depth + nbits;
}

/// One hybrid-uint value of a stream with one code (the cursor's region).
void token(inout Cursor cur, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    if (cur.mode == 0) {
        uint before;
        InterlockedAdd(hist[cur.codeBase + tok], 1u, before);
        return;
    }
    codeToken(cur, cur.codeBase, tok, nbits, raw);
}

uint physicalChannel(uint ci) { return ci == 0 ? 1 : (ci == 1 ? 0 : 2); }

[shader("compute")]
[numthreads(64, 1, 1)]
void dcItems(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.numDcGroups * kDcItemsPerGroup) {
        return;
    }
    const uint g = i / kDcItemsPerGroup;
    const uint l = i % kDcItemsPerGroup;
    const uint ci = l / 65536;
    const uint bx = (l % 65536) % 256;
    const uint by = (l % 65536) / 256;
    const uint4 rect = dcGroups[g];
    Cursor cur;
    cur.mode = params.mode;
    cur.codeBase = dcCodeRegion(g) * kCodeStride;
    cur.pos = 0;
    if (params.mode == 2) {
        cur.pos = dcPartStart[g * 4 + 1] + (prefix[i] - prefix[g * kDcItemsPerGroup]);
    }
    if (bx < rect.z && by < rect.w) {
        const uint c = physicalChannel(ci);
        const uint plane = c * params.nblocks;
        const uint block = (rect.y + by) * params.blocksX + rect.x + bx;
        const bool hasLeft = bx > 0;
        const bool hasTop = by > 0;
        const int w = hasLeft ? dcQuant[plane + block - 1] : 0;
        const int n = hasTop ? dcQuant[plane + block - params.blocksX] : 0;
        const int nw = (hasLeft && hasTop) ? dcQuant[plane + block - params.blocksX - 1] : 0;
        const int v = dcQuant[plane + block];
        token(cur, packSigned(v - gradientPredict(hasLeft, hasTop, w, n, nw)));
    }
    if (params.mode == 1) {
        itemBits[i] = cur.pos;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void metaItems(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.numDcGroups * kMetaItemsPerGroup) {
        return;
    }
    const uint g = i / kMetaItemsPerGroup;
    const uint l = i % kMetaItemsPerGroup;
    const uint4 rect = dcGroups[g];
    Cursor cur;
    cur.mode = params.mode;
    cur.codeBase = metaCodeRegion(g) * kCodeStride;
    cur.pos = 0;
    if (params.mode == 2) {
        cur.pos = dcPartStart[g * 4 + 3] + (prefix[i] - prefix[g * kMetaItemsPerGroup]);
    }
    const uint tilesX = (rect.z + 7) / 8;
    const uint tilesY = (rect.w + 7) / 8;
    if (l < 2 * kMetaTiles) {
        // YtoX then YtoB, one value per 64 px tile, from the chroma-from-luma pass.
        const uint t = l % kMetaTiles;
        const uint tx = t % 32, ty = t / 32;
        if (tx < tilesX && ty < tilesY) {
            const uint tile = (rect.y / 8 + ty) * params.tilesX + rect.x / 8 + tx;
            token(cur, packSigned(l < kMetaTiles ? cmapX[tile] : cmapB[tile]));
        }
    } else {
        const uint b = (l - 2 * kMetaTiles) % kMetaBlocks;
        const bool qfRow = (l - 2 * kMetaTiles) >= kMetaBlocks;
        const uint bx = b % 256, by = b / 256;
        if (bx < rect.z && by < rect.w) {
            // One entry per varblock (its first block): the strategy, then qf - 1.
            const uint block = (rect.y + by) * params.blocksX + rect.x + bx;
            if ((acs[block] & 1) != 0) {
                token(cur, packSigned(qfRow ? int(rawQf[block]) - 1 : int(acs[block] >> 1)));
            }
        }
    }
    if (params.mode == 1) {
        itemBits[i] = cur.pos;
    }
}
