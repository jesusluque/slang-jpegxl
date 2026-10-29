// Copyright (c) 2026 jesus luque.
//
// JPEG recompression, the entropy side: jpegDecodeScan Huffman-decodes a
// baseline scan (one thread per restart interval, the whole scan when there
// are none) into quantised coefficients per component in natural order, and
// jpegToCoeffs lays those out as the VarDCT frame's quantised coefficients
// (libjxl's transposed block layout, subsampled channels at the co-located
// full-grid block), the DC values and the per-block metadata every block is
// a DCT8 with quant field 1 and no chroma-from-luma.
module enc_jpeg;

static const uint kMaxComponents = 4;

struct JpegParams {
    uint numComponents;
    uint mcusX;
    uint mcusY;
    uint restartInterval;   // MCUs per interval (0: one interval)
    uint numIntervals;
    uint scanComponents;    // components in this scan (1: non-interleaved)
    uint dataOffset;        // first scan byte in `data`
    uint dataSize;
    uint4 compIndex;        // per scan component: the component
    uint4 compDcTable;      // per scan component: Huffman table indices
    uint4 compAcTable;
    uint4 compH;            // per component (by component index): sampling factors and block grid
    uint4 compV;
    uint4 compBlocksX;
    uint4 compBlocksY;
    uint4 compOffset;       // per component: first coefficient in `coeffs`
    uint4 compScanBlocksX;  // per component: its block grid without MCU padding (non-interleaved scans)
    uint4 compScanBlocksY;
    // jpegToCoeffs
    uint blocksX;           // the frame's block grid
    uint blocksY;
    uint4 hshift;           // per XL channel (X, Y, B): chroma shifts
    uint4 vshift;
    uint4 jpegOf;           // per XL channel: the JPEG component
    uint4 dcOffset;         // per XL channel: 1024 / q0 for a kNone colour transform, 0 for YCbCr
    uint grayscale;         // 1: channels X and B are empty (zero)
    uint pad0, pad1, pad2;
};

StructuredBuffer<uint>     data;          // the JPEG file, bytes packed little-endian in words
StructuredBuffer<uint>     intervalStart; // per restart interval: byte offset from dataOffset
StructuredBuffer<uint>     huffman;       // 8 tables (dc 0..3, ac 0..3) x kHuffWords: limit[18] (largest code + 1 per length, 0 for none), valoffset[18], values[256]
RWStructuredBuffer<int>    coeffs;        // per component: blocksY x blocksX x 64, natural (row-major) order
RWStructuredBuffer<int>    acQuant;       // jpegToCoeffs: [c][block][64] in libjxl's layout
RWStructuredBuffer<int>    dcQuant;       // [c][block]
RWStructuredBuffer<uint>   nonZeros;      // [c][block]
RWStructuredBuffer<uint>   acs;           // per block: DCT8, first
RWStructuredBuffer<uint>   firstBlock;
RWStructuredBuffer<uint>   rawQf;
RWStructuredBuffer<int>    cmapX;
RWStructuredBuffer<int>    cmapB;
RWStructuredBuffer<uint>   status;        // [0] error flag
StructuredBuffer<int>      rawQuant;      // the DCT8 tables, 3 x 64 in libjxl's transposed layout (X, Y, B)
RWStructuredBuffer<uint>   blockCtx;      // jpegBlockCtx: [0] DC contexts, [1] block contexts, [2..9) luma DC thresholds, [16..) the map
ConstantBuffer<JpegParams> params;

static const uint kHuffWords = 18 + 18 + 256;
static const uint kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                                 41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                                 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

uint byteAt(uint i) { return (data[i >> 2] >> ((i & 3) * 8)) & 0xFFu; }

/// The entropy-coded bit stream: bytes with 0xFF 0x00 unstuffed, stopping at a marker.
struct BitReader {
    uint pos;      // next byte
    uint end;
    uint bits;     // buffered bits, MSB first
    uint count;
    bool marker;   // a marker was reached: zeros from here on
};

void fill(inout BitReader r) {
    while (r.count <= 24) {
        uint b = 0;
        if (!r.marker && r.pos < r.end) {
            b = byteAt(r.pos);
            if (b == 0xFFu) {
                const uint next = r.pos + 1 < r.end ? byteAt(r.pos + 1) : 0xD9u;
                if (next == 0) { r.pos += 2; }
                else { r.marker = true; b = 0; }
            } else {
                ++r.pos;
            }
        }
        r.bits |= b << (24 - r.count);
        r.count += 8;
    }
}

uint peekBits(inout BitReader r, uint n) {
    fill(r);
    return n == 0 ? 0 : r.bits >> (32 - n);
}

void skipBits(inout BitReader r, uint n) {
    r.bits <<= n;
    r.count -= n;
}

uint readBits(inout BitReader r, uint n) {
    const uint v = peekBits(r, n);
    skipBits(r, n);
    return v;
}

/// One Huffman symbol: canonical decode by code length (maxcode / valoffset tables).
uint decodeSymbol(inout BitReader r, uint table) {
    const uint base = table * kHuffWords;
    uint code = 0;
    for (uint len = 1; len <= 16; ++len) {
        code = (code << 1) | readBits(r, 1);
        if (code < huffman[base + len]) {   // a code of this length
            return huffman[base + 36 + ((huffman[base + 18 + len] + code) & 0xFFu)];
        }
    }
    return 0;   // corrupt: treated as EOB / zero
}

int extend(uint v, uint n) {
    if (n == 0) return 0;
    return int(v) < (1 << (n - 1)) ? int(v) - (1 << n) + 1 : int(v);
}

/// One block: DC difference plus AC run/size pairs into natural order.
void decodeBlock(inout BitReader r, uint dcTable, uint acTable, inout int dcPred, uint outBase) {
    for (uint i = 0; i < 64; ++i) coeffs[outBase + i] = 0;
    const uint s = decodeSymbol(r, dcTable);
    const int diff = extend(readBits(r, s & 15), s & 15);
    dcPred += diff;
    coeffs[outBase] = dcPred;
    for (uint k = 1; k < 64;) {
        const uint rs = decodeSymbol(r, 4 + acTable);
        const uint run = rs >> 4, size = rs & 15;
        if (size == 0) {
            if (run == 15) { k += 16; continue; }   // ZRL
            break;                                  // EOB
        }
        k += run;
        if (k > 63) break;
        coeffs[outBase + kZigzag[k]] = extend(readBits(r, size), size);
        ++k;
    }
}

/// One thread per restart interval of a baseline scan.
[shader("compute")]
[numthreads(1, 1, 1)]
void jpegDecodeScan(uint3 tid: SV_DispatchThreadID) {
    const uint interval = tid.x;
    if (interval >= params.numIntervals) return;
    BitReader r;
    r.pos = params.dataOffset + intervalStart[interval];
    r.end = params.dataOffset + params.dataSize;
    r.bits = 0;
    r.count = 0;
    r.marker = false;
    int dcPred[kMaxComponents] = {0, 0, 0, 0};
    const uint totalMcus = params.scanComponents == 1
        ? vec4At(params.compScanBlocksX, params.compIndex.x) * vec4At(params.compScanBlocksY, params.compIndex.x)   // non-interleaved: one block per MCU, unpadded grid
        : params.mcusX * params.mcusY;
    const uint mcusPerInterval = params.restartInterval == 0 ? totalMcus : params.restartInterval;
    const uint first = interval * mcusPerInterval;
    const uint last = min(totalMcus, first + mcusPerInterval);
    for (uint m = first; m < last; ++m) {
        if (params.scanComponents == 1) {
            const uint c = params.compIndex.x;
            const uint sw = vec4At(params.compScanBlocksX, c);
            const uint bw = vec4At(params.compBlocksX, c);
            const uint bx = m % sw, by = m / sw;
            decodeBlock(r, params.compDcTable.x, params.compAcTable.x, dcPred[c], vec4At(params.compOffset, c) + (by * bw + bx) * 64);
            continue;
        }
        const uint mx = m % params.mcusX, my = m / params.mcusX;
        for (uint sc = 0; sc < params.scanComponents; ++sc) {
            const uint c = vec4At(params.compIndex, sc);
            const uint h = vec4At(params.compH, c), v = vec4At(params.compV, c);
            const uint bw = vec4At(params.compBlocksX, c);
            for (uint yy = 0; yy < v; ++yy) {
                for (uint xx = 0; xx < h; ++xx) {
                    const uint bx = mx * h + xx, by = my * v + yy;
                    decodeBlock(r, vec4At(params.compDcTable, sc), vec4At(params.compAcTable, sc), dcPred[c], vec4At(params.compOffset, c) + (by * bw + bx) * 64);
                }
            }
        }
    }
    if (r.marker && r.pos + 2 < r.end && interval + 1 == params.numIntervals) { }   // trailing bytes are fine
}

uint vec4At(uint4 v, uint i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

/// One thread per (XL channel, full-grid block): the frame's quantised
/// coefficients, DC and metadata from the JPEG components.
[shader("compute")]
[numthreads(64, 1, 1)]
void jpegToCoeffs(uint3 tid: SV_DispatchThreadID) {
    const uint nb = params.blocksX * params.blocksY;
    if (tid.x >= 3 * nb) return;
    const uint c = tid.x / nb;
    const uint block = tid.x % nb;
    const uint bx = block % params.blocksX, by = block / params.blocksX;
    if (c == 0) {
        acs[block] = (0u << 1) | 1u;   // DCT8, a first block
        firstBlock[block] = block;
        rawQf[block] = 1;
        if ((bx & 7) == 0 && (by & 7) == 0) {   // the chroma-from-luma factors are per 64 px tile
            const uint tile = (by / 8) * ((params.blocksX + 7) / 8) + bx / 8;
            cmapX[tile] = 0;
            cmapB[tile] = 0;
        }
    }
    const uint hs = vec4At(params.hshift, c), vs = vec4At(params.vshift, c);
    const uint base = (c * nb + block) * 64;
    for (uint i = 0; i < 64; ++i) acQuant[base + i] = 0;
    dcQuant[c * nb + block] = 0;
    nonZeros[c * nb + block] = 0;
    if (((bx >> hs) << hs) != bx || ((by >> vs) << vs) != by) return;   // a subsampled channel codes one block per 2x2
    if (params.grayscale != 0 && c != 1) return;
    const uint comp = vec4At(params.jpegOf, c);
    const uint sbx = bx >> hs, sby = by >> vs;
    const uint bw = vec4At(params.compBlocksX, comp), bh = vec4At(params.compBlocksY, comp);
    if (sbx >= bw || sby >= bh) return;   // beyond the component's grid (zero)
    const uint src = vec4At(params.compOffset, comp) + (sby * bw + sbx) * 64;
    uint nz = 0;
    for (uint y = 0; y < 8; ++y) {
        for (uint x = 0; x < 8; ++x) {
            const int v = coeffs[src + y * 8 + x];
            acQuant[base + x * 8 + y] = v;   // libjxl's [horizontal][vertical] layout for DCT8
            if ((x | y) != 0 && v != 0) ++nz;
        }
    }
    dcQuant[c * nb + block] = coeffs[src] + int(vec4At(params.dcOffset, c));
    nonZeros[c * nb + block] = nz;
}

uint log2Ceil(uint x) {
    uint n = 0;
    while ((1u << n) < x) ++n;
    return n;
}

/// One thread: libjxl's block context map for a JPEG transcode (enc_frame.cc).
/// The luma DC histogram is cut into equal buckets (1..7 thresholds from the
/// block count and the first chroma table's low frequencies); Y gets one AC
/// context per bucket, X and B one per two buckets (one for both when
/// grayscale). Capped at 8 block contexts so the map takes the simple
/// context-map form. blockCtx: [0] DC contexts, [1] block contexts,
/// [2..9) thresholds, [16..) the map (3 x 13 orders x DC contexts).
groupshared uint gDcHist[2048];
[shader("compute")]
[numthreads(1, 1, 1)]
void jpegBlockCtx(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint nb = params.blocksX * params.blocksY;
    for (uint i = 0; i < 2048; ++i) gDcHist[i] = 0;
    for (uint b = 0; b < nb; ++b) {
        gDcHist[uint(clamp(dcQuant[nb + b] + 1024, 0, 2047))]++;
    }
    uint sumQt = 0;
    for (uint i = 1; i <= 5; ++i) sumQt += uint(max(rawQuant[i], 1));
    const int maxThr = params.grayscale != 0 ? 6 : 3;
    const int numThr = clamp(int(log2Ceil(nb)) - int(log2Ceil(sumQt)) - 7, 1, maxThr);
    uint cumsum = 0;
    uint cut = nb / uint(numThr + 1);
    uint n = 0;
    for (uint j = 0; j < 2048; ++j) {
        cumsum += gDcHist[j];
        if (cumsum > cut && n < uint(numThr)) {
            blockCtx[2 + n] = uint(int(j) - 1025);
            ++n;
            cut = nb * (n + 1) / uint(numThr + 1);
        }
    }
    const uint numDc = n + 1;
    blockCtx[0] = numDc;
    for (uint i = 0; i < 3 * 13 * numDc; ++i) blockCtx[16 + i] = 0;
    uint maxCtx = 0;
    for (uint i = 0; i < numDc; ++i) {
        uint x, b;
        if (params.grayscale != 0) {
            x = numDc;
            b = numDc;
        } else {
            x = numDc + i / 2;
            b = numDc + (numDc - 1) / 2 + 1 + i / 2;
        }
        blockCtx[16 + i] = i;
        blockCtx[16 + 13 * numDc + i] = x;
        blockCtx[16 + 2 * 13 * numDc + i] = b;
        maxCtx = max(maxCtx, max(x, b));
    }
    blockCtx[1] = maxCtx + 1;
}
