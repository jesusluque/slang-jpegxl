// Copyright (c) 2026 jesus luque.
//
// The lossless palette transforms of a Modular frame (modular/transform/
// enc_palette.cc), on int planes: paletteCount hashes every pixel's colour
// into a table with counts, paletteBuild (one thread) orders the colours as
// libjxl does (frequent colours by descending luma, then the rest ascending)
// into the palette channel and numbers them, paletteIndex writes the index
// channel. compactStats / compactBuild / compactApply are the single-channel
// palette (FLIF's ChannelCompact): the channel's distinct values, in order,
// become its palette and the samples their ranks.
module enc_palette;

static const uint kTableSize = 8192;      // hash slots (at most 1024 colours are ever kept)
static const uint kEmpty = 0xFFFFFFFFu;
static const uint kMaxColors = 1024;
static const uint kValueBins = 65536;     // compaction: 16-bit values

struct PaletteParams {
    uint width;
    uint height;
    uint numChannels;   // channels the colour palette covers (3 or 4)
    uint maxColors;     // libjxl's limit for this image
    uint channel;       // compaction: the channel
    uint channelOffset; // its samples' offset in planesIn / planesOut
    uint paletteOffset; // where the palette channel's values go in planesOut
    uint indexOffset;   // where the index channel's values go
    uint bitDepth;
    uint pad0, pad1, pad2;
};

StructuredBuffer<int>       planesIn;     // flat channels (channel c at c * width * height)
RWStructuredBuffer<int>     planesOut;
RWStructuredBuffer<uint>    table;        // kTableSize x 4: fingerprint (kEmpty: free), key0, key1, count; then [kTableSize*4] distinct, [+1] overflow, [+2] index map base
RWStructuredBuffer<uint>    bins;         // compaction: used-value bitmap (kValueBins / 32), then ranks (kValueBins)
RWStructuredBuffer<int>     stats;        // compaction: [0] min, [1] max, [2] distinct; paletteBuild: [3] nb_colors
ConstantBuffer<PaletteParams> params;

uint2 colourKey(uint i) {
    const uint n = params.width * params.height;
    // 16 bits per channel, up to four channels.
    uint k0 = 0, k1 = 0;
    for (uint c = 0; c < params.numChannels; ++c) {
        const uint v = uint(planesIn[c * n + i]) & 0xFFFFu;
        if (c < 2) k0 |= v << (16 * c);
        else k1 |= v << (16 * (c - 2));
    }
    return uint2(k0, k1);
}

uint hashKey(uint2 k) {
    uint h = k.x * 0x9E3779B1u ^ (k.y + 0x7F4A7C15u) * 0x85EBCA77u;
    h ^= h >> 15; h *= 0xC2B2AE3Du; h ^= h >> 13;
    return h;
}

/// A 32-bit fingerprint of the colour key (never kEmpty); the slot is claimed on
/// it with one compare-exchange, the full key published after. Threads never
/// wait on each other (a wait inside a SIMD group never ends): paletteVerify
/// checks every pixel's full key against its slot afterwards and aborts the
/// palette on a fingerprint collision.
uint fingerprint(uint2 k) {
    uint h = hashKey(k) ^ (k.y * 0x27D4EB2Fu);
    return h == kEmpty ? 0x12345678u : h;
}

/// One thread per pixel: its colour counted in the table; the distinct count and
/// an overflow flag (more colours than the limit, or the table full).
[shader("compute")]
[numthreads(64, 1, 1)]
void paletteCount(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint2 key = colourKey(tid.x);
    const uint fp = fingerprint(key);
    uint slot = hashKey(key) & (kTableSize - 1);
    for (uint probe = 0; probe < kTableSize; ++probe) {
        const uint base = slot * 4;
        uint prev;
        InterlockedCompareExchange(table[base + 0], kEmpty, fp, prev);
        if (prev == kEmpty || prev == fp) {
            if (prev == kEmpty) {
                table[base + 1] = key.x;
                table[base + 2] = key.y;
                uint distinct;
                InterlockedAdd(table[kTableSize * 4], 1u, distinct);
                if (distinct + 1 > params.maxColors) table[kTableSize * 4 + 1] = 1;
            }
            InterlockedAdd(table[base + 3], 1u, prev);
            return;
        }
        slot = (slot + 1) & (kTableSize - 1);
    }
    table[kTableSize * 4 + 1] = 1;   // table full
}

/// One thread per pixel: the slot its fingerprint claimed must hold its full key
/// (two colours with one fingerprint set the overflow flag: no palette).
[shader("compute")]
[numthreads(64, 1, 1)]
void paletteVerify(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint2 key = colourKey(tid.x);
    const uint fp = fingerprint(key);
    uint slot = hashKey(key) & (kTableSize - 1);
    for (uint probe = 0; probe < kTableSize; ++probe) {
        const uint base = slot * 4;
        if (table[base + 0] == fp) {
            if (table[base + 1] != key.x || table[base + 2] != key.y) table[kTableSize * 4 + 1] = 1;
            return;
        }
        if (table[base + 0] == kEmpty) { table[kTableSize * 4 + 1] = 1; return; }
        slot = (slot + 1) & (kTableSize - 1);
    }
}

groupshared uint gOrder[kMaxColors];
groupshared float gLuma[kMaxColors];

/// One thread: the colours in libjxl's order into the palette channel (one row
/// per component) and their indices in the table.
[shader("compute")]
[numthreads(1, 1, 1)]
void paletteBuild(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint nc = params.numChannels;
    uint count = 0;
    for (uint s = 0; s < kTableSize && count < kMaxColors; ++s) {
        if (table[s * 4] == kEmpty) continue;
        const uint2 key = uint2(table[s * 4 + 1], table[s * 4 + 2]);
        int comp[4];
        for (uint c = 0; c < 4; ++c) comp[c] = c < 2 ? int((key.x >> (16 * c)) & 0xFFFFu) : int((key.y >> (16 * (c - 2))) & 0xFFFFu);
        // Sign-extend 16-bit components (the planes are small ints).
        for (uint c = 0; c < 4; ++c) comp[c] = (comp[c] << 16) >> 16;
        float luma = 0.299f * float(comp[0]) + 0.587f * float(comp[1]) + 0.114f * float(comp[2]) + 0.1f;
        if (nc > 3) luma *= 1.0f + float(comp[3]);
        if (table[s * 4 + 3] > 4) luma = -luma;   // frequent colours first, by descending luma
        gLuma[count] = luma;
        gOrder[count] = s;
        ++count;
    }
    // Insertion sort by luma (stable: libjxl's std::sort is not, but ties are rare).
    for (uint i = 1; i < count; ++i) {
        const float l = gLuma[i];
        const uint o = gOrder[i];
        uint j = i;
        while (j > 0 && gLuma[j - 1] > l) {
            gLuma[j] = gLuma[j - 1];
            gOrder[j] = gOrder[j - 1];
            --j;
        }
        gLuma[j] = l;
        gOrder[j] = o;
    }
    for (uint i = 0; i < count; ++i) {
        const uint s = gOrder[i];
        const uint2 key = uint2(table[s * 4 + 1], table[s * 4 + 2]);
        for (uint c = 0; c < nc; ++c) {
            int v = c < 2 ? int((key.x >> (16 * c)) & 0xFFFFu) : int((key.y >> (16 * (c - 2))) & 0xFFFFu);
            v = (v << 16) >> 16;
            planesOut[params.paletteOffset + c * count + i] = v;
        }
        table[kTableSize * 4 + 2 + s] = i;   // slot -> palette index
    }
    stats[3] = int(count);
}

/// One thread per pixel: its colour's palette index.
[shader("compute")]
[numthreads(64, 1, 1)]
void paletteIndex(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint2 key = colourKey(tid.x);
    const uint fp = fingerprint(key);
    uint slot = hashKey(key) & (kTableSize - 1);
    for (uint probe = 0; probe < kTableSize; ++probe) {
        const uint base = slot * 4;
        if (table[base + 0] == fp) {
            planesOut[params.indexOffset + tid.x] = int(table[kTableSize * 4 + 2 + slot]);
            return;
        }
        slot = (slot + 1) & (kTableSize - 1);
    }
}

// --- single-channel compaction -----------------------------------------------------------

/// One thread per sample of the channel: the used-value bitmap and the range.
/// The raw channels are unsigned up to 16 bits; anything else marks the channel
/// as not compactable (stats[3]).
[shader("compute")]
[numthreads(64, 1, 1)]
void compactStats(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const int v = planesIn[params.channelOffset + tid.x];
    if (v < 0 || v >= int(kValueBins)) { stats[3] = 1; return; }
    const uint u = uint(v);
    uint before;
    InterlockedOr(bins[u >> 5], 1u << (u & 31), before);
    InterlockedMin(stats[0], v);
    InterlockedMax(stats[1], v);
}

/// One thread: the ranks of the used values (the palette: the values in order)
/// and the distinct count.
[shader("compute")]
[numthreads(1, 1, 1)]
void compactBuild(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    uint rank = 0;
    for (uint u = 0; u < kValueBins; ++u) {
        if ((bins[u >> 5] & (1u << (u & 31))) != 0) {
            bins[kValueBins / 32 + u] = rank;
            if (rank < kMaxColors * 64) planesOut[params.paletteOffset + rank] = int(u);
            ++rank;
        }
    }
    stats[2] = int(rank);
}

/// One thread per sample: the value's rank into the index channel.
[shader("compute")]
[numthreads(64, 1, 1)]
void compactApply(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height;
    if (tid.x >= n) return;
    const uint u = uint(max(planesIn[params.channelOffset + tid.x], 0));
    planesOut[params.indexOffset + tid.x] = int(bins[kValueBins / 32 + min(u, kValueBins - 1)]);
}
