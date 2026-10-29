// Copyright (c) 2026 jesus luque.
//
// Ports of enc/dc_tokens.slang (dcTokenize, dcGroupSections), enc/ac_sections.slang
// (acSections) and enc/layout.slang; see EntropyCpu.h.

#include "sjxl/encoder/EntropyCpu.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <future>

#include "sjxl/codestream/WeightedPredictor.h"

namespace sjxl::enc::cpu {
namespace {

constexpr uint32_t kModTreeWords = 8;
constexpr uint32_t kPredLeft = 1, kPredGradient = 5, kPredWeighted = 6;
constexpr uint32_t kMetaTiles = 32 * 32;
constexpr uint32_t kMetaBlocks = 256 * 256;
constexpr uint32_t kMetaItemsPerGroup = 2 * kMetaTiles + 3 * kMetaBlocks;
constexpr uint32_t kNumAcContexts = 15 * (37 + 458);
constexpr uint32_t kSetContexts = 64;
constexpr uint32_t kAnsLogTabSize = 12;
constexpr uint32_t kAnsSignature = 0x13;
constexpr uint32_t kAnsInfoStride = 256;
constexpr uint32_t kAnsReverseStride = 1u << kAnsLogTabSize;
constexpr uint32_t kBlobStrideWords = 8192;
constexpr uint32_t kHybridSplitExponent = 4, kHybridMsbInToken = 2, kHybridLsbInToken = 0;
constexpr uint32_t kHybridSplitToken = 1u << kHybridSplitExponent;
constexpr uint32_t kNumStrategies = 27;
const uint32_t kCoveredX[27] = {1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32};
const uint32_t kCoveredY[27] = {1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16};

uint32_t packSigned(int32_t v) { return (static_cast<uint32_t>(v) << 1) ^ static_cast<uint32_t>(v >> 31); }

/// hybrid_uint.slangh hybridEncode (configuration 4, 2, 0).
void hybridEncode(uint32_t value, uint32_t& token, uint32_t& nbits, uint32_t& bits) {
    if (value < kHybridSplitToken) {
        token = value;
        nbits = 0;
        bits = 0;
        return;
    }
    const uint32_t n = 31 - static_cast<uint32_t>(std::countl_zero(value));
    const uint32_t m = value - (1u << n);
    token = kHybridSplitToken + ((n - kHybridSplitExponent) << (kHybridMsbInToken + kHybridLsbInToken)) + ((m >> (n - kHybridMsbInToken)) << kHybridLsbInToken) +
            (m & ((1u << kHybridLsbInToken) - 1));
    nbits = n - kHybridMsbInToken - kHybridLsbInToken;
    bits = (value >> kHybridLsbInToken) & ((1u << nbits) - 1);
}

/// The tree walk over the properties the DC trees read: 0 channel, 2 y, 6 N, 7 W, 15 WP.
uint32_t treeLeaf(std::span<const uint32_t> tree, uint32_t c, uint32_t y, int32_t top, int32_t left, int32_t wpProp, uint32_t& predictor, int32_t& offset) {
    uint32_t node = 0;
    for (;;) {
        const int32_t prop = static_cast<int32_t>(tree[node * kModTreeWords + 0]);
        if (prop < 0) break;
        int32_t v = 0;
        if (prop == 0) v = static_cast<int32_t>(c);
        else if (prop == 2) v = static_cast<int32_t>(y);
        else if (prop == 6) v = top;
        else if (prop == 7) v = left;
        else if (prop == 15) v = wpProp;
        node = v > static_cast<int32_t>(tree[node * kModTreeWords + 1]) ? tree[node * kModTreeWords + 2] : tree[node * kModTreeWords + 3];
    }
    predictor = tree[node * kModTreeWords + 5];
    offset = static_cast<int32_t>(tree[node * kModTreeWords + 6]);
    return tree[node * kModTreeWords + 4];
}

/// One DC group's three channels through the weighted predictor (dc_tokens.slang dcTokenize).
void dcTokenizeGroup(const FrameGeom& geom, std::span<const int32_t> dcQuant, std::span<const uint32_t> dcTree, const Rect& rect, uint32_t* dcTokens) {
    static const uint32_t physical[3] = {1, 0, 2};
    for (uint32_t ci = 0; ci < 3; ++ci) {
        const uint32_t c = physical[ci];
        const uint32_t plane = c * geom.nblocks;
        const uint32_t hs = geom.hshift[c], vs = geom.vshift[c];
        const uint32_t w = rect.w >> hs, h = rect.h >> vs;
        const uint32_t strideX = 1u << hs, strideY = geom.blocksX << vs;
        codestream::wp::State wp;
        wp.init(codestream::wp::Header{}, w);
        const int32_t* q = dcQuant.data() + plane;
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                const uint32_t block = (rect.y + (y << vs)) * geom.blocksX + rect.x + (x << hs);
                const int32_t v = q[block];
                const int32_t left = x > 0 ? q[block - strideX] : (y > 0 ? q[block - strideY] : 0);
                const int32_t top = y > 0 ? q[block - strideY] : left;
                const int32_t topleft = (x > 0 && y > 0) ? q[block - strideY - strideX] : left;
                const int32_t topright = (x + 1 < w && y > 0) ? q[block - strideY + strideX] : top;
                const int32_t toptop = y > 1 ? q[block - 2 * strideY] : top;
                int32_t wpProp = 0;
                const int64_t pred = wp.predict(x, y, top, left, topright, topleft, toptop, wpProp);
                uint32_t predictor = 0;
                int32_t offset = 0;
                const uint32_t ctx = treeLeaf(dcTree, ci, y, top, left, wpProp, predictor, offset);
                int64_t guess = offset;
                if (predictor == kPredWeighted) guess += pred;
                else if (predictor == kPredGradient) guess += codestream::wp::clampedGradient(left, top, topleft);
                else if (predictor == kPredLeft) guess += left;
                const uint32_t value = packSigned(static_cast<int32_t>(static_cast<int64_t>(v) - guess));
                dcTokens[plane + block] = (ctx << 18) | value;
                wp.updateErrors(v, x, y);
            }
        }
    }
}

// --- ANS chains -----------------------------------------------------------------------------

/// LSB-first bit sink over 32-bit words (what the kernels' emitAtomic produces).
struct BitSink {
    std::vector<uint32_t> words;
    uint64_t              bits = 0;

    void reserveBits(uint64_t n) { words.reserve(static_cast<size_t>((bits + n + 31) / 32) + 1); }
    void write(uint32_t n, uint32_t v) {
        if (n == 0) return;
        if (n < 32) v &= (1u << n) - 1;
        const uint64_t w = bits >> 5;
        const uint32_t s = static_cast<uint32_t>(bits & 31);
        while (words.size() <= w + 1) words.push_back(0);
        words[w] |= v << s;
        if (s + n > 32) words[w + 1] |= v >> (32 - s);
        bits += n;
    }
    void alignByte() { bits = (bits + 7) & ~uint64_t{7}; }
    /// Appends `n` bits of another sink (its words start at bit 0).
    void append(const BitSink& other) {
        uint64_t left = other.bits;
        const uint32_t* src = other.words.data();
        while (left > 0) {
            const uint32_t n = static_cast<uint32_t>(std::min<uint64_t>(32, left));
            write(n, *src++);
            left -= n;
        }
    }
    void appendBits(const uint32_t* src, uint32_t n) {
        for (uint32_t i = 0; i < n; i += 32) write(std::min(32u, n - i), src[i / 32]);
    }
};

/// One ANS stream: tokens fed backwards (the last one first, as the kernels do),
/// then written forwards: the final state, then per token [renormalisation][raw bits].
struct Chain {
    const SectionsInput*  in;
    uint32_t              state = kAnsSignature << 16;
    std::vector<uint32_t> rec;   // per token, pushed in reverse stream order: ansN | nbits << 8, ansBits, raw
    uint64_t              bits = 32;

    explicit Chain(const SectionsInput& input) : in(&input) {}

    void put(uint32_t ctxBase, uint32_t regionBase, uint32_t ctx, uint32_t value) {
        uint32_t tok, nbits, raw;
        hybridEncode(value, tok, nbits, raw);
        putSymbol(regionBase + in->ctxMap[ctxBase + ctx], tok, nbits, raw);
    }

    /// ANSCoder::PutSymbol of `tok` in code region `cluster`, with the token's raw bits.
    void putSymbol(uint32_t cluster, uint32_t tok, uint32_t nbits, uint32_t raw) {
        const uint32_t e = in->ansInfo[cluster * kAnsInfoStride + tok];
        const uint32_t freq = e & 0xFFFFu;
        const uint32_t start = e >> 16;
        uint32_t ansN = 0, ansBits = 0;
        if ((state >> (32 - kAnsLogTabSize)) >= freq) {
            ansBits = state & 0xFFFFu;
            ansN = 16;
            state >>= 16;
        }
        state = ((state / freq) << kAnsLogTabSize) + in->ansReverse[cluster * kAnsReverseStride + start + state % freq];
        rec.push_back(ansN | (nbits << 8));
        rec.push_back(ansBits);
        rec.push_back(raw);
        bits += nbits + ansN;
    }

    void writeTo(BitSink& sink) const {
        sink.write(32, state);
        for (size_t i = rec.size(); i >= 3; i -= 3) {
            const uint32_t h = rec[i - 3];
            sink.write(h & 0xFF, rec[i - 2]);
            sink.write(h >> 8, rec[i - 1]);
        }
    }
};

uint32_t physicalChannel(uint32_t ci) { return ci == 0 ? 1 : (ci == 1 ? 0 : 2); }

/// ac_sections.slang acSections for one (pass, group): the AC stream alone (no alpha group).
BitSink acSection(const SectionsInput& in, uint32_t pass, uint32_t g, uint32_t& acBits) {
    const FrameGeom& geom = in.geom;
    const Rect& rect = geom.acRects[g];
    const uint32_t pb = pass * 3 * geom.nblocks;
    const uint32_t ctxBase = pass * kNumAcContexts;
    const uint32_t regionBase = pass * 32;
    Chain aw(in);
    for (int32_t by = static_cast<int32_t>(rect.h) - 1; by >= 0; --by) {
        for (int32_t bx = static_cast<int32_t>(rect.w) - 1; bx >= 0; --bx) {
            const uint32_t block = (rect.y + static_cast<uint32_t>(by)) * geom.blocksX + rect.x + static_cast<uint32_t>(bx);
            if ((in.acs[block] & 1) == 0) continue;
            const uint32_t strategy = in.acs[block] >> 1;
            const uint32_t cx = kCoveredX[strategy], cy = kCoveredY[strategy];
            const uint32_t covered = cx * cy;
            const uint32_t fbx = block % geom.blocksX, fby = block / geom.blocksX;
            for (int32_t ci = 2; ci >= 0; --ci) {
                const uint32_t c = physicalChannel(static_cast<uint32_t>(ci));
                const uint32_t hs = geom.hshift[c], vs = geom.vshift[c];
                if (((fbx >> hs) << hs) != fbx || ((fby >> vs) << vs) != fby) continue;
                const uint32_t n = in.numTokens[pb + c * geom.nblocks + block];
                for (int32_t k = static_cast<int32_t>(covered + n) - 1; k >= static_cast<int32_t>(covered); --k) {
                    const uint32_t m = static_cast<uint32_t>(k);
                    const uint32_t q = m >> 6;
                    const uint32_t b = (fby + q / cx) * geom.blocksX + fbx + q % cx;
                    const uint32_t t = in.acTokens[uint64_t{pb} * 64 + (uint64_t{c} * geom.nblocks + b) * 64 + (m & 63)];
                    aw.put(ctxBase, regionBase, t >> 18, t & 0x3FFFFu);
                }
                aw.put(ctxBase, regionBase, in.nzCtx[pb + c * geom.nblocks + block], in.nonZeros[pb + c * geom.nblocks + block]);
            }
        }
    }
    acBits = static_cast<uint32_t>(aw.bits);
    BitSink sink;
    sink.reserveBits(aw.bits);
    aw.writeTo(sink);
    return sink;
}

/// dc_tokens.slang dcGroupSections for one group: the section's four parts.
struct DcSection {
    BitSink  bits;
    uint32_t partStart[4] = {};   // relative to the section: pre, DC stream, mid, meta stream
};

DcSection dcGroupSection(const SectionsInput& in, uint32_t g) {
    const FrameGeom& geom = in.geom;
    const Rect& rect = geom.dcRects[g];
    const uint32_t numAcSets = geom.numPasses;
    const auto setCtxBase = [&](uint32_t set) { return numAcSets * kNumAcContexts + (set - numAcSets) * kSetContexts; };
    const uint32_t dcSet = numAcSets + 2 * g, metaSet = numAcSets + 1 + 2 * g;
    DcSection out;
    const uint32_t preRegion = 2 + 2 * g, midRegion = 3 + 2 * g;
    out.partStart[0] = 0;
    out.bits.appendBits(in.blobs.data() + uint64_t{preRegion} * kBlobStrideWords, in.blobBits[preRegion]);
    out.partStart[1] = static_cast<uint32_t>(out.bits.bits);
    if (!geom.useDcFrame) {
        Chain aw(in);
        static const uint32_t physical[3] = {1, 0, 2};
        for (int32_t ci = 2; ci >= 0; --ci) {
            const uint32_t c = physical[ci];
            const uint32_t plane = c * geom.nblocks;
            const uint32_t hs = geom.hshift[c], vs = geom.vshift[c];
            const uint32_t w = rect.w >> hs, h = rect.h >> vs;
            for (int32_t y = static_cast<int32_t>(h) - 1; y >= 0; --y) {
                for (int32_t x = static_cast<int32_t>(w) - 1; x >= 0; --x) {
                    const uint32_t t = in.dcTokens[plane + (rect.y + (static_cast<uint32_t>(y) << vs)) * geom.blocksX + rect.x + (static_cast<uint32_t>(x) << hs)];
                    aw.put(setCtxBase(dcSet), dcSet * 32, t >> 18, t & 0x3FFFFu);
                }
            }
        }
        aw.writeTo(out.bits);
    }
    out.partStart[2] = static_cast<uint32_t>(out.bits.bits);
    out.bits.appendBits(in.blobs.data() + uint64_t{midRegion} * kBlobStrideWords, in.blobBits[midRegion]);
    out.partStart[3] = static_cast<uint32_t>(out.bits.bits);
    Chain mw(in);
    const uint32_t* meta = in.metaTokens.data() + uint64_t{g} * kMetaItemsPerGroup;
    for (int32_t l = static_cast<int32_t>(kMetaItemsPerGroup) - 1; l >= 0; --l) {
        const uint32_t t = meta[l];
        if (t == kNoToken) continue;
        mw.put(setCtxBase(metaSet), metaSet * 32, t >> 18, t & 0x3FFFFu);
    }
    mw.writeTo(out.bits);
    return out;
}

}   // namespace

void dcTokenize(const FrameGeom& geom, std::span<const int32_t> dcQuant, std::span<const uint32_t> dcTree, std::vector<uint32_t>& dcTokens) {
    dcTokens.assign(size_t{3} * geom.nblocks, kNoToken);
    std::vector<std::future<void>> tasks;
    for (uint32_t g = 1; g < geom.numDcGroups; ++g) {
        tasks.push_back(std::async(std::launch::async, [&, g] { dcTokenizeGroup(geom, dcQuant, dcTree, geom.dcRects[g], dcTokens.data()); }));
    }
    if (geom.numDcGroups > 0) dcTokenizeGroup(geom, dcQuant, dcTree, geom.dcRects[0], dcTokens.data());
    for (auto& t : tasks) t.get();
}

Result<SectionsOutput> emitSections(const SectionsInput& in) {
    const FrameGeom& geom = in.geom;
    for (uint32_t block = 0; block < geom.nblocks; ++block) {
        if ((in.acs[block] & 1) != 0 && (in.acs[block] >> 1) >= kNumStrategies) return Error(ErrorCode::InternalError, "invalid strategy in the acs map");
    }
    const uint32_t numAc = geom.numGroups * geom.numPasses;
    // Every chain in parallel.
    std::vector<std::future<DcSection>> dcTasks;
    for (uint32_t g = 0; g < geom.numDcGroups; ++g) dcTasks.push_back(std::async(std::launch::async, [&, g] { return dcGroupSection(in, g); }));
    std::vector<uint32_t> acBits(numAc, 0);
    std::vector<std::future<BitSink>> acTasks;
    for (uint32_t s = 0; s < numAc; ++s) {
        acTasks.push_back(std::async(std::launch::async, [&, s] { return acSection(in, s / geom.numGroups, s % geom.numGroups, acBits[s]); }));
    }
    // Codestream order, each section byte-aligned (or packed into one).
    SectionsOutput out;
    out.acOnlyBits.assign(numAc, 0);
    out.dcPartStart.assign(size_t{4} * geom.numDcGroups, 0);
    BitSink body;
    const auto startSection = [&] {
        if (!geom.packed) body.alignByte();
        return body.bits;
    };
    const auto endSection = [&](uint64_t start) { out.sectionBytes.push_back(static_cast<uint32_t>((body.bits - start + 7) / 8)); };
    {
        const uint64_t start = startSection();
        body.appendBits(in.blobs.data(), in.blobBits[0]);
        endSection(start);
    }
    for (uint32_t g = 0; g < geom.numDcGroups; ++g) {
        const DcSection s = dcTasks[g].get();
        const uint64_t start = startSection();
        for (uint32_t k = 0; k < 4; ++k) out.dcPartStart[g * 4 + k] = static_cast<uint32_t>(start) + s.partStart[k];
        body.append(s.bits);
        endSection(start);
    }
    {
        const uint64_t start = startSection();
        body.appendBits(in.blobs.data() + uint64_t{1} * kBlobStrideWords, in.blobBits[1]);
        body.appendBits(in.blobs.data() + uint64_t{in.qtBlobRegion} * kBlobStrideWords, in.blobBits[in.qtBlobRegion]);
        body.appendBits(in.blobs.data() + uint64_t{in.acPostRegion} * kBlobStrideWords, in.blobBits[in.acPostRegion]);
        endSection(start);
    }
    for (uint32_t s = 0; s < numAc; ++s) {
        const BitSink sec = acTasks[s].get();
        const uint64_t start = startSection();
        body.append(sec);
        endSection(start);
        out.acOnlyBits[s] = acBits[s];
    }
    const uint64_t bodyBytes = (body.bits + 7) / 8;
    out.body.resize(bodyBytes);
    body.words.resize((bodyBytes + 3) / 4 + 1, 0);
    std::memcpy(out.body.data(), body.words.data(), bodyBytes);
    return out;
}

// --- the lossless encoder's chains (enc/modular.slang) -------------------------------------

namespace {

constexpr uint32_t kNumProperties = 16;
constexpr uint32_t kThreshStride = 64;

void hybridEncodeCfg(uint32_t value, uint32_t splitExponent, uint32_t msb, uint32_t lsb, uint32_t& token, uint32_t& nbits, uint32_t& bits) {
    const uint32_t split = 1u << splitExponent;
    if (value < split) {
        token = value;
        nbits = 0;
        bits = 0;
        return;
    }
    const uint32_t n = 31 - static_cast<uint32_t>(std::countl_zero(value));
    const uint32_t m = value - (1u << n);
    token = split + ((n - splitExponent) << (msb + lsb)) + ((m >> (n - msb)) << lsb) + (m & ((1u << lsb) - 1));
    nbits = n - msb - lsb;
    bits = (value >> lsb) & ((1u << nbits) - 1);
}

int32_t modClampedGradient(int32_t n, int32_t w, int32_t l) {
    const int32_t m = std::min(n, w);
    const int32_t M = std::max(n, w);
    const int32_t grad = static_cast<int32_t>(static_cast<uint32_t>(n) + static_cast<uint32_t>(w) - static_cast<uint32_t>(l));
    const int32_t gradClampM = (l < m) ? M : grad;
    return (l > M) ? m : gradClampM;
}

uint32_t hashSample(uint32_t i) {
    uint32_t x = i * 2654435761u + 0x9E3779B9u;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

struct Neighbours {
    int32_t left, top, topleft, topright, toptop, leftleft;
};

Neighbours neighboursAt(const int32_t* planes, uint32_t at, uint32_t stride, uint32_t lx, uint32_t ly, uint32_t w) {
    Neighbours n;
    n.left = lx > 0 ? planes[at - 1] : (ly > 0 ? planes[at - stride] : 0);
    n.top = ly > 0 ? planes[at - stride] : n.left;
    n.topleft = (lx > 0 && ly > 0) ? planes[at - stride - 1] : n.left;
    n.topright = (lx + 1 < w && ly > 0) ? planes[at - stride + 1] : n.top;
    n.toptop = ly > 1 ? planes[at - 2 * stride] : n.top;
    n.leftleft = lx > 1 ? planes[at - 2] : n.left;
    return n;
}

/// The rectangle of channel `ch` that group g codes (modular.slang groupRect).
bool groupRect(const ModGeom& geom, const ModChannel& ch, uint32_t g, uint32_t& x0, uint32_t& y0, uint32_t& w, uint32_t& h) {
    if (!ch.grouped) {
        x0 = 0; y0 = 0; w = ch.w; h = ch.h;
        return g == 0 && w > 0 && h > 0;
    }
    x0 = (g % geom.groupsX) * geom.groupDim;
    y0 = (g / geom.groupsX) * geom.groupDim;
    if (x0 >= ch.w || y0 >= ch.h) { w = 0; h = 0; return false; }
    w = std::min(geom.groupDim, ch.w - x0);
    h = std::min(geom.groupDim, ch.h - y0);
    return true;
}

void modProperties(uint32_t c, uint32_t stream, uint32_t lx, uint32_t ly, const Neighbours& n, int32_t prevGradient, int32_t wpProp, int32_t props[kNumProperties]) {
    props[0] = static_cast<int32_t>(c);
    props[1] = static_cast<int32_t>(stream);
    props[2] = static_cast<int32_t>(ly);
    props[3] = static_cast<int32_t>(lx);
    props[4] = std::abs(n.top);
    props[5] = std::abs(n.left);
    props[6] = n.top;
    props[7] = n.left;
    props[8] = n.left - prevGradient;
    props[9] = static_cast<int32_t>(static_cast<uint32_t>(n.left) + static_cast<uint32_t>(n.top) - static_cast<uint32_t>(n.topleft));
    props[10] = n.left - n.topleft;
    props[11] = n.topleft - n.top;
    props[12] = n.top - n.topright;
    props[13] = n.top - n.toptop;
    props[14] = n.left - n.leftleft;
    props[15] = wpProp;
}

uint32_t quantizeProperty(std::span<const int32_t> thresholds, uint32_t propId, int32_t v) {
    const uint32_t count = static_cast<uint32_t>(thresholds[kNumProperties * kThreshStride + propId]);
    uint32_t q = 0;
    for (uint32_t i = 0; i < count; ++i) if (thresholds[propId * kThreshStride + i] < v) ++q;
    return q;
}

/// Runs `body(c, ch, x0, y0, w, h, stream, chanProp)` for every channel rectangle group g codes.
template <typename F>
void forGroupChannels(const ModGeom& geom, uint32_t g, F&& body) {
    for (uint32_t c = 0; c < geom.channels.size(); ++c) {
        const ModChannel& ch = geom.channels[c];
        uint32_t x0, y0, w, h;
        if (!groupRect(geom, ch, g, x0, y0, w, h)) continue;
        const uint32_t stream = ch.grouped ? geom.streamBase + g : 0;
        const uint32_t chanProp = ch.grouped ? c - geom.nbMeta : c;
        body(ch, x0, y0, w, h, stream, chanProp);
    }
}

void modSamplesGroup(const ModGeom& geom, const int32_t* planes, const ModSamplesParams& p, uint32_t g, std::vector<uint64_t>& out) {
    forGroupChannels(geom, g, [&](const ModChannel& ch, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint32_t stream, uint32_t chanProp) {
        const float fraction = std::max(p.fraction, std::min(1.0f, 1024.0f / static_cast<float>(w * h)));
        codestream::wp::State wp;
        wp.init(codestream::wp::Header{}, w);
        for (uint32_t ly = 0; ly < h; ++ly) {
            int32_t prevGradient = 0;
            for (uint32_t lx = 0; lx < w; ++lx) {
                const uint32_t at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int32_t v = planes[at];
                const Neighbours n = neighboursAt(planes, at, ch.w, lx, ly, w);
                int32_t wpProp = 0;
                const int64_t wpPred = wp.predict(lx, ly, n.top, n.left, n.topright, n.topleft, n.toptop, wpProp);
                if (static_cast<float>(hashSample(at) & 0xFFFFFFu) < fraction * 16777216.0f) {
                    int32_t props[kNumProperties];
                    modProperties(chanProp, stream, lx, ly, n, prevGradient, wpProp, props);
                    uint32_t tokW, nbitsW, raw, tokG, nbitsG;
                    hybridEncodeCfg(packSigned(static_cast<int32_t>(static_cast<int64_t>(v) - wpPred)), 4, 1, 2, tokW, nbitsW, raw);
                    hybridEncodeCfg(packSigned(v - modClampedGradient(n.left, n.top, n.topleft)), 4, 1, 2, tokG, nbitsG, raw);
                    const uint32_t sx = std::min(tokW, 255u) | (std::min(nbitsW, 255u) << 8) | (std::min(tokG, 255u) << 16) | (std::min(nbitsG, 255u) << 24);
                    uint32_t sy = 0;
                    for (uint32_t k = 0; k < 4; ++k) sy |= std::min(quantizeProperty(p.thresholds, p.propIds[k], props[p.propIds[k]]), 255u) << (8 * k);
                    out.push_back(static_cast<uint64_t>(sx) | (static_cast<uint64_t>(sy) << 32));
                }
                prevGradient = static_cast<int32_t>(static_cast<uint32_t>(n.left) + static_cast<uint32_t>(n.top) - static_cast<uint32_t>(n.topleft));
                wp.updateErrors(v, lx, ly);
            }
        }
    });
}

void modTokenizeTreeGroup(const ModGeom& geom, const int32_t* planes, std::span<const uint32_t> tree, uint32_t g, uint32_t* tokens) {
    forGroupChannels(geom, g, [&](const ModChannel& ch, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint32_t stream, uint32_t chanProp) {
        codestream::wp::State wp;
        wp.init(codestream::wp::Header{}, w);
        for (uint32_t ly = 0; ly < h; ++ly) {
            int32_t prevGradient = 0;
            for (uint32_t lx = 0; lx < w; ++lx) {
                const uint32_t at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int32_t v = planes[at];
                const Neighbours n = neighboursAt(planes, at, ch.w, lx, ly, w);
                int32_t wpProp = 0;
                const int64_t wpPred = wp.predict(lx, ly, n.top, n.left, n.topright, n.topleft, n.toptop, wpProp);
                int32_t props[kNumProperties];
                modProperties(chanProp, stream, lx, ly, n, prevGradient, wpProp, props);
                uint32_t node = 0;
                for (;;) {
                    const int32_t prop = static_cast<int32_t>(tree[node * kModTreeWords + 0]);
                    if (prop < 0) break;
                    const int32_t pv = static_cast<uint32_t>(prop) < kNumProperties ? props[prop] : 0;
                    node = pv > static_cast<int32_t>(tree[node * kModTreeWords + 1]) ? tree[node * kModTreeWords + 2] : tree[node * kModTreeWords + 3];
                }
                const uint32_t ctx = tree[node * kModTreeWords + 4];
                const uint32_t predictor = tree[node * kModTreeWords + 5];
                int64_t guess = static_cast<int32_t>(tree[node * kModTreeWords + 6]);
                if (predictor == kPredWeighted) guess += wpPred;
                else if (predictor == kPredGradient) guess += modClampedGradient(n.left, n.top, n.topleft);
                else if (predictor == kPredLeft) guess += n.left;
                else if (predictor == 2) guess += n.top;
                tokens[at] = (ctx << 18) | packSigned(static_cast<int32_t>(static_cast<int64_t>(v) - guess));
                prevGradient = props[9];
                wp.updateErrors(v, lx, ly);
            }
        }
    });
}

/// modular.slangh modTreeLeaf: the fixed trees' properties 0, 2, 6, 7, 9, 15.
uint32_t modTreeLeaf(std::span<const uint32_t> tree, uint32_t c, uint32_t y, int32_t top, int32_t left, int32_t topleft, int32_t wpProp, uint32_t& predictor, int32_t& offset) {
    uint32_t node = 0;
    for (;;) {
        const int32_t prop = static_cast<int32_t>(tree[node * kModTreeWords + 0]);
        if (prop < 0) break;
        int32_t v = 0;
        if (prop == 0) v = static_cast<int32_t>(c);
        else if (prop == 2) v = static_cast<int32_t>(y);
        else if (prop == 6) v = top;
        else if (prop == 7) v = left;
        else if (prop == 9) v = static_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(top) - static_cast<uint32_t>(topleft));
        else if (prop == 15) v = wpProp;
        node = v > static_cast<int32_t>(tree[node * kModTreeWords + 1]) ? tree[node * kModTreeWords + 2] : tree[node * kModTreeWords + 3];
    }
    predictor = tree[node * kModTreeWords + 5];
    offset = static_cast<int32_t>(tree[node * kModTreeWords + 6]);
    return tree[node * kModTreeWords + 4];
}

void modTokenizeFixedGroup(const ModGeom& geom, const int32_t* planes, std::span<const uint32_t> tree, uint32_t g, uint32_t* tokens) {
    forGroupChannels(geom, g, [&](const ModChannel& ch, uint32_t x0, uint32_t y0, uint32_t w, uint32_t h, uint32_t, uint32_t chanProp) {
        codestream::wp::State wp;
        wp.init(codestream::wp::Header{}, w);
        for (uint32_t ly = 0; ly < h; ++ly) {
            for (uint32_t lx = 0; lx < w; ++lx) {
                const uint32_t at = ch.offset + (y0 + ly) * ch.w + x0 + lx;
                const int32_t v = planes[at];
                const Neighbours n = neighboursAt(planes, at, ch.w, lx, ly, w);
                int32_t wpProp = 0;
                const int64_t pred = wp.predict(lx, ly, n.top, n.left, n.topright, n.topleft, n.toptop, wpProp);
                uint32_t predictor = 0;
                int32_t offset = 0;
                const uint32_t ctx = modTreeLeaf(tree, chanProp, ly, n.top, n.left, n.topleft, wpProp, predictor, offset);
                int64_t guess = offset;
                if (predictor == kPredWeighted) guess += pred;
                else if (predictor == kPredGradient) guess += modClampedGradient(n.left, n.top, n.topleft);
                else if (predictor == kPredLeft) guess += n.left;
                tokens[at] = (ctx << 18) | packSigned(static_cast<int32_t>(static_cast<int64_t>(v) - guess));
                wp.updateErrors(v, lx, ly);
            }
        }
    });
}

template <typename F>
void forGroupsParallel(uint32_t numGroups, F&& body) {
    std::vector<std::future<void>> tasks;
    for (uint32_t g = 1; g < numGroups; ++g) tasks.push_back(std::async(std::launch::async, [&, g] { body(g); }));
    if (numGroups > 0) body(0);
    for (auto& t : tasks) t.get();
}

}   // namespace

void modSamples(const ModGeom& geom, std::span<const int32_t> planes, const ModSamplesParams& params, std::vector<uint64_t>& samples) {
    std::vector<std::vector<uint64_t>> perGroup(geom.numGroups);
    forGroupsParallel(geom.numGroups, [&](uint32_t g) { modSamplesGroup(geom, planes.data(), params, g, perGroup[g]); });
    samples.clear();
    for (const auto& v : perGroup) samples.insert(samples.end(), v.begin(), v.end());
    if (samples.size() > params.capacity) samples.resize(params.capacity);
}

void modTokenizeTree(const ModGeom& geom, std::span<const int32_t> planes, std::span<const uint32_t> treeNodes, std::vector<uint32_t>& tokens) {
    tokens.assign(std::max<size_t>(geom.totalSamples, 1), 0);
    forGroupsParallel(geom.numGroups, [&](uint32_t g) { modTokenizeTreeGroup(geom, planes.data(), treeNodes, g, tokens.data()); });
}

void modTokenizeFixed(const ModGeom& geom, std::span<const int32_t> planes, std::span<const uint32_t> tree, uint32_t pass, std::vector<uint32_t>& tokens) {
    if (tokens.size() < size_t{2} * geom.totalSamples) tokens.resize(std::max<size_t>(size_t{2} * geom.totalSamples, 1), 0);
    forGroupsParallel(geom.numGroups, [&](uint32_t g) { modTokenizeFixedGroup(geom, planes.data(), tree, g, tokens.data() + size_t{pass} * geom.totalSamples); });
}

// --- run-length LZ77 and the group sections (enc/lz77.slang, enc/modular.slang) -------------

namespace {

constexpr uint32_t kLengthFlag = 0x80000000u;
constexpr uint32_t kMinSymbol = 224;
constexpr uint32_t kMinLength = 3;

/// lz77.slang streamRect: the rectangle channel c contributes to `stream`.
bool streamRect(const ModGeom& geom, uint32_t stream, uint32_t c, uint32_t& x0, uint32_t& y0, uint32_t& w, uint32_t& h) {
    const ModChannel& ch = geom.channels[c];
    if (stream == geom.numGroups) {
        x0 = 0; y0 = 0; w = ch.w; h = ch.h;
        return !ch.grouped && w > 0 && h > 0;
    }
    if (!ch.grouped) { x0 = 0; y0 = 0; w = 0; h = 0; return false; }
    x0 = (stream % geom.groupsX) * geom.groupDim;
    y0 = (stream / geom.groupsX) * geom.groupDim;
    if (x0 >= ch.w || y0 >= ch.h) { w = 0; h = 0; return false; }
    w = std::min(geom.groupDim, ch.w - x0);
    h = std::min(geom.groupDim, ch.h - y0);
    return true;
}

struct StreamScan {
    uint32_t len = 0, decrease16 = 0, total = 0;
};

StreamScan rleScanStream(const ModGeom& geom, std::span<const uint32_t> tokens, std::span<const uint32_t> ctxHist, const std::vector<uint32_t>& ctxTotal,
                         uint32_t numContexts, uint32_t mode, uint32_t stream, uint32_t* outTokens) {
    const auto symbolBits = [&](uint32_t ctx, uint32_t tok) -> float {
        const uint32_t total = ctxTotal[ctx];
        const uint32_t c = ctxHist[ctx * 128 + std::min(tok, 127u)];
        if (c == 0) return static_cast<float>(kAnsLogTabSize);
        if (c == total) return 0.0f;
        return -std::log2(static_cast<float>(c) / static_cast<float>(total));
    };
    StreamScan r;
    uint32_t out = 0;
    float decrease = 0.0f;
    uint32_t prevValue = 0;
    bool havePrev = false;
    uint32_t runLen = 0, runCtx = 0;
    float runCost = 0.0f;
    const uint32_t numChannels = static_cast<uint32_t>(geom.channels.size());
    for (uint32_t c = 0; c <= numChannels; ++c) {
        uint32_t x0 = 0, y0 = 0, w = 0, h = 0;
        const bool last = c == numChannels;
        if (!last && !streamRect(geom, stream, c, x0, y0, w, h)) continue;
        const uint32_t stride = last ? 1 : geom.channels[c].w;
        const uint32_t offset = last ? 0 : geom.channels[c].offset;
        const uint32_t count = last ? 1 : w * h;
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t t = 0;
            if (!last) {
                const uint32_t lx = i % w, ly = i / w;
                t = tokens[offset + (y0 + ly) * stride + x0 + lx];
                ++r.total;
            }
            const uint32_t value = t & 0x3FFFFu, ctx = t >> 18;
            if (mode == 1 && !last && havePrev && value == prevValue) {
                if (runLen == 0) runCtx = ctx;
                uint32_t tok, nbits, raw;
                hybridEncode(value, tok, nbits, raw);
                runCost += symbolBits(ctx, tok) + static_cast<float>(nbits);
                ++runLen;
                outTokens[out++] = t;
                continue;
            }
            if (runLen != 0) {
                const uint32_t lz77Len = runLen >= kMinLength ? runLen - kMinLength : 0;
                const float lz77Cost = runLen >= kMinLength ? static_cast<float>((31 - std::countl_zero(lz77Len + 1)) + 1 + 1) : 0.0f;
                if (runLen >= kMinLength && runCost > lz77Cost) {
                    out -= runLen;
                    outTokens[out++] = kLengthFlag | (runCtx << 18) | lz77Len;
                    outTokens[out++] = (numContexts << 18) | 1u;
                    decrease += runCost - lz77Cost;
                }
                runLen = 0;
                runCost = 0.0f;
            }
            if (!last) {
                outTokens[out++] = t;
                prevValue = value;
                havePrev = true;
            }
        }
    }
    r.len = out;
    r.decrease16 = static_cast<uint32_t>(decrease * 16.0f);
    return r;
}

/// modular.slang groupHeaderBits / emitGroupHeader.
void writeGroupHeader(BitSink& sink, uint32_t rctType, uint32_t beginC) {
    sink.write(2, 0x3u);   // use_global_tree, wp all_default
    if (rctType == 0) {
        sink.write(2, 0);   // num_transforms = 0
        return;
    }
    sink.write(2, 1);   // num_transforms = 1
    sink.write(2, 0);   // Transform id: RCT
    sink.write(2, 0);   // begin_c: selector 0 (Bits(3))
    sink.write(3, beginC);
    if (rctType == 6) {
        sink.write(2, 0);
    } else if (rctType < 4) {
        sink.write(2, 1);
        sink.write(2, rctType);
    } else if (rctType < 18) {
        sink.write(2, 2);
        sink.write(4, rctType - 2);
    } else {
        sink.write(2, 3);
        sink.write(6, rctType - 10);
    }
}

/// One group's section: header, then its compact list as one ANS stream.
BitSink modGroupSection(const ModSectionsInput& in, const SectionsInput& tables, uint32_t g, uint32_t& bits) {
    const uint32_t rctType = in.groupRct.empty() ? 0 : in.groupRct[g];
    BitSink sink;
    writeGroupHeader(sink, rctType, in.rctBeginC);
    Chain aw(tables);
    aw.rec.reserve(size_t{3} * in.streamLen[g]);
    const uint32_t base = in.streamStart[g];
    for (int64_t i = static_cast<int64_t>(in.streamLen[g]) - 1; i >= 0; --i) {
        const uint32_t t = in.streamTokens[base + static_cast<uint32_t>(i)];
        const uint32_t ctx = (t & ~kLengthFlag) >> 18;
        const uint32_t value = t & 0x3FFFFu;
        const uint32_t cluster = in.ctxMap[ctx];
        uint32_t tok, nbits, raw;
        if ((t & kLengthFlag) != 0) {
            hybridEncode(value, tok, nbits, raw);
            tok += kMinSymbol;
        } else {
            const uint32_t cfg = in.clusterCfg[cluster];
            hybridEncodeCfg(value, cfg & 0xFFu, (cfg >> 8) & 0xFFu, (cfg >> 16) & 0xFFu, tok, nbits, raw);
        }
        aw.putSymbol(cluster, tok, nbits, raw);
    }
    sink.reserveBits(sink.bits + aw.bits);
    aw.writeTo(sink);
    bits = static_cast<uint32_t>(sink.bits);
    return sink;
}

}   // namespace

void rleScan(const ModGeom& geom, std::span<const uint32_t> tokens, std::span<const uint32_t> ctxHist, uint32_t numContexts, uint32_t mode,
             std::span<const uint32_t> streamStart, RleResult& out) {
    const uint32_t numStreams = geom.numGroups + 1;
    std::vector<uint32_t> ctxTotal(numContexts + 1, 0);
    for (uint32_t ctx = 0; ctx <= numContexts && (ctx + 1) * 128 <= ctxHist.size(); ++ctx)
        for (uint32_t s = 0; s < 128; ++s) ctxTotal[ctx] += ctxHist[ctx * 128 + s];
    out.streamTokens.assign(std::max<size_t>(geom.totalSamples, 1), 0);
    out.streamLen.assign(numStreams, 0);
    std::vector<StreamScan> scans(numStreams);
    std::vector<std::future<void>> tasks;
    for (uint32_t s = 1; s < numStreams; ++s) {
        tasks.push_back(std::async(std::launch::async, [&, s] { scans[s] = rleScanStream(geom, tokens, ctxHist, ctxTotal, numContexts, mode, s, out.streamTokens.data() + streamStart[s]); }));
    }
    scans[0] = rleScanStream(geom, tokens, ctxHist, ctxTotal, numContexts, mode, 0, out.streamTokens.data() + streamStart[0]);
    for (auto& t : tasks) t.get();
    out.decrease16 = 0;
    out.total = 0;
    for (uint32_t s = 0; s < numStreams; ++s) {
        out.streamLen[s] = scans[s].len;
        out.decrease16 += scans[s].decrease16;
        out.total += scans[s].total;
    }
}

Result<SectionsOutput> emitModularSections(const ModSectionsInput& in) {
    SectionsInput tables;   // the chain's tables: the modular set is region 0.. with ctxMap at 0
    tables.ctxMap = in.ctxMap;
    tables.ansInfo = in.ansInfo;
    tables.ansReverse = in.ansReverse;
    SectionsOutput out;
    BitSink body;
    body.appendBits(in.blob0.data(), in.blob0Bits);
    body.alignByte();
    out.sectionBytes.push_back(static_cast<uint32_t>(body.bits / 8));
    if (!in.fitsGroup) {
        const uint32_t numGroups = in.geom.numGroups;
        std::vector<uint32_t> bits(numGroups, 0);
        std::vector<std::future<BitSink>> tasks;
        for (uint32_t g = 0; g < numGroups; ++g) tasks.push_back(std::async(std::launch::async, [&, g] { return modGroupSection(in, tables, g, bits[g]); }));
        for (uint32_t g = 0; g < in.numDcGroups; ++g) out.sectionBytes.push_back(0);
        out.sectionBytes.push_back(0);   // AcGlobal
        for (uint32_t g = 0; g < numGroups; ++g) {
            const BitSink sec = tasks[g].get();
            const uint64_t start = body.bits;
            body.append(sec);
            body.alignByte();
            out.sectionBytes.push_back(static_cast<uint32_t>((body.bits - start) / 8));
        }
        out.acOnlyBits = bits;
    }
    const uint64_t bodyBytes = body.bits / 8;
    out.body.resize(bodyBytes);
    body.words.resize((bodyBytes + 3) / 4 + 1, 0);
    std::memcpy(out.body.data(), body.words.data(), bodyBytes);
    return out;
}

}   // namespace sjxl::enc::cpu
