// Copyright (c) 2026 jesus luque.
//
// Ports of enc/dc_tokens.slang (dcTokenize, dcGroupSections), enc/ac_sections.slang
// (acSections) and enc/layout.slang; see EntropyCpu.h.

#include "sjxl/encoder/EntropyCpu.h"

#include <algorithm>
#include <bit>
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
        const uint32_t cluster = regionBase + in->ctxMap[ctxBase + ctx];
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

}   // namespace sjxl::enc::cpu
