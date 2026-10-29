// Copyright (c) 2026 jesus luque.
//
// A learned MA tree for a Modular frame, libjxl's ComputeBestTree (enc_ma.cc)
// as level-synchronous kernels. The samples (modular.slang modSamples) carry
// the residual token of every candidate predictor and the quantised value of
// every candidate property; a level at a time, treeHistograms counts each open
// node's tokens per (property, value, predictor), treeSplit (one thread-group
// per node) sweeps the split thresholds of every property with libjxl's cost
// (EstimateBits per side, extra bits, the predictor-change penalty, the
// static/non-static/no-WP categories and fast_decode_multiplier) and splits
// the node when the saving beats the node threshold, and treeAssign moves the
// samples into the children. propStats and propThresholds quantise the
// properties beforehand as TreeSamples::PreQuantizeProperties does.
module enc_tree;

static const uint kProps = 4;         // candidate properties per tree
static const uint kVals = 48;         // max_property_values at libjxl's hare tier
static const uint kPreds = 2;         // Predictor::Best: Weighted, Gradient
static const uint kToks = 64;         // residual tokens counted, (4, 1, 2) split (larger ones clamp to the last bin)
static const uint kNodeWords = 8;     // propIndex (0xFFFFFFFF leaf), splitQ, lchild, rchild, pred, splitValue, depth, pad
static const uint kThreshStride = 64; // thresholds per property in `thresholds`
static const uint kNumProperties = 16;
static const uint kRange = 512;       // QuantizeSamples clamps pixel and difference samples to +-kRange
static const uint kHistBins = 2 * kRange + 1;
static const uint kThreads = 64;
static const float kMinProb = 1.0f / 4096.0f;
static const uint kSlotCounts = kProps * kVals * kPreds * kToks;
static const uint kSlotExtra = kProps * kVals * kPreds;

struct TreeParams {
    uint  numSamples;
    uint  levelStart;      // first node of this level
    uint  maxNodes;        // node records available
    uint  maxSlots;        // nodes a level may hold (counts buffer)
    float threshold;       // splitting_heuristics_node_threshold
    float fastDecodeMul;   // fast_decode_multiplier
    uint  width;           // propStats: the planes
    uint  height;
    uint  numChannels;
    uint  numGroups;
    uint  groupDim;
    uint  groupsX;
    uint  streamBase;      // stream id of group 0 (0 when the image is one global stream)
    float sampleFraction;  // propStats: the pixel and difference sample fraction
    uint4 propIds;         // the candidate properties (libjxl property numbers)
};

StructuredBuffer<int>       planes;        // propStats
RWStructuredBuffer<uint>    stats;         // [0, kHistBins) pixel samples at value + kRange, [kHistBins, 2 kHistBins) differences
RWStructuredBuffer<int>     thresholds;    // [16][kThreshStride] per property; thresholds[16 * kThreshStride + p] = count
StructuredBuffer<uint2>     samples;       // x: tokW | nbitsW << 8 | tokG << 16 | nbitsG << 24; y: quantised property values, 8 bits each
RWStructuredBuffer<uint>    sampleNode;    // per sample: its node
RWStructuredBuffer<uint>    counts;        // per level slot: [prop][val][pred][tok]
RWStructuredBuffer<uint>    extra;         // per level slot: [prop][val][pred] extra bits
RWStructuredBuffer<uint>    nodes;         // kNodeWords per node
RWStructuredBuffer<uint>    counters;      // [0] node count, [1] splits this level
ConstantBuffer<TreeParams>  params;

uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

/// One thread per pixel of every channel: a subsample of pixel values and
/// horizontal differences (CollectPixelSamples), clamped to +-kRange.
[shader("compute")]
[numthreads(64, 1, 1)]
void propStats(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.width * params.height * params.numChannels;
    if (tid.x >= n) return;
    if (float(hash32(tid.x * 2654435761u + 7u) & 0xFFFFFFu) >= params.sampleFraction * 16777216.0f) return;
    const uint plane = params.width * params.height;
    const uint i = tid.x % plane;
    const uint x = i % params.width;
    const int v = planes[tid.x];
    const uint xp = x == 0 ? 1 : x - 1;
    const int left = x == 0 ? (params.width > 1 ? planes[tid.x + 1] : v) : planes[tid.x - 1];
    const int d = v - left;
    uint before;
    InterlockedAdd(stats[uint(clamp(v, -int(kRange), int(kRange)) + int(kRange))], 1u, before);
    InterlockedAdd(stats[kHistBins + uint(clamp(d, -int(kRange), int(kRange)) + int(kRange))], 1u, before);
}

/// QuantizeHistogram over `hist[0..n)` into numChunks: writes the thresholds (bin
/// indices + bias) for property p and returns how many (the last one dropped).
uint quantizeHistogram(uint histBase, uint n, uint numChunks, int bias, uint p, bool absValues) {
    // abs: fold the bins around kRange first (quantize_abs_*: the samples' absolute values).
    uint64_t sum = 0;
    for (uint i = 0; i < n; ++i) sum += stats[histBase + i];
    if (sum == 0) return 0;
    uint count = 0;
    uint64_t cumsum = 0;
    uint64_t threshold = 1;
    for (uint i = 0; i < n && count < kThreshStride; ++i) {
        cumsum += stats[histBase + i];
        if (cumsum * numChunks >= threshold * sum) {
            thresholds[p * kThreshStride + count++] = int(i) + bias;
            while (cumsum * numChunks >= threshold * sum) ++threshold;
        }
    }
    return count > 0 ? count - 1 : 0;   // thresholds.pop_back()
}

/// QuantizeSamples over a +-kRange histogram: thresholds from the smallest used
/// value up, numChunks chunks.
uint quantizeSamples(uint histBase, uint numChunks, uint p) {
    uint minBin = 0;
    while (minBin < kHistBins && stats[histBase + minBin] == 0) ++minBin;
    if (minBin == kHistBins) return 0;
    return quantizeHistogram(histBase + minBin, kHistBins - minBin, numChunks, int(minBin) - int(kRange), p, false);
}

/// One thread: the thresholds of every candidate property
/// (TreeSamples::PreQuantizeProperties). The absolute-value properties fold the
/// pixel and difference histograms first (into scratch bins after them).
[shader("compute")]
[numthreads(1, 1, 1)]
void propThresholds(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const uint wpTable[29] = {uint(-255), uint(-191), uint(-127), uint(-95), uint(-63), uint(-47), uint(-31), uint(-23), uint(-15), uint(-11),
                              uint(-7), uint(-5), uint(-3), uint(-1), 0, 1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255};
    // Folded histograms: |pixel| at 2 kHistBins + v, |difference| at 2 kHistBins + kRange + 1 + v.
    const uint absPix = 2 * kHistBins, absDiff = 2 * kHistBins + kRange + 1;
    for (uint i = 0; i <= kRange; ++i) { stats[absPix + i] = 0; stats[absDiff + i] = 0; }
    for (uint i = 0; i < kHistBins; ++i) {
        const uint a = uint(abs(int(i) - int(kRange)));
        stats[absPix + a] += stats[i];
        stats[absDiff + a] += stats[kHistBins + i];
    }
    for (uint k = 0; k < kProps; ++k) {
        const uint p = k == 0 ? params.propIds.x : k == 1 ? params.propIds.y : k == 2 ? params.propIds.z : params.propIds.w;
        uint count = 0;
        if (p == 0) {   // channel: equal channel sizes, one bin per channel
            for (uint c = 0; c + 1 < params.numChannels && count < kThreshStride; ++c) thresholds[p * kThreshStride + count++] = int(c);
        } else if (p == 1) {   // group (stream id): QuantizeHistogram over the groups' pixel counts
            uint64_t sum = 0;
            for (uint g = 0; g < params.numGroups; ++g) {
                const uint w = min(params.groupDim, params.width - (g % params.groupsX) * params.groupDim);
                const uint h = min(params.groupDim, params.height - (g / params.groupsX) * params.groupDim);
                sum += uint64_t(w) * h * params.numChannels;
            }
            uint64_t cumsum = 0, threshold = 1;
            for (uint g = 0; g < params.numGroups && count < kThreshStride; ++g) {
                const uint w = min(params.groupDim, params.width - (g % params.groupsX) * params.groupDim);
                const uint h = min(params.groupDim, params.height - (g / params.groupsX) * params.groupDim);
                cumsum += uint64_t(w) * h * params.numChannels;
                if (cumsum * kVals >= threshold * sum) {
                    thresholds[p * kThreshStride + count++] = int(params.streamBase + g);
                    while (cumsum * kVals >= threshold * sum) ++threshold;
                }
            }
            if (count > 0) --count;
        } else if (p == 2 || p == 3) {   // coordinates
            for (uint i = 0; i + 1 < kVals && count < kThreshStride; ++i) thresholds[p * kThreshStride + count++] = int((i + 1) * 256 / kVals) - 1;
        } else if (p == 4 || p == 5) {
            count = quantizeHistogram(absPix, kRange + 1, kVals, 0, p, true);
        } else if (p == 6 || p == 7 || p == 8) {
            count = quantizeSamples(0, kVals, p);
        } else if (p == 15) {
            for (uint i = 0; i < 29; ++i) thresholds[p * kThreshStride + i] = int(wpTable[i]);
            count = 29;
        } else {   // 9..14: differences
            count = quantizeSamples(kHistBins, kVals, p);
        }
        thresholds[kNumProperties * kThreshStride + p] = int(count);
    }
}

uint slotOf(uint node) { return node - params.levelStart; }
uint countIndex(uint slot, uint p, uint val, uint pred, uint tok) { return slot * kSlotCounts + ((p * kVals + val) * kPreds + pred) * kToks + tok; }
uint extraIndex(uint slot, uint p, uint val, uint pred) { return slot * kSlotExtra + (p * kVals + val) * kPreds + pred; }

/// One thread per sample: its tokens counted into its node's histograms.
[shader("compute")]
[numthreads(64, 1, 1)]
void treeHistograms(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.numSamples) return;
    const uint node = sampleNode[tid.x];
    if (node < params.levelStart) return;   // settled in an earlier level
    const uint slot = slotOf(node);
    if (slot >= params.maxSlots) return;
    const uint2 s = samples[tid.x];
    const uint tok[2] = {min(s.x & 0xFFu, kToks - 1), min((s.x >> 16) & 0xFFu, kToks - 1)};
    const uint nbits[2] = {(s.x >> 8) & 0xFFu, s.x >> 24};
    uint before;
    for (uint p = 0; p < kProps; ++p) {
        const uint val = min((s.y >> (8 * p)) & 0xFFu, kVals - 1);
        for (uint pred = 0; pred < kPreds; ++pred) {
            InterlockedAdd(counts[countIndex(slot, p, val, pred, tok[pred])], 1u, before);
            if (nbits[pred] != 0) InterlockedAdd(extra[extraIndex(slot, p, val, pred)], nbits[pred], before);
        }
    }
}

/// EstimateBits: the cost of a histogram under an ideal code with probabilities
/// floored at 1 / 4096; a single-symbol histogram costs nothing.
float estimateBits(uint c[kToks], uint total) {
    if (total == 0) return 0.0f;
    const float invTotal = 1.0f / float(total);
    float bits = 0.0f;
    for (uint i = 0; i < kToks; ++i) {
        if (c[i] == 0 || c[i] == total) continue;
        bits -= float(c[i]) * log2(max(float(c[i]) * invTotal, kMinProb));
    }
    return bits;
}

struct Split {
    float cost;
    uint  prop;    // candidate index
    uint  val;
    uint  lpred;   // the "> split" side's predictor (libjxl's rpred, the lchild)
    uint  rpred;   // the "<= split" side's predictor (libjxl's lpred, the rchild)
};

static const uint kCategories = 4;   // 0 static constant, 1 static, 2 non-static, 3 no WP
groupshared float gCost[kCategories * kThreads];
groupshared uint  gProp[kCategories * kThreads];
groupshared uint  gVal[kCategories * kThreads];
groupshared uint  gLPred[kCategories * kThreads];
groupshared uint  gRPred[kCategories * kThreads];
groupshared uint  gTotal[kPreds * kToks];
groupshared uint  gTotalExtra[kPreds];
groupshared uint  gLastUsed[kProps];

uint propIdOf(uint k) { return k == 0 ? params.propIds.x : k == 1 ? params.propIds.y : k == 2 ? params.propIds.z : params.propIds.w; }

/// One thread-group per node of the level: FindBestSplit for that node.
[shader("compute")]
[numthreads(kThreads, 1, 1)]
void treeSplit(uint3 gid: SV_GroupID, uint3 ltid: SV_GroupThreadID) {
    const uint tid = ltid.x;
    const uint node = params.levelStart + gid.x;
    if (node >= counters[0] || gid.x >= params.maxSlots) return;
    const uint slot = gid.x;
    const uint nodePred = nodes[node * kNodeWords + 4];
    // Totals per predictor from property 0's bins (every sample is in exactly one).
    for (uint i = tid; i < kPreds * kToks; i += kThreads) {
        const uint pred = i / kToks, tok = i % kToks;
        uint t = 0;
        for (uint val = 0; val < kVals; ++val) t += counts[countIndex(slot, 0, val, pred, tok)];
        gTotal[i] = t;
    }
    if (tid < kPreds) {
        uint e = 0;
        for (uint val = 0; val < kVals; ++val) e += extra[extraIndex(slot, 0, val, tid)];
        gTotalExtra[tid] = e;
    }
    if (tid < kProps) {
        uint last = 0;
        for (uint val = 0; val < kVals; ++val) {
            uint used = 0;
            for (uint tok = 0; tok < kToks; ++tok) used += counts[countIndex(slot, tid, val, 0, tok)];
            if (used != 0) last = val;
        }
        gLastUsed[tid] = last;
    }
    GroupMemoryBarrierWithGroupSync();
    uint total = 0;
    uint totalC[kToks];
    for (uint tok = 0; tok < kToks; ++tok) { totalC[tok] = gTotal[nodePred * kToks + tok]; total += totalC[tok]; }
    const float baseBits = estimateBits(totalC, total) + float(gTotalExtra[nodePred]);
    const float changePenalty = 800.0f / (100.0f + params.threshold);

    Split best[kCategories];
    for (uint k = 0; k < kCategories; ++k) { best[k].cost = 3.4e38f; best[k].prop = 0; best[k].val = 0; best[k].lpred = 0; best[k].rpred = 0; }
    if (baseBits > params.threshold) {
        for (uint item = tid; item < kProps * kVals; item += kThreads) {
            const uint p = item / kVals, i = item % kVals;
            if (i >= gLastUsed[p]) continue;
            // Counts of the "<= i" side (below) per predictor; the other side is the total minus it.
            uint below[kPreds][kToks];
            uint extraBelow[kPreds];
            uint usedHere = 0;
            for (uint pred = 0; pred < kPreds; ++pred) {
                extraBelow[pred] = 0;
                for (uint tok = 0; tok < kToks; ++tok) below[pred][tok] = 0;
            }
            for (uint val = 0; val <= i; ++val) {
                for (uint pred = 0; pred < kPreds; ++pred) {
                    for (uint tok = 0; tok < kToks; ++tok) {
                        const uint c = counts[countIndex(slot, p, val, pred, tok)];
                        below[pred][tok] += c;
                        if (pred == 0 && val == i) usedHere += c;
                    }
                    extraBelow[pred] += extra[extraIndex(slot, p, val, pred)];
                }
            }
            if (usedHere == 0) continue;
            float bestL = 3.4e38f, bestR = 3.4e38f, bestLBase = 0.0f, bestRBase = 0.0f;
            uint predL = 0, predR = 0;
            for (uint pred = 0; pred < kPreds; ++pred) {
                uint above[kToks];
                uint nBelow = 0, nAbove = 0;
                for (uint tok = 0; tok < kToks; ++tok) {
                    above[tok] = gTotal[pred * kToks + tok] - below[pred][tok];
                    nBelow += below[pred][tok];
                    nAbove += above[tok];
                }
                const float rcost = estimateBits(above, nAbove) + float(gTotalExtra[pred] - extraBelow[pred]);
                const float lcost = estimateBits(below[pred], nBelow) + float(extraBelow[pred]);
                float penalty = 0.0f;
                if (pred != nodePred && nodePred != 0) penalty = changePenalty;   // predictor index 0 is Weighted
                if (pred == 0) penalty += 1e-8f;
                if (rcost + penalty < bestR) { bestR = rcost + penalty; bestRBase = rcost; predR = pred; }
                if (lcost + penalty < bestL) { bestL = lcost + penalty; bestLBase = lcost; predL = pred; }
            }
            const uint propId = propIdOf(p);
            const bool usesWp = propId == 15 || predL == 0 || predR == 0;
            const bool zeroSide = bestRBase == 0.0f || bestLBase == 0.0f;
            const uint cat = propId < 2 ? (zeroSide ? 0 : 1) : (usesWp ? 2 : 3);
            const float cost = bestLBase + bestRBase;
            if (cost < best[cat].cost) {
                best[cat].cost = cost;
                best[cat].prop = p;
                best[cat].val = i;
                best[cat].lpred = predR;   // the "> split" side (above) is the lchild
                best[cat].rpred = predL;   // the "<= split" side (below) is the rchild
            }
        }
    }
    for (uint k = 0; k < kCategories; ++k) {
        gCost[k * kThreads + tid] = best[k].cost;
        gProp[k * kThreads + tid] = best[k].prop;
        gVal[k * kThreads + tid] = best[k].val;
        gLPred[k * kThreads + tid] = best[k].lpred;
        gRPred[k * kThreads + tid] = best[k].rpred;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid != 0) return;
    Split cat[kCategories];
    for (uint k = 0; k < kCategories; ++k) {
        uint bi = 0;
        for (uint t = 1; t < kThreads; ++t) if (gCost[k * kThreads + t] < gCost[k * kThreads + bi]) bi = t;
        cat[k].cost = gCost[k * kThreads + bi];
        cat[k].prop = gProp[k * kThreads + bi];
        cat[k].val = gVal[k * kThreads + bi];
        cat[k].lpred = gLPred[k * kThreads + bi];
        cat[k].rpred = gRPred[k * kThreads + bi];
    }
    // libjxl's order of preference: non-static, then no-WP and static when close enough, then a static split with a constant side.
    Split chosen = cat[2];
    if (cat[3].cost + params.threshold < baseBits && cat[3].cost <= params.fastDecodeMul * chosen.cost) chosen = cat[3];
    if (cat[1].cost + params.threshold < baseBits && cat[1].cost <= params.fastDecodeMul * chosen.cost) chosen = cat[1];
    if (cat[0].cost + params.threshold < baseBits) chosen = cat[0];
    if (!(chosen.cost + params.threshold < baseBits)) return;
    uint base;
    InterlockedAdd(counters[0], 2u, base);
    if (base + 2 > params.maxNodes) {   // out of room: undo, stay a leaf
        InterlockedAdd(counters[0], uint(-2), base);
        return;
    }
    const uint propId = propIdOf(chosen.prop);
    nodes[node * kNodeWords + 0] = chosen.prop;
    nodes[node * kNodeWords + 1] = chosen.val;
    nodes[node * kNodeWords + 2] = base;
    nodes[node * kNodeWords + 3] = base + 1;
    nodes[node * kNodeWords + 5] = uint(thresholds[propId * kThreshStride + chosen.val]);
    const uint depth = nodes[node * kNodeWords + 6];
    for (uint child = 0; child < 2; ++child) {
        const uint n = (base + child) * kNodeWords;
        nodes[n + 0] = 0xFFFFFFFFu;
        nodes[n + 1] = 0;
        nodes[n + 2] = 0;
        nodes[n + 3] = 0;
        nodes[n + 4] = child == 0 ? chosen.lpred : chosen.rpred;
        nodes[n + 5] = 0;
        nodes[n + 6] = depth + 1;
        nodes[n + 7] = 0;
    }
    InterlockedAdd(counters[1], 1u, base);
}

/// One thread per sample: into the child its property value selects.
[shader("compute")]
[numthreads(64, 1, 1)]
void treeAssign(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.numSamples) return;
    const uint node = sampleNode[tid.x];
    if (node < params.levelStart) return;
    const uint p = nodes[node * kNodeWords + 0];
    if (p == 0xFFFFFFFFu) return;
    const uint val = (samples[tid.x].y >> (8 * p)) & 0xFFu;
    sampleNode[tid.x] = val > nodes[node * kNodeWords + 1] ? nodes[node * kNodeWords + 2] : nodes[node * kNodeWords + 3];
}
