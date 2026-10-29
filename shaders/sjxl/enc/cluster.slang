// Copyright (c) 2026 jesus luque.
//
// Clustering of the AC contexts' histograms into the histograms the AC
// sections are coded with: libjxl's FastClusterHistograms (enc_cluster.cc)
// as one thread-group. Seeds are picked farthest-first by ClusterDistance
// (the bits the merge of two histograms costs over coding them apart, with
// the prefix-code correction for single-symbol histograms) until the
// farthest context is within kMinDistanceForDistinct of a seed or there are
// kMaxClusters; then every context goes to the nearest seed (libjxl assigns
// against the growing clusters in context order; against the fixed seeds the
// assignment is parallel). Clusters are renumbered by their first context,
// and a context without tokens takes the previous context's cluster, which
// keeps the coded context map cheap. The cluster histograms land in the code
// regions 0..numClusters-1 of `hist`, where the prefix codes are built.
module enc_cluster;

#include "sjxl/common/ac_context.slangh"

static const uint kMaxClusters = 32;
static const uint kThreads = 256;
static const uint kCodeStride = 256;
static const float kMinDistanceForDistinct = 48.0f;
static const float kFar = 3.0e38f;

struct ClusterParams {
    uint numContexts;
    uint maxClusters;   // 1..kMaxClusters
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>      ctxHist;      // numContexts * kAcAlphabet
RWStructuredBuffer<float>   ctxTotal;     // scratch, numContexts
RWStructuredBuffer<float>   ctxEntropy;   // scratch
RWStructuredBuffer<float>   ctxDist;      // scratch: distance to the nearest seed
RWStructuredBuffer<uint>    ctxSingle;    // scratch: 1 when the context has at most one symbol
RWStructuredBuffer<uint>    ctxMap;       // numContexts: cluster per context
RWStructuredBuffer<uint>    hist;         // code regions: cluster c's histogram at c * kCodeStride
RWStructuredBuffer<uint>    clusterInfo;  // [0] number of clusters
ConstantBuffer<ClusterParams> params;

groupshared uint  gSeedHist[kMaxClusters * kAcAlphabet];
groupshared float gSeedTotal[kMaxClusters];
groupshared float gSeedEntropy[kMaxClusters];
groupshared uint  gSeedSingle[kMaxClusters];
groupshared uint  gSeedCtx[kMaxClusters];
groupshared uint  gClusterHist[kMaxClusters * kAcAlphabet];
groupshared uint  gRank[kMaxClusters];
groupshared float gRedVal[kThreads];
groupshared uint  gRedIdx[kThreads];
groupshared uint  gNumSeeds;
groupshared uint  gStop;

float xlog2x(float x) { return x > 0.0f ? x * log2(x) : 0.0f; }

/// Shannon cost in bits of a histogram: total log2 total - sum c log2 c.
float entropyOf(uint base) {
    float total = 0.0f;
    float sum = 0.0f;
    for (uint s = 0; s < kAcAlphabet; ++s) {
        const float c = float(ctxHist[base + s]);
        total += c;
        sum += xlog2x(c);
    }
    return xlog2x(total) - sum;
}

/// ClusterDistance(context, seed j) with prefix codes.
float clusterDistance(uint ctx, uint j) {
    const float ta = ctxTotal[ctx];
    const float tb = gSeedTotal[j];
    if (ta == 0.0f || tb == 0.0f) return 0.0f;
    const uint base = ctx * kAcAlphabet;
    float sum = 0.0f;
    uint symbols = 0;
    for (uint s = 0; s < kAcAlphabet; ++s) {
        const uint c = ctxHist[base + s] + gSeedHist[j * kAcAlphabet + s];
        sum += xlog2x(float(c));
        symbols += c != 0 ? 1 : 0;
    }
    const float distance = xlog2x(ta + tb) - sum - ctxEntropy[ctx] - gSeedEntropy[j];
    if (symbols <= 1) return distance;
    float single = 0.0f;
    if (ctxSingle[ctx] != 0) single += ta;
    if (gSeedSingle[j] != 0) single += tb;
    return max(distance, single);
}

/// The first context with the largest value: every thread's best, then thread 0.
void argmaxReduce(uint tid, float val, uint idx) {
    gRedVal[tid] = val;
    gRedIdx[tid] = idx;
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        float best = -1.0f;
        uint bestIdx = 0xFFFFFFFFu;
        for (uint t = 0; t < kThreads; ++t) {
            if (gRedIdx[t] == 0xFFFFFFFFu) continue;
            if (gRedVal[t] > best || (gRedVal[t] == best && gRedIdx[t] < bestIdx)) {
                best = gRedVal[t];
                bestIdx = gRedIdx[t];
            }
        }
        gRedVal[0] = best;
        gRedIdx[0] = bestIdx;
    }
    GroupMemoryBarrierWithGroupSync();
}

void addSeed(uint tid, uint j, uint ctx) {
    for (uint s = tid; s < kAcAlphabet; s += kThreads) {
        gSeedHist[j * kAcAlphabet + s] = ctxHist[ctx * kAcAlphabet + s];
    }
    if (tid == 0) {
        gSeedTotal[j] = ctxTotal[ctx];
        gSeedEntropy[j] = ctxEntropy[ctx];
        gSeedSingle[j] = ctxSingle[ctx];
        gSeedCtx[j] = ctx;
        ctxDist[ctx] = 0.0f;
        ctxMap[ctx] = j;
    }
}

[shader("compute")]
[numthreads(kThreads, 1, 1)]
void clusterContexts(uint3 gtid: SV_GroupThreadID) {
    const uint tid = gtid.x;
    const uint n = params.numContexts;
    for (uint i = tid; i < kMaxClusters * kAcAlphabet; i += kThreads) gClusterHist[i] = 0;

    // Per-context statistics.
    for (uint ctx = tid; ctx < n; ctx += kThreads) {
        float total = 0.0f;
        uint symbols = 0;
        for (uint s = 0; s < kAcAlphabet; ++s) {
            const uint c = ctxHist[ctx * kAcAlphabet + s];
            total += float(c);
            symbols += c != 0 ? 1 : 0;
        }
        ctxTotal[ctx] = total;
        ctxEntropy[ctx] = entropyOf(ctx * kAcAlphabet);
        ctxSingle[ctx] = symbols <= 1 ? 1 : 0;
        ctxDist[ctx] = total == 0.0f ? 0.0f : kFar;
        ctxMap[ctx] = 0xFFFFFFFFu;
    }
    AllMemoryBarrierWithGroupSync();

    // Seed 0: the context with the most tokens.
    {
        float best = -1.0f;
        uint bestIdx = 0xFFFFFFFFu;
        for (uint ctx = tid; ctx < n; ctx += kThreads) {
            if (ctxTotal[ctx] > best) { best = ctxTotal[ctx]; bestIdx = ctx; }
        }
        argmaxReduce(tid, best, bestIdx);
    }
    if (gRedIdx[0] == 0xFFFFFFFFu) {   // no tokens at all
        if (tid == 0) clusterInfo[0] = 1;
        for (uint ctx = tid; ctx < n; ctx += kThreads) ctxMap[ctx] = 0;
        return;
    }
    addSeed(tid, 0, gRedIdx[0]);
    if (tid == 0) { gNumSeeds = 1; gStop = 0; }
    AllMemoryBarrierWithGroupSync();

    // Farthest-first seeding.
    for (uint round = 0; round + 1 < params.maxClusters; ++round) {
        const uint j = gNumSeeds - 1;
        float best = -1.0f;
        uint bestIdx = 0xFFFFFFFFu;
        for (uint ctx = tid; ctx < n; ctx += kThreads) {
            float d = ctxDist[ctx];
            if (d != 0.0f) {
                d = min(d, clusterDistance(ctx, j));
                ctxDist[ctx] = d;
            }
            if (d > best) { best = d; bestIdx = ctx; }
        }
        AllMemoryBarrierWithGroupSync();
        argmaxReduce(tid, best, bestIdx);
        if (gRedIdx[0] == 0xFFFFFFFFu || gRedVal[0] < kMinDistanceForDistinct) {
            break;
        }
        addSeed(tid, gNumSeeds, gRedIdx[0]);
        AllMemoryBarrierWithGroupSync();
        if (tid == 0) ++gNumSeeds;
        AllMemoryBarrierWithGroupSync();
    }
    const uint numSeeds = gNumSeeds;

    // Every context with tokens to its nearest seed; the cluster histograms.
    for (uint ctx = tid; ctx < n; ctx += kThreads) {
        if (ctxTotal[ctx] == 0.0f) continue;
        uint best = ctxMap[ctx];   // a seed is its own cluster
        if (best == 0xFFFFFFFFu) {
            float bestDist = kFar;
            for (uint j = 0; j < numSeeds; ++j) {
                const float d = clusterDistance(ctx, j);
                if (d < bestDist) { bestDist = d; best = j; }
            }
            ctxMap[ctx] = best;
        }
        for (uint s = 0; s < kAcAlphabet; ++s) {
            const uint c = ctxHist[ctx * kAcAlphabet + s];
            if (c != 0) {
                uint before;
                InterlockedAdd(gClusterHist[best * kAcAlphabet + s], c, before);
            }
        }
    }
    AllMemoryBarrierWithGroupSync();

    // Renumber by first appearance; empty contexts follow their predecessor.
    if (tid == 0) {
        for (uint j = 0; j < kMaxClusters; ++j) gRank[j] = 0xFFFFFFFFu;
        uint next = 0;
        uint prev = 0;
        for (uint ctx = 0; ctx < n; ++ctx) {
            const uint m = ctxMap[ctx];
            if (m == 0xFFFFFFFFu) {
                ctxMap[ctx] = prev;
                continue;
            }
            if (gRank[m] == 0xFFFFFFFFu) gRank[m] = next++;
            ctxMap[ctx] = gRank[m];
            prev = gRank[m];
        }
        clusterInfo[0] = next;
    }
    AllMemoryBarrierWithGroupSync();
    for (uint i = tid; i < numSeeds * kAcAlphabet; i += kThreads) {
        const uint j = i / kAcAlphabet;
        const uint s = i % kAcAlphabet;
        if (gRank[j] != 0xFFFFFFFFu) {
            hist[gRank[j] * kCodeStride + s] = gClusterHist[i];
        }
    }
}
