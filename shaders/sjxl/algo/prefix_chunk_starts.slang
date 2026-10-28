// Copyright (c) 2026 jesus luque.
//
// Prefix sum, step 2 of 3: where each chunk starts, and the grand total.
// One invocation: the chunk count is N / 4096, a few thousand at most.
module prefix_chunk_starts;

struct PrefixParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint pad0;
};

StructuredBuffer<uint>       chunkTotals;
RWStructuredBuffer<uint>     chunkStarts;
RWStructuredBuffer<uint>     total;
ConstantBuffer<PrefixParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void prefixChunkStarts(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint running = 0;
    for (uint c = 0; c < params.chunkCount; ++c) {
        chunkStarts[c] = running;
        running += chunkTotals[c];
    }
    total[0] = running;
}
