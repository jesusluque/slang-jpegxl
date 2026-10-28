// Copyright (c) 2026 jesus luque.
//
// Prefix sum, step 1 of 3: the sum of each chunk of the input.
//
// WHY CHUNKS AND NOT A BLELLOCH TREE
//
// A tree scan needs log2(N) dispatches each way and a power-of-two buffer.
// A chunked scan needs three dispatches whatever N is: one invocation per
// chunk walks its chunk in order (4096 elements is a short loop for a GPU
// thread), one invocation walks the chunk totals, and one invocation per chunk
// walks its chunk again writing offsets. No shared group memory, no atomics --
// nothing Metal, CUDA, Vulkan and D3D disagree about.
module prefix_chunk_totals;

struct PrefixParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint pad0;
};

StructuredBuffer<uint>       input;
RWStructuredBuffer<uint>     chunkTotals;
ConstantBuffer<PrefixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void prefixChunkTotals(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    uint sum = 0;
    for (uint i = begin; i < end; ++i) {
        sum += input[i];
    }
    chunkTotals[c] = sum;
}
