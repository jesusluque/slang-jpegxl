// Copyright (c) 2026 jesus luque.
//
// Test kernels: fill a buffer with a pattern, and count mismatches between
// two buffers -- how a test checks a GPU result without reading data back.
module fill;

struct FillParams {
    uint count;
    uint multiplier;
    uint offset;
    uint pad0;
};

RWStructuredBuffer<uint>    output;
ConstantBuffer<FillParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void fill(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    output[i] = i * params.multiplier + params.offset;
}

StructuredBuffer<uint>      a;
StructuredBuffer<uint>      b;
RWStructuredBuffer<uint>    mismatches;

[shader("compute")]
[numthreads(64, 1, 1)]
void countMismatches(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    if (a[i] != b[i]) {
        uint before;
        InterlockedAdd(mismatches[0], 1u, before);
    }
}

/// The exclusive prefix sum of fill(i) = i*m + o, in closed form: how the
/// scan is checked without a CPU oracle.
RWStructuredBuffer<uint>    expected;

[shader("compute")]
[numthreads(64, 1, 1)]
void expectedPrefix(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    // sum_{k<i} (k*m + o) = m*i*(i-1)/2 + o*i   (mod 2^32)
    const uint tri = (i % 2 == 0) ? (i / 2) * (i - 1) : i * ((i - 1) / 2);
    expected[i] = tri * params.multiplier + params.offset * i;
}
