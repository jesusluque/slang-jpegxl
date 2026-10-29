// Copyright (c) 2026 jesus luque.
//
// Two images of 8- or 16-bit samples compared on the device: per thread group
// the sum of squared differences and the largest difference (compareSamples),
// then the first sample reaching that difference (locateMax).
module metrics_compare;

struct CompareParams {
    uint samples;        // width * height * channels
    uint bytesPerSample; // 1 or 2 (little-endian)
    uint pad0, pad1;
};

StructuredBuffer<uint>         a;          // bytes packed little-endian
StructuredBuffer<uint>         b;
RWStructuredBuffer<uint>       groupSums;  // one per thread group
RWStructuredBuffer<uint>       maxDiff;    // [0]: largest difference, [1]: first sample index reaching it
ConstantBuffer<CompareParams>  params;

groupshared uint gSum;

uint byteAt(StructuredBuffer<uint> buf, uint i) { return (buf[i >> 2] >> ((i & 3) * 8)) & 0xFFu; }

uint sampleAt(StructuredBuffer<uint> buf, uint i) {
    if (params.bytesPerSample == 1) return byteAt(buf, i);
    return byteAt(buf, 2 * i) | (byteAt(buf, 2 * i + 1) << 8);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compareSamples(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    if (lid.x == 0) gSum = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint i = tid.x;
    if (i < params.samples) {
        const int d = int(sampleAt(a, i)) - int(sampleAt(b, i));
        const uint ad = uint(abs(d));
        uint before;
        InterlockedAdd(gSum, ad * ad, before);
        InterlockedMax(maxDiff[0], ad, before);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0) groupSums[gid.x] = gSum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void locateMax(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.samples) return;
    const int d = int(sampleAt(a, i)) - int(sampleAt(b, i));
    if (uint(abs(d)) == maxDiff[0]) {
        uint before;
        InterlockedMin(maxDiff[1], i, before);
    }
}
