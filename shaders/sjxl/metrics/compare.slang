// Copyright (c) 2026 jesus luque.
//
// Two 8-bit RGB images compared on the device: per thread group the sum of
// squared differences, and the largest difference with where it is (diff << 24
// | byte index, so images up to 16 MB of pixels name the place).
module metrics_compare;

struct CompareParams {
    uint bytes;   // width * height * 3
    uint pad0, pad1, pad2;
};

StructuredBuffer<uint>         a;          // bytes packed little-endian
StructuredBuffer<uint>         b;
RWStructuredBuffer<uint>       groupSums;  // one per thread group
RWStructuredBuffer<uint>       maxDiff;    // [0]: diff << 24 | index
ConstantBuffer<CompareParams>  params;

groupshared uint gSum;

uint byteAt(StructuredBuffer<uint> buf, uint i) {
    return (buf[i >> 2] >> ((i & 3) * 8)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void compareRgb8(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID) {
    if (lid.x == 0) {
        gSum = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint i = tid.x;
    if (i < params.bytes) {
        const int d = int(byteAt(a, i)) - int(byteAt(b, i));
        const uint ad = uint(abs(d));
        uint before;
        InterlockedAdd(gSum, ad * ad, before);
        InterlockedMax(maxDiff[0], (ad << 24) | (i & 0xFFFFFFu), before);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0) {
        groupSums[gid.x] = gSum;
    }
}
