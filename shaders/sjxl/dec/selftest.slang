// Copyright (c) 2026 jesus luque.
//
// Compiles the decoder libraries and exercises them on a buffer: one thread
// decodes a histogram set and a modular stream from `cs` into `arena`, and
// reports its error code. Used by tests and by slangc checks.
module dec_selftest;

StructuredBuffer<uint>    cs;
RWStructuredBuffer<uint>  arena;

#include "sjxl/dec/entropy.slangh"
#include "sjxl/dec/modular.slangh"

struct SelfTestParams {
    uint startBit;
    uint endBit;
    uint arenaWords;
    uint numContexts;
    uint channelW;
    uint channelH;
    uint numChannels;
    uint streamId;
};

RWStructuredBuffer<uint>      result;   // [0] error, [1] bits consumed, [2] arena used, [3..] channel 0 pixels
ConstantBuffer<SelfTestParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void decodeStream(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    Dec d;
    d.br.pos = params.startBit;
    d.br.end = params.endBit;
    d.base = 0;
    d.top = 0;
    d.cap = params.arenaWords;
    d.error = kErrNone;
    const uint chansOff = alloc(d, kMaxChannels * kChanWords);
    uint numChannels = params.numChannels;
    for (uint i = 0; i < numChannels; ++i) {
        storeChannel(chansOff, i, newChannel(d, params.channelW, params.channelH, 0, 0));
    }
    GlobalTree g;
    g.present = false;
    decodeModularStream(d, chansOff, numChannels, params.streamId, g, 0xFFFFFFFFu, 8);
    result[0] = d.error;
    result[1] = d.br.pos - params.startBit;
    result[2] = d.top;
    const Channel c0 = loadChannel(chansOff, 0);
    const uint n = min(c0.w * c0.h, 4096u);
    for (uint i = 0; i < n; ++i) result[3 + i] = arena[c0.data + i];
}
