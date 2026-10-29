// Copyright (c) 2026 jesus luque.
//
// The AC group sections as ANS streams, one thread per group: the group's
// tokens (varblocks in raster order by their first block, channels Y, X, B,
// the non-zero count then the coefficients, as acTokenize laid them out) coded
// backwards through the ANS state as libjxl's WriteTokens does, so the decoder
// reads them forwards. Mode 1 runs the chain to measure the section (32 bits
// of final state, the raw hybrid-uint bits and 16 bits per renormalisation);
// mode 2 runs it again and places every token's bits from the section's end
// down: in the stream a token is [renormalisation bits][raw bits], tokens in
// forward order after the state.
module enc_ac_sections;

#include "sjxl/common/hybrid_uint.slangh"
#include "sjxl/enc/ac_tokens.slangh"
#include "sjxl/enc/ans.slangh"

struct AcSectionParams {
    uint nblocks;
    uint blocksX;
    uint blocksY;
    uint numGroups;
    uint numDcGroups;
    uint mode;          // 1 count, 2 emit
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint4>    groups;         // AC groups: bx0, by0, w, h (blocks)
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     nonZeros;       // [c][block]
StructuredBuffer<uint>     acTokens;       // [c][block][slot]: ctx << 18 | value at coefficient index k
StructuredBuffer<uint>     numTokens;      // [c][block]
StructuredBuffer<uint>     nzCtx;          // [c][block]
StructuredBuffer<uint>     ctxMap;         // AC context -> cluster
StructuredBuffer<uint>     ansInfo;        // cluster c: freq | start << 16 per symbol at c * 256
StructuredBuffer<uint>     ansReverse;     // cluster c: slots at c * 4096
StructuredBuffer<uint>     sectionStart;   // mode 2: bit offset of each section
RWStructuredBuffer<uint>   acSectionBits;  // mode 1 output: bits per group
RWStructuredBuffer<uint>   body;           // mode 2 output
ConstantBuffer<AcSectionParams> params;

uint physicalChannel(uint ci) { return ci == 0 ? 1 : (ci == 1 ? 0 : 2); }

struct AnsWriter {
    uint mode;
    uint state;
    uint pos;     // mode 2: the next free bit, moving down
    uint bits;    // mode 1: bits so far
};

void put(inout AnsWriter aw, uint ctx, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    const uint cluster = ctxMap[ctx];
    const uint e = ansInfo[cluster * kAnsInfoStride + tok];
    const uint freq = e & 0xFFFFu;
    const uint start = e >> 16;
    if (aw.mode == 2) {
        aw.pos -= nbits;
        emitAtomic(body, aw.pos, nbits, raw);
    }
    uint ansBits, ansN;
    ansPut(aw.state, freq, start, ansReverse, cluster * kAnsReverseStride, ansN, ansBits);
    if (aw.mode == 2) {
        aw.pos -= ansN;
        emitAtomic(body, aw.pos, ansN, ansBits);
    }
    aw.bits += nbits + ansN;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void acSections(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.numGroups) {
        return;
    }
    const uint4 rect = groups[g];
    const uint section = 2 + params.numDcGroups + g;
    AnsWriter aw;
    aw.mode = params.mode;
    aw.state = kAnsSignature << 16;
    aw.bits = 32;
    aw.pos = aw.mode == 2 ? sectionStart[section] + acSectionBits[g] : 0;
    for (int by = int(rect.w) - 1; by >= 0; --by) {
        for (int bx = int(rect.z) - 1; bx >= 0; --bx) {
            const uint block = (rect.y + uint(by)) * params.blocksX + rect.x + uint(bx);
            if ((acs[block] & 1) == 0) continue;
            const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
            for (int ci = 2; ci >= 0; --ci) {
                const uint c = physicalChannel(uint(ci));
                const uint n = numTokens[c * params.nblocks + block];
                for (int k = int(vb.covered + n) - 1; k >= int(vb.covered); --k) {
                    const uint t = acTokens[coeffSlot(vb, c, params.nblocks, params.blocksX, uint(k))];
                    put(aw, t >> 18, t & 0x3FFFFu);
                }
                put(aw, nzCtx[c * params.nblocks + block], nonZeros[c * params.nblocks + block]);
            }
        }
    }
    if (aw.mode == 1) {
        acSectionBits[g] = aw.bits;
    } else {
        emitAtomic(body, sectionStart[section], 32, aw.state);
    }
}
