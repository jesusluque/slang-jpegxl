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
    uint ecInGroups;    // 1: an alpha channel's modular group follows the AC stream in every section
    uint ecCtxBase;     // the alpha set's first context in ctxMap
    uint ecRegionBase;  // its first code region
    uint width;         // image size in pixels (the alpha channel is not block-padded)
    uint height;
    uint groupDim;
    uint numPasses;     // AC sections: pass p of group g is section 2 + numDcGroups + p * numGroups + g
};

StructuredBuffer<uint4>    groups;         // AC groups: bx0, by0, w, h (blocks)
StructuredBuffer<uint>     acs;
StructuredBuffer<uint>     nonZeros;       // [pass][c][block]
StructuredBuffer<uint>     acTokens;       // [pass][c][block][slot]: ctx << 18 | value at coefficient index k
StructuredBuffer<uint>     numTokens;      // [pass][c][block]
StructuredBuffer<uint>     nzCtx;          // [pass][c][block]
StructuredBuffer<uint>     ctxMap;         // AC context -> cluster; pass p's set at p * kNumAcContexts, code region p * 32
StructuredBuffer<uint>     ansInfo;        // cluster c: freq | start << 16 per symbol at c * 256
StructuredBuffer<uint>     ansReverse;     // cluster c: slots at c * 4096
StructuredBuffer<uint>     sectionStart;   // mode 2: bit offset of each section
StructuredBuffer<uint>     ecTokens;       // the alpha channel's tokens per pixel (ctx << 18 | value)
RWStructuredBuffer<uint>   acSectionBits;  // mode 1 output: bits per (pass, group) section (AC stream + the alpha group)
RWStructuredBuffer<uint>   acOnlyBits;     // mode 1 output: the AC stream's bits alone
RWStructuredBuffer<uint>   body;           // mode 2 output
ConstantBuffer<AcSectionParams> params;

uint physicalChannel(uint ci) { return ci == 0 ? 1 : (ci == 1 ? 0 : 2); }

struct AnsWriter {
    uint mode;
    uint state;
    uint pos;     // mode 2: the next free bit, moving down
    uint bits;    // mode 1: bits so far
};

void putIn(inout AnsWriter aw, uint ctxBase, uint regionBase, uint ctx, uint value) {
    uint tok, nbits, raw;
    hybridEncode(value, tok, nbits, raw);
    const uint cluster = regionBase + ctxMap[ctxBase + ctx];
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
    const uint s = tid.x;   // pass * numGroups + g
    if (s >= params.numGroups * params.numPasses) {
        return;
    }
    const uint pass = s / params.numGroups;
    const uint g = s % params.numGroups;
    const uint4 rect = groups[g];
    const uint section = 2 + params.numDcGroups + s;
    const uint pb = pass * 3 * params.nblocks;
    const uint ctxBase = pass * kNumAcContexts;
    const uint regionBase = pass * 32;
    AnsWriter aw;
    aw.mode = params.mode;
    aw.state = kAnsSignature << 16;
    aw.bits = 32;
    aw.pos = aw.mode == 2 ? sectionStart[section] + acOnlyBits[s] : 0;
    for (int by = int(rect.w) - 1; by >= 0; --by) {
        for (int bx = int(rect.z) - 1; bx >= 0; --bx) {
            const uint block = (rect.y + uint(by)) * params.blocksX + rect.x + uint(bx);
            if ((acs[block] & 1) == 0) continue;
            const Varblock vb = varblockOf(block, acs[block] >> 1, params.blocksX);
            for (int ci = 2; ci >= 0; --ci) {
                const uint c = physicalChannel(uint(ci));
                const uint n = numTokens[pb + c * params.nblocks + block];
                for (int k = int(vb.covered + n) - 1; k >= int(vb.covered); --k) {
                    const uint t = acTokens[pb * 64 + coeffSlot(vb, c, params.nblocks, params.blocksX, uint(k))];
                    putIn(aw, ctxBase, regionBase, t >> 18, t & 0x3FFFFu);
                }
                putIn(aw, ctxBase, regionBase, nzCtx[pb + c * params.nblocks + block], nonZeros[pb + c * params.nblocks + block]);
            }
        }
    }
    if (aw.mode == 1) {
        acOnlyBits[s] = aw.bits;
    } else {
        emitAtomic(body, sectionStart[section], 32, aw.state);
    }
    // The alpha channel's modular group goes with the last pass (its shift 0 is
    // in that pass's downsampling bracket).
    if (params.ecInGroups == 0 || pass + 1 != params.numPasses) {
        if (aw.mode == 1) acSectionBits[s] = aw.bits;
        return;
    }
    // The alpha channel's modular group: GroupHeader (global tree, default WP
    // header, no transforms: 4 bits), then its ANS stream over the group's pixels.
    const uint x0 = rect.x * 8, y0 = rect.y * 8;
    const uint w = min(params.groupDim, params.width - x0);
    const uint h = min(params.groupDim, params.height - y0);
    AnsWriter ew;
    ew.mode = params.mode;
    ew.state = kAnsSignature << 16;
    ew.bits = 4 + 32;
    const uint ecStart = sectionStart[section] + acOnlyBits[s];
    ew.pos = ew.mode == 2 ? sectionStart[section] + acSectionBits[s] : 0;
    for (int y = int(h) - 1; y >= 0; --y) {
        for (int x = int(w) - 1; x >= 0; --x) {
            const uint t = ecTokens[(y0 + uint(y)) * params.width + x0 + uint(x)];
            putIn(ew, params.ecCtxBase, params.ecRegionBase, t >> 18, t & 0x3FFFFu);
        }
    }
    if (ew.mode == 1) {
        acSectionBits[s] = aw.bits + ew.bits;
    } else {
        emitAtomic(body, ecStart, 4, 0x3u);
        emitAtomic(body, ecStart + 4, 32, ew.state);
    }
}
