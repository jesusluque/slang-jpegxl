// Copyright (c) 2026 jesus luque.
#include "sjxl/codestream/Headers.h"

#include <cstring>

namespace sjxl::codestream {
namespace {

// SizeHeader dimension (libjxl headers.cc): BitsOffset(9,1)/(13,1)/(18,1)/(30,1).
constexpr U32Enc kDimEnc{{BitsOffset(9, 1), BitsOffset(13, 1), BitsOffset(18, 1), BitsOffset(30, 1)}};
// TOC entry size (libjxl toc.h kTocDist).
constexpr U32Enc kTocDist{{Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408), BitsOffset(30, 4211712)}};

void writeSizeHeader(BitWriter& w, uint32_t xsize, uint32_t ysize) {
    const bool small = xsize % 8 == 0 && ysize % 8 == 0 && xsize <= 256 && ysize <= 256;
    w.writeBool(small);
    if (small) {
        w.write(5, ysize / 8 - 1);
    } else {
        writeU32(w, kDimEnc, ysize);
    }
    w.write(3, 0);   // ratio = 0: xsize sent explicitly
    if (small) {
        w.write(5, xsize / 8 - 1);
    } else {
        writeU32(w, kDimEnc, xsize);
    }
}

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}   // namespace

void writeU32(BitWriter& w, const U32Enc& enc, uint32_t value) {
    uint32_t bestSelector = 0;
    size_t bestBits = 64;
    for (uint32_t s = 0; s < 4; ++s) {
        const Distr& d = enc.d[s];
        if (d.direct) {
            if (d.valueOrOffset == value) {
                bestSelector = s;
                bestBits = 2;
                break;
            }
            continue;
        }
        if (value < d.valueOrOffset || value - d.valueOrOffset >= (uint64_t{1} << d.extraBits)) {
            continue;
        }
        if (2 + d.extraBits < bestBits) {
            bestSelector = s;
            bestBits = 2 + d.extraBits;
        }
    }
    assert(bestBits != 64 && "value not representable by this U32 encoding");
    w.write(2, bestSelector);
    const Distr& chosen = enc.d[bestSelector];
    if (!chosen.direct) {
        w.write(chosen.extraBits, value - chosen.valueOrOffset);
    }
}

void writeU64(BitWriter& w, uint64_t value) {
    if (value == 0) {
        w.write(2, 0);
        return;
    }
    if (value <= 16) {
        w.write(2, 1);
        w.write(4, value - 1);
        return;
    }
    if (value <= 272) {
        w.write(2, 2);
        w.write(8, value - 17);
        return;
    }
    w.write(2, 3);
    value -= 273;
    w.write(12, value & 0xFFF);
    value >>= 12;
    for (int i = 0; i < 8; ++i) {
        if (value == 0) {
            w.write(1, 0);
            return;
        }
        w.write(1, 1);
        const size_t chunk = i < 7 ? 8u : 4u;
        w.write(chunk, value & ((1u << chunk) - 1));
        value >>= chunk;
    }
}

void writeF16(BitWriter& w, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFu;
    uint16_t half = static_cast<uint16_t>(sign);
    if (value != 0.0f) {
        assert(exp > 0 && exp < 31);
        half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
    w.write(16, half);
}

void writeCodestreamHeaders(BitWriter& w, uint32_t xsize, uint32_t ysize) {
    w.write(8, 0xFF);
    w.write(8, 0x0A);
    writeSizeHeader(w, xsize, ysize);
    w.writeBool(true);   // ImageMetadata all_default: 8-bit, xyb_encoded, sRGB
    w.writeBool(true);   // transform_data all_default
    // The decoder jumps to a byte boundary before the frame header.
    w.zeroPadToByte();
}

void writeFrameHeader(BitWriter& w, const FrameParams& params) {
    w.writeBool(false);                                            // all_default
    writeU32(w, U32Enc{{Val(0), Val(1), Val(2), Val(3)}}, 0);      // frame_type = kRegularFrame
    w.writeBool(false);                                            // encoding = kVarDCT
    writeU64(w, 0);                                                // flags
    // xyb_encoded forces color_transform = kXYB: nothing written.
    writeU32(w, U32Enc{{Val(1), Val(2), Val(4), Val(8)}}, 1);      // upsampling = 1
    w.write(3, params.xQmScale);                                   // x_qm_scale
    w.write(3, params.bQmScale);                                   // b_qm_scale
    writeU32(w, U32Enc{{Val(1), Val(2), Val(3), BitsOffset(3, 4)}}, 1);   // num_passes = 1
    w.writeBool(false);                                            // custom_size_or_origin
    writeU32(w, U32Enc{{Val(0), Val(1), Val(2), BitsOffset(2, 3)}}, 0);   // blending mode kReplace
    w.writeBool(true);                                             // is_last
    writeU32(w, U32Enc{{Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48)}}, 0);   // name length
    // LoopFilter (loop_filter.cc VisitFields).
    const bool loopFilterDefault = params.gaborish && params.epfIters == 2;
    w.writeBool(loopFilterDefault);
    if (!loopFilterDefault) {
        w.writeBool(params.gaborish);
        if (params.gaborish) {
            w.writeBool(false);   // gab_custom
        }
        w.write(2, params.epfIters);
        if (params.epfIters > 0) {
            w.writeBool(false);   // epf_sharp_custom
            w.writeBool(false);   // epf_weight_custom
            w.writeBool(false);   // epf_sigma_custom
        }
        writeU64(w, 0);           // LoopFilter extensions
    }
    writeU64(w, 0);               // FrameHeader extensions
}

void writeToc(BitWriter& w, std::span<const uint32_t> sectionBytes) {
    w.writeBool(false);   // no permutation
    w.zeroPadToByte();
    for (uint32_t size : sectionBytes) {
        writeU32(w, kTocDist, size);
    }
    w.zeroPadToByte();
}

FrameDim frameDim(uint32_t xsize, uint32_t ysize, uint32_t groupSizeShift) {
    FrameDim d;
    d.xsize = xsize;
    d.ysize = ysize;
    d.groupDim = 128u << groupSizeShift;
    const uint32_t groupBlocks = d.groupDim / kBlockDim;
    d.xsizeBlocks = ceilDiv(xsize, kBlockDim);
    d.ysizeBlocks = ceilDiv(ysize, kBlockDim);
    d.xsizePadded = d.xsizeBlocks * kBlockDim;
    d.ysizePadded = d.ysizeBlocks * kBlockDim;
    d.xsizeGroups = ceilDiv(d.xsizeBlocks, groupBlocks);
    d.ysizeGroups = ceilDiv(d.ysizeBlocks, groupBlocks);
    d.xsizeDcGroups = ceilDiv(d.xsizeBlocks, d.groupDim);
    d.ysizeDcGroups = ceilDiv(d.ysizeBlocks, d.groupDim);
    return d;
}

}   // namespace sjxl::codestream
