// Copyright (c) 2026 jesus luque.
#include "sjxl/codestream/Parse.h"

#include <algorithm>
#include <array>
#include <format>
#include <cmath>
#include <cstring>

namespace sjxl::codestream {
namespace {

constexpr U32Enc kDimEnc{{BitsOffset(9, 1), BitsOffset(13, 1), BitsOffset(18, 1), BitsOffset(30, 1)}};
constexpr U32Enc kTocDist{{Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408), BitsOffset(30, 4211712)}};

Error unsupported(std::string what) { return Error(ErrorCode::Unsupported, std::move(what)); }
Error corrupt(std::string what) { return Error(ErrorCode::Corrupt, std::move(what)); }

uint32_t be32(const uint8_t* p) {
    return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | p[3];
}

/// A box container: the codestream is the concatenation of jxlc / jxlp payloads
/// (jxlp carries a 4-byte index first). Other boxes are skipped.
Result<std::vector<uint8_t>> unwrapContainer(std::span<const uint8_t> file) {
    static constexpr uint8_t kSignature[12] = {0, 0, 0, 0x0C, 'J', 'X', 'L', ' ', 0x0D, 0x0A, 0x87, 0x0A};
    if (file.size() >= 2 && file[0] == 0xFF && file[1] == 0x0A) {
        return std::vector<uint8_t>(file.begin(), file.end());
    }
    if (file.size() < 12 || std::memcmp(file.data(), kSignature, 12) != 0) {
        return corrupt("not a JPEG XL codestream or container");
    }
    std::vector<uint8_t> out;
    size_t pos = 12;
    while (pos + 8 <= file.size()) {
        uint64_t size = be32(&file[pos]);
        const char* type = reinterpret_cast<const char*>(&file[pos + 4]);
        size_t headerBytes = 8;
        if (size == 1) {
            if (pos + 16 > file.size()) return corrupt("truncated box");
            size = (uint64_t{be32(&file[pos + 8])} << 32) | be32(&file[pos + 12]);
            headerBytes = 16;
        }
        if (size == 0) {
            size = file.size() - pos;   // to the end of the file
        }
        if (size < headerBytes || pos + size > file.size()) return corrupt("box size out of bounds");
        const uint8_t* payload = &file[pos + headerBytes];
        const size_t payloadBytes = static_cast<size_t>(size - headerBytes);
        if (std::memcmp(type, "jxlc", 4) == 0) {
            out.insert(out.end(), payload, payload + payloadBytes);
        } else if (std::memcmp(type, "jxlp", 4) == 0) {
            if (payloadBytes < 4) return corrupt("jxlp box without index");
            out.insert(out.end(), payload + 4, payload + payloadBytes);
        }
        pos += static_cast<size_t>(size);
    }
    if (out.size() < 2 || out[0] != 0xFF || out[1] != 0x0A) return corrupt("container holds no codestream");
    return out;
}

Result<void> readSizeHeader(BitReader& br, uint32_t& xsize, uint32_t& ysize) {
    const bool small = br.readBool();
    if (small) {
        ysize = (br.read(5) + 1) * 8;
    } else {
        ysize = br.readU32(kDimEnc);
    }
    const uint32_t ratio = br.read(3);
    if (ratio == 0) {
        xsize = small ? (br.read(5) + 1) * 8 : br.readU32(kDimEnc);
    } else {
        static constexpr uint32_t kRatios[7][2] = {{1, 1}, {12, 10}, {4, 3}, {3, 2}, {16, 9}, {5, 4}, {2, 1}};
        xsize = static_cast<uint32_t>(uint64_t{ysize} * kRatios[ratio - 1][0] / kRatios[ratio - 1][1]);
    }
    if (xsize == 0 || ysize == 0) return corrupt("zero image size");
    return ok();
}

Result<void> readBitDepth(BitReader& br, ImageMetadata& m) {
    m.floatingPointSample = br.readBool();
    if (!m.floatingPointSample) {
        m.bitsPerSample = br.readU32(U32Enc{{Val(8), Val(10), Val(12), BitsOffset(6, 1)}});
        m.exponentBitsPerSample = 0;
    } else {
        m.bitsPerSample = br.readU32(U32Enc{{Val(32), Val(16), Val(24), BitsOffset(6, 1)}});
        m.exponentBitsPerSample = br.read(4) + 1;
    }
    return ok();
}

Result<void> readCustomXy(BitReader& br) {
    const U32Enc enc{{Bits(19), BitsOffset(19, 524288), BitsOffset(20, 1048576), BitsOffset(21, 2097152)}};
    (void)br.readU32(enc);
    (void)br.readU32(enc);
    return ok();
}

Result<void> readColorEncoding(BitReader& br, ImageMetadata& m) {
    if (br.readBool()) {   // all_default: sRGB
        return ok();
    }
    m.wantIcc = br.readBool();
    m.colorSpace = br.readEnum();
    if (!m.wantIcc) {
        const bool implicitWhitePoint = m.colorSpace == 2;   // XYB
        if (!implicitWhitePoint) {
            m.whitePoint = br.readEnum();
            if (m.whitePoint == 2) SJXL_TRY(readCustomXy(br));
        }
        const bool hasPrimaries = m.colorSpace != 1 && m.colorSpace != 2;   // not grey, not XYB
        if (hasPrimaries) {
            m.primaries = br.readEnum();
            if (m.primaries == 2) {
                SJXL_TRY(readCustomXy(br));
                SJXL_TRY(readCustomXy(br));
                SJXL_TRY(readCustomXy(br));
            }
        }
        // CustomTransferFunction: implicit (linear) for XYB.
        if (m.colorSpace != 2) {
            m.tfHaveGamma = br.readBool();
            if (m.tfHaveGamma) {
                m.tfGamma = br.read(24);
            } else {
                m.transferFunction = br.readEnum();
            }
        }
        m.renderingIntent = br.readEnum();
    }
    return ok();
}

/// Bundle extensions: a U64 mask, one U64 size per set bit, then the bits are skipped.
Result<void> skipExtensions(BitReader& br) {
    const uint64_t extensions = br.readU64();
    if (extensions == 0) return ok();
    uint64_t total = 0;
    for (uint64_t remaining = extensions; remaining != 0; remaining &= remaining - 1) {
        total += br.readU64();
    }
    if (total > br.totalBits()) return corrupt("extension bits overflow");
    br.consume(static_cast<uint32_t>(total));
    return ok();
}

Result<void> readImageMetadata(BitReader& br, ImageMetadata& m) {
    if (br.readBool()) {   // all_default
        return ok();
    }
    const bool extraFields = br.readBool();
    if (extraFields) {
        m.orientation = br.read(3) + 1;
        m.haveIntrinsicSize = br.readBool();
        if (m.haveIntrinsicSize) {
            uint32_t ix, iy;
            SJXL_TRY(readSizeHeader(br, ix, iy));
        }
        m.havePreview = br.readBool();
        if (m.havePreview) {
            // PreviewHeader
            const bool div8 = br.readBool();
            if (div8) (void)br.readU32(U32Enc{{Val(16), Val(32), BitsOffset(5, 1), BitsOffset(9, 33)}});
            else (void)br.readU32(U32Enc{{BitsOffset(6, 1), BitsOffset(8, 65), BitsOffset(10, 321), BitsOffset(12, 1345)}});
            const uint32_t ratio = br.read(3);
            if (ratio == 0) {
                if (div8) (void)br.readU32(U32Enc{{Val(16), Val(32), BitsOffset(5, 1), BitsOffset(9, 33)}});
                else (void)br.readU32(U32Enc{{BitsOffset(6, 1), BitsOffset(8, 65), BitsOffset(10, 321), BitsOffset(12, 1345)}});
            }
        }
        m.haveAnimation = br.readBool();
        if (m.haveAnimation) {
            m.tpsNumerator = br.readU32(U32Enc{{Val(100), Val(1000), BitsOffset(10, 1), BitsOffset(30, 1)}});
            m.tpsDenominator = br.readU32(U32Enc{{Val(1), Val(1001), BitsOffset(8, 1), BitsOffset(10, 1)}});
            m.numLoops = br.readU32(U32Enc{{Val(0), Bits(3), Bits(16), Bits(32)}});
            m.haveTimecodes = br.readBool();
        }
    }
    SJXL_TRY(readBitDepth(br, m));
    m.modular16BitBufferSufficient = br.readBool();
    m.numExtraChannels = br.readU32(U32Enc{{Val(0), Val(1), BitsOffset(4, 2), BitsOffset(12, 1)}});
    m.extraChannels.resize(m.numExtraChannels);
    for (ExtraChannelInfo& ec : m.extraChannels) {
        // ExtraChannelInfo
        if (br.readBool()) continue;   // all_default: 8-bit alpha, not associated
        ec.type = br.readEnum();
        ImageMetadata depth;
        SJXL_TRY(readBitDepth(br, depth));
        if (depth.floatingPointSample) return Error(ErrorCode::Unsupported, "floating point extra channels");
        ec.bitsPerSample = depth.bitsPerSample;
        ec.dimShift = br.readU32(U32Enc{{Val(0), Val(3), Val(4), BitsOffset(3, 1)}});
        if ((1u << ec.dimShift) > 8) return Error(ErrorCode::Corrupt, "extra channel dim_shift");
        const uint32_t nameLength = br.readU32(U32Enc{{Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48)}});
        for (uint32_t k = 0; k < nameLength; ++k) (void)br.read(8);
        if (ec.type == 0) ec.alphaAssociated = br.readBool();
        if (ec.type == 2) for (size_t k = 0; k < 4; ++k) { SJXL_ASSIGN(ec.spotColor[k], br.readF16()); }   // spot colour: RGB and solidity
        if (ec.type == 5) (void)br.readU32(U32Enc{{Val(1), Bits(2), BitsOffset(4, 3), BitsOffset(8, 19)}});   // CFA
    }
    m.xybEncoded = br.readBool();
    SJXL_TRY(readColorEncoding(br, m));
    if (extraFields) {
        // ToneMapping
        if (!br.readBool()) {
            SJXL_ASSIGN(m.intensityTarget, br.readF16());
            SJXL_ASSIGN(m.minNits, br.readF16());
            m.relativeToMaxDisplay = br.readBool();
            SJXL_ASSIGN(m.linearBelow, br.readF16());
        }
    }
    SJXL_TRY(skipExtensions(br));
    return ok();
}

Result<void> readTransformData(BitReader& br, ImageMetadata& m) {
    // Defaults (cms/opsin_params.h).
    m.opsinInverseMatrix = {11.031566901960783f, -9.866943921568629f, -0.16462299647058826f,
                            -3.254147380392157f, 4.418770392156863f,  -0.16462299647058826f,
                            -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f};
    m.opsinBiases = {-0.0037930732552754493f, -0.0037930732552754493f, -0.0037930732552754493f};
    m.quantBiases = {1.0f - 0.05465007330715401f, 1.0f - 0.07005449891748593f, 1.0f - 0.049935103337343655f, 0.145f};
    if (br.readBool()) return ok();   // all_default
    if (m.xybEncoded) {
        // OpsinInverseMatrix
        if (!br.readBool()) {
            m.customOpsinMatrix = true;
            for (float& v : m.opsinInverseMatrix) { SJXL_ASSIGN(v, br.readF16()); }
            for (float& v : m.opsinBiases) { SJXL_ASSIGN(v, br.readF16()); }
            for (float& v : m.quantBiases) { SJXL_ASSIGN(v, br.readF16()); }
        }
    }
    m.customWeightsMask = br.read(3);
    if (m.customWeightsMask & 1) for (float& w : m.upsamplingWeights2) { SJXL_ASSIGN(w, br.readF16()); }
    if (m.customWeightsMask & 2) for (float& w : m.upsamplingWeights4) { SJXL_ASSIGN(w, br.readF16()); }
    if (m.customWeightsMask & 4) for (float& w : m.upsamplingWeights8) { SJXL_ASSIGN(w, br.readF16()); }
    return ok();
}

Result<void> readLoopFilter(BitReader& br, LoopFilter& lf, bool isModular) {
    if (br.readBool()) return ok();   // all_default
    lf.gab = br.readBool();
    if (lf.gab) {
        lf.gabCustom = br.readBool();
        if (lf.gabCustom) {
            SJXL_ASSIGN(lf.gabXWeight1, br.readF16());
            SJXL_ASSIGN(lf.gabXWeight2, br.readF16());
            SJXL_ASSIGN(lf.gabYWeight1, br.readF16());
            SJXL_ASSIGN(lf.gabYWeight2, br.readF16());
            SJXL_ASSIGN(lf.gabBWeight1, br.readF16());
            SJXL_ASSIGN(lf.gabBWeight2, br.readF16());
        }
    }
    lf.epfIters = br.read(2);
    if (lf.epfIters > 0) {
        if (!isModular) {
            lf.epfSharpCustom = br.readBool();
            if (lf.epfSharpCustom) {
                for (float& v : lf.epfSharpLut) { SJXL_ASSIGN(v, br.readF16()); }
            }
        }
        lf.epfWeightCustom = br.readBool();
        if (lf.epfWeightCustom) {
            for (float& v : lf.epfChannelScale) { SJXL_ASSIGN(v, br.readF16()); }
            SJXL_ASSIGN(lf.epfPass1Zeroflush, br.readF16());
            SJXL_ASSIGN(lf.epfPass2Zeroflush, br.readF16());
        }
        lf.epfSigmaCustom = br.readBool();
        if (lf.epfSigmaCustom) {
            if (!isModular) { SJXL_ASSIGN(lf.epfQuantMul, br.readF16()); }
            SJXL_ASSIGN(lf.epfPass0SigmaScale, br.readF16());
            SJXL_ASSIGN(lf.epfPass2SigmaScale, br.readF16());
            SJXL_ASSIGN(lf.epfBorderSadMul, br.readF16());
        }
        if (isModular) { SJXL_ASSIGN(lf.epfSigmaForModular, br.readF16()); }
    }
    SJXL_TRY(skipExtensions(br));
    return ok();
}

Result<void> readFrameHeader(BitReader& br, const ImageMetadata& m, FrameHeader& f) {
    if (br.readBool()) return ok();   // all_default
    f.frameType = br.readU32(U32Enc{{Val(0), Val(1), Val(2), Val(3)}});
    f.isModular = br.readBool();
    f.flags = br.readU64();
    if (m.xybEncoded) {
        f.colorTransform = 0;
    } else {
        f.colorTransform = br.readBool() ? 2 : 1;
    }
    if (f.colorTransform == 2 && (f.flags & FrameHeader::kUseDcFrame) == 0) {
        // YCbCrChromaSubsampling: 3 x Bits(2); kHShift = {0, 1, 1, 0}, kVShift = {0, 1, 0, 1}
        constexpr uint32_t kH[4] = {0, 1, 1, 0}, kV[4] = {0, 1, 0, 1};
        for (uint32_t& mode : f.chromaMode) mode = br.read(2);
        for (uint32_t c = 0; c < 3; ++c) {
            f.maxHShift = std::max(f.maxHShift, kH[f.chromaMode[c]]);
            f.maxVShift = std::max(f.maxVShift, kV[f.chromaMode[c]]);
        }
        for (uint32_t c = 0; c < 3; ++c) {
            f.hshift[c] = f.maxHShift - kH[f.chromaMode[c]];
            f.vshift[c] = f.maxVShift - kV[f.chromaMode[c]];
        }
    }
    if ((f.flags & FrameHeader::kUseDcFrame) == 0) {
        f.upsampling = br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}});
        f.ecUpsampling.resize(m.numExtraChannels);
        for (uint32_t i = 0; i < m.numExtraChannels; ++i) {
            // Coded relative to the channel's dim_shift (frame_header.cc).
            f.ecUpsampling[i] = br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}}) << m.extraChannels[i].dimShift;
        }
    }
    if (f.isModular) {
        f.groupSizeShift = br.read(2);
    }
    if (!f.isModular && f.colorTransform == 0) {
        f.xQmScale = br.read(3);
        f.bQmScale = br.read(3);
    } else {
        f.xQmScale = f.bQmScale = 2;
    }
    if (f.frameType != 2) {
        f.numPasses = br.readU32(U32Enc{{Val(1), Val(2), Val(3), BitsOffset(3, 4)}});
        if (f.numPasses > 11) return corrupt("too many passes");
        if (f.numPasses != 1) {
            f.numDownsample = br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(1, 3)}});
            if (f.numDownsample > 4 || f.numDownsample > f.numPasses) return corrupt("passes downsample count");
            for (uint32_t i = 0; i + 1 < f.numPasses; ++i) f.passShift[i] = br.read(2);
            for (uint32_t i = 0; i < f.numDownsample; ++i) f.downsample[i] = br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}});
            for (uint32_t i = 0; i < f.numDownsample; ++i) {
                f.lastPass[i] = br.readU32(U32Enc{{Val(0), Val(1), Val(2), Bits(3)}});
                if (f.lastPass[i] >= f.numPasses) return corrupt("passes last_pass");
            }
        }
    }
    if (f.frameType == 1) {
        f.dcLevel = br.readU32(U32Enc{{Val(1), Val(2), Val(3), Val(4)}});
    }
    bool isPartial = false;
    if (f.frameType != 1) {
        f.customSizeOrOrigin = br.readBool();
        if (f.customSizeOrOrigin) {
            const U32Enc enc{{Bits(8), BitsOffset(11, 256), BitsOffset(14, 2304), BitsOffset(30, 18688)}};
            if (f.frameType == 0 || f.frameType == 3) {
                const uint32_t ux0 = br.readU32(enc);
                const uint32_t uy0 = br.readU32(enc);
                f.x0 = static_cast<int32_t>((ux0 >> 1) ^ (0u - (ux0 & 1)));
                f.y0 = static_cast<int32_t>((uy0 >> 1) ^ (0u - (uy0 & 1)));
            }
            f.frameXsize = br.readU32(enc);
            f.frameYsize = br.readU32(enc);
            if (f.frameXsize == 0 || f.frameYsize == 0) return corrupt("zero frame size");
            if (f.frameType == 0 || f.frameType == 3) {
                isPartial = f.x0 > 0 || f.y0 > 0 || static_cast<int64_t>(f.frameXsize) + f.x0 < static_cast<int64_t>(m.xsize) ||
                            static_cast<int64_t>(f.frameYsize) + f.y0 < static_cast<int64_t>(m.ysize);
            }
        }
    }
    if (f.frameType == 0 || f.frameType == 3) {
        // BlendingInfo for the colour channels, then one per extra channel
        // (modes: 0 replace, 1 add, 2 blend, 3 alpha-weighted add, 4 multiply).
        const auto readBlending = [&]() -> Result<BlendingInfo> {
            BlendingInfo b;
            b.mode = br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(2, 3)}});
            if (b.mode > 4) return Error(ErrorCode::Corrupt, "blend mode");
            const bool withAlpha = m.numExtraChannels > 0 && (b.mode == 2 || b.mode == 3);
            if (withAlpha) {
                b.alphaChannel = br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(3, 3)}});
                if (b.alphaChannel >= m.numExtraChannels) return Error(ErrorCode::Corrupt, "blending alpha channel");
            }
            if (withAlpha || b.mode == 4) b.clamp = br.readBool();
            if (b.mode != 0 || isPartial) b.source = br.readU32(U32Enc{{Val(0), Val(1), Val(2), Val(3)}});
            return b;
        };
        SJXL_ASSIGN(f.blending, readBlending());
        f.blendMode = f.blending.mode;
        f.ecBlending.resize(m.numExtraChannels);
        for (BlendingInfo& b : f.ecBlending) { SJXL_ASSIGN(b, readBlending()); }
        if (m.haveAnimation) {
            f.duration = br.readU32(U32Enc{{Val(0), Val(1), Bits(8), Bits(32)}});
            if (m.haveTimecodes) (void)br.read(32);
        }
        f.isLast = br.readBool();
    } else {
        f.isLast = false;
    }
    if (f.frameType != 1 && !f.isLast) {
        f.saveAsReference = br.readU32(U32Enc{{Val(0), Val(1), Val(2), Val(3)}});
    }
    if (f.frameType != 1) {
        const bool canBeReferenced = !f.isLast && (f.frameType == 2 || f.saveAsReference != 0) ;
        if (canBeReferenced && f.blendMode == 0 && !isPartial && (f.frameType == 0 || f.frameType == 3)) {
            f.saveBeforeColorTransform = br.readBool();
        } else if (f.frameType == 2) {
            f.saveBeforeColorTransform = br.readBool();
        }
    }
    const uint32_t nameLength = br.readU32(U32Enc{{Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48)}});
    f.name.resize(nameLength);
    for (uint32_t i = 0; i < nameLength; ++i) f.name[i] = static_cast<char>(br.read(8));
    SJXL_TRY(readLoopFilter(br, f.loopFilter, f.isModular));
    SJXL_TRY(skipExtensions(br));
    return ok();
}

}   // namespace

uint32_t BitReader::peek(uint32_t n) const {
    if (n == 0) return 0;
    uint64_t window = 0;
    const size_t byte = pos_ >> 3;
    for (size_t i = 0; i < 8; ++i) {
        if (byte + i < bytes_.size()) window |= uint64_t{bytes_[byte + i]} << (8 * i);
    }
    window >>= (pos_ & 7);
    return static_cast<uint32_t>(window & ((uint64_t{1} << n) - 1));
}

uint32_t BitReader::readU32(const U32Enc& enc) {
    const Distr& d = enc.d[read(2)];
    if (d.direct) return d.valueOrOffset;
    return read(d.extraBits) + d.valueOrOffset;
}

uint64_t BitReader::readU64() {
    const uint32_t selector = read(2);
    if (selector == 0) return 0;
    if (selector == 1) return 1 + read(4);
    if (selector == 2) return 17 + read(8);
    uint64_t result = read(12);
    uint32_t shift = 12;
    while (read(1) != 0) {
        if (shift == 60) {
            result |= uint64_t{read(4)} << shift;
            break;
        }
        result |= uint64_t{read(8)} << shift;
        shift += 8;
    }
    return result;
}

Result<float> BitReader::readF16() {
    const uint32_t bits16 = read(16);
    const uint32_t sign = bits16 >> 15;
    const uint32_t biasedExp = (bits16 >> 10) & 0x1F;
    const uint32_t mantissa = bits16 & 0x3FF;
    if (biasedExp == 31) return corrupt("F16 infinity or NaN");
    float value;
    if (biasedExp == 0) {
        value = (1.0f / 16384) * (static_cast<float>(mantissa) * (1.0f / 1024));
        return sign ? -value : value;
    }
    const uint32_t bits32 = (sign << 31) | ((biasedExp + (127 - 15)) << 23) | (mantissa << 13);
    std::memcpy(&value, &bits32, sizeof value);
    return value;
}

namespace {

/// Section table of a frame from its sizes (codestream order) and permutation (logical ->
/// codestream index; identity when empty), starting at `firstByte`. Returns the end byte.
Result<size_t> buildSections(ParsedFrame& frame, const std::vector<uint32_t>& sizes, std::span<const uint32_t> permutation, size_t firstByte,
                             size_t codestreamSize) {
    const uint32_t n = static_cast<uint32_t>(sizes.size());
    std::vector<size_t> offsets(n);
    size_t offset = firstByte;
    for (uint32_t i = 0; i < n; ++i) {
        offsets[i] = offset;
        offset += sizes[i];
    }
    if (offset > codestreamSize) return Error(ErrorCode::Corrupt, "sections past the end of the codestream");
    frame.sections.resize(n);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t i = permutation.empty() ? j : permutation[j];
        if (i >= n) return Error(ErrorCode::Corrupt, "TOC permutation index");
        frame.sections[j].byteOffset = offsets[i];
        frame.sections[j].byteSize = sizes[i];
    }
    return offset;
}

/// The TOC sizes (byte aligned) at `bit`, then the sections. Returns the end byte.
Result<size_t> readTocSizes(Parsed& p, ParsedFrame& frame, size_t bit, std::span<const uint32_t> permutation) {
    BitReader fr(p.codestream, bit);
    fr.jumpToByteBoundary();
    std::vector<uint32_t> sizes(frame.numSections);
    for (uint32_t& sz : sizes) sz = fr.readU32(kTocDist);
    fr.jumpToByteBoundary();
    if (!fr.withinBounds()) return Error(ErrorCode::Corrupt, "truncated TOC");
    frame.tocEndBit = fr.bitPosition();
    return buildSections(frame, sizes, permutation, frame.tocEndBit / 8, p.codestream.size());
}

/// Frames from `startByte` on: DC frames and reference-only Modular frames (patch
/// dictionaries) may precede the displayed frame; each starts byte-aligned after the
/// previous one's sections. Stops early (p.incomplete) at a permuted TOC.
Result<void> parseFramesFrom(Parsed& p, size_t startByte) {
    const ImageMetadata& m = p.metadata;
    const auto unsupported = [](std::string what) { return Error(ErrorCode::Unsupported, std::move(what)); };
    const auto corrupt = [](std::string what) { return Error(ErrorCode::Corrupt, std::move(what)); };
    std::array<bool, 4> referenceSaved{};
    std::array<bool, 5> dcFrameSaved{};
    for (const ParsedFrame& earlier : p.frames) {
        if (earlier.header.frameType == 2) referenceSaved[earlier.header.saveAsReference] = true;
        if (earlier.header.frameType == 1) dcFrameSaved[earlier.header.dcLevel] = true;
    }
    size_t frameStartByte = startByte;
    for (;;) {
        if (p.frames.size() >= 65536) return unsupported("more than 65536 frames");
        if (frameStartByte >= p.codestream.size()) return corrupt("codestream ends before the last frame");
        BitReader fr(p.codestream, frameStartByte * 8);
        ParsedFrame frame;
        frame.frameHeaderBit = fr.bitPosition();
        SJXL_TRY(readFrameHeader(fr, m, frame.header));
        if (!fr.withinBounds()) return corrupt("truncated frame header");
        const FrameHeader& f = frame.header;
        const bool referenceOnly = f.frameType == 2;
        const bool dcFrame = f.frameType == 1;
        if (f.isModular) {
            if (f.colorTransform == 2) return unsupported("Modular frames in YCbCr");
            if (m.colorSpace != 0 && m.colorSpace != 1) return unsupported("Modular frames in a colour space other than RGB or grey");
        } else {
            if (f.colorTransform == 1) return unsupported("VarDCT frames in RGB without a colour transform");
            if (f.colorTransform == 2 && (f.loopFilter.gab || f.loopFilter.epfIters != 0)) return unsupported("loop filters on YCbCr frames");
            if (referenceOnly) return unsupported("VarDCT reference-only frames");
        }
        if (f.frameType == 3) return unsupported("skip-progressive frames");
        if (f.upsampling != 1 && f.upsampling != 2 && f.upsampling != 4 && f.upsampling != 8) return corrupt("upsampling factor");
        if (m.numExtraChannels > 4) return unsupported("more than 4 extra channels");
        for (uint32_t u : f.ecUpsampling) {
            if (u < f.upsampling || u > 8) return corrupt("extra channel upsampling");
        }
        if (f.customSizeOrOrigin && dcFrame) return corrupt("DC frame with a custom size");
        if (referenceOnly) {
            if (!f.saveBeforeColorTransform) return unsupported("reference frames saved after the colour transform");
            if (f.saveAsReference >= 4) return corrupt("reference slot");
            referenceSaved[f.saveAsReference] = true;
        } else if (dcFrame) {
            if (f.dcLevel < 1 || f.dcLevel > 4) return corrupt("DC frame level");
            dcFrameSaved[f.dcLevel] = true;
        } else if (!f.isLast && f.saveAsReference >= 4) {
            return corrupt("reference slot");
        }
        if ((f.flags & FrameHeader::kPatches) && !referenceSaved[0] && !referenceSaved[1] && !referenceSaved[2] && !referenceSaved[3]) {
            return corrupt("patches without a reference frame");
        }
        if ((f.flags & FrameHeader::kUseDcFrame) && !dcFrameSaved[f.dcLevel + 1]) return corrupt("kUseDcFrame without the DC frame");
        uint32_t fx = f.customSizeOrOrigin ? f.frameXsize : m.xsize;
        uint32_t fy = f.customSizeOrOrigin ? f.frameYsize : m.ysize;
        if (dcFrame) {
            fx = (fx + (1u << (3 * f.dcLevel)) - 1) >> (3 * f.dcLevel);
            fy = (fy + (1u << (3 * f.dcLevel)) - 1) >> (3 * f.dcLevel);
        }
        frame.outXsize = fx;
        frame.outYsize = fy;
        // FrameDimensions::Set: the frame is coded at 1 / upsampling of its size.
        fx = (fx + f.upsampling - 1) / f.upsampling;
        fy = (fy + f.upsampling - 1) / f.upsampling;
        frame.dim = frameDim(fx, fy, f.groupSizeShift, f.maxHShift, f.maxVShift);
        // toc.h NumTocEntries: one entry for one group and one pass; else DcGlobal,
        // DC groups, AcGlobal, then the AC groups of every pass.
        frame.singleSection = frame.dim.numGroups() == 1 && f.numPasses == 1;
        frame.numSections = frame.singleSection ? 1 : 2 + frame.dim.numDcGroups() + frame.dim.numGroups() * f.numPasses;
        // TOC
        if (fr.readBool()) {
            // Entropy coded permutation: a kernel decodes it, finishToc() resumes here.
            frame.tocPermuted = true;
            frame.tocPermutationBit = fr.bitPosition();
            p.frames.push_back(std::move(frame));
            p.frame = p.frames.back();
            p.incomplete = true;
            return ok();
        }
        SJXL_ASSIGN(const size_t end, readTocSizes(p, frame, fr.bitPosition(), {}));
        const bool last = f.isLast;
        p.frames.push_back(std::move(frame));
        if (last) break;
        frameStartByte = end;
    }
    p.frame = p.frames.back();
    p.incomplete = false;
    return ok();
}

}   // namespace

Result<void> finishIcc(Parsed& p, size_t endBit) {
    if (!p.iccPending) return Error(ErrorCode::InternalError, "finishIcc without a pending ICC");
    p.iccPending = false;
    p.incomplete = false;
    return parseFramesFrom(p, (endBit + 7) / 8);   // JumpToByteBoundary after the ICC
}

Result<void> finishToc(Parsed& p, std::span<const uint32_t> permutation, size_t endBit) {
    if (!p.incomplete || p.frames.empty()) return Error(ErrorCode::InternalError, "finishToc without a pending TOC");
    ParsedFrame& frame = p.frames.back();
    if (permutation.size() != frame.numSections) return Error(ErrorCode::Corrupt, "TOC permutation size");
    SJXL_ASSIGN(const size_t end, readTocSizes(p, frame, endBit, permutation));
    frame.tocPermuted = false;   // sections are final now
    const bool last = frame.header.isLast;
    p.incomplete = false;
    p.frame = frame;
    if (last) return ok();
    return parseFramesFrom(p, end);
}

Result<Parsed> parse(std::span<const uint8_t> file) {
    Parsed p;
    SJXL_ASSIGN(p.codestream, unwrapContainer(file));
    BitReader br(p.codestream);
    if (br.read(16) != 0x0AFF) return corrupt("bad signature");
    ImageMetadata& m = p.metadata;
    SJXL_TRY(readSizeHeader(br, m.xsize, m.ysize));
    SJXL_TRY(readImageMetadata(br, m));
    SJXL_TRY(readTransformData(br, m));
    if (!br.withinBounds()) return corrupt("truncated headers");

    // What M2 does not do yet.
    if (m.havePreview) return unsupported("preview frames");
    if (!m.floatingPointSample && m.bitsPerSample > 16) return unsupported("integer samples above 16 bits");
    for (const ExtraChannelInfo& ec : m.extraChannels) {
        if (ec.type == 4) return unsupported("CMYK (a Black extra channel: djxl converts it to RGB through the ICC profile)");
    }
    if (m.floatingPointSample && m.bitsPerSample != 32 && m.bitsPerSample != 16 && m.bitsPerSample != 24) return unsupported("float sample width");
    if (!m.tfHaveGamma && m.transferFunction != 13 && m.transferFunction != 8 && m.transferFunction != 17 && m.transferFunction != 1) {
        return unsupported("transfer functions other than sRGB, linear, gamma, DCI and 709 (PQ/HLG)");
    }

    if (m.wantIcc) {
        // The ICC profile follows the metadata, entropy coded: a kernel decodes it to find
        // the frames (icc_codec.cc ICCReader; the pixels do not depend on its bytes).
        p.iccPending = true;
        p.iccBit = br.bitPosition();
        p.incomplete = true;
        return p;
    }
    SJXL_TRY(parseFramesFrom(p, (br.bitPosition() + 7) / 8));
    return p;
}

}   // namespace sjxl::codestream
