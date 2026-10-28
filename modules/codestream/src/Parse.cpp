// Copyright (c) 2026 jesus luque.
#include "sjxl/codestream/Parse.h"

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
    for (uint32_t i = 0; i < m.numExtraChannels; ++i) {
        // ExtraChannelInfo
        if (br.readBool()) continue;   // all_default: 8-bit alpha
        const uint32_t type = br.readEnum();
        ImageMetadata depth;
        SJXL_TRY(readBitDepth(br, depth));
        const uint32_t dimShift = br.readU32(U32Enc{{Val(0), Val(3), Val(4), BitsOffset(3, 1)}});
        (void)dimShift;
        const uint32_t nameLength = br.readU32(U32Enc{{Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48)}});
        for (uint32_t k = 0; k < nameLength; ++k) (void)br.read(8);
        if (type == 0) (void)br.readBool();            // alpha_associated
        if (type == 2) for (int k = 0; k < 4; ++k) { SJXL_TRY(br.readF16()); }   // spot color
        if (type == 5) (void)br.readU32(U32Enc{{Val(1), Bits(2), BitsOffset(4, 3), BitsOffset(8, 19)}});   // CFA
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
    if (m.customWeightsMask & 1) for (int i = 0; i < 15; ++i) SJXL_TRY(br.readF16());
    if (m.customWeightsMask & 2) for (int i = 0; i < 55; ++i) SJXL_TRY(br.readF16());
    if (m.customWeightsMask & 4) for (int i = 0; i < 210; ++i) SJXL_TRY(br.readF16());
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
        // YCbCrChromaSubsampling: 3 x Bits(2)
        for (int c = 0; c < 3; ++c) (void)br.read(2);
    }
    if ((f.flags & FrameHeader::kUseDcFrame) == 0) {
        f.upsampling = br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}});
        f.ecUpsampling.resize(m.numExtraChannels);
        for (uint32_t& u : f.ecUpsampling) u = br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}});
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
        if (f.numPasses != 1) {
            const uint32_t numDownsample = br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(1, 3)}});
            for (uint32_t i = 0; i + 1 < f.numPasses; ++i) (void)br.read(2);
            for (uint32_t i = 0; i < numDownsample; ++i) (void)br.readU32(U32Enc{{Val(1), Val(2), Val(4), Val(8)}});
            for (uint32_t i = 0; i < numDownsample; ++i) (void)br.readU32(U32Enc{{Val(0), Val(1), Val(2), Bits(3)}});
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
        // BlendingInfo for the colour channels, then one per extra channel.
        const auto readBlending = [&](bool isColor) -> uint32_t {
            const uint32_t mode = br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(2, 3)}});
            const bool withAlpha = m.numExtraChannels > 0 && (mode == 2 || mode == 4);
            if (withAlpha) (void)br.readU32(U32Enc{{Val(0), Val(1), Val(2), BitsOffset(3, 3)}});
            if (withAlpha || mode == 3) (void)br.readBool();   // clamp
            if (mode != 0 || isPartial) (void)br.readU32(U32Enc{{Val(0), Val(1), Val(2), Val(3)}});   // source
            (void)isColor;
            return mode;
        };
        f.blendMode = readBlending(true);
        for (uint32_t i = 0; i < m.numExtraChannels; ++i) (void)readBlending(false);
        if (m.haveAnimation) {
            (void)br.readU32(U32Enc{{Val(0), Val(1), Bits(8), Bits(32)}});   // duration
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
    if (m.wantIcc) return unsupported("ICC colour profiles");
    if (m.numExtraChannels != 0) return unsupported("extra channels (alpha, depth, ...)");
    if (m.havePreview) return unsupported("preview frames");
    if (m.haveAnimation) return unsupported("animation");
    if (m.orientation != 1) return unsupported("orientation other than identity");
    if (m.bitsPerSample > 16 || m.floatingPointSample) return unsupported("floating point or >16-bit samples");
    if (m.customWeightsMask != 0) return unsupported("custom upsampling weights");
    if (!m.tfHaveGamma && m.transferFunction != 13 && m.transferFunction != 8 && m.transferFunction != 17) {
        return unsupported("transfer functions other than sRGB, linear, gamma and DCI (PQ/HLG/709)");
    }

    // Frames: reference-only Modular frames (patch dictionaries) may precede the
    // displayed frame; each starts byte-aligned after the previous one's sections.
    std::array<bool, 4> referenceSaved{};
    size_t frameStartByte = (br.bitPosition() + 7) / 8;
    for (uint32_t frameIndex = 0;; ++frameIndex) {
        if (frameIndex >= 16) return unsupported("more than 16 frames");
        if (frameStartByte >= p.codestream.size()) return corrupt("codestream ends before the last frame");
        BitReader fr(p.codestream, frameStartByte * 8);
        ParsedFrame frame;
        frame.frameHeaderBit = fr.bitPosition();
        SJXL_TRY(readFrameHeader(fr, m, frame.header));
        if (!fr.withinBounds()) return corrupt("truncated frame header");
        const FrameHeader& f = frame.header;
        const bool referenceOnly = f.frameType == 2;
        if (f.isModular) {
            if (f.frameType != 0 && !referenceOnly) return unsupported(std::format("Modular frames of type {}", f.frameType));
            if (f.colorTransform == 2) return unsupported("Modular frames in YCbCr");
            if (m.colorSpace != 0 && m.colorSpace != 1) return unsupported("Modular frames in a colour space other than RGB or grey");
            if (m.floatingPointSample) return unsupported("floating point samples");
        } else {
            if (!m.xybEncoded || f.colorTransform != 0) return unsupported("VarDCT frames not encoded in XYB");
            if (referenceOnly) return unsupported("VarDCT reference-only frames");
        }
        if (f.frameType == 1 || f.frameType == 3) return unsupported("DC or skip-progressive frames");
        if (f.flags & FrameHeader::kNoise) return unsupported("noise synthesis");
        if (f.flags & FrameHeader::kSplines) return unsupported("splines");
        if (f.flags & FrameHeader::kUseDcFrame) return unsupported("DC frames");
        if (f.upsampling != 1) return unsupported("upsampling");
        if (f.numPasses != 1) return unsupported("progressive passes");
        if (f.customSizeOrOrigin && !referenceOnly) return unsupported("frames with a custom size or origin");
        if (referenceOnly) {
            if (!f.saveBeforeColorTransform) return unsupported("reference frames saved after the colour transform");
            if (f.saveAsReference >= 4) return corrupt("reference slot");
            referenceSaved[f.saveAsReference] = true;
        } else if (!f.isLast) {
            return unsupported("multiple displayed frames (animation or blending)");
        }
        if ((f.flags & FrameHeader::kPatches) && !referenceSaved[0] && !referenceSaved[1] && !referenceSaved[2] && !referenceSaved[3]) {
            return corrupt("patches without a reference frame");
        }
        const uint32_t fx = f.customSizeOrOrigin ? f.frameXsize : m.xsize;
        const uint32_t fy = f.customSizeOrOrigin ? f.frameYsize : m.ysize;
        frame.dim = frameDim(fx, fy, f.groupSizeShift);
        const uint32_t numEntries = frame.dim.singleSection() ? 1 : frame.dim.numSections();
        frame.singleSection = frame.dim.singleSection();
        // TOC
        if (fr.readBool()) {
            return unsupported("TOC permutation (entropy coded; decoded on the device in a later milestone)");
        }
        fr.jumpToByteBoundary();
        std::vector<uint32_t> sizes(numEntries);
        for (uint32_t& sz : sizes) sz = fr.readU32(kTocDist);
        fr.jumpToByteBoundary();
        if (!fr.withinBounds()) return corrupt("truncated TOC");
        frame.tocEndBit = fr.bitPosition();
        size_t offset = frame.tocEndBit / 8;
        frame.sections.resize(numEntries);
        for (uint32_t i = 0; i < numEntries; ++i) {
            frame.sections[i].byteOffset = offset;
            frame.sections[i].byteSize = sizes[i];
            offset += sizes[i];
        }
        if (offset > p.codestream.size()) return corrupt("sections past the end of the codestream");
        p.frames.push_back(std::move(frame));
        if (!referenceOnly) break;
        frameStartByte = offset;
    }
    p.frame = p.frames.back();
    return p;
}

}   // namespace sjxl::codestream
