// Copyright (c) 2026 jesus luque.
//
// The codestream syntax the CPU reads: container boxes, signature, SizeHeader,
// ImageMetadata, FrameHeader, TOC. Pure bookkeeping: the result is a section
// table and the parameters the kernels need. Everything entropy-coded is left
// to the device. Ports of libjxl headers.cc, image_metadata.cc,
// color_encoding_internal.cc, frame_header.cc, loop_filter.cc, toc.cc.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "sjxl/codestream/Headers.h"
#include "sjxl/core/Result.h"

namespace sjxl::codestream {

/// LSB-first reader over the codestream bytes, mirroring libjxl's BitReader.
class BitReader {
public:
    explicit BitReader(std::span<const uint8_t> bytes, size_t startBit = 0) : bytes_(bytes), pos_(startBit) {}
    [[nodiscard]] uint32_t peek(uint32_t n) const;
    void consume(uint32_t n) { pos_ += n; }
    [[nodiscard]] uint32_t read(uint32_t n) {
        const uint32_t v = peek(n);
        pos_ += n;
        return v;
    }
    [[nodiscard]] bool readBool() { return read(1) != 0; }
    [[nodiscard]] uint32_t readU32(const U32Enc& enc);
    [[nodiscard]] uint64_t readU64();
    [[nodiscard]] Result<float> readF16();
    [[nodiscard]] uint32_t readEnum() { return readU32(U32Enc{{Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18)}}); }
    void jumpToByteBoundary() { pos_ = (pos_ + 7) & ~size_t{7}; }
    [[nodiscard]] size_t bitPosition() const { return pos_; }
    [[nodiscard]] bool withinBounds() const { return pos_ <= bytes_.size() * 8; }
    [[nodiscard]] size_t totalBits() const { return bytes_.size() * 8; }

private:
    std::span<const uint8_t> bytes_;
    size_t                   pos_;
};

struct ImageMetadata {
    uint32_t xsize = 0, ysize = 0;
    uint32_t orientation = 1;
    bool     haveIntrinsicSize = false, havePreview = false, haveAnimation = false;
    bool     floatingPointSample = false;
    uint32_t bitsPerSample = 8, exponentBitsPerSample = 0;
    bool     modular16BitBufferSufficient = true;
    uint32_t numExtraChannels = 0;
    bool     xybEncoded = true;
    // ColorEncoding
    bool     wantIcc = false;
    uint32_t colorSpace = 0;          // 0 RGB, 1 grey, 2 XYB, 3 unknown
    uint32_t whitePoint = 1;          // 1 D65, 2 custom, 10 E, 11 DCI
    uint32_t primaries = 1;           // 1 sRGB, 2 custom, 9 2100, 11 P3
    bool     tfHaveGamma = false;
    uint32_t tfGamma = 0;             // in units of 1e-7
    uint32_t transferFunction = 13;   // 13 sRGB, 8 linear, 16 PQ, 18 HLG, ...
    uint32_t renderingIntent = 1;
    // ToneMapping
    float    intensityTarget = 255.0f;
    float    minNits = 0.0f;
    bool     relativeToMaxDisplay = false;
    float    linearBelow = 0.0f;
    // transform_data
    bool                 customOpsinMatrix = false;
    std::array<float, 9> opsinInverseMatrix{};   // row-major, libjxl default when not custom
    std::array<float, 3> opsinBiases{};          // kNegOpsinAbsorbanceBiasRGB
    std::array<float, 4> quantBiases{};          // kDefaultQuantBias
    uint32_t             customWeightsMask = 0;
    // Animation header
    uint32_t tpsNumerator = 100, tpsDenominator = 1, numLoops = 0;
    bool     haveTimecodes = false;
};

struct LoopFilter {
    bool     gab = true;
    bool     gabCustom = false;
    float    gabXWeight1 = 1.1f * 0.104699568f, gabXWeight2 = 1.1f * 0.055680538f;
    float    gabYWeight1 = 1.1f * 0.104699568f, gabYWeight2 = 1.1f * 0.055680538f;
    float    gabBWeight1 = 1.1f * 0.104699568f, gabBWeight2 = 1.1f * 0.055680538f;
    uint32_t epfIters = 2;
    bool     epfSharpCustom = false;
    std::array<float, 8> epfSharpLut{0, 1.f / 7, 2.f / 7, 3.f / 7, 4.f / 7, 5.f / 7, 6.f / 7, 1};
    bool     epfWeightCustom = false;
    std::array<float, 3> epfChannelScale{40.0f, 5.0f, 3.5f};
    float    epfPass1Zeroflush = 0.45f, epfPass2Zeroflush = 0.6f;
    bool     epfSigmaCustom = false;
    float    epfQuantMul = 0.46f, epfPass0SigmaScale = 0.9f, epfPass2SigmaScale = 6.5f;
    float    epfBorderSadMul = 2.0f / 3.0f;
    float    epfSigmaForModular = 1.0f;
};

struct FrameHeader {
    uint32_t frameType = 0;      // 0 regular, 1 DC, 2 reference only, 3 skip progressive
    bool     isModular = false;
    uint64_t flags = 0;
    uint32_t colorTransform = 0; // 0 XYB, 1 none, 2 YCbCr
    uint32_t upsampling = 1;
    std::vector<uint32_t> ecUpsampling;
    uint32_t groupSizeShift = 1;
    uint32_t xQmScale = 3, bQmScale = 2;
    uint32_t numPasses = 1;
    uint32_t dcLevel = 0;
    bool     customSizeOrOrigin = false;
    int32_t  x0 = 0, y0 = 0;
    uint32_t frameXsize = 0, frameYsize = 0;   // 0: the image size
    uint32_t blendMode = 0;
    bool     isLast = true;
    uint32_t saveAsReference = 0;
    bool     saveBeforeColorTransform = false;
    std::string name;
    LoopFilter loopFilter;
    // Flags (frame_header.h)
    static constexpr uint64_t kNoise = 1, kPatches = 2, kSplines = 16, kUseDcFrame = 32, kSkipAdaptiveDCSmoothing = 128;
};

struct Section {
    size_t   byteOffset = 0;   // in the codestream
    uint32_t byteSize = 0;
};

struct ParsedFrame {
    FrameHeader           header;
    FrameDim              dim;
    size_t                frameHeaderBit = 0;   // where the frame header starts
    size_t                tocEndBit = 0;        // first section starts here (byte aligned)
    std::vector<Section>  sections;             // in codestream order, after permutation
    bool                  singleSection = false;
};

struct Parsed {
    ImageMetadata            metadata;
    std::vector<uint8_t>     codestream;   // the raw codestream (boxes unwrapped)
    std::vector<ParsedFrame> frames;       // in codestream order; the last one is displayed
    ParsedFrame              frame;        // the displayed (last) frame, a copy of frames.back()
};

/// Unwraps the container if present (jxlc / jxlp boxes), parses the headers
/// and every frame's header and TOC (reference-only frames for patches, then
/// the displayed frame). Features this decoder does not implement are
/// reported as Unsupported, naming them.
[[nodiscard]] Result<Parsed> parse(std::span<const uint8_t> file);

}   // namespace sjxl::codestream
