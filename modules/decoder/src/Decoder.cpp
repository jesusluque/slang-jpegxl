// Copyright (c) 2026 jesus luque.
#include "sjxl/decoder/Decoder.h"

#include "sjxl/decoder/DcGroupCpu.h"

#include "sjxl/image/Image.h"

#include "sjxl/codestream/UpsamplingWeights.h"

#include <cmath>
#include <cstring>
#include <format>
#include <fstream>

#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include <algorithm>
#include <bit>
#include <cstdlib>
#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/StageTimer.h"

namespace sjxl::dec {
namespace {

constexpr uint32_t kGsHeaderSize = 64;
constexpr uint32_t kGlobalsWords = 4u << 20;      // 16 MB: AC code sets, orders, matrices encodings
constexpr uint32_t kDcGroupWords = 1u << 20;      // 4 MB per DC group
constexpr uint32_t kAcGroupWords = 1u << 13;        // per-channel non-zero grids (3 x 32 x 32 words) and reader state
constexpr uint32_t kModularGroupWords = 1u << 19;   // a group's local trees, histograms and predictor state
constexpr float    kInvSigmaNum = -1.1715728752538099024f;
constexpr uint32_t kMaxPatchPositions = 1u << 16;     // frame.slangh kMaxPatchPositions
constexpr uint32_t kLz77WindowWords = 1u << 20;
constexpr uint32_t kDequantTableFloats = 394752;
constexpr uint32_t kGsTreeNumNodesIndex = 33;
constexpr uint32_t kGsGlobalTreeBitIndex = 54;   // frame.slangh kGsGlobalTreeBit
constexpr uint32_t kErrNone = 0, kErrCorrupt = 1, kErrUnsupported = 2, kErrArena = 3, kErrLz77Window = 4, kErrAcNzeros = 5, kErrAcLeftover = 6, kErrAnsState = 7;

struct Uint2 {
    uint32_t x, y;
};

const char* describe(uint32_t code) {
    switch (code) {
    case kErrCorrupt: return "the codestream violates the specification";
    case kErrUnsupported: return "a feature this decoder does not implement (modular Squeeze, RAW quant tables)";
    case kErrArena: return "the decoder ran out of scratch memory";
    case kErrLz77Window: return "LZ77 window";
    case kErrAcNzeros: return "an AC block's non-zero count exceeds its coefficients";
    case kErrAcLeftover: return "an AC block ended with non-zero coefficients outstanding";
    case kErrAnsState: return "an ANS stream's final state is not the signature";
    default: return "unknown";
    }
}

}   // namespace

Result<std::unique_ptr<Decoder>> Decoder::create(std::shared_ptr<gpu::ShaderLibrary> library) {
    std::unique_ptr<Decoder> dec(new Decoder());
    dec->library_ = std::move(library);
    dec->device_ = &dec->library_->device();
    gpu::ShaderLibrary& lib = *dec->library_;
    SJXL_ASSIGN(dec->globals_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeGlobals"));
    SJXL_ASSIGN(dec->dcGroups_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeDcGroups"));
    SJXL_ASSIGN(dec->acGroups_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeAcGroups"));
    SJXL_ASSIGN(dec->tables_, gpu::ComputeKernel::create(lib, "sjxl/dec/tables", "computeDequantTable"));
    SJXL_ASSIGN(dec->smoothDc_, gpu::ComputeKernel::create(lib, "sjxl/dec/dc_finalize", "smoothDc"));
    SJXL_ASSIGN(dec->computeSigma_, gpu::ComputeKernel::create(lib, "sjxl/dec/dc_finalize", "computeSigma"));
    SJXL_ASSIGN(dec->dequantCoeffs_, gpu::ComputeKernel::create(lib, "sjxl/dec/transforms", "dequantCoeffs"));
    SJXL_ASSIGN(dec->llfFromDc_, gpu::ComputeKernel::create(lib, "sjxl/dec/transforms", "llfFromDc"));
    SJXL_ASSIGN(dec->idctRows_, gpu::ComputeKernel::create(lib, "sjxl/dec/transforms", "idctRows"));
    SJXL_ASSIGN(dec->idctCols_, gpu::ComputeKernel::create(lib, "sjxl/dec/transforms", "idctCols"));
    SJXL_ASSIGN(dec->transformSpecial_, gpu::ComputeKernel::create(lib, "sjxl/dec/transforms", "inverseTransformSpecial"));
    SJXL_ASSIGN(dec->modularGlobals_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeModularGlobals"));
    SJXL_ASSIGN(dec->modularDcGroups_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeModularDcGroups"));
    SJXL_ASSIGN(dec->modularAcGroups_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "decodeModularAcGroups"));
    SJXL_ASSIGN(dec->modularFinalize_, gpu::ComputeKernel::create(lib, "sjxl/dec/sections", "modularFinalizeFrame"));
    SJXL_ASSIGN(dec->modularToPlanes_, gpu::ComputeKernel::create(lib, "sjxl/dec/modular_out", "modularToPlanes"));
    SJXL_ASSIGN(dec->ecToPlanes_, gpu::ComputeKernel::create(lib, "sjxl/dec/modular_out", "ecToPlanes"));
    SJXL_ASSIGN(dec->tocPermutation_, gpu::ComputeKernel::create(lib, "sjxl/dec/toc", "decodeTocPermutation"));
    SJXL_ASSIGN(dec->icc_, gpu::ComputeKernel::create(lib, "sjxl/dec/toc", "decodeIcc"));
    SJXL_ASSIGN(dec->toOutputPlanes_, gpu::ComputeKernel::create(lib, "sjxl/dec/output", "toOutputPlanes"));
    SJXL_ASSIGN(dec->blendCanvas_, gpu::ComputeKernel::create(lib, "sjxl/dec/blend", "blendCanvas"));
    SJXL_ASSIGN(dec->upsample_, gpu::ComputeKernel::create(lib, "sjxl/dec/upsample", "upsample"));
    SJXL_ASSIGN(dec->chromaUpsample_, gpu::ComputeKernel::create(lib, "sjxl/dec/filters", "chromaUpsample"));
    SJXL_ASSIGN(dec->noiseRandom_, gpu::ComputeKernel::create(lib, "sjxl/dec/noise", "noiseRandom"));
    SJXL_ASSIGN(dec->noiseConvolve_, gpu::ComputeKernel::create(lib, "sjxl/dec/noise", "noiseConvolve"));
    SJXL_ASSIGN(dec->noiseAdd_, gpu::ComputeKernel::create(lib, "sjxl/dec/noise", "noiseAdd"));
    SJXL_ASSIGN(dec->dcFromFrame_, gpu::ComputeKernel::create(lib, "sjxl/dec/dc_finalize", "dcFromFrame"));
    SJXL_ASSIGN(dec->bucketPatches_, gpu::ComputeKernel::create(lib, "sjxl/dec/patches", "bucketPatches"));
    SJXL_ASSIGN(dec->applyPatches_, gpu::ComputeKernel::create(lib, "sjxl/dec/patches", "applyPatches"));
    SJXL_ASSIGN(dec->splineSegments_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineSegments"));
    SJXL_ASSIGN(dec->splineOffsets_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineOffsets"));
    SJXL_ASSIGN(dec->splineRowCounts_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineRowCounts"));
    SJXL_ASSIGN(dec->splineRowStarts_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineRowStarts"));
    SJXL_ASSIGN(dec->splineScatter_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineScatter"));
    SJXL_ASSIGN(dec->splineDraw_, gpu::ComputeKernel::create(lib, "sjxl/dec/splines", "splineDraw"));
    SJXL_ASSIGN(dec->gaborish_, gpu::ComputeKernel::create(lib, "sjxl/dec/filters", "gaborish"));
    SJXL_ASSIGN(dec->epf_, gpu::ComputeKernel::create(lib, "sjxl/dec/filters", "epf"));
    SJXL_ASSIGN(dec->output_, gpu::ComputeKernel::create(lib, "sjxl/dec/output", "xybToSrgb8"));
    return dec;
}

Result<codestream::Parsed> Decoder::parseAll(std::span<const uint8_t> file) {
    SJXL_ASSIGN(codestream::Parsed parsed, codestream::parse(file));
    if (parsed.iccPending) {
        SJXL_ASSIGN(const size_t end, decodeIccEnd(parsed));
        SJXL_TRY(codestream::finishIcc(parsed, end));
    }
    while (parsed.incomplete) {
        SJXL_ASSIGN(const TocPermutation perm, decodeTocPermutation(parsed, parsed.frames.back()));
        SJXL_TRY(codestream::finishToc(parsed, perm.order, perm.endBit));
    }
    return parsed;
}

Result<size_t> Decoder::decodeIccEnd(const codestream::Parsed& parsed) {
    gpu::Device& dev = *device_;
    std::vector<uint32_t> csWords((parsed.codestream.size() + 3) / 4 + 2, 0);
    std::memcpy(csWords.data(), parsed.codestream.data(), parsed.codestream.size());
    SJXL_ASSIGN(auto cs, gpu::Buffer::fromSpan<uint32_t>(dev, csWords, "dec.icc.cs"));
    constexpr uint32_t kIccArenaWords = (1u << 20) + kLz77WindowWords;
    SJXL_ASSIGN(auto arena, gpu::Buffer::ofType<uint32_t>(dev, kIccArenaWords, "dec.icc.arena"));
    SJXL_ASSIGN(auto out, gpu::Buffer::ofType<uint32_t>(dev, 4, "dec.icc.out"));
    gpu::CommandBatch batch(dev);
    icc_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        c["cs"].setBinding(cs.rhi());
        c["arena"].setBinding(arena.rhi());
        c["tocOut"].setBinding(out.rhi());
        c["params"]["tocBit"].setData(static_cast<uint32_t>(parsed.iccBit));
        c["params"]["tocEntries"].setData(0u);
        c["params"]["arenaCap"].setData(kIccArenaWords);
    });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> words, out.readAll<uint32_t>(dev));
    if (words[0] != kErrNone) return Error::make(ErrorCode::Corrupt, "{} in the ICC profile", describe(words[0]));
    log::debug("decoder: ICC profile of {} encoded bytes, frames start at bit {}", words[2], words[1]);
    return size_t{words[1]};
}

Result<Decoder::TocPermutation> Decoder::decodeTocPermutation(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame) {
    gpu::Device& dev = *device_;
    std::vector<uint32_t> csWords((parsed.codestream.size() + 3) / 4 + 2, 0);
    std::memcpy(csWords.data(), parsed.codestream.data(), parsed.codestream.size());
    SJXL_ASSIGN(auto cs, gpu::Buffer::fromSpan<uint32_t>(dev, csWords, "dec.toc.cs"));
    constexpr uint32_t kTocArenaWords = (1u << 20) + kLz77WindowWords;
    SJXL_ASSIGN(auto arena, gpu::Buffer::ofType<uint32_t>(dev, kTocArenaWords, "dec.toc.arena"));
    SJXL_ASSIGN(auto out, gpu::Buffer::ofType<uint32_t>(dev, 2 + frame.numSections, "dec.toc.out"));
    gpu::CommandBatch batch(dev);
    tocPermutation_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        c["cs"].setBinding(cs.rhi());
        c["arena"].setBinding(arena.rhi());
        c["tocOut"].setBinding(out.rhi());
        c["params"]["tocBit"].setData(static_cast<uint32_t>(frame.tocPermutationBit));
        c["params"]["tocEntries"].setData(frame.numSections);
        c["params"]["arenaCap"].setData(kTocArenaWords);
    });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> words, out.readAll<uint32_t>(dev));
    if (words[0] != kErrNone) return Error::make(ErrorCode::Corrupt, "{} in the TOC permutation", describe(words[0]));
    TocPermutation perm;
    perm.endBit = words[1];
    perm.order.assign(words.begin() + 2, words.end());
    return perm;
}

Result<std::vector<Frame>> Decoder::decodeAll(std::span<const uint8_t> file) {
    SJXL_ASSIGN(const codestream::Parsed parsed, parseAll(file));
    std::array<RefFrame, 4> refs;
    std::array<RefFrame, 5> dcFrames;
    std::vector<Frame> frames;
    FrameIndices indices;
    for (const codestream::ParsedFrame& frame : parsed.frames) {
        // dec_frame.cc: a displayed frame bumps the visible index and resets the other.
        const bool displayed = frame.header.frameType == 0 || frame.header.frameType == 3;
        if (displayed && (frame.header.isLast || frame.header.duration > 0)) { ++indices.visible; indices.nonvisible = 0; }
        else ++indices.nonvisible;
        auto result = run(parsed, frame, false, refs, dcFrames, indices);
        if (!result && result.error().code() == ErrorCode::Unsupported && result.error().message() == "LZ77 window") {
            result = run(parsed, frame, true, refs, dcFrames, indices);
        }
        if (!result) return result.error();
        if (result->emitted) frames.push_back(Frame{std::move(result->image), frame.header.duration});
    }
    if (frames.empty()) return Error(ErrorCode::Corrupt, "no displayed frame");
    return frames;
}

Result<image::Rgb8> Decoder::decode(std::span<const uint8_t> file) {
    SJXL_ASSIGN(std::vector<Frame> frames, decodeAll(file));
    return std::move(frames.front().image);
}

Result<Decoder::FrameOutput> Decoder::run(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame, bool withLz77Windows,
                                 std::array<RefFrame, 4>& refs, std::array<RefFrame, 5>& dcFrames, FrameIndices indices) {
    gpu::Device& dev = *device_;
    const codestream::ImageMetadata& m = parsed.metadata;
    const bool referenceOnly = frame.header.frameType == 2;
    const bool dcFrame = frame.header.frameType == 1;
    const bool useDcFrame = (frame.header.flags & codestream::FrameHeader::kUseDcFrame) != 0;
    const RefFrame* dcSource = useDcFrame ? &dcFrames[frame.header.dcLevel + 1] : nullptr;
    if (useDcFrame && (!dcSource->planes || dcSource->width != frame.dim.xsizeBlocks || dcSource->height != frame.dim.ysizeBlocks)) {
        return Error(ErrorCode::Corrupt, "the DC frame does not match the frame's block grid");
    }
    const codestream::FrameDim& dim = frame.dim;
    const codestream::LoopFilter& lf = frame.header.loopFilter;
    const uint32_t nblocks = dim.numBlocks();
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const uint32_t numThreads = 1 + numDcGroups + numGroups;
    const uint32_t tilesX = (dim.xsizeBlocks + 7) / 8;
    const uint32_t tilesY = (dim.ysizeBlocks + 7) / 8;
    const bool modular = frame.header.isModular;
    const bool modularXyb = modular && frame.header.colorTransform == 0;
    const uint32_t colorChannels = (modular && m.colorSpace == 1 && frame.header.colorTransform == 1) ? 1u : 3u;

    // The codestream and the section table (bit positions).
    std::vector<uint32_t> csWords((parsed.codestream.size() + 3) / 4 + 2, 0);
    std::memcpy(csWords.data(), parsed.codestream.data(), parsed.codestream.size());
    SJXL_ASSIGN(auto cs, gpu::Buffer::fromSpan<uint32_t>(dev, csWords, "dec.cs"));
    std::vector<Uint2> sections;
    for (const codestream::Section& s : frame.sections) {
        sections.push_back({static_cast<uint32_t>(s.byteOffset * 8), static_cast<uint32_t>((s.byteOffset + s.byteSize) * 8)});
    }
    SJXL_ASSIGN(auto sectionBuf, gpu::Buffer::fromSpan<Uint2>(dev, sections, "dec.sections"));

    // Arena partition. A modular frame keeps its full-image channels in the globals
    // region, decodes groups into them through views, and expands palettes at the
    // finalise step in the AC groups' region.
    uint32_t globalsWords = kGlobalsWords, dcGroupWords = kDcGroupWords, acGroupWords = kAcGroupWords;
    if (!modular && frame.header.numPasses > 1) globalsWords += (frame.header.numPasses - 1) * 6156u * 64u;   // one order table per pass
    if (!modular && m.numExtraChannels > 0) {   // the AC and DC groups also decode the extra channels' modular groups
        acGroupWords = std::max(acGroupWords, kModularGroupWords);
        dcGroupWords = std::max(dcGroupWords, kModularGroupWords);
        globalsWords += static_cast<uint32_t>((m.numExtraChannels + 1) * uint64_t{frame.outXsize} * frame.outYsize);
    }
    if (modular) {
        const uint64_t pixels = uint64_t{dim.xsize} * dim.ysize;
        const uint32_t channels = colorChannels + m.numExtraChannels;   // the extra channels are channels of the same modular image
        if (pixels * (channels + 2) > (uint64_t{1} << 31)) return Error(ErrorCode::Unsupported, "modular frames above 512 Mpixel");
        globalsWords = kGlobalsWords + static_cast<uint32_t>((channels + 1) * pixels);
        dcGroupWords = kModularGroupWords;
        acGroupWords = kModularGroupWords + (channels + 1) * dim.groupDim * dim.groupDim;   // the group's trees, histograms and tables, then its channels
        if (uint64_t{acGroupWords} * numGroups < (channels + 1) * pixels) {
            acGroupWords = static_cast<uint32_t>(((channels + 1) * pixels + numGroups - 1) / numGroups);
        }
    }
    const uint32_t dcArenaBase = globalsWords;
    const uint32_t acArenaBase = dcArenaBase + numDcGroups * dcGroupWords;
    const uint32_t windowBase = acArenaBase + numGroups * acGroupWords;
    const uint32_t windowCount = withLz77Windows ? numThreads : 0;
    const uint64_t arenaWords = uint64_t{windowBase} + uint64_t{windowCount} * kLz77WindowWords;
    SJXL_ASSIGN(auto arena, gpu::Buffer::ofType<uint32_t>(dev, arenaWords, "dec.arena"));
    SJXL_ASSIGN(auto status, gpu::Buffer::ofType<uint32_t>(dev, numThreads, "dec.status"));

    SJXL_ASSIGN(auto dc, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks, "dec.dc"));
    SJXL_ASSIGN(auto dcSmooth, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks, "dec.dcSmooth"));
    SJXL_ASSIGN(auto quantDcBucket, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "dec.quantDcBucket"));
    std::vector<uint32_t> invalid(nblocks, 0xFFu);
    SJXL_ASSIGN(auto acs, gpu::Buffer::fromSpan<uint32_t>(dev, invalid, "dec.acs"));
    SJXL_ASSIGN(auto qf, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "dec.qf"));
    SJXL_ASSIGN(auto epfSharp, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "dec.epfSharp"));
    SJXL_ASSIGN(auto firstBlock, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "dec.firstBlock"));
    SJXL_ASSIGN(auto ytox, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "dec.ytox"));
    SJXL_ASSIGN(auto ytob, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "dec.ytob"));
    SJXL_ASSIGN(auto coeffs, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks * 64, "dec.coeffs"));
    SJXL_ASSIGN(auto dequantF, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks * 64, "dec.dequantF"));
    SJXL_ASSIGN(auto dequant, gpu::Buffer::ofType<float>(dev, kDequantTableFloats, "dec.dequant"));
    SJXL_ASSIGN(auto sigma, gpu::Buffer::ofType<float>(dev, nblocks, "dec.sigma"));
    const uint64_t planePixels = uint64_t{dim.xsizePadded} * dim.ysizePadded;
    SJXL_ASSIGN(auto xybA, gpu::Buffer::ofType<float>(dev, 3 * planePixels, "dec.xybA"));
    SJXL_ASSIGN(auto xybB, gpu::Buffer::ofType<float>(dev, 3 * planePixels, "dec.xybB"));
    // After upsampling the frame is outWidth x outHeight; the later stages run at that size.
    const uint32_t upsampling = frame.header.upsampling;
    const uint32_t upShift = upsampling == 8 ? 3 : upsampling == 4 ? 2 : upsampling == 2 ? 1 : 0;
    const uint32_t outWidth = frame.outXsize, outHeight = frame.outYsize;
    const uint32_t outPaddedWidth = dim.xsizePadded * upsampling, outPaddedHeight = dim.ysizePadded * upsampling;
    // Extra channels: modular in every frame, at ceil(output size / ec_upsampling).
    const uint32_t numEc = m.numExtraChannels;
    std::array<uint32_t, 4> ecW{}, ecH{}, ecShift{}, ecUps{};
    uint32_t alphaIndex = numEc;   // first alpha channel, numEc when none
    for (uint32_t e = 0; e < numEc; ++e) {
        ecUps[e] = frame.header.ecUpsampling[e];
        ecW[e] = (outWidth + ecUps[e] - 1) / ecUps[e];
        ecH[e] = (outHeight + ecUps[e] - 1) / ecUps[e];
        const uint32_t ecLog = ecUps[e] == 8 ? 3 : ecUps[e] == 4 ? 2 : ecUps[e] == 2 ? 1 : 0;
        ecShift[e] = ecLog - upShift;
        if (alphaIndex == numEc && m.extraChannels[e].type == 0) alphaIndex = e;
    }
    const bool greyOut = m.colorSpace == 1;   // kGray: one colour channel (djxl writes grey PNGs)
    const uint32_t outChannels = (greyOut ? 1u : 3u) + (alphaIndex < numEc ? 1u : 0u);
    const bool orientationSwaps = m.orientation >= 5;
    // YCbCr chroma subsampling (JPEG recompression): per-channel shifts, zero for XYB.
    const bool ycbcr = !modular && frame.header.colorTransform == 2;
    std::array<uint32_t, 4> hshift{}, vshift{};
    if (ycbcr) {
        for (size_t c = 0; c < 3; ++c) { hshift[c] = frame.header.hshift[c]; vshift[c] = frame.header.vshift[c]; }
    }
    const uint32_t outDepth = m.bitsPerSample > 8 ? 16u : 8u;   // djxl writes 16-bit files for deeper images (floats included)
    // Displayed frames compose onto a canvas of the image's size (frames may be partial).
    const bool displayed = frame.header.frameType == 0 || frame.header.frameType == 3;
    const uint32_t canvasW = m.xsize, canvasH = m.ysize;
    const uint32_t outBytes = canvasW * canvasH * outChannels * (outDepth / 8);
    const codestream::BlendingInfo& blending = frame.header.blending;
    const RefFrame* bgCanvas = nullptr;
    if (displayed) {
        const RefFrame& src0 = refs[blending.source];
        const bool full = !frame.header.customSizeOrOrigin && outWidth == canvasW && outHeight == canvasH;
        bool needsBg = blending.mode != 0 || !full;
        for (const codestream::BlendingInfo& eb : frame.header.ecBlending) needsBg = needsBg || eb.mode != 0;
        if (needsBg && src0.planes && !src0.inXyb) bgCanvas = &src0;
        if (needsBg && src0.planes && src0.inXyb) return Error(ErrorCode::Corrupt, "blending onto a reference frame saved before the colour transform");
    }
    SJXL_ASSIGN(auto rgb8, gpu::Buffer::ofType<uint32_t>(dev, (outBytes + 3) / 4, "dec.rgb8"));
    SJXL_ASSIGN(auto dummy, gpu::Buffer::ofType<float>(dev, 4, "dec.dummy"));   // stands in for absent optional inputs
    std::optional<gpu::Buffer> frameOut, canvas;   // displayed frames: output-encoding planes, then the composited canvas
    if (displayed) {
        SJXL_ASSIGN(frameOut, gpu::Buffer::ofType<float>(dev, uint64_t{3} * outWidth * outHeight, "dec.frameOut"));
        SJXL_ASSIGN(canvas, gpu::Buffer::ofType<float>(dev, uint64_t{3 + numEc} * canvasW * canvasH, "dec.canvas"));
    }
    std::optional<gpu::Buffer> weightsKeep, ecWeightsKeep;   // upsampling weights live until the batch ran

    gpu::CommandBatch batch(dev);
    gpu::StageTimer timer(batch, "decode");

    // The DC groups' modular streams decoded on the CPU (DcGroupCpu.h): set after the
    // globals ran, bound to every sections kernel (the DC group kernel copies them).
    bool dcFromHost = false;
    std::optional<gpu::Buffer> hostDc, hostDcBase;

    const auto bindSections = [&](rhi::ShaderCursor c) {
        c["params"]["dcFromHost"].setData(uint32_t{dcFromHost ? 1u : 0u});
        c["hostDc"].setBinding(hostDc ? hostDc->rhi() : dummy.rhi());
        c["hostDcBase"].setBinding(hostDcBase ? hostDcBase->rhi() : dummy.rhi());
        c["params"]["wideSamples"].setData(uint32_t{m.floatingPointSample || m.bitsPerSample > 16 ? 1u : 0u});
        c["params"]["perfSkip"].setData(static_cast<uint32_t>(std::strtoul(platform::env("SJXL_PERF_SKIP").c_str(), nullptr, 0)));
        c["cs"].setBinding(cs.rhi());
        c["arena"].setBinding(arena.rhi());
        c["dc"].setBinding(dc.rhi());
        c["quantDcBucket"].setBinding(quantDcBucket.rhi());
        c["acs"].setBinding(acs.rhi());
        c["qf"].setBinding(qf.rhi());
        c["epfSharp"].setBinding(epfSharp.rhi());
        c["ytox"].setBinding(ytox.rhi());
        c["ytob"].setBinding(ytob.rhi());
        c["firstBlock"].setBinding(firstBlock.rhi());
        c["coeffs"].setBinding(coeffs.rhi());
        c["sections"].setBinding(sectionBuf.rhi());
        c["status"].setBinding(status.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["singleSection"].setData(uint32_t{frame.singleSection ? 1u : 0u});
        c["params"]["globalsCap"].setData(globalsWords);
        c["params"]["dcArenaBase"].setData(dcArenaBase);
        c["params"]["dcArenaWords"].setData(dcGroupWords);
        c["params"]["acArenaBase"].setData(acArenaBase);
        c["params"]["acArenaWords"].setData(acGroupWords);
        c["params"]["windowBase"].setData(withLz77Windows ? windowBase : 0xFFFFFFFFu);
        c["params"]["windowCount"].setData(windowCount);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["xsize"].setData(dim.xsize);
        c["params"]["ysize"].setData(dim.ysize);
        c["params"]["colorChannels"].setData(colorChannels);
        c["params"]["bitDepth"].setData(m.bitsPerSample);
        c["params"]["frameFlags"].setData(static_cast<uint32_t>(frame.header.flags));
        c["params"]["numPasses"].setData(frame.header.numPasses);
        c["params"]["numEc"].setData(numEc);
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
        c["params"]["ecW"].setData(ecW);
        c["params"]["ecH"].setData(ecH);
        c["params"]["ecShift"].setData(ecShift);
        {
            // passes.shift and GetDownsamplingBracket per pass, 12 slots as three vectors.
            std::array<uint32_t, 12> shift{};
            std::array<int32_t, 12> minShift{}, maxShift{};
            for (uint32_t p = 0; p < frame.header.numPasses; ++p) {
                shift[p] = frame.header.passShift[p];
                int32_t mx = 2, mn = 3;
                for (uint32_t i = 0;; ++i) {
                    for (uint32_t j = 0; j < frame.header.numDownsample; ++j) {
                        if (i == frame.header.lastPass[j]) {
                            const uint32_t ds = frame.header.downsample[j];
                            mn = ds == 8 ? 3 : ds == 4 ? 2 : ds == 2 ? 1 : 0;
                        }
                    }
                    if (i == frame.header.numPasses - 1) mn = 0;
                    if (i == p) break;
                    mx = mn - 1;
                }
                minShift[p] = mn;
                maxShift[p] = mx;
            }
            const auto quad = [](const auto& a, size_t at) { return std::array{a[at], a[at + 1], a[at + 2], a[at + 3]}; };
            c["params"]["passShiftA"].setData(quad(shift, 0));
            c["params"]["passShiftB"].setData(quad(shift, 4));
            c["params"]["passShiftC"].setData(quad(shift, 8));
            c["params"]["passMinA"].setData(quad(minShift, 0));
            c["params"]["passMinB"].setData(quad(minShift, 4));
            c["params"]["passMinC"].setData(quad(minShift, 8));
            c["params"]["passMaxA"].setData(quad(maxShift, 0));
            c["params"]["passMaxB"].setData(quad(maxShift, 4));
            c["params"]["passMaxC"].setData(quad(maxShift, 8));
        }
        std::array<uint32_t, 4> rw{}, rh{};
        for (size_t i = 0; i < 4; ++i) if (refs[i].planes && refs[i].inXyb) { rw[i] = refs[i].width; rh[i] = refs[i].height; }
        c["params"]["refW"].setData(rw);
        c["params"]["refH"].setData(rh);
    };
    if (modular) {
        modularGlobals_.dispatch(batch, {1, 1, 1}, bindSections);
        modularDcGroups_.dispatch(batch, {numDcGroups, 1, 1}, bindSections);
        modularAcGroups_.dispatch(batch, {numGroups, 1, 1}, bindSections);
        modularFinalize_.dispatch(batch, {1, 1, 1}, bindSections);
        modularToPlanes_.dispatch(batch, {dim.xsize * dim.ysize, 1, 1}, [&](rhi::ShaderCursor c) {
            c["arena"].setBinding(arena.rhi());
            c["xyb"].setBinding(xybA.rhi());
            c["params"]["width"].setData(dim.xsize);
            c["params"]["height"].setData(dim.ysize);
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
            c["params"]["xybMode"].setData(uint32_t{modularXyb ? 1u : 0u});
            c["params"]["sampleScale"].setData(m.floatingPointSample ? 1.0f : 1.0f / static_cast<float>((uint64_t{1} << m.bitsPerSample) - 1));
            c["params"]["floatBits"].setData(m.floatingPointSample && !modularXyb ? m.bitsPerSample : 0u);
            c["params"]["floatExpBits"].setData(m.exponentBitsPerSample);
            c["params"]["ecChannel"].setData(0u);
            c["params"]["ecPlane"].setData(0u);
        });
        if (lf.epfIters > 0) {
            computeSigma_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
                c["arena"].setBinding(arena.rhi());
                c["dc"].setBinding(dc.rhi());
                c["dcSmooth"].setBinding(dcSmooth.rhi());
                c["acs"].setBinding(acs.rhi());
                c["qf"].setBinding(qf.rhi());
                c["epfSharp"].setBinding(epfSharp.rhi());
                c["sigma"].setBinding(sigma.rhi());
                c["params"]["blocksX"].setData(dim.xsizeBlocks);
                c["params"]["blocksY"].setData(dim.ysizeBlocks);
                c["params"]["epfIters"].setData(lf.epfIters);
                c["params"]["useConstantSigma"].setData(uint32_t{1});
                c["params"]["constantInvSigma"].setData(kInvSigmaNum / lf.epfSigmaForModular);
                c["dcFrame"].setBinding(dummy.rhi());
                c["params"]["dcFramePaddedWidth"].setData(0u);
                c["params"]["hshift"].setData(hshift);
                c["params"]["vshift"].setData(vshift);
            });
        }
    } else {
    globals_.dispatch(batch, {1, 1, 1}, bindSections);
    timer.mark("globals");
    if (!useDcFrame && numEc == 0 && !frame.singleSection && platform::env("SJXL_DC_CPU") != "0") {
        // The DC groups' serial chains on CPU cores (the project's one exception to GPU-only
        // coded data, docs/decisions.md "M5"): wait for the globals, decode, upload.
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const std::vector<uint32_t> header, arena.readRange<uint32_t>(dev, 0, kGsHeaderSize));
        if (header[0] == 0) {   // kGsError: the globals thread succeeded
            std::vector<cpu::DcGroupGeom> geoms(numDcGroups);
            const uint32_t dcGroupsX = (dim.xsizeBlocks + dim.groupDim - 1) / dim.groupDim;
            for (uint32_t g = 0; g < numDcGroups; ++g) {
                cpu::DcGroupGeom& geo = geoms[g];
                geo.x0 = (g % dcGroupsX) * dim.groupDim;
                geo.y0 = (g / dcGroupsX) * dim.groupDim;
                geo.w = std::min(dim.groupDim, dim.xsizeBlocks - geo.x0);
                geo.h = std::min(dim.groupDim, dim.ysizeBlocks - geo.y0);
                for (size_t c = 0; c < 3; ++c) { geo.hshift[c] = hshift[c]; geo.vshift[c] = vshift[c]; }
            }
            std::vector<cpu::DcGroupData> data;
            SJXL_ASSIGN(const bool done, cpu::decodeDcGroups(parsed.codestream, frame, header[kGsGlobalTreeBitIndex], geoms, data));
            if (done) {
                std::vector<int32_t> words;
                std::vector<uint32_t> bases(numDcGroups);
                for (uint32_t g = 0; g < numDcGroups; ++g) {
                    bases[g] = static_cast<uint32_t>(words.size());
                    words.push_back(static_cast<int32_t>(data[g].extraPrecision));
                    words.push_back(static_cast<int32_t>(data[g].count));
                    words.insert(words.end(), data[g].words.begin(), data[g].words.end());
                }
                SJXL_ASSIGN(hostDc, gpu::Buffer::fromSpan<int32_t>(dev, words, "dec.hostDc"));
                SJXL_ASSIGN(hostDcBase, gpu::Buffer::fromSpan<uint32_t>(dev, bases, "dec.hostDcBase"));
                dcFromHost = true;
            } else {
                log::debug("decoder: DC groups use a feature the CPU port leaves out; the GPU decodes them");
            }
        }
        timer.mark("dc groups (cpu)");
    }
    dcGroups_.dispatch(batch, {numDcGroups, 1, 1}, bindSections);
    timer.mark("dc groups");
    tables_.dispatch(batch, {kDequantTableFloats, 1, 1}, [&](rhi::ShaderCursor c) {
        c["arena"].setBinding(arena.rhi());
        c["dequant"].setBinding(dequant.rhi());
        c["params"]["totalEntries"].setData(kDequantTableFloats);
    });
    const auto bindFinalize = [&](rhi::ShaderCursor c) {
        c["arena"].setBinding(arena.rhi());
        c["dc"].setBinding(dc.rhi());
        c["dcSmooth"].setBinding(dcSmooth.rhi());
        c["acs"].setBinding(acs.rhi());
        c["qf"].setBinding(qf.rhi());
        c["epfSharp"].setBinding(epfSharp.rhi());
        c["sigma"].setBinding(sigma.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["skipSmoothing"].setData(uint32_t{(frame.header.flags & codestream::FrameHeader::kSkipAdaptiveDCSmoothing) ? 1u : 0u});
        c["params"]["epfIters"].setData(lf.epfIters);
        c["params"]["epfQuantMul"].setData(lf.epfQuantMul);
        for (int i = 0; i < 8; ++i) c["params"]["sharpLut"][i].setData(lf.epfSharpLut[static_cast<size_t>(i)]);
        c["params"]["useConstantSigma"].setData(uint32_t{0});
        c["params"]["constantInvSigma"].setData(0.0f);
        c["dcFrame"].setBinding(dcSource ? dcSource->planes->rhi() : dummy.rhi());
        c["params"]["dcFramePaddedWidth"].setData(dcSource ? dcSource->paddedWidth : 0u);
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
    };
    if (useDcFrame) dcFromFrame_.dispatch(batch, {nblocks, 1, 1}, bindFinalize);   // no smoothing with a DC frame (FinalizeDC)
    else smoothDc_.dispatch(batch, {dim.xsizeBlocks, dim.ysizeBlocks, 1}, bindFinalize);
    computeSigma_.dispatch(batch, {nblocks, 1, 1}, bindFinalize);
    timer.mark("dc finalize");
    acGroups_.dispatch(batch, {numGroups, 1, 1}, bindSections);
    if (numEc > 0) modularFinalize_.dispatch(batch, {1, 1, 1}, bindSections);   // the extra channels' inverse transforms
    const auto bindTransforms = [&](rhi::ShaderCursor c) {
        c["arena"].setBinding(arena.rhi());
        c["coeffs"].setBinding(coeffs.rhi());
        c["dequantF"].setBinding(dequantF.rhi());
        c["dequant"].setBinding(dequant.rhi());
        c["acs"].setBinding(acs.rhi());
        c["qf"].setBinding(qf.rhi());
        c["firstBlock"].setBinding(firstBlock.rhi());
        c["ytox"].setBinding(ytox.rhi());
        c["ytob"].setBinding(ytob.rhi());
        c["dcSmooth"].setBinding(dcSmooth.rhi());
        c["tmp"].setBinding(xybB.rhi());
        c["xyb"].setBinding(xybA.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
        c["params"]["xDm"].setData(static_cast<float>(std::pow(1.25, 2 - static_cast<int>(frame.header.xQmScale))));
        c["params"]["bDm"].setData(static_cast<float>(std::pow(1.25, 2 - static_cast<int>(frame.header.bQmScale))));
        for (int i = 0; i < 4; ++i) c["params"]["quantBias"][i].setData(m.quantBiases[static_cast<size_t>(i)]);
    };
    timer.mark("ac groups");
    dequantCoeffs_.dispatch(batch, {3 * nblocks * 64, 1, 1}, bindTransforms);
    llfFromDc_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransforms);
    idctRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransforms);
    idctCols_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransforms);
    transformSpecial_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransforms);
    }   // VarDCT

    // Extra channel planes at the output size: integers scaled by 1 / (2^bits - 1), upsampled when coded smaller.
    std::optional<gpu::Buffer> ecCoded, ecOut;
    if (numEc > 0) {
        SJXL_ASSIGN(ecOut, gpu::Buffer::ofType<float>(dev, uint64_t{numEc} * outPaddedWidth * outPaddedHeight, "dec.ecOut"));
        bool anyUps = false;
        for (uint32_t e = 0; e < numEc; ++e) anyUps = anyUps || ecUps[e] > 1;
        if (anyUps) { SJXL_ASSIGN(ecCoded, gpu::Buffer::ofType<float>(dev, uint64_t{numEc} * outPaddedWidth * outPaddedHeight, "dec.ecCoded")); }
        std::optional<gpu::Buffer> ecWeights;
        for (uint32_t e = 0; e < numEc; ++e) {
            const bool ups = ecUps[e] > 1;
            ecToPlanes_.dispatch(batch, {ecW[e] * ecH[e], 1, 1}, [&](rhi::ShaderCursor c) {
                c["arena"].setBinding(arena.rhi());
                c["xyb"].setBinding(ups ? ecCoded->rhi() : ecOut->rhi());
                c["params"]["width"].setData(ecW[e]);
                c["params"]["height"].setData(ecH[e]);
                c["params"]["paddedWidth"].setData(outPaddedWidth);
                c["params"]["paddedHeight"].setData(outPaddedHeight);
                c["params"]["xybMode"].setData(0u);
                c["params"]["sampleScale"].setData(1.0f / static_cast<float>((uint64_t{1} << m.extraChannels[e].bitsPerSample) - 1));
                c["params"]["ecChannel"].setData(colorChannels * (modular ? 1u : 0u) + e);
                c["params"]["ecPlane"].setData(e);
            });
            if (ups) {
                if (!ecWeights) {
                    std::vector<float> w(280);
                    std::copy(codestream::kUpsampling2.begin(), codestream::kUpsampling2.end(), w.begin());
                    std::copy(codestream::kUpsampling4.begin(), codestream::kUpsampling4.end(), w.begin() + 15);
                    std::copy(codestream::kUpsampling8.begin(), codestream::kUpsampling8.end(), w.begin() + 70);
                    if (m.customWeightsMask & 1) std::copy(m.upsamplingWeights2.begin(), m.upsamplingWeights2.end(), w.begin());
                    if (m.customWeightsMask & 2) std::copy(m.upsamplingWeights4.begin(), m.upsamplingWeights4.end(), w.begin() + 15);
                    if (m.customWeightsMask & 4) std::copy(m.upsamplingWeights8.begin(), m.upsamplingWeights8.end(), w.begin() + 70);
                    SJXL_ASSIGN(ecWeights, gpu::Buffer::fromSpan<float>(dev, w, "dec.ecUpsWeights"));
                }
                const uint32_t shift = ecUps[e] == 8 ? 3 : ecUps[e] == 4 ? 2 : 1;
                upsample_.dispatch(batch, {outWidth, outHeight, 1}, [&](rhi::ShaderCursor c) {
                    c["src"].setBinding(ecCoded->rhi());
                    c["dst"].setBinding(ecOut->rhi());
                    c["weights"].setBinding(ecWeights->rhi());
                    c["params"]["width"].setData(ecW[e]);
                    c["params"]["height"].setData(ecH[e]);
                    c["params"]["paddedWidth"].setData(outPaddedWidth);
                    c["params"]["paddedHeight"].setData(outPaddedHeight);
                    c["params"]["outWidth"].setData(outWidth);
                    c["params"]["outHeight"].setData(outHeight);
                    c["params"]["outPaddedWidth"].setData(outPaddedWidth);
                    c["params"]["outPaddedHeight"].setData(outPaddedHeight);
                    c["params"]["shift"].setData(shift);
                    c["params"]["channels"].setData(1u);
                    c["params"]["srcPlane"].setData(e);
                    c["params"]["dstPlane"].setData(e);
                });
            }
        }
        ecWeightsKeep = std::move(ecWeights);
    }

    // Loop filters, ping-pong between the two XYB buffers.
    gpu::Buffer* src = &xybA;
    gpu::Buffer* dst = &xybB;
    std::array<uint32_t, 4> chromaShift{}, chromaIn{};
    const auto bindFilter = [&](rhi::ShaderCursor c, uint32_t stage, float sigmaScale) {
        c["params"]["chromaShift"].setData(chromaShift);
        c["params"]["chromaIn"].setData(chromaIn);
        c["src"].setBinding(src->rhi());
        c["dst"].setBinding(dst->rhi());
        c["sigma"].setBinding(sigma.rhi());
        c["params"]["width"].setData(dim.xsize);
        c["params"]["height"].setData(dim.ysize);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["stage"].setData(stage);
        c["params"]["sigmaScale"].setData(sigmaScale);
        c["params"]["borderSadMul"].setData(lf.epfBorderSadMul);
        for (int i = 0; i < 3; ++i) c["params"]["channelScale"][i].setData(lf.epfChannelScale[static_cast<size_t>(i)]);
        const float w1s[3] = {lf.gabXWeight1, lf.gabYWeight1, lf.gabBWeight1};
        const float w2s[3] = {lf.gabXWeight2, lf.gabYWeight2, lf.gabBWeight2};
        for (int ch = 0; ch < 3; ++ch) {
            const float div = 1.0f + 4.0f * (w1s[ch] + w2s[ch]);
            c["params"]["gabWeights"][ch * 4 + 0].setData(1.0f / div);
            c["params"]["gabWeights"][ch * 4 + 1].setData(w1s[ch] / div);
            c["params"]["gabWeights"][ch * 4 + 2].setData(w2s[ch] / div);
            c["params"]["gabWeights"][ch * 4 + 3].setData(0.0f);
        }
    };
    timer.mark("idct + ec");
    // Chroma upsampling first (render pipeline order), horizontal then vertical.
    for (uint32_t dir = 0; dir < 2 && ycbcr; ++dir) {
        bool any = false;
        for (size_t c = 0; c < 3; ++c) {
            const uint32_t sh = dir == 0 ? hshift[c] : vshift[c];
            chromaShift[c] = sh;
            const uint32_t full = dir == 0 ? dim.xsize : dim.ysize;
            chromaIn[c] = (full + (1u << sh) - 1) >> sh;
            any = any || sh != 0;
        }
        if (!any) continue;
        chromaUpsample_.dispatch(batch, {dim.xsizePadded, dim.ysizePadded, 1}, [&](rhi::ShaderCursor c) { bindFilter(c, dir, 0.0f); });
        std::swap(src, dst);
    }
    chromaShift = {};
    chromaIn = {};
    if (lf.gab) {
        gaborish_.dispatch(batch, {dim.xsize, dim.ysize, 1}, [&](rhi::ShaderCursor c) { bindFilter(c, 0, 0.0f); });
        std::swap(src, dst);
    }
    if (lf.epfIters >= 3) {
        epf_.dispatch(batch, {dim.xsize, dim.ysize, 1}, [&](rhi::ShaderCursor c) { bindFilter(c, 0, lf.epfPass0SigmaScale * 1.65f); });
        std::swap(src, dst);
    }
    if (lf.epfIters >= 1) {
        epf_.dispatch(batch, {dim.xsize, dim.ysize, 1}, [&](rhi::ShaderCursor c) { bindFilter(c, 1, 1.65f); });
        std::swap(src, dst);
    }
    if (lf.epfIters >= 2) {
        epf_.dispatch(batch, {dim.xsize, dim.ysize, 1}, [&](rhi::ShaderCursor c) { bindFilter(c, 2, lf.epfPass2SigmaScale * 1.65f); });
        std::swap(src, dst);
    }
    timer.mark("filters");
    // Splines, after the loop filters at the frame's size (render pipeline order).
    if (frame.header.flags & codestream::FrameHeader::kSplines) {
        constexpr uint32_t kMaxSplines = 4096, kMaxControlPoints = 1u << 18, kMaxSegments = 1u << 16, kMaxRowEntries = 1u << 20, kSegWords = 10;
        SJXL_ASSIGN(auto ctrl, gpu::Buffer::ofType<float>(dev, uint64_t{2} * kMaxControlPoints, "dec.splineCtrl"));
        SJXL_ASSIGN(auto segCount, gpu::Buffer::ofType<uint32_t>(dev, kMaxSplines, "dec.splineCount"));
        SJXL_ASSIGN(auto segments, gpu::Buffer::ofType<float>(dev, uint64_t{kMaxSegments} * kSegWords, "dec.splineSegments"));
        SJXL_ASSIGN(auto rowDiff, gpu::Buffer::ofType<uint32_t>(dev, dim.ysize + 2, "dec.splineRowDiff"));
        SJXL_ASSIGN(auto rowStart, gpu::Buffer::ofType<uint32_t>(dev, dim.ysize + 2, "dec.splineRowStart"));
        SJXL_ASSIGN(auto rowFill, gpu::Buffer::ofType<uint32_t>(dev, dim.ysize + 2, "dec.splineRowFill"));
        SJXL_ASSIGN(auto segIdx, gpu::Buffer::ofType<uint32_t>(dev, kMaxRowEntries, "dec.splineRowEntries"));
        SJXL_ASSIGN(auto splineStatus, gpu::Buffer::ofType<uint32_t>(dev, 4, "dec.splineStatus"));
        const auto bindSplines = [&](rhi::ShaderCursor c, uint32_t mode) {
            c["arena"].setBinding(arena.rhi());
            c["xyb"].setBinding(src->rhi());
            c["ctrl"].setBinding(ctrl.rhi());
            c["segCount"].setBinding(segCount.rhi());
            c["segments"].setBinding(segments.rhi());
            c["rowDiff"].setBinding(rowDiff.rhi());
            c["rowStart"].setBinding(rowStart.rhi());
            c["rowFill"].setBinding(rowFill.rhi());
            c["segIdx"].setBinding(segIdx.rhi());
            c["status"].setBinding(splineStatus.rhi());
            c["params"]["width"].setData(dim.xsize);
            c["params"]["height"].setData(dim.ysize);
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
            c["params"]["mode"].setData(mode);
            c["params"]["maxSegments"].setData(kMaxSegments);
            c["params"]["maxRowEntries"].setData(kMaxRowEntries);
        };
        splineSegments_.dispatch(batch, {kMaxSplines, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 0); });
        splineOffsets_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 0); });
        splineSegments_.dispatch(batch, {kMaxSplines, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 1); });
        splineRowCounts_.dispatch(batch, {kMaxSegments, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 1); });
        splineRowStarts_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 1); });
        splineScatter_.dispatch(batch, {kMaxSegments, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 1); });
        splineDraw_.dispatch(batch, {dim.xsize * dim.ysize, 1, 1}, [&](rhi::ShaderCursor c) { bindSplines(c, 1); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const auto st, splineStatus.readAll<uint32_t>(dev));
        if (st[2] != 0) return Error::make(ErrorCode::Unsupported, "splines with {} segments over {} rows (capacity {} / {})", st[0], st[1], kMaxSegments, kMaxRowEntries);
        if (!platform::env("SJXL_DEBUG").empty()) log::info("dec debug: splines: {} segments, {} row entries", st[0], st[1]);
    }
    timer.mark("splines");
    // Upsampling (after the loop filters, before patches: render pipeline order).
    std::optional<gpu::Buffer> upsampled;
    if (upsampling > 1) {
        SJXL_ASSIGN(upsampled, gpu::Buffer::ofType<float>(dev, uint64_t{3} * outPaddedWidth * outPaddedHeight, "dec.upsampled"));
        std::vector<float> w(280);
        std::copy(codestream::kUpsampling2.begin(), codestream::kUpsampling2.end(), w.begin());
        std::copy(codestream::kUpsampling4.begin(), codestream::kUpsampling4.end(), w.begin() + 15);
        std::copy(codestream::kUpsampling8.begin(), codestream::kUpsampling8.end(), w.begin() + 70);
        if (m.customWeightsMask & 1) std::copy(m.upsamplingWeights2.begin(), m.upsamplingWeights2.end(), w.begin());
        if (m.customWeightsMask & 2) std::copy(m.upsamplingWeights4.begin(), m.upsamplingWeights4.end(), w.begin() + 15);
        if (m.customWeightsMask & 4) std::copy(m.upsamplingWeights8.begin(), m.upsamplingWeights8.end(), w.begin() + 70);
        SJXL_ASSIGN(auto weights, gpu::Buffer::fromSpan<float>(dev, w, "dec.upsWeights"));
        upsample_.dispatch(batch, {outWidth, outHeight, 1}, [&](rhi::ShaderCursor c) {
            c["src"].setBinding(src->rhi());
            c["dst"].setBinding(upsampled->rhi());
            c["weights"].setBinding(weights.rhi());
            c["params"]["width"].setData(dim.xsize);
            c["params"]["height"].setData(dim.ysize);
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
            c["params"]["outWidth"].setData(outWidth);
            c["params"]["outHeight"].setData(outHeight);
            c["params"]["outPaddedWidth"].setData(outPaddedWidth);
            c["params"]["outPaddedHeight"].setData(outPaddedHeight);
            c["params"]["shift"].setData(upShift);
            c["params"]["channels"].setData(3u);
            c["params"]["srcPlane"].setData(0u);
            c["params"]["dstPlane"].setData(0u);
        });
        weightsKeep = std::move(weights);
        src = &*upsampled;
    }
    if (frame.header.flags & codestream::FrameHeader::kPatches) {
        // After the loop filters, before the colour transform (render pipeline order).
        const auto bindPatches = [&](rhi::ShaderCursor c) {
            c["arena"].setBinding(arena.rhi());
            c["xyb"].setBinding(src->rhi());
            const char* names[4] = {"ref0", "ref1", "ref2", "ref3"};
            std::array<uint32_t, 4> pw{}, ph{};
            for (size_t i = 0; i < 4; ++i) {
                const bool usable = refs[i].planes && refs[i].inXyb;
                c[names[i]].setBinding(usable ? refs[i].planes->rhi() : dummy.rhi());
                pw[i] = usable ? refs[i].paddedWidth : 0u;
                ph[i] = usable ? refs[i].paddedHeight : 0u;
            }
            c["params"]["width"].setData(outWidth);
            c["params"]["height"].setData(outHeight);
            c["params"]["paddedWidth"].setData(outPaddedWidth);
            c["params"]["paddedHeight"].setData(outPaddedHeight);
            c["params"]["arenaCap"].setData(globalsWords);
            c["params"]["refPaddedW"].setData(pw);
            c["params"]["refPaddedH"].setData(ph);
        };
        bucketPatches_.dispatch(batch, {1, 1, 1}, bindPatches);
        applyPatches_.dispatch(batch, {outWidth * outHeight, 1, 1}, bindPatches);
    }
    timer.mark("upsample + patches");
    // Noise, after patches and before the colour transform.
    std::optional<gpu::Buffer> noiseA, noiseB;
    if (frame.header.flags & codestream::FrameHeader::kNoise) {
        SJXL_ASSIGN(noiseA, gpu::Buffer::ofType<float>(dev, uint64_t{3} * outPaddedWidth * outPaddedHeight, "dec.noiseA"));
        SJXL_ASSIGN(noiseB, gpu::Buffer::ofType<float>(dev, uint64_t{3} * outPaddedWidth * outPaddedHeight, "dec.noiseB"));
        const auto bindNoise = [&](rhi::ShaderCursor c) {
            c["arena"].setBinding(arena.rhi());
            c["noise"].setBinding(noiseA->rhi());
            c["conv"].setBinding(noiseB->rhi());
            c["xyb"].setBinding(src->rhi());
            c["params"]["width"].setData(outWidth);
            c["params"]["height"].setData(outHeight);
            c["params"]["paddedWidth"].setData(outPaddedWidth);
            c["params"]["paddedHeight"].setData(outPaddedHeight);
            c["params"]["groupDim"].setData(dim.groupDim);
            c["params"]["visibleFrame"].setData(indices.visible);
            c["params"]["nonvisibleFrame"].setData(indices.nonvisible);
        };
        const uint32_t tilesX = (outWidth + dim.groupDim - 1) / dim.groupDim, tilesY = (outHeight + dim.groupDim - 1) / dim.groupDim;
        noiseRandom_.dispatch(batch, {tilesX, tilesY, 1}, bindNoise);
        noiseConvolve_.dispatch(batch, {outWidth, outHeight, 1}, bindNoise);
        noiseAdd_.dispatch(batch, {outWidth, outHeight, 1}, bindNoise);
    }
    timer.mark("noise");
    const auto bindOutput = [&](rhi::ShaderCursor c, bool fromCanvas) {
        c["xyb"].setBinding(fromCanvas ? canvas->rhi() : src->rhi());
        c["rgb8"].setBinding(rgb8.rhi());
        c["outPlanes"].setBinding(frameOut ? frameOut->rhi() : dummy.rhi());
        c["params"]["width"].setData(fromCanvas ? canvasW : outWidth);
        c["params"]["height"].setData(fromCanvas ? canvasH : outHeight);
        c["params"]["paddedWidth"].setData(fromCanvas ? canvasW : outPaddedWidth);
        c["params"]["paddedHeight"].setData(fromCanvas ? canvasH : outPaddedHeight);
        const float scale = 255.0f / m.intensityTarget;
        for (int r = 0; r < 3; ++r) {
            for (int k = 0; k < 3; ++k) c["params"]["inverseMatrix"][r * 4 + k].setData(m.opsinInverseMatrix[static_cast<size_t>(r * 3 + k)] * scale);
            c["params"]["inverseMatrix"][r * 4 + 3].setData(0.0f);
        }
        for (int k = 0; k < 3; ++k) {
            c["params"]["negBias"][k].setData(m.opsinBiases[static_cast<size_t>(k)]);
            c["params"]["negBiasCbrt"][k].setData(std::cbrt(m.opsinBiases[static_cast<size_t>(k)]));
        }
        c["params"]["negBias"][3].setData(0.0f);
        c["params"]["negBiasCbrt"][3].setData(0.0f);
        // The transfer function of the declared colour encoding (dec_xyb.cc inverse_gamma).
        uint32_t transfer = 0;
        float gamma = 1.0f;
        if (m.wantIcc && m.xybEncoded) {
            // dec_xyb.cc: an ICC-described encoding of an XYB image cannot be output to, so
            // libjxl writes linear sRGB (grey when the ICC is grey).
            transfer = 1;
        } else if (m.tfHaveGamma) {
            transfer = 2;
            gamma = static_cast<float>(m.tfGamma) * 1e-7f;
        } else if (m.transferFunction == 8) {
            transfer = 1;
        } else if (m.transferFunction == 17) {   // DCI
            transfer = 2;
            gamma = 1.0f / 2.6f;
        } else if (m.transferFunction == 1) {   // BT.709
            transfer = 3;
        }
        c["params"]["transfer"].setData(transfer);
        c["params"]["gamma"].setData(gamma);
        c["params"]["mode"].setData(uint32_t{fromCanvas ? 1u : ycbcr ? 2u : (modular && !modularXyb) ? 1u : 0u});
        c["params"]["channels"].setData(outChannels);
        c["params"]["alphaPlane"].setData(fromCanvas ? (alphaIndex < numEc ? 3 + alphaIndex : 0u) : (alphaIndex < numEc ? alphaIndex : 0u));
        c["params"]["bytesPerSample"].setData(outDepth / 8);
        c["params"]["hasAlpha"].setData(uint32_t{alphaIndex < numEc ? 1u : 0u});
        c["params"]["grey"].setData(uint32_t{greyOut ? 1u : 0u});
        c["params"]["orientation"].setData(fromCanvas ? m.orientation : 1u);
        // Spot colours (stage_spot.cc), mixed in when packing the canvas.
        uint32_t numSpot = 0;
        std::array<uint32_t, 4> spotPlane{};
        for (uint32_t e = 0; e < numEc && numSpot < 4; ++e) {
            if (m.extraChannels[e].type != 2 || !fromCanvas) continue;
            spotPlane[numSpot] = 3 + e;
            for (int k = 0; k < 4; ++k) c["params"]["spotColor"][numSpot][k].setData(m.extraChannels[e].spotColor[static_cast<size_t>(k)]);
            ++numSpot;
        }
        c["params"]["numSpot"].setData(numSpot);
        c["params"]["spotPlane"].setData(spotPlane);
        c["alpha"].setBinding(fromCanvas ? canvas->rhi() : ecOut ? ecOut->rhi() : dummy.rhi());
    };
    const bool emitted = displayed && (frame.header.isLast || frame.header.duration > 0);
    if (displayed && !platform::env("SJXL_DEBUG").empty()) {
        std::string ecs;
        for (uint32_t e = 0; e < numEc; ++e) ecs += std::format("[type {} assoc {} blend mode {} alpha {} clamp {}] ", m.extraChannels[e].type, m.extraChannels[e].alphaAssociated, frame.header.ecBlending[e].mode, frame.header.ecBlending[e].alphaChannel, frame.header.ecBlending[e].clamp);
        log::info("dec debug: frame blend mode {} source {} alpha channel {} clamp {} (first alpha ec {}); origin ({},{}) {}x{}; ec {}", blending.mode, blending.source, blending.alphaChannel, blending.clamp, alphaIndex, frame.header.customSizeOrOrigin ? frame.header.x0 : 0, frame.header.customSizeOrOrigin ? frame.header.y0 : 0, outWidth, outHeight, ecs);
    }
    if (displayed) {
        // The frame in the output encoding, composed onto the canvas (stage_blending.cc), then packed.
        toOutputPlanes_.dispatch(batch, {outWidth * outHeight, 1, 1}, [&](rhi::ShaderCursor c) { bindOutput(c, false); });
        blendCanvas_.dispatch(batch, {canvasW * canvasH, 1, 1}, [&](rhi::ShaderCursor c) {
            c["fg"].setBinding(frameOut->rhi());
            c["fgEc"].setBinding(ecOut ? ecOut->rhi() : dummy.rhi());
            c["bg"].setBinding(bgCanvas ? bgCanvas->planes->rhi() : dummy.rhi());
            c["canvas"].setBinding(canvas->rhi());
            c["params"]["width"].setData(canvasW);
            c["params"]["height"].setData(canvasH);
            c["params"]["fw"].setData(outWidth);
            c["params"]["fh"].setData(outHeight);
            c["params"]["x0"].setData(frame.header.customSizeOrOrigin ? frame.header.x0 : 0);
            c["params"]["y0"].setData(frame.header.customSizeOrOrigin ? frame.header.y0 : 0);
            c["params"]["fgPaddedWidth"].setData(outPaddedWidth);
            c["params"]["fgPaddedHeight"].setData(outPaddedHeight);
            c["params"]["numEc"].setData(numEc);
            c["params"]["hasBg"].setData(uint32_t{bgCanvas ? 1u : 0u});
            c["params"]["alphaIndex"].setData(alphaIndex < numEc ? blending.alphaChannel : numEc);
            c["params"]["alphaPremultiplied"].setData(uint32_t{alphaIndex < numEc && m.extraChannels[blending.alphaChannel].alphaAssociated ? 1u : 0u});
            c["params"]["modes"].setData(std::array<uint32_t, 4>{blending.mode, blending.clamp ? 1u : 0u, 0u, 0u});
            std::array<uint32_t, 4> ecModes{};
            for (uint32_t e = 0; e < numEc; ++e) {
                const codestream::BlendingInfo& eb = frame.header.ecBlending[e];
                ecModes[e] = eb.mode | (eb.clamp ? 1u << 8 : 0u) | (eb.alphaChannel << 16);
            }
            c["params"]["ecModes"].setData(ecModes);
        });
        if (emitted) output_.dispatch(batch, {(outBytes + 3) / 4, 1, 1}, [&](rhi::ShaderCursor c) { bindOutput(c, true); });
    }
    SJXL_TRY(batch.submit(true));

    SJXL_ASSIGN(const std::vector<uint32_t> statusHost, status.readAll<uint32_t>(dev));
    timer.mark("output + readback");
    timer.report();
    for (uint32_t i = 0; i < numThreads; ++i) {
        const uint32_t code = statusHost[i] & 0xFF;
        const uint32_t stage = statusHost[i] >> 8;
        if (code == kErrNone) continue;
        if (code == kErrLz77Window) {
            log::debug("decoder: thread {} needs an LZ77 window (stage {})", i, stage);
        } else {
            if (!platform::env("SJXL_DEBUG").empty()) {
                log::info("decoder: thread {} failed with code {} at stage {} (1 DcGlobal, 2 DC group, 3 AcGlobal, 4 AC group, 5 modular finalise)", i, code, stage);
            }
        }
        const std::string where = i == 0 ? "the global sections" : i <= numDcGroups ? std::format("DC group {}", i - 1)
                                                                                    : std::format("AC group {}", i - 1 - numDcGroups);
        if (code == kErrLz77Window) return Error(ErrorCode::Unsupported, "LZ77 window");
        if (code == kErrUnsupported) return Error::make(ErrorCode::Unsupported, "{} in {}", describe(code), where);
        if (code == kErrArena) return Error::make(ErrorCode::OutOfMemory, "{} in {}", describe(code), where);
        return Error::make(ErrorCode::Corrupt, "{} in {}", describe(code), where);
    }
    SJXL_ASSIGN(const std::vector<uint32_t> header, arena.readRange<uint32_t>(dev, 0, kGsHeaderSize));
    stats_.dcGlobalBits = header[45];
    stats_.usedAcs = header[47];
    if (modular && !platform::env("SJXL_DEBUG").empty()) {
        log::info("decoder: modular global tree {} nodes", header[kGsTreeNumNodesIndex]);
        if (!platform::env("SJXL_DEBUG_TREE").empty()) {
            // The nodes: property (-1 leaf), split value, children, predictor, offset, multiplier, leaf id.
            const uint32_t numNodes = header[kGsTreeNumNodesIndex];
            SJXL_ASSIGN(const std::vector<uint32_t> nodes, arena.readRange<uint32_t>(dev, header[kGsTreeNumNodesIndex - 1], uint64_t{numNodes} * 8));
            for (uint32_t n = 0; n < numNodes && n < 400; ++n) {
                const int32_t prop = static_cast<int32_t>(nodes[n * 8]);
                if (prop < 0) log::info("  node {}: leaf {} predictor {} offset {} mul {}", n, nodes[n * 8 + 7], nodes[n * 8 + 4], static_cast<int32_t>(nodes[n * 8 + 5]), nodes[n * 8 + 6]);
                else log::info("  node {}: property {} > {} ? {} : {}", n, prop, static_cast<int32_t>(nodes[n * 8 + 1]), nodes[n * 8 + 2], nodes[n * 8 + 3]);
            }
        }
    }
    if (referenceOnly || dcFrame) {
        // Saved before the colour transform: the XYB planes as they are after the filters
        // (a reference frame for patches, or the DC image of the frame one level down).
        RefFrame& ref = referenceOnly ? refs[frame.header.saveAsReference] : dcFrames[frame.header.dcLevel];
        ref.planes.emplace(std::move(*src));
        ref.paddedWidth = outPaddedWidth;
        ref.paddedHeight = outPaddedHeight;
        ref.width = outWidth;
        ref.height = outHeight;
        ref.inXyb = true;
        return FrameOutput{};
    }
    // CanBeReferenced: not last, and a duration-0 frame or an explicit slot keeps its canvas.
    if (!frame.header.isLast && (frame.header.duration == 0 || frame.header.saveAsReference != 0)) {
        RefFrame& ref = refs[frame.header.saveAsReference];
        ref.planes.emplace(std::move(*canvas));
        ref.paddedWidth = canvasW;
        ref.paddedHeight = canvasH;
        ref.width = canvasW;
        ref.height = canvasH;
        ref.inXyb = false;
        if (!emitted) return FrameOutput{};
        SJXL_ASSIGN(const std::vector<uint32_t> wordsOut, rgb8.readAll<uint32_t>(dev));
        FrameOutput fo;
        fo.emitted = true;
        fo.image.width = orientationSwaps ? canvasH : canvasW;
        fo.image.height = orientationSwaps ? canvasW : canvasH;
        fo.image.channels = outChannels;
        fo.image.bitDepth = outDepth;
        fo.image.pixels.resize(outBytes);
        std::memcpy(fo.image.pixels.data(), wordsOut.data(), outBytes);
        return fo;
    }
    if (!emitted) return FrameOutput{};

    if (!platform::env("SJXL_DEBUG").empty() && (frame.header.flags & codestream::FrameHeader::kPatches)) {
        log::info("dec debug: patches: {} reference positions, {} positions; frame {}x{}", header[56], header[58], dim.xsize, dim.ysize);
        for (int i = 0; i < 4; ++i) if (refs[static_cast<size_t>(i)].planes) log::info("dec debug: reference slot {}: {}x{} (padded {}x{})", i, refs[static_cast<size_t>(i)].width, refs[static_cast<size_t>(i)].height, refs[static_cast<size_t>(i)].paddedWidth, refs[static_cast<size_t>(i)].paddedHeight);
        SJXL_ASSIGN(const auto refsHost, arena.readRange<uint32_t>(dev, header[57], std::min<uint32_t>(header[56], 8) * 5));
        for (uint32_t i = 0; i < std::min<uint32_t>(header[56], 8); ++i) log::info("dec debug: ref patch {}: slot {} at ({},{}) size {}x{}", i, refsHost[i * 5], refsHost[i * 5 + 1], refsHost[i * 5 + 2], refsHost[i * 5 + 3], refsHost[i * 5 + 4]);
        SJXL_ASSIGN(const auto posHost, arena.readRange<uint32_t>(dev, header[59], std::min<uint32_t>(header[58], 12) * 4));
        for (uint32_t i = 0; i < std::min<uint32_t>(header[58], 12); ++i) log::info("dec debug: position {}: ({},{}) ref patch {} mode {}", i, posHost[i * 4], posHost[i * 4 + 1], posHost[i * 4 + 2], posHost[i * 4 + 3]);
    }
    if (!modular && !platform::env("SJXL_DEBUG").empty()) {
        SJXL_ASSIGN(const auto dcHost, dc.readAll<float>(dev));
        SJXL_ASSIGN(const auto dcsHost, dcSmooth.readAll<float>(dev));
        SJXL_ASSIGN(const auto coHost, coeffs.readAll<int32_t>(dev));
        SJXL_ASSIGN(const auto dqHost, dequantF.readAll<float>(dev));
        SJXL_ASSIGN(const auto xybHost, xybA.readAll<float>(dev));
        SJXL_ASSIGN(const auto acsHost, acs.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto qfHost, qf.readAll<uint32_t>(dev));
        log::info("dec debug: gs={} qdc={} invGs={} mulDc=({}, {}, {}) cfl dcX={} dcB={} bctx numCtxs={}", header[1], header[2],
                  std::bit_cast<float>(header[3]), std::bit_cast<float>(header[9]), std::bit_cast<float>(header[10]),
                  std::bit_cast<float>(header[11]), std::bit_cast<float>(header[17]), std::bit_cast<float>(header[18]), header[20]);
        log::info("dec debug: dc block0 = {} {} {}; smooth = {} {} {}; acs0={} qf0={}", dcHost[0], dcHost[nblocks], dcHost[2 * nblocks],
                  dcsHost[0], dcsHost[nblocks], dcsHost[2 * nblocks], acsHost[0], qfHost[0]);
        const uint32_t dbgBlock = static_cast<uint32_t>(std::atoi(platform::env("SJXL_DEBUG_BLOCK").c_str()));
        for (uint32_t ch = 0; ch < 3; ++ch) {
            std::string co;
            for (int k = 0; k < 64; ++k) co += std::format("{} ", coHost[(ch * nblocks + dbgBlock) * 64 + k]);
            log::info("dec debug: coeffs c{} block{} = {}", ch, dbgBlock, co);
        }
        log::info("dec debug: dc block{} = {} {} {} (qf {})", dbgBlock, dcHost[dbgBlock], dcHost[nblocks + dbgBlock], dcHost[2 * nblocks + dbgBlock], qfHost[dbgBlock]);
        SJXL_ASSIGN(const auto fbHost, firstBlock.readAll<uint32_t>(dev));
        {
            SJXL_ASSIGN(const auto orders, arena.readRange<uint32_t>(dev, header[43], 3 * 64));
            for (uint32_t ch = 0; ch < 3; ++ch) {
                std::string o;
                for (int k = 0; k < 64; ++k) o += std::format("{} ", orders[ch * 64 + k]);
                log::info("dec debug: order bucket0 c{} = {}", ch, o);
            }
        }
        log::info("dec debug: loop filter gab={} custom={} epfIters={} sharpCustom={} weightCustom={} sigmaCustom={} flags={:#x} xqm={} bqm={}", lf.gab,
                  lf.gabCustom, lf.epfIters, lf.epfSharpCustom, lf.epfWeightCustom, lf.epfSigmaCustom, frame.header.flags, frame.header.xQmScale,
                  frame.header.bQmScale);
        SJXL_ASSIGN(const auto ytoxHost, ytox.readAll<int32_t>(dev));
        SJXL_ASSIGN(const auto ytobHost, ytob.readAll<int32_t>(dev));
        {
            int minX = 999, maxX = -999, minB = 999, maxB = -999;
            for (size_t i = 0; i < ytoxHost.size(); ++i) { minX = std::min(minX, ytoxHost[i]); maxX = std::max(maxX, ytoxHost[i]); minB = std::min(minB, ytobHost[i]); maxB = std::max(maxB, ytobHost[i]); }
            uint32_t minQf = 9999, maxQf = 0, acsMask = 0;
            for (uint32_t b = 0; b < nblocks; ++b) { minQf = std::min(minQf, qfHost[b]); maxQf = std::max(maxQf, qfHost[b]); if (acsHost[b] & 1) acsMask |= 1u << (acsHost[b] >> 1); }
            log::info("dec debug: ytox [{}, {}] ytob [{}, {}] qf [{}, {}] acs mask 0x{:x} cfl base ({}, {}) cf {} dcCtxs {} nqf {} usedOrders 0x{:x} numHist {}",
                      minX, maxX, minB, maxB, minQf, maxQf, acsMask, std::bit_cast<float>(header[13]), std::bit_cast<float>(header[14]), header[12], header[21], header[22], header[42], header[40]);
        }
        log::info("dec debug: xqm {} bqm {} xDm {} bDm {}; firstBlock[0..4] = {} {} {} {}; acs[0..4] = {} {} {} {}", frame.header.xQmScale,
                  frame.header.bQmScale, std::pow(1.25f, 2 - static_cast<int>(frame.header.xQmScale)),
                  std::pow(1.25f, 2 - static_cast<int>(frame.header.bQmScale)), fbHost[0], fbHost[1], fbHost[2], fbHost[3], acsHost[0],
                  acsHost[1], acsHost[2], acsHost[3]);
        if (!platform::env("SJXL_DEBUG_ACS").empty()) {
            // strategy map: one char per 8x8 block (hex strategy, '.' for a non-first block of a varblock)
            for (uint32_t by = 0; by < dim.ysizeBlocks; ++by) {
                std::string row;
                for (uint32_t bx = 0; bx < dim.xsizeBlocks; ++bx) {
                    const uint32_t a = acsHost[by * dim.xsizeBlocks + bx];
                    row += (a & 1) ? std::format("{:x} ", a >> 1) : ". ";
                }
                log::info("dec debug: acs row {:3} {}", by, row);
            }
        }
        if (const auto mapsPath = platform::env("SJXL_DEBUG_MAPS"); !mapsPath.empty()) {
            // The quant field and strategy map, nblocks uint32 each, for the encoder's oracle experiments.
            std::vector<uint8_t> bytes(nblocks * 8);
            std::memcpy(bytes.data(), qfHost.data(), nblocks * 4);
            std::memcpy(bytes.data() + nblocks * 4, acsHost.data(), nblocks * 4);
            if (auto w = image::writeFile(mapsPath, bytes); !w) log::warn("decoder: cannot write {}", mapsPath);
        }
        if (!platform::env("SJXL_DEBUG_QF").empty()) {
            // quant field: one number per 8x8 block
            for (uint32_t by = 0; by < dim.ysizeBlocks; ++by) {
                std::string row;
                for (uint32_t bx = 0; bx < dim.xsizeBlocks; ++bx) row += std::format("{:3}", qfHost[by * dim.xsizeBlocks + bx]);
                log::info("dec debug: qf row {:3} {}", by, row);
            }
        }
        for (uint32_t ch = 0; ch < 3; ++ch) {
            std::string dq;
            for (int k = 0; k < 16; ++k) dq += std::format("{:.5f} ", dqHost[(ch * nblocks + dbgBlock) * 64 + k]);
            const uint32_t bx = dbgBlock % dim.xsizeBlocks, by = dbgBlock / dim.xsizeBlocks;
            std::string row;
            for (int k = 0; k < 8; ++k) row += std::format("{:.4f} ", xybHost[ch * planePixels + (by * 8) * dim.xsizePadded + bx * 8 + k]);
            log::info("dec debug: c{} dequantF[0..16) = {} / xyb row0 = {}", ch, dq, row);
        }
        log::info("dec debug: xyb(0,0) = {} {} {}", xybHost[0], xybHost[planePixels], xybHost[2 * planePixels]);
        SJXL_ASSIGN(const auto tabHost, dequant.readAll<float>(dev));
        if (const auto dumpPath = platform::env("SJXL_DEBUG_TABLES"); !dumpPath.empty()) {
            std::ofstream out(dumpPath, std::ios::binary);
            out.write(reinterpret_cast<const char*>(tabHost.data()), static_cast<std::streamsize>(tabHost.size() * sizeof(float)));
            log::info("dec debug: dequant tables written to {}", dumpPath);
        }
        log::info("dec debug: dequant DCT8 X[0,1,8,9,63] = {} {} {} {} {}; Y[0,1,8] = {} {} {}; B[0,1,8,63] = {} {} {} {}", tabHost[0], tabHost[1], tabHost[8], tabHost[9], tabHost[63],
                  tabHost[64], tabHost[65], tabHost[72], tabHost[128], tabHost[129], tabHost[136], tabHost[191]);
        log::info("dec debug: dequant DCT16 X[0,1] = {} {}; AFV X[0..4] = {} {} {} {}", tabHost[768], tabHost[769], tabHost[7488], tabHost[7489], tabHost[7490], tabHost[7491]);
    }
    SJXL_ASSIGN(const std::vector<uint32_t> words, rgb8.readAll<uint32_t>(dev));
    FrameOutput fo;
    fo.emitted = true;
    fo.image.width = orientationSwaps ? canvasH : canvasW;
    fo.image.height = orientationSwaps ? canvasW : canvasH;
    fo.image.channels = outChannels;
    fo.image.bitDepth = outDepth;
    fo.image.pixels.resize(outBytes);
    std::memcpy(fo.image.pixels.data(), words.data(), outBytes);
    return fo;
}

}   // namespace sjxl::dec
