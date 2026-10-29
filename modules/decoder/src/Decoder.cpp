// Copyright (c) 2026 jesus luque.
#include "sjxl/decoder/Decoder.h"

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

namespace sjxl::dec {
namespace {

constexpr uint32_t kGsHeaderSize = 64;
constexpr uint32_t kGlobalsWords = 4u << 20;      // 16 MB: AC code sets, orders, matrices encodings
constexpr uint32_t kDcGroupWords = 1u << 20;      // 4 MB per DC group
constexpr uint32_t kAcGroupWords = 1024;
constexpr uint32_t kModularGroupWords = 1u << 19;   // a group's local trees, histograms and predictor state
constexpr float    kInvSigmaNum = -1.1715728752538099024f;
constexpr uint32_t kMaxPatchPositions = 1u << 16;     // frame.slangh kMaxPatchPositions
constexpr uint32_t kLz77WindowWords = 1u << 20;
constexpr uint32_t kDequantTableFloats = 394752;
constexpr uint32_t kErrNone = 0, kErrCorrupt = 1, kErrUnsupported = 2, kErrArena = 3, kErrLz77Window = 4;

struct Uint2 {
    uint32_t x, y;
};

const char* describe(uint32_t code) {
    switch (code) {
    case kErrCorrupt: return "the codestream violates the specification";
    case kErrUnsupported: return "a feature this decoder does not implement (modular Squeeze, RAW quant tables)";
    case kErrArena: return "the decoder ran out of scratch memory";
    case kErrLz77Window: return "LZ77 window";
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
    SJXL_ASSIGN(dec->tocPermutation_, gpu::ComputeKernel::create(lib, "sjxl/dec/toc", "decodeTocPermutation"));
    SJXL_ASSIGN(dec->dcFromFrame_, gpu::ComputeKernel::create(lib, "sjxl/dec/dc_finalize", "dcFromFrame"));
    SJXL_ASSIGN(dec->bucketPatches_, gpu::ComputeKernel::create(lib, "sjxl/dec/patches", "bucketPatches"));
    SJXL_ASSIGN(dec->applyPatches_, gpu::ComputeKernel::create(lib, "sjxl/dec/patches", "applyPatches"));
    SJXL_ASSIGN(dec->gaborish_, gpu::ComputeKernel::create(lib, "sjxl/dec/filters", "gaborish"));
    SJXL_ASSIGN(dec->epf_, gpu::ComputeKernel::create(lib, "sjxl/dec/filters", "epf"));
    SJXL_ASSIGN(dec->output_, gpu::ComputeKernel::create(lib, "sjxl/dec/output", "xybToSrgb8"));
    return dec;
}

Result<codestream::Parsed> Decoder::parseAll(std::span<const uint8_t> file) {
    SJXL_ASSIGN(codestream::Parsed parsed, codestream::parse(file));
    while (parsed.incomplete) {
        SJXL_ASSIGN(const TocPermutation perm, decodeTocPermutation(parsed, parsed.frames.back()));
        SJXL_TRY(codestream::finishToc(parsed, perm.order, perm.endBit));
    }
    return parsed;
}

Result<image::Rgb8> Decoder::decode(std::span<const uint8_t> file) {
    SJXL_ASSIGN(const codestream::Parsed parsed, parseAll(file));
    std::array<RefFrame, 4> refs;
    std::array<RefFrame, 5> dcFrames;
    image::Rgb8 image;
    for (const codestream::ParsedFrame& frame : parsed.frames) {
        auto result = run(parsed, frame, false, refs, dcFrames);
        if (!result && result.error().code() == ErrorCode::Unsupported && result.error().message() == "LZ77 window") {
            result = run(parsed, frame, true, refs, dcFrames);
        }
        if (!result) return result.error();
        image = std::move(*result);
    }
    return image;
}

Result<Decoder::TocPermutation> Decoder::decodeTocPermutation(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame) {
    gpu::Device& dev = *device_;
    std::vector<uint32_t> csWords((parsed.codestream.size() + 3) / 4 + 2, 0);
    std::memcpy(csWords.data(), parsed.codestream.data(), parsed.codestream.size());
    SJXL_ASSIGN(auto cs, gpu::Buffer::fromSpan<uint32_t>(dev, csWords, "dec.toc.cs"));
    constexpr uint32_t kTocArenaWords = 1u << 20;
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

Result<image::Rgb8> Decoder::run(const codestream::Parsed& parsed, const codestream::ParsedFrame& frame, bool withLz77Windows,
                                 std::array<RefFrame, 4>& refs, std::array<RefFrame, 5>& dcFrames) {
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
    if (modular) {
        const uint64_t pixels = uint64_t{dim.xsize} * dim.ysize;
        if (pixels * (colorChannels + 2) > (uint64_t{1} << 31)) return Error(ErrorCode::Unsupported, "modular frames above 512 Mpixel");
        globalsWords = kGlobalsWords + static_cast<uint32_t>((colorChannels + 1) * pixels);
        dcGroupWords = kModularGroupWords;
        acGroupWords = std::max(kModularGroupWords, 4u * dim.groupDim * dim.groupDim);
        if (uint64_t{acGroupWords} * numGroups < (colorChannels + 1) * pixels) {
            acGroupWords = static_cast<uint32_t>(((colorChannels + 1) * pixels + numGroups - 1) / numGroups);
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
    const uint32_t outBytes = dim.xsize * dim.ysize * 3;
    SJXL_ASSIGN(auto rgb8, gpu::Buffer::ofType<uint32_t>(dev, (outBytes + 3) / 4, "dec.rgb8"));
    SJXL_ASSIGN(auto dummy, gpu::Buffer::ofType<float>(dev, 4, "dec.dummy"));   // stands in for absent optional inputs

    gpu::CommandBatch batch(dev);

    const auto bindSections = [&](rhi::ShaderCursor c) {
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
        c["params"]["refW"].setData(std::array<uint32_t, 4>{refs[0].width, refs[1].width, refs[2].width, refs[3].width});
        c["params"]["refH"].setData(std::array<uint32_t, 4>{refs[0].height, refs[1].height, refs[2].height, refs[3].height});
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
            c["params"]["sampleScale"].setData(1.0f / static_cast<float>((uint64_t{1} << m.bitsPerSample) - 1));
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
            });
        }
    } else {
    globals_.dispatch(batch, {1, 1, 1}, bindSections);
    dcGroups_.dispatch(batch, {numDcGroups, 1, 1}, bindSections);
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
    };
    if (useDcFrame) dcFromFrame_.dispatch(batch, {nblocks, 1, 1}, bindFinalize);   // no smoothing with a DC frame (FinalizeDC)
    else smoothDc_.dispatch(batch, {dim.xsizeBlocks, dim.ysizeBlocks, 1}, bindFinalize);
    computeSigma_.dispatch(batch, {nblocks, 1, 1}, bindFinalize);
    acGroups_.dispatch(batch, {numGroups, 1, 1}, bindSections);
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
        c["params"]["xDm"].setData(static_cast<float>(std::pow(1.25, 2 - static_cast<int>(frame.header.xQmScale))));
        c["params"]["bDm"].setData(static_cast<float>(std::pow(1.25, 2 - static_cast<int>(frame.header.bQmScale))));
        for (int i = 0; i < 4; ++i) c["params"]["quantBias"][i].setData(m.quantBiases[static_cast<size_t>(i)]);
    };
    dequantCoeffs_.dispatch(batch, {3 * nblocks * 64, 1, 1}, bindTransforms);
    llfFromDc_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransforms);
    idctRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransforms);
    idctCols_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransforms);
    transformSpecial_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransforms);
    }   // VarDCT

    // Loop filters, ping-pong between the two XYB buffers.
    gpu::Buffer* src = &xybA;
    gpu::Buffer* dst = &xybB;
    const auto bindFilter = [&](rhi::ShaderCursor c, uint32_t stage, float sigmaScale) {
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
    if (frame.header.flags & codestream::FrameHeader::kPatches) {
        // After the loop filters, before the colour transform (render pipeline order).
        const auto bindPatches = [&](rhi::ShaderCursor c) {
            c["arena"].setBinding(arena.rhi());
            c["xyb"].setBinding(src->rhi());
            const char* names[4] = {"ref0", "ref1", "ref2", "ref3"};
            std::array<uint32_t, 4> pw{}, ph{};
            for (size_t i = 0; i < 4; ++i) {
                c[names[i]].setBinding(refs[i].planes ? refs[i].planes->rhi() : dummy.rhi());
                pw[i] = refs[i].paddedWidth;
                ph[i] = refs[i].paddedHeight;
            }
            c["params"]["width"].setData(dim.xsize);
            c["params"]["height"].setData(dim.ysize);
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
            c["params"]["arenaCap"].setData(globalsWords);
            c["params"]["refPaddedW"].setData(pw);
            c["params"]["refPaddedH"].setData(ph);
        };
        bucketPatches_.dispatch(batch, {1, 1, 1}, bindPatches);
        applyPatches_.dispatch(batch, {dim.xsize * dim.ysize, 1, 1}, bindPatches);
    }
    if (!referenceOnly && !dcFrame) output_.dispatch(batch, {(outBytes + 3) / 4, 1, 1}, [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(src->rhi());
        c["rgb8"].setBinding(rgb8.rhi());
        c["params"]["width"].setData(dim.xsize);
        c["params"]["height"].setData(dim.ysize);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
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
        if (m.tfHaveGamma) {
            transfer = 2;
            gamma = static_cast<float>(m.tfGamma) * 1e-7f;
        } else if (m.transferFunction == 8) {
            transfer = 1;
        } else if (m.transferFunction == 17) {   // DCI
            transfer = 2;
            gamma = 1.0f / 2.6f;
        }
        c["params"]["transfer"].setData(transfer);
        c["params"]["gamma"].setData(gamma);
        c["params"]["mode"].setData(uint32_t{(modular && !modularXyb) ? 1u : 0u});
    });
    SJXL_TRY(batch.submit(true));

    SJXL_ASSIGN(const std::vector<uint32_t> statusHost, status.readAll<uint32_t>(dev));
    for (uint32_t i = 0; i < numThreads; ++i) {
        const uint32_t code = statusHost[i] & 0xFF;
        const uint32_t stage = statusHost[i] >> 8;
        if (code == kErrNone) continue;
        if (code == kErrLz77Window) {
            log::debug("decoder: thread {} needs an LZ77 window (stage {})", i, stage);
        } else {
            log::debug("decoder: thread {} failed with code {} at stage {} (1 DcGlobal, 2 DC group, 3 AcGlobal, 4 AC group, 5 modular finalise)", i, code, stage);
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
    if (referenceOnly || dcFrame) {
        // Saved before the colour transform: the XYB planes as they are after the filters
        // (a reference frame for patches, or the DC image of the frame one level down).
        RefFrame& ref = referenceOnly ? refs[frame.header.saveAsReference] : dcFrames[frame.header.dcLevel];
        ref.planes.emplace(std::move(*src));
        ref.paddedWidth = dim.xsizePadded;
        ref.paddedHeight = dim.ysizePadded;
        ref.width = dim.xsize;
        ref.height = dim.ysize;
        return image::Rgb8{};
    }

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
    image::Rgb8 out;
    out.width = dim.xsize;
    out.height = dim.ysize;
    out.pixels.resize(outBytes);
    std::memcpy(out.pixels.data(), words.data(), outBytes);
    return out;
}

}   // namespace sjxl::dec
