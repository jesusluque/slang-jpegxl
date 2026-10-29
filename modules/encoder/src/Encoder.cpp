// Copyright (c) 2026 jesus luque.
#include "sjxl/encoder/Encoder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>

#include "sjxl/codestream/Headers.h"
#include "sjxl/codestream/Tables.h"
#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include <format>
#include <cstdlib>
#include "sjxl/gpu/CommandBatch.h"

namespace sjxl::enc {
namespace {

using codestream::FrameDim;

constexpr uint32_t kMetaItemsPerGroup = 2 * 32 * 32 + 3 * 256 * 256;
constexpr uint32_t kCodeStride = 256;
constexpr uint32_t kBlobStrideWords = 8192;
constexpr uint32_t kAcClusters = 32;
constexpr uint32_t kNumAcContexts = 15 * (37 + 458);
constexpr uint32_t kHistBins = 128;
constexpr uint32_t kDequantTableFloats = 394752;
constexpr uint32_t kCoeffOrderLimit = 6156;

struct Uint4 {
    uint32_t x, y, z, w;
};

/// Group rects in blocks: x0, y0, width, height, raster order.
std::vector<Uint4> groupRects(const FrameDim& dim, uint32_t groupBlocks, uint32_t gx, uint32_t gy) {
    std::vector<Uint4> rects;
    for (uint32_t j = 0; j < gy; ++j) {
        for (uint32_t i = 0; i < gx; ++i) {
            const uint32_t x0 = i * groupBlocks;
            const uint32_t y0 = j * groupBlocks;
            rects.push_back({x0, y0, std::min(groupBlocks, dim.xsizeBlocks - x0), std::min(groupBlocks, dim.ysizeBlocks - y0)});
        }
    }
    return rects;
}

std::vector<uint32_t> fixedTree(uint32_t property, uint32_t predictor, uint64_t totalPixels, uint32_t& numLeaves);
std::vector<uint32_t> acMetaTree(uint32_t& numLeaves);
std::vector<uint32_t> combineTreesByChannel(const std::vector<const std::vector<uint32_t>*>& subtrees, std::vector<uint32_t>& leafMap,
                                            uint32_t& numLeaves);
constexpr uint32_t kSetContexts = 64;
constexpr uint32_t kModTreeWords = 8;
constexpr uint32_t kModPropGradient = 9;
constexpr uint32_t kModPropWp = 15;
constexpr uint32_t kPredZero = 0, kPredLeft = 1, kPredGradient = 5, kPredWeighted = 6;

/// LoopFilterFromParams (enc_frame.cc): Gaborish above distance 0.5, EPF
/// iterations by distance thresholds 0.7, 1.5 and 4. x_qm_scale 3 up to
/// distance 2.5 then 4, 5, 6 (the pixel-statistics adjustment is squirrel+).
codestream::FrameParams frameParamsFor(const Options& options) {
    codestream::FrameParams frame;
    frame.gaborish = options.loopFilter && options.distance > 0.5f;
    frame.epfIters = 0;
    if (options.loopFilter) {
        for (float threshold : {0.7f, 1.5f, 4.0f}) {
            if (options.distance >= threshold) ++frame.epfIters;
        }
    }
    frame.xQmScale = 3;
    for (float step : {2.5f, 5.5f, 9.5f}) {
        if (options.distance > step) ++frame.xQmScale;
    }
    frame.bQmScale = 2;
    return frame;
}

}   // namespace

Result<std::unique_ptr<Encoder>> Encoder::create(std::shared_ptr<gpu::ShaderLibrary> library) {
    std::unique_ptr<Encoder> enc(new Encoder());
    enc->library_ = std::move(library);
    enc->device_ = &enc->library_->device();
    gpu::ShaderLibrary& lib = *enc->library_;
    SJXL_ASSIGN(enc->rgbToXyb_, gpu::ComputeKernel::create(lib, "sjxl/enc/xyb", "rgbToXyb"));
    SJXL_ASSIGN(enc->aqDiff_, gpu::ComputeKernel::create(lib, "sjxl/enc/aq", "aqDiff"));
    SJXL_ASSIGN(enc->aqMap_, gpu::ComputeKernel::create(lib, "sjxl/enc/aq", "aqMap"));
    SJXL_ASSIGN(enc->aqRawField_, gpu::ComputeKernel::create(lib, "sjxl/enc/aq", "aqRawField"));
    SJXL_ASSIGN(enc->gaborishInverse_, gpu::ComputeKernel::create(lib, "sjxl/enc/gaborish", "gaborishInverse"));
    SJXL_ASSIGN(enc->tables_, gpu::ComputeKernel::create(lib, "sjxl/dec/tables", "computeDequantTable"));
    SJXL_ASSIGN(enc->buildOrders_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "buildOrders"));
    SJXL_ASSIGN(enc->selectStrategies_, gpu::ComputeKernel::create(lib, "sjxl/enc/strategy", "selectStrategies"));
    SJXL_ASSIGN(enc->largeQf_, gpu::ComputeKernel::create(lib, "sjxl/enc/large", "largeQf"));
    SJXL_ASSIGN(enc->largeRows_, gpu::ComputeKernel::create(lib, "sjxl/enc/large", "largeRows"));
    SJXL_ASSIGN(enc->largeCols_, gpu::ComputeKernel::create(lib, "sjxl/enc/large", "largeCols"));
    SJXL_ASSIGN(enc->mergeLarge_, gpu::ComputeKernel::create(lib, "sjxl/enc/large", "mergeLarge"));
    SJXL_ASSIGN(enc->adjustQuantField_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "adjustQuantField"));
    SJXL_ASSIGN(enc->dctRows_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "dctRows"));
    SJXL_ASSIGN(enc->dctCols_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "dctCols"));
    SJXL_ASSIGN(enc->dcFromLlf_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "dcFromLlf"));
    SJXL_ASSIGN(enc->specialTransform_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "specialTransform"));
    SJXL_ASSIGN(enc->acTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "acTokenize"));
    SJXL_ASSIGN(enc->cflTile_, gpu::ComputeKernel::create(lib, "sjxl/enc/cfl", "cflTile"));
    SJXL_ASSIGN(enc->quantize_, gpu::ComputeKernel::create(lib, "sjxl/enc/quantize", "quantize"));
    SJXL_ASSIGN(enc->dcTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "dcTokenize"));
    SJXL_ASSIGN(enc->metaTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "metaTokenize"));
    SJXL_ASSIGN(enc->dcGroupSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "dcGroupSections"));
    SJXL_ASSIGN(enc->clusterContexts_, gpu::ComputeKernel::create(lib, "sjxl/enc/cluster", "clusterContexts"));
    SJXL_ASSIGN(enc->accumulateClusters_, gpu::ComputeKernel::create(lib, "sjxl/enc/cluster", "accumulateClusters"));
    SJXL_ASSIGN(enc->acSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/ac_sections", "acSections"));
    SJXL_ASSIGN(enc->rctYCoCg_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "rctYCoCg"));
    SJXL_ASSIGN(enc->modTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modTokenize"));
    SJXL_ASSIGN(enc->channelEntropy_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "channelEntropy"));
    SJXL_ASSIGN(enc->remapTokens_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "remapTokens"));
    SJXL_ASSIGN(enc->modGroupSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modGroupSections"));
    SJXL_ASSIGN(enc->modularGlobalBlob_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildModularGlobalBlob"));
    SJXL_ASSIGN(enc->dcGroupBlobs_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildDcGroupBlobs"));
    SJXL_ASSIGN(enc->globalBlobs_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildGlobalBlobs"));
    SJXL_ASSIGN(enc->layout_, gpu::ComputeKernel::create(lib, "sjxl/enc/layout", "layoutSections"));
    SJXL_ASSIGN(enc->blobEmit_, gpu::ComputeKernel::create(lib, "sjxl/enc/layout", "blobEmit"));
    SJXL_ASSIGN(enc->prefixSum_, gpu::PrefixSum::create(lib));

    SJXL_ASSIGN(enc->dummy_, gpu::Buffer::ofType<uint32_t>(*enc->device_, 4, "enc.dummy"));
    SJXL_ASSIGN(enc->dummyInt_, gpu::Buffer::ofType<int32_t>(*enc->device_, 4, "enc.dummyInt"));

    // libjxl's default dequant tables for every transform kind, from the
    // decoder's table kernel over a globals region whose encodings are all
    // "library" (zero), and the natural coefficient orders of every bucket.
    gpu::Device& dev = *enc->device_;
    std::vector<uint32_t> stubArena(4096, 0);
    stubArena[44] = 64;   // kGsQuantEncOff -> a zeroed region: every kind takes the library default
    SJXL_ASSIGN(auto arena, gpu::Buffer::fromSpan<uint32_t>(dev, stubArena, "enc.stubArena"));
    SJXL_ASSIGN(enc->dequantAll_, gpu::Buffer::ofType<float>(dev, kDequantTableFloats, "enc.dequantAll"));
    SJXL_ASSIGN(enc->orders_, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kCoeffOrderLimit} * 64, "enc.orders"));
    gpu::CommandBatch setup(dev);
    enc->tables_.dispatch(setup, {kDequantTableFloats, 1, 1}, [&](rhi::ShaderCursor c) {
        c["arena"].setBinding(arena.rhi());
        c["dequant"].setBinding(enc->dequantAll_.rhi());
        c["params"]["totalEntries"].setData(kDequantTableFloats);
    });
    enc->buildOrders_.dispatch(setup, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        c["orders"].setBinding(enc->orders_.rhi());
        c["xyb"].setBinding(enc->dummy_.rhi());
        c["acs"].setBinding(enc->dummy_.rhi());
        c["firstBlock"].setBinding(enc->dummy_.rhi());
        c["rowPass"].setBinding(enc->dummy_.rhi());
        c["coeffs"].setBinding(enc->dummy_.rhi());
        c["dcValue"].setBinding(enc->dummy_.rhi());
        c["rawQf"].setBinding(enc->dummy_.rhi());
        c["acQuant"].setBinding(enc->dummyInt_.rhi());
        c["nonZeros"].setBinding(enc->dummy_.rhi());
        c["acTokens"].setBinding(enc->dummy_.rhi());
        c["numTokens"].setBinding(enc->dummy_.rhi());
        c["nzCtx"].setBinding(enc->dummy_.rhi());
        c["ctxHist"].setBinding(enc->dummy_.rhi());
        c["params"]["blocksX"].setData(0u);
    });
    SJXL_TRY(setup.submit(true));
    return enc;
}

Result<std::vector<uint8_t>> Encoder::encode(const image::Rgb8& image, const Options& options) {
    if (image.width == 0 || image.height == 0 || (image.channels != 3 && image.channels != 4) ||
        (image.bitDepth != 8 && image.bitDepth != 16) ||
        image.pixels.size() != size_t{image.width} * image.height * image.channels * (image.bitDepth / 8)) {
        return Error(ErrorCode::InvalidArgument, "empty image or pixel count does not match its size");
    }
    if (options.lossless) {
        return encodeLossless(image, options);
    }
    if (image.channels != 3 || image.bitDepth != 8) {
        return Error(ErrorCode::Unsupported, "lossy encoding of alpha or 16-bit input (use --lossless)");
    }
    if (!(options.distance > 0.0f) || options.distance > 25.0f) {
        return Error::make(ErrorCode::InvalidArgument, "distance {} is not in (0, 25]", options.distance);
    }
    gpu::Device& dev = *device_;
    const FrameDim dim = codestream::frameDim(image.width, image.height);
    const codestream::FrameParams frame = frameParamsFor(options);
    // The quant field's distance: libjxl lowers it when the Gaborish is off.
    const float fieldDistance = options.distance * (frame.gaborish ? 1.0f : 0.62f);
    // The field's unit: libjxl's adaptive scale, or the flat 0.79 / d of its
    // fast tiers. The quantizer's global scale follows 0.39 / d with the
    // adaptive field (enc_heuristics.cc, speeds up to squirrel), 0.79 / d flat.
    const float fieldScale = options.adaptiveQuant ? codestream::initialQuantAc(fieldDistance) : 0.79f / options.distance;
    const codestream::QuantParams qp = codestream::quantParamsFor(options.adaptiveQuant ? 0.39f / options.distance : fieldScale,
                                                                  codestream::initialQuantDc(options.distance));
    const std::array<uint32_t, 4> quantizerWords{qp.globalScale, qp.quantDc, std::bit_cast<uint32_t>(qp.invGlobalScale),
                                                 std::bit_cast<uint32_t>(qp.globalScaleFloat)};
    const uint32_t nblocks = dim.numBlocks();
    const uint32_t tilesX = (dim.xsizeBlocks + 7) / 8;
    const uint32_t tilesY = (dim.ysizeBlocks + 7) / 8;
    const uint32_t cellsX = dim.xsizePadded / 4;
    const uint32_t cellsY = dim.ysizePadded / 4;
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const uint32_t numSections = dim.numSections();

    // Input, padded to whole words.
    std::vector<uint32_t> rgbWords((image.pixels.size() + 3) / 4, 0);
    std::memcpy(rgbWords.data(), image.pixels.data(), image.pixels.size());
    SJXL_ASSIGN(auto rgb, gpu::Buffer::fromSpan<uint32_t>(dev, rgbWords, "enc.rgb8"));

    const uint64_t planePixels = uint64_t{dim.xsizePadded} * dim.ysizePadded;
    SJXL_ASSIGN(auto xyb, gpu::Buffer::ofType<float>(dev, 3 * planePixels, "enc.xyb"));
    SJXL_ASSIGN(auto xybSharp, gpu::Buffer::ofType<float>(dev, frame.gaborish ? 3 * planePixels : 4, "enc.xybSharp"));
    SJXL_ASSIGN(auto preErosion, gpu::Buffer::ofType<float>(dev, uint64_t{cellsX} * cellsY, "enc.preErosion"));
    SJXL_ASSIGN(auto qfField, gpu::Buffer::ofType<float>(dev, nblocks, "enc.qf"));
    SJXL_ASSIGN(auto quantizer, gpu::Buffer::fromSpan<uint32_t>(dev, quantizerWords, "enc.quantizer"));
    SJXL_ASSIGN(auto rawQf, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "enc.rawQf"));
    SJXL_ASSIGN(auto acs, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "enc.acs"));
    SJXL_ASSIGN(auto firstBlock, gpu::Buffer::ofType<uint32_t>(dev, nblocks, "enc.firstBlock"));
    SJXL_ASSIGN(auto rowPass, gpu::Buffer::ofType<float>(dev, 3 * planePixels, "enc.rowPass"));
    SJXL_ASSIGN(auto coeffs, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks * 64, "enc.coeffs"));
    SJXL_ASSIGN(auto dcValue, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks, "enc.dcValue"));
    SJXL_ASSIGN(auto yRecon, gpu::Buffer::ofType<float>(dev, uint64_t{nblocks} * 64, "enc.yRecon"));
    SJXL_ASSIGN(auto acTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks * 64, "enc.acTokens"));
    SJXL_ASSIGN(auto numTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.numTokens"));
    SJXL_ASSIGN(auto nzCtx, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.nzCtx"));
    SJXL_ASSIGN(auto cmapX, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapX"));
    SJXL_ASSIGN(auto cmapB, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapB"));
    SJXL_ASSIGN(auto acQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks * 64, "enc.acQuant"));
    SJXL_ASSIGN(auto dcQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks, "enc.dcQuant"));
    SJXL_ASSIGN(auto nonZeros, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.nonZeros"));

    const std::vector<Uint4> acRects = groupRects(dim, codestream::kGroupDimBlocks, dim.xsizeGroups, dim.ysizeGroups);
    const std::vector<Uint4> dcRects = groupRects(dim, codestream::kDcGroupDimBlocks, dim.xsizeDcGroups, dim.ysizeDcGroups);
    SJXL_ASSIGN(auto groups, gpu::Buffer::fromSpan<Uint4>(dev, acRects, "enc.groups"));
    SJXL_ASSIGN(auto dcGroups, gpu::Buffer::fromSpan<Uint4>(dev, dcRects, "enc.dcGroups"));

    // Histogram sets: 0 the AC contexts, 1 + 2g the DC stream of DC group g, 2 + 2g its AcMetadata.
    const uint32_t numSets = 1 + 2 * numDcGroups;
    const uint32_t numContextsAll = kNumAcContexts + 2 * numDcGroups * kSetContexts;
    const uint32_t numCodeRegions = numSets * kAcClusters;
    const uint32_t numBlobRegions = 2 + 2 * numDcGroups;
    SJXL_ASSIGN(auto hist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.hist"));
    SJXL_ASSIGN(auto codes, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.codes"));
    SJXL_ASSIGN(auto blobs, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numBlobRegions} * kBlobStrideWords, "enc.blobs"));
    SJXL_ASSIGN(auto blobBits, gpu::Buffer::ofType<uint32_t>(dev, numBlobRegions, "enc.blobBits"));
    SJXL_ASSIGN(auto ctxHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numContextsAll} * kHistBins, "enc.ctxHist"));
    SJXL_ASSIGN(auto ctxMap, gpu::Buffer::ofType<uint32_t>(dev, numContextsAll, "enc.ctxMap"));
    SJXL_ASSIGN(auto ctxTotal, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxTotal"));
    SJXL_ASSIGN(auto ctxEntropy, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxEntropy"));
    SJXL_ASSIGN(auto ctxDist, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxDist"));
    SJXL_ASSIGN(auto ctxSingle, gpu::Buffer::ofType<uint32_t>(dev, kNumAcContexts, "enc.ctxSingle"));
    SJXL_ASSIGN(auto clusterInfo, gpu::Buffer::ofType<uint32_t>(dev, numSets, "enc.clusterInfo"));
    SJXL_ASSIGN(auto ctxMapMtf, gpu::Buffer::ofType<uint32_t>(dev, numContextsAll, "enc.ctxMapMtf"));
    SJXL_ASSIGN(auto ansInfo, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions + 1} * 256, "enc.ansInfo"));
    SJXL_ASSIGN(auto ansReverse, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions + 1} * 4096, "enc.ansReverse"));
    // The DC group streams: libjxl's kWPFixedDC tree (weighted predictor) and kACMeta tree, the WP scratch, the tokens.
    uint32_t dcLeaves = 0, metaLeaves = 0;
    const std::vector<uint32_t> dcTreeWords = fixedTree(kModPropWp, kPredWeighted, uint64_t{3} * nblocks, dcLeaves);
    const std::vector<uint32_t> metaTreeWords = acMetaTree(metaLeaves);
    SJXL_ASSIGN(auto dcTree, gpu::Buffer::fromSpan<uint32_t>(dev, dcTreeWords, "enc.dcTree"));
    SJXL_ASSIGN(auto metaTree, gpu::Buffer::fromSpan<uint32_t>(dev, metaTreeWords, "enc.metaTree"));
    const uint32_t wpWordsPerGroup = 5 * 2 * (codestream::kDcGroupDimBlocks + 2);
    SJXL_ASSIGN(auto wpScratch, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numDcGroups} * wpWordsPerGroup, "enc.wpScratch"));
    SJXL_ASSIGN(auto dcTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.dcTokens"));
    SJXL_ASSIGN(auto acSectionBits, gpu::Buffer::ofType<uint32_t>(dev, numGroups, "enc.acSectionBits"));

    const uint32_t metaItems = numDcGroups * kMetaItemsPerGroup;
    SJXL_ASSIGN(auto metaTokens, gpu::Buffer::ofType<uint32_t>(dev, metaItems, "enc.metaTokens"));
    SJXL_ASSIGN(auto dcBits, gpu::Buffer::ofType<uint32_t>(dev, numDcGroups, "enc.dcBits"));
    SJXL_ASSIGN(auto metaBits, gpu::Buffer::ofType<uint32_t>(dev, numDcGroups, "enc.metaBits"));

    SJXL_ASSIGN(auto sectionStart, gpu::Buffer::ofType<uint32_t>(dev, numSections, "enc.sectionStart"));
    SJXL_ASSIGN(auto sectionBits, gpu::Buffer::ofType<uint32_t>(dev, numSections, "enc.sectionBits"));
    SJXL_ASSIGN(auto sectionBytes, gpu::Buffer::ofType<uint32_t>(dev, numSections, "enc.sectionBytes"));
    SJXL_ASSIGN(auto dcPartStart, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{4} * numDcGroups, "enc.dcPartStart"));
    SJXL_ASSIGN(auto totals, gpu::Buffer::ofType<uint32_t>(dev, 2, "enc.totals"));

    gpu::CommandBatch batch(dev);

    rgbToXyb_.dispatch(batch, {dim.xsizePadded, dim.ysizePadded, 1}, [&](rhi::ShaderCursor c) {
        c["rgb8"].setBinding(rgb.rhi());
        c["xyb"].setBinding(xyb.rhi());
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["transfer"].setData(uint32_t{image.gamma > 0.0f ? 2u : 0u});
        c["params"]["gammaInv"].setData(image.gamma > 0.0f ? 1.0f / image.gamma : 1.0f);
    });

    // The quant field (from the planes before the inverse Gaborish), then the
    // integer field under the quantizer's global scale.
    const auto bindAq = [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(xyb.rhi());
        c["preErosion"].setBinding(preErosion.rhi());
        c["qf"].setBinding(qfField.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["cellsX"].setData(cellsX);
        c["params"]["cellsY"].setData(cellsY);
        c["params"]["uniform"].setData(uint32_t{options.adaptiveQuant ? 0u : 1u});
        c["params"]["distance"].setData(fieldDistance);
        c["params"]["scale"].setData(fieldScale);
    };
    if (options.adaptiveQuant) {
        aqDiff_.dispatch(batch, {cellsX * cellsY, 1, 1}, bindAq);
    }
    aqMap_.dispatch(batch, {nblocks, 1, 1}, bindAq);
    aqRawField_.dispatch(batch, {nblocks, 1, 1}, bindAq);

    // Pre-sharpening for the decoder's Gaborish, then the DCT of every block.
    const gpu::Buffer& planes = frame.gaborish ? xybSharp : xyb;
    if (frame.gaborish) {
        gaborishInverse_.dispatch(batch, {dim.xsizePadded, dim.ysizePadded, 1}, [&](rhi::ShaderCursor c) {
            c["src"].setBinding(xyb.rhi());
            c["dst"].setBinding(xybSharp.rhi());
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
        });
    }
    // The strategy of every block, the field per varblock, the
    // varblock DCTs, their DC, chroma-from-luma, quantisation, and the AC tokens
    // with their context histograms.
    const uint32_t superblocksX = (dim.xsizeBlocks + 3) / 4;
    const uint32_t superblocksY = (dim.ysizeBlocks + 3) / 4;
    const uint32_t regionsX = (dim.xsizeBlocks + 7) / 8;
    const uint32_t regionsY = (dim.ysizeBlocks + 7) / 8;
    SJXL_ASSIGN(auto sbCost, gpu::Buffer::ofType<float>(dev, uint64_t{superblocksX} * superblocksY, "enc.sbCost"));
    SJXL_ASSIGN(auto candQf, gpu::Buffer::ofType<float>(dev, uint64_t{regionsX} * regionsY * 5, "enc.candQf"));
    SJXL_ASSIGN(auto candBits, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regionsX} * regionsY * 5, "enc.candBits"));
    SJXL_ASSIGN(auto candDist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regionsX} * regionsY * 5, "enc.candDist"));
    SJXL_ASSIGN(auto candNz, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regionsX} * regionsY * 5, "enc.candNz"));
    uint32_t familyMask = 0x3FF;
    if (const std::string m = platform::env("SJXL_ACS_MASK"); !m.empty()) familyMask = static_cast<uint32_t>(std::strtoul(m.c_str(), nullptr, 0));
    selectStrategies_.dispatchGroups(batch, {superblocksX * superblocksY, 1, 1}, [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(planes.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["dequant"].setBinding(dequantAll_.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["acs"].setBinding(acs.rhi());
        c["firstBlock"].setBinding(firstBlock.rhi());
        c["sbCost"].setBinding(sbCost.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["superblocksX"].setData(superblocksX);
        c["params"]["lambda"].setData(options.acsLambda);
        c["params"]["distance"].setData(options.distance);
        c["params"]["enabled"].setData(uint32_t{options.acStrategy ? 1u : 0u});
        c["params"]["familyMask"].setData(familyMask & 0x7Fu);
        c["params"]["specials"].setData(uint32_t{options.acsSpecials ? 1u : 0u});
    });
    // The 64-pixel families against the superblocks' layouts.
    if (options.acStrategy && (familyMask >> 7) != 0) {
        const auto bindLarge = [&](rhi::ShaderCursor c, uint32_t family) {
            c["xyb"].setBinding(planes.rhi());
            c["rawQf"].setBinding(rawQf.rhi());
            c["dequant"].setBinding(dequantAll_.rhi());
            c["quantizer"].setBinding(quantizer.rhi());
            c["sbCost"].setBinding(sbCost.rhi());
            c["rowPass"].setBinding(rowPass.rhi());
            c["candQf"].setBinding(candQf.rhi());
            c["candBits"].setBinding(candBits.rhi());
            c["candDist"].setBinding(candDist.rhi());
            c["candNz"].setBinding(candNz.rhi());
            c["acs"].setBinding(acs.rhi());
            c["firstBlock"].setBinding(firstBlock.rhi());
            c["params"]["blocksX"].setData(dim.xsizeBlocks);
            c["params"]["blocksY"].setData(dim.ysizeBlocks);
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
            c["params"]["regionsX"].setData(regionsX);
            c["params"]["family"].setData(family);
            c["params"]["lambda"].setData(options.acsLambda);
            c["params"]["distance"].setData(options.distance);
            c["params"]["largeMask"].setData(familyMask >> 7);
        };
        largeQf_.dispatch(batch, {regionsX * regionsY * 5, 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, 0); });
        for (uint32_t family = 0; family < 3; ++family) {
            largeRows_.dispatch(batch, {static_cast<uint32_t>(planePixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, family); });
            largeCols_.dispatch(batch, {static_cast<uint32_t>(planePixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, family); });
        }
        mergeLarge_.dispatch(batch, {regionsX * regionsY, 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, 0); });
    }
    const auto bindTransform = [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(planes.rhi());
        c["acs"].setBinding(acs.rhi());
        c["firstBlock"].setBinding(firstBlock.rhi());
        c["orders"].setBinding(orders_.rhi());
        c["rowPass"].setBinding(rowPass.rhi());
        c["coeffs"].setBinding(coeffs.rhi());
        c["dcValue"].setBinding(dcValue.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["acTokens"].setBinding(acTokens.rhi());
        c["numTokens"].setBinding(numTokens.rhi());
        c["nzCtx"].setBinding(nzCtx.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["distance"].setData(options.distance);
    };
    adjustQuantField_.dispatch(batch, {nblocks, 1, 1}, bindTransform);
    // Oracle experiment: another encoder's quant field and strategy map (as the
    // decoder's SJXL_DEBUG_MAPS dumps them) replace ours.
    const std::string oracleMaps = platform::env("SJXL_ENC_MAPS");
    gpu::Buffer oracleQf, oracleAcs, oracleFirst;
    if (!oracleMaps.empty()) {
        SJXL_ASSIGN(const auto bytes, image::readFile(oracleMaps));
        if (bytes.size() != size_t{nblocks} * 8) return Error(ErrorCode::InvalidArgument, "SJXL_ENC_MAPS: wrong size");
        std::vector<uint32_t> qfMap(nblocks), acsMap(nblocks), firstMap(nblocks);
        std::memcpy(qfMap.data(), bytes.data(), nblocks * 4);
        std::memcpy(acsMap.data(), bytes.data() + nblocks * 4, nblocks * 4);
        for (uint32_t b = 0; b < nblocks; ++b) {
            if (acsMap[b] & 1) {
                const uint32_t st = acsMap[b] >> 1;
                static constexpr uint32_t kCx[27] = {1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32};
                static constexpr uint32_t kCy[27] = {1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16};
                for (uint32_t y = 0; y < kCy[st]; ++y)
                    for (uint32_t x = 0; x < kCx[st]; ++x) firstMap[b + y * dim.xsizeBlocks + x] = b;
            }
        }
        SJXL_ASSIGN(oracleQf, gpu::Buffer::fromSpan<uint32_t>(dev, qfMap, "enc.oracleQf"));
        SJXL_ASSIGN(oracleAcs, gpu::Buffer::fromSpan<uint32_t>(dev, acsMap, "enc.oracleAcs"));
        SJXL_ASSIGN(oracleFirst, gpu::Buffer::fromSpan<uint32_t>(dev, firstMap, "enc.oracleFirst"));
        batch.copy(oracleQf, 0, rawQf, 0, uint64_t{nblocks} * 4);
        batch.copy(oracleAcs, 0, acs, 0, uint64_t{nblocks} * 4);
        batch.copy(oracleFirst, 0, firstBlock, 0, uint64_t{nblocks} * 4);
    }
    dctRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
    dctCols_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
    specialTransform_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransform);
    dcFromLlf_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransform);
    if (options.chromaFromLuma) {
        cflTile_.dispatch(batch, {tilesX * tilesY, 1, 1}, [&](rhi::ShaderCursor c) {
            c["coeffs"].setBinding(coeffs.rhi());
            c["dequant"].setBinding(dequantAll_.rhi());
            c["acs"].setBinding(acs.rhi());
            c["rawQf"].setBinding(rawQf.rhi());
            c["quantizer"].setBinding(quantizer.rhi());
            c["cmapX"].setBinding(cmapX.rhi());
            c["cmapB"].setBinding(cmapB.rhi());
            c["params"]["blocksX"].setData(dim.xsizeBlocks);
            c["params"]["blocksY"].setData(dim.ysizeBlocks);
            c["params"]["tilesX"].setData(tilesX);
            c["params"]["tilesY"].setData(tilesY);
        });
    }
    quantize_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
        c["coeffs"].setBinding(coeffs.rhi());
        c["dcValue"].setBinding(dcValue.rhi());
        c["dequant"].setBinding(dequantAll_.rhi());
        c["acs"].setBinding(acs.rhi());
        c["firstBlock"].setBinding(firstBlock.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
        c["yRecon"].setBinding(yRecon.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["xQmMultiplier"].setData(static_cast<float>(std::pow(1.25, static_cast<int>(frame.xQmScale) - 2)));
        c["params"]["bQmMultiplier"].setData(static_cast<float>(std::pow(1.25, static_cast<int>(frame.bQmScale) - 2)));
        c["params"]["noAdjust"].setData(uint32_t{platform::env("SJXL_ENC_NO_ADJUST").empty() ? 0u : 1u});
    });
    acTokenize_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransform);

    // The DC group streams' tokens: the DC with the weighted predictor (one
    // thread per group, as the error state is serial) and the AcMetadata per item.
    const auto bindDc = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& body) {
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["acs"].setBinding(acs.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
        c["dcTree"].setBinding(dcTree.rhi());
        c["metaTree"].setBinding(metaTree.rhi());
        c["wpScratch"].setBinding(wpScratch.rhi());
        c["dcTokens"].setBinding(dcTokens.rhi());
        c["metaTokens"].setBinding(metaTokens.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["dcPartStart"].setBinding(dcPartStart.rhi());
        c["dcBits"].setBinding(dcBits.rhi());
        c["metaBits"].setBinding(metaBits.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["nblocks"].setData(nblocks);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["mode"].setData(mode);
        c["params"]["dcTreeNodes"].setData(static_cast<uint32_t>(dcTreeWords.size() / kModTreeWords));
        c["params"]["metaTreeNodes"].setData(static_cast<uint32_t>(metaTreeWords.size() / kModTreeWords));
        c["params"]["wpWordsPerGroup"].setData(wpWordsPerGroup);
    };
    dcTokenize_.dispatch(batch, {numDcGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindDc(c, 0, dummy_); });
    metaTokenize_.dispatch(batch, {metaItems, 1, 1}, [&](rhi::ShaderCursor c) { bindDc(c, 0, dummy_); });
    const auto bindSections = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& body) {
        c["groups"].setBinding(groups.rhi());
        c["acs"].setBinding(acs.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["acTokens"].setBinding(acTokens.rhi());
        c["numTokens"].setBinding(numTokens.rhi());
        c["nzCtx"].setBinding(nzCtx.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["sectionStart"].setBinding(sectionStart.rhi());
        c["acSectionBits"].setBinding(acSectionBits.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["nblocks"].setData(nblocks);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["mode"].setData(mode);
    };
    // The ANS sections (one thread per AC group, one per DC group) measured in mode 1 and emitted in mode 2.
    const auto sectionPasses = [&](uint32_t mode, const gpu::Buffer& body) {
        acSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindSections(c, mode, body); });
        dcGroupSections_.dispatch(batch, {numDcGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindDc(c, mode, body); });
    };

    // Every histogram set's contexts clustered into its codes' histograms.
    const auto bindCluster = [&](rhi::ShaderCursor c, uint32_t set, uint32_t numContexts) {
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxTotal"].setBinding(ctxTotal.rhi());
        c["ctxEntropy"].setBinding(ctxEntropy.rhi());
        c["ctxDist"].setBinding(ctxDist.rhi());
        c["ctxSingle"].setBinding(ctxSingle.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["hist"].setBinding(hist.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["params"]["numContexts"].setData(numContexts);
        c["params"]["maxClusters"].setData(std::min<uint32_t>(options.acClusters, kAcClusters));
        c["params"]["ctxBase"].setData(set == 0 ? 0u : kNumAcContexts + (set - 1) * kSetContexts);
        c["params"]["infoIndex"].setData(set);
        c["params"]["regionBase"].setData(set * kAcClusters);
    };
    for (uint32_t set = 0; set < numSets; ++set) {
        const uint32_t numContexts = set == 0 ? kNumAcContexts : kSetContexts;
        clusterContexts_.dispatchGroups(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindCluster(c, set, numContexts); });
        accumulateClusters_.dispatch(batch, {numContexts, 1, 1}, [&](rhi::ShaderCursor c) { bindCluster(c, set, numContexts); });
    }

    const auto bindCodes = [&](rhi::ShaderCursor c) {
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["acs"].setBinding(acs.rhi());
        c["hist"].setBinding(hist.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["ctxMapMtf"].setBinding(ctxMapMtf.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["planes"].setBinding(dummyInt_.rhi());
        c["treeNodes"].setBinding(dummy_.rhi());
        c["modTokens"].setBinding(dummy_.rhi());
        c["dcTree"].setBinding(dcTree.rhi());
        c["metaTree"].setBinding(metaTree.rhi());
        c["codes"].setBinding(codes.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["blobBits"].setBinding(blobBits.rhi());
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["dcTreeNodes"].setData(static_cast<uint32_t>(dcTreeWords.size() / kModTreeWords));
        c["params"]["dcLeaves"].setData(dcLeaves);
        c["params"]["metaTreeNodes"].setData(static_cast<uint32_t>(metaTreeWords.size() / kModTreeWords));
        c["params"]["metaLeaves"].setData(metaLeaves);
        c["params"]["numAcContexts"].setData(kNumAcContexts);
    };
    dcGroupBlobs_.dispatchGroups(batch, {numDcGroups, 1, 1}, bindCodes);
    globalBlobs_.dispatchGroups(batch, {1, 1, 1}, bindCodes);

    sectionPasses(1, dummy_);   // bit counts

    const auto bindLayout = [&](rhi::ShaderCursor c, const gpu::Buffer& body) {
        c["blobBits"].setBinding(blobBits.rhi());
        c["acSectionBits"].setBinding(acSectionBits.rhi());
        c["dcBits"].setBinding(dcBits.rhi());
        c["metaBits"].setBinding(metaBits.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["sectionStart"].setBinding(sectionStart.rhi());
        c["sectionBits"].setBinding(sectionBits.rhi());
        c["sectionBytes"].setBinding(sectionBytes.rhi());
        c["dcPartStart"].setBinding(dcPartStart.rhi());
        c["totals"].setBinding(totals.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["packed"].setData(uint32_t{dim.singleSection() ? 1u : 0u});
    };
    layout_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindLayout(c, dummy_); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> clusterInfoHost, clusterInfo.readAll<uint32_t>(dev));
    stats_.numAcClusters = clusterInfoHost[0];
    stats_.globalScale = qp.globalScale;
    stats_.quantDc = qp.quantDc;
    stats_.gaborish = frame.gaborish;
    stats_.epfIters = frame.epfIters;
    if (!platform::env("SJXL_DEBUG").empty()) {
        SJXL_ASSIGN(const auto rawQfHost, rawQf.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto cmapXHost, cmapX.readAll<int32_t>(dev));
        SJXL_ASSIGN(const auto cmapBHost, cmapB.readAll<int32_t>(dev));
        uint32_t qfMin = 256, qfMax = 0;
        for (uint32_t v : rawQfHost) { qfMin = std::min(qfMin, v); qfMax = std::max(qfMax, v); }
        std::string cm;
        for (size_t i = 0; i < std::min<size_t>(cmapXHost.size(), 12); ++i) cm += std::format("({},{}) ", cmapXHost[i], cmapBHost[i]);
        log::info("enc debug: global_scale {} quant_dc {} raw qf [{}, {}] gab {} epf {} xqm {} ac clusters {} cmap(x,b) {}", qp.globalScale,
                  qp.quantDc, qfMin, qfMax, frame.gaborish, frame.epfIters, frame.xQmScale, clusterInfoHost[0], cm);
        SJXL_ASSIGN(const auto acsHost, acs.readAll<uint32_t>(dev));
        std::array<uint32_t, 27> strategyCount{};
        uint32_t firsts = 0;
        for (uint32_t a : acsHost) {
            if (a & 1) { ++firsts; if ((a >> 1) < 27) ++strategyCount[a >> 1]; }
        }
        std::string sc;
        for (uint32_t i = 0; i < 27; ++i) if (strategyCount[i]) sc += std::format("s{}:{} ", i, strategyCount[i]);
        log::info("enc debug: {} varblocks of {} blocks: {}", firsts, nblocks, sc);
        for (uint32_t by = 0; by < std::min<uint32_t>(dim.ysizeBlocks, 8); ++by) {
            std::string row;
            for (uint32_t bx = 0; bx < std::min<uint32_t>(dim.xsizeBlocks, 40); ++bx) {
                const uint32_t a = acsHost[by * dim.xsizeBlocks + bx];
                row += (a & 1) ? std::format("{:x} ", a >> 1) : ". ";
            }
            log::info("enc debug: acs row {:3} {}", by, row);
        }
        SJXL_ASSIGN(const auto blobBitsHost, blobBits.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto sizesHost, sectionBytes.readAll<uint32_t>(dev));
        std::string sections;
        for (uint32_t b : sizesHost) sections += std::format("{} ", b);
        log::info("enc debug: blob bits DcGlobal {} AcGlobal {}; section bytes {}", blobBitsHost[0], blobBitsHost[1], sections);
        SJXL_ASSIGN(const auto partsHost, dcPartStart.readAll<uint32_t>(dev));
        uint64_t dcTokenBits = 0, metaTokenBits = 0, headBits = 0;
        for (uint32_t g = 0; g < numDcGroups; ++g) {
            const uint32_t pre = partsHost[g * 4], dcs = partsHost[g * 4 + 1], mid = partsHost[g * 4 + 2], meta = partsHost[g * 4 + 3];
            headBits += (dcs - pre) + (meta - mid);
            dcTokenBits += mid - dcs;
            metaTokenBits += sizesHost[1 + g] * 8 - (meta - pre);
        }
        log::info("enc debug: DC groups: headers {} B, DC tokens {} B, AcMetadata tokens {} B", headBits / 8, dcTokenBits / 8, metaTokenBits / 8);
    }

    SJXL_ASSIGN(const std::vector<uint32_t> totalsHost, totals.readAll<uint32_t>(dev));
    SJXL_ASSIGN(std::vector<uint32_t> sizes, sectionBytes.readAll<uint32_t>(dev));
    const uint32_t bodyBytes = totalsHost[0];
    if (bodyBytes == 0) {
        return Error(ErrorCode::InternalError, "the layout produced an empty body");
    }

    // The body: zero, then every section and blob ORs its bits in.
    SJXL_ASSIGN(auto body, gpu::Buffer::ofType<uint32_t>(dev, (bodyBytes + 3) / 4 + 1, "enc.body"));
    sectionPasses(2, body);
    blobEmit_.dispatch(batch, {numBlobRegions, 1, 1}, [&](rhi::ShaderCursor c) { bindLayout(c, body); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> bodyWords, body.readAll<uint32_t>(dev));

    // Headers and TOC on the CPU, then the body.
    codestream::BitWriter head;
    codestream::writeCodestreamHeaders(head, image.width, image.height, image.gamma);
    codestream::writeFrameHeader(head, frame);
    if (dim.singleSection()) {
        sizes.assign(1, bodyBytes);
    }
    codestream::writeToc(head, sizes);
    std::vector<uint8_t> out = head.take();
    stats_.headBytes = static_cast<uint32_t>(out.size());
    stats_.bodyBytes = bodyBytes;
    stats_.sectionBytes = sizes;
    const auto* bodyBytesPtr = reinterpret_cast<const uint8_t*>(bodyWords.data());
    out.insert(out.end(), bodyBytesPtr, bodyBytesPtr + bodyBytes);
    return out;
}

// --- lossless --------------------------------------------------------------------------

namespace {


/// libjxl MakeFixedTree(property, cutoffs, predictor): a balanced binary
/// search over one property, shallower for small images, in the
/// breadth-first order DecodeTree expects. Returns the nodes as words and the
/// leaf count.
std::vector<uint32_t> fixedTree(uint32_t property, uint32_t predictor, uint64_t totalPixels, uint32_t& numLeaves) {
    static constexpr int32_t kCutoffs[] = {-500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0,
                                           1, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500};
    constexpr size_t kNumCutoffs = sizeof(kCutoffs) / sizeof(kCutoffs[0]);
    uint32_t logPx = 0;
    while ((uint64_t{1} << logPx) < totalPixels) ++logPx;
    const size_t minGap = logPx < 14 ? 8 * (14 - logPx) : 0;
    struct Node {
        int32_t prop = -1, split = 0;
        uint32_t left = 0, right = 0;
    };
    std::vector<Node> nodes(1);
    struct Pending {
        size_t begin, end, pos;
    };
    std::vector<Pending> queue{{0, kNumCutoffs, 0}};
    for (size_t head = 0; head < queue.size(); ++head) {
        const Pending info = queue[head];
        if (info.begin + minGap >= info.end) continue;
        const size_t split = (info.begin + info.end) / 2;
        nodes[info.pos] = Node{static_cast<int32_t>(property), kCutoffs[split], static_cast<uint32_t>(nodes.size()),
                               static_cast<uint32_t>(nodes.size() + 1)};
        queue.push_back({split + 1, info.end, nodes.size()});
        nodes.emplace_back();
        queue.push_back({info.begin, split, nodes.size()});
        nodes.emplace_back();
    }
    std::vector<uint32_t> words(nodes.size() * kModTreeWords, 0);
    numLeaves = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        uint32_t* n = &words[i * kModTreeWords];
        n[0] = static_cast<uint32_t>(nodes[i].prop);
        n[1] = static_cast<uint32_t>(nodes[i].split);
        n[2] = nodes[i].left;
        n[3] = nodes[i].right;
        n[4] = nodes[i].prop < 0 ? numLeaves++ : 0;
        n[5] = predictor;
        n[6] = 0;
    }
    return words;
}

std::vector<uint32_t> fixedGradientTree(uint64_t totalPixels, uint32_t& numLeaves) {
    return fixedTree(kModPropGradient, kPredGradient, totalPixels, numLeaves);
}

/// One tree per channel joined under a chain of splits on property 0 (the
/// channel): "c > k" sends channels above k on, else to channel k's subtree.
/// Re-serialised breadth-first for DecodeTree; leafMap[c * 64 + j] gives the
/// combined leaf of subtree c's leaf j (its own breadth-first id).
std::vector<uint32_t> combineTreesByChannel(const std::vector<const std::vector<uint32_t>*>& subtrees, std::vector<uint32_t>& leafMap,
                                            uint32_t& numLeaves) {
    struct N {
        int32_t prop = -1, split = 0;
        uint32_t left = 0, right = 0;
        uint32_t predictor = 0;
        int32_t offset = 0;
        uint32_t channel = 0, leaf = 0;   // for leaves: the subtree and its leaf id
    };
    std::vector<N> table;
    const size_t nc = subtrees.size();
    // The chain: node k (k < nc - 1) tests c > k; its right child is subtree k, its left the next chain node (or subtree nc - 1).
    std::vector<uint32_t> chain(nc - 1);
    for (size_t k = 0; k + 1 < nc; ++k) {
        chain[k] = static_cast<uint32_t>(table.size());
        table.push_back(N{0, static_cast<int32_t>(k), 0, 0, 0, 0, 0, 0});
    }
    std::vector<uint32_t> subtreeRoot(nc);
    for (size_t c = 0; c < nc; ++c) {
        const std::vector<uint32_t>& t = *subtrees[c];
        const uint32_t base = static_cast<uint32_t>(table.size());
        subtreeRoot[c] = base;
        for (size_t i = 0; i < t.size() / kModTreeWords; ++i) {
            const uint32_t* w = &t[i * kModTreeWords];
            N n;
            n.prop = static_cast<int32_t>(w[0]);
            n.split = static_cast<int32_t>(w[1]);
            n.left = base + w[2];
            n.right = base + w[3];
            n.predictor = w[5];
            n.offset = static_cast<int32_t>(w[6]);
            n.channel = static_cast<uint32_t>(c);
            n.leaf = w[4];
            table.push_back(n);
        }
    }
    for (size_t k = 0; k + 1 < nc; ++k) {
        table[chain[k]].right = subtreeRoot[k];
        table[chain[k]].left = k + 2 < nc ? chain[k + 1] : subtreeRoot[nc - 1];
    }
    // Breadth-first order.
    std::vector<uint32_t> order{0};
    std::vector<uint32_t> newIndex(table.size(), 0);
    for (size_t head = 0; head < order.size(); ++head) {
        const N& n = table[order[head]];
        newIndex[order[head]] = static_cast<uint32_t>(head);
        if (n.prop >= 0) {
            order.push_back(n.left);
            order.push_back(n.right);
        }
    }
    leafMap.assign(nc * 64, 0);
    std::vector<uint32_t> words(order.size() * kModTreeWords, 0);
    numLeaves = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const N& n = table[order[i]];
        uint32_t* w = &words[i * kModTreeWords];
        w[0] = static_cast<uint32_t>(n.prop);
        w[1] = static_cast<uint32_t>(n.split);
        w[2] = n.prop >= 0 ? newIndex[n.left] : 0;
        w[3] = n.prop >= 0 ? newIndex[n.right] : 0;
        if (n.prop < 0) {
            leafMap[n.channel * 64 + n.leaf] = numLeaves;
            w[4] = numLeaves++;
        }
        w[5] = n.predictor;
        w[6] = static_cast<uint32_t>(n.offset);
    }
    return words;
}

/// libjxl's kACMeta tree (enc_encoding.cc PredefinedTree): splits on the
/// channel, the row (strategy or qf) and the left or top value; Gradient for
/// the chroma-from-luma tiles, Left for the qf, Zero for the strategies and
/// the EPF sharpness (whose leaves carry offset 4 here, so it costs nothing).
/// Re-serialised in the breadth-first order DecodeTree expects.
std::vector<uint32_t> acMetaTree(uint32_t& numLeaves) {
    struct N {
        int32_t prop, split;   // prop -1: a leaf
        uint32_t left;         // right child = left + 1
        uint32_t predictor;
        int32_t offset;
    };
    // libjxl's table: node, Split(property, value, left child) or Leaf(predictor).
    const std::vector<N> table = {
        {0, 1, 1, 0, 0},  {0, 2, 3, 0, 0},  {0, 0, 5, 0, 0},   {6, 3, 21, 0, 0}, {2, 0, 7, 0, 0},
        {-1, 0, 0, kPredGradient, 0}, {-1, 0, 0, kPredGradient, 0},
        {7, 5, 9, 0, 0},  {7, 5, 15, 0, 0}, {7, 11, 11, 0, 0}, {7, 3, 13, 0, 0},
        {-1, 0, 0, kPredLeft, 0}, {-1, 0, 0, kPredLeft, 0}, {-1, 0, 0, kPredLeft, 0}, {-1, 0, 0, kPredLeft, 0},
        {7, 11, 17, 0, 0}, {7, 3, 19, 0, 0},
        {-1, 0, 0, kPredZero, 0}, {-1, 0, 0, kPredZero, 0}, {-1, 0, 0, kPredZero, 0}, {-1, 0, 0, kPredZero, 0},
        {7, 3, 23, 0, 0}, {7, 3, 25, 0, 0},
        {-1, 0, 0, kPredZero, 4}, {-1, 0, 0, kPredZero, 4}, {-1, 0, 0, kPredZero, 4}, {-1, 0, 0, kPredZero, 4},
    };
    // Breadth-first renumbering: the decoder places a split's children after every pending node.
    std::vector<uint32_t> order{0};
    std::vector<uint32_t> newIndex(table.size(), 0);
    for (size_t head = 0; head < order.size(); ++head) {
        const N& n = table[order[head]];
        newIndex[order[head]] = static_cast<uint32_t>(head);
        if (n.prop >= 0) {
            order.push_back(n.left);
            order.push_back(n.left + 1);
        }
    }
    std::vector<uint32_t> words(order.size() * kModTreeWords, 0);
    numLeaves = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const N& n = table[order[i]];
        uint32_t* w = &words[i * kModTreeWords];
        w[0] = static_cast<uint32_t>(n.prop);
        w[1] = static_cast<uint32_t>(n.split);
        w[2] = n.prop >= 0 ? newIndex[n.left] : 0;
        w[3] = n.prop >= 0 ? newIndex[n.left + 1] : 0;
        w[4] = n.prop < 0 ? numLeaves++ : 0;
        w[5] = n.predictor;
        w[6] = static_cast<uint32_t>(n.offset);
    }
    return words;
}

}   // namespace

Result<std::vector<uint8_t>> Encoder::encodeLossless(const image::Rgb8& image, const Options& options) {
    gpu::Device& dev = *device_;
    const FrameDim dim = codestream::frameDim(image.width, image.height);
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const bool fitsGroup = image.width <= dim.groupDim && image.height <= dim.groupDim;
    const uint64_t pixels = uint64_t{image.width} * image.height;
    const uint32_t channels = image.channels;

    // libjxl's Predictor::Best per channel: the weighted predictor with the fixed
    // WP-property tree, or Gradient with the gradient tree, whichever the
    // channel's tokens estimate cheaper; the frame's tree joins the choices.
    uint32_t wpLeaves = 0, gradLeaves = 0;
    const std::vector<uint32_t> wpTree = fixedTree(kModPropWp, kPredWeighted, channels * pixels, wpLeaves);
    const std::vector<uint32_t> gradTree = fixedTree(kModPropGradient, kPredGradient, channels * pixels, gradLeaves);
    SJXL_ASSIGN(auto wpTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, wpTree, "enc.wpTree"));
    SJXL_ASSIGN(auto gradTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, gradTree, "enc.gradTree"));
    uint32_t numLeaves = 0;
    uint32_t numTreeNodes = 0;
    gpu::Buffer treeNodes;

    std::vector<uint32_t> rgbWords((image.pixels.size() + 3) / 4, 0);
    std::memcpy(rgbWords.data(), image.pixels.data(), image.pixels.size());
    SJXL_ASSIGN(auto rgb, gpu::Buffer::fromSpan<uint32_t>(dev, rgbWords, "enc.rgb8"));
    SJXL_ASSIGN(auto planes, gpu::Buffer::ofType<int32_t>(dev, channels * pixels, "enc.planes"));
    SJXL_ASSIGN(auto modTokens, gpu::Buffer::ofType<uint32_t>(dev, channels * pixels, "enc.modTokens"));
    SJXL_ASSIGN(auto passTokens, gpu::Buffer::ofType<uint32_t>(dev, 2 * channels * pixels, "enc.passTokens"));
    SJXL_ASSIGN(auto passHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{2} * channels * kSetContexts * kHistBins, "enc.passHist"));
    SJXL_ASSIGN(auto passBits, gpu::Buffer::ofType<float>(dev, 2 * channels, "enc.passBits"));
    gpu::Buffer choice, leafMapBuf;
    const uint32_t wpWordsPerGroup = 5 * 2 * (dim.groupDim + 2);
    SJXL_ASSIGN(auto wpScratch, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numGroups} * wpWordsPerGroup, "enc.wpScratchMod"));
    const uint32_t maxLeaves = channels * kSetContexts;   // the combined tree's leaves at most
    SJXL_ASSIGN(auto ctxHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{maxLeaves} * kHistBins, "enc.ctxHist"));
    SJXL_ASSIGN(auto ctxMap, gpu::Buffer::ofType<uint32_t>(dev, maxLeaves, "enc.ctxMap"));
    SJXL_ASSIGN(auto ctxTotal, gpu::Buffer::ofType<float>(dev, maxLeaves, "enc.ctxTotal"));
    SJXL_ASSIGN(auto ctxEntropy, gpu::Buffer::ofType<float>(dev, maxLeaves, "enc.ctxEntropy"));
    SJXL_ASSIGN(auto ctxDist, gpu::Buffer::ofType<float>(dev, maxLeaves, "enc.ctxDist"));
    SJXL_ASSIGN(auto ctxSingle, gpu::Buffer::ofType<uint32_t>(dev, maxLeaves, "enc.ctxSingle"));
    SJXL_ASSIGN(auto clusterInfo, gpu::Buffer::ofType<uint32_t>(dev, 4, "enc.clusterInfo"));
    SJXL_ASSIGN(auto ctxMapMtf, gpu::Buffer::ofType<uint32_t>(dev, maxLeaves, "enc.ctxMapMtf"));
    SJXL_ASSIGN(auto hist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * kCodeStride, "enc.hist"));
    SJXL_ASSIGN(auto codes, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * kCodeStride, "enc.codes"));
    // Regions: one set of clusters (the map's own code lands at (1 + 2 numDcGroups) * 32).
    const uint32_t regions = (1 + 2 * numDcGroups) * kAcClusters + 1;
    SJXL_ASSIGN(auto ansInfo, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regions} * 256, "enc.ansInfo"));
    SJXL_ASSIGN(auto ansReverse, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regions} * 4096, "enc.ansReverse"));
    // The global blob holds the whole stream when the image fits a group.
    const uint64_t blobWords = kBlobStrideWords + (fitsGroup ? channels * pixels * (image.bitDepth / 8) + 1024 : 0);
    SJXL_ASSIGN(auto blobs, gpu::Buffer::ofType<uint32_t>(dev, blobWords, "enc.blobs"));
    SJXL_ASSIGN(auto blobBits, gpu::Buffer::ofType<uint32_t>(dev, 2, "enc.blobBits"));
    SJXL_ASSIGN(auto groupBits, gpu::Buffer::ofType<uint32_t>(dev, numGroups, "enc.groupBits"));

    gpu::CommandBatch batch(dev);
    const auto bindMod = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& groupStart, const gpu::Buffer& body,
                             const gpu::Buffer& tree, uint32_t pass) {
        c["rgb8"].setBinding(rgb.rhi());
        c["planes"].setBinding(planes.rhi());
        c["planesIn"].setBinding(planes.rhi());
        c["treeNodes"].setBinding(tree.rhi());
        c["wpScratch"].setBinding(wpScratch.rhi());
        c["passTokens"].setBinding(passTokens.rhi());
        c["passHist"].setBinding(passHist.rhi());
        c["passBits"].setBinding(passBits.rhi());
        c["choice"].setBinding(choice.rhi());
        c["leafMap"].setBinding(leafMapBuf.rhi());
        c["modTokens"].setBinding(modTokens.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["groupStart"].setBinding(groupStart.rhi());
        c["groupBits"].setBinding(groupBits.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["groupsX"].setData(dim.xsizeGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numLeaves"].setData(numLeaves);
        c["params"]["mode"].setData(mode);
        c["params"]["numChannels"].setData(channels);
        c["params"]["bitDepth"].setData(image.bitDepth);
        c["params"]["wpWordsPerGroup"].setData(wpWordsPerGroup);
        c["params"]["pass"].setData(pass);
    };
    // Both predictor passes and their per-channel entropies; the choice is bookkeeping on the host.
    SJXL_ASSIGN(choice, gpu::Buffer::ofType<uint32_t>(dev, channels, "enc.choice"));
    SJXL_ASSIGN(leafMapBuf, gpu::Buffer::ofType<uint32_t>(dev, channels * kSetContexts, "enc.leafMap"));
    rctYCoCg_.dispatch(batch, {image.width, image.height, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, wpTreeBuf, 0); });
    modTokenize_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, wpTreeBuf, 0); });
    modTokenize_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, gradTreeBuf, 1); });
    channelEntropy_.dispatch(batch, {2 * channels, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, wpTreeBuf, 0); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<float> passBitsHost, passBits.readAll<float>(dev));
    std::vector<uint32_t> choiceHost(channels);
    std::vector<const std::vector<uint32_t>*> subtrees(channels);
    for (uint32_t c = 0; c < channels; ++c) {
        choiceHost[c] = passBitsHost[c] <= passBitsHost[channels + c] ? 0u : 1u;
        subtrees[c] = choiceHost[c] == 0 ? &wpTree : &gradTree;
    }
    std::vector<uint32_t> leafMapHost;
    std::vector<uint32_t> treeWords;
    bool unanimous = true;
    for (uint32_t c = 1; c < channels; ++c) unanimous = unanimous && choiceHost[c] == choiceHost[0];
    if (unanimous) {
        // One tree for every channel: fewer leaves, so fewer histograms to carry.
        treeWords = *subtrees[0];
        numLeaves = choiceHost[0] == 0 ? wpLeaves : gradLeaves;
        leafMapHost.assign(size_t{channels} * kSetContexts, 0);
        for (uint32_t c = 0; c < channels; ++c)
            for (uint32_t j = 0; j < kSetContexts; ++j) leafMapHost[c * kSetContexts + j] = j;
    } else {
        treeWords = combineTreesByChannel(subtrees, leafMapHost, numLeaves);
    }
    numTreeNodes = static_cast<uint32_t>(treeWords.size() / kModTreeWords);
    SJXL_ASSIGN(treeNodes, gpu::Buffer::fromSpan<uint32_t>(dev, treeWords, "enc.treeNodes"));
    SJXL_ASSIGN(choice, gpu::Buffer::fromSpan<uint32_t>(dev, choiceHost, "enc.choice"));
    SJXL_ASSIGN(leafMapBuf, gpu::Buffer::fromSpan<uint32_t>(dev, leafMapHost, "enc.leafMap"));
    remapTokens_.dispatch(batch, {static_cast<uint32_t>(channels * pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, treeNodes, 0); });
    const auto bindCluster = [&](rhi::ShaderCursor c) {
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxTotal"].setBinding(ctxTotal.rhi());
        c["ctxEntropy"].setBinding(ctxEntropy.rhi());
        c["ctxDist"].setBinding(ctxDist.rhi());
        c["ctxSingle"].setBinding(ctxSingle.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["hist"].setBinding(hist.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["params"]["numContexts"].setData(numLeaves);
        c["params"]["maxClusters"].setData(std::min<uint32_t>(options.acClusters, kAcClusters));
        c["params"]["ctxBase"].setData(0u);
        c["params"]["infoIndex"].setData(0u);
        c["params"]["regionBase"].setData(0u);
    };
    clusterContexts_.dispatchGroups(batch, {1, 1, 1}, bindCluster);
    accumulateClusters_.dispatch(batch, {numLeaves, 1, 1}, bindCluster);
    const auto bindCodes = [&](rhi::ShaderCursor c) {
        c["dcGroups"].setBinding(dummy_.rhi());
        c["acs"].setBinding(dummy_.rhi());
        c["hist"].setBinding(hist.rhi());
        c["quantizer"].setBinding(dummy_.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["ctxMapMtf"].setBinding(ctxMapMtf.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["planes"].setBinding(planes.rhi());
        c["treeNodes"].setBinding(treeNodes.rhi());
        c["modTokens"].setBinding(modTokens.rhi());
        c["dcTree"].setBinding(dummy_.rhi());
        c["metaTree"].setBinding(dummy_.rhi());
        c["codes"].setBinding(codes.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["blobBits"].setBinding(blobBits.rhi());
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["numTreeNodes"].setData(numTreeNodes);
        c["params"]["numLeaves"].setData(numLeaves);
        c["params"]["numChannels"].setData(image.channels);
        c["params"]["bitDepth"].setData(image.bitDepth);
        c["params"]["dcTreeNodes"].setData(0u);
        c["params"]["dcLeaves"].setData(0u);
        c["params"]["metaTreeNodes"].setData(0u);
        c["params"]["metaLeaves"].setData(0u);
        c["params"]["numAcContexts"].setData(numLeaves);
    };
    modularGlobalBlob_.dispatchGroups(batch, {1, 1, 1}, bindCodes);
    if (!fitsGroup) {
        modGroupSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 1, dummy_, dummy_, treeNodes, 0); });
    }
    SJXL_TRY(batch.submit(true));

    // The layout is bookkeeping: byte-aligned sections DcGlobal, DC groups
    // (empty), AcGlobal (empty), AC groups; or one section when the image fits a group.
    SJXL_ASSIGN(const std::vector<uint32_t> blobBitsHost, blobBits.readAll<uint32_t>(dev));
    SJXL_ASSIGN(const std::vector<uint32_t> groupBitsHost, groupBits.readAll<uint32_t>(dev));
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> groupStartHost(numGroups, 0);
    uint32_t bodyBytes = (blobBitsHost[0] + 7) / 8;
    if (fitsGroup) {
        sizes.assign(1, bodyBytes);
    } else {
        sizes.push_back(bodyBytes);
        for (uint32_t g = 0; g < numDcGroups; ++g) sizes.push_back(0);
        sizes.push_back(0);
        for (uint32_t g = 0; g < numGroups; ++g) {
            groupStartHost[g] = bodyBytes * 8;
            const uint32_t bytes = (groupBitsHost[g] + 7) / 8;
            sizes.push_back(bytes);
            bodyBytes += bytes;
        }
    }
    SJXL_ASSIGN(auto groupStart, gpu::Buffer::fromSpan<uint32_t>(dev, groupStartHost, "enc.groupStart"));
    SJXL_ASSIGN(auto body, gpu::Buffer::ofType<uint32_t>(dev, bodyBytes / 4 + 2, "enc.body"));
    SJXL_ASSIGN(auto sectionStart, gpu::Buffer::ofType<uint32_t>(dev, 4, "enc.sectionStart"));   // DcGlobal at bit 0
    blobEmit_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        c["blobBits"].setBinding(blobBits.rhi());
        c["acSectionBits"].setBinding(dummy_.rhi());
        c["dcBits"].setBinding(dummy_.rhi());
        c["metaBits"].setBinding(dummy_.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["sectionStart"].setBinding(sectionStart.rhi());
        c["sectionBits"].setBinding(dummy_.rhi());
        c["sectionBytes"].setBinding(dummy_.rhi());
        c["dcPartStart"].setBinding(dummy_.rhi());
        c["totals"].setBinding(dummy_.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(0u);
        c["params"]["packed"].setData(0u);
    });
    if (!fitsGroup) {
        modGroupSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 2, groupStart, body, treeNodes, 0); });
    }
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> bodyWords, body.readAll<uint32_t>(dev));
    SJXL_ASSIGN(const std::vector<uint32_t> clusterInfoHost, clusterInfo.readAll<uint32_t>(dev));

    codestream::BitWriter head;
    codestream::writeCodestreamHeadersRgb(head, image.width, image.height, image.bitDepth, channels == 4, image.gamma);
    codestream::writeModularFrameHeader(head, channels == 4 ? 1 : 0);
    codestream::writeToc(head, sizes);
    std::vector<uint8_t> out = head.take();
    stats_ = Stats{};
    stats_.headBytes = static_cast<uint32_t>(out.size());
    stats_.bodyBytes = bodyBytes;
    stats_.sectionBytes = sizes;
    stats_.numAcClusters = clusterInfoHost[0];
    if (!platform::env("SJXL_DEBUG").empty()) {
        std::string ch;
        for (uint32_t c = 0; c < channels; ++c) ch += std::format("{}:{} ", c, choiceHost[c] == 0 ? "WP" : "grad");
        log::info("enc debug: lossless {}x{} x{} {}-bit, predictors {}, {} tree nodes, {} leaves, {} clusters, DcGlobal {} bits, {} groups",
                  image.width, image.height, channels, image.bitDepth, ch, numTreeNodes, numLeaves, clusterInfoHost[0], blobBitsHost[0], numGroups);
    }
    const auto* bodyBytesPtr = reinterpret_cast<const uint8_t*>(bodyWords.data());
    out.insert(out.end(), bodyBytesPtr, bodyBytesPtr + bodyBytes);
    return out;
}

}   // namespace sjxl::enc
