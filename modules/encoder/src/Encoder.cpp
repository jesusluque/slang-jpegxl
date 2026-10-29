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

constexpr uint32_t kAcItemsPerGroup = 32 * 32 * 3;
constexpr uint32_t kDcItemsPerGroup = 256 * 256 * 3;
constexpr uint32_t kMetaItemsPerGroup = 2 * 32 * 32 + 2 * 256 * 256;
constexpr uint32_t kCodeStride = 256;
constexpr uint32_t kBlobStrideWords = 8192;
constexpr uint32_t kAcClusters = 32;
constexpr uint32_t kNumAcContexts = 15 * (37 + 458);
constexpr uint32_t kAcAlphabet = 64;

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
    SJXL_ASSIGN(enc->dct_, gpu::ComputeKernel::create(lib, "sjxl/enc/cfl", "dct"));
    SJXL_ASSIGN(enc->cflTile_, gpu::ComputeKernel::create(lib, "sjxl/enc/cfl", "cflTile"));
    SJXL_ASSIGN(enc->quantize_, gpu::ComputeKernel::create(lib, "sjxl/enc/quantize", "quantize"));
    SJXL_ASSIGN(enc->acItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "acItems"));
    SJXL_ASSIGN(enc->dcItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "dcItems"));
    SJXL_ASSIGN(enc->metaItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "metaItems"));
    SJXL_ASSIGN(enc->clusterContexts_, gpu::ComputeKernel::create(lib, "sjxl/enc/cluster", "clusterContexts"));
    SJXL_ASSIGN(enc->acSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/ac_sections", "acSections"));
    SJXL_ASSIGN(enc->dcGroupBlobs_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildDcGroupBlobs"));
    SJXL_ASSIGN(enc->globalBlobs_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildGlobalBlobs"));
    SJXL_ASSIGN(enc->layout_, gpu::ComputeKernel::create(lib, "sjxl/enc/layout", "layoutSections"));
    SJXL_ASSIGN(enc->blobEmit_, gpu::ComputeKernel::create(lib, "sjxl/enc/layout", "blobEmit"));
    SJXL_ASSIGN(enc->prefixSum_, gpu::PrefixSum::create(lib));

    const auto weights = codestream::dct8DequantWeights();
    SJXL_ASSIGN(enc->dequant_, gpu::Buffer::fromSpan<float>(*enc->device_, weights, "enc.dequant"));
    const auto order = codestream::dct8NaturalOrder();
    SJXL_ASSIGN(enc->order_, gpu::Buffer::fromSpan<uint32_t>(*enc->device_, order, "enc.order"));
    SJXL_ASSIGN(enc->dummy_, gpu::Buffer::ofType<uint32_t>(*enc->device_, 4, "enc.dummy"));
    return enc;
}

Result<std::vector<uint8_t>> Encoder::encode(const image::Rgb8& image, const Options& options) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() != size_t{image.width} * image.height * 3) {
        return Error(ErrorCode::InvalidArgument, "empty image or pixel count does not match its size");
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
    SJXL_ASSIGN(auto coeffs, gpu::Buffer::ofType<float>(dev, uint64_t{3} * nblocks * 64, "enc.coeffs"));
    SJXL_ASSIGN(auto cmapX, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapX"));
    SJXL_ASSIGN(auto cmapB, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapB"));
    SJXL_ASSIGN(auto acQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks * 64, "enc.acQuant"));
    SJXL_ASSIGN(auto dcQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks, "enc.dcQuant"));
    SJXL_ASSIGN(auto nonZeros, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.nonZeros"));

    const std::vector<Uint4> acRects = groupRects(dim, codestream::kGroupDimBlocks, dim.xsizeGroups, dim.ysizeGroups);
    const std::vector<Uint4> dcRects = groupRects(dim, codestream::kDcGroupDimBlocks, dim.xsizeDcGroups, dim.ysizeDcGroups);
    SJXL_ASSIGN(auto groups, gpu::Buffer::fromSpan<Uint4>(dev, acRects, "enc.groups"));
    SJXL_ASSIGN(auto dcGroups, gpu::Buffer::fromSpan<Uint4>(dev, dcRects, "enc.dcGroups"));

    const uint32_t numCodeRegions = kAcClusters + 2 * numDcGroups;
    const uint32_t numBlobRegions = 2 + 2 * numDcGroups;
    SJXL_ASSIGN(auto hist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.hist"));
    SJXL_ASSIGN(auto codes, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.codes"));
    SJXL_ASSIGN(auto blobs, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numBlobRegions} * kBlobStrideWords, "enc.blobs"));
    SJXL_ASSIGN(auto blobBits, gpu::Buffer::ofType<uint32_t>(dev, numBlobRegions, "enc.blobBits"));
    SJXL_ASSIGN(auto ctxHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kNumAcContexts} * kAcAlphabet, "enc.ctxHist"));
    SJXL_ASSIGN(auto ctxMap, gpu::Buffer::ofType<uint32_t>(dev, kNumAcContexts, "enc.ctxMap"));
    SJXL_ASSIGN(auto ctxTotal, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxTotal"));
    SJXL_ASSIGN(auto ctxEntropy, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxEntropy"));
    SJXL_ASSIGN(auto ctxDist, gpu::Buffer::ofType<float>(dev, kNumAcContexts, "enc.ctxDist"));
    SJXL_ASSIGN(auto ctxSingle, gpu::Buffer::ofType<uint32_t>(dev, kNumAcContexts, "enc.ctxSingle"));
    SJXL_ASSIGN(auto clusterInfo, gpu::Buffer::ofType<uint32_t>(dev, 4, "enc.clusterInfo"));
    SJXL_ASSIGN(auto ctxMapMtf, gpu::Buffer::ofType<uint32_t>(dev, kNumAcContexts, "enc.ctxMapMtf"));
    SJXL_ASSIGN(auto ansInfo, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters + 1} * 256, "enc.ansInfo"));
    SJXL_ASSIGN(auto ansReverse, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters + 1} * 4096, "enc.ansReverse"));
    SJXL_ASSIGN(auto acSectionBits, gpu::Buffer::ofType<uint32_t>(dev, numGroups, "enc.acSectionBits"));

    const uint32_t acItems = numGroups * kAcItemsPerGroup;
    const uint32_t dcItems = numDcGroups * kDcItemsPerGroup;
    const uint32_t metaItems = numDcGroups * kMetaItemsPerGroup;
    SJXL_ASSIGN(auto dcBits, gpu::Buffer::ofType<uint32_t>(dev, dcItems + 1, "enc.dcBits"));
    SJXL_ASSIGN(auto metaBits, gpu::Buffer::ofType<uint32_t>(dev, metaItems + 1, "enc.metaBits"));
    SJXL_ASSIGN(auto dcPrefix, gpu::Buffer::ofType<uint32_t>(dev, dcItems + 1, "enc.dcPrefix"));
    SJXL_ASSIGN(auto metaPrefix, gpu::Buffer::ofType<uint32_t>(dev, metaItems + 1, "enc.metaPrefix"));
    SJXL_ASSIGN(auto scanTotals, gpu::Buffer::ofType<uint32_t>(dev, 3, "enc.scanTotals"));

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
    const auto bindCfl = [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(planes.rhi());
        c["coeffs"].setBinding(coeffs.rhi());
        c["dequant"].setBinding(dequant_.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["tilesY"].setData(tilesY);
    };
    dct_.dispatch(batch, {nblocks, 1, 1}, bindCfl);
    if (options.chromaFromLuma) {
        cflTile_.dispatch(batch, {tilesX * tilesY, 1, 1}, bindCfl);
    }
    quantize_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
        c["coeffs"].setBinding(coeffs.rhi());
        c["dequant"].setBinding(dequant_.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["xQmMultiplier"].setData(static_cast<float>(std::pow(1.25, static_cast<int>(frame.xQmScale) - 2)));
        c["params"]["bQmMultiplier"].setData(static_cast<float>(std::pow(1.25, static_cast<int>(frame.bQmScale) - 2)));
    });

    const auto bindItems = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& bits, const gpu::Buffer& prefix,
                               const gpu::Buffer& body) {
        c["groups"].setBinding(groups.rhi());
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["order"].setBinding(order_.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
        c["codes"].setBinding(codes.rhi());
        c["hist"].setBinding(hist.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["itemBits"].setBinding(bits.rhi());
        c["prefix"].setBinding(prefix.rhi());
        c["sectionStart"].setBinding(sectionStart.rhi());
        c["dcPartStart"].setBinding(dcPartStart.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["nblocks"].setData(nblocks);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["mode"].setData(mode);
        c["params"]["tilesX"].setData(tilesX);
    };
    const auto bindSections = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& body) {
        c["groups"].setBinding(groups.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["order"].setBinding(order_.rhi());
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
    // The AC groups: context histograms in mode 0 (items), then the ANS
    // sections (one thread per group) measured in mode 1 and emitted in mode 2.
    const auto itemPasses = [&](uint32_t mode, const gpu::Buffer& body) {
        if (mode == 0) {
            acItems_.dispatch(batch, {acItems, 1, 1}, [&](rhi::ShaderCursor c) { bindItems(c, mode, dcBits, dcPrefix, body); });
        } else {
            acSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindSections(c, mode, body); });
        }
        dcItems_.dispatch(batch, {dcItems, 1, 1}, [&](rhi::ShaderCursor c) { bindItems(c, mode, dcBits, dcPrefix, body); });
        metaItems_.dispatch(batch, {metaItems, 1, 1},
                            [&](rhi::ShaderCursor c) { bindItems(c, mode, metaBits, metaPrefix, body); });
    };

    itemPasses(0, dummy_);   // histograms

    // The AC contexts' histograms clustered into the codes' histograms.
    clusterContexts_.dispatchGroups(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["ctxTotal"].setBinding(ctxTotal.rhi());
        c["ctxEntropy"].setBinding(ctxEntropy.rhi());
        c["ctxDist"].setBinding(ctxDist.rhi());
        c["ctxSingle"].setBinding(ctxSingle.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["hist"].setBinding(hist.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["params"]["numContexts"].setData(kNumAcContexts);
        c["params"]["maxClusters"].setData(std::min<uint32_t>(options.acClusters, kAcClusters));
    });

    const auto bindCodes = [&](rhi::ShaderCursor c) {
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["hist"].setBinding(hist.rhi());
        c["quantizer"].setBinding(quantizer.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["clusterInfo"].setBinding(clusterInfo.rhi());
        c["ctxMapMtf"].setBinding(ctxMapMtf.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["codes"].setBinding(codes.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["blobBits"].setBinding(blobBits.rhi());
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
    };
    dcGroupBlobs_.dispatchGroups(batch, {numDcGroups, 1, 1}, bindCodes);
    globalBlobs_.dispatchGroups(batch, {1, 1, 1}, bindCodes);

    itemPasses(1, dummy_);   // bit counts

    SJXL_TRY(prefixSum_.apply(batch, dcBits, dcPrefix, scanTotals, dcItems + 1));
    SJXL_TRY(prefixSum_.apply(batch, metaBits, metaPrefix, scanTotals, metaItems + 1));

    const auto bindLayout = [&](rhi::ShaderCursor c, const gpu::Buffer& body) {
        c["blobBits"].setBinding(blobBits.rhi());
        c["acSectionBits"].setBinding(acSectionBits.rhi());
        c["dcPrefix"].setBinding(dcPrefix.rhi());
        c["metaPrefix"].setBinding(metaPrefix.rhi());
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
        SJXL_ASSIGN(const auto blobBitsHost, blobBits.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto sizesHost, sectionBytes.readAll<uint32_t>(dev));
        std::string sections;
        for (uint32_t b : sizesHost) sections += std::format("{} ", b);
        log::info("enc debug: blob bits DcGlobal {} AcGlobal {}; section bytes {}", blobBitsHost[0], blobBitsHost[1], sections);
    }

    SJXL_ASSIGN(const std::vector<uint32_t> totalsHost, totals.readAll<uint32_t>(dev));
    SJXL_ASSIGN(std::vector<uint32_t> sizes, sectionBytes.readAll<uint32_t>(dev));
    const uint32_t bodyBytes = totalsHost[0];
    if (bodyBytes == 0) {
        return Error(ErrorCode::InternalError, "the layout produced an empty body");
    }

    // The body: zero, then every item and blob ORs its bits in.
    SJXL_ASSIGN(auto body, gpu::Buffer::ofType<uint32_t>(dev, (bodyBytes + 3) / 4 + 1, "enc.body"));
    itemPasses(2, body);
    blobEmit_.dispatch(batch, {numBlobRegions, 1, 1}, [&](rhi::ShaderCursor c) { bindLayout(c, body); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> bodyWords, body.readAll<uint32_t>(dev));

    // Headers and TOC on the CPU, then the body.
    codestream::BitWriter head;
    codestream::writeCodestreamHeaders(head, image.width, image.height);
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

}   // namespace sjxl::enc
