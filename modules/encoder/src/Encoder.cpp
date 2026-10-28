// Copyright (c) 2026 jesus luque.
#include "sjxl/encoder/Encoder.h"

#include <array>
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
constexpr uint32_t kBlobStrideWords = 2048;

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

}   // namespace

Result<std::unique_ptr<Encoder>> Encoder::create(std::shared_ptr<gpu::ShaderLibrary> library) {
    std::unique_ptr<Encoder> enc(new Encoder());
    enc->library_ = std::move(library);
    enc->device_ = &enc->library_->device();
    gpu::ShaderLibrary& lib = *enc->library_;
    SJXL_ASSIGN(enc->rgbToXyb_, gpu::ComputeKernel::create(lib, "sjxl/enc/xyb", "rgbToXyb"));
    SJXL_ASSIGN(enc->dctQuant_, gpu::ComputeKernel::create(lib, "sjxl/enc/dct_quant", "dctQuant"));
    SJXL_ASSIGN(enc->acItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "acItems"));
    SJXL_ASSIGN(enc->dcItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "dcItems"));
    SJXL_ASSIGN(enc->metaItems_, gpu::ComputeKernel::create(lib, "sjxl/enc/items", "metaItems"));
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
    const codestream::QuantParams qp = codestream::quantParamsForDistance(options.distance);
    const uint32_t nblocks = dim.numBlocks();
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const uint32_t numSections = dim.numSections();

    // Input, padded to whole words.
    std::vector<uint32_t> rgbWords((image.pixels.size() + 3) / 4, 0);
    std::memcpy(rgbWords.data(), image.pixels.data(), image.pixels.size());
    SJXL_ASSIGN(auto rgb, gpu::Buffer::fromSpan<uint32_t>(dev, rgbWords, "enc.rgb8"));

    const uint64_t planePixels = uint64_t{dim.xsizePadded} * dim.ysizePadded;
    SJXL_ASSIGN(auto xyb, gpu::Buffer::ofType<float>(dev, 3 * planePixels, "enc.xyb"));
    SJXL_ASSIGN(auto acQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks * 64, "enc.acQuant"));
    SJXL_ASSIGN(auto dcQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks, "enc.dcQuant"));
    SJXL_ASSIGN(auto nonZeros, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.nonZeros"));

    const std::vector<Uint4> acRects = groupRects(dim, codestream::kGroupDimBlocks, dim.xsizeGroups, dim.ysizeGroups);
    const std::vector<Uint4> dcRects = groupRects(dim, codestream::kDcGroupDimBlocks, dim.xsizeDcGroups, dim.ysizeDcGroups);
    SJXL_ASSIGN(auto groups, gpu::Buffer::fromSpan<Uint4>(dev, acRects, "enc.groups"));
    SJXL_ASSIGN(auto dcGroups, gpu::Buffer::fromSpan<Uint4>(dev, dcRects, "enc.dcGroups"));

    const uint32_t numCodeRegions = 1 + 2 * numDcGroups;
    const uint32_t numBlobRegions = 2 + 2 * numDcGroups;
    SJXL_ASSIGN(auto hist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.hist"));
    SJXL_ASSIGN(auto codes, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions} * kCodeStride, "enc.codes"));
    SJXL_ASSIGN(auto blobs, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numBlobRegions} * kBlobStrideWords, "enc.blobs"));
    SJXL_ASSIGN(auto blobBits, gpu::Buffer::ofType<uint32_t>(dev, numBlobRegions, "enc.blobBits"));

    const uint32_t acItems = numGroups * kAcItemsPerGroup;
    const uint32_t dcItems = numDcGroups * kDcItemsPerGroup;
    const uint32_t metaItems = numDcGroups * kMetaItemsPerGroup;
    SJXL_ASSIGN(auto acBits, gpu::Buffer::ofType<uint32_t>(dev, acItems + 1, "enc.acBits"));
    SJXL_ASSIGN(auto dcBits, gpu::Buffer::ofType<uint32_t>(dev, dcItems + 1, "enc.dcBits"));
    SJXL_ASSIGN(auto metaBits, gpu::Buffer::ofType<uint32_t>(dev, metaItems + 1, "enc.metaBits"));
    SJXL_ASSIGN(auto acPrefix, gpu::Buffer::ofType<uint32_t>(dev, acItems + 1, "enc.acPrefix"));
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
    dctQuant_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
        c["xyb"].setBinding(xyb.rhi());
        c["dequant"].setBinding(dequant_.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["quantMul"].setData(static_cast<float>(qp.quantField) * qp.globalScaleFloat);
        c["params"]["dcMul"].setData(qp.globalScaleFloat * static_cast<float>(qp.quantDc));
    });

    const auto bindItems = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& bits, const gpu::Buffer& prefix,
                               const gpu::Buffer& body) {
        c["groups"].setBinding(groups.rhi());
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["acQuant"].setBinding(acQuant.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["nonZeros"].setBinding(nonZeros.rhi());
        c["order"].setBinding(order_.rhi());
        c["codes"].setBinding(codes.rhi());
        c["hist"].setBinding(hist.rhi());
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
        c["params"]["quantField"].setData(qp.quantField);
    };
    const auto itemPasses = [&](uint32_t mode, const gpu::Buffer& body) {
        acItems_.dispatch(batch, {acItems, 1, 1}, [&](rhi::ShaderCursor c) { bindItems(c, mode, acBits, acPrefix, body); });
        dcItems_.dispatch(batch, {dcItems, 1, 1}, [&](rhi::ShaderCursor c) { bindItems(c, mode, dcBits, dcPrefix, body); });
        metaItems_.dispatch(batch, {metaItems, 1, 1},
                            [&](rhi::ShaderCursor c) { bindItems(c, mode, metaBits, metaPrefix, body); });
    };

    itemPasses(0, dummy_);   // histograms

    const auto bindCodes = [&](rhi::ShaderCursor c) {
        c["dcGroups"].setBinding(dcGroups.rhi());
        c["hist"].setBinding(hist.rhi());
        c["codes"].setBinding(codes.rhi());
        c["blobs"].setBinding(blobs.rhi());
        c["blobBits"].setBinding(blobBits.rhi());
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["globalScale"].setData(qp.globalScale);
        c["params"]["quantDc"].setData(qp.quantDc);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
    };
    dcGroupBlobs_.dispatchGroups(batch, {numDcGroups, 1, 1}, bindCodes);
    globalBlobs_.dispatchGroups(batch, {1, 1, 1}, bindCodes);

    itemPasses(1, dummy_);   // bit counts

    SJXL_TRY(prefixSum_.apply(batch, acBits, acPrefix, scanTotals, acItems + 1));
    SJXL_TRY(prefixSum_.apply(batch, dcBits, dcPrefix, scanTotals, dcItems + 1));
    SJXL_TRY(prefixSum_.apply(batch, metaBits, metaPrefix, scanTotals, metaItems + 1));

    const auto bindLayout = [&](rhi::ShaderCursor c, const gpu::Buffer& body) {
        c["blobBits"].setBinding(blobBits.rhi());
        c["acPrefix"].setBinding(acPrefix.rhi());
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
    if (!platform::env("SJXL_DEBUG").empty()) {
        SJXL_ASSIGN(const auto xybHost, xyb.readAll<float>(dev));
        SJXL_ASSIGN(const auto dcHost, dcQuant.readAll<int32_t>(dev));
        SJXL_ASSIGN(const auto acHost, acQuant.readAll<int32_t>(dev));
        log::info("enc debug: xyb(0,0) = {} {} {}; dcQuant block0 = {} {} {}; qp gs={} qdc={} qf={}", xybHost[0], xybHost[planePixels],
                  xybHost[2 * planePixels], dcHost[0], dcHost[nblocks], dcHost[2 * nblocks], qp.globalScale, qp.quantDc, qp.quantField);
        const uint32_t dbgBlock = static_cast<uint32_t>(std::atoi(platform::env("SJXL_DEBUG_BLOCK").c_str()));
        for (uint32_t ch = 0; ch < 3; ++ch) {
            std::string ac;
            for (int k = 0; k < 64; ++k) ac += std::format("{} ", acHost[(ch * nblocks + dbgBlock) * 64 + k]);
            log::info("enc debug: acQuant c{} block{} = {}", ch, dbgBlock, ac);
        }
        log::info("enc debug: dcQuant block{} = {} {} {}", dbgBlock, dcHost[dbgBlock], dcHost[nblocks + dbgBlock], dcHost[2 * nblocks + dbgBlock]);
        std::string row;
        for (int k = 0; k < 10; ++k) row += std::format("({:.3f},{:.3f},{:.3f}) ", xybHost[k], xybHost[planePixels + k], xybHost[2 * planePixels + k]);
        log::info("enc debug: xyb row0[0..10) = {}", row);
        SJXL_ASSIGN(const auto rgbHost, rgb.readAll<uint32_t>(dev));
        log::info("enc debug: rgb words = {:08x} {:08x} {:08x}; dims {}x{} padded {}x{}", rgbHost[0], rgbHost[1], rgbHost[2], image.width, image.height, dim.xsizePadded, dim.ysizePadded);
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
    codestream::FrameParams frame;
    frame.gaborish = false;
    frame.epfIters = 0;
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
