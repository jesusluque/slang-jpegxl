// Copyright (c) 2026 jesus luque.
#include "sjxl/encoder/Encoder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "sjxl/codestream/Headers.h"
#include "sjxl/codestream/Tables.h"
#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include <format>
#include <cstdlib>
#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/StageTimer.h"

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
constexpr uint32_t kDefaultUintCfg = 4u | (2u << 8);   // HybridUintConfig(4, 2, 0) as clusterCfg holds it
constexpr uint32_t kModTreeWords = 8;
constexpr uint32_t kModPropGradient = 9;
constexpr uint32_t kModPropWp = 15;
constexpr uint32_t kPredZero = 0, kPredLeft = 1, kPredGradient = 5, kPredWeighted = 6;

/// LoopFilterFromParams (enc_frame.cc): Gaborish above distance 0.5, EPF
/// iterations by distance thresholds 0.7, 1.5 and 4. x_qm_scale 3 up to
/// distance 2.5 then 4, 5, 6 (the pixel-statistics adjustment is squirrel+).
/// A progressive pass as libjxl defines it (enc_progressive_split.h PassDefinition).
struct PassDef {
    uint32_t ncoeffs;      // side of the coefficient square per 8x8 block
    uint32_t shift;
    uint32_t downsampling; // 0: not a downsampling stop
};

/// libjxl's pass lists (enc_frame.cc SetProgressiveMode): qprogressive codes every
/// coefficient at shift 1 then the remainder; progressive the 2x2 lowest
/// frequencies, then 3x3, then all.
std::vector<PassDef> passesFor(const Options& options) {
    if (const std::string custom = platform::env("SJXL_PASSES"); !custom.empty()) {   // debug: "ncoeffs:shift:downsample,..."
        std::vector<PassDef> passes;
        size_t pos = 0;
        while (pos < custom.size()) {
            PassDef d{};
            int n = 0;
            if (std::sscanf(custom.c_str() + pos, "%u:%u:%u%n", &d.ncoeffs, &d.shift, &d.downsampling, &n) != 3) break;
            passes.push_back(d);
            pos += static_cast<size_t>(n) + 1;
        }
        if (!passes.empty()) return passes;
    }
    if (options.progressive == 1) return {{8, 1, 2}, {8, 0, 0}};
    if (options.progressive == 2) return {{2, 0, 4}, {3, 0, 2}, {8, 0, 0}};
    return {{8, 0, 0}};
}

codestream::FrameParams frameParamsFor(const Options& options) {
    codestream::FrameParams frame;
    // Passes (ProgressiveSplitter::InitPasses).
    const std::vector<PassDef> passes = passesFor(options);
    frame.numPasses = static_cast<uint32_t>(passes.size());
    frame.numDownsample = 0;
    for (uint32_t i = 0; i + 1 < frame.numPasses; ++i) {
        frame.shift[i] = passes[i].shift;
        if (passes[i].downsampling > 1) {
            frame.downsample[frame.numDownsample] = passes[i].downsampling;
            frame.lastPass[frame.numDownsample] = i;
            if (passes[i + 1].downsampling < passes[i].downsampling) ++frame.numDownsample;
        }
    }
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
    SJXL_ASSIGN(enc->passNonZeroCount_, gpu::ComputeKernel::create(lib, "sjxl/enc/transform", "passNonZeroCount"));
    SJXL_ASSIGN(enc->cflTile_, gpu::ComputeKernel::create(lib, "sjxl/enc/cfl", "cflTile"));
    SJXL_ASSIGN(enc->quantize_, gpu::ComputeKernel::create(lib, "sjxl/enc/quantize", "quantize"));
    SJXL_ASSIGN(enc->dcTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "dcTokenize"));
    SJXL_ASSIGN(enc->metaTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "metaTokenize"));
    SJXL_ASSIGN(enc->dcGroupSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/dc_tokens", "dcGroupSections"));
    SJXL_ASSIGN(enc->clusterContexts_, gpu::ComputeKernel::create(lib, "sjxl/enc/cluster", "clusterContexts"));
    SJXL_ASSIGN(enc->accumulateClusters_, gpu::ComputeKernel::create(lib, "sjxl/enc/cluster", "accumulateClusters"));
    SJXL_ASSIGN(enc->acSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/ac_sections", "acSections"));
    SJXL_ASSIGN(enc->unpackPlanes_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "unpackPlanes"));
    SJXL_ASSIGN(enc->paletteCount_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "paletteCount"));
    SJXL_ASSIGN(enc->paletteBuild_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "paletteBuild"));
    SJXL_ASSIGN(enc->paletteIndex_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "paletteIndex"));
    SJXL_ASSIGN(enc->paletteVerify_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "paletteVerify"));
    SJXL_ASSIGN(enc->compactStats_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "compactStats"));
    SJXL_ASSIGN(enc->compactBuild_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "compactBuild"));
    SJXL_ASSIGN(enc->compactApply_, gpu::ComputeKernel::create(lib, "sjxl/enc/palette", "compactApply"));
    SJXL_ASSIGN(enc->rleScan_, gpu::ComputeKernel::create(lib, "sjxl/enc/lz77", "rleScan"));
    SJXL_ASSIGN(enc->jpegDecodeScan_, gpu::ComputeKernel::create(lib, "sjxl/enc/jpeg", "jpegDecodeScan"));
    SJXL_ASSIGN(enc->jpegToCoeffs_, gpu::ComputeKernel::create(lib, "sjxl/enc/jpeg", "jpegToCoeffs"));
    SJXL_ASSIGN(enc->jpegBlockCtx_, gpu::ComputeKernel::create(lib, "sjxl/enc/jpeg", "jpegBlockCtx"));
    SJXL_ASSIGN(enc->changeBounds_, gpu::ComputeKernel::create(lib, "sjxl/enc/crop", "changeBounds"));
    SJXL_ASSIGN(enc->cropBytes_, gpu::ComputeKernel::create(lib, "sjxl/enc/crop", "cropBytes"));
    SJXL_ASSIGN(enc->rleHistograms_, gpu::ComputeKernel::create(lib, "sjxl/enc/lz77", "rleHistograms"));
    SJXL_ASSIGN(enc->uintConfigHist_, gpu::ComputeKernel::create(lib, "sjxl/enc/lz77", "uintConfigHist"));
    SJXL_ASSIGN(enc->uintConfigChoose_, gpu::ComputeKernel::create(lib, "sjxl/enc/lz77", "uintConfigChoose"));
    SJXL_ASSIGN(enc->rctCost_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "rctCost"));
    SJXL_ASSIGN(enc->rctApply_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "rctApply"));
    SJXL_ASSIGN(enc->dcFramePlanes_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "dcFramePlanes"));
    SJXL_ASSIGN(enc->modTokenize_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modTokenize"));
    SJXL_ASSIGN(enc->modSamples_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modSamples"));
    SJXL_ASSIGN(enc->modTokenizeTree_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modTokenizeTree"));
    SJXL_ASSIGN(enc->propStats_, gpu::ComputeKernel::create(lib, "sjxl/enc/tree", "propStats"));
    SJXL_ASSIGN(enc->propThresholds_, gpu::ComputeKernel::create(lib, "sjxl/enc/tree", "propThresholds"));
    SJXL_ASSIGN(enc->treeHistograms_, gpu::ComputeKernel::create(lib, "sjxl/enc/tree", "treeHistograms"));
    SJXL_ASSIGN(enc->treeSplit_, gpu::ComputeKernel::create(lib, "sjxl/enc/tree", "treeSplit"));
    SJXL_ASSIGN(enc->treeAssign_, gpu::ComputeKernel::create(lib, "sjxl/enc/tree", "treeAssign"));
    SJXL_ASSIGN(enc->extractAlpha_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "extractAlpha"));
    SJXL_ASSIGN(enc->channelEntropy_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "channelEntropy"));
    SJXL_ASSIGN(enc->remapTokens_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "remapTokens"));
    SJXL_ASSIGN(enc->modGroupSections_, gpu::ComputeKernel::create(lib, "sjxl/enc/modular", "modGroupSections"));
    SJXL_ASSIGN(enc->modularGlobalBlob_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildModularGlobalBlob"));
    SJXL_ASSIGN(enc->dcGroupPre_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildDcGroupPre"));
    SJXL_ASSIGN(enc->dcGroupMid_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildDcGroupMid"));
    SJXL_ASSIGN(enc->globalBlobs_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildGlobalBlobs"));
    SJXL_ASSIGN(enc->quantTableBlob_, gpu::ComputeKernel::create(lib, "sjxl/enc/codes", "buildQuantTableBlob"));
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

namespace {

Result<void> checkImage(const image::Rgb8& image) {
    if (image.width == 0 || image.height == 0 || (image.channels != 3 && image.channels != 4) ||
        (image.bitDepth != 8 && image.bitDepth != 16) ||
        image.pixels.size() != size_t{image.width} * image.height * image.channels * (image.bitDepth / 8)) {
        return Error(ErrorCode::InvalidArgument, "empty image or pixel count does not match its size");
    }
    return ok();
}

}   // namespace

Result<std::vector<uint8_t>> Encoder::encode(const image::Rgb8& image, const Options& options) {
    SJXL_TRY(checkImage(image));
    codestream::BitWriter head;
    if (options.lossless) {
        codestream::writeCodestreamHeadersRgb(head, image.width, image.height, image.bitDepth, image.channels == 4, image.gamma);
    } else {
        codestream::writeCodestreamHeaders(head, image.width, image.height, image.gamma, image.bitDepth, image.channels == 4);
    }
    std::vector<uint8_t> out = head.take();
    SJXL_ASSIGN(const std::vector<uint8_t> frame, encodeFrame(image, options, codestream::FrameInfo{}));
    stats_.headBytes += static_cast<uint32_t>(out.size());
    out.insert(out.end(), frame.begin(), frame.end());
    return out;
}

Result<std::vector<uint8_t>> Encoder::encodeAnimation(const std::vector<image::Rgb8>& frames, const std::vector<uint32_t>& durations,
                                                      uint32_t ticksPerSecond, uint32_t numLoops, const Options& options) {
    if (frames.empty() || durations.size() != frames.size()) {
        return Error(ErrorCode::InvalidArgument, "an animation needs frames and one duration per frame");
    }
    const image::Rgb8& first = frames.front();
    SJXL_TRY(checkImage(first));
    for (const image::Rgb8& f : frames) {
        if (f.width != first.width || f.height != first.height || f.channels != first.channels || f.bitDepth != first.bitDepth) {
            return Error(ErrorCode::InvalidArgument, "every frame of an animation must have the first frame's size, channels and depth");
        }
    }
    codestream::Animation animation;
    animation.tpsNumerator = ticksPerSecond;
    animation.tpsDenominator = 1;
    animation.numLoops = numLoops;
    codestream::BitWriter head;
    if (options.lossless) {
        codestream::writeCodestreamHeadersRgb(head, first.width, first.height, first.bitDepth, first.channels == 4, first.gamma, &animation);
    } else {
        codestream::writeCodestreamHeaders(head, first.width, first.height, first.gamma, first.bitDepth, first.channels == 4, &animation);
    }
    std::vector<uint8_t> out = head.take();
    gpu::Device& dev = *device_;
    const uint32_t bytesPerPixel = first.channels * (first.bitDepth / 8);
    // Which frames to emit and what rectangle each one changes: a frame identical
    // to the one before it only lengthens that frame's duration; a frame that
    // differs in a sub-rectangle is coded as that rectangle (a partial frame,
    // kReplace over the canvas kept in reference slot 1 by the frame before it).
    struct Emitted {
        size_t   index;
        uint32_t duration;
        uint32_t x0, y0, w, h;   // the rectangle coded (the whole frame for the first one)
    };
    std::vector<Emitted> emitted;
    emitted.push_back({0, durations[0], 0, 0, first.width, first.height});
    std::vector<gpu::Buffer> uploads(frames.size());
    const auto upload = [&](size_t i) -> Result<void> {
        if (uploads[i].valid()) return ok();
        std::vector<uint32_t> words(std::max<size_t>(1, (frames[i].pixels.size() + 3) / 4), 0);
        std::memcpy(words.data(), frames[i].pixels.data(), frames[i].pixels.size());
        SJXL_ASSIGN(uploads[i], gpu::Buffer::fromSpan<uint32_t>(dev, words, "enc.animFrame"));
        return ok();
    };
    SJXL_TRY(upload(0));
    for (size_t i = 1; i < frames.size(); ++i) {
        SJXL_TRY(upload(i));
        const size_t prev = emitted.back().index;
        const std::array<uint32_t, 4> start{0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u};
        SJXL_ASSIGN(auto bounds, gpu::Buffer::fromSpan<uint32_t>(dev, start, "enc.changeBounds"));
        gpu::CommandBatch batch(dev);
        changeBounds_.dispatch(batch, {first.width, first.height, 1}, [&](rhi::ShaderCursor c) {
            c["src"].setBinding(uploads[i].rhi());
            c["other"].setBinding(uploads[prev].rhi());
            c["bounds"].setBinding(bounds.rhi());
            c["params"]["srcWidth"].setData(first.width);
            c["params"]["srcHeight"].setData(first.height);
            c["params"]["bytesPerPixel"].setData(bytesPerPixel);
        });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const auto box, bounds.readAll<uint32_t>(dev));
        if (!platform::env("SJXL_DEBUG").empty()) log::info("enc debug: frame {} vs {}: change box x {}..{} y {}..{}", i, prev, box[0], box[2], box[1], box[3]);
        if (box[2] < box[0]) {   // nothing differs
            emitted.back().duration += durations[i];
            uploads[i] = gpu::Buffer{};
            continue;
        }
        Emitted e{i, durations[i], 0, 0, first.width, first.height};
        if (options.cropFrames) {
            e.x0 = box[0];
            e.y0 = box[1];
            e.w = box[2] - box[0] + 1;
            e.h = box[3] - box[1] + 1;
        }
        emitted.push_back(e);
        uploads[prev] = gpu::Buffer{};
    }
    for (size_t j = 0; j < emitted.size(); ++j) {
        const Emitted& e = emitted[j];
        const image::Rgb8& image = frames[e.index];
        const bool partial = e.w != first.width || e.h != first.height;
        const bool nextPartial = j + 1 < emitted.size() && (emitted[j + 1].w != first.width || emitted[j + 1].h != first.height);
        codestream::FrameInfo info;
        info.animated = true;
        info.isLast = j + 1 == emitted.size();
        info.duration = e.duration;
        info.saveAsReference = nextPartial ? 1 : 0;
        std::vector<uint8_t> frame;
        if (!partial) {
            SJXL_ASSIGN(frame, encodeFrame(image, options, info));
        } else {
            info.haveCrop = true;
            info.x0 = static_cast<int32_t>(e.x0);
            info.y0 = static_cast<int32_t>(e.y0);
            info.cropWidth = e.w;
            info.cropHeight = e.h;
            info.blendSource = 1;
            SJXL_TRY(upload(e.index));
            const uint32_t cropWords = (e.w * e.h * bytesPerPixel + 3) / 4;
            SJXL_ASSIGN(auto cropped, gpu::Buffer::ofType<uint32_t>(dev, cropWords, "enc.cropFrame"));
            gpu::CommandBatch batch(dev);
            cropBytes_.dispatch(batch, {cropWords, 1, 1}, [&](rhi::ShaderCursor c) {
                c["src"].setBinding(uploads[e.index].rhi());
                c["dst"].setBinding(cropped.rhi());
                c["params"]["srcWidth"].setData(first.width);
                c["params"]["srcHeight"].setData(first.height);
                c["params"]["bytesPerPixel"].setData(bytesPerPixel);
                c["params"]["x0"].setData(e.x0);
                c["params"]["y0"].setData(e.y0);
                c["params"]["width"].setData(e.w);
                c["params"]["height"].setData(e.h);
            });
            SJXL_TRY(batch.submit(true));
            image::Rgb8 stub;
            stub.width = e.w;
            stub.height = e.h;
            stub.channels = image.channels;
            stub.bitDepth = image.bitDepth;
            stub.gamma = image.gamma;
            SJXL_ASSIGN(frame, encodeFrame(stub, options, info, nullptr, &cropped));
        }
        out.insert(out.end(), frame.begin(), frame.end());
        uploads[e.index] = gpu::Buffer{};
    }
    stats_.frames = static_cast<uint32_t>(emitted.size());
    return out;
}

Result<std::vector<uint8_t>> Encoder::encodeJpeg(const image::JpegFile& jpeg, const Options& options) {
    if (jpeg.progressive) return Error(ErrorCode::Unsupported, "progressive JPEG (only baseline Huffman scans are decoded)");
    if (jpeg.components.size() != 1 && jpeg.components.size() != 3) return Error(ErrorCode::Unsupported, "JPEG component count");
    for (const image::JpegScan& scan : jpeg.scans) {
        if (scan.ss != 0 || scan.se != 63 || scan.ah != 0 || scan.al != 0) return Error(ErrorCode::Unsupported, "JPEG scan spectral selection");
        for (const image::JpegScan::Component& sc : scan.components) {
            if (!jpeg.dcTables[sc.dcTable].present || !jpeg.acTables[sc.acTable].present) return Error(ErrorCode::Corrupt, "JPEG scan without its Huffman tables");
        }
    }
    // The headers: an 8-bit sRGB image, not XYB; then one VarDCT frame fed by the JPEG.
    image::Rgb8 stub;
    stub.width = jpeg.width;
    stub.height = jpeg.height;
    stub.channels = 3;
    stub.bitDepth = 8;
    codestream::BitWriter head;
    codestream::writeCodestreamHeadersRgb(head, jpeg.width, jpeg.height, 8, false);
    std::vector<uint8_t> out = head.take();
    SJXL_ASSIGN(const std::vector<uint8_t> frame, encodeFrame(stub, options, codestream::FrameInfo{}, &jpeg));
    stats_.headBytes += static_cast<uint32_t>(out.size());
    out.insert(out.end(), frame.begin(), frame.end());
    return out;
}

Result<std::vector<uint8_t>> Encoder::encodeFrame(const image::Rgb8& image, const Options& options, const codestream::FrameInfo& info,
                                                   const image::JpegFile* jpeg, const gpu::Buffer* rgbDevice) {
    if (options.lossless && jpeg == nullptr) {
        return encodeLossless(image, options, info, rgbDevice);
    }
    if (jpeg == nullptr && (!(options.distance > 0.0f) || options.distance > 25.0f)) {
        return Error::make(ErrorCode::InvalidArgument, "distance {} is not in (0, 25]", options.distance);
    }
    gpu::Device& dev = *device_;
    codestream::FrameParams frame = frameParamsFor(options);
    frame.useDcFrame = options.progressiveDc && image.channels == 3 && jpeg == nullptr;   // libjxl: no DC frames with extra channels
    // JPEG recompression: a YCbCr (or RGB) frame with the JPEG's subsampling, no filters, no
    // DC smoothing unless 4:4:4 (libjxl's defaults), every block a DCT8 with quant field 1.
    std::array<uint32_t, 4> hshift{}, vshift{};
    std::array<uint32_t, 4> jpegOf{0, 1, 2, 0};
    std::array<float, 3> dcQuantJpeg{};
    std::array<int32_t, 192> rawQuant{};
    std::array<uint32_t, 4> dcOffsetJpeg{};
    uint32_t maxHShift = 0, maxVShift = 0;
    const bool grayJpeg = jpeg != nullptr && jpeg->components.size() == 1;
    if (jpeg != nullptr) {
        frame.xyb = false;
        frame.ycbcr = !jpeg->rgb || grayJpeg;
        frame.gaborish = false;
        frame.epfIters = 0;
        frame.xQmScale = frame.bQmScale = 2;
        // Channel order: X <- Cb, Y <- Y, B <- Cr (JpegOrder); RGB frames keep the component order.
        if (frame.ycbcr && !grayJpeg) jpegOf = {1, 0, 2, 0};
        if (grayJpeg) jpegOf = {0, 0, 0, 0};
        static constexpr uint32_t kH[4] = {0, 1, 1, 0}, kV[4] = {0, 1, 0, 1};
        for (uint32_t c = 0; c < 3; ++c) {
            const image::JpegComponent& comp = jpeg->components[jpegOf[c]];
            uint32_t mode = 4;
            for (uint32_t i = 0; i < 4; ++i) {
                if ((1u << kH[i]) == comp.hSamp && (1u << kV[i]) == comp.vSamp) mode = i;
            }
            if (mode == 4) return Error(ErrorCode::Unsupported, "JPEG sampling factors");
            frame.chromaMode[c] = mode;
            maxHShift = std::max(maxHShift, kH[mode]);
            maxVShift = std::max(maxVShift, kV[mode]);
        }
        for (uint32_t c = 0; c < 3; ++c) {
            hshift[c] = maxHShift - kH[frame.chromaMode[c]];
            vshift[c] = maxVShift - kV[frame.chromaMode[c]];
            const image::JpegComponent& comp = jpeg->components[jpegOf[c]];
            const std::array<int32_t, 64>& q = jpeg->quant[comp.quantIndex];
            dcQuantJpeg[c] = static_cast<float>(q[0]) / (8.0f * 255.0f);
            for (uint32_t y = 0; y < 8; ++y)
                for (uint32_t x = 0; x < 8; ++x) rawQuant[c * 64 + 8 * x + y] = q[8 * y + x];   // libjxl's transposed layout
            dcOffsetJpeg[c] = frame.ycbcr ? 0u : static_cast<uint32_t>(1024 / q[0]);
        }
        frame.skipDcSmoothing = true;   // as cjxl writes every transcode (its lfs option off)
        if (!frame.ycbcr) frame.chromaMode = {0, 0, 0};
    }
    const FrameDim dim = codestream::frameDim(image.width, image.height, 1, maxHShift, maxVShift);
    // The quant field's distance: libjxl lowers it when the Gaborish is off.
    const float fieldDistance = options.distance * (frame.gaborish ? 1.0f : 0.62f);
    // The field's unit: libjxl's adaptive scale, or the flat 0.79 / d of its
    // fast tiers. The quantizer's global scale follows 0.39 / d with the
    // adaptive field (enc_heuristics.cc, speeds up to squirrel), 0.79 / d flat.
    const float fieldScale = options.adaptiveQuant ? codestream::initialQuantAc(fieldDistance) : 0.79f / options.distance;
    const codestream::QuantParams qp = codestream::quantParamsFor(options.adaptiveQuant ? 0.39f / options.distance : fieldScale,
                                                                  codestream::initialQuantDc(options.distance));
    gpu::Buffer rawQuantDev;
    const std::array<uint32_t, 4> quantizerWords = jpeg != nullptr
        ? std::array<uint32_t, 4>{65536u, 1u, std::bit_cast<uint32_t>(1.0f), std::bit_cast<uint32_t>(1.0f)}
        : std::array<uint32_t, 4>{qp.globalScale, qp.quantDc, std::bit_cast<uint32_t>(qp.invGlobalScale), std::bit_cast<uint32_t>(qp.globalScaleFloat)};
    const uint32_t nblocks = dim.numBlocks();
    const uint32_t tilesX = (dim.xsizeBlocks + 7) / 8;
    const uint32_t tilesY = (dim.ysizeBlocks + 7) / 8;
    const uint32_t cellsX = dim.xsizePadded / 4;
    const uint32_t cellsY = dim.ysizePadded / 4;
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const std::vector<PassDef> passes = passesFor(options);
    const uint32_t numPasses = frame.numPasses;
    const uint32_t numSections = dim.numSections(numPasses);
    const bool singleSection = dim.singleSection(numPasses);

    // Input, padded to whole words (or already on the device: a cropped frame).
    gpu::Buffer rgbOwned;
    if (rgbDevice == nullptr) {
        std::vector<uint32_t> rgbWords(std::max<size_t>(1, (image.pixels.size() + 3) / 4), 0);
        std::memcpy(rgbWords.data(), image.pixels.data(), image.pixels.size());
        SJXL_ASSIGN(rgbOwned, gpu::Buffer::fromSpan<uint32_t>(dev, rgbWords, "enc.rgb8"));
    }
    const gpu::Buffer& rgb = rgbDevice != nullptr ? *rgbDevice : rgbOwned;

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
    SJXL_ASSIGN(auto acTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numPasses} * 3 * nblocks * 64, "enc.acTokens"));
    SJXL_ASSIGN(auto numTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numPasses} * 3 * nblocks, "enc.numTokens"));
    SJXL_ASSIGN(auto nzCtx, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numPasses} * 3 * nblocks, "enc.nzCtx"));
    SJXL_ASSIGN(auto passNonZeros, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numPasses} * 3 * nblocks, "enc.passNonZeros"));
    SJXL_ASSIGN(auto cmapX, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapX"));
    SJXL_ASSIGN(auto cmapB, gpu::Buffer::ofType<int32_t>(dev, uint64_t{tilesX} * tilesY, "enc.cmapB"));
    SJXL_ASSIGN(auto acQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks * 64, "enc.acQuant"));
    SJXL_ASSIGN(auto dcQuant, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks, "enc.dcQuant"));
    // The block context map: libjxl's default (15 contexts) unless a JPEG transcode
    // builds one from the luma DC (jpegBlockCtx; read back as one counter).
    SJXL_ASSIGN(auto bctxMap, gpu::Buffer::ofType<uint32_t>(dev, 16 + 3 * 13 * 8, "enc.bctxMap"));
    uint32_t numBlockCtxs = 15;
    uint32_t numAcCtxsUsed = kNumAcContexts;
    SJXL_ASSIGN(auto nonZeros, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.nonZeros"));

    const std::vector<Uint4> acRects = groupRects(dim, codestream::kGroupDimBlocks, dim.xsizeGroups, dim.ysizeGroups);
    const std::vector<Uint4> dcRects = groupRects(dim, codestream::kDcGroupDimBlocks, dim.xsizeDcGroups, dim.ysizeDcGroups);
    SJXL_ASSIGN(auto groups, gpu::Buffer::fromSpan<Uint4>(dev, acRects, "enc.groups"));
    SJXL_ASSIGN(auto dcGroups, gpu::Buffer::fromSpan<Uint4>(dev, dcRects, "enc.dcGroups"));

    // Histogram sets: one per AC pass (the AC contexts), then P + 2g the DC stream of
    // DC group g, P + 1 + 2g its AcMetadata, then the alpha channel's modular stream
    // when there is one.
    const bool hasAlpha = image.channels == 4;
    const bool ecFits = image.width <= dim.groupDim && image.height <= dim.groupDim;
    const uint32_t numEcSets = hasAlpha ? 1 : 0;
    const uint32_t ecSet = numPasses + 2 * numDcGroups;
    const uint32_t numSets = numPasses + 2 * numDcGroups + numEcSets;
    const uint32_t numContextsAll = numPasses * kNumAcContexts + (2 * numDcGroups + numEcSets) * kSetContexts;
    const auto setCtxBase = [&](uint32_t set) {
        return set < numPasses ? set * kNumAcContexts : numPasses * kNumAcContexts + (set - numPasses) * kSetContexts;
    };
    const uint32_t numCodeRegions = numSets * kAcClusters;
    const uint64_t pixels = uint64_t{image.width} * image.height;
    const uint32_t numBlobRegions = 2 + 2 * numDcGroups + 2;   // + the quant table stream and AcGlobal's second part
    const uint32_t qtBlobRegion = 2 + 2 * numDcGroups, acPostRegion = qtBlobRegion + 1;
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
    SJXL_ASSIGN(auto ansInfo, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions + 2} * 256, "enc.ansInfo"));
    SJXL_ASSIGN(auto ansReverse, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numCodeRegions + 2} * 4096, "enc.ansReverse"));
    const std::array<uint32_t, kModTreeWords> qtTreeWords{0xFFFFFFFFu, 0, 0, 0, 0, kPredZero, 0, 0};   // one leaf, predictor Zero
    SJXL_ASSIGN(auto qtTree, gpu::Buffer::fromSpan<uint32_t>(dev, qtTreeWords, "enc.qtTree"));
    // The DC group streams: libjxl's kWPFixedDC tree (weighted predictor) and kACMeta tree, the WP scratch, the tokens.
    uint32_t dcLeaves = 0, metaLeaves = 0;
    const std::vector<uint32_t> dcTreeWords = fixedTree(kModPropWp, kPredWeighted, uint64_t{3} * nblocks, dcLeaves);
    const std::vector<uint32_t> metaTreeWords = acMetaTree(metaLeaves);
    SJXL_ASSIGN(auto dcTree, gpu::Buffer::fromSpan<uint32_t>(dev, dcTreeWords, "enc.dcTree"));
    SJXL_ASSIGN(auto metaTree, gpu::Buffer::fromSpan<uint32_t>(dev, metaTreeWords, "enc.metaTree"));
    const uint32_t wpWordsPerGroup = 5 * 2 * (codestream::kDcGroupDimBlocks + 2);
    SJXL_ASSIGN(auto wpScratch, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numDcGroups} * wpWordsPerGroup, "enc.wpScratch"));
    SJXL_ASSIGN(auto dcTokens, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * nblocks, "enc.dcTokens"));
    // The alpha channel as a modular stream (the lossless machinery over one plane).
    const uint64_t alphaPixels = hasAlpha ? pixels : 4;
    SJXL_ASSIGN(auto alphaPlane, gpu::Buffer::ofType<int32_t>(dev, alphaPixels, "enc.alphaPlane"));
    SJXL_ASSIGN(auto alphaTokens, gpu::Buffer::ofType<uint32_t>(dev, alphaPixels, "enc.alphaTokens"));
    SJXL_ASSIGN(auto alphaPassTokens, gpu::Buffer::ofType<uint32_t>(dev, 2 * alphaPixels, "enc.alphaPassTokens"));
    SJXL_ASSIGN(auto alphaPassHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{2} * kSetContexts * kHistBins, "enc.alphaPassHist"));
    SJXL_ASSIGN(auto alphaPassBits, gpu::Buffer::ofType<float>(dev, 2, "enc.alphaPassBits"));
    const uint32_t alphaWpWords = 5 * 2 * (dim.groupDim + 2);
    SJXL_ASSIGN(auto alphaWpScratch, gpu::Buffer::ofType<uint32_t>(dev, hasAlpha ? uint64_t{numGroups} * alphaWpWords : 4, "enc.alphaWp"));
    SJXL_ASSIGN(auto acOnlyBits, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numGroups} * numPasses, "enc.acOnlyBits"));
    uint32_t alphaLeaves = 0, alphaNodes = 0;
    gpu::Buffer alphaTree, alphaChanTable, alphaStreamTokens, alphaStreamStart, alphaStreamLen;
    const std::vector<uint32_t> cfgDefault(numCodeRegions + 1, kDefaultUintCfg);
    SJXL_ASSIGN(auto clusterCfg, gpu::Buffer::fromSpan<uint32_t>(dev, cfgDefault, "enc.clusterCfg"));
    SJXL_ASSIGN(alphaTree, gpu::Buffer::ofType<uint32_t>(dev, 8, "enc.alphaTree"));
    SJXL_ASSIGN(auto acSectionBits, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numGroups} * numPasses, "enc.acSectionBits"));

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
    gpu::StageTimer timer(batch, "encode");

    if (jpeg == nullptr) rgbToXyb_.dispatch(batch, {dim.xsizePadded, dim.ysizePadded, 1}, [&](rhi::ShaderCursor c) {
        c["rgb8"].setBinding(rgb.rhi());
        c["xyb"].setBinding(xyb.rhi());
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["transfer"].setData(uint32_t{image.gamma > 0.0f ? 2u : 0u});
        c["params"]["gammaInv"].setData(image.gamma > 0.0f ? 1.0f / image.gamma : 1.0f);
        c["params"]["numChannels"].setData(image.channels);
        c["params"]["bitDepth"].setData(image.bitDepth);
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
        timer.mark("upload + xyb");
        if (jpeg == nullptr) aqDiff_.dispatch(batch, {cellsX * cellsY, 1, 1}, bindAq);
    }
    if (jpeg == nullptr) aqMap_.dispatch(batch, {nblocks, 1, 1}, bindAq);
    if (jpeg == nullptr) aqRawField_.dispatch(batch, {nblocks, 1, 1}, bindAq);

    // Pre-sharpening for the decoder's Gaborish, then the DCT of every block.
    timer.mark("adaptive quant");
    const gpu::Buffer& planes = frame.gaborish ? xybSharp : xyb;
    if (frame.gaborish) {
        if (jpeg == nullptr) gaborishInverse_.dispatch(batch, {dim.xsizePadded, dim.ysizePadded, 1}, [&](rhi::ShaderCursor c) {
            c["src"].setBinding(xyb.rhi());
            c["dst"].setBinding(xybSharp.rhi());
            c["params"]["paddedWidth"].setData(dim.xsizePadded);
            c["params"]["paddedHeight"].setData(dim.ysizePadded);
        });
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
        c["passNonZeros"].setBinding(passNonZeros.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["bctxMap"].setBinding(bctxMap.rhi());
        c["dcQuant"].setBinding(dcQuant.rhi());
        c["params"]["numBlockCtxs"].setData(numBlockCtxs);
        c["params"]["jpegCtx"].setData(uint32_t{jpeg != nullptr ? 1u : 0u});
        c["params"]["paddedWidth"].setData(dim.xsizePadded);
        c["params"]["paddedHeight"].setData(dim.ysizePadded);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["distance"].setData(options.distance);
    };
    // The pass split (SplitACCoefficients): what this pass codes and what the earlier ones did.
    const auto bindPass = [&](rhi::ShaderCursor c, uint32_t p) {
        bindTransform(c);
        uint32_t prevExact = 1;   // ncoeffs_all_done_from_earlier_passes
        for (uint32_t i = 0; i < p; ++i) {
            if (passes[i].shift == 0) prevExact = passes[i].ncoeffs;
        }
        c["params"]["pass"].setData(p);
        c["params"]["shift"].setData(passes[p].shift);
        c["params"]["prevShift"].setData(p == 0 ? 0u : passes[p - 1].shift);
        c["params"]["ncoeffs"].setData(passes[p].ncoeffs);
        c["params"]["prevNcoeffs"].setData(prevExact);
        c["params"]["ctxBase"].setData(p * kNumAcContexts);
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
    };
    const auto bindCfl = [&](rhi::ShaderCursor c) {
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
    };
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
    SJXL_ASSIGN(auto candNz, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{3} * regionsX * regionsY * 5, "enc.candNz"));
    SJXL_ASSIGN(auto largeY, gpu::Buffer::ofType<float>(dev, static_cast<uint64_t>(planePixels), "enc.largeY"));
    uint32_t familyMask = 0x3FF;
    if (const std::string m = platform::env("SJXL_ACS_MASK"); !m.empty()) familyMask = static_cast<uint32_t>(std::strtoul(m.c_str(), nullptr, 0));
    timer.mark("inverse gaborish");
    const uint32_t threeChannels = platform::env("SJXL_ACS_YONLY").empty() ? 1u : 0u;
    const uint32_t acsDeadZone = platform::env("SJXL_ACS_ROUND").empty() ? 1u : 0u;   // the coder's dead zones in the estimate (+0.3 SSIMULACRA2 over libjxl's rounding)
    const auto bindStrategy = [&](rhi::ShaderCursor c, bool enabled) {
        c["xyb"].setBinding(planes.rhi());
        c["rawQf"].setBinding(rawQf.rhi());
        c["qf"].setBinding(qfField.rhi());
        c["cmapX"].setBinding(cmapX.rhi());
        c["cmapB"].setBinding(cmapB.rhi());
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
        c["params"]["enabled"].setData(uint32_t{enabled ? 1u : 0u});
        c["params"]["familyMask"].setData(familyMask & 0x7Fu);
        c["params"]["specials"].setData(uint32_t{options.acsSpecials ? 1u : 0u});
        c["params"]["tilesX"].setData(tilesX);
        c["params"]["threeChannels"].setData(threeChannels);
        c["params"]["deadZone"].setData(acsDeadZone);
    };
    // A first chroma-from-luma pass on DCT8 coefficients (libjxl's ACS runs after
    // an initial CfL): the strategy search scores X and B as residuals of it.
    if (jpeg == nullptr && options.acStrategy && options.chromaFromLuma && threeChannels != 0) {
        selectStrategies_.dispatchGroups(batch, {superblocksX * superblocksY, 1, 1}, [&](rhi::ShaderCursor c) { bindStrategy(c, false); });
        dctRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
        dctCols_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
        cflTile_.dispatch(batch, {tilesX * tilesY, 1, 1}, bindCfl);
    }
    timer.mark("first cfl pass");
    if (jpeg == nullptr) selectStrategies_.dispatchGroups(batch, {superblocksX * superblocksY, 1, 1}, [&](rhi::ShaderCursor c) { bindStrategy(c, true); });
    // The 64-pixel families against the superblocks' layouts.
    timer.mark("strategy 8..32");
    if (options.acStrategy && (familyMask >> 7) != 0) {
        const auto bindLarge = [&](rhi::ShaderCursor c, uint32_t family, uint32_t channelPass) {
            c["xyb"].setBinding(planes.rhi());
            c["rawQf"].setBinding(rawQf.rhi());
            c["qf"].setBinding(qfField.rhi());
            c["cmapX"].setBinding(cmapX.rhi());
            c["cmapB"].setBinding(cmapB.rhi());
            c["largeY"].setBinding(largeY.rhi());
            c["params"]["channelPass"].setData(channelPass);
            c["params"]["tilesX"].setData(tilesX);
            c["params"]["threeChannels"].setData(threeChannels);
            c["params"]["deadZone"].setData(acsDeadZone);
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
        if (jpeg == nullptr) largeQf_.dispatch(batch, {regionsX * regionsY * 5, 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, 0, 0); });
        for (uint32_t family = 0; family < 3; ++family) {
            if (jpeg == nullptr) largeRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, family, 0); });
            if (jpeg == nullptr) largeCols_.dispatch(batch, {static_cast<uint32_t>(planePixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, family, 0); });
            if (jpeg == nullptr && threeChannels != 0) {
                largeCols_.dispatch(batch, {static_cast<uint32_t>(2 * planePixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, family, 1); });
            }
        }
        if (jpeg == nullptr) mergeLarge_.dispatch(batch, {regionsX * regionsY, 1, 1}, [&](rhi::ShaderCursor c) { bindLarge(c, 0, 0); });
    }
    timer.mark("strategy 64");
    if (jpeg == nullptr) adjustQuantField_.dispatch(batch, {nblocks, 1, 1}, bindTransform);
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
    if (jpeg == nullptr) dctRows_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
    if (jpeg == nullptr) dctCols_.dispatch(batch, {static_cast<uint32_t>(3 * planePixels), 1, 1}, bindTransform);
    timer.mark("dct");
    if (jpeg == nullptr) specialTransform_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransform);
    if (jpeg == nullptr) dcFromLlf_.dispatch(batch, {3 * nblocks, 1, 1}, bindTransform);
    if (options.chromaFromLuma) {
        if (jpeg == nullptr) cflTile_.dispatch(batch, {tilesX * tilesY, 1, 1}, bindCfl);
    }
    timer.mark("cfl");
    if (jpeg == nullptr) quantize_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
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
    if (jpeg != nullptr) {
        // The JPEG's Huffman scans decoded on the device, then its coefficients laid out as the frame's.
        std::vector<uint32_t> fileWords((jpeg->data.size() + 3) / 4, 0);
        std::memcpy(fileWords.data(), jpeg->data.data(), jpeg->data.size());
        SJXL_ASSIGN(auto jpegData, gpu::Buffer::fromSpan<uint32_t>(dev, fileWords, "enc.jpegData"));
        // Huffman tables: limit (largest code + 1) and value offset per length, then the values.
        constexpr uint32_t kHuffWords = 18 + 18 + 256;
        std::vector<uint32_t> huff(8 * kHuffWords, 0);
        for (uint32_t t = 0; t < 8; ++t) {
            const image::JpegHuffmanTable& table = t < 4 ? jpeg->dcTables[t] : jpeg->acTables[t - 4];
            if (!table.present) continue;
            uint32_t code = 0, index = 0;
            for (uint32_t len = 1; len <= 16; ++len) {
                const uint32_t n = table.counts[len];
                huff[t * kHuffWords + len] = n ? code + n : 0;                                   // limit
                huff[t * kHuffWords + 18 + len] = n ? static_cast<uint32_t>(index - code) : 0;   // value index = offset + code (mod 256)
                code += n;
                index += n;
                code <<= 1;
            }
            for (uint32_t i = 0; i < table.values.size() && i < 256; ++i) huff[t * kHuffWords + 36 + i] = table.values[i];
        }
        SJXL_ASSIGN(auto huffman, gpu::Buffer::fromSpan<uint32_t>(dev, huff, "enc.jpegHuffman"));
        std::array<uint32_t, 4> compH{1, 1, 1, 1}, compV{1, 1, 1, 1}, compBlocksX{}, compBlocksY{}, compOffset{}, compScanBlocksX{}, compScanBlocksY{};
        uint32_t totalCoeffs = 0;
        uint32_t hmax = 1, vmax = 1;
        for (const image::JpegComponent& comp : jpeg->components) { hmax = std::max(hmax, comp.hSamp); vmax = std::max(vmax, comp.vSamp); }
        for (uint32_t c = 0; c < jpeg->components.size(); ++c) {
            const image::JpegComponent& comp = jpeg->components[c];
            compH[c] = comp.hSamp;
            compV[c] = comp.vSamp;
            compBlocksX[c] = comp.blocksX;
            compBlocksY[c] = comp.blocksY;
            compOffset[c] = totalCoeffs;
            const uint32_t cw = (jpeg->width * comp.hSamp + hmax - 1) / hmax, ch = (jpeg->height * comp.vSamp + vmax - 1) / vmax;
            compScanBlocksX[c] = (cw + 7) / 8;
            compScanBlocksY[c] = (ch + 7) / 8;
            totalCoeffs += comp.blocksX * comp.blocksY * 64;
        }
        SJXL_ASSIGN(auto coeffs, gpu::Buffer::ofType<int32_t>(dev, totalCoeffs, "enc.jpegCoeffs"));
        SJXL_ASSIGN(auto rawQuantBuf, gpu::Buffer::fromSpan<int32_t>(dev, rawQuant, "enc.rawQuant"));
        rawQuantDev = std::move(rawQuantBuf);
        const auto bindJpeg = [&](rhi::ShaderCursor c, const gpu::Buffer& intervals, const image::JpegScan* scan) {
            c["data"].setBinding(jpegData.rhi());
            c["intervalStart"].setBinding(intervals.rhi());
            c["huffman"].setBinding(huffman.rhi());
            c["coeffs"].setBinding(coeffs.rhi());
            c["acQuant"].setBinding(acQuant.rhi());
            c["dcQuant"].setBinding(dcQuant.rhi());
            c["nonZeros"].setBinding(nonZeros.rhi());
            c["acs"].setBinding(acs.rhi());
            c["firstBlock"].setBinding(firstBlock.rhi());
            c["rawQf"].setBinding(rawQf.rhi());
            c["cmapX"].setBinding(cmapX.rhi());
            c["cmapB"].setBinding(cmapB.rhi());
            c["status"].setBinding(dummy_.rhi());
            c["rawQuant"].setBinding(rawQuantDev.rhi());
            c["blockCtx"].setBinding(bctxMap.rhi());
            c["params"]["numComponents"].setData(static_cast<uint32_t>(jpeg->components.size()));
            c["params"]["mcusX"].setData(jpeg->mcusX);
            c["params"]["mcusY"].setData(jpeg->mcusY);
            c["params"]["restartInterval"].setData(jpeg->restartInterval);
            std::array<uint32_t, 4> idx{}, dct{}, act{};
            uint32_t numIntervals = 1, scanComponents = 0, dataOffset = 0, dataSize = 0;
            if (scan != nullptr) {
                for (uint32_t i = 0; i < scan->components.size() && i < 4; ++i) {
                    idx[i] = scan->components[i].index;
                    dct[i] = scan->components[i].dcTable;
                    act[i] = scan->components[i].acTable;
                }
                numIntervals = static_cast<uint32_t>(scan->restartOffsets.size());
                scanComponents = static_cast<uint32_t>(scan->components.size());
                dataOffset = static_cast<uint32_t>(scan->dataOffset);
                dataSize = static_cast<uint32_t>(scan->dataSize);
            }
            c["params"]["numIntervals"].setData(numIntervals);
            c["params"]["scanComponents"].setData(scanComponents);
            c["params"]["dataOffset"].setData(dataOffset);
            c["params"]["dataSize"].setData(dataSize);
            c["params"]["compIndex"].setData(idx);
            c["params"]["compDcTable"].setData(dct);
            c["params"]["compAcTable"].setData(act);
            c["params"]["compH"].setData(compH);
            c["params"]["compV"].setData(compV);
            c["params"]["compBlocksX"].setData(compBlocksX);
            c["params"]["compBlocksY"].setData(compBlocksY);
            c["params"]["compOffset"].setData(compOffset);
            c["params"]["compScanBlocksX"].setData(compScanBlocksX);
            c["params"]["compScanBlocksY"].setData(compScanBlocksY);
            c["params"]["blocksX"].setData(dim.xsizeBlocks);
            c["params"]["blocksY"].setData(dim.ysizeBlocks);
            c["params"]["hshift"].setData(hshift);
            c["params"]["vshift"].setData(vshift);
            c["params"]["jpegOf"].setData(jpegOf);
            c["params"]["dcOffset"].setData(dcOffsetJpeg);
            c["params"]["grayscale"].setData(uint32_t{grayJpeg ? 1u : 0u});
        };
        std::vector<gpu::Buffer> intervalBuffers;
        for (const image::JpegScan& scan : jpeg->scans) {
            SJXL_ASSIGN(auto intervals, gpu::Buffer::fromSpan<uint32_t>(dev, scan.restartOffsets, "enc.jpegIntervals"));
            intervalBuffers.push_back(std::move(intervals));
            jpegDecodeScan_.dispatch(batch, {static_cast<uint32_t>(scan.restartOffsets.size()), 1, 1},
                                     [&](rhi::ShaderCursor c) { bindJpeg(c, intervalBuffers.back(), &scan); });
        }
        jpegToCoeffs_.dispatch(batch, {3 * nblocks, 1, 1}, [&](rhi::ShaderCursor c) { bindJpeg(c, intervalBuffers.back(), nullptr); });
        jpegBlockCtx_.dispatchGroups(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindJpeg(c, intervalBuffers.back(), nullptr); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const auto bctxHead, bctxMap.readRange<uint32_t>(dev, 0, 2));
        numBlockCtxs = bctxHead[1];
        numAcCtxsUsed = numBlockCtxs * (37 + 458);
        if (!platform::env("SJXL_DEBUG").empty()) log::info("enc debug: jpeg block contexts: {} DC buckets, {} contexts", bctxHead[0], numBlockCtxs);
    }
    // The DC frame: the DC image as a Modular XYB frame at 1/8 size, scaled by the
    // frame's DC steps (F16 in its DcGlobal); the VarDCT frame then codes no DC.
    std::vector<uint8_t> dcFrameBytes;
    if (frame.useDcFrame) {
        std::array<float, 3> dcQuantF{};
        for (uint32_t c = 0; c < 3; ++c) dcQuantF[c] = 1.0f / (codestream::kInvDcQuant[c] * qp.globalScaleFloat * static_cast<float>(qp.quantDc));
        SJXL_ASSIGN(auto dcPlanes, gpu::Buffer::ofType<int32_t>(dev, uint64_t{3} * nblocks, "enc.dcFramePlanes"));
        dcFramePlanes_.dispatch(batch, {nblocks, 1, 1}, [&](rhi::ShaderCursor c) {
            for (const char* name : {"rgb8", "planesIn", "treeNodes", "wpScratch", "passTokens", "passHist", "passBits", "choice", "leafMap", "modTokens",
                                     "ctxHist", "ctxMap", "ansInfo", "ansReverse", "groupStart", "groupBits", "body", "thresholds", "samples", "sampleCount",
                                     "rctCosts", "groupRct", "chanTable"}) {
                c[name].setBinding(dummy_.rhi());
            }
            c["dcValue"].setBinding(dcValue.rhi());
            c["planes"].setBinding(dcPlanes.rhi());
            c["params"]["width"].setData(dim.xsizeBlocks);
            c["params"]["height"].setData(dim.ysizeBlocks);
            c["params"]["dcQuantX"].setData(dcQuantF[0]);
            c["params"]["dcQuantY"].setData(dcQuantF[1]);
            c["params"]["dcQuantB"].setData(dcQuantF[2]);
        });
        SJXL_TRY(batch.submit(true));
        ModularSpec spec;
        spec.width = dim.xsizeBlocks;
        spec.height = dim.ysizeBlocks;
        spec.bitDepth = 16;
        spec.groupSizeShift = codestream::modularGroupSizeShift(dim.xsizeBlocks, dim.ysizeBlocks);
        const codestream::FrameDim dcDim = codestream::frameDim(dim.xsizeBlocks, dim.ysizeBlocks, spec.groupSizeShift);
        const bool dcGrouped = !(dim.xsizeBlocks <= dcDim.groupDim && dim.ysizeBlocks <= dcDim.groupDim);
        for (uint32_t c = 0; c < 3; ++c) spec.channels.push_back(ModularChannel{dim.xsizeBlocks, dim.ysizeBlocks, c * nblocks, dcGrouped});
        spec.rct = false;
        spec.dcFrame = true;
        spec.dcQuant = dcQuantF;
        const Stats mainStats = stats_;
        SJXL_ASSIGN(dcFrameBytes, encodeModularFrame(spec, dcPlanes, options, codestream::FrameInfo{}));
        stats_ = mainStats;
        stats_.dcFrameBytes = static_cast<uint32_t>(dcFrameBytes.size());
    }
    for (uint32_t p = 0; p < numPasses; ++p) {
        passNonZeroCount_.dispatch(batch, {3 * nblocks, 1, 1}, [&](rhi::ShaderCursor c) { bindPass(c, p); });
        acTokenize_.dispatch(batch, {3 * nblocks, 1, 1}, [&](rhi::ShaderCursor c) { bindPass(c, p); });
    }

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
        c["params"]["numAcSets"].setData(numPasses);
        c["params"]["useDcFrame"].setData(uint32_t{frame.useDcFrame ? 1u : 0u});
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
    };
    timer.mark("quantize + ac tokens");
    if (!frame.useDcFrame) dcTokenize_.dispatch(batch, {numDcGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindDc(c, 0, dummy_); });
    metaTokenize_.dispatch(batch, {metaItems, 1, 1}, [&](rhi::ShaderCursor c) { bindDc(c, 0, dummy_); });
    if (hasAlpha) {
        // The alpha channel: both predictor passes, the choice (bookkeeping on two
        // numbers), the frame's global tree and the remapped tokens in the alpha set.
        uint32_t wpLeaves = 0, gradLeaves = 0;
        const std::vector<uint32_t> wpTree = fixedTree(kModPropWp, kPredWeighted, pixels, wpLeaves);
        const std::vector<uint32_t> gradTree = fixedTree(kModPropGradient, kPredGradient, pixels, gradLeaves);
        SJXL_ASSIGN(auto wpTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, wpTree, "enc.alphaWpTree"));
        SJXL_ASSIGN(auto gradTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, gradTree, "enc.alphaGradTree"));
        SJXL_ASSIGN(auto choice, gpu::Buffer::ofType<uint32_t>(dev, 1, "enc.alphaChoice"));
        SJXL_ASSIGN(auto leafMapBuf, gpu::Buffer::ofType<uint32_t>(dev, kSetContexts, "enc.alphaLeafMap"));
        const std::array<Uint4, 1> alphaTableHost{Uint4{image.width, image.height, 0, ecFits ? 0u : 1u}};
        SJXL_ASSIGN(alphaChanTable, gpu::Buffer::fromSpan<Uint4>(dev, alphaTableHost, "enc.alphaChanTable"));
        const auto bindAlpha = [&](rhi::ShaderCursor c, const gpu::Buffer& tree, uint32_t pass, const gpu::Buffer& choiceBuf,
                                   const gpu::Buffer& leafMapB) {
            c["rgb8"].setBinding(rgb.rhi());
            c["chanTable"].setBinding(alphaChanTable.rhi());
            c["params"]["totalSamples"].setData(static_cast<uint32_t>(pixels));
            c["params"]["rctBeginC"].setData(0u);
            c["params"]["nbMeta"].setData(0u);
            c["params"]["rct"].setData(0u);
            c["planes"].setBinding(alphaPlane.rhi());
            c["planesIn"].setBinding(alphaPlane.rhi());
            c["treeNodes"].setBinding(tree.rhi());
            c["wpScratch"].setBinding(alphaWpScratch.rhi());
            c["passTokens"].setBinding(alphaPassTokens.rhi());
            c["passHist"].setBinding(alphaPassHist.rhi());
            c["passBits"].setBinding(alphaPassBits.rhi());
            c["choice"].setBinding(choiceBuf.rhi());
            c["leafMap"].setBinding(leafMapB.rhi());
            c["modTokens"].setBinding(alphaTokens.rhi());
            c["ctxHist"].setBinding(ctxHist.rhi());
            c["ctxMap"].setBinding(ctxMap.rhi());
            c["ansInfo"].setBinding(ansInfo.rhi());
            c["ansReverse"].setBinding(ansReverse.rhi());
            c["groupStart"].setBinding(dummy_.rhi());
            c["groupBits"].setBinding(dummy_.rhi());
            c["body"].setBinding(dummy_.rhi());
            for (const char* name : {"dcValue", "thresholds", "samples", "sampleCount", "rctCosts", "groupRct", "streamTokens", "streamStart", "streamLen"}) {
                c[name].setBinding(dummy_.rhi());
            }
            c["clusterCfg"].setBinding(clusterCfg.rhi());
            c["params"]["lz77"].setData(0u);
            c["params"]["width"].setData(image.width);
            c["params"]["height"].setData(image.height);
            c["params"]["groupDim"].setData(dim.groupDim);
            c["params"]["groupsX"].setData(dim.xsizeGroups);
            c["params"]["numGroups"].setData(numGroups);
            c["params"]["numLeaves"].setData(alphaLeaves);
            c["params"]["mode"].setData(0u);
            c["params"]["numChannels"].setData(image.channels);
            c["params"]["bitDepth"].setData(image.bitDepth);
            c["params"]["wpWordsPerGroup"].setData(alphaWpWords);
            c["params"]["pass"].setData(pass);
            c["params"]["ctxBase"].setData(setCtxBase(ecSet));
        };
        extractAlpha_.dispatch(batch, {image.width, image.height, 1}, [&](rhi::ShaderCursor c) { bindAlpha(c, wpTreeBuf, 0, choice, leafMapBuf); });
        // modTokenize reads one channel: the alpha plane is plane 0 of a one-channel image.
        const auto bindAlphaOne = [&](rhi::ShaderCursor c, const gpu::Buffer& tree, uint32_t pass, const gpu::Buffer& choiceBuf,
                                      const gpu::Buffer& leafMapB) {
            bindAlpha(c, tree, pass, choiceBuf, leafMapB);
            c["params"]["numChannels"].setData(1u);
        };
        modTokenize_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindAlphaOne(c, wpTreeBuf, 0, choice, leafMapBuf); });
        modTokenize_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindAlphaOne(c, gradTreeBuf, 1, choice, leafMapBuf); });
        channelEntropy_.dispatch(batch, {2, 1, 1}, [&](rhi::ShaderCursor c) { bindAlphaOne(c, wpTreeBuf, 0, choice, leafMapBuf); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const std::vector<float> bitsHost, alphaPassBits.readAll<float>(dev));
        const uint32_t chosen = bitsHost[0] <= bitsHost[1] ? 0u : 1u;
        const std::vector<uint32_t> treeWords = chosen == 0 ? wpTree : gradTree;
        alphaLeaves = chosen == 0 ? wpLeaves : gradLeaves;
        alphaNodes = static_cast<uint32_t>(treeWords.size() / kModTreeWords);
        std::vector<uint32_t> leafMapHost(kSetContexts);
        for (uint32_t j = 0; j < kSetContexts; ++j) leafMapHost[j] = j;
        const std::vector<uint32_t> choiceHost{chosen};
        SJXL_ASSIGN(alphaTree, gpu::Buffer::fromSpan<uint32_t>(dev, treeWords, "enc.alphaTree"));
        SJXL_ASSIGN(auto choiceB, gpu::Buffer::fromSpan<uint32_t>(dev, choiceHost, "enc.alphaChoice"));
        SJXL_ASSIGN(auto leafMapB, gpu::Buffer::fromSpan<uint32_t>(dev, leafMapHost, "enc.alphaLeafMap"));
        remapTokens_.dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindAlphaOne(c, alphaTree, 0, choiceB, leafMapB); });
        stats_.alphaPredictor = chosen;
        // The global modular stream's writer reads a compact token list: the alpha
        // channel's tokens in stream order (plain, no run-length coding here).
        {
            const std::vector<uint32_t> starts(numGroups + 1, 0);
            SJXL_ASSIGN(alphaStreamStart, gpu::Buffer::fromSpan<uint32_t>(dev, starts, "enc.alphaStreamStart"));
            SJXL_ASSIGN(alphaStreamLen, gpu::Buffer::ofType<uint32_t>(dev, numGroups + 1, "enc.alphaStreamLen"));
            SJXL_ASSIGN(alphaStreamTokens, gpu::Buffer::ofType<uint32_t>(dev, pixels, "enc.alphaStreamTokens"));
            SJXL_ASSIGN(auto rleStats, gpu::Buffer::ofType<uint32_t>(dev, 2, "enc.alphaRleStats"));
            rleScan_.dispatch(batch, {numGroups + 1, 1, 1}, [&](rhi::ShaderCursor c) {
                c["modTokens"].setBinding(alphaTokens.rhi());
                c["chanTable"].setBinding(alphaChanTable.rhi());
                c["streamStart"].setBinding(alphaStreamStart.rhi());
                c["streamLen"].setBinding(alphaStreamLen.rhi());
                c["streamTokens"].setBinding(alphaStreamTokens.rhi());
                c["rleStats"].setBinding(rleStats.rhi());
                for (const char* name : {"ctxHist", "ctxMap", "cfgHist", "cfgExtra", "lengthHist", "hist", "clusterCfg", "maxToken"}) c[name].setBinding(dummy_.rhi());
                c["params"]["numStreams"].setData(numGroups + 1);
                c["params"]["numContexts"].setData(alphaLeaves);
                c["params"]["mode"].setData(0u);
                c["params"]["numClusters"].setData(0u);
                c["params"]["totalTokens"].setData(static_cast<uint32_t>(pixels));
                c["params"]["lz77"].setData(0u);
                c["params"]["regionBase"].setData(0u);
                c["params"]["ctxBase"].setData(0u);
                c["params"]["numChannels"].setData(1u);
                c["params"]["groupDim"].setData(dim.groupDim);
                c["params"]["groupsX"].setData(dim.xsizeGroups);
                c["params"]["numGroups"].setData(numGroups);
                c["params"]["width"].setData(image.width);
                c["params"]["height"].setData(image.height);
            });
        }
    }
    const auto bindSections = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& body) {
        c["groups"].setBinding(groups.rhi());
        c["acs"].setBinding(acs.rhi());
        c["nonZeros"].setBinding(passNonZeros.rhi());
        c["acTokens"].setBinding(acTokens.rhi());
        c["numTokens"].setBinding(numTokens.rhi());
        c["nzCtx"].setBinding(nzCtx.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["sectionStart"].setBinding(sectionStart.rhi());
        c["ecTokens"].setBinding(alphaTokens.rhi());
        c["acSectionBits"].setBinding(acSectionBits.rhi());
        c["acOnlyBits"].setBinding(acOnlyBits.rhi());
        c["body"].setBinding(body.rhi());
        c["params"]["nblocks"].setData(nblocks);
        c["params"]["blocksX"].setData(dim.xsizeBlocks);
        c["params"]["blocksY"].setData(dim.ysizeBlocks);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numDcGroups"].setData(numDcGroups);
        c["params"]["mode"].setData(mode);
        c["params"]["ecInGroups"].setData(uint32_t{hasAlpha && !ecFits ? 1u : 0u});
        c["params"]["ecCtxBase"].setData(setCtxBase(ecSet));
        c["params"]["ecRegionBase"].setData(ecSet * kAcClusters);
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["numPasses"].setData(numPasses);
        c["params"]["hshift"].setData(hshift);
        c["params"]["vshift"].setData(vshift);
    };
    timer.mark("dc + meta tokens");
    // The ANS sections (one thread per AC group, one per DC group) measured in mode 1 and emitted in mode 2.
    const auto sectionPasses = [&](uint32_t mode, const gpu::Buffer& body) {
        acSections_.dispatch(batch, {numGroups * numPasses, 1, 1}, [&](rhi::ShaderCursor c) { bindSections(c, mode, body); });
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
        c["params"]["ctxBase"].setData(setCtxBase(set));
        c["params"]["infoIndex"].setData(set);
        c["params"]["regionBase"].setData(set * kAcClusters);
    };
    for (uint32_t set = 0; set < numSets; ++set) {
        const uint32_t numContexts = set < numPasses ? numAcCtxsUsed : kSetContexts;
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
        c["bctxMap"].setBinding(bctxMap.rhi());
        c["params"]["numAcCtxsUsed"].setData(numAcCtxsUsed);
        c["ctxMapMtf"].setBinding(ctxMapMtf.rhi());
        c["groupRct"].setBinding(dummy_.rhi());
        c["chanTable"].setBinding(alphaChanTable.valid() ? alphaChanTable.rhi() : dummy_.rhi());
        c["transforms"].setBinding(dummy_.rhi());
        c["streamTokens"].setBinding(alphaStreamTokens.valid() ? alphaStreamTokens.rhi() : dummy_.rhi());
        c["streamStart"].setBinding(alphaStreamStart.valid() ? alphaStreamStart.rhi() : dummy_.rhi());
        c["streamLen"].setBinding(alphaStreamLen.valid() ? alphaStreamLen.rhi() : dummy_.rhi());
        c["clusterCfg"].setBinding(clusterCfg.rhi());
        c["params"]["numTransforms"].setData(0u);
        c["params"]["totalSamples"].setData(static_cast<uint32_t>(pixels));
        c["params"]["lz77Modular"].setData(0u);
        c["params"]["modLogAlpha"].setData(uint32_t{image.bitDepth > 8 ? 7u : 6u});
        c["params"]["histogramTries"].setData(7u);
        c["rawQuant"].setBinding(rawQuantDev.valid() ? rawQuantDev.rhi() : dummyInt_.rhi());
        c["qtTree"].setBinding(qtTree.rhi());
        c["params"]["jpeg"].setData(uint32_t{jpeg != nullptr ? 1u : 0u});
        c["params"]["qtRegion"].setData(numCodeRegions + 1);
        c["params"]["qtCount"].setData(192u);
        c["params"]["numQuantKinds"].setData(17u);
        c["params"]["qtBlobRegion"].setData(qtBlobRegion);
        c["params"]["acPostRegion"].setData(acPostRegion);
        c["params"]["dcQuantJpegX"].setData(dcQuantJpeg[0]);
        c["params"]["dcQuantJpegY"].setData(dcQuantJpeg[1]);
        c["params"]["dcQuantJpegB"].setData(dcQuantJpeg[2]);
        c["params"]["rctBeginC"].setData(0u);
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["planes"].setBinding(dummyInt_.rhi());
        c["treeNodes"].setBinding(alphaTree.rhi());
        c["modTokens"].setBinding(alphaTokens.rhi());
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
        c["params"]["numEcSets"].setData(numEcSets);
        c["params"]["numAcSets"].setData(numPasses);
        c["params"]["useDcFrame"].setData(uint32_t{frame.useDcFrame ? 1u : 0u});
        c["params"]["rct"].setData(0u);
        c["params"]["dcQuantCustom"].setData(0u);
        c["params"]["dcQuantX"].setData(0.0f);
        c["params"]["dcQuantY"].setData(0.0f);
        c["params"]["dcQuantB"].setData(0.0f);
        c["params"]["ecInGlobal"].setData(uint32_t{hasAlpha && ecFits ? 1u : 0u});
        c["params"]["ecLogAlpha"].setData(uint32_t{image.bitDepth > 8 ? 7u : 6u});
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["numTreeNodes"].setData(alphaNodes);
        c["params"]["numLeaves"].setData(alphaLeaves);
        c["params"]["numChannels"].setData(1u);
        c["params"]["bitDepth"].setData(image.bitDepth);
    };
    timer.mark("sections + cluster");
    dcGroupPre_.dispatchGroups(batch, {numDcGroups, 1, 1}, bindCodes);
    dcGroupMid_.dispatchGroups(batch, {numDcGroups, 1, 1}, bindCodes);
    globalBlobs_.dispatchGroups(batch, {1, 1, 1}, bindCodes);
    if (jpeg != nullptr) quantTableBlob_.dispatchGroups(batch, {1, 1, 1}, bindCodes);

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
        c["params"]["packed"].setData(uint32_t{singleSection ? 1u : 0u});
        c["params"]["numPasses"].setData(numPasses);
        c["params"]["qtBlobRegion"].setData(qtBlobRegion);
        c["params"]["acPostRegion"].setData(acPostRegion);
        c["params"]["numBlobRegions"].setData(numBlobRegions);
    };
    timer.mark("codes");
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
        log::info("enc debug: blob bits DcGlobal {} AcGlobal {}; section bytes {}", blobBitsHost[0],
                  blobBitsHost[1] + blobBitsHost[qtBlobRegion] + blobBitsHost[acPostRegion], sections);
        SJXL_ASSIGN(const auto passNzHost, passNonZeros.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto numTokensHost, numTokens.readAll<uint32_t>(dev));
        SJXL_ASSIGN(const auto acOnlyHost, acOnlyBits.readAll<uint32_t>(dev));
        for (uint32_t p = 0; p < numPasses; ++p) {
            uint64_t nzSum = 0, tokSum = 0;
            for (uint32_t i = 0; i < 3 * nblocks; ++i) { nzSum += passNzHost[p * 3 * nblocks + i]; tokSum += numTokensHost[p * 3 * nblocks + i]; }
            std::string bits;
            for (uint32_t g = 0; g < numGroups; ++g) bits += std::format("{} ", acOnlyHost[p * numGroups + g]);
            log::info("enc debug: pass {}: shift {} ncoeffs {} non-zeros {} tokens {} clusters {} AC bits per group {}", p, passes[p].shift, passes[p].ncoeffs,
                      nzSum, tokSum, clusterInfoHost[p], bits);
            SJXL_ASSIGN(const auto ctxHistHost, ctxHist.readAll<uint32_t>(dev));
            SJXL_ASSIGN(const auto histHost, hist.readAll<uint32_t>(dev));
            uint64_t zeroTok = 0, otherTok = 0;
            for (uint32_t ctx = 0; ctx < kNumAcContexts; ++ctx) {
                for (uint32_t b = 0; b < kHistBins; ++b) (b == 0 ? zeroTok : otherTok) += ctxHistHost[(uint64_t{p} * kNumAcContexts + ctx) * kHistBins + b];
            }
            std::string h;
            for (uint32_t b = 0; b < 8; ++b) h += std::format("{} ", histHost[(uint64_t{p} * kAcClusters) * kCodeStride + b]);
            log::info("enc debug: pass {} ctxHist token0 {} others {}; cluster 0 hist {}", p, zeroTok, otherTok, h);
            SJXL_ASSIGN(const auto ctxMapHost, ctxMap.readAll<uint32_t>(dev));
            SJXL_ASSIGN(const auto ansInfoHost, ansInfo.readAll<uint32_t>(dev));
            SJXL_ASSIGN(const auto ansReverseHost, ansReverse.readAll<uint32_t>(dev));
            std::map<uint32_t, uint32_t> mapCensus;
            for (uint32_t ctx = 0; ctx < kNumAcContexts; ++ctx) ++mapCensus[ctxMapHost[uint64_t{p} * kNumAcContexts + ctx]];
            std::string census, info, rev;
            for (const auto& [k, v] : mapCensus) census += std::format("{}x{} ", k, v);
            for (uint32_t b = 0; b < 4; ++b) info += std::format("{:x} ", ansInfoHost[(uint64_t{p} * kAcClusters) * 256 + b]);
            for (uint32_t b = 0; b < 6; ++b) rev += std::format("{} ", ansReverseHost[(uint64_t{p} * kAcClusters) * 4096 + b]);
            log::info("enc debug: pass {} ctxMap census {}; ansInfo {}; reverse {}", p, census, info, rev);
        }
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
    timer.mark("layout + size readback");
    blobEmit_.dispatch(batch, {numBlobRegions, 1, 1}, [&](rhi::ShaderCursor c) { bindLayout(c, body); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> bodyWords, body.readAll<uint32_t>(dev));
    timer.mark("emit + body readback");
    timer.report();

    // The frame header and TOC on the CPU, then the body.
    codestream::BitWriter head;
    codestream::writeFrameHeader(head, frame, hasAlpha ? 1u : 0u, info);
    if (singleSection) {
        sizes.assign(1, bodyBytes);
    }
    codestream::writeToc(head, sizes);
    std::vector<uint8_t> out = head.take();
    stats_.headBytes = static_cast<uint32_t>(out.size());
    stats_.bodyBytes = bodyBytes;
    stats_.sectionBytes = sizes;
    const auto* bodyBytesPtr = reinterpret_cast<const uint8_t*>(bodyWords.data());
    out.insert(out.end(), bodyBytesPtr, bodyBytesPtr + bodyBytes);
    if (!dcFrameBytes.empty()) out.insert(out.begin(), dcFrameBytes.begin(), dcFrameBytes.end());
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

Result<std::vector<uint8_t>> Encoder::encodeLossless(const image::Rgb8& image, const Options& options, const codestream::FrameInfo& info,
                                                      const gpu::Buffer* rgbDevice) {
    gpu::Device& dev = *device_;
    const uint32_t pixels = image.width * image.height;
    const uint32_t channels = image.channels;
    const uint32_t groupSizeShift = codestream::modularGroupSizeShift(image.width, image.height);
    const FrameDim dim = codestream::frameDim(image.width, image.height, groupSizeShift);
    const uint32_t numGroups = dim.numGroups();
    const bool grouped = !(image.width <= dim.groupDim && image.height <= dim.groupDim);
    gpu::Buffer rgbOwned;
    if (rgbDevice == nullptr) {
        std::vector<uint32_t> rgbWords(std::max<size_t>(1, (image.pixels.size() + 3) / 4), 0);
        std::memcpy(rgbWords.data(), image.pixels.data(), image.pixels.size());
        SJXL_ASSIGN(rgbOwned, gpu::Buffer::fromSpan<uint32_t>(dev, rgbWords, "enc.rgb8"));
    }
    const gpu::Buffer& rgb = rgbDevice != nullptr ? *rgbDevice : rgbOwned;
    // The raw channels as flat int planes, then (libjxl's hare tier, try_palettes) a
    // palette over every channel when the image has few enough colours, else a
    // compaction palette per channel that uses few of its values; the final planes
    // hold the meta channels first, then the image channels.
    SJXL_ASSIGN(auto rawPlanes, gpu::Buffer::ofType<int32_t>(dev, uint64_t{channels} * pixels, "enc.rawPlanes"));
    gpu::CommandBatch batch(dev);
    const auto bindBasic = [&](rhi::ShaderCursor c, const gpu::Buffer& planesBuf, const gpu::Buffer& chanTable) {
        for (const char* name : {"dcValue", "treeNodes", "wpScratch", "passTokens", "passHist", "passBits", "choice", "leafMap", "modTokens", "ctxHist",
                                 "ctxMap", "ansInfo", "ansReverse", "groupStart", "groupBits", "body", "thresholds", "samples", "sampleCount", "rctCosts",
                                 "groupRct"}) {
            c[name].setBinding(dummy_.rhi());
        }
        c["rgb8"].setBinding(rgb.rhi());
        c["planes"].setBinding(planesBuf.rhi());
        c["planesIn"].setBinding(planesBuf.rhi());
        c["chanTable"].setBinding(chanTable.rhi());
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["numChannels"].setData(channels);
        c["params"]["bitDepth"].setData(image.bitDepth);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["groupsX"].setData(dim.xsizeGroups);
        c["params"]["numGroups"].setData(numGroups);
    };
    unpackPlanes_.dispatch(batch, {image.width, image.height, 1}, [&](rhi::ShaderCursor c) { bindBasic(c, rawPlanes, dummy_); });

    ModularSpec spec;
    spec.width = image.width;
    spec.height = image.height;
    spec.bitDepth = image.bitDepth;
    spec.groupSizeShift = groupSizeShift;
    gpu::Buffer planes;
    constexpr uint32_t kTableSize = 8192, kMaxColors = 1024, kValueBins = 65536;
    // The colour limit: min(cost_before * 0.0005 + pixels / 128 + 128, 1024) with
    // cost_before the flat estimate 0.2 bits x depth x channels per pixel.
    const float costBefore = static_cast<float>(pixels) * 0.2f * static_cast<float>(image.bitDepth) * static_cast<float>(channels);
    const uint32_t maxColors = std::min<uint32_t>(static_cast<uint32_t>(costBefore * 0.0005f + static_cast<float>(pixels) / 128.0f + 128.0f), kMaxColors);
    gpu::Buffer table;
    SJXL_ASSIGN(auto palStats, gpu::Buffer::ofType<int32_t>(dev, 4, "enc.paletteStats"));
    const auto bindPalette = [&](rhi::ShaderCursor c, const gpu::Buffer& in, const gpu::Buffer& out, const gpu::Buffer& bins, uint32_t numC, uint32_t channel,
                                 uint32_t channelOffset, uint32_t paletteOffset, uint32_t indexOffset) {
        c["planesIn"].setBinding(in.rhi());
        c["planesOut"].setBinding(out.rhi());
        c["table"].setBinding(table.rhi());
        c["bins"].setBinding(bins.rhi());
        c["stats"].setBinding(palStats.rhi());
        c["params"]["width"].setData(image.width);
        c["params"]["height"].setData(image.height);
        c["params"]["numChannels"].setData(numC);
        c["params"]["maxColors"].setData(maxColors);
        c["params"]["channel"].setData(channel);
        c["params"]["channelOffset"].setData(channelOffset);
        c["params"]["paletteOffset"].setData(paletteOffset);
        c["params"]["indexOffset"].setData(indexOffset);
        c["params"]["bitDepth"].setData(image.bitDepth);
    };
    // The table's slots start free (kEmpty).
    {
        std::vector<uint32_t> empty(uint64_t{kTableSize} * 4 + 2 + kTableSize, 0);
        for (uint32_t i = 0; i < kTableSize; ++i) empty[uint64_t{i} * 4] = 0xFFFFFFFFu;
        SJXL_ASSIGN(table, gpu::Buffer::fromSpan<uint32_t>(dev, empty, "enc.paletteTable"));
    }
    paletteCount_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, dummyInt_, dummy_, channels, 0, 0, 0, 0); });
    paletteVerify_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, dummyInt_, dummy_, channels, 0, 0, 0, 0); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> tableHead, table.readRange<uint32_t>(dev, uint64_t{kTableSize} * 4, 2));
    const uint32_t distinct = tableHead[0];
    const bool paletteOk = tableHead[1] == 0 && distinct >= 2 && distinct <= maxColors;
    if (!platform::env("SJXL_DEBUG").empty()) log::info("enc debug: palette: {} colours (limit {}), overflow {}", distinct, maxColors, tableHead[1]);
    if (paletteOk) {
        // Channels: the palette (distinct x channels) then the index channel.
        SJXL_ASSIGN(planes, gpu::Buffer::ofType<int32_t>(dev, uint64_t{distinct} * channels + pixels, "enc.planes"));
        paletteBuild_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, planes, dummy_, channels, 0, 0, 0, distinct * channels); });
        paletteIndex_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, planes, dummy_, channels, 0, 0, 0, distinct * channels); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const std::vector<int32_t> statsHost, palStats.readAll<int32_t>(dev));
        const uint32_t nbColors = static_cast<uint32_t>(statsHost[3]);
        spec.channels.push_back(ModularChannel{nbColors, channels, 0, false});
        spec.channels.push_back(ModularChannel{image.width, image.height, nbColors * channels, grouped});
        spec.transforms.push_back(ModularTransform{1, 0, channels, nbColors});
        spec.nbMeta = 1;
        spec.rct = false;
        SJXL_ASSIGN(std::vector<uint8_t> out, encodeModularFrame(spec, planes, options, info, nullptr));
        stats_.palette = nbColors;   // encodeModularFrame resets the stats
        return out;
    }
    // Compaction per channel: its distinct values when fewer than 95 % of its range
    // (and under pixels / 16) become a one-row palette and the samples their ranks.
    SJXL_ASSIGN(auto bins, gpu::Buffer::ofType<uint32_t>(dev, kValueBins / 32 + kValueBins, "enc.compactBins"));
    std::vector<uint32_t> compactColors(channels, 0);
    std::vector<gpu::Buffer> palettes;   // per compacted channel: its values
    for (uint32_t ch = 0; ch < channels; ++ch) {
        const std::array<int32_t, 4> init{std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(), 0, 0};
        SJXL_ASSIGN(palStats, gpu::Buffer::fromSpan<int32_t>(dev, init, "enc.paletteStats"));
        batch.clear(bins);
        compactStats_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, dummyInt_, bins, 1, ch, ch * pixels, 0, 0); });
        SJXL_ASSIGN(auto values, gpu::Buffer::ofType<int32_t>(dev, kValueBins, "enc.compactValues"));
        compactBuild_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, values, bins, 1, ch, ch * pixels, 0, 0); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const std::vector<int32_t> st, palStats.readAll<int32_t>(dev));
        const int64_t range = int64_t{st[1]} - st[0] + 1;
        const uint32_t used = static_cast<uint32_t>(st[2]);
        const uint32_t limit = std::min<uint32_t>(pixels / 16, static_cast<uint32_t>(0.95 * static_cast<double>(range)));
        if (st[3] == 0 && used >= 1 && used <= limit) {
            compactColors[ch] = used;
            palettes.push_back(std::move(values));
        } else {
            palettes.emplace_back();
        }
    }
    // The final channel list: the compaction palettes (last applied first, as
    // the decoder inserts each at the front), then the image channels.
    uint32_t numCompacted = 0;
    for (uint32_t n : compactColors) numCompacted += n != 0 ? 1 : 0;
    uint32_t total = pixels * channels;
    for (uint32_t n : compactColors) total += n;
    SJXL_ASSIGN(planes, gpu::Buffer::ofType<int32_t>(dev, total, "enc.planes"));
    std::vector<ModularChannel> metas;   // in application order
    uint32_t offset = 0;
    for (uint32_t ch = 0; ch < channels; ++ch) {
        if (compactColors[ch] == 0) continue;
        metas.push_back(ModularChannel{compactColors[ch], 1, 0, false});
    }
    // Lay the metas out in the decoder's order (reverse application), then the image channels.
    std::vector<ModularChannel> ordered(metas.rbegin(), metas.rend());
    for (ModularChannel& m : ordered) { m.offset = offset; offset += m.width * m.height; }
    std::vector<uint32_t> imageOffsets(channels);
    for (uint32_t ch = 0; ch < channels; ++ch) { imageOffsets[ch] = offset; offset += pixels; }
    uint32_t metaSoFar = 0;
    for (uint32_t ch = 0; ch < channels; ++ch) {
        const uint32_t metaIndexFromEnd = numCompacted - 1 - metaSoFar;   // where this palette sits in `ordered`
        if (compactColors[ch] != 0) {
            // The palette values (compactBuild wrote them at offset 0 of `values`) into their meta channel, the ranks into the image channel.
            batch.copy(palettes[ch], 0, planes, uint64_t{ordered[metaIndexFromEnd].offset} * 4, uint64_t{compactColors[ch]} * 4);
            batch.clear(bins);
            compactStats_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, dummyInt_, bins, 1, ch, ch * pixels, 0, 0); });
            compactBuild_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, dummyInt_, bins, 1, ch, ch * pixels, 0, 0); });
            compactApply_.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor c) { bindPalette(c, rawPlanes, planes, bins, 1, ch, ch * pixels, 0, imageOffsets[ch]); });
            spec.transforms.push_back(ModularTransform{1, ch + metaSoFar, 1, compactColors[ch]});
            ++metaSoFar;
        } else {
            batch.copy(rawPlanes, uint64_t{ch} * pixels * 4, planes, uint64_t{imageOffsets[ch]} * 4, uint64_t{pixels} * 4);
        }
    }
    spec.channels = ordered;
    for (uint32_t ch = 0; ch < channels; ++ch) spec.channels.push_back(ModularChannel{image.width, image.height, imageOffsets[ch], grouped});
    spec.nbMeta = numCompacted;
    // The RCT per group over the three colour channels (after the meta channels):
    // libjxl's four hare-tier candidates by the entropy estimate of their gradient residuals.
    SJXL_ASSIGN(auto rctCosts, gpu::Buffer::ofType<float>(dev, uint64_t{numGroups} * 4, "enc.rctCosts"));
    SJXL_ASSIGN(auto groupRct, gpu::Buffer::ofType<uint32_t>(dev, numGroups, "enc.groupRct"));
    std::vector<Uint4> tableHost;
    for (const ModularChannel& c : spec.channels) tableHost.push_back(Uint4{c.width, c.height, c.offset, c.grouped ? 1u : 0u});
    SJXL_ASSIGN(auto chanTable, gpu::Buffer::fromSpan<Uint4>(dev, tableHost, "enc.chanTable"));
    const auto bindRct = [&](rhi::ShaderCursor c) {
        bindBasic(c, planes, chanTable);
        c["rctCosts"].setBinding(rctCosts.rhi());
        c["groupRct"].setBinding(groupRct.rhi());
        c["params"]["numChannels"].setData(static_cast<uint32_t>(spec.channels.size()));
        c["params"]["rctBeginC"].setData(spec.nbMeta);
    };
    rctCost_.dispatchGroups(batch, {numGroups * 4, 1, 1}, bindRct);
    rctApply_.dispatch(batch, {image.width, image.height, 1}, bindRct);
    SJXL_TRY(batch.submit(true));
    spec.rct = true;
    SJXL_ASSIGN(std::vector<uint8_t> out, encodeModularFrame(spec, planes, options, info, &groupRct));
    stats_.compacted = numCompacted;
    return out;
}

Result<std::vector<uint8_t>> Encoder::encodeModularFrame(const ModularSpec& spec, const gpu::Buffer& planes, const Options& options,
                                                         const codestream::FrameInfo& info, const gpu::Buffer* groupRct) {
    gpu::Device& dev = *device_;
    const FrameDim dim = codestream::frameDim(spec.width, spec.height, spec.groupSizeShift);
    const uint32_t numGroups = dim.numGroups();
    const uint32_t numDcGroups = dim.numDcGroups();
    const bool fitsGroup = spec.width <= dim.groupDim && spec.height <= dim.groupDim;
    const uint32_t channels = static_cast<uint32_t>(spec.channels.size());
    const uint64_t pixels = spec.totalSamples();   // samples over every channel
    uint32_t maxChannelWidth = dim.groupDim;
    for (const ModularChannel& c : spec.channels) maxChannelWidth = std::max(maxChannelWidth, c.width);
    const uint32_t wpWordsPerGroup = 5 * 2 * (maxChannelWidth + 2);
    SJXL_ASSIGN(auto wpScratch, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numGroups} * wpWordsPerGroup, "enc.wpScratchMod"));
    SJXL_ASSIGN(auto modTokens, gpu::Buffer::ofType<uint32_t>(dev, std::max<uint64_t>(pixels, 1), "enc.modTokens"));
    std::vector<Uint4> tableHost;
    for (const ModularChannel& c : spec.channels) tableHost.push_back(Uint4{c.width, c.height, c.offset, c.grouped ? 1u : 0u});
    SJXL_ASSIGN(auto chanTable, gpu::Buffer::fromSpan<Uint4>(dev, tableHost, "enc.chanTable"));
    std::vector<Uint4> transformsHost;
    for (const ModularTransform& t : spec.transforms) transformsHost.push_back(Uint4{t.id, t.beginC, t.numC, t.nbColors});
    if (transformsHost.empty()) transformsHost.push_back(Uint4{0, 0, 0, 0});
    SJXL_ASSIGN(auto transforms, gpu::Buffer::fromSpan<Uint4>(dev, transformsHost, "enc.transforms"));
    // The tree: learned (tree.slang) or libjxl's fixed WP / gradient trees per channel.
    gpu::Buffer treeNodes, ctxHist, choice, leafMapBuf, passTokens, passHist, passBits, thresholds, samples, sampleCount;
    uint32_t numLeaves = 0, numTreeNodes = 0;
    std::string treeDesc;
    const auto orDummy = [&](const gpu::Buffer& b) -> const gpu::Buffer& { return b.valid() ? b : dummy_; };
    // libjxl's hare-tier tree learning: Predictor::Best (Weighted, Gradient); the
    // first four of its property order {0, 1, 15, 9, 10, ...}, the stream (1)
    // dropped when the frame has fewer than 30 streams; 48 property values; a
    // quarter of the pixels sampled; node threshold 75 + 14 * 5.
    const uint32_t numStreams = 18 + 3 * numDcGroups + numGroups;
    const uint32_t streamBase = fitsGroup ? 0 : 18 + 3 * numDcGroups;
    const std::array<uint32_t, 4> propIds = numStreams < 30 ? std::array<uint32_t, 4>{0, 15, 9, 10} : std::array<uint32_t, 4>{0, 1, 15, 9};
    const float sampleFraction = 0.25f;
    float nodeThreshold = 75.0f + 14.0f * 5.0f;
    if (const std::string t = platform::env("SJXL_TREE_THRESHOLD"); !t.empty()) nodeThreshold = std::strtof(t.c_str(), nullptr);   // experiments

    gpu::CommandBatch batch(dev);
    const auto bindMod = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& groupStart, const gpu::Buffer& body, const gpu::Buffer& tree,
                             uint32_t pass) {
        c["rgb8"].setBinding(dummy_.rhi());
        c["dcValue"].setBinding(dummy_.rhi());
        c["thresholds"].setBinding(orDummy(thresholds).rhi());
        c["samples"].setBinding(orDummy(samples).rhi());
        c["sampleCount"].setBinding(orDummy(sampleCount).rhi());
        c["rctCosts"].setBinding(dummy_.rhi());
        c["groupRct"].setBinding(groupRct != nullptr ? groupRct->rhi() : dummy_.rhi());
        c["chanTable"].setBinding(chanTable.rhi());
        c["planes"].setBinding(planes.rhi());
        c["planesIn"].setBinding(planes.rhi());
        c["treeNodes"].setBinding(orDummy(tree).rhi());
        c["wpScratch"].setBinding(wpScratch.rhi());
        c["passTokens"].setBinding(orDummy(passTokens).rhi());
        c["passHist"].setBinding(orDummy(passHist).rhi());
        c["passBits"].setBinding(orDummy(passBits).rhi());
        c["choice"].setBinding(orDummy(choice).rhi());
        c["leafMap"].setBinding(orDummy(leafMapBuf).rhi());
        c["modTokens"].setBinding(modTokens.rhi());
        c["ctxHist"].setBinding(orDummy(ctxHist).rhi());
        c["ctxMap"].setBinding(dummy_.rhi());
        c["ansInfo"].setBinding(dummy_.rhi());
        c["ansReverse"].setBinding(dummy_.rhi());
        c["groupStart"].setBinding(groupStart.rhi());
        c["groupBits"].setBinding(dummy_.rhi());
        c["body"].setBinding(body.rhi());
        for (const char* name : {"streamTokens", "streamStart", "streamLen", "clusterCfg"}) c[name].setBinding(dummy_.rhi());
        c["params"]["lz77"].setData(0u);
        c["params"]["width"].setData(spec.width);
        c["params"]["height"].setData(spec.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["groupsX"].setData(dim.xsizeGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["numLeaves"].setData(numLeaves);
        c["params"]["mode"].setData(mode);
        c["params"]["numChannels"].setData(channels);
        c["params"]["bitDepth"].setData(spec.bitDepth);
        c["params"]["wpWordsPerGroup"].setData(wpWordsPerGroup);
        c["params"]["pass"].setData(pass);
        c["params"]["ctxBase"].setData(0u);
        c["params"]["dcQuantX"].setData(0.0f);
        c["params"]["dcQuantY"].setData(0.0f);
        c["params"]["dcQuantB"].setData(0.0f);
        c["params"]["sampleFraction"].setData(sampleFraction);
        c["params"]["sampleCapacity"].setData(uint32_t{samples.valid() ? static_cast<uint32_t>(samples.bytes() / 8) : 0u});
        c["params"]["streamBase"].setData(streamBase);
        c["params"]["rct"].setData(uint32_t{spec.rct && groupRct != nullptr ? 1u : 0u});
        c["params"]["rctBeginC"].setData(spec.nbMeta);
        c["params"]["nbMeta"].setData(spec.nbMeta);
        c["params"]["totalSamples"].setData(static_cast<uint32_t>(pixels));
        c["params"]["propIds"].setData(propIds);
    };

    if (options.learnedTree) {
        constexpr uint32_t kStatBins = 2 * 512 + 1, kThreshStride = 64, kNumProps = 16;
        constexpr uint32_t kMaxNodes = 4096, kMaxSlots = 384, kLearnNodeWords = 8;
        constexpr uint32_t kSlotCounts = 4 * 48 * 2 * 64, kSlotExtra = 4 * 48 * 2;
        SJXL_ASSIGN(auto treeStats, gpu::Buffer::ofType<uint32_t>(dev, 3 * kStatBins + 16, "enc.treeStats"));
        SJXL_ASSIGN(thresholds, gpu::Buffer::ofType<int32_t>(dev, (kNumProps + 1) * kThreshStride, "enc.treeThresholds"));
        const uint64_t capacity = std::min<uint64_t>(pixels, static_cast<uint64_t>(static_cast<double>(pixels) * sampleFraction * 1.25) + 65536);
        SJXL_ASSIGN(samples, gpu::Buffer::ofType<uint64_t>(dev, capacity, "enc.treeSamples"));
        SJXL_ASSIGN(sampleCount, gpu::Buffer::ofType<uint32_t>(dev, 1, "enc.treeSampleCount"));
        SJXL_ASSIGN(auto counts, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kMaxSlots} * kSlotCounts, "enc.treeCounts"));
        SJXL_ASSIGN(auto extra, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kMaxSlots} * kSlotExtra, "enc.treeExtra"));
        std::vector<uint32_t> nodesHost(uint64_t{kMaxNodes} * kLearnNodeWords, 0);
        nodesHost[0] = 0xFFFFFFFFu;   // the root: a leaf predicting Weighted (predictor index 0)
        SJXL_ASSIGN(auto nodes, gpu::Buffer::fromSpan<uint32_t>(dev, nodesHost, "enc.treeNodesLearn"));
        gpu::Buffer sampleNode, counters;
        uint32_t numSamples = 0;
        const auto bindTree = [&](rhi::ShaderCursor c, uint32_t levelStart) {
            c["planes"].setBinding(planes.rhi());
            c["chanTable"].setBinding(chanTable.rhi());
            c["stats"].setBinding(treeStats.rhi());
            c["thresholds"].setBinding(thresholds.rhi());
            c["samples"].setBinding(samples.rhi());
            c["sampleNode"].setBinding(orDummy(sampleNode).rhi());
            c["counts"].setBinding(counts.rhi());
            c["extra"].setBinding(extra.rhi());
            c["nodes"].setBinding(nodes.rhi());
            c["counters"].setBinding(orDummy(counters).rhi());
            c["params"]["numSamples"].setData(numSamples);
            c["params"]["levelStart"].setData(levelStart);
            c["params"]["maxNodes"].setData(kMaxNodes);
            c["params"]["maxSlots"].setData(kMaxSlots);
            c["params"]["threshold"].setData(nodeThreshold);
            c["params"]["fastDecodeMul"].setData(1.001f);
            c["params"]["width"].setData(spec.width);
            c["params"]["height"].setData(spec.height);
            c["params"]["numChannels"].setData(channels);
            c["params"]["totalSamples"].setData(static_cast<uint32_t>(pixels));
            c["params"]["numGroups"].setData(numGroups);
            c["params"]["groupDim"].setData(dim.groupDim);
            c["params"]["groupsX"].setData(dim.xsizeGroups);
            c["params"]["streamBase"].setData(streamBase);
            c["params"]["sampleFraction"].setData(std::min(0.99f, sampleFraction * 0.1f));   // CollectPixelSamples
            c["params"]["propIds"].setData(propIds);
        };
        // The properties' quantisation thresholds, then the samples.
        propStats_.dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindTree(c, 0); });
        propThresholds_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) { bindTree(c, 0); });
        modSamples_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, dummy_, 0); });
        SJXL_TRY(batch.submit(true));
        SJXL_ASSIGN(const std::vector<uint32_t> sampleCountHost, sampleCount.readAll<uint32_t>(dev));
        numSamples = static_cast<uint32_t>(std::min<uint64_t>(sampleCountHost[0], capacity));
        // LearnTree: the node threshold scales with the fraction of pixels sampled (0.9 f + 0.1).
        const float pixelFraction = static_cast<float>(numSamples) / static_cast<float>(std::max<uint64_t>(pixels, 1));
        nodeThreshold *= pixelFraction * 0.9f + 0.1f;
        SJXL_ASSIGN(sampleNode, gpu::Buffer::ofType<uint32_t>(dev, std::max<uint32_t>(numSamples, 1), "enc.treeSampleNode"));
        // The tree, a level at a time: histograms per open node, the split decision per node, the samples into the children.
        uint32_t levelStart = 0, nodeCount = 1, depth = 0;
        for (; depth < 40 && numSamples > 0; ++depth) {
            const uint32_t inLevel = nodeCount - levelStart;
            if (inLevel == 0) break;
            const std::array<uint32_t, 2> countersHost{nodeCount, 0};
            SJXL_ASSIGN(counters, gpu::Buffer::fromSpan<uint32_t>(dev, countersHost, "enc.treeCounters"));
            batch.clear(counts);
            batch.clear(extra);
            treeHistograms_.dispatch(batch, {numSamples, 1, 1}, [&](rhi::ShaderCursor c) { bindTree(c, levelStart); });
            treeSplit_.dispatchGroups(batch, {std::min(inLevel, kMaxSlots), 1, 1}, [&](rhi::ShaderCursor c) { bindTree(c, levelStart); });
            treeAssign_.dispatch(batch, {numSamples, 1, 1}, [&](rhi::ShaderCursor c) { bindTree(c, levelStart); });
            SJXL_TRY(batch.submit(true));
            SJXL_ASSIGN(const std::vector<uint32_t> after, counters.readAll<uint32_t>(dev));
            levelStart = nodeCount;
            nodeCount = after[0];
            if (after[1] == 0 || nodeCount + 2 > kMaxNodes) break;
        }
        // The decoder's layout: breadth first, a node's children consecutive, leaves numbered in that order.
        SJXL_ASSIGN(const std::vector<uint32_t> learned, nodes.readRange<uint32_t>(dev, 0, uint64_t{nodeCount} * kLearnNodeWords));
        std::vector<uint32_t> treeWords;
        std::vector<uint32_t> queue{0};
        for (size_t head = 0; head < queue.size(); ++head) {
            const uint32_t n = queue[head];
            const uint32_t* w = &learned[uint64_t{n} * kLearnNodeWords];
            const size_t at = treeWords.size();
            treeWords.resize(at + kModTreeWords, 0);
            if (w[0] == 0xFFFFFFFFu) {
                treeWords[at + 0] = 0xFFFFFFFFu;
                treeWords[at + 4] = numLeaves++;
                treeWords[at + 5] = w[4] == 0 ? kPredWeighted : kPredGradient;
            } else {
                const uint32_t pending = static_cast<uint32_t>(queue.size() - head - 1);
                const uint32_t self = static_cast<uint32_t>(at / kModTreeWords);
                treeWords[at + 0] = propIds[std::min<uint32_t>(w[0], 3)];
                treeWords[at + 1] = w[5];
                treeWords[at + 2] = self + pending + 1;
                treeWords[at + 3] = self + pending + 2;
                queue.push_back(w[2]);
                queue.push_back(w[3]);
            }
        }
        numTreeNodes = static_cast<uint32_t>(treeWords.size() / kModTreeWords);
        SJXL_ASSIGN(treeNodes, gpu::Buffer::fromSpan<uint32_t>(dev, treeWords, "enc.treeNodes"));
        treeDesc = std::format("learned tree over properties {} {} {} {}: {} samples, {} nodes, depth {}", propIds[0], propIds[1], propIds[2], propIds[3],
                               numSamples, numTreeNodes, depth);
        SJXL_ASSIGN(ctxHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{numLeaves + 1} * kHistBins, "enc.ctxHist"));   // + the LZ77 distance context
        modTokenizeTree_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, treeNodes, 0); });
    } else {
        // libjxl's Predictor::Best per channel: the weighted predictor with the fixed
        // WP-property tree, or Gradient with the gradient tree, whichever the
        // channel's tokens estimate cheaper; the frame's tree joins the choices.
        uint32_t wpLeaves = 0, gradLeaves = 0;
        const std::vector<uint32_t> wpTree = fixedTree(kModPropWp, kPredWeighted, pixels, wpLeaves);
        const std::vector<uint32_t> gradTree = fixedTree(kModPropGradient, kPredGradient, pixels, gradLeaves);
        SJXL_ASSIGN(auto wpTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, wpTree, "enc.wpTree"));
        SJXL_ASSIGN(auto gradTreeBuf, gpu::Buffer::fromSpan<uint32_t>(dev, gradTree, "enc.gradTree"));
        SJXL_ASSIGN(passTokens, gpu::Buffer::ofType<uint32_t>(dev, 2 * pixels, "enc.passTokens"));
        SJXL_ASSIGN(passHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{2} * channels * kSetContexts * kHistBins, "enc.passHist"));
        SJXL_ASSIGN(passBits, gpu::Buffer::ofType<float>(dev, 2 * channels, "enc.passBits"));
        const uint32_t maxLeaves = channels * kSetContexts;   // the combined tree's leaves at most
        SJXL_ASSIGN(ctxHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{maxLeaves + 1} * kHistBins, "enc.ctxHist"));
        SJXL_ASSIGN(choice, gpu::Buffer::ofType<uint32_t>(dev, channels, "enc.choice"));
        SJXL_ASSIGN(leafMapBuf, gpu::Buffer::ofType<uint32_t>(dev, channels * kSetContexts, "enc.leafMap"));
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
            treeDesc += std::format("{}:{} ", c, choiceHost[c] == 0 ? "WP" : "grad");
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
        remapTokens_.dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindMod(c, 0, dummy_, dummy_, treeNodes, 0); });
    }

    // The streams (the groups', then the global one) and their compact token lists
    // (lz77.slang): run-length coded when libjxl's rule accepts it, plain otherwise.
    const uint32_t numTokenStreams = numGroups + 1;
    std::vector<uint32_t> streamStartHost(numTokenStreams, 0);
    {
        uint32_t pos = 0;
        for (uint32_t st = 0; st < numTokenStreams; ++st) {
            streamStartHost[st] = pos;
            for (const ModularChannel& c : spec.channels) {
                if (st == numGroups) {
                    if (!c.grouped) pos += c.width * c.height;
                } else if (c.grouped) {
                    const uint32_t x0 = (st % dim.xsizeGroups) * dim.groupDim, y0 = (st / dim.xsizeGroups) * dim.groupDim;
                    if (x0 < c.width && y0 < c.height) pos += std::min(dim.groupDim, c.width - x0) * std::min(dim.groupDim, c.height - y0);
                }
            }
        }
    }
    SJXL_ASSIGN(auto streamStart, gpu::Buffer::fromSpan<uint32_t>(dev, streamStartHost, "enc.streamStart"));
    SJXL_ASSIGN(auto streamLen, gpu::Buffer::ofType<uint32_t>(dev, numTokenStreams, "enc.streamLen"));
    SJXL_ASSIGN(auto streamTokens, gpu::Buffer::ofType<uint32_t>(dev, std::max<uint64_t>(pixels, 1), "enc.streamTokens"));
    SJXL_ASSIGN(auto rleStats, gpu::Buffer::ofType<uint32_t>(dev, 2, "enc.rleStats"));
    SJXL_ASSIGN(auto ctxMap, gpu::Buffer::ofType<uint32_t>(dev, numLeaves + 1, "enc.ctxMap"));
    SJXL_ASSIGN(auto cfgHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * 4 * 256, "enc.cfgHist"));
    SJXL_ASSIGN(auto cfgExtra, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * 4 * 2, "enc.cfgExtra"));
    SJXL_ASSIGN(auto lengthHist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * 256, "enc.lengthHist"));
    SJXL_ASSIGN(auto maxToken, gpu::Buffer::ofType<uint32_t>(dev, 1, "enc.maxToken"));
    SJXL_ASSIGN(auto hist, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * kCodeStride, "enc.hist"));
    // Regions: one set of clusters (the map's own code lands at (1 + 2 numDcGroups) * 32).
    const uint32_t regions = (1 + 2 * numDcGroups) * kAcClusters + 1;
    const std::vector<uint32_t> cfgDefault(regions, kDefaultUintCfg);
    SJXL_ASSIGN(auto clusterCfg, gpu::Buffer::fromSpan<uint32_t>(dev, cfgDefault, "enc.clusterCfg"));
    const auto bindLz77 = [&](rhi::ShaderCursor c, uint32_t mode, uint32_t numClusters, bool lz77) {
        c["modTokens"].setBinding(modTokens.rhi());
        c["chanTable"].setBinding(chanTable.rhi());
        c["streamStart"].setBinding(streamStart.rhi());
        c["streamLen"].setBinding(streamLen.rhi());
        c["streamTokens"].setBinding(streamTokens.rhi());
        c["ctxHist"].setBinding(ctxHist.rhi());
        c["rleStats"].setBinding(rleStats.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["cfgHist"].setBinding(cfgHist.rhi());
        c["cfgExtra"].setBinding(cfgExtra.rhi());
        c["lengthHist"].setBinding(lengthHist.rhi());
        c["hist"].setBinding(hist.rhi());
        c["clusterCfg"].setBinding(clusterCfg.rhi());
        c["maxToken"].setBinding(maxToken.rhi());
        c["params"]["numStreams"].setData(numTokenStreams);
        c["params"]["numContexts"].setData(numLeaves);
        c["params"]["mode"].setData(mode);
        c["params"]["numClusters"].setData(numClusters);
        c["params"]["totalTokens"].setData(static_cast<uint32_t>(pixels));
        c["params"]["lz77"].setData(uint32_t{lz77 ? 1u : 0u});
        c["params"]["regionBase"].setData(0u);
        c["params"]["ctxBase"].setData(0u);
        c["params"]["numChannels"].setData(channels);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["groupsX"].setData(dim.xsizeGroups);
        c["params"]["numGroups"].setData(numGroups);
        c["params"]["width"].setData(spec.width);
        c["params"]["height"].setData(spec.height);
        c["params"]["histogramTries"].setData(7u);
        c["params"]["numCandidatesRt"].setData(4u);
    };
    rleScan_.dispatch(batch, {numTokenStreams, 1, 1}, [&](rhi::ShaderCursor c) { bindLz77(c, 1, 0, false); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> rleHost, rleStats.readAll<uint32_t>(dev));
    const bool lz77 = static_cast<double>(rleHost[0]) / 16.0 > static_cast<double>(rleHost[1]) * 0.2 + 16.0;
    if (lz77) {
        batch.clear(ctxHist);
        rleHistograms_.dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLz77(c, 1, 0, true); });
    } else {
        rleScan_.dispatch(batch, {numTokenStreams, 1, 1}, [&](rhi::ShaderCursor c) { bindLz77(c, 0, 0, false); });
    }
    const uint32_t numContexts = numLeaves + (lz77 ? 1 : 0);

    // The contexts' histograms clustered, the hybrid-uint configuration per cluster,
    // the codes and the global blob, the group sections' sizes.
    SJXL_ASSIGN(auto ctxTotal, gpu::Buffer::ofType<float>(dev, numContexts, "enc.ctxTotal"));
    SJXL_ASSIGN(auto ctxEntropy, gpu::Buffer::ofType<float>(dev, numContexts, "enc.ctxEntropy"));
    SJXL_ASSIGN(auto ctxDist, gpu::Buffer::ofType<float>(dev, numContexts, "enc.ctxDist"));
    SJXL_ASSIGN(auto ctxSingle, gpu::Buffer::ofType<uint32_t>(dev, numContexts, "enc.ctxSingle"));
    SJXL_ASSIGN(auto clusterInfo, gpu::Buffer::ofType<uint32_t>(dev, 4, "enc.clusterInfo"));
    SJXL_ASSIGN(auto ctxMapMtf, gpu::Buffer::ofType<uint32_t>(dev, numContexts, "enc.ctxMapMtf"));
    SJXL_ASSIGN(auto codes, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{kAcClusters} * kCodeStride, "enc.codes"));
    SJXL_ASSIGN(auto ansInfo, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regions} * 256, "enc.ansInfo"));
    SJXL_ASSIGN(auto ansReverse, gpu::Buffer::ofType<uint32_t>(dev, uint64_t{regions} * 4096, "enc.ansReverse"));
    // The global blob holds the whole stream when the image fits a group.
    const uint64_t blobWords = kBlobStrideWords + pixels * (spec.bitDepth / 8) + 1024;
    SJXL_ASSIGN(auto blobs, gpu::Buffer::ofType<uint32_t>(dev, blobWords, "enc.blobs"));
    SJXL_ASSIGN(auto blobBits, gpu::Buffer::ofType<uint32_t>(dev, 2, "enc.blobBits"));
    SJXL_ASSIGN(auto groupBits, gpu::Buffer::ofType<uint32_t>(dev, numGroups, "enc.groupBits"));
    const auto bindSections = [&](rhi::ShaderCursor c, uint32_t mode, const gpu::Buffer& groupStart, const gpu::Buffer& body) {
        bindMod(c, mode, groupStart, body, treeNodes, 0);
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["ansInfo"].setBinding(ansInfo.rhi());
        c["ansReverse"].setBinding(ansReverse.rhi());
        c["groupBits"].setBinding(groupBits.rhi());
        c["streamTokens"].setBinding(streamTokens.rhi());
        c["streamStart"].setBinding(streamStart.rhi());
        c["streamLen"].setBinding(streamLen.rhi());
        c["clusterCfg"].setBinding(clusterCfg.rhi());
        c["params"]["lz77"].setData(uint32_t{lz77 ? 1u : 0u});
    };
    const auto bindCluster = [&](rhi::ShaderCursor c) {
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
        c["params"]["ctxBase"].setData(0u);
        c["params"]["infoIndex"].setData(0u);
        c["params"]["regionBase"].setData(0u);
    };
    clusterContexts_.dispatchGroups(batch, {1, 1, 1}, bindCluster);
    accumulateClusters_.dispatch(batch, {numContexts, 1, 1}, bindCluster);
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> clusterInfoEarly, clusterInfo.readAll<uint32_t>(dev));
    const uint32_t numClusters = clusterInfoEarly[0];
    // ChooseUintConfigs (the fast candidates) per cluster: the histograms of every
    // candidate from the final tokens, the cheapest kept as the cluster's code region.
    uintConfigHist_.dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) { bindLz77(c, 0, numClusters, lz77); });
    uintConfigChoose_.dispatchGroups(batch, {numClusters, 1, 1}, [&](rhi::ShaderCursor c) { bindLz77(c, 0, numClusters, lz77); });
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> maxTokenHost, maxToken.readAll<uint32_t>(dev));
    uint32_t logAlpha = 5;
    while (maxTokenHost[0] >= (1u << logAlpha) && logAlpha < 8) ++logAlpha;
    if (lz77) logAlpha = 8;
    const auto bindCodes = [&](rhi::ShaderCursor c) {
        c["dcGroups"].setBinding(dummy_.rhi());
        c["acs"].setBinding(dummy_.rhi());
        c["hist"].setBinding(hist.rhi());
        c["quantizer"].setBinding(dummy_.rhi());
        c["ctxMap"].setBinding(ctxMap.rhi());
        c["groupRct"].setBinding(groupRct != nullptr ? groupRct->rhi() : dummy_.rhi());
        c["chanTable"].setBinding(chanTable.rhi());
        c["transforms"].setBinding(transforms.rhi());
        c["streamTokens"].setBinding(streamTokens.rhi());
        c["streamStart"].setBinding(streamStart.rhi());
        c["streamLen"].setBinding(streamLen.rhi());
        c["clusterCfg"].setBinding(clusterCfg.rhi());
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
        c["params"]["width"].setData(spec.width);
        c["params"]["height"].setData(spec.height);
        c["params"]["groupDim"].setData(dim.groupDim);
        c["params"]["numTreeNodes"].setData(numTreeNodes);
        c["params"]["numLeaves"].setData(numLeaves);
        c["params"]["numChannels"].setData(channels);
        c["params"]["bitDepth"].setData(spec.bitDepth);
        c["params"]["rctBeginC"].setData(spec.nbMeta);
        c["params"]["numTransforms"].setData(static_cast<uint32_t>(spec.transforms.size()));
        c["params"]["totalSamples"].setData(static_cast<uint32_t>(pixels));
        c["params"]["lz77Modular"].setData(uint32_t{lz77 ? 1u : 0u});
        c["params"]["modLogAlpha"].setData(logAlpha);
        c["params"]["histogramTries"].setData(7u);
        c["rawQuant"].setBinding(dummyInt_.rhi());
        c["qtTree"].setBinding(dummy_.rhi());
        c["params"]["jpeg"].setData(0u);
        c["params"]["qtRegion"].setData(0u);
        c["params"]["qtCount"].setData(0u);
        c["params"]["numQuantKinds"].setData(17u);
        c["params"]["qtBlobRegion"].setData(0u);
        c["params"]["acPostRegion"].setData(0u);
        c["params"]["dcQuantJpegX"].setData(0.0f);
        c["params"]["dcQuantJpegY"].setData(0.0f);
        c["params"]["dcQuantJpegB"].setData(0.0f);
        c["params"]["dcTreeNodes"].setData(0u);
        c["params"]["dcLeaves"].setData(0u);
        c["params"]["metaTreeNodes"].setData(0u);
        c["params"]["metaLeaves"].setData(0u);
        c["params"]["numAcContexts"].setData(numContexts);
        c["params"]["numEcSets"].setData(0u);
        c["params"]["numAcSets"].setData(1u);
        c["params"]["ecInGlobal"].setData(0u);
        c["params"]["ecLogAlpha"].setData(6u);
        c["params"]["useDcFrame"].setData(0u);
        c["params"]["rct"].setData(uint32_t{spec.rct && groupRct != nullptr ? 1u : 0u});
        c["params"]["dcQuantCustom"].setData(uint32_t{spec.dcFrame ? 1u : 0u});
        c["params"]["dcQuantX"].setData(spec.dcQuant[0]);
        c["params"]["dcQuantY"].setData(spec.dcQuant[1]);
        c["params"]["dcQuantB"].setData(spec.dcQuant[2]);
    };
    modularGlobalBlob_.dispatchGroups(batch, {1, 1, 1}, bindCodes);
    if (!fitsGroup) {
        modGroupSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindSections(c, 1, dummy_, dummy_); });
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
        c["params"]["qtBlobRegion"].setData(0xFFFFFFFFu);
        c["params"]["acPostRegion"].setData(0xFFFFFFFFu);
        c["params"]["numBlobRegions"].setData(1u);
    });
    if (!fitsGroup) {
        modGroupSections_.dispatch(batch, {numGroups, 1, 1}, [&](rhi::ShaderCursor c) { bindSections(c, 2, groupStart, body); });
    }
    SJXL_TRY(batch.submit(true));
    SJXL_ASSIGN(const std::vector<uint32_t> bodyWords, body.readAll<uint32_t>(dev));
    SJXL_ASSIGN(const std::vector<uint32_t> clusterInfoHost, clusterInfo.readAll<uint32_t>(dev));

    codestream::BitWriter head;
    const uint32_t imageChannels = channels - spec.nbMeta;
    const bool hasAlphaChannel = !spec.dcFrame && (imageChannels == 4 || (spec.nbMeta == 1 && spec.transforms.size() == 1 && spec.transforms[0].numC == 4));
    if (spec.dcFrame) codestream::writeDcFrameHeader(head, spec.groupSizeShift);
    else codestream::writeModularFrameHeader(head, hasAlphaChannel ? 1 : 0, info, spec.groupSizeShift);
    codestream::writeToc(head, sizes);
    std::vector<uint8_t> out = head.take();
    stats_ = Stats{};
    stats_.headBytes = static_cast<uint32_t>(out.size());
    stats_.bodyBytes = bodyBytes;
    stats_.sectionBytes = sizes;
    stats_.numAcClusters = clusterInfoHost[0];
    if (!platform::env("SJXL_DEBUG").empty()) {
        log::info("enc debug: {} {}x{} x{} {}-bit, {}, {} tree nodes, {} leaves, {} clusters, lz77 {} (saving {} of {} symbols), log alpha {}, DcGlobal {} bits, {} groups",
                  spec.dcFrame ? "DC frame" : "lossless", spec.width, spec.height, channels, spec.bitDepth, treeDesc, numTreeNodes, numLeaves, clusterInfoHost[0],
                  lz77, rleHost[0] / 16, rleHost[1], logAlpha, blobBitsHost[0], numGroups);
    }
    const auto* bodyBytesPtr = reinterpret_cast<const uint8_t*>(bodyWords.data());
    out.insert(out.end(), bodyBytesPtr, bodyBytesPtr + bodyBytes);
    return out;
}

}   // namespace sjxl::enc
