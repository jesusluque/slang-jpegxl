// Copyright (c) 2026 jesus luque.
//
// sjxl: the command line.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <string_view>

#include "sjxl/core/Log.h"
#include "sjxl/decoder/Decoder.h"
#include "sjxl/encoder/Encoder.h"
#include "sjxl/gpu/Device.h"
#include "sjxl/image/Image.h"
#include "sjxl/metrics/Compare.h"

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  sjxl info [in.jxl]                          describe the GPU device, or a file's headers and frames\n"
                 "  sjxl encode in.png out.jxl [--distance d] [--lossless] [--no-aq] [--no-cfl] [--no-filters] [--no-acs] [--no-specials] [--lambda x] [--clusters n]\n"
                 "                                              lossy VarDCT (default distance 1.0)\n"
                 "  sjxl decode in.jxl out.png                  any frame type; animations write out_000.png, out_001.png, ...\n"
                 "  sjxl compare a.png b.png                    PSNR and largest difference, measured on the GPU\n"
                 "  sjxl pixel a.png x y                        one pixel's RGB\n"
                 "  sjxl convert in.png out.png                 re-save as 8-bit RGB (drops alpha)\n");
    return 2;
}

int fail(const sjxl::Error& error) {
    std::fprintf(stderr, "sjxl: %s\n", error.toString().c_str());
    return 1;
}

int info(int argc, char** argv) {
    if (argc >= 3) {
        auto bytes = sjxl::image::readFile(argv[2]);
        if (!bytes) return fail(bytes.error());
        auto parsed = sjxl::dec::Decoder::info(*bytes);
        if (!parsed) return fail(parsed.error());
        if (parsed->incomplete) {   // a permuted TOC: the device decodes the permutation
            auto device = sjxl::gpu::Device::create();
            if (!device) return fail(device.error());
            auto library = std::make_shared<sjxl::gpu::ShaderLibrary>(*device);
            auto decoder = sjxl::dec::Decoder::create(library);
            if (!decoder) return fail(decoder.error());
            parsed = (*decoder)->parseAll(*bytes);
            if (!parsed) return fail(parsed.error());
        }
        const auto& m = parsed->metadata;
        std::printf("image     %ux%u, %u-bit%s, %s, %s%s\n", m.xsize, m.ysize, m.bitsPerSample, m.floatingPointSample ? " float" : "",
                    m.colorSpace == 0 ? "RGB" : m.colorSpace == 1 ? "grey" : "other colour space", m.xybEncoded ? "XYB" : "not XYB",
                    m.numExtraChannels ? ", extra channels" : "");
        std::printf("frames    %zu\n", parsed->frames.size());
        for (size_t i = 0; i < parsed->frames.size(); ++i) {
            const auto& f = parsed->frames[i];
            const auto& h = f.header;
            const char* type = h.frameType == 0 ? "regular" : h.frameType == 1 ? "DC" : h.frameType == 2 ? "reference-only" : "skip-progressive";
            std::printf("  frame %zu %s %s %ux%u, groups %u px (%u AC, %u DC), %zu sections, %u pass%s%s%s%s%s, gab %d, epf %u%s\n", i, type,
                        h.isModular ? "Modular" : h.colorTransform == 2 ? "VarDCT YCbCr" : "VarDCT", f.dim.xsize, f.dim.ysize, f.dim.groupDim, f.dim.numGroups(), f.dim.numDcGroups(),
                        f.sections.size(), h.numPasses, h.numPasses == 1 ? "" : "es", (h.flags & sjxl::codestream::FrameHeader::kPatches) ? ", patches" : "",
                        (h.flags & sjxl::codestream::FrameHeader::kSplines) ? ", splines" : "", (h.flags & sjxl::codestream::FrameHeader::kNoise) ? ", noise" : "",
                        h.frameType == 2 ? (h.saveBeforeColorTransform ? ", saved before the colour transform" : ", saved after the colour transform")
                                         : h.frameType == 1 ? ", DC frame" : (h.flags & sjxl::codestream::FrameHeader::kUseDcFrame) ? ", DC from a DC frame" : "",
                        h.loopFilter.gab ? 1 : 0, h.loopFilter.epfIters, h.isLast ? ", last" : "");
            if (f.sections.size() > 1 && f.sections.size() <= 64) {
                std::printf("    section bytes:");
                for (const auto& sct : f.sections) std::printf(" %u", sct.byteSize);
                std::printf("\n");
            }
            if (h.duration != 0 || h.customSizeOrOrigin || h.blending.mode != 0) {
                const char* modes[5] = {"replace", "add", "blend", "alpha-weighted add", "multiply"};
                std::printf("          duration %u ticks, origin (%d,%d), blend %s from slot %u%s\n", h.duration, h.x0, h.y0,
                            modes[h.blending.mode < 5 ? h.blending.mode : 0], h.blending.source,
                            (!h.isLast && (h.duration == 0 || h.saveAsReference != 0)) ? std::format(", saved as reference {}", h.saveAsReference).c_str() : "");
            }
        }
        return 0;
    }
    auto device = sjxl::gpu::Device::create();
    if (!device) return fail(device.error());
    const auto& caps = (*device)->caps();
    std::printf("backend   %s\napi       %s\nadapter   %s\nhalf      %s\nshaders   %s\n",
                sjxl::gpu::toString((*device)->backend()), caps.apiName.c_str(), caps.adapterName.c_str(),
                caps.half ? "yes" : "no", sjxl::gpu::shaderDirectory().string().c_str());
    return 0;
}

int encode(int argc, char** argv) {
    if (argc < 4) return usage();
    sjxl::enc::Options options;
    for (int i = 4; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if ((arg == "--distance" || arg == "-d") && i + 1 < argc) {
            options.distance = std::strtof(argv[++i], nullptr);
        } else if (arg == "--no-aq") {
            options.adaptiveQuant = false;
        } else if (arg == "--no-cfl") {
            options.chromaFromLuma = false;
        } else if (arg == "--no-filters") {
            options.loopFilter = false;
        } else if (arg == "--lossless") {
            options.lossless = true;
        } else if (arg == "--no-acs") {
            options.acStrategy = false;
        } else if (arg == "--no-specials") {
            options.acsSpecials = false;
        } else if (arg == "--lambda" && i + 1 < argc) {
            options.acsLambda = std::strtof(argv[++i], nullptr);
        } else if (arg == "--clusters" && i + 1 < argc) {
            options.acClusters = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            return usage();
        }
    }
    // Lossless keeps alpha and 16-bit samples; lossy takes 8-bit RGB.
    auto image = options.lossless ? sjxl::image::readImage(argv[2]) : sjxl::image::readRgb8(argv[2]);
    if (!image) return fail(image.error());
    auto device = sjxl::gpu::Device::create();
    if (!device) return fail(device.error());
    auto library = std::make_shared<sjxl::gpu::ShaderLibrary>(*device);
    auto encoder = sjxl::enc::Encoder::create(library);
    if (!encoder) return fail(encoder.error());
    const auto start = std::chrono::steady_clock::now();
    auto bytes = (*encoder)->encode(*image, options);
    if (!bytes) return fail(bytes.error());
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (auto written = sjxl::image::writeFile(argv[3], *bytes); !written) return fail(written.error());
    std::printf("%ux%u -> %zu bytes (%.3f bpp) in %.1f ms\n", image->width, image->height, bytes->size(),
                8.0 * static_cast<double>(bytes->size()) / (static_cast<double>(image->width) * image->height), ms);
    return 0;
}

int decode(int argc, char** argv) {
    if (argc < 4) return usage();
    auto bytes = sjxl::image::readFile(argv[2]);
    if (!bytes) return fail(bytes.error());
    auto device = sjxl::gpu::Device::create();
    if (!device) return fail(device.error());
    auto library = std::make_shared<sjxl::gpu::ShaderLibrary>(*device);
    auto decoder = sjxl::dec::Decoder::create(library);
    if (!decoder) return fail(decoder.error());
    const auto start = std::chrono::steady_clock::now();
    auto frames = (*decoder)->decodeAll(*bytes);
    if (!frames) return fail(frames.error());
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto& first = frames->front().image;
    if (frames->size() == 1) {
        if (auto written = sjxl::image::writePng(argv[3], first); !written) return fail(written.error());
    } else {
        // An animation: one PNG per displayed frame, out.png -> out_000.png, out_001.png, ...
        const std::filesystem::path out(argv[3]);
        for (size_t i = 0; i < frames->size(); ++i) {
            std::filesystem::path path = out.parent_path() / (out.stem().string() + std::format("_{:03}", i) + out.extension().string());
            if (auto written = sjxl::image::writePng(path, (*frames)[i].image); !written) return fail(written.error());
        }
    }
    std::printf("%zu bytes -> %ux%u, %zu frame%s in %.1f ms\n", bytes->size(), first.width, first.height, frames->size(),
                frames->size() == 1 ? "" : "s", ms);
    return 0;
}

int pixel(int argc, char** argv) {
    if (argc < 5) return usage();
    auto a = sjxl::image::readRgb8(argv[2]);
    if (!a) return fail(a.error());
    const unsigned x = static_cast<unsigned>(std::atoi(argv[3]));
    const unsigned y = static_cast<unsigned>(std::atoi(argv[4]));
    if (x >= a->width || y >= a->height) return usage();
    const size_t i = (size_t{y} * a->width + x) * 3;
    std::printf("%u %u %u\n", a->pixels[i], a->pixels[i + 1], a->pixels[i + 2]);
    return 0;
}

int convert(int argc, char** argv) {
    if (argc < 4) return usage();
    auto a = sjxl::image::readRgb8(argv[2]);
    if (!a) return fail(a.error());
    if (auto written = sjxl::image::writePng(argv[3], *a); !written) return fail(written.error());
    return 0;
}

int compare(int argc, char** argv) {
    if (argc < 4) return usage();
    auto a = sjxl::image::readImage(argv[2]);
    if (!a) return fail(a.error());
    auto b = sjxl::image::readImage(argv[3]);
    if (!b) return fail(b.error());
    auto device = sjxl::gpu::Device::create();
    if (!device) return fail(device.error());
    sjxl::gpu::ShaderLibrary library(*device);
    auto m = sjxl::metrics::compare(library, *a, *b);
    if (!m) return fail(m.error());
    std::printf("PSNR %.2f dB, max abs diff %u at (%u,%u) channel %u\n", m->psnr, m->maxAbsDiff, m->maxAtX, m->maxAtY,
                m->maxAtChannel);
    return 0;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string_view command = argv[1];
    if (command == "info") return info(argc, argv);
    if (command == "encode") return encode(argc, argv);
    if (command == "compare") return compare(argc, argv);
    if (command == "pixel") return pixel(argc, argv);
    if (command == "convert") return convert(argc, argv);
    if (command == "decode") return decode(argc, argv);
    return usage();
}
