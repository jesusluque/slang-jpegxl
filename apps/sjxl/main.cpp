// Copyright (c) 2026 jesus luque.
//
// sjxl: the command line.
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
                 "  sjxl info                                   open the GPU device and describe it\n"
                 "  sjxl encode in.png out.jxl [--distance d]   lossy VarDCT (default distance 1.0)\n"
                 "  sjxl decode in.jxl out.png                  VarDCT frames\n"
                 "  sjxl compare a.png b.png                    PSNR and largest difference, measured on the GPU\n"
                 "  sjxl pixel a.png x y                        one pixel's RGB\n"
                 "  sjxl convert in.png out.png                 re-save as 8-bit RGB (drops alpha)\n");
    return 2;
}

int fail(const sjxl::Error& error) {
    std::fprintf(stderr, "sjxl: %s\n", error.toString().c_str());
    return 1;
}

int info() {
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
        } else {
            return usage();
        }
    }
    auto image = sjxl::image::readRgb8(argv[2]);
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
    auto image = (*decoder)->decode(*bytes);
    if (!image) return fail(image.error());
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (auto written = sjxl::image::writePng(argv[3], *image); !written) return fail(written.error());
    std::printf("%zu bytes -> %ux%u in %.1f ms\n", bytes->size(), image->width, image->height, ms);
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
    auto a = sjxl::image::readRgb8(argv[2]);
    if (!a) return fail(a.error());
    auto b = sjxl::image::readRgb8(argv[3]);
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
    if (command == "info") return info();
    if (command == "encode") return encode(argc, argv);
    if (command == "compare") return compare(argc, argv);
    if (command == "pixel") return pixel(argc, argv);
    if (command == "convert") return convert(argc, argv);
    if (command == "decode") return decode(argc, argv);
    return usage();
}
