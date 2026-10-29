// Copyright (c) 2026 jesus luque.
//
// Conformance of the encoder against libjxl's decoder (Homebrew djxl): every
// test image must decode, and decode close to the source.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <string>

#include "Compare.h"
#include "TestDevice.h"
#include "sjxl/encoder/Encoder.h"
#include "sjxl/image/Image.h"

using namespace sjxl;

namespace {

std::unique_ptr<enc::Encoder>& encoder() {
    static std::unique_ptr<enc::Encoder> instance;
    if (!instance) {
        auto made = enc::Encoder::create(test::testDevice().library);
        REQUIRE(made);
        instance = std::move(*made);
    }
    return instance;
}

}   // namespace

TEST_CASE("djxl decodes what the encoder writes", "[encoder][conformance]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) {
        SKIP("djxl (Homebrew jpeg-xl) is not installed");
    }
    const std::string name = GENERATE(std::string("berlin_7x5.png"), std::string("sveti_64.png"),
                                      std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    const float distance = GENERATE(1.0f, 3.0f);
    CAPTURE(name, distance);
    auto source = image::readRgb8(test::testData(name));
    REQUIRE(source);

    enc::Options options;
    options.distance = distance;
    auto encoded = encoder()->encode(*source, options);
    REQUIRE(encoded);
    const auto& stats = encoder()->stats();
    INFO(std::format("{} d={}: {} bytes (head {}, body {}, {} sections)", name, distance, encoded->size(),
                     stats.headBytes, stats.bodyBytes, stats.sectionBytes.size()));

    const std::filesystem::path jxl = test::scratchDir() / std::format("{}_{}.jxl", name, distance);
    const std::filesystem::path png = test::scratchDir() / std::format("{}_{}.decoded.png", name, distance);
    REQUIRE(image::writeFile(jxl, *encoded));
    std::filesystem::remove(png);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto decoded = image::readRgb8(png);
    REQUIRE(decoded);
    REQUIRE(decoded->width == source->width);
    REQUIRE(decoded->height == source->height);

    auto metrics = test::compare(*source, *decoded);
    REQUIRE(metrics);
    WARN(std::format("{} d={}: PSNR {:.2f} dB, max diff {} at ({},{}) c{}, {} bytes", name, distance, metrics->psnr,
                     metrics->maxAbsDiff, metrics->maxAtX, metrics->maxAtY, metrics->maxAtChannel, encoded->size()));
    // DCT8 only, adaptive quant field, CfL, Gaborish + EPF. Floors are
    // recorded in docs/decisions.md.
    CHECK(metrics->psnr >= (distance <= 1.0f ? 30.0 : 24.0));
}

TEST_CASE("lossless encodes decode bit-exact", "[encoder][lossless][conformance]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) {
        SKIP("djxl (Homebrew jpeg-xl) is not installed");
    }
    const std::string name = GENERATE(std::string("berlin_7x5.png"), std::string("sveti_64.png"),
                                      std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    CAPTURE(name);
    auto source = image::readRgb8(test::testData(name));
    REQUIRE(source);
    enc::Options options;
    options.lossless = true;
    auto encoded = encoder()->encode(*source, options);
    REQUIRE(encoded);
    const std::filesystem::path jxl = test::scratchDir() / std::format("{}_lossless.jxl", name);
    const std::filesystem::path png = test::scratchDir() / std::format("{}_lossless.decoded.png", name);
    REQUIRE(image::writeFile(jxl, *encoded));
    std::filesystem::remove(png);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto decoded = image::readRgb8(png);
    REQUIRE(decoded);
    auto metrics = test::compare(*source, *decoded);
    REQUIRE(metrics);
    WARN(std::format("{} lossless: {} bytes ({:.2f} bpp), {} clusters", name, encoded->size(),
                     8.0 * static_cast<double>(encoded->size()) / (static_cast<double>(source->width) * source->height),
                     encoder()->stats().numAcClusters));
    CHECK(metrics->maxAbsDiff == 0);
}

namespace {

/// berlin_200 with a gradient alpha and a transparent hole (RGBA), or as 16-bit
/// RGB with information in the low bytes; the decoder tests use the same images.
image::Rgb8 variantImage(bool alpha, bool deep) {
    auto base = image::readRgb8(test::testData("berlin_200.png"));
    REQUIRE(base);
    image::Rgb8 out;
    out.width = base->width;
    out.height = base->height;
    out.channels = alpha ? 4 : 3;
    out.bitDepth = deep ? 16 : 8;
    const size_t n = size_t{out.width} * out.height;
    out.pixels.resize(n * out.channels * (deep ? 2 : 1));
    for (size_t p = 0; p < n; ++p) {
        const uint32_t x = static_cast<uint32_t>(p % out.width), y = static_cast<uint32_t>(p / out.width);
        for (uint32_t c = 0; c < out.channels; ++c) {
            uint32_t v;
            if (c < 3) {
                v = base->pixels[p * 3 + c];
                if (deep) v = std::min<uint32_t>(65535, v * 257u + (x * 97u + c * 31u) % 251u);
            } else {
                const bool hole = (x > 60 && x < 100 && y > 30 && y < 70);
                v = hole ? 0 : 40 + (x * 215) / out.width;
                if (deep) v = v * 257u;
            }
            const size_t s = p * out.channels + c;
            if (deep) {
                out.pixels[2 * s] = static_cast<uint8_t>(v & 0xFF);
                out.pixels[2 * s + 1] = static_cast<uint8_t>(v >> 8);
            } else {
                out.pixels[s] = static_cast<uint8_t>(v);
            }
        }
    }
    return out;
}

}   // namespace

TEST_CASE("lossless encodes of alpha and 16-bit images decode bit-exact", "[encoder][lossless][conformance]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) {
        SKIP("djxl (Homebrew jpeg-xl) is not installed");
    }
    const auto [alpha, deep] = GENERATE(table<bool, bool>({{true, false}, {false, true}, {true, true}}));
    CAPTURE(alpha, deep);
    const image::Rgb8 source = variantImage(alpha, deep);
    enc::Options options;
    options.lossless = true;
    auto encoded = encoder()->encode(source, options);
    REQUIRE(encoded);
    const std::filesystem::path jxl = test::scratchDir() / std::format("variant_{}_{}.jxl", alpha, deep);
    const std::filesystem::path png = test::scratchDir() / std::format("variant_{}_{}.decoded.png", alpha, deep);
    REQUIRE(image::writeFile(jxl, *encoded));
    std::filesystem::remove(png);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto decoded = image::readImage(png);
    REQUIRE(decoded);
    REQUIRE(decoded->channels == source.channels);
    REQUIRE(decoded->bitDepth == source.bitDepth);
    auto metrics = test::compare(source, *decoded);
    REQUIRE(metrics);
    WARN(std::format("lossless alpha {} 16-bit {}: {} bytes", alpha, deep, encoded->size()));
    CHECK(metrics->maxAbsDiff == 0);
}

TEST_CASE("lossy encodes of alpha and 16-bit images decode with their channels", "[encoder][alpha][depth16][conformance]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) {
        SKIP("djxl (Homebrew jpeg-xl) is not installed");
    }
    const auto [alpha, deep] = GENERATE(table<bool, bool>({{true, false}, {false, true}, {true, true}}));
    CAPTURE(alpha, deep);
    const image::Rgb8 source = variantImage(alpha, deep);
    enc::Options options;
    options.distance = 1.0f;
    auto encoded = encoder()->encode(source, options);
    REQUIRE(encoded);
    const std::filesystem::path jxl = test::scratchDir() / std::format("lossy_variant_{}_{}.jxl", alpha, deep);
    const std::filesystem::path png = test::scratchDir() / std::format("lossy_variant_{}_{}.decoded.png", alpha, deep);
    REQUIRE(image::writeFile(jxl, *encoded));
    std::filesystem::remove(png);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto decoded = image::readImage(png);
    REQUIRE(decoded);
    REQUIRE(decoded->channels == source.channels);
    REQUIRE(decoded->bitDepth == source.bitDepth);
    auto metrics = test::compare(source, *decoded);
    REQUIRE(metrics);
    WARN(std::format("lossy alpha {} 16-bit {}: {} bytes, PSNR {:.2f} dB, max diff {} on channel {}", alpha, deep, encoded->size(),
                     metrics->psnr, metrics->maxAbsDiff, metrics->maxAtChannel));
    CHECK(metrics->psnr >= 30.0);
    // The alpha channel is coded losslessly: the largest difference must be in a colour channel.
    if (alpha) CHECK(metrics->maxAtChannel < 3);
}

TEST_CASE("encoding is deterministic", "[encoder]") {
    auto source = image::readRgb8(test::testData("sveti_crop_257x129.png"));
    REQUIRE(source);
    enc::Options options;
    auto first = encoder()->encode(*source, options);
    auto second = encoder()->encode(*source, options);
    REQUIRE(first);
    REQUIRE(second);
    CHECK(*first == *second);
}

TEST_CASE("the CPU entropy chains write the same file as the GPU kernels", "[encoder][enc-cpu]") {
    // DC tokens through the weighted predictor and every ANS section (EntropyCpu.cpp)
    // against the kernels: single-section, several groups, progressive passes, a DC frame.
    struct Case { std::string name; uint32_t progressive; bool dcFrame; float distance; bool lossless; bool fixedTree; };
    const Case c = GENERATE(Case{"berlin_7x5.png", 0, false, 1.0f, false, false}, Case{"berlin_200.png", 0, false, 1.0f, false, false},
                            Case{"sveti_stefan_320.png", 0, false, 0.5f, false, false}, Case{"sveti_stefan_320.png", 1, false, 1.0f, false, false},
                            Case{"sveti_stefan_320.png", 2, true, 1.0f, false, false}, Case{"sveti_stefan_320.png", 0, false, 0.0f, true, false},
                            Case{"berlin_200.png", 0, false, 0.0f, true, true});
    CAPTURE(c.name, c.progressive, c.dcFrame, c.lossless, c.fixedTree);
    auto source = image::readRgb8(test::testData(c.name));
    REQUIRE(source);
    enc::Options options;
    options.distance = c.distance;
    options.progressive = c.progressive;
    options.progressiveDc = c.dcFrame;
    options.lossless = c.lossless;
    options.learnedTree = !c.fixedTree;
    setenv("SJXL_ENC_CPU", "0", 1);
    auto gpu = encoder()->encode(*source, options);
    setenv("SJXL_ENC_CPU", "1", 1);
    auto cpu = encoder()->encode(*source, options);
    unsetenv("SJXL_ENC_CPU");
    REQUIRE(gpu);
    REQUIRE(cpu);
    CHECK(gpu->size() == cpu->size());
    CHECK(*gpu == *cpu);
}
