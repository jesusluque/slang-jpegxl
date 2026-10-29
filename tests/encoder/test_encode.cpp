// Copyright (c) 2026 jesus luque.
//
// Conformance of the encoder against libjxl's decoder (Homebrew djxl): every
// test image must decode, and decode close to the source.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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
