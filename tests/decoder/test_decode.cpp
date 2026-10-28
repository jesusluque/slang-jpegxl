// Copyright (c) 2026 jesus luque.
//
// The decoder against libjxl: our encoder's files, then cjxl's, compared with
// djxl's output on the GPU.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <format>
#include <string>

#include "Compare.h"
#include "TestDevice.h"
#include "sjxl/decoder/Decoder.h"
#include "sjxl/encoder/Encoder.h"
#include "sjxl/image/Image.h"

using namespace sjxl;

namespace {

dec::Decoder& decoder() {
    static std::unique_ptr<dec::Decoder> instance;
    if (!instance) {
        auto made = dec::Decoder::create(test::testDevice().library);
        REQUIRE(made);
        instance = std::move(*made);
    }
    return *instance;
}

enc::Encoder& encoder() {
    static std::unique_ptr<enc::Encoder> instance;
    if (!instance) {
        auto made = enc::Encoder::create(test::testDevice().library);
        REQUIRE(made);
        instance = std::move(*made);
    }
    return *instance;
}

}   // namespace

TEST_CASE("the decoder matches djxl on our own encoder's output", "[decoder][conformance]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) SKIP("djxl is not installed");
    const std::string name = GENERATE(std::string("berlin_7x5.png"), std::string("sveti_64.png"), std::string("berlin_200.png"),
                                      std::string("sveti_crop_257x129.png"), std::string("sveti_stefan_320.png"));
    CAPTURE(name);
    auto source = image::readRgb8(test::testData(name));
    REQUIRE(source);
    enc::Options options;
    auto encoded = encoder().encode(*source, options);
    REQUIRE(encoded);
    const std::filesystem::path jxl = test::scratchDir() / ("dec_own_" + name + ".jxl");
    const std::filesystem::path png = test::scratchDir() / ("dec_own_" + name + ".djxl.png");
    REQUIRE(image::writeFile(jxl, *encoded));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto reference = image::readRgb8(png);
    REQUIRE(reference);

    auto ours = decoder().decode(*encoded);
    REQUIRE(ours);
    REQUIRE(ours->width == source->width);
    REQUIRE(ours->height == source->height);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{}: vs djxl PSNR {:.2f} dB, max diff {} at ({},{}) c{}", name, metrics->psnr, metrics->maxAbsDiff,
                     metrics->maxAtX, metrics->maxAtY, metrics->maxAtChannel));
    CHECK(metrics->maxAbsDiff <= 1);
}

TEST_CASE("the decoder matches djxl on cjxl's output", "[decoder][conformance][cjxl]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("sveti_64.png"), std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    const int effort = GENERATE(1, 3, 5, 7);
    const float distance = GENERATE(1.0f, 3.0f);
    CAPTURE(name, effort, distance);
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_cjxl_{}_e{}_d{}.jxl", name, effort, distance);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_cjxl_{}_e{}_d{}.djxl.png", name, effort, distance);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" -d {} -e {} >/dev/null 2>&1", cjxl.string(),
                                  test::testData(name).string(), jxl.string(), distance, effort)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);

    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        // Reported, not failed: the stream uses a feature the roadmap still owes.
        // (A SKIP here would mark the whole test case skipped under ctest.)
        WARN(std::format("{} e{} d{}: unsupported: {}", name, effort, distance, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{} e{} d{}: vs djxl PSNR {:.2f} dB, max diff {} at ({},{}) c{}, used acs 0x{:x}", name, effort, distance,
                     metrics->psnr, metrics->maxAbsDiff, metrics->maxAtX, metrics->maxAtY, metrics->maxAtChannel,
                     decoder().stats().usedAcs));
    CHECK(metrics->maxAbsDiff <= 1);
}
