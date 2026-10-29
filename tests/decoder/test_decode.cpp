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

TEST_CASE("the decoder reproduces cjxl's lossless output exactly", "[decoder][conformance][cjxl][modular]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("sveti_64.png"), std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    const int effort = GENERATE(1, 3, 7, 9);
    const int responsive = GENERATE(0, 1);   // 1: Squeeze (progressive lossless)
    CAPTURE(name, effort, responsive);
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_lossless_{}_e{}_r{}.jxl", name, effort, responsive);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_lossless_{}_e{}_r{}.djxl.png", name, effort, responsive);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" -d 0 -e {} --responsive={} >/dev/null 2>&1", cjxl.string(),
                                  test::testData(name).string(), jxl.string(), effort, responsive)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("{} e{} r{} lossless: unsupported: {}", name, effort, responsive, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{} e{} r{} lossless: vs djxl max diff {}", name, effort, responsive, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff == 0);
}

TEST_CASE("the decoder matches djxl on cjxl's lossy modular output", "[decoder][conformance][cjxl][modular]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("sveti_64.png"), std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    const int effort = GENERATE(3, 7);
    CAPTURE(name, effort);
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_lossymod_{}_e{}.jxl", name, effort);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_lossymod_{}_e{}.djxl.png", name, effort);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" -m 1 -d 1 -e {} >/dev/null 2>&1", cjxl.string(), test::testData(name).string(),
                                  jxl.string(), effort)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("{} e{} lossy modular: unsupported: {}", name, effort, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{} e{} lossy modular: vs djxl PSNR {:.2f} dB, max diff {}", name, effort, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= 1);
}

namespace {

/// A flat page of repeated glyph-like bitmaps: cjxl builds a patch dictionary for it.
std::filesystem::path glyphPage() {
    const std::filesystem::path path = test::scratchDir() / "glyph_page.ppm";
    constexpr uint32_t w = 480, h = 320;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) { rgb[i] = 235; rgb[i + 1] = 235; rgb[i + 2] = 230; }
    uint32_t seed = 12345;
    const auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7FFF; };
    std::vector<std::vector<uint8_t>> glyphs(10, std::vector<uint8_t>(14 * 18));
    for (auto& g : glyphs) for (auto& bit : g) bit = rnd() < 0x4000 ? 1 : 0;
    for (uint32_t ty = 0; ty < 8; ++ty) {
        for (uint32_t tx = 0; tx < 24; ++tx) {
            const auto& g = glyphs[(tx * 3 + ty * 5) % 10];
            for (uint32_t y = 0; y < 18; ++y) {
                for (uint32_t x = 0; x < 14; ++x) {
                    if (!g[y * 14 + x]) continue;
                    const size_t at = (static_cast<size_t>(16 + ty * 36 + y) * w + 12 + tx * 19 + x) * 3;
                    rgb[at] = 30; rgb[at + 1] = 30; rgb[at + 2] = 60;
                }
            }
        }
    }
    std::string header = std::format("P6\n{} {}\n255\n", w, h);
    std::vector<uint8_t> file(header.begin(), header.end());
    file.insert(file.end(), rgb.begin(), rgb.end());
    REQUIRE(image::writeFile(path, file));
    return path;
}

}   // namespace

TEST_CASE("the decoder applies cjxl's patches like djxl", "[decoder][conformance][cjxl][patches]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::filesystem::path page = glyphPage();
    const std::string options = GENERATE(std::string("-d 1 -e 7"), std::string("-d 0 -e 7"), std::string("-m 1 -d 1 -e 7"));
    CAPTURE(options);
    const std::string tag = options == "-d 1 -e 7" ? "vardct" : options == "-d 0 -e 7" ? "lossless" : "modular";
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_patches_{}.jxl", tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_patches_{}.djxl.png", tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), page.string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = dec::Decoder::info(*bytes);
    REQUIRE(parsed);
    // The point of the test: a reference frame and a displayed frame with patches.
    if (parsed->frames.size() != 2 || !(parsed->frame.header.flags & codestream::FrameHeader::kPatches)) {
        WARN(std::format("cjxl {} did not use patches on the glyph page", options));
    }
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("patches {}: {} frames, vs djxl PSNR {:.2f} dB, max diff {}", options, parsed->frames.size(), metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= (tag == "lossless" ? 0 : 1));
}

TEST_CASE("the decoder matches djxl on cjxl's progressive output", "[decoder][conformance][cjxl][progressive]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("sveti_64.png"), std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"),
                                      std::string("sveti_stefan_320.png"));
    // -p: passes with a DC frame (VarDCT), a permuted TOC (modular).
    const std::string options = GENERATE(std::string("-p -d 1 -e 5"), std::string("-p -d 1 -e 7"), std::string("-p -d 0 -e 7"),
                                         std::string("-p -m 1 -d 1 -e 7"), std::string("-d 1 -e 7 --progressive_dc=2"));
    CAPTURE(name, options);
    std::string tag = options;
    for (char& ch : tag) if (ch == ' ' || ch == '-') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_prog_{}{}.jxl", name, tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_prog_{}{}.djxl.png", name, tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), test::testData(name).string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("{} {}: unsupported: {}", name, options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{} {}: vs djxl PSNR {:.2f} dB, max diff {}", name, options, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= (options.find("-d 0") != std::string::npos ? 0 : 1));
}

TEST_CASE("the decoder matches djxl on cjxl's upsampled and noisy output", "[decoder][conformance][cjxl][features]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"), std::string("sveti_stefan_320.png"));
    const std::string options = GENERATE(std::string("-d 1 -e 5 --resampling=2"), std::string("-d 1 -e 5 --resampling=4"),
                                         std::string("-d 1 -e 7 --resampling=8"), std::string("-d 0 -e 7 --resampling=2"),
                                         std::string("-d 1 -e 5 --photon_noise_iso=3200"), std::string("-d 1 -e 7 --photon_noise_iso=800 --resampling=2"));
    CAPTURE(name, options);
    std::string tag = options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_feat_{}{}.jxl", name, tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_feat_{}{}.djxl.png", name, tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), test::testData(name).string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("{} {}: unsupported: {}", name, options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("{} {}: vs djxl PSNR {:.2f} dB, max diff {}", name, options, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= 1);
}

namespace {

/// berlin_200 with a gradient alpha and a transparent hole, as an RGBA PNG.
std::filesystem::path alphaImage() {
    const std::filesystem::path path = test::scratchDir() / "berlin_alpha.png";
    auto base = image::readRgb8(test::testData("berlin_200.png"));
    REQUIRE(base);
    image::Rgb8 rgba;
    rgba.width = base->width;
    rgba.height = base->height;
    rgba.channels = 4;
    rgba.pixels.resize(size_t{rgba.width} * rgba.height * 4);
    for (uint32_t y = 0; y < rgba.height; ++y) {
        for (uint32_t x = 0; x < rgba.width; ++x) {
            const size_t src = (size_t{y} * rgba.width + x) * 3, dst = (size_t{y} * rgba.width + x) * 4;
            rgba.pixels[dst] = base->pixels[src];
            rgba.pixels[dst + 1] = base->pixels[src + 1];
            rgba.pixels[dst + 2] = base->pixels[src + 2];
            const bool hole = (x > 60 && x < 100 && y > 30 && y < 70);
            rgba.pixels[dst + 3] = hole ? 0 : static_cast<uint8_t>(40 + (x * 215) / rgba.width);
        }
    }
    REQUIRE(image::writePng(path, rgba));
    return path;
}

}   // namespace

TEST_CASE("the decoder matches djxl on images with alpha", "[decoder][conformance][cjxl][alpha]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::filesystem::path png0 = alphaImage();
    const std::string options = GENERATE(std::string("-d 1 -e 5"), std::string("-d 1 -e 7"), std::string("-d 0 -e 7"), std::string("-m 1 -d 1 -e 7"),
                                         std::string("-d 1 -e 5 --resampling=2"), std::string("-d 1 -e 7 -p"));
    CAPTURE(options);
    std::string tag = options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_alpha{}.jxl", tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_alpha{}.djxl.png", tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), png0.string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readImage8(png);
    REQUIRE(reference);
    REQUIRE(reference->channels == 4);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("alpha {}: unsupported: {}", options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    REQUIRE(ours->channels == 4);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("alpha {}: vs djxl PSNR {:.2f} dB, max diff {} (channel {})", options, metrics->psnr, metrics->maxAbsDiff, metrics->maxAtChannel));
    CHECK(metrics->maxAbsDiff <= (options == "-d 0 -e 7" ? 0 : 1));
}

TEST_CASE("the decoder matches djxl on recompressed JPEGs", "[decoder][conformance][cjxl][jpeg]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    const std::filesystem::path sips = test::findTool("sips");   // macOS: makes 4:2:0 JPEGs from the PNGs
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    if (sips.empty()) SKIP("no JPEG encoder (sips) to make test JPEGs");
    const std::string name = GENERATE(std::string("berlin_200.png"), std::string("sveti_stefan_320.png"), std::string("sveti_crop_257x129.png"));
    const int quality = GENERATE(75, 92);
    CAPTURE(name, quality);
    const std::filesystem::path jpg = test::scratchDir() / std::format("dec_jpeg_{}_q{}.jpg", name, quality);
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_jpeg_{}_q{}.jxl", name, quality);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_jpeg_{}_q{}.djxl.png", name, quality);
    REQUIRE(test::run(std::format("\"{}\" -s format jpeg -s formatOptions {} \"{}\" --out \"{}\" >/dev/null 2>&1", sips.string(), quality,
                                  test::testData(name).string(), jpg.string())));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", cjxl.string(), jpg.string(), jxl.string())));   // lossless transcode
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = dec::Decoder::info(*bytes);
    REQUIRE(parsed);
    REQUIRE(parsed->frame.header.colorTransform == 2);   // YCbCr, as a JPEG transcode is
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("jpeg {} q{}: unsupported: {}", name, quality, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("jpeg {} q{}: vs djxl PSNR {:.2f} dB, max diff {}", name, quality, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= 1);
}

TEST_CASE("the decoder matches djxl on images with an ICC profile", "[decoder][conformance][cjxl][icc]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    // djxl writes JPEGs with an embedded ICC profile; cjxl keeps it in the codestream.
    const std::filesystem::path tmpJxl = test::scratchDir() / "icc_source.jxl";
    const std::filesystem::path jpg = test::scratchDir() / "icc_source.jpg";
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" -d 0 -e 1 >/dev/null 2>&1", cjxl.string(), test::testData("berlin_200.png").string(), tmpJxl.string())));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), tmpJxl.string(), jpg.string())));
    // "" transcodes the JPEG (cjxl's default even with -d 0); --lossless_jpeg=0 encodes its pixels.
    const std::string options = GENERATE(std::string(""), std::string("--lossless_jpeg=0 -d 0 -e 3"),
                                         std::string("--lossless_jpeg=0 -d 0 -e 7 --responsive=1"));
    CAPTURE(options);
    std::string tag = options.empty() ? "transcode" : options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_icc_{}.jxl", tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_icc_{}.djxl.png", tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), jpg.string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = dec::Decoder::info(*bytes);
    REQUIRE(parsed);
    REQUIRE(parsed->metadata.wantIcc);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("icc {}: unsupported: {}", tag, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("icc {}: vs djxl PSNR {:.2f} dB, max diff {}", tag, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= (options.empty() ? 1 : 0));   // the transcode goes through YCbCr floats; lossless is exact
}

namespace {

/// berlin_200 as a 16-bit RGB PNG: samples * 257 plus a smooth low-byte ramp, so the
/// low bytes carry information.
std::filesystem::path deepImage() {
    const std::filesystem::path path = test::scratchDir() / "berlin_16.png";
    auto base = image::readRgb8(test::testData("berlin_200.png"));
    REQUIRE(base);
    image::Rgb8 deep;
    deep.width = base->width;
    deep.height = base->height;
    deep.channels = 3;
    deep.bitDepth = 16;
    deep.pixels.resize(size_t{deep.width} * deep.height * 3 * 2);
    for (size_t i = 0; i < size_t{deep.width} * deep.height * 3; ++i) {
        const uint32_t x = static_cast<uint32_t>((i / 3) % deep.width);
        const uint32_t v = std::min<uint32_t>(65535, base->pixels[i] * 257u + (x * 97u) % 251u);
        deep.pixels[2 * i] = static_cast<uint8_t>(v & 0xFF);
        deep.pixels[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
    REQUIRE(image::writePng(path, deep));
    return path;
}

}   // namespace

TEST_CASE("the decoder matches djxl on 16-bit images", "[decoder][conformance][cjxl][depth16]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::filesystem::path png0 = deepImage();
    const std::string options = GENERATE(std::string("-d 0 -e 3"), std::string("-d 0 -e 7 --responsive=1"), std::string("-d 1 -e 5"),
                                         std::string("-m 1 -d 1 -e 7"));
    CAPTURE(options);
    std::string tag = options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_16{}.jxl", tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_16{}.djxl.png", tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), png0.string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readImage(png);
    REQUIRE(reference);
    REQUIRE(reference->bitDepth == 16);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("16-bit {}: unsupported: {}", options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    REQUIRE(ours->bitDepth == 16);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("16-bit {}: vs djxl PSNR {:.2f} dB, max diff {} / 65535", options, metrics->psnr, metrics->maxAbsDiff));
    // Lossless must be exact; lossy goes through floats on both sides, so a few 16-bit steps may differ.
    CHECK(metrics->maxAbsDiff <= (options.find("-d 0") != std::string::npos ? 0u : 8u));
}

TEST_CASE("the decoder matches djxl on animations", "[decoder][conformance][cjxl][animation]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    const std::filesystem::path python = test::findTool("python3");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    if (python.empty()) SKIP("python3 is needed to make and split APNGs");
    const std::filesystem::path support = std::filesystem::path(SJXL_TEST_DATA_DIR) / ".." / "support";
    // "full": every frame a whole canvas (replace); "partial": frames 1.. are sub-rectangles blended over with alpha.
    const std::string layout = GENERATE(std::string("full"), std::string("partial"));
    const std::filesystem::path apng = test::scratchDir() / std::format("anim_source_{}.png", layout);
    REQUIRE(test::run(std::format("\"{}\" -I \"{}\" \"{}\" 4 96 64 {}", python.string(), (support / "apng_make.py").string(), apng.string(), layout)));
    const std::string options = GENERATE(std::string("-d 0 -e 3"), std::string("-d 1 -e 5"), std::string("-m 1 -d 1 -e 7"));
    CAPTURE(layout, options);
    std::string tag = layout + options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_anim{}.jxl", tag);
    const std::filesystem::path out = test::scratchDir() / std::format("dec_anim{}.djxl.png", tag);
    const std::filesystem::path prefix = test::scratchDir() / std::format("dec_anim{}_djxl", tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), apng.string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), out.string())));
    REQUIRE(test::run(std::format("\"{}\" -I \"{}\" \"{}\" \"{}\" >/dev/null", python.string(), (support / "apng_split.py").string(), out.string(), prefix.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = dec::Decoder::info(*bytes);
    REQUIRE(parsed);
    REQUIRE(parsed->metadata.haveAnimation);
    auto ours = decoder().decodeAll(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("animation {}: unsupported: {}", options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    REQUIRE(ours->size() == 4);
    for (size_t i = 0; i < ours->size(); ++i) {
        auto reference = image::readImage(std::format("{}_{:03}.png", prefix.string(), i));
        REQUIRE(reference);
        auto metrics = test::compare(*reference, (*ours)[i].image);
        REQUIRE(metrics);
        WARN(std::format("animation {} {} frame {} (duration {}): vs djxl PSNR {:.2f} dB, max diff {}", layout, options, i, (*ours)[i].duration,
                         metrics->psnr, metrics->maxAbsDiff));
        // Lossless full frames are exact; alpha blending runs in floats on both sides, so a blended frame may differ by one.
        const bool exact = options.find("-d 0") != std::string::npos && (layout == "full" || i == 0);
        CHECK(metrics->maxAbsDiff <= (exact ? 0u : 1u));
    }
}

TEST_CASE("the decoder matches djxl across modular group sizes", "[decoder][conformance][cjxl][groups]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (djxl.empty() || cjxl.empty()) SKIP("cjxl/djxl are not installed");
    const std::string name = GENERATE(std::string("berlin_200.png"), std::string("sveti_stefan_320.png"));
    // -g 0: 128 px groups, where cjxl chooses an RCT per group (group-local transforms).
    const std::string options = GENERATE(std::string("-d 0 -e 7 -g 0"), std::string("-d 0 -e 9 -g 0"), std::string("-m 1 -d 1 -e 7 -g 0"),
                                         std::string("-d 0 -e 7 -g 3"));
    CAPTURE(name, options);
    std::string tag = options;
    for (char& ch : tag) if (ch == ' ' || ch == '-' || ch == '=') ch = '_';
    const std::filesystem::path jxl = test::scratchDir() / std::format("dec_groups_{}{}.jxl", name, tag);
    const std::filesystem::path png = test::scratchDir() / std::format("dec_groups_{}{}.djxl.png", name, tag);
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" {} >/dev/null 2>&1", cjxl.string(), test::testData(name).string(), jxl.string(), options)));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), png.string())));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto reference = image::readRgb8(png);
    REQUIRE(reference);
    auto ours = decoder().decode(*bytes);
    if (!ours && ours.error().code() == ErrorCode::Unsupported) {
        WARN(std::format("groups {} {}: unsupported: {}", name, options, ours.error().message()));
        SUCCEED();
        return;
    }
    REQUIRE(ours);
    auto metrics = test::compare(*reference, *ours);
    REQUIRE(metrics);
    WARN(std::format("groups {} {}: vs djxl PSNR {:.2f} dB, max diff {}", name, options, metrics->psnr, metrics->maxAbsDiff));
    CHECK(metrics->maxAbsDiff <= (options.find("-d 0") != std::string::npos ? 0u : 1u));
}

TEST_CASE("the encoder's animations round-trip and djxl reads them", "[encoder][decoder][conformance][animation]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) SKIP("djxl is not installed");
    const bool lossless = GENERATE(false, true);
    CAPTURE(lossless);
    auto base = image::readRgb8(test::testData("berlin_200.png"));
    REQUIRE(base);
    // Three frames: the image and two channel rotations of it (the frames must differ to tell them apart).
    std::vector<image::Rgb8> frames;
    for (uint32_t r = 0; r < 3; ++r) {
        image::Rgb8 f = *base;
        for (size_t p = 0; p + 2 < f.pixels.size(); p += 3) {
            for (uint32_t c = 0; c < 3; ++c) f.pixels[p + c] = base->pixels[p + (c + r) % 3];
        }
        frames.push_back(std::move(f));
    }
    const std::vector<uint32_t> durations{1, 2, 3};
    enc::Options options;
    options.lossless = lossless;
    options.distance = 1.0f;
    auto encoded = encoder().encodeAnimation(frames, durations, 10, 0, options);
    REQUIRE(encoded);
    const std::filesystem::path jxl = test::scratchDir() / std::format("enc_anim_{}.jxl", lossless);
    const std::filesystem::path out = test::scratchDir() / std::format("enc_anim_{}.djxl.png", lossless);
    REQUIRE(image::writeFile(jxl, *encoded));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), out.string())));
    auto parsed = dec::Decoder::info(*encoded);
    REQUIRE(parsed);
    REQUIRE(parsed->metadata.haveAnimation);
    REQUIRE(parsed->frames.size() == 3);
    auto ours = decoder().decodeAll(*encoded);
    REQUIRE(ours);
    REQUIRE(ours->size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        CHECK((*ours)[i].duration == durations[i]);
        auto metrics = test::compare(frames[i], (*ours)[i].image);
        REQUIRE(metrics);
        WARN(std::format("animation lossless {} frame {}: {} bytes total, PSNR {:.2f} dB, max diff {}", lossless, i, encoded->size(),
                         metrics->psnr, metrics->maxAbsDiff));
        if (lossless) CHECK(metrics->maxAbsDiff == 0);
        else CHECK(metrics->psnr >= 30.0);
    }
}

TEST_CASE("the encoder's progressive passes decode like its single pass", "[encoder][decoder][conformance][progressive]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) SKIP("djxl is not installed");
    // 1: two precision passes (libjxl qprogressive); 2: three frequency passes (libjxl progressive).
    const uint32_t progressive = GENERATE(1u, 2u);
    const std::string name = GENERATE(std::string("berlin_200.png"), std::string("sveti_crop_257x129.png"));
    CAPTURE(progressive, name);
    auto source = image::readRgb8(test::testData(name));
    REQUIRE(source);
    enc::Options options;
    options.distance = 1.0f;
    auto single = encoder().encode(*source, options);
    REQUIRE(single);
    options.progressive = progressive;
    auto encoded = encoder().encode(*source, options);
    REQUIRE(encoded);
    auto parsed = dec::Decoder::info(*encoded);
    REQUIRE(parsed);
    REQUIRE(parsed->frames.size() == 1);
    CHECK(parsed->frames[0].header.numPasses == (progressive == 1 ? 2u : 3u));
    const std::filesystem::path jxl = test::scratchDir() / std::format("enc_prog_{}_{}.jxl", progressive, name);
    const std::filesystem::path out = test::scratchDir() / std::format("enc_prog_{}_{}.djxl.png", progressive, name);
    REQUIRE(image::writeFile(jxl, *encoded));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), out.string())));
    auto reference = image::readRgb8(out);
    REQUIRE(reference);
    // The passes add up to the single pass's coefficients: djxl's output equals its decode of the single-pass file.
    auto singleDecoded = decoder().decode(*single);
    REQUIRE(singleDecoded);
    auto ours = decoder().decode(*encoded);
    REQUIRE(ours);
    auto vsDjxl = test::compare(*reference, *ours);
    REQUIRE(vsDjxl);
    auto vsSingle = test::compare(*singleDecoded, *ours);
    REQUIRE(vsSingle);
    WARN(std::format("progressive {} {}: {} bytes vs {} single pass; vs djxl max diff {}, vs single pass max diff {}", progressive, name,
                     encoded->size(), single->size(), vsDjxl->maxAbsDiff, vsSingle->maxAbsDiff));
    CHECK(vsDjxl->maxAbsDiff <= 1);
    CHECK(vsSingle->maxAbsDiff == 0);
}

TEST_CASE("the encoder's DC frame feeds the VarDCT frame like djxl", "[encoder][decoder][conformance][progressive][dcframe]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) SKIP("djxl is not installed");
    const std::string name = GENERATE(std::string("berlin_7x5.png"), std::string("berlin_200.png"), std::string("sveti_stefan_320.png"));
    const bool passes = GENERATE(false, true);   // with the three frequency passes as well: what cjxl -p emits
    CAPTURE(name, passes);
    auto source = image::readRgb8(test::testData(name));
    REQUIRE(source);
    enc::Options options;
    options.distance = 1.0f;
    options.progressiveDc = true;
    options.progressive = passes ? 2u : 0u;
    auto encoded = encoder().encode(*source, options);
    REQUIRE(encoded);
    auto parsed = dec::Decoder::info(*encoded);
    REQUIRE(parsed);
    REQUIRE(parsed->frames.size() == 2);
    CHECK(parsed->frames[0].header.frameType == 1);   // kDCFrame
    CHECK(parsed->frames[0].header.isModular);
    CHECK((parsed->frames[1].header.flags & codestream::FrameHeader::kUseDcFrame) != 0);
    const std::filesystem::path jxl = test::scratchDir() / std::format("enc_dcframe_{}_{}.jxl", passes, name);
    const std::filesystem::path out = test::scratchDir() / std::format("enc_dcframe_{}_{}.djxl.png", passes, name);
    REQUIRE(image::writeFile(jxl, *encoded));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), out.string())));
    auto reference = image::readRgb8(out);
    REQUIRE(reference);
    auto ours = decoder().decode(*encoded);
    REQUIRE(ours);
    auto vsDjxl = test::compare(*reference, *ours);
    REQUIRE(vsDjxl);
    auto vsSource = test::compare(*source, *ours);
    REQUIRE(vsSource);
    WARN(std::format("DC frame {} passes {}: {} bytes (DC frame {}), vs djxl max diff {}, PSNR {:.2f} dB", name, passes, encoded->size(),
                     encoder().stats().dcFrameBytes, vsDjxl->maxAbsDiff, vsSource->psnr));
    CHECK(vsDjxl->maxAbsDiff <= 1);
    CHECK(vsSource->psnr >= 30.0);
}

namespace {

/// A few-colour image: `numColors` colours in small blocks (what a palette is for).
image::Rgb8 fewColourImage(uint32_t width, uint32_t height, uint32_t numColors, bool alpha) {
    image::Rgb8 img;
    img.width = width;
    img.height = height;
    img.channels = alpha ? 4 : 3;
    img.bitDepth = 8;
    img.pixels.resize(size_t{width} * height * img.channels);
    uint32_t seed = 12345;
    std::vector<std::array<uint8_t, 4>> colours(numColors);
    for (auto& c : colours) {
        for (uint8_t& v : c) { seed = seed * 1664525u + 1013904223u; v = static_cast<uint8_t>(seed >> 24); }
        c[3] = (seed & 1) ? 255 : 128;
    }
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto& c = colours[((x / 7) * 3 + (y / 5) * 5 + (x * y) / 300) % numColors];
            for (uint32_t k = 0; k < img.channels; ++k) img.pixels[(size_t{y} * width + x) * img.channels + k] = c[k];
        }
    }
    return img;
}

}   // namespace

TEST_CASE("lossless palettes decode bit-exact", "[encoder][decoder][conformance][lossless][palette]") {
    const std::filesystem::path djxl = test::findTool("djxl");
    if (djxl.empty()) SKIP("djxl is not installed");
    // 16 colours over two groups (the palette in the global stream, the indices in the groups), 200 colours in one
    // group, and 16 colours with alpha (a four-channel palette).
    const auto [numColors, width, height, alpha] =
        GENERATE(table<uint32_t, uint32_t, uint32_t, bool>({{16, 600, 300, false}, {200, 100, 80, false}, {16, 300, 200, true}}));
    CAPTURE(numColors, width, height, alpha);
    const image::Rgb8 source = fewColourImage(width, height, numColors, alpha);
    enc::Options options;
    options.lossless = true;
    auto encoded = encoder().encode(source, options);
    REQUIRE(encoded);
    CHECK(encoder().stats().palette >= 2);
    const std::filesystem::path jxl = test::scratchDir() / std::format("palette_{}_{}_{}.jxl", numColors, width, alpha);
    const std::filesystem::path out = test::scratchDir() / std::format("palette_{}_{}_{}.djxl.png", numColors, width, alpha);
    REQUIRE(image::writeFile(jxl, *encoded));
    REQUIRE(test::run(std::format("\"{}\" \"{}\" \"{}\" >/dev/null 2>&1", djxl.string(), jxl.string(), out.string())));
    auto reference = image::readImage(out);
    REQUIRE(reference);
    REQUIRE(reference->channels == source.channels);
    auto vsDjxl = test::compare(source, *reference);
    REQUIRE(vsDjxl);
    auto ours = decoder().decode(*encoded);
    REQUIRE(ours);
    auto vsOurs = test::compare(source, *ours);
    REQUIRE(vsOurs);
    WARN(std::format("palette {} colours {}x{} alpha {}: {} bytes, {} palette entries", numColors, width, height, alpha, encoded->size(),
                     encoder().stats().palette));
    CHECK(vsDjxl->maxAbsDiff == 0);
    CHECK(vsOurs->maxAbsDiff == 0);
}
