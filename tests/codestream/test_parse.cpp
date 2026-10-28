// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "Compare.h"
#include "sjxl/codestream/Parse.h"
#include "sjxl/image/Image.h"

using namespace sjxl;

TEST_CASE("the parser reads cjxl's headers and TOC", "[codestream][parse]") {
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (cjxl.empty()) {
        SKIP("cjxl (Homebrew jpeg-xl) is not installed");
    }
    const std::filesystem::path jxl = test::scratchDir() / "parse_257x129.jxl";
    REQUIRE(test::run(cjxl.string() + " \"" + test::testData("sveti_crop_257x129.png").string() + "\" \"" + jxl.string() +
                      "\" -d 1 -e 3 >/dev/null 2>&1"));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = codestream::parse(*bytes);
    REQUIRE(parsed);
    CHECK(parsed->metadata.xsize == 257);
    CHECK(parsed->metadata.ysize == 129);
    CHECK(parsed->metadata.xybEncoded);
    CHECK(parsed->metadata.bitsPerSample == 8);
    CHECK(!parsed->frame.header.isModular);
    CHECK(parsed->frame.dim.numGroups() == 2);
    // Sections: DcGlobal, 1 DC group, AcGlobal, 2 AC groups.
    REQUIRE(parsed->frame.sections.size() == 5);
    size_t end = 0;
    for (const auto& s : parsed->frame.sections) {
        CHECK(s.byteSize > 0);
        end = s.byteOffset + s.byteSize;
    }
    CHECK(end == parsed->codestream.size());
}

TEST_CASE("the parser reads a single-section codestream", "[codestream][parse]") {
    const std::filesystem::path cjxl = test::findTool("cjxl");
    if (cjxl.empty()) {
        SKIP("cjxl (Homebrew jpeg-xl) is not installed");
    }
    const std::filesystem::path jxl = test::scratchDir() / "parse_64.jxl";
    REQUIRE(test::run(cjxl.string() + " \"" + test::testData("sveti_64.png").string() + "\" \"" + jxl.string() +
                      "\" -d 1 -e 3 >/dev/null 2>&1"));
    auto bytes = image::readFile(jxl);
    REQUIRE(bytes);
    auto parsed = codestream::parse(*bytes);
    REQUIRE(parsed);
    CHECK(parsed->frame.singleSection);
    REQUIRE(parsed->frame.sections.size() == 1);
    CHECK(parsed->frame.sections[0].byteOffset + parsed->frame.sections[0].byteSize == parsed->codestream.size());
}
