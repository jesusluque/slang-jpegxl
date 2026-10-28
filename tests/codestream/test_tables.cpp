// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>

#include "sjxl/codestream/Headers.h"
#include "sjxl/codestream/Tables.h"

using namespace sjxl::codestream;

#include "cujpegxl_dct8.inc"

TEST_CASE("DCT8 dequant weights match libjxl's extracted table", "[codestream][tables]") {
    const auto ours = dct8DequantWeights();
    for (size_t c = 0; c < 3; ++c) {
        for (size_t k = 0; k < 64; ++k) {
            CAPTURE(c, k);
            // libjxl interpolates with FastPowf; this is the exact pow.
            CHECK_THAT(ours[c * 64 + k], Catch::Matchers::WithinRel(kCujpegxlDct8[c][k], 2e-3f));
        }
    }
}

TEST_CASE("the natural order is a permutation starting at DC", "[codestream][tables]") {
    const auto order = dct8NaturalOrder();
    std::array<int, 64> seen{};
    for (uint32_t k : order) {
        REQUIRE(k < 64);
        ++seen[k];
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
    CHECK(order[0] == 0);
    // The first diagonal: memory index 1 then 8 (libjxl swaps on odd diagonals).
    CHECK(order[1] == 1);
    CHECK(order[2] == 8);
    CHECK(order[63] == 63);
}

TEST_CASE("quantizer parameters follow libjxl for a uniform field", "[codestream][tables]") {
    const QuantParams p = quantParamsForDistance(1.0f);
    CHECK(p.globalScale >= 1);
    CHECK(p.globalScale <= 32768);
    CHECK(p.quantDc >= 1);
    CHECK(p.quantField >= 1);
    CHECK(p.quantField <= 256);
    CHECK_THAT(p.globalScaleFloat * p.invGlobalScale, Catch::Matchers::WithinRel(1.0f, 1e-5f));
    // Smaller distance, finer quantisation.
    const QuantParams fine = quantParamsForDistance(0.5f);
    CHECK(fine.globalScale * fine.quantField >= p.globalScale * p.quantField);
}

TEST_CASE("frame geometry counts groups and sections", "[codestream]") {
    const FrameDim d = frameDim(257, 129);
    CHECK(d.xsizeBlocks == 33);
    CHECK(d.ysizeBlocks == 17);
    CHECK(d.xsizePadded == 264);
    CHECK(d.numGroups() == 2);
    CHECK(d.numDcGroups() == 1);
    CHECK(!d.singleSection());
    CHECK(d.numSections() == 5);
    const FrameDim s = frameDim(64, 36);
    CHECK(s.numGroups() == 1);
    CHECK(s.singleSection());
}

TEST_CASE("U32 and U64 fields encode like libjxl", "[codestream]") {
    BitWriter w;
    writeU64(w, 0);
    CHECK(w.bitsWritten() == 2);
    writeU64(w, 16);     // 2 + 4 bits
    writeU64(w, 272);    // 2 + 8 bits
    CHECK(w.bitsWritten() == 2 + 6 + 10);
    BitWriter t;
    writeU32(t, U32Enc{{Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408), BitsOffset(30, 4211712)}}, 1023);
    CHECK(t.bitsWritten() == 12);
    writeU32(t, U32Enc{{Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408), BitsOffset(30, 4211712)}}, 1024);
    CHECK(t.bitsWritten() == 12 + 16);
}
