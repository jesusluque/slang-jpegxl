// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "TestDevice.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/algo/PrefixSum.h"

using namespace sjxl;

TEST_CASE("prefix sum matches the closed form", "[gpu][algo]") {
    auto& gpu = test::testDevice();
    const uint32_t count = GENERATE(1u, 63u, 64u, 1000u, 4097u, 300000u);
    auto fill = gpu::ComputeKernel::create(*gpu.library, "sjxl/test/fill", "fill");
    auto expected = gpu::ComputeKernel::create(*gpu.library, "sjxl/test/fill", "expectedPrefix");
    auto mismatch = gpu::ComputeKernel::create(*gpu.library, "sjxl/test/fill", "countMismatches");
    auto scan = gpu::PrefixSum::create(*gpu.library);
    REQUIRE(fill);
    REQUIRE(expected);
    REQUIRE(mismatch);
    REQUIRE(scan);

    auto input = gpu::Buffer::ofType<uint32_t>(*gpu.device, count, "input");
    auto output = gpu::Buffer::ofType<uint32_t>(*gpu.device, count, "output");
    auto want = gpu::Buffer::ofType<uint32_t>(*gpu.device, count, "want");
    auto total = gpu::Buffer::ofType<uint32_t>(*gpu.device, 1, "total");
    auto mismatches = gpu::Buffer::ofType<uint32_t>(*gpu.device, 1, "mismatches");
    REQUIRE(input);
    REQUIRE(output);
    REQUIRE(want);
    REQUIRE(total);
    REQUIRE(mismatches);

    const uint32_t multiplier = 5;
    const uint32_t offset = 2;
    gpu::CommandBatch batch(*gpu.device);
    fill->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["output"].setBinding(input->rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["multiplier"].setData(multiplier);
        cursor["params"]["offset"].setData(offset);
    });
    expected->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["expected"].setBinding(want->rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["multiplier"].setData(multiplier);
        cursor["params"]["offset"].setData(offset);
    });
    REQUIRE(scan->apply(batch, *input, *output, *total, count));
    mismatch->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(output->rhi());
        cursor["b"].setBinding(want->rhi());
        cursor["mismatches"].setBinding(mismatches->rhi());
        cursor["params"]["count"].setData(count);
    });
    REQUIRE(batch.submit(true));

    auto bad = mismatches->readAll<uint32_t>(*gpu.device);
    REQUIRE(bad);
    CHECK((*bad)[0] == 0);
    auto sum = total->readAll<uint32_t>(*gpu.device);
    REQUIRE(sum);
    // The total is the closed form at i = count.
    const uint64_t n = count;
    const uint32_t wantTotal = static_cast<uint32_t>((n * (n - 1) / 2) * multiplier + offset * n);
    CHECK((*sum)[0] == wantTotal);
}
