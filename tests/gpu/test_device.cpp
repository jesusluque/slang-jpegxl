// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>

#include "TestDevice.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/ComputeKernel.h"

using namespace sjxl;

TEST_CASE("device opens and names its adapter", "[gpu]") {
    auto& gpu = test::testDevice();
    CHECK(!gpu.device->caps().apiName.empty());
    CHECK(!gpu.device->caps().adapterName.empty());
}

TEST_CASE("a kernel fills a buffer and a kernel checks it", "[gpu]") {
    auto& gpu = test::testDevice();
    auto fill = gpu::ComputeKernel::create(*gpu.library, "sjxl/test/fill", "fill");
    REQUIRE(fill);
    auto count = gpu::ComputeKernel::create(*gpu.library, "sjxl/test/fill", "countMismatches");
    REQUIRE(count);

    constexpr uint32_t kCount = 100000;
    auto a = gpu::Buffer::ofType<uint32_t>(*gpu.device, kCount, "a");
    auto b = gpu::Buffer::ofType<uint32_t>(*gpu.device, kCount, "b");
    auto mismatches = gpu::Buffer::ofType<uint32_t>(*gpu.device, 1, "mismatches");
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(mismatches);

    gpu::CommandBatch batch(*gpu.device);
    for (const auto* target : {&*a, &*b}) {
        fill->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["output"].setBinding(target->rhi());
            cursor["params"]["count"].setData(kCount);
            cursor["params"]["multiplier"].setData(uint32_t{3});
            cursor["params"]["offset"].setData(uint32_t{7});
        });
    }
    count->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a->rhi());
        cursor["b"].setBinding(b->rhi());
        cursor["mismatches"].setBinding(mismatches->rhi());
        cursor["params"]["count"].setData(kCount);
    });
    REQUIRE(batch.submit(true));
    auto read = mismatches->readAll<uint32_t>(*gpu.device);
    REQUIRE(read);
    CHECK((*read)[0] == 0);

    // A different pattern in b: every element differs (offset moves them all).
    fill->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["output"].setBinding(b->rhi());
        cursor["params"]["count"].setData(kCount);
        cursor["params"]["multiplier"].setData(uint32_t{3});
        cursor["params"]["offset"].setData(uint32_t{8});
    });
    batch.clear(*mismatches);
    count->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a->rhi());
        cursor["b"].setBinding(b->rhi());
        cursor["mismatches"].setBinding(mismatches->rhi());
        cursor["params"]["count"].setData(kCount);
    });
    REQUIRE(batch.submit(true));
    read = mismatches->readAll<uint32_t>(*gpu.device);
    REQUIRE(read);
    CHECK((*read)[0] == kCount);
}
