// Copyright (c) 2026 jesus luque.
#include "TestDevice.h"

#include <catch2/catch_test_macros.hpp>

namespace sjxl::test {

Gpu& testDevice() {
    static Gpu gpu;
    static std::string failure;
    if (gpu.device == nullptr && failure.empty()) {
        auto made = gpu::Device::create();
        if (!made) {
            failure = made.error().toString();
        } else {
            gpu.device = *made;
            gpu.library = std::make_shared<gpu::ShaderLibrary>(gpu.device);
        }
    }
    if (gpu.device == nullptr) {
        SKIP("no GPU device: " << failure);
    }
    return gpu;
}

}   // namespace sjxl::test
