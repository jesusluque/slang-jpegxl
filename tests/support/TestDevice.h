// Copyright (c) 2026 jesus luque.
//
// One device and shader library for a test binary. A test that needs the GPU
// calls `testDevice()`; when no device can be opened the case SKIPs with the
// reason, and never runs anything on the CPU instead.
#pragma once

#include <memory>

#include "sjxl/gpu/Device.h"
#include "sjxl/gpu/ShaderLibrary.h"

namespace sjxl::test {

struct Gpu {
    std::shared_ptr<gpu::Device>        device;
    std::shared_ptr<gpu::ShaderLibrary> library;
};

/// The process's device, opened on first use. SKIPs the current Catch2 case
/// when there is none.
Gpu& testDevice();

}   // namespace sjxl::test
