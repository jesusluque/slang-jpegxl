// Copyright (c) 2026 jesus luque.
//
// Test helpers around sjxl::metrics and the oracle tools.
#pragma once

#include <filesystem>
#include <string>

#include "TestDevice.h"
#include "sjxl/image/Image.h"
#include "sjxl/metrics/Compare.h"

namespace sjxl::test {

using metrics::Metrics;

Result<Metrics> compare(const image::Rgb8& a, const image::Rgb8& b);

/// A Homebrew/PATH tool, or empty.
std::filesystem::path findTool(const std::string& name);

/// Runs a command line; true on exit status 0.
bool run(const std::string& command);

/// A scratch directory for this test binary.
std::filesystem::path scratchDir();

std::filesystem::path testData(const std::string& file);

}   // namespace sjxl::test
