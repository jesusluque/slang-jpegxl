// Copyright (c) 2026 jesus luque.
//
// Image files in and out. The CPU reads the bytes and hands them to the device
// as they are; it does not convert, scale or compare pixels.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "sjxl/core/Result.h"

namespace sjxl::image {

/// 8-bit sRGB, interleaved, 3 channels, row-major, top-left origin.
struct Rgb8 {
    uint32_t             width = 0;
    uint32_t             height = 0;
    std::vector<uint8_t> pixels;   ///< width * height * 3

    [[nodiscard]] size_t bytes() const { return pixels.size(); }
};

/// PNG, PPM/PGM, JPEG, BMP, TGA through stb_image; alpha is dropped, grey expanded.
[[nodiscard]] Result<Rgb8> readRgb8(const std::filesystem::path& path);
[[nodiscard]] Result<void> writePng(const std::filesystem::path& path, const Rgb8& image);

[[nodiscard]] Result<std::vector<uint8_t>> readFile(const std::filesystem::path& path);
[[nodiscard]] Result<void> writeFile(const std::filesystem::path& path, std::span<const uint8_t> bytes);

}   // namespace sjxl::image
