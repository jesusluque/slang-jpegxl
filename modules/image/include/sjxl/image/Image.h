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

/// Interleaved sRGB samples, 3 (RGB) or 4 (RGBA) channels, 8 or 16 bits (16-bit samples
/// are little-endian pairs of bytes), row-major, top-left origin. The name is historical.
struct Rgb8 {
    uint32_t             width = 0;
    uint32_t             height = 0;
    uint32_t             channels = 3;
    uint32_t             bitDepth = 8;   ///< 8 or 16
    std::vector<uint8_t> pixels;        ///< width * height * channels * (bitDepth / 8)

    [[nodiscard]] size_t bytes() const { return pixels.size(); }
    [[nodiscard]] uint32_t bytesPerSample() const { return bitDepth / 8; }
    [[nodiscard]] uint32_t sample(size_t index) const {   // sample `index` as an integer
        return bitDepth == 8 ? pixels[index] : static_cast<uint32_t>(pixels[2 * index]) | (static_cast<uint32_t>(pixels[2 * index + 1]) << 8);
    }
};

/// PNG, PPM/PGM, JPEG, BMP, TGA through stb_image, as 8-bit RGB; alpha is dropped, grey expanded.
[[nodiscard]] Result<Rgb8> readRgb8(const std::filesystem::path& path);
/// The same, keeping alpha when the file has it (4 channels), still 8-bit.
[[nodiscard]] Result<Rgb8> readImage8(const std::filesystem::path& path);
/// Keeping alpha and the file's depth (16-bit PNG/PPM give 16-bit samples).
[[nodiscard]] Result<Rgb8> readImage(const std::filesystem::path& path);
/// 8-bit through stb_image_write; 16-bit with the module's own PNG writer (stored deflate).
[[nodiscard]] Result<void> writePng(const std::filesystem::path& path, const Rgb8& image);

[[nodiscard]] Result<std::vector<uint8_t>> readFile(const std::filesystem::path& path);
[[nodiscard]] Result<void> writeFile(const std::filesystem::path& path, std::span<const uint8_t> bytes);

}   // namespace sjxl::image
