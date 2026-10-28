// Copyright (c) 2026 jesus luque.
#include "sjxl/image/Image.h"

#include <cstdio>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PIC
#define STBI_NO_PSD
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace sjxl::image {

Result<Rgb8> readRgb8(const std::filesystem::path& path) {
    int w = 0, h = 0, channels = 0;
    stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &channels, 3);
    if (data == nullptr) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
    }
    Rgb8 image;
    image.width = static_cast<uint32_t>(w);
    image.height = static_cast<uint32_t>(h);
    image.pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(data);
    return image;
}

Result<void> writePng(const std::filesystem::path& path, const Rgb8& image) {
    if (image.pixels.size() != size_t{image.width} * image.height * 3) {
        return Error(ErrorCode::InvalidArgument, "image pixel count does not match its size");
    }
    if (stbi_write_png(path.string().c_str(), static_cast<int>(image.width), static_cast<int>(image.height), 3,
                       image.pixels.data(), static_cast<int>(image.width * 3)) == 0) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", path.string());
    }
    return ok();
}

Result<std::vector<uint8_t>> readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return bytes;
}

Result<void> writeFile(const std::filesystem::path& path, std::span<const uint8_t> bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "cannot create '{}'", path.string());
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "cannot write '{}'", path.string());
    }
    return ok();
}

}   // namespace sjxl::image
