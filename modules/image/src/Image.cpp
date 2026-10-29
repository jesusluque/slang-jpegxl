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

namespace {

Result<Rgb8> readWithChannels(const std::filesystem::path& path, bool keepAlpha) {
    int w = 0, h = 0, channels = 0;
    if (!stbi_info(path.string().c_str(), &w, &h, &channels)) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
    }
    const int want = (keepAlpha && (channels == 2 || channels == 4)) ? 4 : 3;
    stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &channels, want);
    if (data == nullptr) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
    }
    Rgb8 image;
    image.width = static_cast<uint32_t>(w);
    image.height = static_cast<uint32_t>(h);
    image.channels = static_cast<uint32_t>(want);
    image.pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(want));
    stbi_image_free(data);
    return image;
}

}   // namespace

Result<Rgb8> readRgb8(const std::filesystem::path& path) { return readWithChannels(path, false); }
Result<Rgb8> readImage8(const std::filesystem::path& path) { return readWithChannels(path, true); }

Result<void> writePng(const std::filesystem::path& path, const Rgb8& image) {
    if ((image.channels != 3 && image.channels != 4) || image.pixels.size() != size_t{image.width} * image.height * image.channels) {
        return Error(ErrorCode::InvalidArgument, "image pixel count does not match its size");
    }
    if (stbi_write_png(path.string().c_str(), static_cast<int>(image.width), static_cast<int>(image.height), static_cast<int>(image.channels),
                       image.pixels.data(), static_cast<int>(image.width * image.channels)) == 0) {
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
