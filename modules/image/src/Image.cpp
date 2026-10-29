// Copyright (c) 2026 jesus luque.
#include "sjxl/image/Image.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <fstream>
#include <span>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_PIC
#define STBI_NO_PSD
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace sjxl::image {

Result<void> writeFile(const std::filesystem::path& path, std::span<const uint8_t> bytes);

namespace {

uint32_t be32(const uint8_t* p) { return (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) | (uint32_t{p[2]} << 8) | p[3]; }

/// The colour chunks of a PNG before its image data (libjxl's precedence: cICP,
/// iCCP, sRGB, then gAMA with cHRM). Only a gAMA transfer over sRGB primaries is
/// carried; an ICC profile or other primaries fall back to sRGB with a note.
void scanPngColor(const std::filesystem::path& path, Rgb8& image) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> head(1 << 16);
    in.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
    const size_t n = static_cast<size_t>(in.gcount());
    static constexpr uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (n < 8 || !std::equal(kSig, kSig + 8, head.data())) return;
    float gamma = 0.0f;
    bool srgb = false, icc = false, cicp = false, chrmOk = true;
    for (size_t pos = 8; pos + 12 <= n;) {
        const uint32_t len = be32(&head[pos]);
        const std::string type(reinterpret_cast<const char*>(&head[pos + 4]), 4);
        if (type == "IDAT" || type == "IEND" || pos + 12 + len > n) break;
        const uint8_t* data = &head[pos + 8];
        if (type == "gAMA" && len == 4) gamma = static_cast<float>(be32(data)) / 100000.0f;
        if (type == "sRGB") srgb = true;
        if (type == "iCCP") icc = true;
        if (type == "cICP") cicp = true;
        if (type == "cHRM" && len == 32) {
            // sRGB primaries and D65 within the chunk's precision.
            static constexpr uint32_t kSrgb[8] = {31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000};
            for (int i = 0; i < 8; ++i) {
                const int32_t v = static_cast<int32_t>(be32(data + 4 * i));
                if (std::abs(v - static_cast<int32_t>(kSrgb[i])) > 100) chrmOk = false;
            }
        }
        pos += 12 + len;
    }
    if (cicp || icc || srgb) {
        if (icc || cicp) std::fprintf(stderr, "sjxl: %s: an ICC profile or cICP chunk is not honoured; treating the samples as sRGB\n", path.string().c_str());
        return;
    }
    if (!chrmOk) {
        std::fprintf(stderr, "sjxl: %s: cHRM primaries differ from sRGB; treating them as sRGB\n", path.string().c_str());
    }
    if (gamma > 0.0f && std::abs(gamma - 1.0f / 2.2f) < 1e-6f + 0.002f) {
        image.gamma = gamma;   // a plain gamma 2.2 transfer, not the sRGB curve, as libjxl reads it
    } else if (gamma > 0.0f) {
        image.gamma = gamma;
    }
}

Result<Rgb8> readWithChannels(const std::filesystem::path& path, bool keepAlpha, bool keepDepth) {
    int w = 0, h = 0, channels = 0;
    if (!stbi_info(path.string().c_str(), &w, &h, &channels)) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
    }
    const int want = (keepAlpha && (channels == 2 || channels == 4)) ? 4 : 3;
    const bool sixteen = keepDepth && stbi_is_16_bit(path.string().c_str());
    Rgb8 image;
    image.width = static_cast<uint32_t>(w);
    image.height = static_cast<uint32_t>(h);
    image.channels = static_cast<uint32_t>(want);
    scanPngColor(path, image);
    const size_t samples = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(want);
    if (sixteen) {
        stbi_us* data = stbi_load_16(path.string().c_str(), &w, &h, &channels, want);
        if (data == nullptr) return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
        image.bitDepth = 16;
        image.pixels.resize(samples * 2);
        for (size_t i = 0; i < samples; ++i) {
            image.pixels[2 * i] = static_cast<uint8_t>(data[i] & 0xFF);
            image.pixels[2 * i + 1] = static_cast<uint8_t>(data[i] >> 8);
        }
        stbi_image_free(data);
        return image;
    }
    stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &channels, want);
    if (data == nullptr) {
        return Error::make(ErrorCode::IoFailure, "cannot read image '{}': {}", path.string(), stbi_failure_reason());
    }
    image.pixels.assign(data, data + samples);
    stbi_image_free(data);
    return image;
}

uint32_t crc32Of(std::span<const uint8_t> bytes, uint32_t crc = 0xFFFFFFFFu) {
    for (uint8_t b : bytes) {
        crc ^= b;
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

void putBe32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

void putChunk(std::vector<uint8_t>& out, const char* type, std::span<const uint8_t> data) {
    putBe32(out, static_cast<uint32_t>(data.size()));
    std::vector<uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    putBe32(out, crc32Of(body) ^ 0xFFFFFFFFu);
}

/// A 16-bit PNG (stb_image_write only does 8): filter 0 rows, big-endian samples,
/// zlib with stored deflate blocks.
Result<void> writePng16(const std::filesystem::path& path, const Rgb8& image) {
    const size_t rowBytes = size_t{image.width} * image.channels * 2;
    std::vector<uint8_t> raw;
    raw.reserve((rowBytes + 1) * image.height);
    for (uint32_t y = 0; y < image.height; ++y) {
        raw.push_back(0);   // filter: none
        const uint8_t* row = image.pixels.data() + y * rowBytes;
        for (size_t i = 0; i < rowBytes; i += 2) {   // little-endian in memory, big-endian in PNG
            raw.push_back(row[i + 1]);
            raw.push_back(row[i]);
        }
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t pos = 0; pos < raw.size() || pos == 0;) {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        const bool last = pos + n >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>(n >> 8));
        z.push_back(static_cast<uint8_t>(~n & 0xFF));
        z.push_back(static_cast<uint8_t>((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
        if (last) break;
    }
    putBe32(z, (b << 16) | a);
    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    putBe32(ihdr, image.width);
    putBe32(ihdr, image.height);
    ihdr.push_back(16);
    ihdr.push_back(image.channels == 4 ? 6 : 2);
    ihdr.push_back(0); ihdr.push_back(0); ihdr.push_back(0);
    putChunk(png, "IHDR", ihdr);
    putChunk(png, "IDAT", z);
    putChunk(png, "IEND", {});
    return writeFile(path, png);
}

}   // namespace

Result<Rgb8> readRgb8(const std::filesystem::path& path) { return readWithChannels(path, false, false); }
Result<Rgb8> readImage8(const std::filesystem::path& path) { return readWithChannels(path, true, false); }
Result<Rgb8> readImage(const std::filesystem::path& path) { return readWithChannels(path, true, true); }

Result<void> writePng(const std::filesystem::path& path, const Rgb8& image) {
    if ((image.channels != 3 && image.channels != 4) || (image.bitDepth != 8 && image.bitDepth != 16) ||
        image.pixels.size() != size_t{image.width} * image.height * image.channels * (image.bitDepth / 8)) {
        return Error(ErrorCode::InvalidArgument, "image pixel count does not match its size");
    }
    if (image.bitDepth == 16) return writePng16(path, image);
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
