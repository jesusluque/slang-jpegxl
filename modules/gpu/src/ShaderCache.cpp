// Copyright (c) 2026 jesus luque.
#include "ShaderCache.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace sjxl::gpu {
namespace {

/// FNV-1a over the key's bytes: a file name, not a security boundary -- the
/// file carries the whole key and a mismatch is a miss.
uint64_t hashOf(const void* data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h = (h ^ bytes[i]) * 1099511628211ULL;
    }
    return h;
}

}   // namespace

DiskShaderCache::DiskShaderCache(std::filesystem::path directory) : directory_(std::move(directory)) {
    std::error_code ignored;
    std::filesystem::create_directories(directory_, ignored);
}

std::filesystem::path DiskShaderCache::pathFor(ISlangBlob* key) const {
    char name[32];
    std::snprintf(name, sizeof name, "%016llx.bin",
                  static_cast<unsigned long long>(hashOf(key->getBufferPointer(), key->getBufferSize())));
    return directory_ / name;
}

rhi::Result DiskShaderCache::writeCache(ISlangBlob* key, ISlangBlob* data) {
    const std::filesystem::path path = pathFor(key);
    std::filesystem::path partial = path;
    partial += ".partial." + std::to_string(writes_.fetch_add(1));
    {
        std::ofstream out(partial, std::ios::binary | std::ios::trunc);
        if (!out) {
            return SLANG_FAIL;
        }
        const uint64_t keyBytes = key->getBufferSize();
        out.write(reinterpret_cast<const char*>(&keyBytes), sizeof keyBytes);
        out.write(static_cast<const char*>(key->getBufferPointer()), static_cast<std::streamsize>(keyBytes));
        out.write(static_cast<const char*>(data->getBufferPointer()), static_cast<std::streamsize>(data->getBufferSize()));
        if (!out) {
            return SLANG_FAIL;
        }
    }
    std::error_code ec;
    std::filesystem::rename(partial, path, ec);
    return ec ? SLANG_FAIL : SLANG_OK;
}

rhi::Result DiskShaderCache::queryCache(ISlangBlob* key, ISlangBlob** outData) {
    *outData = nullptr;
    std::ifstream in(pathFor(key), std::ios::binary);
    if (!in) {
        misses_.fetch_add(1);
        return SLANG_E_NOT_FOUND;
    }
    std::vector<char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    uint64_t keyBytes = 0;
    if (file.size() < sizeof keyBytes) {
        misses_.fetch_add(1);
        return SLANG_E_NOT_FOUND;
    }
    std::memcpy(&keyBytes, file.data(), sizeof keyBytes);
    if (keyBytes != key->getBufferSize() || file.size() < sizeof keyBytes + keyBytes ||
        std::memcmp(file.data() + sizeof keyBytes, key->getBufferPointer(), keyBytes) != 0) {
        misses_.fetch_add(1);
        return SLANG_E_NOT_FOUND;
    }
    const size_t at = sizeof keyBytes + keyBytes;
    if (SLANG_FAILED(rhi::getRHI()->createBlob(file.data() + at, file.size() - at, outData))) {
        misses_.fetch_add(1);
        return SLANG_E_NOT_FOUND;
    }
    hits_.fetch_add(1);
    return SLANG_OK;
}

rhi::Result DiskShaderCache::queryInterface(const SlangUUID& uuid, void** outObject) {
    if (uuid == rhi::IPersistentCache::getTypeGuid() || uuid == ISlangUnknown::getTypeGuid()) {
        addRef();
        *outObject = static_cast<rhi::IPersistentCache*>(this);
        return SLANG_OK;
    }
    return SLANG_E_NO_INTERFACE;
}

uint32_t DiskShaderCache::addRef() { return references_.fetch_add(1) + 1; }

uint32_t DiskShaderCache::release() {
    const uint32_t left = references_.fetch_sub(1) - 1;
    if (left == 0) {
        delete this;
    }
    return left;
}

ShaderCacheStats DiskShaderCache::stats() const noexcept { return {hits_.load(), misses_.load(), writes_.load()}; }

}   // namespace sjxl::gpu
