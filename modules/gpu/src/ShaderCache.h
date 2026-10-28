// Copyright (c) 2026 jesus luque.
//
// slang-rhi's persistent shader cache, on disk: one file per compiled
// program, named by a hash of its key, holding the key (to tell hash
// collisions apart) and the data.
#pragma once

#include <atomic>
#include <filesystem>

#include <slang-rhi.h>

#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {

class DiskShaderCache final : public rhi::IPersistentCache {
public:
    explicit DiskShaderCache(std::filesystem::path directory);

    SLANG_NO_THROW rhi::Result SLANG_MCALL writeCache(ISlangBlob* key, ISlangBlob* data) override;
    SLANG_NO_THROW rhi::Result SLANG_MCALL queryCache(ISlangBlob* key, ISlangBlob** outData) override;
    SLANG_NO_THROW rhi::Result SLANG_MCALL queryInterface(const SlangUUID& uuid, void** outObject) override;
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override;
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override;

    [[nodiscard]] ShaderCacheStats stats() const noexcept;

private:
    [[nodiscard]] std::filesystem::path pathFor(ISlangBlob* key) const;

    std::filesystem::path directory_;
    std::atomic<uint32_t> references_{1};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> writes_{0};
};

}   // namespace sjxl::gpu
