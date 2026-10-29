// Copyright (c) 2026 jesus luque.
//
// The compute device: slang-rhi, one per process. Metal on Apple, CUDA then
// Vulkan on Linux. Slang is compiled at run time for whichever opened, and
// kernels are bound by name through reflection (ComputeKernel).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <slang-rhi.h>

#include "sjxl/core/Result.h"

namespace sjxl::gpu {

enum class Backend { Metal, CUDA, Vulkan, D3D12 };

[[nodiscard]] const char* toString(Backend) noexcept;

struct DeviceDesc {
    /// Empty means the platform's preference: Metal on Apple, CUDA then Vulkan
    /// on Linux, D3D12 then Vulkan on Windows. There is no CPU entry.
    std::vector<Backend> backends;
    /// slang-rhi's own validation layer. Slow; for tests and debugging.
    bool validation = false;
    /// Extra directories searched for `import`ed Slang modules, after the
    /// codec's own shader directory.
    std::vector<std::filesystem::path> shaderPaths;
    /// Compiled shaders kept on disk between runs. Empty: $SJXL_SHADER_CACHE,
    /// else sjxl/shaders in the platform's cache directory.
    std::filesystem::path shaderCache;
    bool                  useShaderCache = true;
};

struct ShaderCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t writes = 0;
};

/// What the device can do, asked once at creation.
struct Caps {
    std::string apiName;
    std::string adapterName;
    bool half = false;            ///< 16-bit floats in kernels
    bool unifiedMemory = false;
};

class Device {
public:
    [[nodiscard]] static Result<std::shared_ptr<Device>> create(const DeviceDesc& desc = {});

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    [[nodiscard]] Backend backend() const noexcept { return backend_; }
    [[nodiscard]] const Caps& caps() const noexcept { return caps_; }
    [[nodiscard]] ShaderCacheStats shaderCacheStats() const;

    [[nodiscard]] rhi::IDevice* rhi() const noexcept { return device_.get(); }
    [[nodiscard]] rhi::ICommandQueue* queue() const noexcept { return queue_.get(); }
    [[nodiscard]] slang::ISession* slangSession() const noexcept { return session_.get(); }
    /// Submits an empty command buffer without waiting: the queue's first use costs
    /// tens of milliseconds on Metal, which this lets overlap with the caller's CPU work.
    void warmUp();

    /// The directories Slang searches, in order, the codec's shaders first.
    [[nodiscard]] const std::vector<std::string>& shaderSearchPaths() const noexcept { return searchPaths_; }
    /// Where ShaderLibrary keeps precompiled Slang modules (empty: no cache).
    [[nodiscard]] const std::filesystem::path& moduleCacheDirectory() const noexcept { return moduleCacheDir_; }

    /// Blocks until everything submitted has run. A command buffer that
    /// failed meanwhile is not lost: the next CommandBatch::submit reports it
    /// (takeQueueError).
    void waitIdle();

    /// waitIdle, and what was freed actually given back: on Metal slang-rhi
    /// keeps a freed buffer resident until the next submit commits the
    /// residency set, so an empty batch is submitted and waited for.
    void releaseFreed();

    /// A slang-rhi result from the queue as a Result: SLANG_E_OUT_OF_MEMORY
    /// (with the patch in cmake/patches) is OutOfMemory, any other failure
    /// DeviceFailure.
    [[nodiscard]] static Result<void> queueResult(SlangResult result, std::string_view what);
    /// A failure waitIdle met and could not return, once.
    [[nodiscard]] Result<void> takeQueueError();

private:
    Device() = default;

    Backend                            backend_ = Backend::Metal;
    Caps                               caps_;
    std::vector<std::string>           searchPaths_;
    std::filesystem::path              moduleCacheDir_;
    rhi::ComPtr<rhi::IPersistentCache> shaderCache_;
    rhi::ComPtr<rhi::IDevice>          device_;
    rhi::ComPtr<rhi::ICommandQueue>    queue_;
    rhi::ComPtr<slang::ISession>       session_;
    std::mutex                         queueErrorGuard_;
    std::optional<Error>               queueError_;
};

/// Where the codec's own .slang files are: $SJXL_SHADER_DIR, then a `shaders`
/// directory holding `sjxl/` found from the executable or up to three
/// directories above it (<exe>/../shaders for a program in bin/), then the
/// build tree.
[[nodiscard]] std::filesystem::path shaderDirectory();

}   // namespace sjxl::gpu
