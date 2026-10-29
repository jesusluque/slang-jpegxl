// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/Device.h"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ShaderCache.h"
#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"

namespace sjxl::gpu {
namespace {

rhi::DeviceType toRhi(Backend backend) {
    switch (backend) {
    case Backend::Metal: return rhi::DeviceType::Metal;
    case Backend::CUDA: return rhi::DeviceType::CUDA;
    case Backend::Vulkan: return rhi::DeviceType::Vulkan;
    case Backend::D3D12: return rhi::DeviceType::D3D12;
    }
    return rhi::DeviceType::Default;
}

std::vector<Backend> platformPreference() {
#if defined(__APPLE__)
    return {Backend::Metal};
#elif defined(_WIN32)
    return {Backend::CUDA, Backend::D3D12, Backend::Vulkan};
#else
    return {Backend::CUDA, Backend::Vulkan};
#endif
}

/// Messages from slang-rhi's validation and the drivers, into the log.
class DebugLog final : public rhi::IDebugCallback {
public:
    std::string lastError;

    SLANG_NO_THROW void SLANG_MCALL handleMessage(rhi::DebugMessageType type, rhi::DebugMessageSource,
                                                  const char* message) override {
        switch (type) {
        case rhi::DebugMessageType::Error:
            log::error("rhi: {}", message);
            lastError = message;
            break;
        case rhi::DebugMessageType::Warning:
            // The Metal compiler's "unused variable" notes come from results Slang stores and
            // the kernels ignore (atomics' previous value, counts); they are noise at warn level.
            if (std::string_view(message).find("-Wunused-variable") != std::string_view::npos) log::debug("rhi: {}", message);
            else log::warn("rhi: {}", message);
            break;
        default: log::debug("rhi: {}", message); break;
        }
    }
};

DebugLog gDebugLog;

std::filesystem::path findShaderDirectory(const std::filesystem::path& start) {
    constexpr int kLevelsUp = 3;
    std::error_code ignored;
    if (start.empty()) {
        return {};
    }
    std::filesystem::path at = start;
    for (int level = 0; level <= kLevelsUp; ++level) {
        if (auto candidate = at / "shaders"; std::filesystem::is_directory(candidate / "sjxl", ignored)) {
            return std::filesystem::weakly_canonical(candidate, ignored);
        }
        if (!at.has_parent_path() || at.parent_path() == at) {
            break;
        }
        at = at.parent_path();
    }
    return {};
}

}   // namespace

const char* toString(Backend backend) noexcept {
    switch (backend) {
    case Backend::Metal: return "Metal";
    case Backend::CUDA: return "CUDA";
    case Backend::Vulkan: return "Vulkan";
    case Backend::D3D12: return "D3D12";
    }
    return "?";
}

std::filesystem::path shaderDirectory() {
    if (std::string fromEnv = platform::env("SJXL_SHADER_DIR"); !fromEnv.empty()) {
        return fromEnv;
    }
    if (auto found = findShaderDirectory(platform::executableDir()); !found.empty()) {
        return found;
    }
    return SJXL_BUILD_SHADER_DIR;
}

Result<std::shared_ptr<Device>> Device::create(const DeviceDesc& desc) {
    std::vector<Backend> order = desc.backends.empty() ? platformPreference() : desc.backends;
    // SJXL_BACKEND=metal|cuda|vulkan|d3d12 (a comma-separated order) chooses
    // for a whole run what a caller left to the platform.
    if (desc.backends.empty()) {
        const std::string fromEnv = platform::env("SJXL_BACKEND");
        std::vector<Backend> named;
        size_t start = 0;
        while (!fromEnv.empty() && start <= fromEnv.size()) {
            const size_t comma = fromEnv.find(',', start);
            const std::string word = fromEnv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (word == "metal") named.push_back(Backend::Metal);
            else if (word == "cuda") named.push_back(Backend::CUDA);
            else if (word == "vulkan") named.push_back(Backend::Vulkan);
            else if (word == "d3d12") named.push_back(Backend::D3D12);
            else if (!word.empty()) log::warn("SJXL_BACKEND: '{}' is not a backend (metal, cuda, vulkan, d3d12)", word);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (!named.empty()) {
            order = named;
        }
    }

    auto device = std::shared_ptr<Device>(new Device());
    device->searchPaths_.push_back(shaderDirectory().string());
    for (const auto& extra : desc.shaderPaths) {
        device->searchPaths_.push_back(extra.string());
    }
    std::vector<const char*> searchPaths;
    for (const std::string& path : device->searchPaths_) {
        searchPaths.push_back(path.c_str());
    }

    // SJXL_GPU_VALIDATION: the API's own validation and debug layers. Slow;
    // for finding a failure, not for running.
    const bool validate = desc.validation || !platform::env("SJXL_GPU_VALIDATION").empty();
    if (validate) {
        rhi::getRHI()->enableDebugLayers();
    }
    if (desc.useShaderCache) {
        std::filesystem::path directory = desc.shaderCache;
        if (directory.empty()) {
            const std::string fromEnv = platform::env("SJXL_SHADER_CACHE");
            directory = !fromEnv.empty() ? std::filesystem::path(fromEnv) : platform::cacheDirectory() / "sjxl" / "shaders";
        }
        directory /= "gen1";
        device->shaderCache_.attach(new DiskShaderCache(directory));
    }

    std::string failures;
    for (Backend backend : order) {
        if (!rhi::getRHI()->isDeviceTypeSupported(toRhi(backend))) {
            failures += std::format(" {} (not built in)", toString(backend));
            continue;
        }
        rhi::DeviceDesc rhiDesc;
        rhiDesc.deviceType = toRhi(backend);
        rhiDesc.enableValidation = validate;
        rhiDesc.debugCallback = &gDebugLog;
        rhiDesc.slang.searchPaths = searchPaths.data();
        rhiDesc.slang.searchPathCount = static_cast<uint32_t>(searchPaths.size());
        rhiDesc.slang.optimizationLevel = SLANG_OPTIMIZATION_LEVEL_HIGH;
        rhiDesc.persistentShaderCache = device->shaderCache_.get();

        rhi::ComPtr<rhi::IDevice> made;
        gDebugLog.lastError.clear();
        if (SLANG_FAILED(rhi::getRHI()->createDevice(rhiDesc, made.writeRef())) || made == nullptr) {
            failures += gDebugLog.lastError.empty() ? std::format(" {} (no device)", toString(backend))
                                                    : std::format(" {} ({})", toString(backend), gDebugLog.lastError);
            continue;
        }
        device->backend_ = backend;
        device->device_ = made;
        break;
    }
    if (device->device_ == nullptr) {
        return Error::make(ErrorCode::DeviceFailure, "no GPU device could be opened:{}", failures);
    }

    rhi::IDevice* rhiDevice = device->device_.get();
    if (SLANG_FAILED(rhiDevice->getQueue(rhi::QueueType::Graphics, device->queue_.writeRef()))) {
        return Error(ErrorCode::DeviceFailure, "the device has no queue");
    }
    if (SLANG_FAILED(rhiDevice->getSlangSession(device->session_.writeRef()))) {
        return Error(ErrorCode::ShaderFailure, "the device has no Slang session");
    }

    const rhi::DeviceInfo& info = rhiDevice->getInfo();
    Caps& caps = device->caps_;
    caps.apiName = info.apiName != nullptr ? info.apiName : "";
    caps.adapterName = info.adapterName != nullptr ? info.adapterName : "";
    caps.half = rhiDevice->hasFeature(rhi::Feature::Half);
    caps.unifiedMemory = device->backend_ == Backend::Metal;
    log::info("GPU: {} on {}", caps.apiName, caps.adapterName);
    return device;
}

void Device::releaseFreed() {
    waitIdle();
    if (queue_ != nullptr) {
        if (auto submitted = queueResult(queue_->submit(rhi::SubmitDesc{}), "the work submitted"); !submitted) {
            const std::lock_guard<std::mutex> held(queueErrorGuard_);
            if (!queueError_.has_value()) {
                queueError_ = std::move(submitted).error();
            }
        }
    }
    waitIdle();
}

Result<void> Device::queueResult(SlangResult result, std::string_view what) {
    if (SLANG_SUCCEEDED(result)) {
        return ok();
    }
    if (result == SLANG_E_OUT_OF_MEMORY) {
        return Error::make(ErrorCode::OutOfMemory, "the GPU ran out of memory running {}", what);
    }
    return Error::make(ErrorCode::DeviceFailure, "the GPU failed running {}", what);
}

Result<void> Device::takeQueueError() {
    const std::lock_guard<std::mutex> held(queueErrorGuard_);
    if (!queueError_.has_value()) {
        return ok();
    }
    Error error = std::move(*queueError_);
    queueError_.reset();
    return error;
}

Device::~Device() {
    if (queue_ != nullptr) {
        queue_->waitOnHost();
    }
}

ShaderCacheStats Device::shaderCacheStats() const {
    return shaderCache_ != nullptr ? static_cast<const DiskShaderCache*>(shaderCache_.get())->stats() : ShaderCacheStats{};
}

void Device::waitIdle() {
    if (queue_ != nullptr) {
        if (auto waited = queueResult(queue_->waitOnHost(), "the work submitted"); !waited) {
            const std::lock_guard<std::mutex> held(queueErrorGuard_);
            if (!queueError_.has_value()) {
                queueError_ = std::move(waited).error();
            }
        }
    }
}

}   // namespace sjxl::gpu
