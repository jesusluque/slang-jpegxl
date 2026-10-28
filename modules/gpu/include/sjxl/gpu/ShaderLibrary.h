// Copyright (c) 2026 jesus luque.
//
// Slang programs, compiled once per (module, entry points) and kept.
#pragma once

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "sjxl/core/Result.h"

namespace sjxl::gpu {

class Device;

/// A linked program and what reflection says about it.
struct Program {
    rhi::ComPtr<rhi::IShaderProgram>   program;
    rhi::ComPtr<slang::IComponentType> linked;
    /// [numthreads] of the first (compute) entry point. Read, not assumed:
    /// dispatch sizes are threads divided by this, rounded up.
    std::array<uint32_t, 3>            threadGroup{1, 1, 1};
};

/// A link-time constant: a module declares `extern static const uint kName;`
/// and the program is linked with `export static const uint kName = value;`.
struct LinkConstant {
    std::string type;    ///< "uint", "int", "float", "bool"
    std::string name;
    std::string value;   ///< as Slang source: "16", "1.5", "true"
};

class ShaderLibrary {
public:
    explicit ShaderLibrary(std::shared_ptr<Device> device);

    /// `module` is a Slang module name as `import` would write it
    /// ("sjxl/algo/prefix_local"); `entries` its entry points.
    [[nodiscard]] Result<std::shared_ptr<const Program>> load(const std::string& module,
                                                              const std::vector<std::string>& entries,
                                                              const std::vector<LinkConstant>& constants = {});

    [[nodiscard]] Device& device() noexcept { return *device_; }
    [[nodiscard]] std::shared_ptr<Device> deviceShared() const noexcept { return device_; }

private:
    [[nodiscard]] Result<std::shared_ptr<const Program>> link(const std::string& key, slang::IModule* module,
                                                              const std::string& name,
                                                              const std::vector<std::string>& entries,
                                                              const std::vector<LinkConstant>& constants);

    std::shared_ptr<Device>                               device_;
    std::mutex                                            guard_;
    std::map<std::string, std::shared_ptr<const Program>> programs_;
};

}   // namespace sjxl::gpu
