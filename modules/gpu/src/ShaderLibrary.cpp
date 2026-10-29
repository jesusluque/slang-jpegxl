// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/ShaderLibrary.h"

#include <functional>
#include <string_view>

#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {
namespace {

std::string diagnostics(slang::IBlob* blob) {
    if (blob == nullptr || blob->getBufferSize() == 0) {
        return {};
    }
    return std::string(static_cast<const char*>(blob->getBufferPointer()), blob->getBufferSize());
}

std::string keyOf(const std::string& module, const std::vector<std::string>& entries,
                  const std::vector<LinkConstant>& constants) {
    std::string key = module;
    for (const std::string& entry : entries) {
        key += "|" + entry;
    }
    for (const LinkConstant& c : constants) {
        key += "|" + c.type + " " + c.name + "=" + c.value;
    }
    return key;
}

}   // namespace

ShaderLibrary::ShaderLibrary(std::shared_ptr<Device> device) : device_(std::move(device)) {}

Result<std::shared_ptr<const Program>> ShaderLibrary::load(const std::string& module,
                                                            const std::vector<std::string>& entries,
                                                            const std::vector<LinkConstant>& constants) {
    const std::string key = keyOf(module, entries, constants);
    const std::lock_guard<std::mutex> held(guard_);
    if (auto found = programs_.find(key); found != programs_.end()) {
        return found->second;
    }
    rhi::ComPtr<slang::IBlob> diag;
    slang::IModule* loaded = device_->slangSession()->loadModule(module.c_str(), diag.writeRef());
    if (loaded == nullptr) {
        return Error::make(ErrorCode::ShaderFailure, "cannot load Slang module '{}':\n{}", module,
                           diagnostics(diag.get()));
    }
    return link(key, loaded, module, entries, constants);
}

Result<std::shared_ptr<const Program>> ShaderLibrary::link(const std::string& key, slang::IModule* loaded,
                                                            const std::string& module,
                                                            const std::vector<std::string>& entries,
                                                            const std::vector<LinkConstant>& constants) {
    slang::ISession* session = device_->slangSession();
    rhi::ComPtr<slang::IBlob> diag;
    std::vector<rhi::ComPtr<slang::IEntryPoint>> entryPoints;
    std::vector<slang::IComponentType*> components{loaded};
    for (const std::string& entry : entries) {
        rhi::ComPtr<slang::IEntryPoint> found;
        if (SLANG_FAILED(loaded->findEntryPointByName(entry.c_str(), found.writeRef()))) {
            return Error::make(ErrorCode::ShaderFailure, "no entry point '{}' in '{}'", entry, module);
        }
        components.push_back(found.get());
        entryPoints.push_back(found);
    }
    if (!constants.empty()) {
        std::string source;
        for (const LinkConstant& c : constants) {
            source += "export static const " + c.type + " " + c.name + " = " + c.value + ";\n";
        }
        const std::string name = "sjxl_constants_" + std::to_string(std::hash<std::string>{}(key));
        slang::IModule* exports = session->loadModuleFromSourceString(name.c_str(), (name + ".slang").c_str(),
                                                                       source.c_str(), diag.writeRef());
        if (exports == nullptr) {
            return Error::make(ErrorCode::ShaderFailure, "cannot make the link constants of '{}':\n{}", module,
                               diagnostics(diag.get()));
        }
        components.push_back(exports);
    }

    rhi::ComPtr<slang::IComponentType> composed;
    if (SLANG_FAILED(session->createCompositeComponentType(components.data(), static_cast<SlangInt>(components.size()),
                                                           composed.writeRef(), diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot compose '{}':\n{}", module, diagnostics(diag.get()));
    }
    auto program = std::make_shared<Program>();
    if (SLANG_FAILED(composed->link(program->linked.writeRef(), diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot link '{}':\n{}", module, diagnostics(diag.get()));
    }

    if (slang::ProgramLayout* layout = program->linked->getLayout();
        layout != nullptr && layout->getEntryPointCount() > 0) {
        SlangUInt sizes[3] = {1, 1, 1};
        layout->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, sizes);
        for (int i = 0; i < 3; ++i) {
            program->threadGroup[static_cast<size_t>(i)] = static_cast<uint32_t>(sizes[i] == 0 ? 1 : sizes[i]);
        }
    }

    if (!platform::env("SJXL_DEBUG").empty()) log::info("shaders: compiling '{}' [{}]", module, entries.empty() ? std::string() : entries.front());
    rhi::ShaderProgramDesc desc;
    desc.slangGlobalScope = program->linked.get();
    if (SLANG_FAILED(device_->rhi()->createShaderProgram(desc, program->program.writeRef(), diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot compile '{}' for {}:\n{}", module,
                           toString(device_->backend()), diagnostics(diag.get()));
    }
    if (std::string warnings = diagnostics(diag.get()); !warnings.empty()) {
        log::warn("slang '{}': {}", module, warnings);
    }
    programs_.emplace(key, program);
    return std::shared_ptr<const Program>(program);
}

}   // namespace sjxl::gpu
