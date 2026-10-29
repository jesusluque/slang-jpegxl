// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/ShaderLibrary.h"

#include <filesystem>

#include <fstream>

#include <format>

#include <cstring>

#include <algorithm>

#include <chrono>

#include <functional>
#include <string_view>

#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {

namespace {

uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

}   // namespace

/// A hash of every .slang/.slangh under the search paths plus the Slang version: the
/// precompiled modules' validity.
std::string ShaderLibrary::sourceStamp() {
    if (!stamp_.empty()) return stamp_;
    uint64_t h = 1469598103934665603ull;
    const char* tag = spGetBuildTagString();
    h = fnv1a(h, tag, std::strlen(tag));
    std::vector<std::filesystem::path> files;
    for (const std::string& root : device_->shaderSearchPaths()) {
        std::error_code ec;
        for (std::filesystem::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file()) continue;
            const std::string ext = it->path().extension().string();
            if (ext == ".slang" || ext == ".slangh") files.push_back(it->path());
        }
    }
    std::sort(files.begin(), files.end());
    for (const std::filesystem::path& f : files) {
        const std::string name = f.string();
        h = fnv1a(h, name.data(), name.size());
        std::ifstream in(f, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        h = fnv1a(h, bytes.data(), bytes.size());
    }
    stamp_ = std::format("{:016x}", h);
    return stamp_;
}

slang::IModule* ShaderLibrary::loadModuleCached(const std::string& module, rhi::ComPtr<slang::IBlob>& diag) {
    if (auto found = modules_.find(module); found != modules_.end()) return found->second;
    slang::ISession* session = device_->slangSession();
    const std::filesystem::path dir = device_->moduleCacheDirectory();
    slang::IModule* loaded = nullptr;
    std::filesystem::path cached;
    if (!dir.empty()) {
        std::string name = module;
        std::replace(name.begin(), name.end(), '/', '_');
        cached = dir / (name + "." + sourceStamp() + ".slang-module");
        if (std::ifstream in(cached, std::ios::binary); in) {
            std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            rhi::ComPtr<ISlangBlob> blob;
            if (SLANG_SUCCEEDED(rhi::getRHI()->createBlob(bytes.data(), bytes.size(), blob.writeRef()))) {
                rhi::ComPtr<slang::IBlob> ignored;
                loaded = session->loadModuleFromIRBlob(module.c_str(), (module + ".slang").c_str(), blob.get(), ignored.writeRef());
            }
        }
    }
    if (loaded == nullptr) {
        loaded = session->loadModule(module.c_str(), diag.writeRef());
        if (loaded != nullptr && !cached.empty()) {
            rhi::ComPtr<ISlangBlob> blob;
            if (SLANG_SUCCEEDED(loaded->serialize(blob.writeRef()))) {
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                const std::filesystem::path partial = cached.string() + ".partial";
                if (std::ofstream out(partial, std::ios::binary); out) {
                    out.write(static_cast<const char*>(blob->getBufferPointer()), static_cast<std::streamsize>(blob->getBufferSize()));
                    out.close();
                    std::filesystem::rename(partial, cached, ec);
                }
            }
        }
    }
    if (loaded != nullptr) modules_[module] = loaded;
    return loaded;
}
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
    slang::IModule* loaded = loadModuleCached(module, diag);
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
    const auto t0 = std::chrono::steady_clock::now();
    if (SLANG_FAILED(device_->rhi()->createShaderProgram(desc, program->program.writeRef(), diag.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot compile '{}' for {}:\n{}", module,
                           toString(device_->backend()), diagnostics(diag.get()));
    }
    if (const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ms > 2.0 && !platform::env("SJXL_TIMING").empty()) {
        log::info("timing shader program {}:{}: {:.1f} ms", module, entries.empty() ? std::string() : entries.front(), ms);
    }
    if (std::string warnings = diagnostics(diag.get()); !warnings.empty()) {
        log::warn("slang '{}': {}", module, warnings);
    }
    programs_.emplace(key, program);
    return std::shared_ptr<const Program>(program);
}

}   // namespace sjxl::gpu
