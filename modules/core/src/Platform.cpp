// Copyright (c) 2026 jesus luque.
#include "sjxl/core/Platform.h"

#include <cstdlib>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace sjxl::platform {

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

std::filesystem::path cacheDirectory() {
#if defined(__APPLE__)
    return std::filesystem::path(env("HOME")) / "Library" / "Caches";
#elif defined(_WIN32)
    return std::filesystem::path(env("LOCALAPPDATA"));
#else
    if (std::string xdg = env("XDG_CACHE_HOME"); !xdg.empty()) {
        return xdg;
    }
    return std::filesystem::path(env("HOME")) / ".cache";
#endif
}

std::filesystem::path executableDir() {
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        std::error_code ignored;
        return std::filesystem::weakly_canonical(buffer.c_str(), ignored).parent_path();
    }
    return {};
#elif defined(_WIN32)
    wchar_t name[MAX_PATH];
    const DWORD length = ::GetModuleFileNameW(nullptr, name, MAX_PATH);
    return length > 0 && length < MAX_PATH ? std::filesystem::path(name).parent_path() : std::filesystem::path{};
#else
    std::error_code failed;
    auto self = std::filesystem::read_symlink("/proc/self/exe", failed);
    return failed ? std::filesystem::path{} : self.parent_path();
#endif
}

}   // namespace sjxl::platform
