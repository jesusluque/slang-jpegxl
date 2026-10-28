// Copyright (c) 2026 jesus luque.
//
// The few OS calls the codec makes, in one place.
#pragma once

#include <filesystem>
#include <string>

namespace sjxl::platform {

/// `name` from the environment, or empty.
[[nodiscard]] std::string env(const char* name);

/// Where a user's caches go: ~/Library/Caches on macOS, $XDG_CACHE_HOME or
/// ~/.cache on Linux, %LOCALAPPDATA% on Windows.
[[nodiscard]] std::filesystem::path cacheDirectory();

/// The directory holding the running executable; empty if it cannot be named.
[[nodiscard]] std::filesystem::path executableDir();

}   // namespace sjxl::platform
