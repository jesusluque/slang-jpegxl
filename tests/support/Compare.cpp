// Copyright (c) 2026 jesus luque.
#include "Compare.h"

#include <cstdlib>

namespace sjxl::test {

Result<Metrics> compare(const image::Rgb8& a, const image::Rgb8& b) {
    return metrics::compare(*testDevice().library, a, b);
}

std::filesystem::path findTool(const std::string& name) {
    for (const char* dir : {"/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"}) {
        const std::filesystem::path candidate = std::filesystem::path(dir) / name;
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return {};
}

bool run(const std::string& command) { return std::system(command.c_str()) == 0; }

std::filesystem::path scratchDir() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "sjxl-tests";
    std::error_code ignored;
    std::filesystem::create_directories(dir, ignored);
    return dir;
}

std::filesystem::path testData(const std::string& file) { return std::filesystem::path(SJXL_TEST_DATA_DIR) / file; }

}   // namespace sjxl::test
