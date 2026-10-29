# Copyright (c) 2026 jesus luque.
cmake_minimum_required(VERSION 3.24)
project(sjxl VERSION 0.1.0 LANGUAGES C CXX)
if(APPLE)
    # slang-rhi's Metal backend is Objective-C++; the language is enabled at the
    # top so every directory has its compile rule at generate time.
    enable_language(OBJC OBJCXX)
endif()

# sjxl: a JPEG XL encoder and decoder whose every numeric step is a Slang
# compute kernel run through slang-rhi (Metal on Apple, CUDA or Vulkan on
# Linux). The CPU parses and writes headers, the table of contents and files;
# transforms, quantisation, tokens, histograms, code tables, bit emission and
# symbol decoding are kernels. docs/decisions.md holds the design and the
# roadmap; CLAUDE.md the short version.

set(SJXL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(SJXL_BUILD "${CMAKE_CURRENT_BINARY_DIR}")

option(SJXL_BUILD_APPS "the sjxl command line tool" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SJXL_BUILD}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SJXL_BUILD}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${SJXL_BUILD}/lib")

list(PREPEND CMAKE_MODULE_PATH "${SJXL_ROOT}/cmake")

include("${SJXL_ROOT}/cmake/Warnings.cmake")
# Before Dependencies: Catch2 is fetched only when BUILD_TESTING is on, which
# CTest defines.
include(CTest)
include("${SJXL_ROOT}/cmake/Dependencies.cmake")
include("${SJXL_ROOT}/cmake/Shaders.cmake")
sjxl_shader_copy_target()

find_package(ZLIB REQUIRED)   # the PNG writer's deflate
add_subdirectory(modules)
if(SJXL_BUILD_APPS)
    add_subdirectory(apps/sjxl)
endif()
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
