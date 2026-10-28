# Copyright (c) 2026 jesus luque.
# Third-party dependencies, fetched at configure time and pinned.
#
# Slang itself is not fetched: one Slang runtime per process, the release at
# ~/tools/slang (2026.14.1), which slang-rhi links too. The same recipe as
# athenea (the sibling renderer), so a machine set up for one builds the other.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# --- Slang --------------------------------------------------------------------

set(SLANG_ROOT "$ENV{SLANG_ROOT}" CACHE PATH "Slang release install (bin/, lib/, include/)")
if(NOT SLANG_ROOT)
    set(SLANG_ROOT "$ENV{HOME}/tools/slang" CACHE PATH "" FORCE)
endif()
if(NOT EXISTS "${SLANG_ROOT}/include/slang.h")
    message(FATAL_ERROR
        "Slang not found at ${SLANG_ROOT}. Unpack the 2026.14.1 release from "
        "github.com/shader-slang/slang there, or set SLANG_ROOT.")
endif()
set(ENV{SLANG_ROOT} "${SLANG_ROOT}")
# slangc is not used by the build; it is how a shader is compile-checked by
# hand (CLAUDE.md). Found here so the path is one place.
find_program(SJXL_SLANGC slangc HINTS "${SLANG_ROOT}/bin" NO_DEFAULT_PATH)

# --- slang-rhi -----------------------------------------------------------------
#
# Pinned to a commit rather than a tag: slang-rhi does not cut releases, and
# its API moves. e17f6d7 is 2026-09-03 ("Add opacity micromap support").
#
# Patches (cmake/patches, from athenea):
#   - slang-rhi-metal-command-buffer-errors.patch: a Metal command buffer that
#     fails (the device out of memory) is reported by the next submit or
#     waitOnHost as SLANG_E_OUT_OF_MEMORY instead of asserting; a buffer that
#     cannot be allocated is OOM rather than SLANG_FAIL.
#   - slang-rhi-cuda-driver-symbols.patch: slang-rhi's dlopen'd cu* pointers
#     in a namespace, so another CUDA user in the process binds libcuda's.

set(SLANG_RHI_FETCH_SLANG OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_SLANG_INCLUDE_DIR "${SLANG_ROOT}/include" CACHE STRING "" FORCE)
set(SLANG_RHI_SLANG_BINARY_DIR "${SLANG_ROOT}" CACHE STRING "" FORCE)
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_TESTS_WITH_GLFW OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_INSTALL OFF CACHE BOOL "" FORCE)
# No CPU backend: nothing here may run its numbers on the CPU, and a backend
# that exists gets used as a fallback by someone. No WebGPU: Dawn is a large
# download for a target this codec does not have yet.
set(SLANG_RHI_ENABLE_CPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_WGPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
if(APPLE)
    # Metal only. Vulkan on macOS is MoltenVK over Metal: the device we have,
    # minus a translation layer.
    set(SLANG_RHI_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    set(SLANG_RHI_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(SLANG_RHI_ENABLE_OPTIX OFF CACHE BOOL "" FORCE)
    set(SLANG_RHI_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
else()
    # A codec traces no rays.
    set(SLANG_RHI_ENABLE_OPTIX OFF CACHE BOOL "" FORCE)
endif()

set(SJXL_RHI_PATCHES
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-metal-command-buffer-errors.patch"
    "${CMAKE_CURRENT_LIST_DIR}/patches/slang-rhi-cuda-driver-symbols.patch")
string(JOIN "$<SEMICOLON>" SJXL_RHI_PATCHES ${SJXL_RHI_PATCHES})

if(NOT TARGET slang-rhi)
FetchContent_Declare(slang_rhi
    GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
    GIT_TAG        e17f6d75f858f9b7cb91bc102a7b8c6fda0435dc
    GIT_SHALLOW    FALSE
    GIT_SUBMODULES ""
    PATCH_COMMAND  ${CMAKE_COMMAND}
                   "-DPATCHES=${SJXL_RHI_PATCHES}"
                   -P "${CMAKE_CURRENT_LIST_DIR}/patches/apply.cmake"
    UPDATE_DISCONNECTED TRUE
    SYSTEM)
FetchContent_MakeAvailable(slang_rhi)
endif()

# --- small libraries ------------------------------------------------------------

# stb_image / stb_image_write: PNG and PPM files in and out. File I/O only; the
# pixels go to the device as they are read.
FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        f58f558c120e9b32c217290b80bad1a0729fbb2c
    GIT_SHALLOW    FALSE
    SYSTEM)
FetchContent_MakeAvailable(stb)
add_library(sjxl_stb INTERFACE)
target_include_directories(sjxl_stb SYSTEM INTERFACE ${stb_SOURCE_DIR})

if(BUILD_TESTING AND NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1
        GIT_SHALLOW    TRUE
        SYSTEM)
    FetchContent_MakeAvailable(catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()
