# Copyright (c) 2026 jesus luque.
sjxl_add_module(gpu
    SOURCES
        src/Device.cpp
        src/Buffer.cpp
        src/ShaderLibrary.cpp
        src/ComputeKernel.cpp
        src/CommandBatch.cpp
        src/ShaderCache.cpp
        src/algo/PrefixSum.cpp
    PUBLIC_LIBS
        sjxl::core
        slang-rhi)
target_compile_definitions(sjxl_gpu PRIVATE
    SJXL_BUILD_SHADER_DIR="${SJXL_SHADER_OUTPUT_DIR}")
