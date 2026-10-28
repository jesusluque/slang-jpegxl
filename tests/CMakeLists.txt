# Copyright (c) 2026 jesus luque.
include(Catch)

# sjxl_test(<name> SOURCES ... LIBS ...)
#
# A Catch2 executable registered with ctest case by case. GPU tests skip when
# the machine has no device; they never fall back to anything else.
function(sjxl_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain sjxl_test_support ${ARG_LIBS})
    sjxl_set_target_defaults(${name})
    set_target_properties(${name} PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib")
    add_dependencies(${name} sjxl_shaders_copy)
    catch_discover_tests(${name}
        PROPERTIES ENVIRONMENT "SJXL_SHADER_DIR=${SJXL_SHADER_OUTPUT_DIR}" LABELS gpu
        DISCOVERY_MODE PRE_TEST)
endfunction()

# What every test shares: one device per process, the compare kernel, the
# oracle tools, the test images.
add_library(sjxl_test_support STATIC support/TestDevice.cpp support/Compare.cpp)
target_include_directories(sjxl_test_support PUBLIC support)
target_link_libraries(sjxl_test_support PUBLIC sjxl::gpu sjxl::image sjxl::metrics Catch2::Catch2)
target_compile_definitions(sjxl_test_support PUBLIC SJXL_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
sjxl_set_target_defaults(sjxl_test_support)

sjxl_test(sjxl_gpu_tests
    SOURCES gpu/test_device.cpp gpu/test_prefix.cpp
    LIBS sjxl::gpu)

sjxl_test(sjxl_codestream_tests
    SOURCES codestream/test_tables.cpp codestream/test_parse.cpp
    LIBS sjxl::codestream sjxl::image)

sjxl_test(sjxl_encoder_tests
    SOURCES encoder/test_encode.cpp
    LIBS sjxl::encoder)

sjxl_test(sjxl_decoder_tests
    SOURCES decoder/test_decode.cpp
    LIBS sjxl::decoder sjxl::encoder)
