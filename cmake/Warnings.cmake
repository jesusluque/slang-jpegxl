# Copyright (c) 2026 jesus luque.
# Strict warnings on this project's own targets, and only on those: third-party
# code (slang-rhi, Catch2, stb) keeps whatever its authors chose.

function(sjxl_set_target_defaults target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
            -Wold-style-cast -Wcast-align -Woverloaded-virtual
            -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough)
    endif()
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
endfunction()

# sjxl_add_module(<name> SOURCES ... [PUBLIC_LIBS ...] [PRIVATE_LIBS ...])
#
# A static library `sjxl_<name>`, aliased `sjxl::<name>`, whose public headers
# are modules/<name>/include/sjxl/<name>/*.h.
function(sjxl_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_LIBS;PRIVATE_LIBS" ${ARGN})
    add_library(sjxl_${name} STATIC ${ARG_SOURCES})
    add_library(sjxl::${name} ALIAS sjxl_${name})
    target_include_directories(sjxl_${name}
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(sjxl_${name} PUBLIC cxx_std_20)
    if(ARG_PUBLIC_LIBS)
        target_link_libraries(sjxl_${name} PUBLIC ${ARG_PUBLIC_LIBS})
    endif()
    if(ARG_PRIVATE_LIBS)
        target_link_libraries(sjxl_${name} PRIVATE ${ARG_PRIVATE_LIBS})
    endif()
    sjxl_set_target_defaults(sjxl_${name})
endfunction()
