# Copyright (c) 2026 jesus luque.
# Slang sources compiled at run time by slang-rhi.
#
# The kernels are not embedded: they are compiled by the device the process
# opened, for the backend it opened. They are copied beside the binaries and
# found through SJXL_SHADER_DIR (set at build) or <exe>/../shaders.

set(SJXL_SHADER_SOURCE_DIR "${SJXL_ROOT}/shaders")
if(NOT DEFINED SJXL_SHADER_OUTPUT_DIR)
    set(SJXL_SHADER_OUTPUT_DIR "${SJXL_BUILD}/shaders")
endif()

# One target that copies every shader, so tests and apps depend on the same one.
function(sjxl_shader_copy_target)
    if(TARGET sjxl_shaders_copy)
        return()
    endif()
    file(GLOB_RECURSE _shaders CONFIGURE_DEPENDS
        "${SJXL_SHADER_SOURCE_DIR}/*.slang" "${SJXL_SHADER_SOURCE_DIR}/*.slangh")
    set(_outputs)
    foreach(_src IN LISTS _shaders)
        file(RELATIVE_PATH _rel "${SJXL_SHADER_SOURCE_DIR}" "${_src}")
        set(_dst "${SJXL_SHADER_OUTPUT_DIR}/${_rel}")
        add_custom_command(OUTPUT "${_dst}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "${_dst}"
            DEPENDS "${_src}" VERBATIM)
        list(APPEND _outputs "${_dst}")
    endforeach()
    add_custom_target(sjxl_shaders_copy ALL DEPENDS ${_outputs})
endfunction()

function(sjxl_copy_shaders target)
    sjxl_shader_copy_target()
    add_dependencies(${target} sjxl_shaders_copy)
endfunction()
