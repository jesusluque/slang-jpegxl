# Copyright (c) 2026 jesus luque.
# Applies the patches named in PATCHES (a ;-list of files) to the current
# directory, in order, skipping any already applied. FetchContent runs its
# PATCH_COMMAND again whenever the command line changes, on a tree that may
# already carry the earlier patches; `git apply` alone would fail there.
foreach(patch IN LISTS PATCHES)
    execute_process(COMMAND git apply --ignore-whitespace --reverse --check "${patch}"
                    RESULT_VARIABLE already OUTPUT_QUIET ERROR_QUIET)
    if(already EQUAL 0)
        message(STATUS "patch already applied: ${patch}")
        continue()
    endif()
    execute_process(COMMAND git apply --ignore-whitespace "${patch}" RESULT_VARIABLE failed)
    if(NOT failed EQUAL 0)
        message(FATAL_ERROR "cannot apply ${patch}")
    endif()
    message(STATUS "applied: ${patch}")
endforeach()
