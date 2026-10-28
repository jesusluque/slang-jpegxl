# Copyright (c) 2026 jesus luque.
add_executable(sjxl main.cpp)
target_link_libraries(sjxl PRIVATE sjxl::encoder sjxl::metrics)
sjxl_set_target_defaults(sjxl)
set_target_properties(sjxl PROPERTIES BUILD_RPATH "${SLANG_ROOT}/lib")
sjxl_copy_shaders(sjxl)
