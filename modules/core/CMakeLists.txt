# Copyright (c) 2026 jesus luque.
sjxl_add_module(core
    SOURCES
        src/Error.cpp
        src/Log.cpp
        src/Platform.cpp)
find_package(Threads REQUIRED)
target_link_libraries(sjxl_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
