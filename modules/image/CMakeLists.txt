# Copyright (c) 2026 jesus luque.
sjxl_add_module(image
    SOURCES
        src/Image.cpp
        src/Jpeg.cpp
    PUBLIC_LIBS
        sjxl::core
    PRIVATE_LIBS
        sjxl_stb
        ZLIB::ZLIB)
