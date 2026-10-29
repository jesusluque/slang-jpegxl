# Copyright (c) 2026 jesus luque.
sjxl_add_module(encoder
    SOURCES
        src/Encoder.cpp
        src/EntropyCpu.cpp
    PUBLIC_LIBS
        sjxl::gpu
        sjxl::codestream
        sjxl::image)
