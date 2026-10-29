# Copyright (c) 2026 jesus luque.
sjxl_add_module(decoder
    SOURCES
        src/Decoder.cpp
        src/DcGroupCpu.cpp
    PUBLIC_LIBS
        sjxl::gpu
        sjxl::codestream
        sjxl::image)
