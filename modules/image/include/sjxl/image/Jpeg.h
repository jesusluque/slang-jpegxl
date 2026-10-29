// Copyright (c) 2026 jesus luque.
//
// The structure of a baseline JPEG file for recompression: the headers parsed
// on the CPU (frame, components, quantisation and Huffman tables, scans,
// restart interval, colour transform) and the entropy-coded scan bytes kept as
// they are for the Huffman-decoding kernel.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "sjxl/core/Result.h"

namespace sjxl::image {

struct JpegComponent {
    uint32_t id = 0;
    uint32_t hSamp = 1, vSamp = 1;   ///< sampling factors
    uint32_t quantIndex = 0;
    uint32_t blocksX = 0, blocksY = 0;   ///< the component's block grid, padded to whole MCUs
};

struct JpegHuffmanTable {
    bool                     present = false;
    std::array<uint32_t, 17> counts{};   ///< codes of each length 1..16
    std::vector<uint32_t>    values;     ///< symbols by increasing code length
};

struct JpegScan {
    struct Component {
        uint32_t index = 0;     ///< into JpegFile::components
        uint32_t dcTable = 0, acTable = 0;
    };
    std::vector<Component> components;
    uint32_t ss = 0, se = 63, ah = 0, al = 0;   ///< spectral selection / successive approximation (baseline: 0, 63, 0, 0)
    size_t   dataOffset = 0, dataSize = 0;      ///< the entropy-coded bytes in `data` (restart markers included)
    std::vector<uint32_t> restartOffsets;       ///< byte offsets (from dataOffset) where each restart interval starts
};

struct JpegFile {
    uint32_t width = 0, height = 0;
    uint32_t mcuWidth = 0, mcuHeight = 0;   ///< MCU size in pixels (8 x the largest sampling factors)
    uint32_t mcusX = 0, mcusY = 0;
    uint32_t restartInterval = 0;          ///< MCUs per restart interval (0: none)
    bool     progressive = false;
    bool     rgb = false;                  ///< Adobe transform 0 or R/G/B component ids: no YCbCr
    std::vector<JpegComponent>          components;
    std::array<std::array<int32_t, 64>, 4> quant{};   ///< natural (row-major) order
    std::array<bool, 4>                 quantPresent{};
    std::array<JpegHuffmanTable, 4>     dcTables, acTables;
    std::vector<JpegScan>               scans;
    std::vector<uint8_t>                data;   ///< the whole file; scans index into it
};

/// The headers of a JPEG file. Progressive and arithmetic-coded files are read
/// but not decodable here (`progressive`); 12-bit and lossless JPEG fail.
[[nodiscard]] Result<JpegFile> readJpeg(const std::filesystem::path& path);

}   // namespace sjxl::image
