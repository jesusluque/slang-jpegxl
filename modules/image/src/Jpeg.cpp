// Copyright (c) 2026 jesus luque.
#include "sjxl/image/Jpeg.h"

#include <algorithm>

#include "sjxl/image/Image.h"

namespace sjxl::image {

namespace {

Error corrupt(const char* what) { return Error::make(ErrorCode::Corrupt, "JPEG: {}", what); }

uint32_t be16(const uint8_t* p) { return (uint32_t{p[0]} << 8) | p[1]; }

}   // namespace

Result<JpegFile> readJpeg(const std::filesystem::path& path) {
    JpegFile jpg;
    SJXL_ASSIGN(jpg.data, readFile(path));
    const std::vector<uint8_t>& d = jpg.data;
    if (d.size() < 4 || d[0] != 0xFF || d[1] != 0xD8) return corrupt("no SOI marker");
    size_t pos = 2;
    bool frameSeen = false;
    bool adobe = false;
    while (pos + 4 <= d.size()) {
        if (d[pos] != 0xFF) return corrupt("marker expected");
        const uint8_t marker = d[pos + 1];
        if (marker == 0xFF) { ++pos; continue; }   // fill byte
        pos += 2;
        if (marker == 0xD9) break;   // EOI
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7)) continue;   // SOI, RSTn: no length
        if (pos + 2 > d.size()) return corrupt("truncated segment");
        const uint32_t length = be16(&d[pos]);
        if (length < 2 || pos + length > d.size()) return corrupt("bad segment length");
        const uint8_t* seg = &d[pos + 2];
        const size_t segLen = length - 2;
        switch (marker) {
        case 0xDB: {   // DQT
            size_t i = 0;
            while (i < segLen) {
                const uint32_t pq = seg[i] >> 4, tq = seg[i] & 15;
                ++i;
                if (tq > 3) return corrupt("quant table index");
                if (i + (pq ? 128 : 64) > segLen) return corrupt("truncated DQT");
                static constexpr uint8_t kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                                                        41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                                                        30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
                for (uint32_t k = 0; k < 64; ++k) {
                    const int32_t v = pq ? static_cast<int32_t>(be16(&seg[i + 2 * k])) : seg[i + k];
                    if (v <= 0) return corrupt("zero quant value");
                    jpg.quant[tq][kZigzag[k]] = v;
                }
                jpg.quantPresent[tq] = true;
                i += pq ? 128 : 64;
            }
            break;
        }
        case 0xC4: {   // DHT
            size_t i = 0;
            while (i + 17 <= segLen) {
                const uint32_t tc = seg[i] >> 4, th = seg[i] & 15;
                if (tc > 1 || th > 3) return corrupt("Huffman table class or index");
                JpegHuffmanTable& t = tc == 0 ? jpg.dcTables[th] : jpg.acTables[th];
                t = JpegHuffmanTable{};
                t.present = true;
                uint32_t total = 0;
                for (uint32_t l = 1; l <= 16; ++l) { t.counts[l] = seg[i + l]; total += t.counts[l]; }
                if (total > 256 || i + 17 + total > segLen) return corrupt("Huffman table size");
                t.values.assign(seg + i + 17, seg + i + 17 + total);
                i += 17 + total;
            }
            break;
        }
        case 0xDD:   // DRI
            if (segLen < 2) return corrupt("truncated DRI");
            jpg.restartInterval = be16(seg);
            break;
        case 0xC0: case 0xC1: case 0xC2: case 0xC3: case 0xC5: case 0xC6: case 0xC7:
        case 0xC9: case 0xCA: case 0xCB: case 0xCD: case 0xCE: case 0xCF: {   // SOFn
            if (frameSeen) return corrupt("two frames");
            frameSeen = true;
            jpg.progressive = marker != 0xC0 && marker != 0xC1;   // only baseline / extended sequential Huffman decode here
            if (segLen < 6) return corrupt("truncated SOF");
            if (seg[0] != 8) return Error::make(ErrorCode::Unsupported, "JPEG: {}-bit samples", seg[0]);
            jpg.height = be16(&seg[1]);
            jpg.width = be16(&seg[3]);
            const uint32_t nc = seg[5];
            if (nc != 1 && nc != 3) return Error::make(ErrorCode::Unsupported, "JPEG: {} components", nc);
            if (segLen < 6 + 3 * nc) return corrupt("truncated SOF components");
            for (uint32_t c = 0; c < nc; ++c) {
                JpegComponent comp;
                comp.id = seg[6 + 3 * c];
                comp.hSamp = seg[7 + 3 * c] >> 4;
                comp.vSamp = seg[7 + 3 * c] & 15;
                comp.quantIndex = seg[8 + 3 * c];
                if (comp.hSamp < 1 || comp.hSamp > 4 || comp.vSamp < 1 || comp.vSamp > 4 || comp.quantIndex > 3) return corrupt("component sampling");
                jpg.components.push_back(comp);
            }
            break;
        }
        case 0xEE:   // APP14 Adobe: transform flag 0 means no YCbCr
            if (segLen >= 12 && seg[0] == 'A' && seg[1] == 'd' && seg[2] == 'o' && seg[3] == 'b' && seg[4] == 'e') {
                adobe = true;
                jpg.rgb = seg[11] == 0;
            }
            break;
        case 0xDA: {   // SOS: the scan header, then entropy-coded data up to the next non-RST marker
            if (!frameSeen) return corrupt("scan before frame");
            JpegScan scan;
            const uint32_t ns = seg[0];
            if (ns < 1 || ns > 4 || segLen < 1 + 2 * ns + 3) return corrupt("scan components");
            for (uint32_t i = 0; i < ns; ++i) {
                JpegScan::Component sc;
                const uint32_t id = seg[1 + 2 * i];
                sc.dcTable = seg[2 + 2 * i] >> 4;
                sc.acTable = seg[2 + 2 * i] & 15;
                bool found = false;
                for (uint32_t c = 0; c < jpg.components.size(); ++c) {
                    if (jpg.components[c].id == id) { sc.index = c; found = true; }
                }
                if (!found || sc.dcTable > 3 || sc.acTable > 3) return corrupt("scan component");
                scan.components.push_back(sc);
            }
            scan.ss = seg[1 + 2 * ns];
            scan.se = seg[2 + 2 * ns];
            scan.ah = seg[3 + 2 * ns] >> 4;
            scan.al = seg[3 + 2 * ns] & 15;
            pos += length;
            scan.dataOffset = pos;
            scan.restartOffsets.push_back(0);
            // The entropy-coded segment: 0xFF 0x00 is a stuffed byte, 0xFF 0xD0..D7 a restart marker, anything else ends it.
            while (pos + 1 < d.size()) {
                if (d[pos] != 0xFF) { ++pos; continue; }
                const uint8_t m = d[pos + 1];
                if (m == 0x00 || m == 0xFF) { pos += m == 0x00 ? 2 : 1; continue; }
                if (m >= 0xD0 && m <= 0xD7) { pos += 2; scan.restartOffsets.push_back(static_cast<uint32_t>(pos - scan.dataOffset)); continue; }
                break;
            }
            scan.dataSize = pos - scan.dataOffset;
            jpg.scans.push_back(std::move(scan));
            continue;   // pos already at the next marker
        }
        default:
            break;   // APPn, COM, others: skipped
        }
        pos += length;
    }
    if (!frameSeen || jpg.scans.empty()) return corrupt("no frame or scan");
    if (!adobe && jpg.components.size() == 3 && jpg.components[0].id == 'R' && jpg.components[1].id == 'G' && jpg.components[2].id == 'B') jpg.rgb = true;
    uint32_t hmax = 1, vmax = 1;
    for (const JpegComponent& c : jpg.components) { hmax = std::max(hmax, c.hSamp); vmax = std::max(vmax, c.vSamp); }
    jpg.mcuWidth = 8 * hmax;
    jpg.mcuHeight = 8 * vmax;
    jpg.mcusX = (jpg.width + jpg.mcuWidth - 1) / jpg.mcuWidth;
    jpg.mcusY = (jpg.height + jpg.mcuHeight - 1) / jpg.mcuHeight;
    for (JpegComponent& c : jpg.components) {
        c.blocksX = jpg.mcusX * c.hSamp;
        c.blocksY = jpg.mcusY * c.vSamp;
        if (!jpg.quantPresent[c.quantIndex]) return corrupt("component without a quant table");
    }
    return jpg;
}

}   // namespace sjxl::image
