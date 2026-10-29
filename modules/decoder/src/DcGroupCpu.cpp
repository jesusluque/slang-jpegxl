// Copyright (c) 2026 jesus luque.
//
// A C++ port of the parts of shaders/sjxl/dec/{entropy,modular,modular_channel,
// dcgroup}.slangh a DC group section needs: the bit reader, DecodeHistograms
// (prefix codes and ANS with alias tables, simple and MTF-coded context maps),
// the MA tree, the weighted predictor and the channel loop with libjxl's 64-bit
// pixel arithmetic. Everything the GPU path rejects or that this port leaves
// out (LZ77, transforms in the stream, extra channels) returns false so the
// GPU kernel decodes the group instead. The outputs must match the GPU path
// bit for bit: the `[dc-cpu]` test compares the two.

#include "sjxl/decoder/DcGroupCpu.h"

#include "sjxl/codestream/WeightedPredictor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>

namespace sjxl::dec::cpu {
namespace {

using codestream::ParsedFrame;

constexpr uint32_t kAnsLogTabSize = 12;
constexpr uint32_t kAnsTabSize = 1u << kAnsLogTabSize;
constexpr uint32_t kAnsSignature = 0x13;
constexpr uint32_t kHuffmanTableBits = 8;
constexpr uint32_t kPrefixMaxBits = 15;
constexpr uint32_t kMaxTreeNodes = 8192;
constexpr uint32_t kNumNonrefProperties = 16;
constexpr uint32_t kWpProp = 15;
constexpr uint32_t kPredWeighted = 6;

/// The port reports what it cannot do as `Unsupported` (the caller falls back)
/// and corrupt streams as `Corrupt`.
struct Unsupported {};
struct Corrupt {};

// --- bit reader ---------------------------------------------------------------------

struct BitReader {
    const uint32_t* words = nullptr;
    size_t          numWords = 0;
    size_t          pos = 0;   // bits

    uint32_t wordAt(size_t w) const { return w < numWords ? words[w] : 0u; }
    uint32_t peek(uint32_t n) const {
        if (n == 0) return 0;
        const size_t w = pos >> 5;
        const uint32_t s = static_cast<uint32_t>(pos & 31);
        uint64_t v = (static_cast<uint64_t>(wordAt(w + 1)) << 32) | wordAt(w);
        v >>= s;
        return n < 32 ? static_cast<uint32_t>(v & ((1ull << n) - 1)) : static_cast<uint32_t>(v);
    }
    uint32_t read(uint32_t n) {
        const uint32_t v = peek(n);
        pos += n;
        return v;
    }
    void consume(uint32_t n) { pos += n; }
};

struct Distr {
    bool     direct;
    uint32_t valueOrOffset;
    uint32_t extraBits;
};
Distr val(uint32_t v) { return {true, v, 0}; }
Distr bitsOffset(uint32_t bits, uint32_t offset) { return {false, offset, bits}; }

uint32_t readU32(BitReader& br, Distr d0, Distr d1, Distr d2, Distr d3) {
    const uint32_t sel = br.read(2);
    const Distr d = sel == 0 ? d0 : sel == 1 ? d1 : sel == 2 ? d2 : d3;
    if (d.direct) return d.valueOrOffset;
    return br.read(d.extraBits) + d.valueOrOffset;
}

uint32_t ceilLog2Nonzero(uint32_t x) {
    uint32_t n = 0;
    while ((1u << n) < x) ++n;
    return n;
}

int32_t unpackSigned(uint32_t v) { return static_cast<int32_t>((v >> 1) ^ (0u - (v & 1))); }

// --- hybrid uint --------------------------------------------------------------------

uint32_t packConfig(uint32_t splitExp, uint32_t msb, uint32_t lsb) { return splitExp | (msb << 8) | (lsb << 16); }

uint32_t readHybridUintConfig(BitReader& br, uint32_t config, uint32_t token) {
    const uint32_t splitExp = config & 0xFF, msb = (config >> 8) & 0xFF, lsb = (config >> 16) & 0xFF;
    const uint32_t splitToken = 1u << splitExp;
    if (token < splitToken) return token;
    uint32_t nbits = splitExp - (msb + lsb) + ((token - splitToken) >> (msb + lsb));
    nbits &= 31u;
    const uint32_t low = token & ((1u << lsb) - 1);
    token >>= lsb;
    const uint32_t bits = br.read(nbits);
    return (((((1u << msb) | (token & ((1u << msb) - 1))) << nbits) | bits) << lsb) | low;
}

uint32_t decodeUintConfig(BitReader& br, uint32_t logAlphaSize) {
    const uint32_t splitExp = br.read(ceilLog2Nonzero(logAlphaSize + 1));
    uint32_t msb = 0, lsb = 0;
    if (splitExp != logAlphaSize) {
        msb = br.read(ceilLog2Nonzero(splitExp + 1));
        if (msb > splitExp) throw Corrupt{};
        lsb = br.read(ceilLog2Nonzero(splitExp - msb + 1));
    }
    if (lsb + msb > splitExp) throw Corrupt{};
    return packConfig(splitExp, msb, lsb);
}

// --- prefix codes -------------------------------------------------------------------

uint32_t huffEntry(uint32_t bits, uint32_t value) { return (bits << 16) | value; }
uint32_t huffBits(uint32_t e) { return e >> 16; }
uint32_t huffValue(uint32_t e) { return e & 0xFFFF; }

int getNextKey(int key, int len) {
    int step = 1 << (len - 1);
    while ((key & step) != 0) step >>= 1;
    return (key & (step - 1)) + step;
}

void replicateValue(uint32_t* table, int step, int end, uint32_t code) {
    do {
        end -= step;
        table[end] = code;
    } while (end > 0);
}

uint32_t nextTableBitSize(const uint32_t* count, uint32_t len, uint32_t rootBits) {
    uint32_t left = 1u << (len - rootBits);
    while (len < kPrefixMaxBits) {
        if (left <= count[len]) break;
        left -= count[len];
        ++len;
        left <<= 1;
    }
    return len - rootBits;
}

/// BuildHuffmanTable; `table` has room for alphabet + 376 entries. Returns the size (0 on failure).
uint32_t buildHuffmanTable(uint32_t* tableOff, uint32_t rootBits, const uint32_t* lengths, uint32_t alphabet) {
    uint32_t count[16] = {};
    uint32_t offset[16] = {};
    std::vector<uint32_t> sorted(alphabet);
    for (uint32_t s = 0; s < alphabet; ++s) ++count[lengths[s]];
    uint32_t maxLength = 1;
    {
        uint32_t sum = 0;
        for (uint32_t len = 1; len <= kPrefixMaxBits; ++len) {
            offset[len] = sum;
            if (count[len] != 0) {
                sum += count[len];
                maxLength = len;
            }
        }
    }
    for (uint32_t s = 0; s < alphabet; ++s) {
        const uint32_t l = lengths[s];
        if (l != 0) sorted[offset[l]++] = s;
    }
    uint32_t* table = tableOff;
    uint32_t tableBits = rootBits;
    int tableSize = 1 << tableBits;
    int totalSize = tableSize;
    if (offset[kPrefixMaxBits] == 1) {
        const uint32_t code = huffEntry(0, sorted[0]);
        for (int key = 0; key < totalSize; ++key) tableOff[key] = code;
        return static_cast<uint32_t>(totalSize);
    }
    if (tableBits > maxLength) {
        tableBits = maxLength;
        tableSize = 1 << tableBits;
    }
    int key = 0;
    uint32_t symbol = 0;
    uint32_t bits = 1;
    int step = 2;
    do {
        for (; count[bits] != 0; --count[bits]) {
            replicateValue(table + key, step, tableSize, huffEntry(bits, sorted[symbol]));
            ++symbol;
            key = getNextKey(key, static_cast<int>(bits));
        }
        step <<= 1;
    } while (++bits <= tableBits);
    while (totalSize != tableSize) {
        for (int i = 0; i < tableSize; ++i) table[tableSize + i] = table[i];
        tableSize <<= 1;
    }
    const int mask = totalSize - 1;
    int low = -1;
    step = 2;
    for (uint32_t len = rootBits + 1; len <= maxLength; ++len, step <<= 1) {
        for (; count[len] != 0; --count[len]) {
            if ((key & mask) != low) {
                table += tableSize;
                tableBits = nextTableBitSize(count, len, rootBits);
                tableSize = 1 << tableBits;
                totalSize += tableSize;
                low = key & mask;
                tableOff[low] = huffEntry(tableBits + rootBits, static_cast<uint32_t>(table - tableOff) - static_cast<uint32_t>(low));
            }
            replicateValue(table + (key >> rootBits), step, tableSize, huffEntry(len - rootBits, sorted[symbol]));
            ++symbol;
            key = getNextKey(key, static_cast<int>(len));
        }
    }
    return static_cast<uint32_t>(totalSize);
}

bool readHuffmanCodeLengths(BitReader& br, const uint32_t* clc, uint32_t numSymbols, uint32_t* lengths) {
    uint32_t table[32 + 400] = {};
    if (buildHuffmanTable(table, 5, clc, 18) == 0) return false;
    uint32_t symbol = 0;
    uint32_t prevCodeLen = 8;
    int repeat = 0;
    uint32_t repeatCodeLen = 0;
    int space = 32768;
    while (symbol < numSymbols && space > 0) {
        const uint32_t e = table[br.peek(5)];
        br.consume(huffBits(e));
        const uint32_t codeLen = huffValue(e);
        if (codeLen < 16) {
            repeat = 0;
            lengths[symbol++] = codeLen;
            if (codeLen != 0) {
                prevCodeLen = codeLen;
                space -= static_cast<int>(32768u >> codeLen);
            }
        } else {
            const uint32_t extraBits = codeLen - 14;
            uint32_t newLen = 0;
            if (codeLen == 16) newLen = prevCodeLen;
            if (repeatCodeLen != newLen) {
                repeat = 0;
                repeatCodeLen = newLen;
            }
            const int oldRepeat = repeat;
            if (repeat > 0) {
                repeat -= 2;
                repeat <<= extraBits;
            }
            repeat += static_cast<int>(br.read(extraBits)) + 3;
            const int repeatDelta = repeat - oldRepeat;
            if (symbol + static_cast<uint32_t>(repeatDelta) > numSymbols) return false;
            for (int i = 0; i < repeatDelta; ++i) lengths[symbol + static_cast<uint32_t>(i)] = repeatCodeLen;
            symbol += static_cast<uint32_t>(repeatDelta);
            if (repeatCodeLen != 0) space -= repeatDelta << (15 - repeatCodeLen);
        }
    }
    if (space != 0) return false;
    for (; symbol < numSymbols; ++symbol) lengths[symbol] = 0;
    return true;
}

bool readSimpleCode(BitReader& br, uint32_t alphabet, std::vector<uint32_t>& table) {
    uint32_t maxBits = 0;
    if (alphabet > 1) {
        uint32_t v = alphabet - 1;
        while (v != 0) { ++maxBits; v >>= 1; }
    }
    uint32_t numSymbols = br.read(2) + 1;
    uint32_t symbols[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < numSymbols; ++i) {
        const uint32_t s = br.read(maxBits);
        if (s >= alphabet) return false;
        symbols[i] = s;
    }
    for (uint32_t i = 0; i + 1 < numSymbols; ++i)
        for (uint32_t j = i + 1; j < numSymbols; ++j)
            if (symbols[i] == symbols[j]) return false;
    if (numSymbols == 4) numSymbols += br.read(1);
    uint32_t tableSize = 1;
    table.assign(1u << kHuffmanTableBits, 0);
    switch (numSymbols) {
    case 1: table[0] = huffEntry(0, symbols[0]); break;
    case 2:
        if (symbols[0] > symbols[1]) std::swap(symbols[0], symbols[1]);
        table[0] = huffEntry(1, symbols[0]);
        table[1] = huffEntry(1, symbols[1]);
        tableSize = 2;
        break;
    case 3:
        if (symbols[1] > symbols[2]) std::swap(symbols[1], symbols[2]);
        table[0] = huffEntry(1, symbols[0]);
        table[2] = huffEntry(1, symbols[0]);
        table[1] = huffEntry(2, symbols[1]);
        table[3] = huffEntry(2, symbols[2]);
        tableSize = 4;
        break;
    case 4:
        for (uint32_t i = 0; i < 3; ++i)
            for (uint32_t j = i + 1; j < 4; ++j)
                if (symbols[i] > symbols[j]) std::swap(symbols[i], symbols[j]);
        table[0] = huffEntry(2, symbols[0]);
        table[2] = huffEntry(2, symbols[1]);
        table[1] = huffEntry(2, symbols[2]);
        table[3] = huffEntry(2, symbols[3]);
        tableSize = 4;
        break;
    case 5:
        if (symbols[2] > symbols[3]) std::swap(symbols[2], symbols[3]);
        table[0] = huffEntry(1, symbols[0]);
        table[1] = huffEntry(2, symbols[1]);
        table[2] = huffEntry(1, symbols[0]);
        table[3] = huffEntry(3, symbols[2]);
        table[4] = huffEntry(1, symbols[0]);
        table[5] = huffEntry(2, symbols[1]);
        table[6] = huffEntry(1, symbols[0]);
        table[7] = huffEntry(3, symbols[3]);
        tableSize = 8;
        break;
    default: return false;
    }
    const uint32_t goal = 1u << kHuffmanTableBits;
    while (tableSize != goal) {
        for (uint32_t i = 0; i < tableSize; ++i) table[tableSize + i] = table[i];
        tableSize <<= 1;
    }
    return true;
}

std::vector<uint32_t> readHuffmanCode(BitReader& br, uint32_t alphabet) {
    if (alphabet > (1u << kPrefixMaxBits)) throw Corrupt{};
    std::vector<uint32_t> table(std::max<uint32_t>(alphabet + 376, 256), 0);
    const uint32_t simpleOrSkip = br.read(2);
    if (simpleOrSkip == 1) {
        if (!readSimpleCode(br, alphabet, table)) throw Corrupt{};
        return table;
    }
    std::vector<uint32_t> lengths(alphabet, 0);
    uint32_t clc[18] = {};
    static const uint32_t order[18] = {1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const uint32_t huffBitsTab[16] = {2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4};
    static const uint32_t huffValTab[16] = {0, 4, 3, 2, 0, 4, 3, 1, 0, 4, 3, 2, 0, 4, 3, 5};
    int space = 32;
    uint32_t numCodes = 0;
    for (uint32_t i = simpleOrSkip; i < 18 && space > 0; ++i) {
        const uint32_t idx = br.peek(4);
        br.consume(huffBitsTab[idx]);
        const uint32_t v = huffValTab[idx];
        clc[order[i]] = v;
        if (v != 0) {
            space -= static_cast<int>(32u >> v);
            ++numCodes;
        }
    }
    if (!(numCodes == 1 || space == 0)) throw Corrupt{};
    if (!readHuffmanCodeLengths(br, clc, alphabet, lengths.data())) throw Corrupt{};
    if (buildHuffmanTable(table.data(), kHuffmanTableBits, lengths.data(), alphabet) == 0) throw Corrupt{};
    return table;
}

uint32_t readPrefixSymbol(BitReader& br, const uint32_t* table) {
    uint32_t at = br.peek(kHuffmanTableBits);
    uint32_t e = table[at];
    uint32_t nBits = huffBits(e);
    if (nBits > kHuffmanTableBits) {
        br.consume(kHuffmanTableBits);
        nBits -= kHuffmanTableBits;
        at += huffValue(e);
        at += br.peek(nBits);
        e = table[at];
    }
    br.consume(huffBits(e));
    return huffValue(e);
}

// --- ANS histograms and alias tables ------------------------------------------------

uint32_t decodeVarLenUint8(BitReader& br) {
    if (br.read(1) != 0) {
        const uint32_t nbits = br.read(3);
        if (nbits == 0) return 1;
        return br.read(nbits) + (1u << nbits);
    }
    return 0;
}

uint32_t decodeVarLenUint16(BitReader& br) {
    if (br.read(1) != 0) {
        const uint32_t nbits = br.read(4);
        if (nbits == 0) return 1;
        return br.read(nbits) + (1u << nbits);
    }
    return 0;
}

uint32_t populationCountPrecision(uint32_t logcount, uint32_t shift) {
    const int r = std::min(static_cast<int>(logcount), static_cast<int>(shift) - static_cast<int>((kAnsLogTabSize - logcount) >> 1));
    return r < 0 ? 0 : static_cast<uint32_t>(r);
}

/// ReadHistogram into counts[256]; returns the alphabet length, 0 on error.
uint32_t readAnsHistogram(BitReader& br, uint32_t* counts) {
    const uint32_t range = kAnsTabSize;
    std::fill(counts, counts + 256, 0u);
    if (br.read(1) == 1) {
        uint32_t symbols[2] = {0, 0};
        uint32_t maxSymbol = 0;
        const uint32_t numSymbols = br.read(1) + 1;
        for (uint32_t i = 0; i < numSymbols; ++i) {
            symbols[i] = decodeVarLenUint8(br);
            maxSymbol = std::max(maxSymbol, symbols[i]);
        }
        if (maxSymbol >= 256) return 0;
        if (numSymbols == 1) {
            counts[symbols[0]] = range;
        } else {
            if (symbols[0] == symbols[1]) return 0;
            counts[symbols[0]] = br.read(kAnsLogTabSize);
            counts[symbols[1]] = range - counts[symbols[0]];
        }
        return maxSymbol + 1;
    }
    if (br.read(1) == 1) {
        const uint32_t alphabet = decodeVarLenUint8(br) + 1;
        if (alphabet > range) return 0;
        const uint32_t count = range / alphabet;
        const uint32_t rem = range % alphabet;
        for (uint32_t i = 0; i < alphabet; ++i) counts[i] = count + (i < rem ? 1 : 0);
        return alphabet;
    }
    uint32_t shift;
    {
        uint32_t upperBoundLog = 0;
        {
            uint32_t v = kAnsLogTabSize + 1;
            while (v > 1) { ++upperBoundLog; v >>= 1; }
        }
        uint32_t lg = 0;
        for (; lg < upperBoundLog; ++lg) {
            if (br.read(1) == 0) break;
        }
        shift = (br.read(lg) | (1u << lg)) - 1;
        if (shift > kAnsLogTabSize + 1) return 0;
    }
    const uint32_t length = decodeVarLenUint8(br) + 3;
    if (length > 256) return 0;
    uint32_t logs[256] = {};
    uint32_t same[256] = {};
    int omitLog = -1;
    int omitPos = -1;
    for (uint32_t i = 0; i < length; ++i) {
        const uint32_t idx = br.peek(7);
        uint32_t bits, value;
        const uint32_t col = idx & 7;
        if (col == 1) {
            static const uint32_t b[16] = {7, 4, 5, 4, 6, 4, 5, 4, 7, 4, 5, 4, 6, 4, 5, 4};
            static const uint32_t v[16] = {12, 4, 0, 4, 11, 4, 0, 4, 13, 4, 0, 4, 11, 4, 0, 4};
            const uint32_t row = idx >> 3;
            bits = b[row];
            value = v[row];
        } else {
            static const uint32_t b[8] = {3, 0, 3, 4, 3, 3, 3, 4};
            static const uint32_t v[8] = {10, 0, 7, 3, 6, 8, 9, 5};
            bits = b[col];
            value = v[col];
            if (col == 3 || col == 7) {
                const uint32_t row = idx >> 3;
                if (col == 3) value = (row & 1) == 0 ? 3 : 1;
                else value = (row & 1) == 0 ? 5 : 2;
            }
        }
        br.consume(bits);
        const int logcount = static_cast<int>(value) - 1;
        logs[i] = static_cast<uint32_t>(logcount);
        if (logcount == static_cast<int>(kAnsLogTabSize)) {
            const uint32_t rleLength = decodeVarLenUint8(br);
            same[i] = rleLength + 5;
            i += rleLength + 3;
            continue;
        }
        if (logcount > omitLog) {
            omitLog = logcount;
            omitPos = static_cast<int>(i);
        }
    }
    if (omitPos < 0) return 0;
    if (static_cast<uint32_t>(omitPos) + 1 < length && static_cast<int>(logs[static_cast<uint32_t>(omitPos) + 1]) == static_cast<int>(kAnsLogTabSize)) return 0;
    int prev = 0;
    int numsame = 0;
    int total = 0;
    for (uint32_t i = 0; i < length; ++i) {
        if (same[i] != 0) {
            numsame = static_cast<int>(same[i]) - 1;
            prev = i > 0 ? static_cast<int>(counts[i - 1]) : 0;
        }
        if (numsame > 0) {
            counts[i] = static_cast<uint32_t>(prev);
            --numsame;
        } else {
            const int code = static_cast<int>(logs[i]);
            if (static_cast<int>(i) == omitPos || code < 0) {
                continue;
            } else if (shift == 0 || code == 0) {
                counts[i] = 1u << code;
            } else {
                const uint32_t bitcount = populationCountPrecision(static_cast<uint32_t>(code), shift);
                counts[i] = (1u << code) + (br.read(bitcount) << (static_cast<uint32_t>(code) - bitcount));
            }
        }
        total += static_cast<int>(counts[i]);
    }
    const int omitted = static_cast<int>(range) - total;
    if (omitted <= 0) return 0;
    counts[static_cast<uint32_t>(omitPos)] = static_cast<uint32_t>(omitted);
    return length;
}

void aliasSet(uint32_t* t, uint32_t i, uint32_t cutoff, uint32_t right, uint32_t freq0, uint32_t offsets1, uint32_t freq1xor) {
    t[2 * i] = (cutoff & 0xFF) | ((right & 0xFF) << 8) | ((freq0 & 0xFFFF) << 16);
    t[2 * i + 1] = (offsets1 & 0xFFFF) | ((freq1xor & 0xFFFF) << 16);
}

/// InitAliasTable; `table` holds 2 << logAlpha words.
bool initAliasTable(uint32_t* counts, uint32_t length, uint32_t logAlpha, uint32_t* table) {
    const uint32_t tableSize = 1u << logAlpha;
    while (length > 0 && counts[length - 1] == 0) --length;
    if (length == 0) {
        counts[0] = kAnsTabSize;
        length = 1;
    }
    if (length > tableSize) return false;
    const uint32_t entrySize = kAnsTabSize >> logAlpha;
    int single = -1;
    uint32_t sum = 0;
    for (uint32_t s = 0; s < length; ++s) {
        sum += counts[s];
        if (counts[s] == kAnsTabSize) single = static_cast<int>(s);
    }
    if (sum != kAnsTabSize) return false;
    if (single >= 0) {
        for (uint32_t i = 0; i < tableSize; ++i) aliasSet(table, i, 0, static_cast<uint32_t>(single), 0, entrySize * i, kAnsTabSize);
        return true;
    }
    std::vector<uint32_t> cutoffs(tableSize), under(tableSize), over(tableSize), right(tableSize), offsets1(tableSize, 0);
    uint32_t nUnder = 0, nOver = 0;
    for (uint32_t i = 0; i < length; ++i) {
        cutoffs[i] = counts[i];
        if (cutoffs[i] > entrySize) over[nOver++] = i;
        else if (cutoffs[i] < entrySize) under[nUnder++] = i;
    }
    for (uint32_t i = length; i < tableSize; ++i) {
        cutoffs[i] = 0;
        under[nUnder++] = i;
    }
    for (uint32_t i = 0; i < tableSize; ++i) right[i] = i;
    while (nOver > 0) {
        const uint32_t overI = over[--nOver];
        if (nUnder == 0) return false;
        const uint32_t underI = under[--nUnder];
        const uint32_t underBy = entrySize - cutoffs[underI];
        cutoffs[overI] -= underBy;
        right[underI] = overI;
        offsets1[underI] = cutoffs[overI];
        if (cutoffs[overI] < entrySize) under[nUnder++] = overI;
        else if (cutoffs[overI] > entrySize) over[nOver++] = overI;
    }
    for (uint32_t i = 0; i < tableSize; ++i) {
        uint32_t r = right[i];
        uint32_t o1 = offsets1[i];
        uint32_t cutoff;
        if (cutoffs[i] == entrySize) {
            r = i;
            o1 = 0;
            cutoff = 0;
        } else {
            o1 -= cutoffs[i];
            cutoff = cutoffs[i];
        }
        const uint32_t freq0 = i < length ? counts[i] : 0;
        const uint32_t freq1 = r < length ? counts[r] : 0;
        aliasSet(table, i, cutoff, r, freq0, o1, freq1 ^ freq0);
    }
    return true;
}

// --- code sets (DecodeHistograms) ---------------------------------------------------

struct CodeSet {
    bool                               lz77 = false;
    uint32_t                           numContexts = 0;
    uint32_t                           numClusters = 1;
    bool                               usePrefix = false;
    uint32_t                           logAlpha = 0;
    std::vector<uint32_t>              ctxMap;
    std::vector<uint32_t>              configs;
    std::vector<std::vector<uint32_t>> tables;   // per cluster: alias entries (2 words each) or a prefix table
};

struct SymReader {
    const CodeSet* cs = nullptr;
    uint32_t       state = 0;
    uint32_t       logEntrySize = 0;
    uint32_t       entryMask = 0;
};

void decodeCodeTail(BitReader& br, CodeSet& cs, uint32_t numClusters) {
    cs.usePrefix = br.read(1) != 0;
    cs.logAlpha = cs.usePrefix ? kPrefixMaxBits : br.read(2) + 5;
    cs.numClusters = numClusters;
    cs.configs.resize(numClusters);
    cs.tables.resize(numClusters);
    for (uint32_t c = 0; c < numClusters; ++c) cs.configs[c] = decodeUintConfig(br, cs.logAlpha);
    if (cs.usePrefix) {
        std::vector<uint32_t> sizes(numClusters);
        for (uint32_t c = 0; c < numClusters; ++c) {
            sizes[c] = decodeVarLenUint16(br) + 1;
            if (sizes[c] > (1u << kPrefixMaxBits)) throw Corrupt{};
        }
        for (uint32_t c = 0; c < numClusters; ++c) {
            if (sizes[c] > 1) {
                cs.tables[c] = readHuffmanCode(br, sizes[c]);
            } else {
                cs.tables[c].assign(256, huffEntry(0, 0));
            }
        }
    } else {
        uint32_t counts[256];
        for (uint32_t c = 0; c < numClusters; ++c) {
            const uint32_t length = readAnsHistogram(br, counts);
            if (length == 0 || length > (1u << cs.logAlpha)) throw Corrupt{};
            cs.tables[c].assign(2u << cs.logAlpha, 0);
            if (!initAliasTable(counts, length, cs.logAlpha, cs.tables[c].data())) throw Corrupt{};
        }
    }
}

SymReader makeSymReader(BitReader& br, const CodeSet& cs) {
    if (cs.lz77) throw Unsupported{};   // the GPU path keeps LZ77 streams
    SymReader r;
    r.cs = &cs;
    if (!cs.usePrefix) {
        r.state = br.read(32);
        r.logEntrySize = kAnsLogTabSize - cs.logAlpha;
        r.entryMask = (1u << r.logEntrySize) - 1;
    } else {
        r.state = kAnsSignature << 16;
    }
    return r;
}

uint32_t readSymbol(BitReader& br, SymReader& r, uint32_t cluster) {
    const uint32_t* table = r.cs->tables[cluster].data();
    if (r.cs->usePrefix) return readPrefixSymbol(br, table);
    const uint32_t res = r.state & (kAnsTabSize - 1);
    const uint32_t i = res >> r.logEntrySize;
    const uint32_t pos = res & r.entryMask;
    const uint32_t w0 = table[2 * i];
    const uint32_t w1 = table[2 * i + 1];
    const uint32_t cutoff = w0 & 0xFF;
    const uint32_t right = (w0 >> 8) & 0xFF;
    const uint32_t freq0 = w0 >> 16;
    const bool greater = pos >= cutoff;
    const uint32_t offsets1 = greater ? (w1 & 0xFFFF) : 0;
    const uint32_t freq1xor = greater ? (w1 >> 16) : 0;
    const uint32_t value = greater ? right : i;
    const uint32_t offset = offsets1 + pos;
    const uint32_t freq = freq0 ^ freq1xor;
    r.state = freq * (r.state >> kAnsLogTabSize) + offset;
    if (r.state < (1u << 16)) {
        r.state = (r.state << 16) | br.peek(16);
        br.consume(16);
    }
    return value;
}

uint32_t readHybridUintClustered(BitReader& br, SymReader& r, uint32_t cluster) {
    const uint32_t token = readSymbol(br, r, cluster);
    return readHybridUintConfig(br, r.cs->configs[cluster], token);
}

uint32_t readHybridUint(BitReader& br, SymReader& r, uint32_t ctx) { return readHybridUintClustered(br, r, r.cs->ctxMap[ctx]); }

bool checkFinalState(const SymReader& r) { return r.cs->usePrefix || r.state == (kAnsSignature << 16); }

void inverseMtf(std::vector<uint32_t>& map) {
    uint32_t mtf[256];
    for (uint32_t i = 0; i < 256; ++i) mtf[i] = i;
    for (uint32_t& m : map) {
        const uint32_t index = m;
        const uint32_t value = mtf[index];
        m = value;
        for (uint32_t j = index; j > 0; --j) mtf[j] = mtf[j - 1];
        mtf[0] = value;
    }
}

std::unique_ptr<CodeSet> decodeHistograms(BitReader& br, uint32_t numContexts) {
    auto cs = std::make_unique<CodeSet>();
    cs->lz77 = br.read(1) != 0;
    if (cs->lz77) {
        (void)readU32(br, val(224), val(512), val(4096), bitsOffset(15, 8));
        (void)readU32(br, val(3), val(4), bitsOffset(2, 5), bitsOffset(8, 9));
        ++numContexts;
        (void)decodeUintConfig(br, 8);
        throw Unsupported{};   // LZ77 streams stay on the GPU
    }
    cs->numContexts = numContexts;
    cs->ctxMap.assign(numContexts, 0);
    uint32_t numClusters = 1;
    if (numContexts > 1) {
        const bool isSimple = br.read(1) != 0;
        if (isSimple) {
            const uint32_t bitsPerEntry = br.read(2);
            if (bitsPerEntry != 0) {
                for (uint32_t i = 0; i < numContexts; ++i) cs->ctxMap[i] = br.read(bitsPerEntry);
            }
        } else {
            const bool useMtf = br.read(1) != 0;
            CodeSet inner;
            inner.lz77 = br.read(1) != 0;
            if (inner.lz77) throw Unsupported{};
            inner.numContexts = 1;
            inner.ctxMap.assign(1, 0);
            decodeCodeTail(br, inner, 1);
            SymReader r = makeSymReader(br, inner);
            uint32_t maxSym = 0;
            for (uint32_t i = 0; i < numContexts; ++i) {
                const uint32_t sym = readHybridUint(br, r, 0);
                maxSym = std::max(maxSym, sym);
                cs->ctxMap[i] = sym;
            }
            if (maxSym >= 256 || !checkFinalState(r)) throw Corrupt{};
            if (useMtf) inverseMtf(cs->ctxMap);
        }
        uint32_t maxEntry = 0;
        for (uint32_t v : cs->ctxMap) maxEntry = std::max(maxEntry, v);
        numClusters = maxEntry + 1;
        std::vector<uint8_t> seen(numClusters, 0);
        for (uint32_t v : cs->ctxMap) seen[v] = 1;
        for (uint8_t s : seen) if (s == 0) throw Corrupt{};
    }
    decodeCodeTail(br, *cs, numClusters);
    return cs;
}

// --- MA trees -------------------------------------------------------------------------

struct Node {
    int32_t  prop = -1;   // -1: leaf
    int32_t  split = 0;
    uint32_t lchild = 0, rchild = 0;
    uint32_t predictor = 0;
    int32_t  offset = 0;
    int32_t  multiplier = 1;
    uint32_t ctx = 0;
};

struct Tree {
    std::vector<Node>        nodes;
    uint32_t                 numLeaves = 0;
    bool                     usesWp = false;
    std::unique_ptr<CodeSet> codeSet;
};

void decodeTree(BitReader& br, Tree& t) {
    auto treeCode = decodeHistograms(br, 6);
    SymReader r = makeSymReader(br, *treeCode);
    uint32_t toDecode = 1;
    uint32_t leafId = 0;
    while (toDecode > 0) {
        if (t.nodes.size() >= kMaxTreeNodes) throw Corrupt{};
        --toDecode;
        const uint32_t prop1 = readHybridUint(br, r, 1);
        if (prop1 > 256) throw Corrupt{};
        Node n;
        if (prop1 == 0) {
            n.predictor = readHybridUint(br, r, 2);
            if (n.predictor >= 14) throw Corrupt{};
            n.offset = unpackSigned(readHybridUint(br, r, 3));
            const uint32_t mulLog = readHybridUint(br, r, 4);
            if (mulLog >= 31) throw Corrupt{};
            const uint32_t mulBits = readHybridUint(br, r, 5);
            if (mulBits >= (1u << (31u - mulLog)) - 1u) throw Corrupt{};
            n.prop = -1;
            n.multiplier = static_cast<int32_t>((mulBits + 1) << mulLog);
            n.ctx = leafId;
            if (n.predictor == kPredWeighted) t.usesWp = true;
            ++leafId;
            t.nodes.push_back(n);
            continue;
        }
        n.prop = static_cast<int32_t>(prop1 - 1);
        n.split = unpackSigned(readHybridUint(br, r, 0));
        n.lchild = static_cast<uint32_t>(t.nodes.size()) + toDecode + 1;
        n.rchild = static_cast<uint32_t>(t.nodes.size()) + toDecode + 2;
        if (prop1 - 1 == kWpProp) t.usesWp = true;
        t.nodes.push_back(n);
        toDecode += 2;
    }
    t.numLeaves = leafId;
    if (!checkFinalState(r)) throw Corrupt{};
}

// --- the weighted predictor: codestream/WeightedPredictor.h ----------------------------

using WpHeader = codestream::wp::Header;
using WpState = codestream::wp::State;
using codestream::wp::clampedGradient;

WpHeader readWpHeader(BitReader& br) {
    WpHeader h;
    if (br.read(1) != 0) return h;
    h.p1C = static_cast<int32_t>(br.read(5));
    h.p2C = static_cast<int32_t>(br.read(5));
    h.p3Ca = static_cast<int32_t>(br.read(5));
    h.p3Cb = static_cast<int32_t>(br.read(5));
    h.p3Cc = static_cast<int32_t>(br.read(5));
    h.p3Cd = static_cast<int32_t>(br.read(5));
    h.p3Ce = static_cast<int32_t>(br.read(5));
    h.w0 = br.read(4);
    h.w1 = br.read(4);
    h.w2 = br.read(4);
    h.w3 = br.read(4);
    return h;
}

int64_t selectPredictor(int64_t a, int64_t b, int64_t c) {
    const int64_t p = a + b - c;
    const int64_t pa = std::llabs(p - a);
    const int64_t pb = std::llabs(p - b);
    return pa < pb ? a : b;
}

int64_t predictOne(uint32_t p, int64_t left, int64_t top, int64_t toptop, int64_t topleft, int64_t topright, int64_t leftleft,
                   int64_t toprightright, int64_t wpPred) {
    switch (p) {
    case 0: return 0;
    case 1: return left;
    case 2: return top;
    case 3: return (left + top) / 2;
    case 4: return selectPredictor(left, top, topleft);
    case 5: return clampedGradient(left, top, topleft);
    case 6: return wpPred;
    case 7: return topright;
    case 8: return topleft;
    case 9: return leftleft;
    case 10: return (left + topleft) / 2;
    case 11: return (topleft + top) / 2;
    case 12: return (top + topright) / 2;
    case 13: return (6 * top - 2 * toptop + 7 * left + 1 * leftleft + 1 * toprightright + 3 * topright + 8) / 16;
    default: return 0;
    }
}

// --- channels -------------------------------------------------------------------------

struct Channel {
    uint32_t             w = 0, h = 0;
    int32_t              hshift = 0, vshift = 0;
    std::vector<int32_t> data;
    int32_t px(uint32_t x, uint32_t y) const { return data[y * w + x]; }
};

/// DecodeModularChannelMAANS, exactly as modular_channel.slangh (64-bit variant).
void decodeChannel(BitReader& br, SymReader& r, const Tree& tree, const WpHeader& wpHeader, std::vector<Channel>& channels, uint32_t chan,
                   uint32_t streamId) {
    Channel& ch = channels[chan];
    if (ch.w == 0 || ch.h == 0) return;
    std::vector<uint32_t> refs;
    for (int j = static_cast<int>(chan) - 1; j >= 0; --j) {
        const Channel& cj = channels[static_cast<size_t>(j)];
        if (cj.w == ch.w && cj.h == ch.h && cj.hshift == ch.hshift && cj.vshift == ch.vshift) refs.push_back(static_cast<uint32_t>(j));
    }
    WpState wp;
    if (tree.usesWp) wp.init(wpHeader, ch.w);
    const Node* nodes = tree.nodes.data();
    int32_t* out = ch.data.data();
    for (uint32_t y = 0; y < ch.h; ++y) {
        int32_t prevGrad = 0;
        const int32_t* rowUp = y > 0 ? out + (y - 1) * ch.w : nullptr;
        const int32_t* rowUp2 = y > 1 ? out + (y - 2) * ch.w : nullptr;
        int32_t* row = out + y * ch.w;
        for (uint32_t x = 0; x < ch.w; ++x) {
            const int32_t left = x > 0 ? row[x - 1] : (y > 0 ? rowUp[x] : 0);
            const int32_t top = y > 0 ? rowUp[x] : left;
            const int32_t topleft = (x > 0 && y > 0) ? rowUp[x - 1] : left;
            const int32_t topright = (x + 1 < ch.w && y > 0) ? rowUp[x + 1] : top;
            const int32_t leftleft = x > 1 ? row[x - 2] : left;
            const int32_t toptop = y > 1 ? rowUp2[x] : top;
            const int32_t toprightright = (x + 2 < ch.w && y > 0) ? rowUp[x + 2] : topright;
            int64_t wpPred = 0;
            int32_t wpProp = 0;
            if (tree.usesWp) wpPred = wp.predict(x, y, top, left, topright, topleft, toptop, wpProp);
            uint32_t node = 0;
            for (;;) {
                const Node& n = nodes[node];
                if (n.prop < 0) break;
                int32_t v = 0;
                switch (static_cast<uint32_t>(n.prop)) {
                case 0: v = static_cast<int32_t>(chan); break;
                case 1: v = static_cast<int32_t>(streamId); break;
                case 2: v = static_cast<int32_t>(y); break;
                case 3: v = static_cast<int32_t>(x); break;
                case 4: v = static_cast<int32_t>(std::llabs(static_cast<int64_t>(top))); break;
                case 5: v = static_cast<int32_t>(std::llabs(static_cast<int64_t>(left))); break;
                case 6: v = top; break;
                case 7: v = left; break;
                case 8: v = left - prevGrad; break;
                case 9: v = left + top - topleft; break;
                case 10: v = left - topleft; break;
                case 11: v = topleft - top; break;
                case 12: v = top - topright; break;
                case 13: v = top - toptop; break;
                case 14: v = left - leftleft; break;
                case 15: v = wpProp; break;
                default: {
                    const uint32_t k = (static_cast<uint32_t>(n.prop) - kNumNonrefProperties) / 4;
                    if (k < refs.size()) {
                        const Channel& cj = channels[refs[k]];
                        const int32_t rv = cj.px(x, y);
                        const uint32_t which = (static_cast<uint32_t>(n.prop) - kNumNonrefProperties) % 4;
                        if (which == 0) v = static_cast<int32_t>(std::llabs(static_cast<int64_t>(rv)));
                        else if (which == 1) v = rv;
                        else {
                            const int32_t vleft = x > 0 ? cj.px(x - 1, y) : 0;
                            const int32_t vtop = y > 0 ? cj.px(x, y - 1) : vleft;
                            const int32_t vtopleft = (x > 0 && y > 0) ? cj.px(x - 1, y - 1) : vleft;
                            const int64_t vpred = clampedGradient(vleft, vtop, vtopleft);
                            v = which == 2 ? static_cast<int32_t>(std::llabs(static_cast<int64_t>(rv) - vpred)) : static_cast<int32_t>(static_cast<int64_t>(rv) - vpred);
                        }
                    }
                }
                }
                node = v > n.split ? n.lchild : n.rchild;
            }
            prevGrad = left + top - topleft;
            const Node& leaf = nodes[node];
            const int64_t guess = static_cast<int64_t>(leaf.offset) + predictOne(leaf.predictor, left, top, toptop, topleft, topright, leftleft, toprightright, wpPred);
            const uint32_t token = readHybridUint(br, r, leaf.ctx);
            const int64_t value = static_cast<int64_t>(unpackSigned(token)) * static_cast<int64_t>(leaf.multiplier) + guess;
            row[x] = static_cast<int32_t>(value);
            if (tree.usesWp) wp.updateErrors(row[x], x, y);
        }
    }
}

/// ModularDecode of one stream over `channels`: GroupHeader (no transforms here), the
/// tree (local or the frame's global one), every channel.
void decodeModularStream(BitReader& br, std::vector<Channel>& channels, uint32_t streamId, const Tree* global) {
    if (channels.empty()) return;
    const bool useGlobalTree = br.read(1) != 0;
    const WpHeader wp = readWpHeader(br);
    const uint32_t numTransforms = readU32(br, val(0), val(1), bitsOffset(4, 2), bitsOffset(8, 18));
    if (numTransforms != 0) throw Unsupported{};
    bool any = false;
    for (const Channel& c : channels) any = any || (c.w != 0 && c.h != 0);
    if (!any) return;
    Tree local;
    const Tree* tree = global;
    if (!useGlobalTree) {
        decodeTree(br, local);
        local.codeSet = decodeHistograms(br, (static_cast<uint32_t>(local.nodes.size()) + 1) / 2);
        tree = &local;
    } else if (global == nullptr) {
        throw Corrupt{};
    }
    SymReader r = makeSymReader(br, *tree->codeSet);
    for (uint32_t i = 0; i < channels.size(); ++i) decodeChannel(br, r, *tree, wp, channels, i, streamId);
    if (!checkFinalState(r)) throw Corrupt{};
}

const uint32_t kNumStrategies = 27;
const uint32_t kCoveredX[27] = {1, 1, 1, 1, 2, 4, 1, 2, 1, 4, 2, 4, 1, 1, 1, 1, 1, 1, 8, 4, 8, 16, 8, 16, 32, 16, 32};
const uint32_t kCoveredY[27] = {1, 1, 1, 1, 2, 4, 2, 1, 4, 1, 4, 2, 1, 1, 1, 1, 1, 1, 8, 8, 4, 16, 16, 8, 32, 32, 16};

/// The raster walk of DecodeAcMetadata: every block not covered yet starts the next
/// varblock; checks the coverage as dcgroup.slangh does.
void walkStrategies(const DcGroupGeom& geo, const Channel& cAcs, const Channel& cEpf, uint32_t count, DcGroupData& r) {
    const bool is444 = (geo.hshift[0] | geo.hshift[1] | geo.hshift[2] | geo.vshift[0] | geo.vshift[1] | geo.vshift[2]) == 0;
    std::vector<uint8_t> covered(size_t{geo.w} * geo.h, 0);
    uint32_t num = 0;
    for (uint32_t iy = 0; iy < geo.h; ++iy) {
        const uint32_t y = geo.y0 + iy;
        for (uint32_t ix = 0; ix < geo.w; ++ix) {
            const uint32_t x = geo.x0 + ix;
            const int32_t sharpness = cEpf.px(ix, iy);
            if (sharpness < 0 || sharpness >= 8) throw Corrupt{};
            if (covered[iy * geo.w + ix] != 0) continue;
            if (num >= count) throw Corrupt{};
            const int32_t raw = cAcs.px(num, 0);
            const int32_t qfv = cAcs.px(num, 1);
            if (raw < 0 || raw >= static_cast<int32_t>(kNumStrategies)) throw Corrupt{};
            const uint32_t strategy = static_cast<uint32_t>(raw);
            r.usedAcs |= 1u << strategy;
            const uint32_t cx = kCoveredX[strategy], cy = kCoveredY[strategy];
            if ((cx > 1 || cy > 1) && !is444) throw Corrupt{};
            const uint32_t nextXGroup = (x / geo.groupBlocks + 1) * geo.groupBlocks;
            const uint32_t nextYGroup = (y / geo.groupBlocks + 1) * geo.groupBlocks;
            if (x + cx > nextXGroup || x + cx > geo.x0 + geo.w || y + cy > nextYGroup || y + cy > geo.y0 + geo.h) throw Corrupt{};
            for (uint32_t by = 0; by < cy; ++by) {
                for (uint32_t bx = 0; bx < cx; ++bx) {
                    uint8_t& c = covered[(iy + by) * geo.w + ix + bx];
                    if (c != 0) throw Corrupt{};
                    c = 1;
                }
            }
            r.varblocks.push_back(static_cast<int32_t>(y * geo.blocksX + x));
            r.varblocks.push_back(static_cast<int32_t>(strategy));
            r.varblocks.push_back(1 + std::clamp(qfv, 0, 255));
            ++num;
        }
    }
}

Channel makeChannel(uint32_t w, uint32_t h, int32_t hshift, int32_t vshift) {
    Channel c;
    c.w = w;
    c.h = h;
    c.hshift = hshift;
    c.vshift = vshift;
    c.data.assign(size_t{w} * h, 0);
    return c;
}

}   // namespace

Result<bool> decodeDcGroups(std::span<const uint8_t> codestream, const ParsedFrame& frame, size_t globalTreeBit,
                            std::span<const DcGroupGeom> geoms, std::vector<DcGroupData>& out) {
    std::vector<uint32_t> words((codestream.size() + 3) / 4 + 2, 0);
    std::memcpy(words.data(), codestream.data(), codestream.size());
    const bool useDcFrame = (frame.header.flags & codestream::FrameHeader::kUseDcFrame) != 0;
    try {
        // The global tree (DcGlobal's modular part), if the frame has one.
        Tree global;
        bool hasGlobal = false;
        {
            BitReader br{words.data(), words.size(), globalTreeBit};
            if (br.read(1) != 0) {
                decodeTree(br, global);
                global.codeSet = decodeHistograms(br, (static_cast<uint32_t>(global.nodes.size()) + 1) / 2);
                hasGlobal = true;
            }
        }
        const uint32_t numDcGroups = static_cast<uint32_t>(geoms.size());
        out.assign(numDcGroups, DcGroupData{});
        // The groups are independent chains: one task each (the global tree is read only).
        enum class Outcome { Ok, Unsupported, Corrupt, Overrun };
        const auto decodeOne = [&](uint32_t g) -> Outcome {
            try {
                const DcGroupGeom& geo = geoms[g];
                const codestream::Section& s = frame.sections[1 + g];
                BitReader br{words.data(), words.size(), s.byteOffset * 8};
                DcGroupData& r = out[g];
                static const uint32_t phys[3] = {1, 0, 2};
                std::vector<Channel> dcCh;
                if (!useDcFrame) {
                    r.extraPrecision = br.read(2);
                    for (uint32_t i = 0; i < 3; ++i) {
                        const uint32_t c = phys[i];
                        dcCh.push_back(makeChannel(geo.w >> geo.hshift[c], geo.h >> geo.vshift[c], 0, 0));
                    }
                    decodeModularStream(br, dcCh, 1 + g, hasGlobal ? &global : nullptr);
                }
                // AcMetadata.
                const uint32_t upperBound = geo.w * geo.h;
                r.count = br.read(ceilLog2Nonzero(upperBound)) + 1;
                const uint32_t tw = (geo.w + 7) >> 3, th = (geo.h + 7) >> 3;
                std::vector<Channel> meta;
                meta.push_back(makeChannel(tw, th, 3, 3));
                meta.push_back(makeChannel(tw, th, 3, 3));
                meta.push_back(makeChannel(r.count, 2, 0, 0));
                meta.push_back(makeChannel(geo.w, geo.h, 0, 0));
                decodeModularStream(br, meta, 1 + 2 * numDcGroups + g, hasGlobal ? &global : nullptr);
                if (br.pos > (s.byteOffset + s.byteSize) * 8) return Outcome::Overrun;
                walkStrategies(geo, meta[2], meta[3], r.count, r);
                for (const Channel& c : dcCh) r.words.insert(r.words.end(), c.data.begin(), c.data.end());
                for (const Channel& c : meta) r.words.insert(r.words.end(), c.data.begin(), c.data.end());
                return Outcome::Ok;
            } catch (const Unsupported&) {
                return Outcome::Unsupported;
            } catch (const Corrupt&) {
                return Outcome::Corrupt;
            }
        };
        std::vector<std::future<Outcome>> tasks;
        for (uint32_t g = 1; g < numDcGroups; ++g) tasks.push_back(std::async(std::launch::async, decodeOne, g));
        std::vector<Outcome> outcomes(numDcGroups, Outcome::Ok);
        outcomes[0] = decodeOne(0);
        for (uint32_t g = 1; g < numDcGroups; ++g) outcomes[g] = tasks[g - 1].get();
        for (Outcome o : outcomes) {
            if (o == Outcome::Unsupported) { out.clear(); return false; }
            if (o == Outcome::Overrun) return Error(ErrorCode::Corrupt, "DC group section overrun");
            if (o == Outcome::Corrupt) return Error(ErrorCode::Corrupt, "the codestream violates the specification in a DC group (CPU path)");
        }
    } catch (const Unsupported&) {
        out.clear();
        return false;
    } catch (const Corrupt&) {
        return Error(ErrorCode::Corrupt, "the codestream violates the specification in a DC group (CPU path)");
    }
    return true;
}

}   // namespace sjxl::dec::cpu
