// Copyright (c) 2026 jesus luque.
//
// LSB-first bit writer with libjxl's byte/bit order: bits go into increasing
// byte addresses, least-significant bit first. For headers and the TOC only;
// the frame's body is emitted by kernels.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sjxl::codestream {

class BitWriter {
public:
    void write(size_t nBits, uint64_t bits) {
        assert(nBits <= 56);
        assert(nBits == 64 || (bits >> nBits) == 0);
        while (nBits > 0) {
            if (bitPos_ == 0) {
                bytes_.push_back(0);
            }
            const size_t take = nBits < (8 - bitPos_) ? nBits : (8 - bitPos_);
            const uint8_t mask = static_cast<uint8_t>((1u << take) - 1);
            bytes_.back() |= static_cast<uint8_t>((bits & mask) << bitPos_);
            bits >>= take;
            nBits -= take;
            bitPos_ = (bitPos_ + take) & 7;
        }
    }
    void writeBool(bool value) { write(1, value ? 1 : 0); }
    void zeroPadToByte() { bitPos_ = 0; }
    [[nodiscard]] bool byteAligned() const { return bitPos_ == 0; }
    [[nodiscard]] size_t bitsWritten() const {
        return bytes_.empty() ? 0 : (bytes_.size() - 1) * 8 + (bitPos_ == 0 ? 8 : bitPos_);
    }
    [[nodiscard]] const std::vector<uint8_t>& bytes() const { return bytes_; }
    [[nodiscard]] std::vector<uint8_t> take() { bitPos_ = 0; return std::move(bytes_); }

private:
    std::vector<uint8_t> bytes_;
    size_t               bitPos_ = 0;
};

/// One of the four distributions a 2-bit U32 selector picks (libjxl U32Distr).
struct Distr {
    bool     direct = false;
    uint32_t valueOrOffset = 0;
    uint32_t extraBits = 0;
};
constexpr Distr Val(uint32_t v) { return Distr{true, v, 0}; }
constexpr Distr BitsOffset(uint32_t bits, uint32_t offset) { return Distr{false, offset, bits}; }
constexpr Distr Bits(uint32_t bits) { return BitsOffset(bits, 0); }

struct U32Enc {
    Distr d[4];
};

/// Writes `value` with the cheapest distribution that holds it.
void writeU32(BitWriter& w, const U32Enc& enc, uint32_t value);
/// libjxl U64Coder.
void writeU64(BitWriter& w, uint64_t value);
void writeF16(BitWriter& w, float value);

}   // namespace sjxl::codestream
