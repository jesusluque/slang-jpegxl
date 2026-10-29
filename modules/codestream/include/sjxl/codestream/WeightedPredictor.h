// Copyright (c) 2026 jesus luque.
//
// libjxl's weighted predictor (modular/encoding/context_predict.h
// weighted::State) with pixel_type_w = int64 arithmetic and uint32 error sums,
// as the kernels implement it (dec/modular_channel.slangh, enc/wp.slangh). The
// CPU paths of the decoder (DC groups) and the encoder (DC tokens) share it;
// both must stay bit-identical to the kernels, which the [dc-cpu] and
// [enc-cpu] tests check.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sjxl::codestream::wp {

constexpr int     kExtraBits = 3;
constexpr int64_t kRound = ((1 << kExtraBits) >> 1) - 1;

struct Header {
    int32_t  p1C = 16, p2C = 10, p3Ca = 7, p3Cb = 7, p3Cc = 7, p3Cd = 0, p3Ce = 0;
    uint32_t w0 = 0xd, w1 = 0xc, w2 = 0xc, w3 = 0xc;
};

inline const uint32_t kDivLookup[64] = {
    16777216, 8388608, 5592405, 4194304, 3355443, 2796202, 2396745, 2097152, 1864135, 1677721, 1525201, 1398101, 1290555,
    1198372, 1118481, 1048576, 986895,  932067,  883011,  838860,  798915,  762600,  729444,  699050,  671088,  645277,  621378,
    599186,  578524,  559240,  541200,  524288,  508400,  493447,  479349,  466033,  453438,  441505,  430185,  419430,  409200,
    399457,  390167,  381300,  372827,  364722,  356962,  349525,  342392,  335544,  328965,  322638,  316551,  310689,  305040,
    299593,  294337,  289262,  284359,  279620,  275036,  270600,  266305,  262144};

inline uint32_t floorLog2Nonzero(uint64_t x) {
    uint32_t n = 0;
    while (x > 1) { ++n; x >>= 1; }
    return n;
}

struct State {
    Header                header;
    uint32_t              xsize = 0;
    std::vector<uint32_t> predErrors[4];   // (xsize + 2) * 2 each
    std::vector<int32_t>  error;
    int64_t               prediction[4] = {};
    int64_t               pred = 0;

    void init(const Header& h, uint32_t w) {
        header = h;
        xsize = w;
        for (auto& pe : predErrors) pe.assign((w + 2) * 2, 0);
        error.assign((w + 2) * 2, 0);
    }

    uint32_t errorWeight(uint64_t x, uint32_t maxweight) const {
        int shift = static_cast<int>(floorLog2Nonzero(x + 1)) - 5;
        if (shift < 0) shift = 0;
        return 4 + static_cast<uint32_t>((static_cast<uint64_t>(maxweight) * kDivLookup[x >> shift]) >> shift);
    }

    static int64_t addBits(int64_t x) { return static_cast<int64_t>(static_cast<uint64_t>(x) << kExtraBits); }

    int64_t weightedAverage(const int64_t* p, uint32_t w[4]) const {
        uint32_t weightSum = 0;
        for (uint32_t i = 0; i < 4; ++i) weightSum += w[i];
        const uint32_t logWeight = floorLog2Nonzero(weightSum);
        weightSum = 0;
        for (uint32_t i = 0; i < 4; ++i) {
            w[i] >>= (logWeight - 4);
            weightSum += w[i];
        }
        int64_t sum = static_cast<int64_t>(weightSum >> 1) - 1;
        for (uint32_t i = 0; i < 4; ++i) sum += p[i] * static_cast<int64_t>(w[i]);
        return (sum * static_cast<int64_t>(kDivLookup[weightSum - 1])) >> 24;
    }

    /// The prediction for (x, y) from the neighbours (edge rules applied by the
    /// caller) and the WP property (largest absolute neighbouring error).
    int64_t predict(uint32_t x, uint32_t y, int64_t N, int64_t W, int64_t NE, int64_t NW, int64_t NN, int32_t& wpProp) {
        const uint32_t curRow = (y & 1) != 0 ? 0 : (xsize + 2);
        const uint32_t prevRow = (y & 1) != 0 ? (xsize + 2) : 0;
        const uint32_t posN = prevRow + x;
        const uint32_t posNE = x < xsize - 1 ? posN + 1 : posN;
        const uint32_t posNW = x > 0 ? posN - 1 : posN;
        uint32_t weights[4];
        const uint32_t hw[4] = {header.w0, header.w1, header.w2, header.w3};
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t sum = predErrors[i][posN] + predErrors[i][posNE] + predErrors[i][posNW];   // uint32, wrapping
            weights[i] = errorWeight(sum, hw[i]);
        }
        N = addBits(N);
        W = addBits(W);
        NE = addBits(NE);
        NW = addBits(NW);
        NN = addBits(NN);
        const int64_t teW = x == 0 ? 0 : error[curRow + x - 1];
        const int64_t teN = error[posN];
        const int64_t teNW = error[posNW];
        const int64_t sumWN = teN + teW;
        const int64_t teNE = error[posNE];
        {
            int64_t p = teW;
            if (std::llabs(teN) > std::llabs(p)) p = teN;
            if (std::llabs(teNW) > std::llabs(p)) p = teNW;
            if (std::llabs(teNE) > std::llabs(p)) p = teNE;
            wpProp = static_cast<int32_t>(p);
        }
        prediction[0] = W + NE - N;
        prediction[1] = N - (((sumWN + teNE) * header.p1C) >> 5);
        prediction[2] = W - (((sumWN + teNW) * header.p2C) >> 5);
        prediction[3] = N - ((teNW * header.p3Ca + teN * header.p3Cb + teNE * header.p3Cc + (NN - N) * header.p3Cd + (NW - W) * header.p3Ce) >> 5);
        pred = weightedAverage(prediction, weights);
        if (((teN ^ teW) | (teN ^ teNW)) > 0) return (pred + kRound) >> kExtraBits;
        const int64_t mx = std::max(W, std::max(NE, N));
        const int64_t mn = std::min(W, std::min(NE, N));
        pred = std::max(mn, std::min(mx, pred));
        return (pred + kRound) >> kExtraBits;
    }

    void updateErrors(int64_t val, uint32_t x, uint32_t y) {
        const uint32_t curRow = (y & 1) != 0 ? 0 : (xsize + 2);
        const uint32_t prevRow = (y & 1) != 0 ? (xsize + 2) : 0;
        val = addBits(val);
        error[curRow + x] = static_cast<int32_t>(pred - val);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t err = static_cast<uint32_t>((std::llabs(prediction[i] - val) + kRound) >> kExtraBits);
            predErrors[i][curRow + x] = err;
            predErrors[i][prevRow + x + 1] += err;
        }
    }
};

inline int64_t clampedGradient(int64_t n, int64_t w, int64_t l) {
    const int64_t m = std::min(n, w);
    const int64_t M = std::max(n, w);
    const int64_t grad = n + w - l;
    const int64_t gradClampM = (l < m) ? M : grad;
    return (l > M) ? m : gradClampM;
}

}   // namespace sjxl::codestream::wp
