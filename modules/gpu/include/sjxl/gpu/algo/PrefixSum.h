// Copyright (c) 2026 jesus luque.
//
// Exclusive prefix sum over uint32, on the GPU, in three dispatches.
// See shaders/sjxl/algo/prefix_chunk_totals.slang for why chunks.
#pragma once

#include <cstdint>

#include "sjxl/core/Result.h"
#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/ComputeKernel.h"

namespace sjxl::gpu {

class CommandBatch;
class ShaderLibrary;

class PrefixSum {
public:
    /// Elements per chunk for `count` elements: each chunk is one invocation
    /// walking its elements in order, so the chunk count is the parallelism.
    [[nodiscard]] static uint32_t chunkFor(uint32_t count) noexcept {
        uint32_t chunk = kMinChunk;
        while (chunk < kMaxChunk && count / chunk > kTargetChunks) {
            chunk *= 2;
        }
        return chunk;
    }
    static constexpr uint32_t kMinChunk = 256;
    static constexpr uint32_t kMaxChunk = 4096;
    static constexpr uint32_t kTargetChunks = 2048;

    [[nodiscard]] static Result<PrefixSum> create(ShaderLibrary& library);

    /// output[i] = input[0] + ... + input[i-1]; total[0] = the sum of all.
    /// `input` and `output` must be different buffers; `total` holds at least
    /// one uint. Queued into `batch`, not waited for.
    [[nodiscard]] Result<void> apply(CommandBatch& batch, const Buffer& input, const Buffer& output,
                                     const Buffer& total, uint32_t count);

private:
    [[nodiscard]] Result<void> reserve(uint32_t chunks);

    Device*       device_ = nullptr;
    ComputeKernel totals_;
    ComputeKernel starts_;
    ComputeKernel local_;
    Buffer        chunkTotals_;
    Buffer        chunkStarts_;
    uint32_t      capacity_ = 0;
};

}   // namespace sjxl::gpu
