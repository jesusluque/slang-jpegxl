// Copyright (c) 2026 jesus luque.
#pragma once

#include <slang-rhi.h>

#include "sjxl/core/Result.h"

namespace sjxl::gpu {

class Device;
class Buffer;

/// Commands recorded together and submitted once: one per pipeline stage, not
/// one per dispatch.
class CommandBatch {
public:
    explicit CommandBatch(Device& device);
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch();

    [[nodiscard]] Device& device() noexcept { return device_; }
    [[nodiscard]] rhi::ICommandEncoder* encoder() noexcept { return encoder_.get(); }
    /// For callers that record straight into `encoder()`.
    void markDirty() noexcept { dirty_ = true; }

    /// Records a fill with zeros of the whole buffer.
    void clear(const Buffer& buffer);
    /// Records a copy of `bytes` from `from` at `fromOffset` into `to` at `toOffset`.
    void copy(const Buffer& from, uint64_t fromOffset, const Buffer& to, uint64_t toOffset, uint64_t bytes);

    /// Submits what was recorded and starts a fresh encoder. `wait` blocks
    /// until the device has run it.
    [[nodiscard]] Result<void> submit(bool wait = false);

private:
    Device&                           device_;
    rhi::ComPtr<rhi::ICommandEncoder> encoder_;
    bool                              dirty_ = false;
    friend class ComputeKernel;
};

}   // namespace sjxl::gpu
