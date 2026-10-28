// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/CommandBatch.h"

#include "sjxl/gpu/Buffer.h"
#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {

CommandBatch::CommandBatch(Device& device) : device_(device), encoder_(device.queue()->createCommandEncoder()) {}

CommandBatch::~CommandBatch() {
    if (dirty_) {
        (void)submit(false);
    }
}

void CommandBatch::clear(const Buffer& buffer) {
    encoder_->clearBuffer(buffer.rhi());
    dirty_ = true;
}

void CommandBatch::copy(const Buffer& from, uint64_t fromOffset, const Buffer& to, uint64_t toOffset, uint64_t bytes) {
    encoder_->copyBuffer(to.rhi(), toOffset, from.rhi(), fromOffset, bytes);
    dirty_ = true;
}

Result<void> CommandBatch::submit(bool wait) {
    // A command buffer that failed before this one -- the device out of
    // memory -- is reported by the first submit or wait after it.
    Result<void> earlier = device_.takeQueueError();
    if (dirty_) {
        const SlangResult submitted = device_.queue()->submit(encoder_->finish());
        encoder_ = device_.queue()->createCommandEncoder();
        dirty_ = false;
        SJXL_TRY(earlier);
        SJXL_TRY(Device::queueResult(submitted, "a command batch"));
    }
    SJXL_TRY(earlier);
    if (wait) {
        SJXL_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

}   // namespace sjxl::gpu
