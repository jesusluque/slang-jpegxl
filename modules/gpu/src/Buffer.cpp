// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/Buffer.h"

#include <cstring>
#include <format>

#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {
namespace {

rhi::BufferDesc toRhi(const BufferDesc& desc) {
    rhi::BufferDesc out;
    out.size = desc.bytes;
    out.elementSize = desc.elementBytes;
    out.label = desc.label.empty() ? nullptr : desc.label.c_str();
    switch (desc.memory) {
    case Memory::Device:
        out.memoryType = rhi::MemoryType::DeviceLocal;
        out.usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess | rhi::BufferUsage::CopySource |
                    rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::UnorderedAccess;
        break;
    case Memory::Upload:
        out.memoryType = rhi::MemoryType::Upload;
        out.usage = rhi::BufferUsage::CopySource | rhi::BufferUsage::ShaderResource;
        out.defaultState = rhi::ResourceState::General;
        break;
    case Memory::Readback:
        out.memoryType = rhi::MemoryType::ReadBack;
        out.usage = rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::CopyDestination;
        break;
    }
    return out;
}

}   // namespace

Result<Buffer> Buffer::create(Device& device, const BufferDesc& desc, const void* initial) {
    if (desc.bytes == 0) {
        return Error::make(ErrorCode::InvalidArgument, "buffer '{}' of zero bytes", desc.label);
    }
    Buffer buffer;
    const rhi::BufferDesc rhiDesc = toRhi(desc);
    if (SLANG_FAILED(device.rhi()->createBuffer(rhiDesc, initial, buffer.buffer_.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate buffer '{}' ({} bytes)", desc.label, desc.bytes);
    }
    buffer.bytes_ = desc.bytes;
    buffer.elementBytes_ = desc.elementBytes;
    // A BUFFER WITHOUT DATA STARTS AT ZERO, on every backend. Metal hands out
    // new memory cleared; CUDA's and Vulkan's allocations hold whatever was
    // there before. Histograms and counters are made and then added to, so
    // the clear goes on the queue, which everything that uses the buffer
    // comes after.
    if (initial == nullptr && device.backend() != Backend::Metal &&
        (rhiDesc.usage & rhi::BufferUsage::CopyDestination) != rhi::BufferUsage::None) {
        rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
        encoder->clearBuffer(buffer.buffer_.get());
        SJXL_TRY(Device::queueResult(device.queue()->submit(encoder->finish()), "a new buffer's clear"));
    }
    return buffer;
}

Result<void> Buffer::read(Device& device, uint64_t offset, uint64_t bytes, void* into) const {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "read outside the buffer");
    }
    if (bytes == 0) {
        return ok();
    }
    // Waiting first is how a command buffer that failed on the way -- the
    // device out of memory -- is reported instead of read.
    device.waitIdle();
    SJXL_TRY(device.takeQueueError());
    if (const SlangResult read = device.rhi()->readBuffer(buffer_.get(), offset, bytes, into); SLANG_FAILED(read)) {
        return read == SLANG_E_OUT_OF_MEMORY
                   ? Error(ErrorCode::OutOfMemory, "the GPU ran out of memory reading a buffer back")
                   : Error(ErrorCode::DeviceFailure, "readBuffer failed");
    }
    return ok();
}

Result<void> Buffer::write(Device& device, uint64_t offset, uint64_t bytes, const void* from) {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "write outside the buffer");
    }
    if (bytes == 0) {
        return ok();
    }
    rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
    encoder->uploadBufferData(buffer_.get(), offset, bytes, const_cast<void*>(from));
    return Device::queueResult(device.queue()->submit(encoder->finish()), "an upload");
}

}   // namespace sjxl::gpu
