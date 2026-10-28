// Copyright (c) 2026 jesus luque.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "sjxl/core/Result.h"

namespace sjxl::gpu {

class Device;

enum class Memory {
    Device,     ///< GPU-only. The default for everything a kernel writes.
    Upload,     ///< CPU writes, GPU reads.
    Readback,   ///< GPU writes (by copy), CPU reads.
};

struct BufferDesc {
    uint64_t    bytes = 0;
    /// Bytes per element for a structured buffer. Zero for raw bytes.
    uint32_t    elementBytes = 0;
    Memory      memory = Memory::Device;
    std::string label;
};

/// A device buffer. Owns its slang-rhi object; cheap to copy (ref-counted).
/// A buffer made without initial data starts at zero on every backend.
class Buffer {
public:
    Buffer() = default;

    [[nodiscard]] static Result<Buffer> create(Device& device, const BufferDesc& desc, const void* initial = nullptr);

    /// `count` elements of T, zeroed, on the device.
    template <typename T>
    [[nodiscard]] static Result<Buffer> ofType(Device& device, uint64_t count, std::string label,
                                               Memory memory = Memory::Device) {
        BufferDesc desc;
        desc.bytes = count * sizeof(T);
        desc.elementBytes = sizeof(T);
        desc.memory = memory;
        desc.label = std::move(label);
        return create(device, desc, nullptr);
    }

    template <typename T>
    [[nodiscard]] static Result<Buffer> fromSpan(Device& device, std::span<const T> data, std::string label,
                                                 Memory memory = Memory::Device) {
        BufferDesc desc;
        desc.bytes = data.size_bytes();
        desc.elementBytes = sizeof(T);
        desc.memory = memory;
        desc.label = std::move(label);
        return create(device, desc, data.data());
    }

    [[nodiscard]] bool valid() const noexcept { return buffer_ != nullptr; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] uint32_t elementBytes() const noexcept { return elementBytes_; }
    [[nodiscard]] uint64_t count() const noexcept { return elementBytes_ != 0 ? bytes_ / elementBytes_ : bytes_; }
    [[nodiscard]] rhi::IBuffer* rhi() const noexcept { return buffer_.get(); }

    /// Synchronous readback: for the final image, section sizes, and tests'
    /// metric counters.
    [[nodiscard]] Result<void> read(Device& device, uint64_t offset, uint64_t bytes, void* into) const;

    template <typename T>
    [[nodiscard]] Result<std::vector<T>> readAll(Device& device) const {
        std::vector<T> out(bytes_ / sizeof(T));
        SJXL_TRY(read(device, 0, out.size() * sizeof(T), out.data()));
        return out;
    }

    template <typename T>
    [[nodiscard]] Result<std::vector<T>> readRange(Device& device, uint64_t first, uint64_t count) const {
        std::vector<T> out(count);
        SJXL_TRY(read(device, first * sizeof(T), count * sizeof(T), out.data()));
        return out;
    }

    /// Synchronous upload into a device buffer.
    [[nodiscard]] Result<void> write(Device& device, uint64_t offset, uint64_t bytes, const void* from);

private:
    rhi::ComPtr<rhi::IBuffer> buffer_;
    uint64_t                  bytes_ = 0;
    uint32_t                  elementBytes_ = 0;
};

}   // namespace sjxl::gpu
