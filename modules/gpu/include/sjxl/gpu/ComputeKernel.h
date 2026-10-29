// Copyright (c) 2026 jesus luque.
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "sjxl/core/Result.h"
#include "sjxl/gpu/ShaderLibrary.h"

namespace sjxl::gpu {

class CommandBatch;

/// One compute entry point, as a pipeline.
///
/// Bound by name through reflection -- `cursor["coeffs"].setBinding(buffer)` --
/// never by slot.
class ComputeKernel {
public:
    using Bind = std::function<void(rhi::ShaderCursor)>;

    [[nodiscard]] static Result<ComputeKernel> create(ShaderLibrary& library, const std::string& module,
                                                      const std::string& entry,
                                                      const std::vector<LinkConstant>& constants = {});

    ComputeKernel() = default;

    /// Queues the kernel over `threads` (in threads, not groups) into `batch`.
    /// The group count is threads / [numthreads], rounded up; the kernel
    /// bounds-checks.
    void dispatch(CommandBatch& batch, std::array<uint32_t, 3> threads, const Bind& bind) const;

    /// Queues exactly `groups` thread groups.
    void dispatchGroups(CommandBatch& batch, std::array<uint32_t, 3> groups, const Bind& bind) const;

    /// The thread group size from reflection (loads the program if it is not yet).
    [[nodiscard]] const std::array<uint32_t, 3>& threadGroup() const;
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] bool valid() const noexcept { return library_ != nullptr; }

    /// Loads the program and builds the pipeline now (the first dispatch does it
    /// otherwise): loading a module and compiling a big kernel's backend code take
    /// from milliseconds to seconds, so they are paid only by the kernels a run uses.
    [[nodiscard]] Result<void> prepare() const;

private:
    std::string                                name_, module_, entry_;
    std::vector<LinkConstant>                  constants_;
    ShaderLibrary*                             library_ = nullptr;
    mutable std::shared_ptr<const Program>     program_;
    mutable rhi::ComPtr<rhi::IComputePipeline> pipeline_;
    mutable bool                               failed_ = false;
};

[[nodiscard]] constexpr uint32_t groupsFor(uint32_t threads, uint32_t perGroup) noexcept {
    return perGroup == 0 ? 0 : (threads + perGroup - 1) / perGroup;
}

}   // namespace sjxl::gpu
