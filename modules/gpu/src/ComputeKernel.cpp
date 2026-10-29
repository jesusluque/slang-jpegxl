// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/ComputeKernel.h"
#include <chrono>
#include "sjxl/core/Platform.h"
#include "sjxl/core/Log.h"

#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/Device.h"

namespace sjxl::gpu {

Result<ComputeKernel> ComputeKernel::create(ShaderLibrary& library, const std::string& module, const std::string& entry,
                                            const std::vector<LinkConstant>& constants) {
    auto program = library.load(module, {entry}, constants);
    if (!program) {
        return std::move(program).error();
    }
    ComputeKernel kernel;
    kernel.name_ = module + ":" + entry;
    kernel.program_ = *program;
    rhi::ComputePipelineDesc desc;
    desc.program = kernel.program_->program.get();
    const auto t0 = std::chrono::steady_clock::now();
    if (SLANG_FAILED(library.device().rhi()->createComputePipeline(desc, kernel.pipeline_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a pipeline for {}", kernel.name_);
    }
    // The backend compile (Metal's AGX compiler) can take seconds on the biggest single-thread
    // kernels; SJXL_DEBUG lists the slow ones so they get split before they become hangs.
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 500.0 && !platform::env("SJXL_DEBUG").empty()) log::info("shaders: pipeline {} took {:.0f} ms", kernel.name_, ms);
    return kernel;
}

void ComputeKernel::dispatch(CommandBatch& batch, std::array<uint32_t, 3> threads, const Bind& bind) const {
    const auto& group = program_->threadGroup;
    dispatchGroups(batch, {groupsFor(threads[0], group[0]), groupsFor(threads[1], group[1]), groupsFor(threads[2], group[2])},
                   bind);
}

void ComputeKernel::dispatchGroups(CommandBatch& batch, std::array<uint32_t, 3> groups, const Bind& bind) const {
    if (groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return;
    }
    static const bool traceDispatch = !platform::env("SJXL_DEBUG_DISPATCH").empty();
    if (traceDispatch) log::info("dispatch {}", name_);
    rhi::IComputePassEncoder* pass = batch.encoder()->beginComputePass();
    rhi::IShaderObject* root = pass->bindPipeline(pipeline_.get());
    if (bind) {
        bind(rhi::ShaderCursor(root));
    }
    pass->dispatchCompute(groups[0], groups[1], groups[2]);
    pass->end();
    // slang-rhi's resources are untracked (Metal MTLHazardTrackingModeUntracked,
    // and the same contract elsewhere): what one dispatch wrote is visible to
    // the next only after a barrier. Every kernel here feeds the next, so the
    // barrier is part of a dispatch.
    batch.encoder()->globalBarrier();
    batch.dirty_ = true;
}

}   // namespace sjxl::gpu
