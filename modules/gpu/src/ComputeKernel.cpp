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
    kernel.device_ = library.device().rhi();
    return kernel;
}

Result<void> ComputeKernel::prepare() const {
    if (pipeline_ != nullptr) return ok();
    if (failed_ || program_ == nullptr) return Error::make(ErrorCode::ShaderFailure, "cannot make a pipeline for {}", name_);
    rhi::ComputePipelineDesc desc;
    desc.program = program_->program.get();
    const auto t0 = std::chrono::steady_clock::now();
    if (SLANG_FAILED(device_->createComputePipeline(desc, pipeline_.writeRef()))) {
        failed_ = true;
        return Error::make(ErrorCode::ShaderFailure, "cannot make a pipeline for {}", name_);
    }
    // The backend compile (Metal's AGX compiler) takes seconds on the biggest single-thread
    // kernels (it inlines everything they reach), so pipelines are built at first use and
    // SJXL_DEBUG lists the slow ones so they get split.
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > 500.0 && !platform::env("SJXL_DEBUG").empty()) log::info("shaders: pipeline {} took {:.0f} ms", name_, ms);
    return ok();
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
    if (const Result<void> made = prepare(); !made) {
        log::error("{}: {}; the dispatch is skipped", name_, made.error().message());
        return;
    }
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
