// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/ComputeKernel.h"

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
    if (SLANG_FAILED(library.device().rhi()->createComputePipeline(desc, kernel.pipeline_.writeRef()))) {
        return Error::make(ErrorCode::ShaderFailure, "cannot make a pipeline for {}", kernel.name_);
    }
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
