// Copyright (c) 2026 jesus luque.
#include "sjxl/gpu/algo/PrefixSum.h"

#include "sjxl/gpu/CommandBatch.h"
#include "sjxl/gpu/Device.h"
#include "sjxl/gpu/ShaderLibrary.h"

namespace sjxl::gpu {
namespace {

void setParams(rhi::ShaderCursor cursor, uint32_t count, uint32_t chunks) {
    cursor["params"]["count"].setData(count);
    cursor["params"]["chunkSize"].setData(PrefixSum::chunkFor(count));
    cursor["params"]["chunkCount"].setData(chunks);
}

}   // namespace

Result<PrefixSum> PrefixSum::create(ShaderLibrary& library) {
    PrefixSum sum;
    sum.device_ = &library.device();
    SJXL_ASSIGN(sum.totals_, ComputeKernel::create(library, "sjxl/algo/prefix_chunk_totals", "prefixChunkTotals"));
    SJXL_ASSIGN(sum.starts_, ComputeKernel::create(library, "sjxl/algo/prefix_chunk_starts", "prefixChunkStarts"));
    SJXL_ASSIGN(sum.local_, ComputeKernel::create(library, "sjxl/algo/prefix_local", "prefixLocal"));
    return sum;
}

Result<void> PrefixSum::reserve(uint32_t chunks) {
    if (chunks <= capacity_) {
        return ok();
    }
    uint32_t grown = capacity_ == 0 ? 16 : capacity_;
    while (grown < chunks) {
        grown *= 2;
    }
    SJXL_ASSIGN(chunkTotals_, Buffer::ofType<uint32_t>(*device_, grown, "prefix.chunkTotals"));
    SJXL_ASSIGN(chunkStarts_, Buffer::ofType<uint32_t>(*device_, grown, "prefix.chunkStarts"));
    capacity_ = grown;
    return ok();
}

Result<void> PrefixSum::apply(CommandBatch& batch, const Buffer& input, const Buffer& output, const Buffer& total,
                              uint32_t count) {
    if (input.rhi() == output.rhi()) {
        return Error(ErrorCode::InvalidArgument, "prefix sum input and output must differ");
    }
    if (input.count() < count || output.count() < count || total.count() < 1) {
        return Error(ErrorCode::InvalidArgument, "prefix sum buffers are too small");
    }
    const uint32_t chunk = chunkFor(count);
    const uint32_t chunks = count == 0 ? 1 : (count + chunk - 1) / chunk;
    SJXL_TRY(reserve(chunks));

    totals_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        setParams(cursor, count, chunks);
    });
    starts_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["total"].setBinding(total.rhi());
        setParams(cursor, count, chunks);
    });
    local_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["output"].setBinding(output.rhi());
        setParams(cursor, count, chunks);
    });
    return ok();
}

}   // namespace sjxl::gpu
