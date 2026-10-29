// Copyright (c) 2026 jesus luque.
//
// Per-stage timing of a pipeline: with SJXL_TIMING set, every mark submits the
// batch, waits for the GPU and records the time since the previous mark; the
// report lists the stages. Without the variable, marks cost nothing and the
// batch stays unsynchronised.
#pragma once

#include <chrono>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "sjxl/core/Log.h"
#include "sjxl/core/Platform.h"
#include "sjxl/gpu/CommandBatch.h"

namespace sjxl::gpu {

class StageTimer {
public:
    explicit StageTimer(CommandBatch& batch, const char* pipeline)
        : batch_(&batch), pipeline_(pipeline), enabled_(!platform::env("SJXL_TIMING").empty()), last_(std::chrono::steady_clock::now()) {}

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// The work queued since the previous mark belongs to `stage`.
    void mark(const char* stage) {
        if (!enabled_) return;
        (void)batch_->submit(true);
        const auto now = std::chrono::steady_clock::now();
        stages_.emplace_back(stage, std::chrono::duration<double, std::milli>(now - last_).count());
        last_ = now;
    }

    void report() const {
        if (!enabled_) return;
        double total = 0.0;
        for (const auto& [name, ms] : stages_) total += ms;
        std::string lines;
        for (const auto& [name, ms] : stages_) lines += std::format("\n  {:<28} {:8.1f} ms  {:5.1f} %", name, ms, total > 0.0 ? 100.0 * ms / total : 0.0);
        log::info("timing {}: {:.1f} ms{}", pipeline_, total, lines);
    }

private:
    CommandBatch*                                   batch_;
    const char*                                     pipeline_;
    bool                                            enabled_;
    std::chrono::steady_clock::time_point           last_;
    std::vector<std::pair<std::string, double>>     stages_;
};

}   // namespace sjxl::gpu
