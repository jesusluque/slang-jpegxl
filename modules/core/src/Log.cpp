// Copyright (c) 2026 jesus luque.
#include "sjxl/core/Log.h"

#include <atomic>
#include <cstdio>
#include <mutex>

namespace sjxl::log {
namespace {

std::mutex& guard() {
    static std::mutex kGuard;
    return kGuard;
}

Sink& sink() {
    static Sink kSink;
    return kSink;
}

std::atomic<Level>& minimum() {
    static std::atomic<Level> kMinimum{Level::Info};
    return kMinimum;
}

const char* tag(Level level) {
    switch (level) {
    case Level::Debug: return "debug";
    case Level::Info: return "info";
    case Level::Warn: return "warn";
    case Level::Error: return "error";
    }
    return "?";
}

}   // namespace

void setSink(Sink replacement) { sink() = std::move(replacement); }
void setMinimum(Level level) noexcept { minimum().store(level); }

void write(Level level, std::string_view line) {
    if (level < minimum().load()) {
        return;
    }
    const std::lock_guard<std::mutex> held(guard());
    if (sink()) {
        sink()(level, line);
        return;
    }
    std::fprintf(stderr, "sjxl [%s] %.*s\n", tag(level), static_cast<int>(line.size()), line.data());
}

}   // namespace sjxl::log
