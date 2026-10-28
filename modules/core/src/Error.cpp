// Copyright (c) 2026 jesus luque.
#include "sjxl/core/Error.h"

namespace sjxl {

std::string_view toString(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::Unknown: return "Unknown";
    case ErrorCode::InvalidArgument: return "InvalidArgument";
    case ErrorCode::Unsupported: return "Unsupported";
    case ErrorCode::Corrupt: return "Corrupt";
    case ErrorCode::IoFailure: return "IoFailure";
    case ErrorCode::DeviceFailure: return "DeviceFailure";
    case ErrorCode::ShaderFailure: return "ShaderFailure";
    case ErrorCode::OutOfMemory: return "OutOfMemory";
    case ErrorCode::InternalError: return "InternalError";
    }
    return "?";
}

std::string Error::toString() const {
    return std::string(sjxl::toString(code_)) + ": " + message_;
}

}   // namespace sjxl
