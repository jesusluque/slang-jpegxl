// Copyright (c) 2026 jesus luque.
#pragma once

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace sjxl {

/// Broad failure categories. Coarse on purpose: the message carries the
/// detail, the code exists so a caller can branch without parsing a string.
enum class ErrorCode {
    Unknown,
    InvalidArgument,
    Unsupported,     ///< a codestream feature this decoder does not implement yet; names it
    Corrupt,         ///< the codestream violates the specification
    IoFailure,
    DeviceFailure,   ///< the GPU or its driver refused
    ShaderFailure,   ///< Slang could not compile or link
    OutOfMemory,
    InternalError,
};

[[nodiscard]] std::string_view toString(ErrorCode) noexcept;

/// A failure value. Cheap to move, never thrown.
class Error {
public:
    Error() = default;
    Error(ErrorCode code, std::string message) : code_(code), message_(std::move(message)) {}

    template <typename... Args>
    [[nodiscard]] static Error make(ErrorCode code, std::format_string<Args...> fmt, Args&&... args) {
        return Error(code, std::format(fmt, std::forward<Args>(args)...));
    }

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    /// "Unsupported: modular transform Squeeze"
    [[nodiscard]] std::string toString() const;

private:
    ErrorCode   code_ = ErrorCode::Unknown;
    std::string message_;
};

}   // namespace sjxl
