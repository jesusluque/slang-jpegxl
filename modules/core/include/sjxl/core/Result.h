// Copyright (c) 2026 jesus luque.
#pragma once

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include "sjxl/core/Error.h"

namespace sjxl {

/// A value or an Error. `std::expected<T, Error>` in spirit, hand-rolled so the
/// build does not depend on which libc++ has it. Errors are returned, never
/// thrown.
template <typename T>
class [[nodiscard]] Result {
    static_assert(!std::is_same_v<std::remove_cv_t<T>, Error>, "Result<Error> is meaningless; use Result<void>");

public:
    using value_type = T;

    Result(T value) : storage_(std::move(value)) {}       // NOLINT(*-explicit-*)
    Result(Error error) : storage_(std::move(error)) {}   // NOLINT(*-explicit-*)

    [[nodiscard]] bool hasValue() const noexcept { return storage_.index() == 0; }
    explicit operator bool() const noexcept { return hasValue(); }

    [[nodiscard]] T& value() & {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(storage_);
    }
    [[nodiscard]] const T& value() const& {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(storage_);
    }
    [[nodiscard]] T&& value() && {
        assert(hasValue() && "Result::value() on an error");
        return std::get<0>(std::move(storage_));
    }
    [[nodiscard]] const Error& error() const& {
        assert(!hasValue() && "Result::error() on a value");
        return std::get<1>(storage_);
    }
    [[nodiscard]] Error&& error() && {
        assert(!hasValue() && "Result::error() on a value");
        return std::get<1>(std::move(storage_));
    }

    T& operator*() & { return value(); }
    const T& operator*() const& { return value(); }
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }

private:
    std::variant<T, Error> storage_;
};

template <>
class [[nodiscard]] Result<void> {
public:
    using value_type = void;

    Result() = default;
    Result(Error error) : error_(std::move(error)) {}   // NOLINT(*-explicit-*)

    [[nodiscard]] bool hasValue() const noexcept { return !error_.has_value(); }
    explicit operator bool() const noexcept { return hasValue(); }
    [[nodiscard]] const Error& error() const& {
        assert(!hasValue() && "Result::error() on a value");
        return *error_;
    }
    [[nodiscard]] Error&& error() && {
        assert(!hasValue() && "Result::error() on a value");
        return std::move(*error_);
    }

private:
    std::optional<Error> error_;
};

inline Result<void> ok() { return {}; }

/// Returns the error of `expr` from the enclosing function if it has one.
#define SJXL_TRY(expr)                                   \
    do {                                                 \
        if (auto sjxl_try_ = (expr); !sjxl_try_) {       \
            return std::move(sjxl_try_).error();         \
        }                                                \
    } while (false)

#define SJXL_CONCAT_(a, b) a##b
#define SJXL_CONCAT(a, b) SJXL_CONCAT_(a, b)

/// `SJXL_ASSIGN(auto x, f())`: x is f()'s value, or the function returns its error.
#define SJXL_ASSIGN(decl, expr)                                            \
    auto SJXL_CONCAT(sjxl_assign_, __LINE__) = (expr);                     \
    if (!SJXL_CONCAT(sjxl_assign_, __LINE__)) {                            \
        return std::move(SJXL_CONCAT(sjxl_assign_, __LINE__)).error();     \
    }                                                                      \
    decl = std::move(*SJXL_CONCAT(sjxl_assign_, __LINE__))

}   // namespace sjxl
