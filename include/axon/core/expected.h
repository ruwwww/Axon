#pragma once

#include <string>
#include <variant>
#include <utility>

namespace axon {

struct Error {
    std::string message;
};

template <typename T>
class Expected {
public:
    Expected(T value) : data_(std::move(value)) {}
    Expected(Error error) : data_(std::move(error)) {}

    explicit operator bool() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    T& value() {
        return std::get<T>(data_);
    }

    const T& value() const {
        return std::get<T>(data_);
    }

    Error& error() {
        return std::get<Error>(data_);
    }

    const Error& error() const {
        return std::get<Error>(data_);
    }

    T& operator*() { return value(); }
    const T& operator*() const { return value(); }

private:
    std::variant<T, Error> data_;
};

template <>
class Expected<void> {
public:
    Expected() : ok_(true) {}
    Expected(Error error) : ok_(false), error_(std::move(error)) {}

    explicit operator bool() const noexcept { return ok_; }

    Error& error() { return error_; }
    const Error& error() const { return error_; }

private:
    bool ok_;
    Error error_;
};

} // namespace axon

#define RETURN_IF_ERROR(expr) \
    do { \
        auto _result = (expr); \
        if (!_result) return std::move(_result.error()); \
    } while (false)
