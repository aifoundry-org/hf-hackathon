#pragma once

#include <optional>
#include <utility>

template <typename E>
class ExpectedVoid {
 public:
  ExpectedVoid() = default;
  ExpectedVoid(bool ok) {
    if (!ok) {
      error_ = E{};
    }
  }

  static ExpectedVoid success() { return ExpectedVoid(); }

  static ExpectedVoid failure(E error) {
    ExpectedVoid result;
    result.error_ = std::move(error);
    return result;
  }

  bool has_value() const { return !error_.has_value(); }
  explicit operator bool() const { return has_value(); }

  const E& error() const { return *error_; }
  E& error() { return *error_; }

 private:
  std::optional<E> error_;
};

template <typename T, typename E>
class Expected {
 public:
  Expected(T value) : value_(std::move(value)) {}

  static Expected failure(E error) {
    Expected result;
    result.error_ = std::move(error);
    return result;
  }

  bool has_value() const { return value_.has_value(); }
  explicit operator bool() const { return has_value(); }

  const T& value() const { return *value_; }
  T& value() { return *value_; }
  const E& error() const { return *error_; }
  E& error() { return *error_; }

 private:
  Expected() = default;

  std::optional<T> value_;
  std::optional<E> error_;
};
