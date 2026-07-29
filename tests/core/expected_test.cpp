#include <catch2/catch_test_macros.hpp>
#include "axon/core/expected.h"

using namespace axon;

TEST_CASE("Expected holds a value", "[core][expected]") {
    Expected<int> e(42);
    REQUIRE(e);
    REQUIRE(e.value() == 42);
}

TEST_CASE("Expected holds an error", "[core][expected]") {
    Expected<int> e(Error{"something failed"});
    REQUIRE_FALSE(e);
    REQUIRE(e.error().message == "something failed");
}

TEST_CASE("Expected<void> success", "[core][expected]") {
    Expected<void> e;
    REQUIRE(e);
}

TEST_CASE("Expected<void> failure", "[core][expected]") {
    Expected<void> e(Error{"failed"});
    REQUIRE_FALSE(e);
    REQUIRE(e.error().message == "failed");
}

TEST_CASE("RETURN_IF_ERROR macro propagates errors", "[core][expected]") {
    auto func = []() -> Expected<int> {
        auto dummy = []() -> Expected<void> {
            return Error{"nope"};
        };
        RETURN_IF_ERROR(dummy());
        return 42;
    };
    auto result = func();
    REQUIRE_FALSE(result);
    REQUIRE(result.error().message == "nope");
}
