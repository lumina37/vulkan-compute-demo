#pragma once

#include <concepts>

namespace vkc {

template <std::unsigned_integral Tv>
[[nodiscard]] static constexpr bool isPowOf2(const Tv v) noexcept {
    return (v & (v - 1)) == 0;
}

template <std::integral Tv>
[[nodiscard]] static constexpr Tv alignUp(const Tv v, const size_t to) noexcept {
    return (Tv)(((size_t)v + (to - 1)) & ((~to) + 1));
}

template <std::integral Tv>
[[nodiscard]] static constexpr Tv ceilDiv(const Tv dividend, const Tv divisor) noexcept {
    return (dividend + (divisor - 1)) / divisor;
}

}  // namespace vkc
