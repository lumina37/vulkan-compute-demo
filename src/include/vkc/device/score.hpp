#pragma once

#include <compare>

namespace vkc {

template <typename TAttach>
class Score {
public:
    float score;
    TAttach attachment;

    friend constexpr std::partial_ordering operator<=>(const Score& lhs, const Score& rhs) noexcept {
        return lhs.score <=> rhs.score;
    }
};

}  // namespace vkc
