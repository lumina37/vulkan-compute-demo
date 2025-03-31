#pragma once

#include <compare>
#include <cstdint>

class ScoreWithIndex {
public:
    int64_t score;
    uint64_t index;

    static friend constexpr std::weak_ordering operator<=>(const ScoreWithIndex& lhs,
                                                           const ScoreWithIndex& rhs) noexcept {
        return lhs.score <=> rhs.score;
    }
};
