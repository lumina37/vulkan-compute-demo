#pragma once

#include <concepts>

#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename Self>
concept CQueryPoolManager = requires {
    requires requires(Self& self) {
        { self.getQueryPool() } noexcept -> std::same_as<vk::QueryPool&>;
        { self.addQueryIndex() } noexcept;
        { self.resetQueryIndex() } noexcept;
    };
    requires requires(const Self& self) {
        { self.getQueryPool() } noexcept -> std::same_as<const vk::QueryPool&>;
        { self.getQueryIndex() } noexcept -> std::same_as<int>;
        { self.getQueryCount() } noexcept -> std::same_as<int>;
    };
};

}  // namespace vkc
