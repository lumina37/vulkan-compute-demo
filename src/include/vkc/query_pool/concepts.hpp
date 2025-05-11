#pragma once

#include <concepts>
#include <expected>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename Self>
concept CQueryPoolManager = requires {
    requires requires(Self& self) {
        { self.getQueryPool() } noexcept -> std::same_as<vk::QueryPool&>;
        { self.addQueryIndex() } noexcept -> std::same_as<std::expected<void, Error>>;
        { self.resetQueryIndex() } noexcept;
    };
    requires requires(const Self& self) {
        { self.getQueryPool() } noexcept -> std::same_as<const vk::QueryPool&>;
        { self.getQueryIndex() } noexcept -> std::same_as<int>;
        { self.getQueryCount() } noexcept -> std::same_as<int>;
    };
};

}  // namespace vkc
