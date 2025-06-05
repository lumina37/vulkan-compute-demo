#pragma once

#include <concepts>
#include <expected>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename Self>
concept CImageBox = requires {
    requires requires(const Self& self) {
        { self.getImageAccessMask() } noexcept -> std::same_as<vk::AccessFlags>;
        { self.getImageLayout() } noexcept -> std::same_as<vk::ImageLayout>;
        { self.getStagingAccessMask() } noexcept -> std::same_as<vk::AccessFlags>;
    };
    requires requires(Self& self) {
        { self.getImage() } noexcept -> std::same_as<vk::Image>;
        { self.getStagingBuffer() } noexcept -> std::same_as<vk::Buffer>;
        { self.setImageAccessMask(std::declval<vk::AccessFlags>()) } noexcept;
        { self.setImageLayout(std::declval<vk::ImageLayout>()) } noexcept;
        { self.setStagingAccessMask(std::declval<vk::AccessFlags>()) } noexcept;
    };
};

template <typename Self>
concept CQueryPoolBox = requires {
    requires requires(Self& self) {
        { self.addQueryIndex() } noexcept -> std::same_as<std::expected<void, Error>>;
        { self.resetQueryIndex() } noexcept;
    };
    requires requires(const Self& self) {
        { self.getQueryPool() } noexcept -> std::same_as<vk::QueryPool>;
        { self.getQueryIndex() } noexcept -> std::same_as<int>;
        { self.getQueryCount() } noexcept -> std::same_as<int>;
    };
};

}  // namespace vkc
