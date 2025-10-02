#pragma once

#include <concepts>
#include <expected>

#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename Self>
concept CImageBox = requires {
    requires requires(const Self& self) {
        { self.getExtent() } noexcept -> std::same_as<const Extent&>;
        { self.getAccessMask() } noexcept -> std::same_as<vk::AccessFlags>;
        { self.getImageLayout() } noexcept -> std::same_as<vk::ImageLayout>;
    };
    requires requires(Self& self) {
        { self.getVkImage() } noexcept -> std::same_as<vk::Image>;
        { self.getExtent() } noexcept -> std::same_as<Extent&>;
        { self.setAccessMask(std::declval<vk::AccessFlags>()) } noexcept;
        { self.setImageLayout(std::declval<vk::ImageLayout>()) } noexcept;
    };
};

template <typename Self>
concept CBufferBox = requires {
    requires requires(const Self& self) {
        { self.getSize() } noexcept -> std::same_as<vk::DeviceSize>;
        { self.getAccessMask() } noexcept -> std::same_as<vk::AccessFlags>;
    };
    requires requires(Self& self) {
        { self.getVkBuffer() } noexcept -> std::same_as<vk::Buffer>;
        { self.setAccessMask(std::declval<vk::AccessFlags>()) } noexcept;
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
