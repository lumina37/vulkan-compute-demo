#pragma once

#include <concepts>

#include <vulkan/vulkan.hpp>

namespace vkc {

template <typename Self>
concept CSupportDraftWriteDescSet = requires {
    requires requires(const Self& self) {
        { self.draftWriteDescSet() } noexcept -> std::same_as<vk::WriteDescriptorSet>;
    };
};

template <typename Self>
concept CSupportGetDescType = requires {
    requires requires(const Self& self) {
        { self.getDescType() } noexcept -> std::same_as<vk::DescriptorType>;
    };
};

}  // namespace vkc
