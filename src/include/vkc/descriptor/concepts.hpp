#pragma once

#include <concepts>

#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename Self>
concept CSupportDraftWriteDescSet = requires {
    requires requires(const Self& self) {
        { self.draftWriteDescSet() } noexcept -> std::same_as<vk::WriteDescriptorSet>;
    };
};

template <typename Self>
concept CSupportStaticGetDescType = requires {
    requires requires {
        { Self::getDescType() } noexcept -> std::same_as<vk::DescriptorType>;
    };
};

template <typename Self>
concept CSupportGetDescType = requires {
    requires requires(const Self& self) {
        { self.getDescType() } noexcept -> std::same_as<vk::DescriptorType>;
    };
};

template <typename Self>
concept CSupportDraftDescSetLayoutBinding = requires {
    requires requires(const Self& self) {
        { self.draftDescSetLayoutBinding() } noexcept -> std::same_as<vk::DescriptorSetLayoutBinding>;
    };
};

template <typename Self>
concept CSupportStaticDraftDescSetLayoutBinding = requires {
    requires requires {
        { Self::draftDescSetLayoutBinding() } noexcept -> std::same_as<vk::DescriptorSetLayoutBinding>;
    };
};

}  // namespace vkc
