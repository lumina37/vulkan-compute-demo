#pragma once

#include <concepts>

#include <vulkan/vulkan.hpp>

template <typename Self>
concept CSupDraftWriteDescSet = requires {
    requires requires(const Self& self) {
        { self.draftWriteDescSet() } -> std::same_as<vk::WriteDescriptorSet>;
    };
};
