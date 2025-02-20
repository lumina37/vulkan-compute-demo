#pragma once

#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"

namespace vkc {

template <CSupportGetDescType... TManager>
[[nodiscard]] static inline auto genDescSetLayoutBindings(const TManager&... mgrs) {
    std::array descSetLayoutBindings{[](const auto& mgr) {
        vk::DescriptorSetLayoutBinding binding;
        binding.setDescriptorCount(1);
        binding.setDescriptorType(mgr.getDescType());
        binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
        return binding;
    }(mgrs)...};

    for (auto [index, binding] : rgs::views::enumerate(descSetLayoutBindings)) {
        binding.setBinding(index);
    }

    return descSetLayoutBindings;
}

}  // namespace vkc
