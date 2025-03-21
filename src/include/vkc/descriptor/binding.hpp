#pragma once

#include <cstddef>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"

namespace vkc {

template <CSupportDraftDescSetLayoutBinding... TManager>
[[nodiscard]] static constexpr inline auto genDescSetLayoutBindings(const TManager&... mgrs) {
    const auto genDescSetLayoutBinding = [](const auto& mgr, size_t index) {
        vk::DescriptorSetLayoutBinding binding = mgr.draftDescSetLayoutBinding();
        binding.setBinding(index);
        return binding;
    };

    const auto genDescSetLayoutBindingHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genDescSetLayoutBinding(mgrs, Is)...};
    };

    return genDescSetLayoutBindingHelper(std::index_sequence_for<TManager...>{});
}

}  // namespace vkc
