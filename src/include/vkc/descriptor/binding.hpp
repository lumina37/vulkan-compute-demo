#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <CSupportDraftDescSetLayoutBinding... TManager>
[[nodiscard]] static constexpr auto genDescSetLayoutBindings(const TManager&... mgrs) noexcept {
    constexpr auto genDescSetLayoutBinding = [](const auto& mgr, const size_t index) {
        vk::DescriptorSetLayoutBinding binding = mgr.draftDescSetLayoutBinding();
        binding.setBinding((uint32_t)index);
        return binding;
    };

    const auto genDescSetLayoutBindingHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genDescSetLayoutBinding(mgrs, Is)...};
    };

    return genDescSetLayoutBindingHelper(std::index_sequence_for<TManager...>{});
}

}  // namespace vkc
