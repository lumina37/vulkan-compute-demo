#pragma once

#include <array>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <CSupportDraftDescSetLayoutBinding... TBox>
[[nodiscard]] static constexpr auto genDescSetLayoutBindings(const TBox&... boxes) noexcept {
    constexpr auto genDescSetLayoutBinding = [](const auto& box, const size_t index) {
        vk::DescriptorSetLayoutBinding binding = box.draftDescSetLayoutBinding();
        binding.setBinding((uint32_t)index);
        return binding;
    };

    const auto genDescSetLayoutBindingHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genDescSetLayoutBinding(boxes, Is)...};
    };

    return genDescSetLayoutBindingHelper(std::index_sequence_for<TBox...>{});
}

template <CSupportStaticDraftDescSetLayoutBinding... TBox>
[[nodiscard]] static constexpr auto genDescSetLayoutBindings() noexcept {
    constexpr auto genDescSetLayoutBinding = []<CSupportStaticDraftDescSetLayoutBinding T>(const size_t index) {
        vk::DescriptorSetLayoutBinding binding = T::draftDescSetLayoutBinding();
        binding.setBinding((uint32_t)index);
        return binding;
    };

    const auto genDescSetLayoutBindingHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array { genDescSetLayoutBinding.template operator()<TBox>(Is)... };
    };

    return genDescSetLayoutBindingHelper(std::index_sequence_for<TBox...>{});
}

}  // namespace vkc
