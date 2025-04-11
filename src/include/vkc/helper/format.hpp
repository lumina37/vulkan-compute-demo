#pragma once

#include <vulkan/vulkan.hpp>

namespace vkc {

[[nodiscard]] constexpr int mapVkFormatToBpp(const vk::Format format) noexcept {
    switch (format) {
        case vk::Format::eR8Unorm:
            return 1;
        case vk::Format::eR8G8Unorm:
            return 2;
        case vk::Format::eR8G8B8Unorm:
            return 3;
        case vk::Format::eR8G8B8A8Unorm:
            return 4;
        default:
            std::unreachable();
    }
}

}  // namespace vkc
