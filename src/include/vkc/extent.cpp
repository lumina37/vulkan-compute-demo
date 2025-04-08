#include <utility>

#include <vulkan/vulkan.hpp>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/extent.hpp"
#endif

namespace vkc {

vk::Format Extent::formatUnorm() const noexcept {
    switch (comps_) {
        case 1:
            return vk::Format::eR8Unorm;
        case 2:
            return vk::Format::eR8G8Unorm;
        case 3:
            return vk::Format::eR8G8B8Unorm;
        case 4:
            return vk::Format::eR8G8B8A8Unorm;
        default:
            std::unreachable();
    }
}

}  // namespace vkc
