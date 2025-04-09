#pragma once

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace vkc {

class Extent {
public:
    Extent() = default;
    Extent(const int width, const int height, const int comps, const vk::Format format)
        : extent_(width, height), format_(format), comps_(comps), size_(width * height * comps) {}

    [[nodiscard]] uint32_t width() const noexcept { return extent_.width; }
    [[nodiscard]] uint32_t height() const noexcept { return extent_.height; }
    [[nodiscard]] size_t comps() const noexcept { return comps_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] vk::Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] vk::Extent3D extent3D() const noexcept { return {extent_.width, extent_.height, 1}; }
    [[nodiscard]] vk::Format format() const noexcept { return format_; }

private:
    vk::Extent2D extent_;
    vk::Format format_;
    size_t comps_;
    size_t size_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/extent.cpp"
#endif
