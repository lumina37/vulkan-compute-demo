#pragma once

#include <cstddef>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/math.hpp"

namespace vkc {

class Extent {
public:
    constexpr Extent() = default;
    constexpr Extent(const int width, const int height, const int comps, const vk::Format format)
        : extent_(width, height),
          comps_(comps),
          format_(format),
          rowAlign_(1),
          rowPitch_(width * comps),
          size_(rowPitch_ * height) {}
    constexpr Extent(const int width, const int height, const int comps, const vk::Format format, const int rowAlign)
        : extent_(width, height),
          comps_(comps),
          format_(format),
          rowAlign_(rowAlign),
          rowPitch_((size_t)alignUp(width * comps, rowAlign)),
          size_(rowPitch_ * height) {}

    [[nodiscard]] int width() const noexcept { return (int)extent_.width; }
    [[nodiscard]] int height() const noexcept { return (int)extent_.height; }
    [[nodiscard]] int comps() const noexcept { return comps_; }
    [[nodiscard]] int rowAlign() const noexcept { return rowAlign_; }
    [[nodiscard]] size_t rowPitch() const noexcept { return rowPitch_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] vk::Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] vk::Extent3D extent3D() const noexcept { return {extent_.width, extent_.height, 1}; }
    [[nodiscard]] vk::Format format() const noexcept { return format_; }

private:
    vk::Extent2D extent_;
    vk::Format format_;
    int comps_;
    int rowAlign_;
    size_t rowPitch_;
    size_t size_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/extent.cpp"
#endif
