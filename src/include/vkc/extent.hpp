#pragma once

#include <cstddef>

#include "vkc/helper/format.hpp"
#include "vkc/helper/math.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class Extent {
public:
    constexpr Extent() = default;
    constexpr Extent(const int width, const int height, const vk::Format format)
        : extent_(width, height),
          format_(format),
          bpp_(mapVkFormatToBpp(format)),
          rowAlign_(1),
          rowPitch_(width * bpp_),
          size_(rowPitch_ * height) {}
    constexpr Extent(const int width, const int height, const vk::Format format, const int rowAlign)
        : extent_(width, height),
          format_(format),
          bpp_(mapVkFormatToBpp(format)),
          rowAlign_(rowAlign),
          rowPitch_((size_t)alignUp(width * bpp_, rowAlign)),
          size_(rowPitch_ * height) {}

    [[nodiscard]] constexpr int width() const noexcept { return (int)extent_.width; }
    [[nodiscard]] constexpr int height() const noexcept { return (int)extent_.height; }
    [[nodiscard]] constexpr int bpp() const noexcept { return bpp_; }
    [[nodiscard]] constexpr int rowAlign() const noexcept { return rowAlign_; }
    [[nodiscard]] constexpr size_t rowPitch() const noexcept { return rowPitch_; }
    [[nodiscard]] constexpr size_t elemCount() const noexcept { return extent_.width * extent_.height; }
    [[nodiscard]] constexpr size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr vk::Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] constexpr vk::Extent3D extent3D() const noexcept { return {extent_.width, extent_.height, 1}; }
    [[nodiscard]] constexpr vk::Format format() const noexcept { return format_; }
    [[nodiscard]] constexpr size_t calculateBufferOffset(const vk::Offset2D offset) const noexcept {
        return offset.y * rowPitch() + offset.x * bpp();
    }

private:
    vk::Extent2D extent_;
    vk::Format format_;
    int bpp_;
    int rowAlign_;
    size_t rowPitch_;
    size_t size_;
};

class Roi {
public:
    Roi() = delete;
    constexpr Roi(const int x, const int y, const int width, const int height)
        : offset_(x, y), extent_(width, height) {}

    [[nodiscard]] constexpr vk::Offset2D offset() const noexcept { return offset_; }
    [[nodiscard]] constexpr vk::Offset3D offset3D() const noexcept { return {offset_.x, offset_.y, 0}; }
    [[nodiscard]] constexpr vk::Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] constexpr vk::Extent3D extent3D() const noexcept { return {extent_.width, extent_.height, 1}; }

private:
    vk::Offset2D offset_;
    vk::Extent2D extent_;
};

}  // namespace vkc
