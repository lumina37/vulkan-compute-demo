#pragma once

#include <memory>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

namespace _hp {

static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
static constexpr vk::ImageSubresourceLayers SUBRESOURCE_LAYERS{vk::ImageAspectFlagBits::eColor, 0, 0, 1};

}  // namespace _hp

class ImageBox {
    ImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Image image, const Extent& extent,
             vk::ImageUsageFlags usage) noexcept;

public:
    ImageBox(const ImageBox&) = delete;
    ImageBox(ImageBox&& rhs) noexcept;
    ~ImageBox() noexcept;

    [[nodiscard]] static std::expected<ImageBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                               const Extent& extent,
                                                               vk::ImageUsageFlags usage) noexcept;
    [[nodiscard]] static ImageBox createWithoutOwning(vk::Image image, const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto getVkImage(this Self&& self) noexcept {
        return std::forward_like<Self>(self).image_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::MemoryRequirements getMemoryRequirements() const noexcept;
    [[nodiscard]] std::expected<void, Error> bind(MemoryBox& memoryBox) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::Image image_;
    Extent extent_;
    vk::ImageUsageFlags usage_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.cpp"
#endif
