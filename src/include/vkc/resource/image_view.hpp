#pragma once

#include <expected>
#include <memory>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image.hpp"

namespace vkc {

class ImageViewBox {
    ImageViewBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::ImageView imageView) noexcept;

public:
    ImageViewBox(const ImageViewBox&) = delete;
    ImageViewBox(ImageViewBox&& rhs) noexcept;
    ~ImageViewBox() noexcept;

    [[nodiscard]] static std::expected<ImageViewBox, Error> create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                   const ImageBox& imageBox) noexcept;
    [[nodiscard]] static std::expected<ImageViewBox, Error> createFromVkImage(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                              vk::Image image,
                                                                              const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto getVkImageView(this Self&& self) noexcept {
        return std::forward_like<Self>(self).imageView_;
    }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::ImageView imageView_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image_view.cpp"
#endif
