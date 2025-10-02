#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image_view.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/present_image.hpp"
#endif

namespace vkc {

PresentImageBox::PresentImageBox(ImageBox&& imageBox, ImageViewBox&& imageViewBox) noexcept
    : imageBox_(std::move(imageBox)),
      imageViewBox_(std::move(imageViewBox)),
      accessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined) {}

std::expected<PresentImageBox, Error> PresentImageBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                              ImageBox& imageBox) noexcept {
    auto imageViewBoxRes = ImageViewBox::create(pDeviceBox, imageBox);
    if (!imageViewBoxRes) return std::unexpected{std::move(imageViewBoxRes.error())};
    ImageViewBox& imageViewBox = imageViewBoxRes.value();

    return PresentImageBox{std::move(imageBox), std::move(imageViewBox)};
}

}  // namespace vkc
