#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/present_image.hpp"
#endif

namespace vkc {

PresentImageBox::PresentImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, const Extent& extent, vk::Image image,
                                 vk::ImageView imageView) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      imageAccessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined) {}

PresentImageBox::PresentImageBox(PresentImageBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageAccessMask_(rhs.imageAccessMask_),
      imageLayout_(rhs.imageLayout_) {}

PresentImageBox::~PresentImageBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();

    if (imageView_ != nullptr) {
        device.destroyImageView(imageView_);
        imageView_ = nullptr;
    }
    image_ = nullptr;
}

std::expected<PresentImageBox, Error> PresentImageBox::create(std::shared_ptr<DeviceBox> pDeviceBox, vk::Image image,
                                                              const Extent& extent) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    // Image View
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setBaseMipLevel(0);
    subresourceRange.setLevelCount(1);
    subresourceRange.setBaseArrayLayer(0);
    subresourceRange.setLayerCount(1);

    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(extent.format());
    imageViewInfo.setSubresourceRange(subresourceRange);
    const auto [imageViewRes, imageView] = device.createImageView(imageViewInfo);
    if (imageViewRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, imageViewRes}};
    }

    return PresentImageBox{std::move(pDeviceBox), extent, image, imageView};
}

}  // namespace vkc
