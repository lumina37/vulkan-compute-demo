#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image_view.hpp"
#endif

namespace vkc {

ImageViewBox::ImageViewBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::ImageView imageView) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), imageView_(imageView) {}

ImageViewBox::ImageViewBox(ImageViewBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), imageView_(std::exchange(rhs.imageView_, nullptr)) {}

ImageViewBox::~ImageViewBox() noexcept {
    if (imageView_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyImageView(imageView_);
    imageView_ = nullptr;
}

std::expected<ImageViewBox, Error> ImageViewBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                        const ImageBox& imageBox) noexcept {
    return createFromVkImage(pDeviceBox, imageBox.getVkImage(), imageBox.getExtent());
}

std::expected<ImageViewBox, Error> ImageViewBox::createFromVkImage(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                   vk::Image image, const Extent& extent) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(image);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(extent.format());
    imageViewInfo.setSubresourceRange(_hp::SUBRESOURCE_RANGE);
    const auto [imageViewRes, imageView] = device.createImageView(imageViewInfo);
    if (imageViewRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, imageViewRes}};
    }

    return ImageViewBox{std::move(pDeviceBox), imageView};
}

}  // namespace vkc
