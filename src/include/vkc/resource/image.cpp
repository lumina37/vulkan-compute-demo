#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.hpp"
#endif

namespace vkc {

ImageBox::ImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Image image, const Extent& extent,
                   vk::ImageUsageFlags usage) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), image_(image), extent_(extent), usage_(usage) {}

ImageBox::ImageBox(ImageBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      image_(std::exchange(rhs.image_, nullptr)),
      extent_(rhs.extent_),
      usage_(rhs.usage_) {}

ImageBox::~ImageBox() noexcept {
    if (image_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyImage(image_);
    image_ = nullptr;
}

std::expected<ImageBox, Error> ImageBox::create(std::shared_ptr<DeviceBox> pDeviceBox, const Extent& extent,
                                                vk::ImageUsageFlags usage) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setFormat(extent.format());
    imageInfo.setExtent(extent.extent3D());
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setTiling(vk::ImageTiling::eOptimal);
    imageInfo.setUsage(usage);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    auto [imageRes, image] = device.createImage(imageInfo);
    if (imageRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, imageRes}};
    }

    return ImageBox{std::move(pDeviceBox), image, extent, usage};
}

vk::MemoryRequirements ImageBox::getMemoryRequirements() const noexcept {
    return _hp::getMemoryRequirements(*pDeviceBox_, image_);
}

std::expected<void, Error> ImageBox::bind(MemoryBox& memoryBox) noexcept {
    vk::Device device = pDeviceBox_->getDevice();

    const auto bindRes = device.bindImageMemory(image_, memoryBox.getVkDeviceMemory(), 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
    }

    return {};
}

}  // namespace vkc
