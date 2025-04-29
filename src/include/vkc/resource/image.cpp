#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.hpp"
#endif

namespace vkc {

ImageManager::ImageManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, Extent extent, ImageType imageType,
                           vk::DescriptorType descType, vk::Image image, vk::ImageView imageView,
                           vk::DeviceMemory imageMemory, vk::Buffer stagingBuffer, vk::DeviceMemory stagingMemory,
                           vk::DescriptorImageInfo descImageInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      extent_(extent),
      imageType_(imageType),
      descType_(descType),
      image_(image),
      imageView_(imageView),
      imageMemory_(imageMemory),
      stagingBuffer_(stagingBuffer),
      stagingMemory_(stagingMemory),
      descImageInfo_(descImageInfo) {}

ImageManager::ImageManager(ImageManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      extent_(rhs.extent_),
      imageType_(rhs.imageType_),
      descType_(rhs.descType_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageMemory_(std::exchange(rhs.imageMemory_, nullptr)),
      stagingBuffer_(std::exchange(rhs.stagingBuffer_, nullptr)),
      stagingMemory_(std::exchange(rhs.stagingMemory_, nullptr)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})) {}

ImageManager::~ImageManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    if (stagingBuffer_ != nullptr) {
        device.destroyBuffer(stagingBuffer_);
        stagingBuffer_ = nullptr;
    }
    if (stagingMemory_ != nullptr) {
        device.freeMemory(stagingMemory_);
        stagingMemory_ = nullptr;
    }
    if (imageView_ != nullptr) {
        device.destroyImageView(imageView_);
        imageView_ = nullptr;
    }
    if (image_ != nullptr) {
        device.destroyImage(image_);
        image_ = nullptr;
    }
    if (imageMemory_ != nullptr) {
        device.freeMemory(imageMemory_);
        imageMemory_ = nullptr;
    }
    descImageInfo_.setImageView(nullptr);
}

std::expected<ImageManager, Error> ImageManager::create(const PhysicalDeviceManager& phyDeviceMgr,
                                                        std::shared_ptr<DeviceManager> pDeviceMgr, const Extent& extent,
                                                        ImageType imageType) noexcept {
    auto& device = pDeviceMgr->getDevice();

    vk::DescriptorType descType;
    vk::ImageUsageFlags imageUsage;
    vk::BufferUsageFlags bufferUsage;
    vk::ImageLayout imageLayout;
    switch (imageType) {
        case ImageType::Read:
            descType = vk::DescriptorType::eSampledImage;
            imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
            bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
            imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            break;
        case ImageType::Write:
            descType = vk::DescriptorType::eStorageImage;
            imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
            bufferUsage = vk::BufferUsageFlagBits::eTransferDst;
            imageLayout = vk::ImageLayout::eGeneral;
            break;
        default:
            std::unreachable();
    }

    // Image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setFormat(extent.format());
    imageInfo.setExtent(extent.extent3D());
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setTiling(vk::ImageTiling::eOptimal);
    imageInfo.setUsage(imageUsage);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    vk::Image image = device.createImage(imageInfo);

    // Device Memory
    vk::DeviceMemory imageMemory;
    _hp::allocImageMemory(phyDeviceMgr, *pDeviceMgr, image, vk::MemoryPropertyFlagBits::eDeviceLocal, imageMemory);
    device.bindImageMemory(image, imageMemory, 0);

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
    vk::ImageView imageView = device.createImageView(imageViewInfo);

    // Staging Memory
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(extent.size());
    bufferInfo.setUsage(bufferUsage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);

    vk::DeviceMemory stagingMemory;
    _hp::allocBufferMemory(phyDeviceMgr, *pDeviceMgr, stagingBuffer,
                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                           stagingMemory);
    device.bindBufferMemory(stagingBuffer, stagingMemory, 0);

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageView);
    descImageInfo.setImageLayout(imageLayout);

    return ImageManager{std::move(pDeviceMgr), extent,        imageType,     descType,     image, imageView,
                        imageMemory,           stagingBuffer, stagingMemory, descImageInfo};
}

vk::WriteDescriptorSet ImageManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(descType_);
    writeDescSet.setImageInfo(descImageInfo_);
    return writeDescSet;
}

vk::DescriptorSetLayoutBinding ImageManager::draftDescSetLayoutBinding() const noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(descType_);
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

vk::Result ImageManager::uploadFrom(const std::span<const std::byte> data) {
    return _hp::uploadFrom(*pDeviceMgr_, stagingMemory_, data);
}

vk::Result ImageManager::downloadTo(const std::span<std::byte> data) {
    return _hp::downloadTo(*pDeviceMgr_, stagingMemory_, data);
}

}  // namespace vkc
