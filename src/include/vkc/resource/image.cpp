#include <cstddef>
#include <cstring>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.hpp"
#endif

namespace vkc {

ImageManager::ImageManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const ExtentManager& extent,
                           const ImageType imageType)
    : deviceMgr_(deviceMgr), extent_(extent), imageType_(imageType) {
    auto& device = deviceMgr.getDevice();

    vk::ImageUsageFlags imageUsage;
    vk::BufferUsageFlags bufferUsage;
    vk::ImageLayout imageLayout;
    switch (imageType) {
        case ImageType::ReadOnly:
            descType_ = vk::DescriptorType::eSampledImage;
            imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
            bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
            imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            break;
        case ImageType::WriteOnly:
            descType_ = vk::DescriptorType::eStorageImage;
            imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
            bufferUsage = vk::BufferUsageFlagBits::eTransferDst;
            imageLayout = vk::ImageLayout::eGeneral;
            break;
        case ImageType::ReadWrite:
            descType_ = vk::DescriptorType::eStorageImage;
            imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
            bufferUsage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
            imageLayout = vk::ImageLayout::eGeneral;
            break;
        default:
            std::unreachable();
    }

    // Image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setFormat(extent.formatUnorm());
    imageInfo.setExtent(extent.extent3D());
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setTiling(vk::ImageTiling::eOptimal);
    imageInfo.setUsage(imageUsage);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    image_ = device.createImage(imageInfo);

    // Device Memory
    _hp::allocImageMemory(phyDeviceMgr, deviceMgr, image_, vk::MemoryPropertyFlagBits::eDeviceLocal, imageMemory_);
    device.bindImageMemory(image_, imageMemory_, 0);

    // Image View
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setBaseMipLevel(0);
    subresourceRange.setLevelCount(1);
    subresourceRange.setBaseArrayLayer(0);
    subresourceRange.setLayerCount(1);

    vk::ImageViewCreateInfo imageViewInfo;
    imageViewInfo.setImage(image_);
    imageViewInfo.setViewType(vk::ImageViewType::e2D);
    imageViewInfo.setFormat(extent.formatUnorm());
    imageViewInfo.setSubresourceRange(subresourceRange);
    imageView_ = device.createImageView(imageViewInfo);

    // Staging Memory
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(extent.size());
    bufferInfo.setUsage(bufferUsage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    stagingBuffer_ = device.createBuffer(bufferInfo);

    _hp::allocBufferMemory(phyDeviceMgr, deviceMgr, stagingBuffer_,
                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                           stagingMemory_);
    device.bindBufferMemory(stagingBuffer_, stagingMemory_, 0);

    // Descriptor Image Info
    descImageInfo_.setImageView(imageView_);
    descImageInfo_.setImageLayout(imageLayout);
}

ImageManager::~ImageManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyBuffer(stagingBuffer_);
    device.freeMemory(stagingMemory_);
    device.destroyImageView(imageView_);
    device.destroyImage(image_);
    device.freeMemory(imageMemory_);
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

vk::Result ImageManager::uploadFrom(const std::span<std::byte> data) {
    return _hp::uploadFrom(deviceMgr_, stagingMemory_, data);
}

vk::Result ImageManager::downloadTo(std::span<std::byte> data) {
    return _hp::downloadTo(deviceMgr_, stagingMemory_, data);
}

}  // namespace vkc
