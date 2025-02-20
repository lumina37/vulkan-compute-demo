#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/helper/memory.hpp"

namespace vkc {

class ImageManager {
public:
    inline ImageManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                        const vk::Extent2D& extent, const vk::ImageUsageFlags usage);
    inline ~ImageManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getImage(this Self& self) noexcept {
        return std::forward_like<Self>(self).image_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getImageView(this Self& self) noexcept {
        return std::forward_like<Self>(self).imageView_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getStagingMemory(this Self& self) noexcept {
        return std::forward_like<Self>(self).stagingMemory_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getStagingBuffer(this Self& self) noexcept {
        return std::forward_like<Self>(self).stagingBuffer_;
    }

    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Image image_;
    vk::DeviceMemory imageMemory_;
    vk::ImageView imageView_;
    vk::DeviceMemory stagingMemory_;
    vk::Buffer stagingBuffer_;
    vk::DescriptorImageInfo imageInfo_;
};

ImageManager::ImageManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                           const vk::Extent2D& extent, const vk::ImageUsageFlags usage)
    : deviceMgr_(deviceMgr) {
    const auto& device = deviceMgr.getDevice();

    // Image
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setFormat(vk::Format::eR8Uint);
    imageInfo.setExtent({extent.width, extent.height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setTiling(vk::ImageTiling::eOptimal);
    imageInfo.setUsage(usage);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    imageInfo.setInitialLayout(vk::ImageLayout::eUndefined);
    image_ = device.createImage(imageInfo);

    // Device Memory
    const vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image_);
    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(memRequirements.size);
    const auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();
    const auto memTypeIndex =
        findMemoryType(physicalDevice, memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    allocInfo.setMemoryTypeIndex(memTypeIndex);
    imageMemory_ = device.allocateMemory(allocInfo);
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
    imageViewInfo.setFormat(vk::Format::eR8Uint);
    imageViewInfo.setSubresourceRange(subresourceRange);
    imageView_ = device.createImageView(imageViewInfo);

    // Staging Memory
    vk::BufferUsageFlags bufferUsage;
    if (usage & vk::ImageUsageFlagBits::eTransferSrc) {
        bufferUsage = vk::BufferUsageFlagBits::eTransferDst;
    } else {
        bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
    }
    createBuffer(physicalDevice, device, extent.width * extent.height, bufferUsage,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer_,
                 stagingMemory_);

    // Image Info
    imageInfo_.setImageView(imageView_);
    imageInfo_.setImageLayout(vk::ImageLayout::eGeneral);
}

ImageManager::~ImageManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyImageView(imageView_);
    device.destroyImage(image_);
    device.freeMemory(imageMemory_);
    device.destroyBuffer(stagingBuffer_);
    device.freeMemory(stagingMemory_);
}

inline vk::WriteDescriptorSet ImageManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eStorageImage);
    writeDescSet.setImageInfo(imageInfo_);
    return writeDescSet;
}

}  // namespace vkc
