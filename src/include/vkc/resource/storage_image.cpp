#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.hpp"
#endif

namespace vkc {

StorageImageManager::StorageImageManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, Extent extent, vk::Image image,
                                         vk::ImageView imageView, vk::DeviceMemory imageMemory,
                                         vk::Buffer stagingBuffer, vk::DeviceMemory stagingMemory,
                                         vk::DescriptorImageInfo descImageInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      imageMemory_(imageMemory),
      stagingBuffer_(stagingBuffer),
      stagingMemory_(stagingMemory),
      descImageInfo_(descImageInfo) {}

StorageImageManager::StorageImageManager(StorageImageManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageMemory_(std::exchange(rhs.imageMemory_, nullptr)),
      stagingBuffer_(std::exchange(rhs.stagingBuffer_, nullptr)),
      stagingMemory_(std::exchange(rhs.stagingMemory_, nullptr)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})) {}

StorageImageManager::~StorageImageManager() noexcept {
    if (pDeviceMgr_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();

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

std::expected<StorageImageManager, Error> StorageImageManager::create(const PhyDeviceManager& phyDeviceMgr,
                                                                      std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                      const Extent& extent,
                                                                      StorageImageType imageType) noexcept {
    vk::Device device = pDeviceMgr->getDevice();

    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eStorage;
    vk::BufferUsageFlags bufferUsage{};
    constexpr vk::ImageLayout imageLayout = vk::ImageLayout::eGeneral;

    if (StorageImageType::Read & imageType) {
        imageUsage |= vk::ImageUsageFlagBits::eTransferDst;
        bufferUsage |= vk::BufferUsageFlagBits::eTransferSrc;
    }
    if (StorageImageType::Write & imageType) {
        imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
        bufferUsage |= vk::BufferUsageFlagBits::eTransferDst;
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
    auto [imageRes, image] = device.createImage(imageInfo);
    if (imageRes != vk::Result::eSuccess) {
        return std::unexpected{Error{imageRes}};
    }

    // Device Memory
    vk::DeviceMemory imageMemory;
    auto allocRes =
        _hp::allocImageMemory(phyDeviceMgr, *pDeviceMgr, image, vk::MemoryPropertyFlagBits::eDeviceLocal, imageMemory);
    if (!allocRes) return std::unexpected{std::move(allocRes.error())};

    const vk::Result bindRes = device.bindImageMemory(image, imageMemory, 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bindRes}};
    }

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
        return std::unexpected{Error{imageViewRes}};
    }

    // Staging Memory
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(extent.size());
    bufferInfo.setUsage(bufferUsage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [stagingBufferRes, stagingBuffer] = device.createBuffer(bufferInfo);
    if (stagingBufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{stagingBufferRes}};
    }

    vk::DeviceMemory stagingMemory;
    auto allocStagingRes = _hp::allocBufferMemory(
        phyDeviceMgr, *pDeviceMgr, stagingBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingMemory);
    if (!allocStagingRes) return std::unexpected{std::move(allocStagingRes.error())};

    const vk::Result bindStagingRes = device.bindBufferMemory(stagingBuffer, stagingMemory, 0);
    if (bindStagingRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bindStagingRes}};
    }

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageView);
    descImageInfo.setImageLayout(imageLayout);

    return StorageImageManager{std::move(pDeviceMgr), extent,        image,         imageView,
                               imageMemory,           stagingBuffer, stagingMemory, descImageInfo};
}

vk::WriteDescriptorSet StorageImageManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eStorageImage);
    writeDescSet.setImageInfo(descImageInfo_);
    return writeDescSet;
}

vk::DescriptorSetLayoutBinding StorageImageManager::draftDescSetLayoutBinding() const noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(vk::DescriptorType::eStorageImage);
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

std::expected<void, Error> StorageImageManager::uploadFrom(const std::span<const std::byte> data) noexcept {
    return _hp::uploadFrom(*pDeviceMgr_, stagingMemory_, data);
}

std::expected<void, Error> StorageImageManager::downloadTo(const std::span<std::byte> data) noexcept {
    return _hp::downloadTo(*pDeviceMgr_, stagingMemory_, data);
}

}  // namespace vkc
