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
#    include "vkc/resource/storage_image.hpp"
#endif

namespace vkc {

StorageImageBox::StorageImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, Extent extent, vk::Image image,
                                         vk::ImageView imageView, vk::DeviceMemory imageMemory,
                                         vk::Buffer stagingBuffer, vk::DeviceMemory stagingMemory,
                                         vk::DescriptorImageInfo descImageInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      imageMemory_(imageMemory),
      stagingBuffer_(stagingBuffer),
      stagingMemory_(stagingMemory),
      descImageInfo_(descImageInfo),
      imageAccessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined),
      stagingAccessMask_(vk::AccessFlagBits::eNone) {}

StorageImageBox::StorageImageBox(StorageImageBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageMemory_(std::exchange(rhs.imageMemory_, nullptr)),
      stagingBuffer_(std::exchange(rhs.stagingBuffer_, nullptr)),
      stagingMemory_(std::exchange(rhs.stagingMemory_, nullptr)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})),
      imageAccessMask_(rhs.imageAccessMask_),
      imageLayout_(rhs.imageLayout_),
      stagingAccessMask_(rhs.stagingAccessMask_) {}

StorageImageBox::~StorageImageBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();

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

std::expected<StorageImageBox, Error> StorageImageBox::create(const PhyDeviceBox& phyDeviceBox,
                                                                      std::shared_ptr<DeviceBox> pDeviceBox,
                                                                      const Extent& extent,
                                                                      StorageImageType imageType) noexcept {
    vk::Device device = pDeviceBox->getDevice();

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

    // Image Memory
    vk::DeviceMemory imageMemory;
    auto allocRes =
        _hp::allocImageMemory(phyDeviceBox, *pDeviceBox, image, vk::MemoryPropertyFlagBits::eDeviceLocal, imageMemory);
    if (!allocRes) return std::unexpected{std::move(allocRes.error())};

    const auto bindRes = device.bindImageMemory(image, imageMemory, 0);
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
        phyDeviceBox, *pDeviceBox, stagingBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingMemory);
    if (!allocStagingRes) return std::unexpected{std::move(allocStagingRes.error())};

    const auto bindStagingRes = device.bindBufferMemory(stagingBuffer, stagingMemory, 0);
    if (bindStagingRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bindStagingRes}};
    }

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageView);
    descImageInfo.setImageLayout(imageLayout);

    return StorageImageBox{std::move(pDeviceBox), extent,        image,         imageView,
                               imageMemory,           stagingBuffer, stagingMemory, descImageInfo};
}

vk::WriteDescriptorSet StorageImageBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eStorageImage);
    writeDescSet.setImageInfo(descImageInfo_);
    return writeDescSet;
}

std::expected<void, Error> StorageImageBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(mmapBox.getMapPtr(), pSrc, extent_.size());

    return {};
}

std::expected<void, Error> StorageImageBox::uploadWithRoi(const std::byte* pSrc, const Roi roi,
                                                              const size_t bufferRowPitch) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    size_t srcOffset = 0;
    size_t dstOffset = extent_.calculateBufferOffset(roi.offset());
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = pSrc + srcOffset;
        std::byte* dstCursor = (std::byte*)mmapBox.getMapPtr() + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent_.bpp());
        srcOffset += bufferRowPitch;
        dstOffset += extent_.rowPitch();
    }

    return {};
}

std::expected<void, Error> StorageImageBox::download(std::byte* pDst) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(pDst, mmapBox.getMapPtr(), extent_.size());

    return {};
}

std::expected<void, Error> StorageImageBox::downloadWithRoi(std::byte* pDst, const Roi roi,
                                                                const size_t bufferRowPitch) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    size_t srcOffset = extent_.calculateBufferOffset(roi.offset());
    size_t dstOffset = 0;
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = (std::byte*)mmapBox.getMapPtr() + srcOffset;
        std::byte* dstCursor = pDst + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent_.bpp());
        srcOffset += extent_.rowPitch();
        dstOffset += bufferRowPitch;
    }

    return {};
}

}  // namespace vkc
