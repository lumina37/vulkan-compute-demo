#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/buffer.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.hpp"
#endif

namespace vkc {

StorageImageBox::StorageImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, const Extent& extent, vk::Image image,
                                 vk::ImageView imageView, MemoryBox&& imageMemoryBox,
                                 const vk::DescriptorImageInfo& descImageInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      imageMemoryBox_(std::move(imageMemoryBox)),
      descImageInfo_(descImageInfo),
      imageAccessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined) {}

StorageImageBox::StorageImageBox(StorageImageBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageMemoryBox_(std::move(rhs.imageMemoryBox_)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})),
      imageAccessMask_(rhs.imageAccessMask_),
      imageLayout_(rhs.imageLayout_) {}

StorageImageBox::~StorageImageBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();

    if (imageView_ != nullptr) {
        device.destroyImageView(imageView_);
        imageView_ = nullptr;
    }
    if (image_ != nullptr) {
        device.destroyImage(image_);
        image_ = nullptr;
    }
    descImageInfo_.setImageView(nullptr);
}

std::expected<StorageImageBox, Error> StorageImageBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                              const Extent& extent, StorageType imageType) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
    vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
    constexpr vk::ImageLayout imageLayout = vk::ImageLayout::eGeneral;

    if (StorageType::ReadWrite == imageType) {
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
        return std::unexpected{Error{ECate::eVk, imageRes}};
    }

    // Image Memory
    const vk::MemoryRequirements imageMemoryReq = _hp::getMemoryRequirements(*pDeviceBox, image);
    auto imageMemoryBoxRes = MemoryBox::create(pDeviceBox, imageMemoryReq, vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!imageMemoryBoxRes) return std::unexpected{std::move(imageMemoryBoxRes.error())};
    MemoryBox& imageMemoryBox = imageMemoryBoxRes.value();

    const auto bindRes = device.bindImageMemory(image, imageMemoryBox.getVkDeviceMemory(), 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
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
        return std::unexpected{Error{ECate::eVk, imageViewRes}};
    }

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageView);
    descImageInfo.setImageLayout(imageLayout);

    return StorageImageBox{std::move(pDeviceBox), extent, image, imageView, std::move(imageMemoryBox), descImageInfo};
}

vk::WriteDescriptorSet StorageImageBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setImageInfo(descImageInfo_);
    return writeDescSet;
}

std::expected<void, Error> StorageImageBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = imageMemoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(mapPtr, pSrc, extent_.size());

    imageMemoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StorageImageBox::uploadWithRoi(const std::byte* pSrc, const Roi& roi, size_t bufferOffset,
                                                          size_t bufferRowPitch) noexcept {
    auto mmapRes = imageMemoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    size_t srcOffset = 0;
    size_t dstOffset = bufferOffset;
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = pSrc + srcOffset;
        std::byte* dstCursor = (std::byte*)mapPtr + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent_.bpp());
        srcOffset += bufferRowPitch;
        dstOffset += extent_.rowPitch();
    }

    imageMemoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StorageImageBox::download(std::byte* pDst) noexcept {
    auto mmapRes = imageMemoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(pDst, mapPtr, extent_.size());

    imageMemoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StorageImageBox::downloadWithRoi(std::byte* pDst, const Roi& roi, size_t bufferOffset,
                                                            const size_t bufferRowPitch) noexcept {
    auto mmapRes = imageMemoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    size_t srcOffset = bufferOffset;
    size_t dstOffset = 0;
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = (std::byte*)mapPtr + srcOffset;
        std::byte* dstCursor = pDst + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent_.bpp());
        srcOffset += extent_.rowPitch();
        dstOffset += bufferRowPitch;
    }

    imageMemoryBox_.memUnmap();

    return {};
}

}  // namespace vkc
