#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.hpp"
#endif

namespace vkc {

StorageImageBox::StorageImageBox(ImageBox&& imageBox, ImageViewBox&& imageViewBox, MemoryBox&& imageMemoryBox,
                                 const vk::DescriptorImageInfo& descImageInfo) noexcept
    : imageBox_(std::move(imageBox)),
      imageViewBox_(std::move(imageViewBox)),
      imageMemoryBox_(std::move(imageMemoryBox)),
      descImageInfo_(descImageInfo),
      accessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined) {}

std::expected<StorageImageBox, Error> StorageImageBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                              const Extent& extent, StorageType imageType) noexcept {
    vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst;
    constexpr vk::ImageLayout imageLayout = vk::ImageLayout::eGeneral;

    if (StorageType::ReadWrite == imageType) {
        imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
    }

    auto imageBoxRes = ImageBox::create(pDeviceBox, extent, imageUsage);
    if (!imageBoxRes) return std::unexpected{std::move(imageBoxRes.error())};
    ImageBox& imageBox = imageBoxRes.value();

    auto memoryBoxRes =
        MemoryBox::create(pDeviceBox, imageBox.getMemoryRequirements(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox& memoryBox = memoryBoxRes.value();

    auto bindRes = imageBox.bind(memoryBox);
    if (!bindRes) return std::unexpected{std::move(bindRes.error())};

    auto imageViewBoxRes = ImageViewBox::create(pDeviceBox, imageBox);
    if (!imageViewBoxRes) return std::unexpected{std::move(imageViewBoxRes.error())};
    ImageViewBox& imageViewBox = imageViewBoxRes.value();

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageViewBox.getVkImageView());
    descImageInfo.setImageLayout(imageLayout);

    return StorageImageBox{std::move(imageBox), std::move(imageViewBox), std::move(memoryBox), descImageInfo};
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

    std::memcpy(mapPtr, pSrc, getExtent().size());

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
        std::memcpy(dstCursor, srcCursor, roi.extent().width * getExtent().bpp());
        srcOffset += bufferRowPitch;
        dstOffset += getExtent().rowPitch();
    }

    imageMemoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StorageImageBox::download(std::byte* pDst) noexcept {
    auto mmapRes = imageMemoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(pDst, mapPtr, getExtent().size());

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
        std::memcpy(dstCursor, srcCursor, roi.extent().width * getExtent().bpp());
        srcOffset += getExtent().rowPitch();
        dstOffset += bufferRowPitch;
    }

    imageMemoryBox_.memUnmap();

    return {};
}

}  // namespace vkc
