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

PresentImageManager::PresentImageManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, Extent extent, vk::Image image,
                                         vk::ImageView imageView, vk::Buffer stagingBuffer,
                                         vk::DeviceMemory stagingMemory, vk::DescriptorImageInfo descImageInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      stagingBuffer_(stagingBuffer),
      stagingMemory_(stagingMemory),
      descImageInfo_(descImageInfo),
      imageAccessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined),
      stagingAccessMask_(vk::AccessFlagBits::eNone) {}

PresentImageManager::PresentImageManager(PresentImageManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      stagingBuffer_(std::exchange(rhs.stagingBuffer_, nullptr)),
      stagingMemory_(std::exchange(rhs.stagingMemory_, nullptr)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})),
      imageAccessMask_(rhs.imageAccessMask_),
      imageLayout_(rhs.imageLayout_),
      stagingAccessMask_(rhs.imageAccessMask_) {}

PresentImageManager::~PresentImageManager() noexcept {
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
    image_ = nullptr;
    descImageInfo_.setImageView(nullptr);
}

std::expected<PresentImageManager, Error> PresentImageManager::create(const PhyDeviceManager& phyDeviceMgr,
                                                                      std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                      vk::Image image, const Extent& extent) noexcept {
    vk::Device device = pDeviceMgr->getDevice();

    constexpr vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
    constexpr vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

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

    return PresentImageManager{std::move(pDeviceMgr), extent,        image,        imageView,
                               stagingBuffer,         stagingMemory, descImageInfo};
}

std::expected<void, Error> PresentImageManager::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = _hp::MemMapManager::create(pDeviceMgr_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapMgr = mmapRes.value();

    std::memcpy(mmapMgr.getMapPtr(), pSrc, extent_.size());

    return {};
}

std::expected<void, Error> PresentImageManager::uploadWithRoi(const std::byte* pSrc, const Roi roi,
                                                              const size_t bufferRowPitch) noexcept {
    auto mmapRes = _hp::MemMapManager::create(pDeviceMgr_, stagingMemory_, extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapMgr = mmapRes.value();

    size_t srcOffset = 0;
    size_t dstOffset = extent_.calculateBufferOffset(roi.offset());
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = pSrc + srcOffset;
        std::byte* dstCursor = (std::byte*)mmapMgr.getMapPtr() + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent_.bpp());
        srcOffset += bufferRowPitch;
        dstOffset += extent_.rowPitch();
    }

    return {};
}

}  // namespace vkc
