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
#    include "vkc/resource/sampled_image.hpp"
#endif

namespace vkc {

SampledImageBox::SampledImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, Extent extent, vk::Image image,
                                 vk::ImageView imageView, MemoryBox&& imageMemoryBox, vk::Buffer stagingBuffer,
                                 MemoryBox&& stagingMemoryBox, vk::DescriptorImageInfo descImageInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      extent_(extent),
      image_(image),
      imageView_(imageView),
      imageMemoryBox_(std::move(imageMemoryBox)),
      stagingBuffer_(stagingBuffer),
      stagingMemoryBox_(std::move(stagingMemoryBox)),
      descImageInfo_(descImageInfo),
      imageAccessMask_(vk::AccessFlagBits::eNone),
      imageLayout_(vk::ImageLayout::eUndefined),
      stagingAccessMask_(vk::AccessFlagBits::eNone) {}

SampledImageBox::SampledImageBox(SampledImageBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      extent_(rhs.extent_),
      image_(std::exchange(rhs.image_, nullptr)),
      imageView_(std::exchange(rhs.imageView_, nullptr)),
      imageMemoryBox_(std::move(rhs.imageMemoryBox_)),
      stagingBuffer_(std::exchange(rhs.stagingBuffer_, nullptr)),
      stagingMemoryBox_(std::move(rhs.stagingMemoryBox_)),
      descImageInfo_(std::exchange(rhs.descImageInfo_, {})),
      imageAccessMask_(rhs.imageAccessMask_),
      imageLayout_(rhs.imageLayout_),
      stagingAccessMask_(rhs.stagingAccessMask_) {}

SampledImageBox::~SampledImageBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();

    if (stagingBuffer_ != nullptr) {
        device.destroyBuffer(stagingBuffer_);
        stagingBuffer_ = nullptr;
    }
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

std::expected<SampledImageBox, Error> SampledImageBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                              const Extent& extent) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    constexpr vk::ImageUsageFlags imageUsage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    constexpr vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;
    constexpr vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

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
    MemoryBox imageMemoryBox = std::move(imageMemoryBoxRes.value());

    const auto bindRes = device.bindImageMemory(image, imageMemoryBox.getDeviceMemory(), 0);
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

    // Staging Memory
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(extent.size());
    bufferInfo.setUsage(bufferUsage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [stagingBufferRes, stagingBuffer] = device.createBuffer(bufferInfo);
    if (stagingBufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, stagingBufferRes}};
    }

    const vk::MemoryRequirements stagingMemoryReq = _hp::getMemoryRequirements(*pDeviceBox, stagingBuffer);
    auto stagingMemoryBoxRes =
        MemoryBox::create(pDeviceBox, stagingMemoryReq,
                          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!stagingMemoryBoxRes) return std::unexpected{std::move(stagingMemoryBoxRes.error())};
    MemoryBox stagingMemoryBox = std::move(stagingMemoryBoxRes.value());

    const auto bindStagingRes = device.bindBufferMemory(stagingBuffer, stagingMemoryBox.getDeviceMemory(), 0);
    if (bindStagingRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindStagingRes}};
    }

    // Descriptor Image Info
    vk::DescriptorImageInfo descImageInfo;
    descImageInfo.setImageView(imageView);
    descImageInfo.setImageLayout(imageLayout);

    return SampledImageBox{
        std::move(pDeviceBox),       extent,       image, imageView, std::move(imageMemoryBox), stagingBuffer,
        std::move(stagingMemoryBox), descImageInfo};
}

vk::WriteDescriptorSet SampledImageBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setImageInfo(descImageInfo_);
    return writeDescSet;
}

std::expected<void, Error> SampledImageBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemoryBox_.getDeviceMemory(), extent_.size());
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(mmapBox.getMapPtr(), pSrc, extent_.size());

    return {};
}

std::expected<void, Error> SampledImageBox::uploadWithRoi(const std::byte* pSrc, const Roi roi,
                                                          const size_t bufferRowPitch) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, stagingMemoryBox_.getDeviceMemory(), extent_.size());
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

}  // namespace vkc
