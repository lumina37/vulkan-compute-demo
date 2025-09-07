#pragma once

#include <expected>
#include <functional>
#include <memory>
#include <span>

#include "vkc/command/concepts.hpp"
#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource.hpp"

namespace vkc {

class CommandBufferBox {
    CommandBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, std::shared_ptr<CommandPoolBox>&& pCommandPoolBox,
                     vk::CommandBuffer commandBuffer) noexcept;

public:
    CommandBufferBox(const CommandBufferBox&) = delete;
    CommandBufferBox(CommandBufferBox&& rhs) noexcept;
    ~CommandBufferBox() noexcept;

    [[nodiscard]] static std::expected<CommandBufferBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, std::shared_ptr<CommandPoolBox> pCommandPoolBox) noexcept;

    [[nodiscard]] vk::CommandBuffer getCommandBuffer() const noexcept { return commandBuffer_; }

    void bindPipeline(PipelineBox& pipelineBox) noexcept;
    void bindDescSets(DescSetsBox& descSetsBox, const PipelineLayoutBox& pipelineLayoutBox,
                      vk::PipelineBindPoint bindPoint) noexcept;

    template <typename TPc>
    void pushConstant(const PushConstantBox<TPc>& pushConstantBox, const PipelineLayoutBox& pipelineLayoutBox) noexcept;

    [[nodiscard]] std::expected<void, Error> begin() noexcept;

    using TStorageImageBoxRef = std::reference_wrapper<StorageImageBox>;
    using TPresentImageBoxRef = std::reference_wrapper<PresentImageBox>;

    template <CImageBox TImageBox>
    void recordPrepareReceiveBeforeDispatch(
        std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept;

    template <CImageBox TImageBox>
    void recordPrepareReceiveAfterDispatch(std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept;

    template <CImageBox TImageBox>
    void recordSrcPrepareShaderRead(std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept;

    void recordDstPrepareShaderWrite(std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept;
    void recordPrepareSendBeforeDispatch(std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept;
    void recordPrepareSendAfterDispatch(std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept;
    void recordPreparePresent(std::span<const TPresentImageBoxRef> imageBoxRefs) noexcept;

    void recordDispatch(int groupNumX, int groupNumY) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStagingToSrc(const TImageBox& srcImageBox) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStagingToSrcWithRoi(const TImageBox& srcImageBox, Roi roi) noexcept;

    void recordCopyDstToStaging(StorageImageBox& dstImageBox) noexcept;
    void recordCopyDstToStagingWithRoi(StorageImageBox& dstImageBox, Roi roi) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStorageToAnother(const StorageImageBox& srcImageBox, TImageBox& dstImageBox) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStorageToAnotherWithRoi(const StorageImageBox& srcImageBox, TImageBox& dstImageBox,
                                           Roi roi) noexcept;

    void recordWaitDownloadComplete(std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept;

    template <CQueryPoolBox TQueryPoolBox>
    void recordResetQueryPool(TQueryPoolBox& queryPoolBox) noexcept;

    [[nodiscard]] std::expected<void, Error> recordTimestampStart(TimestampQueryPoolBox& queryPoolBox,
                                                                  vk::PipelineStageFlagBits pipelineStage) noexcept;
    [[nodiscard]] std::expected<void, Error> recordTimestampEnd(TimestampQueryPoolBox& queryPoolBox,
                                                                vk::PipelineStageFlagBits pipelineStage) noexcept;
    [[nodiscard]] std::expected<void, Error> recordPerfQueryStart(PerfQueryPoolBox& queryPoolBox) noexcept;
    [[nodiscard]] std::expected<void, Error> recordPerfQueryEnd(PerfQueryPoolBox& queryPoolBox) noexcept;

    [[nodiscard]] std::expected<void, Error> end() noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;
    std::shared_ptr<CommandPoolBox> pCommandPoolBox_;

    vk::CommandBuffer commandBuffer_;

    static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
};

template <typename TPc>
void CommandBufferBox::pushConstant(const PushConstantBox<TPc>& pushConstantBox,
                                    const PipelineLayoutBox& pipelineLayoutBox) noexcept {
    vk::PipelineLayout piplelineLayout = pipelineLayoutBox.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, pushConstantBox.getPushConstantRange().stageFlags, 0, sizeof(TPc),
                                 pushConstantBox.getPPushConstant());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordPrepareReceiveBeforeDispatch(
    const std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordPrepareReceiveAfterDispatch(
    const std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordSrcPrepareShaderRead(
    const std::span<const std::reference_wrapper<TImageBox>> srcImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderRead;
    constexpr vk::ImageLayout newImageLayout = std::is_same_v<TImageBox, SampledImageBox>
                                                   ? vk::ImageLayout::eShaderReadOnlyOptimal
                                                   : vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStagingToSrc(const TImageBox& srcImageBox) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(srcImageBox.getExtent().extent3D());

    commandBuffer_.copyBufferToImage(srcImageBox.getStagingBuffer(), srcImageBox.getImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStagingToSrcWithRoi(const TImageBox& srcImageBox, const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = srcImageBox.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyBufferToImage(srcImageBox.getStagingBuffer(), srcImageBox.getImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStorageToAnother(const StorageImageBox& srcImageBox, TImageBox& dstImageBox) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(subresourceLayers);
    copyRegion.setDstSubresource(subresourceLayers);
    copyRegion.setExtent(srcImageBox.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageBox.getImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageBox.getImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStorageToAnotherWithRoi(const StorageImageBox& srcImageBox, TImageBox& dstImageBox,
                                                         const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(subresourceLayers);
    copyRegion.setSrcOffset(roi.offset3D());
    copyRegion.setDstSubresource(subresourceLayers);
    copyRegion.setDstOffset(roi.offset3D());
    copyRegion.setExtent(srcImageBox.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageBox.getImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageBox.getImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CQueryPoolBox TQueryPoolBox>
void CommandBufferBox::recordResetQueryPool(TQueryPoolBox& queryPoolBox) noexcept {
    vk::QueryPool queryPool = queryPoolBox.getQueryPool();
    queryPoolBox.resetQueryIndex();
    commandBuffer_.resetQueryPool(queryPool, 0, queryPoolBox.getQueryCount());
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.cpp"
#endif
