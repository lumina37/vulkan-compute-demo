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

    using TStagingBufferBoxRef = std::reference_wrapper<StagingBufferBox>;
    using TStorageImageBoxRef = std::reference_wrapper<StorageImageBox>;
    using TPresentImageBoxRef = std::reference_wrapper<PresentImageBox>;

    template <CImageBox TImageBox>
    void recordPrepareReceiveBeforeDispatch(std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept;

    template <CImageBox TImageBox>
    void recordPrepareReceiveAfterDispatch(std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept;

    template <CImageBox TImageBox>
    void recordPrepareShaderRead(std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept;

    void recordPrepareShaderWrite(std::span<const TStorageImageBoxRef> imageBoxRefs) noexcept;
    void recordPrepareSendBeforeDispatch(std::span<const TStorageImageBoxRef> imageBoxRefs) noexcept;
    void recordPrepareSendAfterDispatch(std::span<const TStorageImageBoxRef> imageBoxRefs) noexcept;
    void recordPreparePresent(std::span<const TPresentImageBoxRef> imageBoxRefs) noexcept;

    void recordDispatch(int groupNumX, int groupNumY) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStagingToImage(const StagingBufferBox& stagingBufferBox, TImageBox& imageBox) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStagingToImageWithRoi(const StagingBufferBox& stagingBufferBox, TImageBox& imageBox,
                                         const Roi& roi) noexcept;

    void recordCopyImageToStaging(const StorageImageBox& imageBox, StagingBufferBox& stagingBufferBox) noexcept;
    void recordCopyImageToStagingWithRoi(const StorageImageBox& imageBox, StagingBufferBox& stagingBufferBox,
                                         const Roi& roi) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStorageToAnother(const StorageImageBox& srcImageBox, TImageBox& dstImageBox) noexcept;

    template <CImageBox TImageBox>
    void recordCopyStorageToAnotherWithRoi(const StorageImageBox& srcImageBox, TImageBox& dstImageBox,
                                           const Roi& roi) noexcept;

    void recordWaitDownloadComplete(std::span<const TStagingBufferBoxRef> stagingBufferBoxRefs) noexcept;

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
    const std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setImage(box.getVkImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordPrepareReceiveAfterDispatch(
    const std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getVkImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordPrepareShaderRead(
    const std::span<const std::reference_wrapper<TImageBox>> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderRead;
    constexpr vk::ImageLayout newImageLayout = std::is_same_v<TImageBox, SampledImageBox>
                                                   ? vk::ImageLayout::eShaderReadOnlyOptimal
                                                   : vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageBox> boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getVkImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStagingToImage(const StagingBufferBox& stagingBufferBox,
                                                TImageBox& imageBox) noexcept {
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setImageExtent(imageBox.getExtent().extent3D());

    commandBuffer_.copyBufferToImage(stagingBufferBox.getVkBuffer(), imageBox.getVkImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStagingToImageWithRoi(const StagingBufferBox& stagingBufferBox, TImageBox& imageBox,
                                                       const Roi& roi) noexcept {
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = imageBox.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyBufferToImage(stagingBufferBox.getVkBuffer(), imageBox.getVkImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStorageToAnother(const StorageImageBox& srcImageBox, TImageBox& dstImageBox) noexcept {
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setDstSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setExtent(srcImageBox.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageBox.getVkImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageBox.getVkImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageBox TImageBox>
void CommandBufferBox::recordCopyStorageToAnotherWithRoi(const StorageImageBox& srcImageBox, TImageBox& dstImageBox,
                                                         const Roi& roi) noexcept {
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setSrcOffset(roi.offset3D());
    copyRegion.setDstSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setDstOffset(roi.offset3D());
    copyRegion.setExtent(srcImageBox.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageBox.getVkImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageBox.getVkImage(),
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
