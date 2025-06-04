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
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource.hpp"

namespace vkc {

struct BlockSize {
    using Tv = uint32_t;
    Tv x, y, z;
};

class CommandBufferManager {
    CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                         std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr,
                         vk::CommandBuffer commandBuffer) noexcept;

public:
    CommandBufferManager(CommandBufferManager&& rhs) noexcept;
    ~CommandBufferManager() noexcept;

    [[nodiscard]] static std::expected<CommandBufferManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::shared_ptr<CommandPoolManager> pCommandPoolMgr) noexcept;

    [[nodiscard]] vk::CommandBuffer getCommandBuffer() const noexcept { return commandBuffer_; }

    void bindPipeline(PipelineManager& pipelineMgr) noexcept;
    void bindDescSets(DescSetsManager& descSetsMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                      vk::PipelineBindPoint bindPoint) noexcept;

    template <typename TPc>
    void pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                      const PipelineLayoutManager& pipelineLayoutMgr) noexcept;

    [[nodiscard]] std::expected<void, Error> begin() noexcept;

    using TStorageImageMgrRef = std::reference_wrapper<StorageImageManager>;
    using TPresentImageMgrRef = std::reference_wrapper<PresentImageManager>;

    template <CImageManager TImageManager>
    void recordPrepareReceiveBeforeDispatch(
        std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept;

    template <CImageManager TImageManager>
    void recordPrepareReceiveAfterDispatch(
        std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept;

    template <CImageManager TImageManager>
    void recordSrcPrepareShaderRead(std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept;

    void recordDstPrepareShaderWrite(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;
    void recordPrepareSendBeforeDispatch(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;
    void recordPrepareSendAfterDispatch(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;
    void recordPreparePresent(std::span<const TPresentImageMgrRef> imageMgrRefs) noexcept;

    void recordDispatch(vk::Extent2D extent, BlockSize blockSize) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStagingToSrc(const TImageManager& srcImageMgr) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStagingToSrcWithRoi(const TImageManager& srcImageMgr, Roi roi) noexcept;

    void recordCopyDstToStaging(StorageImageManager& dstImageMgr) noexcept;
    void recordCopyDstToStagingWithRoi(StorageImageManager& dstImageMgr, Roi roi) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStorageToAnother(const StorageImageManager& srcImageMgr, TImageManager& dstImageMgr) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStorageToAnotherWithRoi(const StorageImageManager& srcImageMgr, TImageManager& dstImageMgr,
                                           Roi roi) noexcept;

    void recordWaitDownloadComplete(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;

    template <typename TQueryPoolManager>
        requires CQueryPoolManager<TQueryPoolManager>
    void recordResetQueryPool(TQueryPoolManager& queryPoolMgr) noexcept;

    [[nodiscard]] std::expected<void, Error> recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr,
                                                                  vk::PipelineStageFlagBits pipelineStage) noexcept;
    [[nodiscard]] std::expected<void, Error> recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr,
                                                                vk::PipelineStageFlagBits pipelineStage) noexcept;

    [[nodiscard]] std::expected<void, Error> end() noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;
    std::shared_ptr<CommandPoolManager> pCommandPoolMgr_;

    vk::CommandBuffer commandBuffer_;

    static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
};

template <typename TPc>
void CommandBufferManager::pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) noexcept {
    vk::PipelineLayout piplelineLayout = pipelineLayoutMgr.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, pushConstantMgr.getPushConstantRange().stageFlags, 0, sizeof(TPc),
                                 pushConstantMgr.getPPushConstant());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordPrepareReceiveBeforeDispatch(
    const std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept {
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

    const auto fillout = [&](const std::reference_wrapper<TImageManager> mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordPrepareReceiveAfterDispatch(
    const std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferDstOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageManager> mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(mgr.getImageAccessMask());
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordSrcPrepareShaderRead(
    const std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderRead;
    constexpr vk::ImageLayout newImageLayout = std::is_same_v<TImageManager, SampledImageManager>
                                                   ? vk::ImageLayout::eShaderReadOnlyOptimal
                                                   : vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const std::reference_wrapper<TImageManager> mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(mgr.getImageAccessMask());
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = srcImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordCopyStagingToSrc(const TImageManager& srcImageMgr) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(srcImageMgr.getExtent().extent3D());

    commandBuffer_.copyBufferToImage(srcImageMgr.getStagingBuffer(), srcImageMgr.getImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageManager TImageManager>
void CommandBufferManager::recordCopyStagingToSrcWithRoi(const TImageManager& srcImageMgr, const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = srcImageMgr.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyBufferToImage(srcImageMgr.getStagingBuffer(), srcImageMgr.getImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageManager TImageManager>
void CommandBufferManager::recordCopyStorageToAnother(const StorageImageManager& srcImageMgr,
                                                      TImageManager& dstImageMgr) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(subresourceLayers);
    copyRegion.setDstSubresource(subresourceLayers);
    copyRegion.setExtent(srcImageMgr.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageMgr.getImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <CImageManager TImageManager>
void CommandBufferManager::recordCopyStorageToAnotherWithRoi(const StorageImageManager& srcImageMgr,
                                                             TImageManager& dstImageMgr, const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource(subresourceLayers);
    copyRegion.setSrcOffset(roi.offset3D());
    copyRegion.setDstSubresource(subresourceLayers);
    copyRegion.setDstOffset(roi.offset3D());
    copyRegion.setExtent(srcImageMgr.getExtent().extent3D());

    commandBuffer_.copyImage(srcImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal, dstImageMgr.getImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
}

template <typename TQueryPoolManager>
    requires CQueryPoolManager<TQueryPoolManager>
void CommandBufferManager::recordResetQueryPool(TQueryPoolManager& queryPoolMgr) noexcept {
    vk::QueryPool queryPool = queryPoolMgr.getQueryPool();
    queryPoolMgr.resetQueryIndex();
    commandBuffer_.resetQueryPool(queryPool, 0, queryPoolMgr.getQueryCount());
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.cpp"
#endif
