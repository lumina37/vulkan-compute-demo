#pragma once

#include <expected>
#include <functional>
#include <memory>
#include <span>

#include "vkc/command/concepts.hpp"
#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource.hpp"
#include "vkc/sync/fence.hpp"
#include "vkc/sync/semaphore.hpp"

namespace vkc {

struct BlockSize {
    using Tv = uint32_t;
    Tv x, y, z;
};

class CommandBufferManager {
    CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                         std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr,
                         vk::CommandBuffer commandBuffer) noexcept;

    [[nodiscard]] std::expected<void, Error> _submit(vk::Queue queue, vk::Semaphore waitSemaphore,
                                                     vk::PipelineStageFlags waitDstStage, vk::Semaphore signalSemaphore,
                                                     vk::Fence fence) noexcept;

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

    template <CImageManager TImageManager>
    void recordSrcPrepareTranfer(std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept;

    template <CImageManager TImageManager>
    void recordSrcPrepareShaderRead(std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept;

    void recordDstPrepareShaderWrite(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;
    void recordDispatch(vk::Extent2D extent, BlockSize blockSize) noexcept;
    void recordDstPrepareTransfer(std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStagingToSrc(const TImageManager& srcImageMgr) noexcept;

    template <CImageManager TImageManager>
    void recordCopyStagingToSrcWithRoi(const TImageManager& srcImageMgr, Roi roi) noexcept;

    void recordCopyDstToStaging(StorageImageManager& dstImageMgr) noexcept;
    void recordCopyDstToStagingWithRoi(StorageImageManager& dstImageMgr, Roi roi) noexcept;
    void recordCopyStorageToSampled(const StorageImageManager& srcImageMgr, SampledImageManager& dstImageMgr) noexcept;
    void recordCopyStorageToSampledWithRoi(const StorageImageManager& srcImageMgr, SampledImageManager& dstImageMgr,
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

    [[nodiscard]] std::expected<void, Error> submitAndWaitPreTask(QueueManager& queueMgr,
                                                                  const SemaphoreManager& waitSemaphoreMgr,
                                                                  vk::PipelineStageFlags waitDstStage,
                                                                  SemaphoreManager& signalSemaphoreMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submitAndWaitPreTask(QueueManager& queueMgr,
                                                                  const SemaphoreManager& waitSemaphoreMgr,
                                                                  vk::PipelineStageFlags waitDstStage,
                                                                  FenceManager& fenceMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(QueueManager& queueMgr,
                                                    SemaphoreManager& signalSemaphoreMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(QueueManager& queueMgr, FenceManager& fenceMgr) noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;
    std::shared_ptr<CommandPoolManager> pCommandPoolMgr_;

    vk::CommandBuffer commandBuffer_;

    static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
};

template <typename TPc>
void CommandBufferManager::pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) noexcept {
    const auto& piplelineLayout = pipelineLayoutMgr.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, pushConstantMgr.getPushConstantRange().stageFlags, 0, sizeof(TPc),
                                 pushConstantMgr.getPPushConstant());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordSrcPrepareTranfer(
    const std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept {
    vk::ImageMemoryBarrier uploadConvBarrierTemplate;
    uploadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    uploadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    uploadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    uploadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    uploadConvBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    uploadConvBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    uploadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&uploadConvBarrierTemplate](const std::reference_wrapper<TImageManager> mgrRef) {
        vk::ImageMemoryBarrier uploadConvBarrier = uploadConvBarrierTemplate;
        uploadConvBarrier.setImage(mgrRef.get().getImage());
        return uploadConvBarrier;
    };

    const auto uploadConvBarriers = srcImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)uploadConvBarriers.size(),
                                   uploadConvBarriers.data());
}

template <CImageManager TImageManager>
void CommandBufferManager::recordSrcPrepareShaderRead(
    const std::span<const std::reference_wrapper<TImageManager>> srcImageMgrRefs) noexcept {
    vk::ImageMemoryBarrier shaderCompatibleBarrierTemplate;
    shaderCompatibleBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    shaderCompatibleBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    shaderCompatibleBarrierTemplate.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    if constexpr (std::is_same_v<TImageManager, SampledImageManager>) {
        shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    } else {
        shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eGeneral);
    }
    shaderCompatibleBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&shaderCompatibleBarrierTemplate](const std::reference_wrapper<TImageManager> mgrRef) {
        vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
        shaderCompatibleBarrier.setImage(mgrRef.get().getImage());
        return shaderCompatibleBarrier;
    };

    const auto shaderCompatibleBarriers = srcImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)shaderCompatibleBarriers.size(), shaderCompatibleBarriers.data());
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
    const auto& imageExtent = srcImageMgr.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyBufferToImage(srcImageMgr.getStagingBuffer(), srcImageMgr.getImage(),
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
