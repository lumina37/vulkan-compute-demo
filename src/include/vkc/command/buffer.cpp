#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

CommandBufferManager::CommandBufferManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                           const std::shared_ptr<CommandPoolManager>& pCommandPoolMgr)
    : pDeviceMgr_(pDeviceMgr), pCommandPoolMgr_(pCommandPoolMgr) {
    auto& device = pDeviceMgr->getDevice();
    auto& commandPool = pCommandPoolMgr->getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    commandBuffer_ = device.allocateCommandBuffers(allocInfo)[0];
    completeFence_ = device.createFence({});
}

CommandBufferManager::~CommandBufferManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    auto& commandPool = pCommandPoolMgr_->getCommandPool();
    device.freeCommandBuffers(commandPool, commandBuffer_);
    device.destroyFence(completeFence_);
}

void CommandBufferManager::bindPipeline(PipelineManager& pipelineMgr) {
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineMgr.getPipeline());
}

void CommandBufferManager::bindDescSets(DescSetsManager& descSetsMgr, const PipelineLayoutManager& pipelineLayoutMgr) {
    auto& descSets = descSetsMgr.getDescSets();
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutMgr.getPipelineLayout(), 0,
                                      (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
}

void CommandBufferManager::begin() {
    commandBuffer_.reset();

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer_.begin(cmdBufBeginInfo);
}

void CommandBufferManager::recordSrcPrepareTranfer(const std::span<const TImageMgrCRef> srcImageMgrRefs) {
    vk::ImageMemoryBarrier uploadConvBarrierTemplate;
    uploadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    uploadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    uploadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    uploadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    uploadConvBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    uploadConvBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    uploadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto uploadConvBarriers = srcImageMgrRefs |
                                    rgs::views::transform([&uploadConvBarrierTemplate](const TImageMgrCRef mgrRef) {
                                        vk::ImageMemoryBarrier uploadConvBarrier = uploadConvBarrierTemplate;
                                        uploadConvBarrier.setImage(mgrRef.get().getImage());
                                        return uploadConvBarrier;
                                    }) |
                                    rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)uploadConvBarriers.size(),
                                   uploadConvBarriers.data());
}

void CommandBufferManager::recordUploadToSrc(const std::span<const TImageMgrCRef> srcImageMgrRefs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegionTemplate;
    copyRegionTemplate.setImageSubresource(subresourceLayers);

    for (const auto mgrRef : srcImageMgrRefs) {
        const auto& mgr = mgrRef.get();
        vk::BufferImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setImageExtent(mgr.getExtent().extent3D());
        commandBuffer_.copyBufferToImage(mgr.getStagingBuffer(), mgr.getImage(), vk::ImageLayout::eTransferDstOptimal,
                                         1, &copyRegion);
    }
}

void CommandBufferManager::recordImageCopy(const std::span<const ImageManagerPair> imageMgrPairs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegionTemplate;
    copyRegionTemplate.setSrcSubresource(subresourceLayers);
    copyRegionTemplate.setDstSubresource(subresourceLayers);

    for (const auto mgrPair : imageMgrPairs) {
        vk::ImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setExtent(mgrPair.copyFrom.getExtent().extent3D());
        commandBuffer_.copyImage(mgrPair.copyFrom.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 mgrPair.copyTo.getImage(), vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
    }
}

void CommandBufferManager::recordSrcPrepareShaderRead(const std::span<const TImageMgrCRef> srcImageMgrRefs) {
    vk::ImageMemoryBarrier shaderCompatibleBarrierTemplate;
    shaderCompatibleBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    shaderCompatibleBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    shaderCompatibleBarrierTemplate.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    shaderCompatibleBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto shaderCompatibleBarriers =
        srcImageMgrRefs | rgs::views::transform([&shaderCompatibleBarrierTemplate](const TImageMgrCRef mgrRef) {
            vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
            shaderCompatibleBarrier.setImage(mgrRef.get().getImage());
            return shaderCompatibleBarrier;
        }) |
        rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)shaderCompatibleBarriers.size(), shaderCompatibleBarriers.data());
}

void CommandBufferManager::recordDstPrepareShaderWrite(const std::span<const TImageMgrCRef> dstImageMgrRefs) {
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(1);

    // Shader Compatible Image Layout
    vk::ImageMemoryBarrier shaderCompatibleBarrierTemplate;
    shaderCompatibleBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    shaderCompatibleBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    shaderCompatibleBarrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eGeneral);
    shaderCompatibleBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    shaderCompatibleBarrierTemplate.setSubresourceRange(subresourceRange);

    const auto shaderCompatibleBarriers =
        dstImageMgrRefs | rgs::views::transform([&shaderCompatibleBarrierTemplate](const TImageMgrCRef mgrRef) {
            vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
            shaderCompatibleBarrier.setImage(mgrRef.get().getImage());
            return shaderCompatibleBarrier;
        }) |
        rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)shaderCompatibleBarriers.size(), shaderCompatibleBarriers.data());
}

void CommandBufferManager::recordDispatch(const Extent extent, const BlockSize blockSize) {
    uint32_t groupSizeX = (extent.width() + (blockSize.x - 1)) / blockSize.x;
    uint32_t groupSizeY = (extent.height() + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferManager::recordDstPrepareTransfer(const std::span<const TImageMgrCRef> dstImageMgrRefs) {
    vk::ImageMemoryBarrier downloadConvBarrierTemplate;
    downloadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    downloadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    downloadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eGeneral);
    downloadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    downloadConvBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadConvBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto downloadConvBarriers = dstImageMgrRefs |
                                      rgs::views::transform([&downloadConvBarrierTemplate](const TImageMgrCRef mgrRef) {
                                          vk::ImageMemoryBarrier downloadConvBarrier = downloadConvBarrierTemplate;
                                          downloadConvBarrier.setImage(mgrRef.get().getImage());
                                          return downloadConvBarrier;
                                      }) |
                                      rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)downloadConvBarriers.size(), downloadConvBarriers.data());
}

void CommandBufferManager::recordDownloadToDst(std::span<const TImageMgrCRef> dstImageMgrRefs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegionTemplate;
    copyRegionTemplate.setImageSubresource(subresourceLayers);

    for (const auto mgrRef : dstImageMgrRefs) {
        const auto& mgr = mgrRef.get();
        vk::BufferImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setImageExtent(mgr.getExtent().extent3D());
        commandBuffer_.copyImageToBuffer(mgr.getImage(), vk::ImageLayout::eTransferSrcOptimal, mgr.getStagingBuffer(),
                                         1, &copyRegion);
    }
}

void CommandBufferManager::recordWaitDownloadComplete(const std::span<const TImageMgrCRef> dstImageMgrRefs) {
    vk::BufferMemoryBarrier downloadCompleteBarrierTemplate;
    downloadCompleteBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    downloadCompleteBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eHostRead);
    downloadCompleteBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadCompleteBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto downloadCompleteBarriers =
        dstImageMgrRefs | rgs::views::transform([&downloadCompleteBarrierTemplate](const TImageMgrCRef mgrRef) {
            const auto& mgr = mgrRef.get();
            vk::BufferMemoryBarrier downloadCompleteBarrier = downloadCompleteBarrierTemplate;
            downloadCompleteBarrier.setBuffer(mgr.getStagingBuffer());
            downloadCompleteBarrier.setSize(mgr.getExtent().size());
            return downloadCompleteBarrier;
        }) |
        rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, (uint32_t)downloadCompleteBarriers.size(),
                                   downloadCompleteBarriers.data(), 0, nullptr);
}

void CommandBufferManager::recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr,
                                                const vk::PipelineStageFlagBits pipelineStage) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);
}

void CommandBufferManager::recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr,
                                              const vk::PipelineStageFlagBits pipelineStage) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);
}

void CommandBufferManager::end() { commandBuffer_.end(); }

void CommandBufferManager::submitTo(QueueManager& queueMgr) {
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer_);

    auto& computeQueue = queueMgr.getComputeQueue();
    computeQueue.submit(submitInfo, completeFence_);
}

vk::Result CommandBufferManager::waitFence() {
    auto& device = pDeviceMgr_->getDevice();

    auto waitFenceResult = device.waitForFences(completeFence_, true, std::numeric_limits<uint64_t>::max());
    if (waitFenceResult != vk::Result::eSuccess) {
        return waitFenceResult;
    }

    device.resetFences(completeFence_);

    return vk::Result::eSuccess;
}

}  // namespace vkc
