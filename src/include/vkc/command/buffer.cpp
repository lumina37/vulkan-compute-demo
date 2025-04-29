#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <ranges>
#include <utility>

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

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

CommandBufferManager::CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                                           std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr,
                                           vk::CommandBuffer commandBuffer, vk::Fence completeFence) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      pCommandPoolMgr_(std::move(pCommandPoolMgr)),
      commandBuffer_(commandBuffer),
      completeFence_(completeFence) {}

CommandBufferManager::CommandBufferManager(CommandBufferManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      pCommandPoolMgr_(std::move(rhs.pCommandPoolMgr_)),
      commandBuffer_(std::exchange(rhs.commandBuffer_, nullptr)),
      completeFence_(std::exchange(rhs.completeFence_, nullptr)) {}

CommandBufferManager::~CommandBufferManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    auto& commandPool = pCommandPoolMgr_->getCommandPool();
    if (commandBuffer_ != nullptr) {
        device.freeCommandBuffers(commandPool, commandBuffer_);
        commandBuffer_ = nullptr;
    }
    if (completeFence_ != nullptr) {
        device.destroyFence(completeFence_);
        completeFence_ = nullptr;
    }
}

std::expected<CommandBufferManager, Error> CommandBufferManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::shared_ptr<CommandPoolManager> pCommandPoolMgr) noexcept {
    auto& device = pDeviceMgr->getDevice();
    auto& commandPool = pCommandPoolMgr->getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    const auto [commandBuffersRes, commandBuffers] = device.allocateCommandBuffers(allocInfo);
    if (commandBuffersRes != vk::Result::eSuccess) {
        return std::unexpected{Error{commandBuffersRes}};
    }
    vk::CommandBuffer commandBuffer = commandBuffers[0];

    const auto [completeFenceRes, completeFence] = device.createFence({});
    if (completeFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{completeFenceRes}};
    }

    return CommandBufferManager{std::move(pDeviceMgr), std::move(pCommandPoolMgr), commandBuffer, completeFence};
}

void CommandBufferManager::bindPipeline(PipelineManager& pipelineMgr) noexcept {
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineMgr.getPipeline());
}

void CommandBufferManager::bindDescSets(DescSetsManager& descSetsMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) noexcept {
    auto& descSets = descSetsMgr.getDescSets();
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutMgr.getPipelineLayout(), 0,
                                      (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
}

std::expected<void, Error> CommandBufferManager::begin() noexcept {
    const vk::Result resetRes = commandBuffer_.reset();
    if (resetRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetRes}};
    }

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    const vk::Result beginRes = commandBuffer_.begin(cmdBufBeginInfo);
    if (beginRes != vk::Result::eSuccess) {
        return std::unexpected{Error{beginRes}};
    }

    return {};
}

void CommandBufferManager::recordSrcPrepareTranfer(const std::span<const TImageMgrCRef> srcImageMgrRefs) noexcept {
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

void CommandBufferManager::recordUploadToSrc(const std::span<const TImageMgrCRef> srcImageMgrRefs) noexcept {
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

void CommandBufferManager::recordImageCopy(const std::span<const ImageManagerPair> imageMgrPairs) noexcept {
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

void CommandBufferManager::recordSrcPrepareShaderRead(const std::span<const TImageMgrCRef> srcImageMgrRefs) noexcept {
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

void CommandBufferManager::recordDstPrepareShaderWrite(const std::span<const TImageMgrCRef> dstImageMgrRefs) noexcept {
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

void CommandBufferManager::recordDispatch(const Extent extent, const BlockSize blockSize) noexcept {
    uint32_t groupSizeX = (extent.width() + (blockSize.x - 1)) / blockSize.x;
    uint32_t groupSizeY = (extent.height() + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferManager::recordDstPrepareTransfer(const std::span<const TImageMgrCRef> dstImageMgrRefs) noexcept {
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

void CommandBufferManager::recordDownloadToDst(std::span<const TImageMgrCRef> dstImageMgrRefs) noexcept {
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

void CommandBufferManager::recordWaitDownloadComplete(const std::span<const TImageMgrCRef> dstImageMgrRefs) noexcept {
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
                                                const vk::PipelineStageFlagBits pipelineStage) noexcept {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);
}

void CommandBufferManager::recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr,
                                              const vk::PipelineStageFlagBits pipelineStage) noexcept {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);
}

std::expected<void, Error> CommandBufferManager::end() noexcept {
    const vk::Result endRes = commandBuffer_.end();
    if (endRes != vk::Result::eSuccess) {
        return std::unexpected{Error{endRes}};
    }
    return {};
}

std::expected<void, Error> CommandBufferManager::submitTo(QueueManager& queueMgr) noexcept {
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer_);

    auto& computeQueue = queueMgr.getComputeQueue();
    const vk::Result submitRes = computeQueue.submit(submitInfo, completeFence_);
    if (submitRes != vk::Result::eSuccess) {
        return std::unexpected{Error{submitRes}};
    }

    return {};
}

std::expected<void, Error> CommandBufferManager::waitFence() noexcept {
    auto& device = pDeviceMgr_->getDevice();

    const auto waitFenceRes = device.waitForFences(completeFence_, true, std::numeric_limits<uint64_t>::max());
    if (waitFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{waitFenceRes}};
    }

    const auto resetFenceRes = device.resetFences(completeFence_);
    if (resetFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetFenceRes}};
    }

    return {};
}

}  // namespace vkc
