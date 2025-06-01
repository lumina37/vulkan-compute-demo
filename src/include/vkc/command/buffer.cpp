#include <expected>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
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
#include "vkc/sync/fence.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

CommandBufferManager::CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                                           std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr,
                                           const vk::CommandBuffer commandBuffer) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), pCommandPoolMgr_(std::move(pCommandPoolMgr)), commandBuffer_(commandBuffer) {}

CommandBufferManager::CommandBufferManager(CommandBufferManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      pCommandPoolMgr_(std::move(rhs.pCommandPoolMgr_)),
      commandBuffer_(std::exchange(rhs.commandBuffer_, nullptr)) {}

CommandBufferManager::~CommandBufferManager() noexcept {
    if (commandBuffer_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();
    vk::CommandPool commandPool = pCommandPoolMgr_->getCommandPool();
    device.freeCommandBuffers(commandPool, commandBuffer_);
    commandBuffer_ = nullptr;
}

std::expected<CommandBufferManager, Error> CommandBufferManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::shared_ptr<CommandPoolManager> pCommandPoolMgr) noexcept {
    vk::Device device = pDeviceMgr->getDevice();
    vk::CommandPool commandPool = pCommandPoolMgr->getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    const auto [commandBuffersRes, commandBuffers] = device.allocateCommandBuffers(allocInfo);
    if (commandBuffersRes != vk::Result::eSuccess) {
        return std::unexpected{Error{commandBuffersRes}};
    }
    vk::CommandBuffer commandBuffer = commandBuffers[0];

    return CommandBufferManager{std::move(pDeviceMgr), std::move(pCommandPoolMgr), commandBuffer};
}

void CommandBufferManager::bindPipeline(PipelineManager& pipelineMgr) noexcept {
    commandBuffer_.bindPipeline(pipelineMgr.getBindPoint(), pipelineMgr.getPipeline());
}

void CommandBufferManager::bindDescSets(DescSetsManager& descSetsMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                                        const vk::PipelineBindPoint bindPoint) noexcept {
    auto& descSets = descSetsMgr.getDescSets();
    commandBuffer_.bindDescriptorSets(bindPoint, pipelineLayoutMgr.getPipelineLayout(), 0, (uint32_t)descSets.size(),
                                      descSets.data(), 0, nullptr);
}

std::expected<void, Error> CommandBufferManager::begin() noexcept {
    const auto resetRes = commandBuffer_.reset();
    if (resetRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetRes}};
    }

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    const auto beginRes = commandBuffer_.begin(cmdBufBeginInfo);
    if (beginRes != vk::Result::eSuccess) {
        return std::unexpected{Error{beginRes}};
    }

    return {};
}

void CommandBufferManager::recordDstPrepareShaderWrite(
    const std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageMgrRef mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferManager::recordDispatch(const vk::Extent2D extent, const BlockSize blockSize) noexcept {
    const uint32_t groupSizeX = (extent.width + (blockSize.x - 1)) / blockSize.x;
    const uint32_t groupSizeY = (extent.height + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferManager::recordPrepareSendBeforeDispatch(
    const std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageMgrRef mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferManager::recordPrepareSendAfterDispatch(
    const std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageMgrRef mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(mgr.getImageAccessMask());
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferManager::recordPreparePresent(std::span<const TPresentImageMgrRef> imageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eMemoryRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TPresentImageMgrRef mgrRef) {
        auto& mgr = mgrRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(mgr.getImageAccessMask());
        barrier.setOldLayout(mgr.getImageLayout());
        barrier.setImage(mgr.getImage());

        mgr.setImageAccessMask(newAccessMask);
        mgr.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferManager::recordCopyDstToStaging(StorageImageManager& dstImageMgr) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(dstImageMgr.getExtent().extent3D());

    commandBuffer_.copyImageToBuffer(dstImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageMgr.getStagingBuffer(), 1, &copyRegion);
}

void CommandBufferManager::recordCopyDstToStagingWithRoi(StorageImageManager& dstImageMgr, const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = dstImageMgr.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyImageToBuffer(dstImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageMgr.getStagingBuffer(), 1, &copyRegion);
}

void CommandBufferManager::recordWaitDownloadComplete(
    const std::span<const TStorageImageMgrRef> dstImageMgrRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eHostRead;

    vk::BufferMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&](const TStorageImageMgrRef mgrRef) {
        auto& mgr = mgrRef.get();

        vk::BufferMemoryBarrier barrier = barrierTemplate;
        barrier.setBuffer(mgr.getStagingBuffer());
        barrier.setSize(mgr.getExtent().size());

        mgr.setStagingAccessMask(newAccessMask);

        return barrier;
    };

    const auto barriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, (uint32_t)barriers.size(), barriers.data(), 0,
                                   nullptr);
}

std::expected<void, Error> CommandBufferManager::recordTimestampStart(
    TimestampQueryPoolManager& queryPoolMgr, const vk::PipelineStageFlagBits pipelineStage) noexcept {
    vk::QueryPool queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();

    auto addIndexRes = queryPoolMgr.addQueryIndex();
    if (!addIndexRes) return std::unexpected{std::move(addIndexRes.error())};

    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);

    return {};
}

std::expected<void, Error> CommandBufferManager::recordTimestampEnd(
    TimestampQueryPoolManager& queryPoolMgr, const vk::PipelineStageFlagBits pipelineStage) noexcept {
    vk::QueryPool queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();

    auto addIndexRes = queryPoolMgr.addQueryIndex();
    if (!addIndexRes) return std::unexpected{std::move(addIndexRes.error())};

    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);

    return {};
}

std::expected<void, Error> CommandBufferManager::end() noexcept {
    const auto endRes = commandBuffer_.end();
    if (endRes != vk::Result::eSuccess) {
        return std::unexpected{Error{endRes}};
    }
    return {};
}

std::expected<void, Error> CommandBufferManager::_submit(vk::Queue queue, vk::Semaphore waitSemaphore,
                                                         vk::PipelineStageFlags waitDstStage,
                                                         vk::Semaphore signalSemaphore, vk::Fence fence) noexcept {
    vk::SubmitInfo submitInfo;
    if (waitSemaphore != nullptr) {
        submitInfo.setWaitSemaphores(waitSemaphore);
        submitInfo.setWaitDstStageMask(waitDstStage);
    }
    submitInfo.setCommandBuffers(commandBuffer_);
    if (signalSemaphore != nullptr) {
        submitInfo.setSignalSemaphores(signalSemaphore);
    }

    const auto submitRes = queue.submit(submitInfo, fence);
    if (submitRes != vk::Result::eSuccess) {
        return std::unexpected{Error{submitRes}};
    }

    return {};
}

std::expected<void, Error> CommandBufferManager::submitAndWaitPreTask(QueueManager& queueMgr,
                                                                      const SemaphoreManager& waitSemaphoreMgr,
                                                                      const vk::PipelineStageFlags waitDstStage,
                                                                      SemaphoreManager& signalSemaphoreMgr) noexcept {
    return _submit(queueMgr.getQueue(), waitSemaphoreMgr.getSemaphore(), waitDstStage,
                   signalSemaphoreMgr.getSemaphore(), nullptr);
}

std::expected<void, Error> CommandBufferManager::submitAndWaitPreTask(QueueManager& queueMgr,
                                                                      const SemaphoreManager& waitSemaphoreMgr,
                                                                      const vk::PipelineStageFlags waitDstStage,
                                                                      FenceManager& fenceMgr) noexcept {
    return _submit(queueMgr.getQueue(), waitSemaphoreMgr.getSemaphore(), waitDstStage, nullptr, fenceMgr.getFence());
}

std::expected<void, Error> CommandBufferManager::submit(QueueManager& queueMgr,
                                                        SemaphoreManager& signalSemaphoreMgr) noexcept {
    return _submit(queueMgr.getQueue(), nullptr, vk::PipelineStageFlagBits::eNone, signalSemaphoreMgr.getSemaphore(),
                   nullptr);
}

std::expected<void, Error> CommandBufferManager::submit(QueueManager& queueMgr, FenceManager& fenceMgr) noexcept {
    return _submit(queueMgr.getQueue(), nullptr, vk::PipelineStageFlagBits::eNone, nullptr, fenceMgr.getFence());
}

template void CommandBufferManager::recordPrepareReceiveBeforeDispatch<SampledImageManager>(
    std::span<const std::reference_wrapper<SampledImageManager>>) noexcept;
template void CommandBufferManager::recordPrepareReceiveBeforeDispatch<StorageImageManager>(
    std::span<const std::reference_wrapper<StorageImageManager>>) noexcept;

template void CommandBufferManager::recordSrcPrepareShaderRead<SampledImageManager>(
    std::span<const std::reference_wrapper<SampledImageManager>>) noexcept;
template void CommandBufferManager::recordSrcPrepareShaderRead<StorageImageManager>(
    std::span<const std::reference_wrapper<StorageImageManager>>) noexcept;

template void CommandBufferManager::recordCopyStagingToSrc<SampledImageManager>(
    const SampledImageManager& srcImageMgr) noexcept;
template void CommandBufferManager::recordCopyStagingToSrc<StorageImageManager>(
    const StorageImageManager& srcImageMgr) noexcept;

}  // namespace vkc
