#include <expected>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <utility>

#include "vkc/command/concepts.hpp"
#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/fence.hpp"
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
                                           vk::CommandBuffer commandBuffer) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), pCommandPoolMgr_(std::move(pCommandPoolMgr)), commandBuffer_(commandBuffer) {}

CommandBufferManager::CommandBufferManager(CommandBufferManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      pCommandPoolMgr_(std::move(rhs.pCommandPoolMgr_)),
      commandBuffer_(std::exchange(rhs.commandBuffer_, nullptr)) {}

CommandBufferManager::~CommandBufferManager() noexcept {
    if (commandBuffer_ == nullptr) return;

    auto device = pDeviceMgr_->getDevice();
    auto commandPool = pCommandPoolMgr_->getCommandPool();
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

void CommandBufferManager::recordDstPrepareShaderWrite(
    const std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept {
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

    const auto fillout = [&shaderCompatibleBarrierTemplate](const TStorageImageMgrCRef mgrRef) {
        vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
        shaderCompatibleBarrier.setImage(mgrRef.get().getImage());
        return shaderCompatibleBarrier;
    };

    const auto shaderCompatibleBarriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)shaderCompatibleBarriers.size(), shaderCompatibleBarriers.data());
}

void CommandBufferManager::recordDispatch(const vk::Extent2D extent, const BlockSize blockSize) noexcept {
    uint32_t groupSizeX = (extent.width + (blockSize.x - 1)) / blockSize.x;
    uint32_t groupSizeY = (extent.height + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferManager::recordDstPrepareTransfer(
    const std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept {
    vk::ImageMemoryBarrier downloadConvBarrierTemplate;
    downloadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    downloadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    downloadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eGeneral);
    downloadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    downloadConvBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadConvBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&downloadConvBarrierTemplate](const TStorageImageMgrCRef mgrRef) {
        vk::ImageMemoryBarrier downloadConvBarrier = downloadConvBarrierTemplate;
        downloadConvBarrier.setImage(mgrRef.get().getImage());
        return downloadConvBarrier;
    };

    const auto downloadConvBarriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr,
                                   (uint32_t)downloadConvBarriers.size(), downloadConvBarriers.data());
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
    const int bufferRowLen = dstImageMgr.getExtent().width();
    copyRegion.setBufferOffset(roi.offset().y * bufferRowLen + roi.offset().x);
    copyRegion.setBufferRowLength(bufferRowLen);
    copyRegion.setBufferImageHeight(dstImageMgr.getExtent().height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyImageToBuffer(dstImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageMgr.getStagingBuffer(), 1, &copyRegion);
}

void CommandBufferManager::recordCopyStorageToSampled(const StorageImageManager& srcImageMgr,
                                                      SampledImageManager& dstImageMgr) noexcept {
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

void CommandBufferManager::recordCopyStorageToSampledWithRoi(const StorageImageManager& srcImageMgr,
                                                             SampledImageManager& dstImageMgr, const Roi roi) noexcept {
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

void CommandBufferManager::recordWaitDownloadComplete(
    const std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept {
    vk::BufferMemoryBarrier downloadCompleteBarrierTemplate;
    downloadCompleteBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    downloadCompleteBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eHostRead);
    downloadCompleteBarrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    downloadCompleteBarrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&downloadCompleteBarrierTemplate](const TStorageImageMgrCRef mgrRef) {
        const auto& mgr = mgrRef.get();
        vk::BufferMemoryBarrier downloadCompleteBarrier = downloadCompleteBarrierTemplate;
        downloadCompleteBarrier.setBuffer(mgr.getStagingBuffer());
        downloadCompleteBarrier.setSize(mgr.getExtent().size());
        return downloadCompleteBarrier;
    };

    const auto downloadCompleteBarriers = dstImageMgrRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, (uint32_t)downloadCompleteBarriers.size(),
                                   downloadCompleteBarriers.data(), 0, nullptr);
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
    const vk::Result endRes = commandBuffer_.end();
    if (endRes != vk::Result::eSuccess) {
        return std::unexpected{Error{endRes}};
    }
    return {};
}

std::expected<void, Error> CommandBufferManager::submitTo(QueueManager& queueMgr, FenceManager& fenceMgr) noexcept {
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer_);

    vk::Queue computeQueue = queueMgr.getComputeQueue();
    const vk::Result submitRes = computeQueue.submit(submitInfo, fenceMgr.getFence());
    if (submitRes != vk::Result::eSuccess) {
        return std::unexpected{Error{submitRes}};
    }

    return {};
}

template void CommandBufferManager::recordSrcPrepareTranfer<SampledImageManager>(
    std::span<const std::reference_wrapper<const SampledImageManager>>) noexcept;
template void CommandBufferManager::recordSrcPrepareTranfer<StorageImageManager>(
    std::span<const std::reference_wrapper<const StorageImageManager>>) noexcept;

template void CommandBufferManager::recordSrcPrepareShaderRead<SampledImageManager>(
    std::span<const std::reference_wrapper<const SampledImageManager>>) noexcept;
template void CommandBufferManager::recordSrcPrepareShaderRead<StorageImageManager>(
    std::span<const std::reference_wrapper<const StorageImageManager>>) noexcept;

template void CommandBufferManager::recordCopyStagingToSrc<SampledImageManager>(
    const SampledImageManager& srcImageMgr) noexcept;
template void CommandBufferManager::recordCopyStagingToSrc<StorageImageManager>(
    const StorageImageManager& srcImageMgr) noexcept;

}  // namespace vkc
