#pragma once

#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/queue.hpp"
#include "vkc/resource/image.hpp"
#include "vkc/resource/push_constant.hpp"

namespace vkc {

class CommandBufferManager {
public:
    inline CommandBufferManager(DeviceManager& deviceMgr, CommandPoolManager& commandPoolMgr);
    inline ~CommandBufferManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandBuffer_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getCompleteFence(this Self&& self) noexcept {
        return std::forward_like<Self>(self).completeFence_;
    }

    inline void begin();
    inline void bindPipeline(PipelineManager& pipelineMgr);
    inline void bindDescSet(DescSetManager& descSetMgr, const PipelineLayoutManager& pipelineLayoutMgr);

    template <typename TPc>
    inline void pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                             const PipelineLayoutManager& pipelineLayoutMgr);

    inline void recordUpload(ImageManager& srcImageMgr);
    inline void recordDstLayoutTrans(ImageManager& dstImageMgr);
    inline void recordDispatch(const ExtentManager extent, const BlockSize blockSize);
    inline void recordDownload(ImageManager& dstImageMgr);

    template <typename TQueryPoolManager>
        requires CQueryPoolManager<TQueryPoolManager>
    inline void recordResetQueryPool(TQueryPoolManager& queryPoolMgr);

    inline void recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr);
    inline void recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr);
    inline void end();
    inline void submitTo(QueueManager& queueMgr);
    inline vk::Result waitFence();

private:
    DeviceManager& deviceMgr_;            // FIXME: UAF
    CommandPoolManager& commandPoolMgr_;  // FIXME: UAF
    vk::CommandBuffer commandBuffer_;
    vk::Fence completeFence_;
};

CommandBufferManager::CommandBufferManager(DeviceManager& deviceMgr, CommandPoolManager& commandPoolMgr)
    : deviceMgr_(deviceMgr), commandPoolMgr_(commandPoolMgr) {
    auto& device = deviceMgr.getDevice();
    auto& commandPool = commandPoolMgr.getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    commandBuffer_ = device.allocateCommandBuffers(allocInfo)[0];
    completeFence_ = device.createFence({});
}

CommandBufferManager::~CommandBufferManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    auto& commandPool = commandPoolMgr_.getCommandPool();
    device.freeCommandBuffers(commandPool, commandBuffer_);
    device.destroyFence(completeFence_);
}

void CommandBufferManager::bindPipeline(PipelineManager& pipelineMgr) {
    commandBuffer_.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineMgr.getPipeline());
}

void CommandBufferManager::bindDescSet(DescSetManager& descSetMgr, const PipelineLayoutManager& pipelineLayoutMgr) {
    auto& descSet = descSetMgr.getDescSet();
    commandBuffer_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutMgr.getPipelineLayout(), 0, 1,
                                      &descSet, 0, nullptr);
}

template <typename TPc>
void CommandBufferManager::pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) {
    const auto& piplelineLayout = pipelineLayoutMgr.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TPc),
                                 pushConstantMgr.getPPushConstant());
}

void CommandBufferManager::begin() {
    commandBuffer_.reset();

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer_.begin(cmdBufBeginInfo);
}

void CommandBufferManager::recordUpload(ImageManager& srcImageMgr) {
    // Copy Staging Buffer to Image
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(1);

    vk::ImageMemoryBarrier uploadConvBarrier;
    uploadConvBarrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
    uploadConvBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    uploadConvBarrier.setOldLayout(vk::ImageLayout::eUndefined);
    uploadConvBarrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    uploadConvBarrier.setImage(srcImageMgr.getImage());
    uploadConvBarrier.setSubresourceRange(subresourceRange);

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &uploadConvBarrier);

    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(srcImageMgr.getExtent().extent3D());

    commandBuffer_.copyBufferToImage(srcImageMgr.getStagingBuffer(), srcImageMgr.getImage(),
                                     vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

    // Shader Compatible Image Layout
    vk::ImageMemoryBarrier srcShaderCompatibleBarrier;
    srcShaderCompatibleBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    srcShaderCompatibleBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    srcShaderCompatibleBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    srcShaderCompatibleBarrier.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    srcShaderCompatibleBarrier.setImage(srcImageMgr.getImage());
    srcShaderCompatibleBarrier.setSubresourceRange(subresourceRange);

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &srcShaderCompatibleBarrier);
}

void CommandBufferManager::recordDstLayoutTrans(ImageManager& dstImageMgr) {
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(1);

    // Shader Compatible Image Layout
    vk::ImageMemoryBarrier dstShaderCompatibleBarrier;
    dstShaderCompatibleBarrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
    dstShaderCompatibleBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    dstShaderCompatibleBarrier.setOldLayout(vk::ImageLayout::eUndefined);
    dstShaderCompatibleBarrier.setNewLayout(vk::ImageLayout::eGeneral);
    dstShaderCompatibleBarrier.setImage(dstImageMgr.getImage());
    dstShaderCompatibleBarrier.setSubresourceRange(subresourceRange);

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &dstShaderCompatibleBarrier);
}

void CommandBufferManager::recordDispatch(const ExtentManager extent, const BlockSize blockSize) {
    uint32_t groupSizeX = (extent.width() + (blockSize.x - 1)) / blockSize.x;
    uint32_t groupSizeY = (extent.height() + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferManager::recordDownload(ImageManager& dstImageMgr) {
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(1);

    // Download to Staging Buffer
    vk::ImageMemoryBarrier downloadConvBarrier;
    downloadConvBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    downloadConvBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    downloadConvBarrier.setOldLayout(vk::ImageLayout::eGeneral);
    downloadConvBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    downloadConvBarrier.setImage(dstImageMgr.getImage());
    downloadConvBarrier.setSubresourceRange(subresourceRange);

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &downloadConvBarrier);

    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(dstImageMgr.getExtent().extent3D());

    commandBuffer_.copyImageToBuffer(dstImageMgr.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageMgr.getStagingBuffer(), 1, &copyRegion);

    vk::BufferMemoryBarrier downloadCompleteBarrier;
    downloadCompleteBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    downloadCompleteBarrier.setDstAccessMask(vk::AccessFlagBits::eHostRead);
    downloadCompleteBarrier.setBuffer(dstImageMgr.getStagingBuffer());
    downloadCompleteBarrier.setSize(dstImageMgr.getExtent().size());

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, 1, &downloadCompleteBarrier, 0, nullptr);
}

template <typename TQueryPoolManager>
    requires CQueryPoolManager<TQueryPoolManager>
void CommandBufferManager::recordResetQueryPool(TQueryPoolManager& queryPoolMgr) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    commandBuffer_.resetQueryPool(queryPool, 0, queryPoolMgr.getQueryCount());
}

void CommandBufferManager::recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, queryIndex);
}

void CommandBufferManager::recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    const int queryIndex = queryPoolMgr.getQueryIndex();
    queryPoolMgr.addQueryIndex();
    commandBuffer_.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, queryIndex);
}

void CommandBufferManager::end() { commandBuffer_.end(); }

void CommandBufferManager::submitTo(QueueManager& queueMgr) {
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(commandBuffer_);

    auto& computeQueue = queueMgr.getComputeQueue();
    computeQueue.submit(submitInfo, completeFence_);
}

vk::Result CommandBufferManager::waitFence() {
    auto& device = deviceMgr_.getDevice();

    auto waitFenceResult = device.waitForFences(completeFence_, true, std::numeric_limits<uint64_t>::max());
    if (waitFenceResult != vk::Result::eSuccess) {
        return waitFenceResult;
    }

    device.resetFences(completeFence_);

    return vk::Result::eSuccess;
}

}  // namespace vkc
