#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <print>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/image.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/queue.hpp"

namespace vkc {

class Context {
public:
    inline Context(const DeviceManager& deviceMgr, const CommandPoolManager& commandPoolMgr,
                   const PipelineManager& pipelineMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                   const DescSetManager& descSetMgr, const QueueManager& queueMgr, const vk::Extent2D& extent);
    inline ~Context() noexcept;

    inline void execute(const std::span<uint8_t> src, std::span<uint8_t> dst, BufferManager& bufferMgr);

private:
    // FIXME: lots of UAF
    const DeviceManager& deviceMgr_;
    const CommandPoolManager& commandPoolMgr_;
    const PipelineManager& pipelineMgr_;
    const PipelineLayoutManager& pipelineLayoutMgr_;
    const DescSetManager& descSetMgr_;
    vk::Extent2D extent_;

    const QueueManager& queueMgr_;
    CommandBufferManager commandBufferMgr_;
    vk::Fence commandCompleteFence_;
};

Context::Context(const DeviceManager& deviceMgr, const CommandPoolManager& commandPoolMgr,
                 const PipelineManager& pipelineMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                 const DescSetManager& descSetMgr, const QueueManager& queueMgr, const vk::Extent2D& extent)
    : deviceMgr_(deviceMgr),
      commandPoolMgr_(commandPoolMgr),
      pipelineMgr_(pipelineMgr),
      pipelineLayoutMgr_(pipelineLayoutMgr),
      descSetMgr_(descSetMgr),
      extent_(extent),
      queueMgr_(queueMgr),
      commandBufferMgr_(deviceMgr, commandPoolMgr) {
    vk::FenceCreateInfo fenceInfo;
    const auto& device = deviceMgr.getDevice();
    commandCompleteFence_ = device.createFence(fenceInfo);
}

inline Context::~Context() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyFence(commandCompleteFence_);
}

void Context::execute(const std::span<uint8_t> src, std::span<uint8_t> dst, BufferManager& bufferMgr) {
    const auto& device = deviceMgr_.getDevice();
    const auto& srcImage = bufferMgr.srcImageMgr_.getImage();
    const auto& dstImage = bufferMgr.dstImageMgr_.getImage();

    // Upload to Staging Buffer
    void* mapPtr;
    const auto& uploadMapResult =
        device.mapMemory(bufferMgr.srcStagingMemory_, 0, src.size(), (vk::MemoryMapFlags)0, &mapPtr);
    memcpy(mapPtr, src.data(), src.size());
    device.unmapMemory(bufferMgr.srcStagingMemory_);

    // Begin Command Buffer
    const auto& cmdBuf = commandBufferMgr_.getCommandBuffers()[0];
    cmdBuf.reset();

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    cmdBuf.begin(cmdBufBeginInfo);

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
    uploadConvBarrier.setImage(srcImage);
    uploadConvBarrier.setSubresourceRange(subresourceRange);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                           (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &uploadConvBarrier);

    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent({extent_.width, extent_.height, 1});

    cmdBuf.copyBufferToImage(bufferMgr.srcStagingBuffer_, bufferMgr.srcImageMgr_.getImage(),
                             vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

    // Shader Compatible Image Layout
    vk::ImageMemoryBarrier srcShaderCompatibleBarrier;
    srcShaderCompatibleBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    srcShaderCompatibleBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    srcShaderCompatibleBarrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    srcShaderCompatibleBarrier.setNewLayout(vk::ImageLayout::eGeneral);
    srcShaderCompatibleBarrier.setImage(srcImage);
    srcShaderCompatibleBarrier.setSubresourceRange(subresourceRange);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                           (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &srcShaderCompatibleBarrier);

    vk::ImageMemoryBarrier dstShaderCompatibleBarrier;
    dstShaderCompatibleBarrier.setSrcAccessMask(vk::AccessFlagBits::eNone);
    dstShaderCompatibleBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    dstShaderCompatibleBarrier.setOldLayout(vk::ImageLayout::eUndefined);
    dstShaderCompatibleBarrier.setNewLayout(vk::ImageLayout::eGeneral);
    dstShaderCompatibleBarrier.setImage(dstImage);
    dstShaderCompatibleBarrier.setSubresourceRange(subresourceRange);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                           (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &dstShaderCompatibleBarrier);

    // Execute Pipeline
    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipelineMgr_.getPipeline());
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayoutMgr_.getPipelineLayout(), 0, 1,
                              &descSetMgr_.getDescSets()[0], 0, nullptr);

    uint32_t groupSizeX = (extent_.width + 15) / 16;
    uint32_t groupSizeY = (extent_.height + 15) / 16;
    cmdBuf.dispatch(groupSizeX, groupSizeY, 1);

    // Download to Staging Buffer
    vk::ImageMemoryBarrier downloadConvBarrier;
    downloadConvBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    downloadConvBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    downloadConvBarrier.setOldLayout(vk::ImageLayout::eGeneral);
    downloadConvBarrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    downloadConvBarrier.setImage(dstImage);
    downloadConvBarrier.setSubresourceRange(subresourceRange);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, 1, &downloadConvBarrier);

    cmdBuf.copyImageToBuffer(dstImage, vk::ImageLayout::eTransferSrcOptimal, bufferMgr.dstStagingBuffer_, 1,
                             &copyRegion);

    vk::BufferMemoryBarrier downloadCompleteBarrier;
    downloadCompleteBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    downloadCompleteBarrier.setDstAccessMask(vk::AccessFlagBits::eHostRead);
    downloadCompleteBarrier.setBuffer(bufferMgr.dstStagingBuffer_);
    downloadCompleteBarrier.setSize(bufferMgr.size_);

    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                           (vk::DependencyFlags)0, 0, nullptr, 1, &downloadCompleteBarrier, 0, nullptr);

    cmdBuf.end();

    // Submit and Wait
    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(cmdBuf);

    const auto& computeQueue = queueMgr_.getComputeQueue();
    computeQueue.submit(submitInfo, commandCompleteFence_);

    const auto& waitFenceResult =
        device.waitForFences(commandCompleteFence_, true, std::numeric_limits<uint64_t>::max());
    if constexpr (ENABLE_DEBUG) {
        if (waitFenceResult != vk::Result::eSuccess) {
            std::println(std::cerr, "Command fence timeout!");
        }
    }
    device.resetFences(commandCompleteFence_);

    // Download from Staging Buffer
    const auto& downloadMapResult =
        device.mapMemory(bufferMgr.dstStagingMemory_, 0, bufferMgr.size_, (vk::MemoryMapFlags)0, &mapPtr);
    memcpy(dst.data(), mapPtr, dst.size());
    device.unmapMemory(bufferMgr.dstStagingMemory_);
}

}  // namespace vkc
