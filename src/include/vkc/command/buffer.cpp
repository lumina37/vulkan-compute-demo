#include <limits>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/queue.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

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

void CommandBufferManager::begin() {
    commandBuffer_.reset();

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    commandBuffer_.begin(cmdBufBeginInfo);
}

void CommandBufferManager::recordDispatch(const ExtentManager extent, const BlockSize blockSize) {
    uint32_t groupSizeX = (extent.width() + (blockSize.x - 1)) / blockSize.x;
    uint32_t groupSizeY = (extent.height() + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
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
    auto& device = deviceMgr_.getDevice();

    auto waitFenceResult = device.waitForFences(completeFence_, true, std::numeric_limits<uint64_t>::max());
    if (waitFenceResult != vk::Result::eSuccess) {
        return waitFenceResult;
    }

    device.resetFences(completeFence_);

    return vk::Result::eSuccess;
}

}  // namespace vkc
