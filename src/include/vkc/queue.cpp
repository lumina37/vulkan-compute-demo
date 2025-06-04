#pragma once

#include <expected>

#include "vkc/command/buffer.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/sync/fence.hpp"
#include "vkc/sync/semaphore.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.hpp"
#endif

namespace vkc {

QueueManager::QueueManager(vk::Queue queue) noexcept : queue_(queue) {}

std::expected<QueueManager, Error> QueueManager::create(DeviceManager& deviceMgr, vk::QueueFlags type) noexcept {
    auto queueRes = deviceMgr.getQueue(type);
    if (!queueRes) return std::unexpected{std::move(queueRes.error())};
    const vk::Queue queue = queueRes.value();

    return QueueManager{queue};
}

std::expected<void, Error> QueueManager::_submit(vk::CommandBuffer commandBuffer, vk::Semaphore waitSemaphore,
                                                 vk::PipelineStageFlags waitDstStage, vk::Semaphore signalSemaphore,
                                                 vk::Fence fence) noexcept {
    vk::SubmitInfo submitInfo;
    if (waitSemaphore != nullptr) {
        submitInfo.setWaitSemaphores(waitSemaphore);
        submitInfo.setWaitDstStageMask(waitDstStage);
    }
    submitInfo.setCommandBuffers(commandBuffer);
    if (signalSemaphore != nullptr) {
        submitInfo.setSignalSemaphores(signalSemaphore);
    }

    const auto submitRes = queue_.submit(submitInfo, fence);
    if (submitRes != vk::Result::eSuccess) {
        return std::unexpected{Error{submitRes}};
    }

    return {};
}

std::expected<void, Error> QueueManager::submitAndWaitSemaphore(CommandBufferManager& commandBufferMgr,
                                                              const SemaphoreManager& waitSemaphoreMgr,
                                                              const vk::PipelineStageFlags waitDstStage,
                                                              SemaphoreManager& signalSemaphoreMgr) noexcept {
    return _submit(commandBufferMgr.getCommandBuffer(), waitSemaphoreMgr.getSemaphore(), waitDstStage,
                   signalSemaphoreMgr.getSemaphore(), nullptr);
}

std::expected<void, Error> QueueManager::submitAndWaitSemaphore(CommandBufferManager& commandBufferMgr,
                                                              const SemaphoreManager& waitSemaphoreMgr,
                                                              const vk::PipelineStageFlags waitDstStage,
                                                              FenceManager& fenceMgr) noexcept {
    return _submit(commandBufferMgr.getCommandBuffer(), waitSemaphoreMgr.getSemaphore(), waitDstStage, nullptr,
                   fenceMgr.getFence());
}

std::expected<void, Error> QueueManager::submit(CommandBufferManager& commandBufferMgr,
                                                SemaphoreManager& signalSemaphoreMgr) noexcept {
    return _submit(commandBufferMgr.getCommandBuffer(), nullptr, vk::PipelineStageFlagBits::eNone,
                   signalSemaphoreMgr.getSemaphore(), nullptr);
}

std::expected<void, Error> QueueManager::submit(CommandBufferManager& commandBufferMgr,
                                                FenceManager& fenceMgr) noexcept {
    return _submit(commandBufferMgr.getCommandBuffer(), nullptr, vk::PipelineStageFlagBits::eNone, nullptr,
                   fenceMgr.getFence());
}

std::expected<void, Error> QueueManager::present(SwapchainManager& swapchainMgr, uint32_t imageIndex) noexcept {
    vk::PresentInfoKHR presentInfo;
    presentInfo.setImageIndices(imageIndex);
    vk::SwapchainKHR swapchain = swapchainMgr.getSwapchain();
    presentInfo.setSwapchains(swapchain);

    const auto presentRes = queue_.presentKHR(presentInfo);
    if (presentRes != vk::Result::eSuccess) {
        return std::unexpected{Error{presentRes}};
    }

    return {};
}

}  // namespace vkc
