#pragma once

#include "vkc/command/buffer.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/sync/fence.hpp"
#include "vkc/sync/semaphore.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.hpp"
#endif

namespace vkc {

QueueBox::QueueBox(vk::Queue queue) noexcept : queue_(queue) {}

std::expected<QueueBox, Error> QueueBox::create(DeviceBox& deviceBox, vk::QueueFlags type) noexcept {
    auto queueRes = deviceBox.getQueue(type);
    if (!queueRes) return std::unexpected{std::move(queueRes.error())};
    const vk::Queue queue = queueRes.value();

    return QueueBox{queue};
}

std::expected<void, Error> QueueBox::_submit(vk::CommandBuffer commandBuffer, vk::Semaphore waitSemaphore,
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
        return std::unexpected{Error{ECate::eVk, submitRes}};
    }

    return {};
}

std::expected<void, Error> QueueBox::submitAndWaitSemaphore(CommandBufferBox& commandBufferBox,
                                                            const SemaphoreBox& waitSemaphoreBox,
                                                            const vk::PipelineStageFlags waitDstStage,
                                                            SemaphoreBox& signalSemaphoreBox) noexcept {
    return _submit(commandBufferBox.getCommandBuffer(), waitSemaphoreBox.getSemaphore(), waitDstStage,
                   signalSemaphoreBox.getSemaphore(), nullptr);
}

std::expected<void, Error> QueueBox::submitAndWaitSemaphore(CommandBufferBox& commandBufferBox,
                                                            const SemaphoreBox& waitSemaphoreBox,
                                                            const vk::PipelineStageFlags waitDstStage,
                                                            FenceBox& fenceBox) noexcept {
    return _submit(commandBufferBox.getCommandBuffer(), waitSemaphoreBox.getSemaphore(), waitDstStage, nullptr,
                   fenceBox.getFence());
}

std::expected<void, Error> QueueBox::submit(CommandBufferBox& commandBufferBox,
                                            SemaphoreBox& signalSemaphoreBox) noexcept {
    return _submit(commandBufferBox.getCommandBuffer(), nullptr, vk::PipelineStageFlagBits::eNone,
                   signalSemaphoreBox.getSemaphore(), nullptr);
}

std::expected<void, Error> QueueBox::submit(CommandBufferBox& commandBufferBox, FenceBox& fenceBox) noexcept {
    return _submit(commandBufferBox.getCommandBuffer(), nullptr, vk::PipelineStageFlagBits::eNone, nullptr,
                   fenceBox.getFence());
}

std::expected<void, Error> QueueBox::present(SwapchainBox& swapchainBox, uint32_t imageIndex) noexcept {
    vk::PresentInfoKHR presentInfo;
    presentInfo.setImageIndices(imageIndex);
    vk::SwapchainKHR swapchain = swapchainBox.getSwapchain();
    presentInfo.setSwapchains(swapchain);

    const auto presentRes = queue_.presentKHR(presentInfo);
    if (presentRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, presentRes}};
    }

    return {};
}

}  // namespace vkc
