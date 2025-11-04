#pragma once

#include "vkc/command/buffer.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/sync/fence.hpp"
#include "vkc/sync/semaphore.hpp"

namespace vkc {

class QueueBox {
    QueueBox(vk::Queue queue) noexcept;

    [[nodiscard]] std::expected<void, Error> _submit(vk::CommandBuffer commandBuffer, vk::Semaphore waitSemaphore,
                                                     vk::PipelineStageFlags waitDstStage, vk::Semaphore signalSemaphore,
                                                     vk::Fence fence) noexcept;

public:
    [[nodiscard]] static std::expected<QueueBox, Error> create(DeviceBox& deviceBox, vk::QueueFlags type) noexcept;

    [[nodiscard]] vk::Queue getQueue() const noexcept { return queue_; }

    [[nodiscard]] std::expected<void, Error> submitAndWaitSemaphore(CommandBufferBox& queueBox,
                                                                    const SemaphoreBox& waitSemaphoreBox,
                                                                    vk::PipelineStageFlags waitDstStage,
                                                                    SemaphoreBox& signalSemaphoreBox) noexcept;
    [[nodiscard]] std::expected<void, Error> submitAndWaitSemaphore(CommandBufferBox& queueBox,
                                                                    const SemaphoreBox& waitSemaphoreBox,
                                                                    vk::PipelineStageFlags waitDstStage,
                                                                    FenceBox& fenceBox) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(CommandBufferBox& queueBox,
                                                    SemaphoreBox& signalSemaphoreBox) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(CommandBufferBox& queueBox, FenceBox& fenceBox) noexcept;

    [[nodiscard]] std::expected<void, Error> present(SwapchainBox& swapchainBox, uint32_t imageIndex) noexcept;

private:
    vk::Queue queue_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.cpp"
#endif
