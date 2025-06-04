#pragma once

#include <expected>

#include "vkc/command/buffer.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/sync/fence.hpp"
#include "vkc/sync/semaphore.hpp"

namespace vkc {

class QueueManager {
    QueueManager(vk::Queue queue) noexcept;

    [[nodiscard]] std::expected<void, Error> _submit(vk::CommandBuffer commandBuffer, vk::Semaphore waitSemaphore,
                                                     vk::PipelineStageFlags waitDstStage, vk::Semaphore signalSemaphore,
                                                     vk::Fence fence) noexcept;

public:
    [[nodiscard]] static std::expected<QueueManager, Error> create(DeviceManager& deviceMgr,
                                                                   vk::QueueFlags type) noexcept;

    [[nodiscard]] vk::Queue getQueue() const noexcept { return queue_; }

    [[nodiscard]] std::expected<void, Error> submitAndWaitSemaphore(CommandBufferManager& queueMgr,
                                                                    const SemaphoreManager& waitSemaphoreMgr,
                                                                    vk::PipelineStageFlags waitDstStage,
                                                                    SemaphoreManager& signalSemaphoreMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submitAndWaitSemaphore(CommandBufferManager& queueMgr,
                                                                    const SemaphoreManager& waitSemaphoreMgr,
                                                                    vk::PipelineStageFlags waitDstStage,
                                                                    FenceManager& fenceMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(CommandBufferManager& queueMgr,
                                                    SemaphoreManager& signalSemaphoreMgr) noexcept;
    [[nodiscard]] std::expected<void, Error> submit(CommandBufferManager& queueMgr, FenceManager& fenceMgr) noexcept;

    [[nodiscard]] std::expected<void, Error> present(SwapchainManager& swapchainMgr, uint32_t imageIndex) noexcept;

private:
    vk::Queue queue_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.cpp"
#endif
