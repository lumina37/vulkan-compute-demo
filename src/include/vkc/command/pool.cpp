#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/queue_family.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.hpp"
#endif

namespace vkc {

CommandPoolManager::CommandPoolManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr)
    : deviceMgr_(deviceMgr) {
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyMgr.getComputeQFamilyIndex());

    auto& device = deviceMgr.getDevice();
    commandPool_ = device.createCommandPool(commandPoolInfo);
}

CommandPoolManager::~CommandPoolManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyCommandPool(commandPool_);
}

}  // namespace vkc
