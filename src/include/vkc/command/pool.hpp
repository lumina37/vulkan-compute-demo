#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/queue_family.hpp"

namespace vkc {

class CommandPoolManager {
public:
    inline CommandPoolManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr);
    inline ~CommandPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandPool_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::CommandPool commandPool_;
};

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
