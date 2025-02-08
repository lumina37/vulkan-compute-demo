#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/queue_family.hpp"

namespace vkc {

class CommandPoolManager {
public:
    inline CommandPoolManager(const DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr);
    inline ~CommandPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandPool(this Self& self) noexcept {
        return std::forward_like<Self>(self).commandPool_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::CommandPool commandPool_;
};

CommandPoolManager::CommandPoolManager(const DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr)
    : deviceMgr_(deviceMgr) {
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyMgr.getComputeQFamilyIndex());

    const auto& device = deviceMgr.getDevice();
    commandPool_ = device.createCommandPool(commandPoolInfo);
}

CommandPoolManager::~CommandPoolManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyCommandPool(commandPool_);
}

}  // namespace vkc
