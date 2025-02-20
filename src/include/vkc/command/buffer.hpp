#pragma once

#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class CommandBufferManager {
public:
    inline CommandBufferManager(DeviceManager& deviceMgr, CommandPoolManager& commandPoolMgr);
    inline ~CommandBufferManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandBuffers(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandBuffers_;
    }

private:
    DeviceManager& deviceMgr_;            // FIXME: UAF
    CommandPoolManager& commandPoolMgr_;  // FIXME: UAF
    std::vector<vk::CommandBuffer> commandBuffers_;
};

CommandBufferManager::CommandBufferManager(DeviceManager& deviceMgr, CommandPoolManager& commandPoolMgr)
    : deviceMgr_(deviceMgr), commandPoolMgr_(commandPoolMgr) {
    auto& device = deviceMgr.getDevice();
    auto& commandPool = commandPoolMgr.getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    commandBuffers_ = device.allocateCommandBuffers(allocInfo);
}

CommandBufferManager::~CommandBufferManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    auto& commandPool = commandPoolMgr_.getCommandPool();
    device.freeCommandBuffers(commandPool, commandBuffers_);
}

}  // namespace vkc
