#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.hpp"
#endif

namespace vkc {

CommandPoolManager::CommandPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::CommandPool commandPool,
                                       uint32_t queueFamilyIdx) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), commandPool_(commandPool), queueFamilyIdx_(queueFamilyIdx) {}

CommandPoolManager::CommandPoolManager(CommandPoolManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      commandPool_(std::exchange(rhs.commandPool_, nullptr)),
      queueFamilyIdx_(rhs.queueFamilyIdx_) {}

CommandPoolManager::~CommandPoolManager() noexcept {
    if (commandPool_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroyCommandPool(commandPool_);
    commandPool_ = nullptr;
}

std::expected<CommandPoolManager, Error> CommandPoolManager::create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                    uint32_t queueFamilyIdx) noexcept {
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIdx);

    auto& device = pDeviceMgr->getDevice();
    vk::CommandPool commandPool = device.createCommandPool(commandPoolInfo);

    return CommandPoolManager{std::move(pDeviceMgr), commandPool, queueFamilyIdx};
}

}  // namespace vkc
