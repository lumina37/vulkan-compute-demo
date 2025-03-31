#include <cstdint>
#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.hpp"
#endif

namespace vkc {

CommandPoolManager::CommandPoolManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const uint32_t queueFamilyIdx)
    : pDeviceMgr_(pDeviceMgr) {
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIdx);

    auto& device = pDeviceMgr->getDevice();
    commandPool_ = device.createCommandPool(commandPoolInfo);
}

CommandPoolManager::~CommandPoolManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyCommandPool(commandPool_);
}

}  // namespace vkc
