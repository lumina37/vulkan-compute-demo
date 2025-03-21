#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.hpp"
#endif

namespace vkc {

DescPoolManager::DescPoolManager(DeviceManager& deviceMgr, const std::span<const vk::DescriptorPoolSize> poolSizes)
    : deviceMgr_(deviceMgr) {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    auto& device = deviceMgr.getDevice();
    descPool_ = device.createDescriptorPool(poolInfo);
}

DescPoolManager::~DescPoolManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorPool(descPool_);
}

}  // namespace vkc
