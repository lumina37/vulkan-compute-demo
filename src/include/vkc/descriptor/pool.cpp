#include <cstdint>
#include <memory>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.hpp"
#endif

namespace vkc {

DescPoolManager::DescPoolManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                 const std::span<const vk::DescriptorPoolSize> poolSizes)
    : pDeviceMgr_(pDeviceMgr) {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets((uint32_t)poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    auto& device = pDeviceMgr->getDevice();
    descPool_ = device.createDescriptorPool(poolInfo);
}

DescPoolManager::~DescPoolManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyDescriptorPool(descPool_);
}

}  // namespace vkc
